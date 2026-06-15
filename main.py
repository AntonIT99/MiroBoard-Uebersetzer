#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import requests
from dotenv import load_dotenv


MIRO_API_BASE = "https://api.miro.com/v2"
DEFAULT_CT2_MODEL_DIR = "models/opus-mt-de-en-ct2"
DEFAULT_HF_TOKENIZER_MODEL = "Helsinki-NLP/opus-mt-de-en"
TRANSLATION_CACHE_FILE = Path("translation_cache_ct2_de_en.json")
CT2_CONVERSION_COMMAND = (
    "ct2-transformers-converter --model Helsinki-NLP/opus-mt-de-en "
    "--output_dir models/opus-mt-de-en-ct2 --quantization int8 --force"
)
PYTHON_DEPENDENCIES_COMMAND = (
    "pip install ctranslate2 transformers sentencepiece sacremoses beautifulsoup4 "
    "requests python-dotenv"
)
TRANSLATION_CACHE_SAVE_INTERVAL = 500

# Item type -> endpoint segment
MIRO_UPDATE_ENDPOINTS = {
    "sticky_note": "sticky_notes",
    "text": "texts",
    "shape": "shapes",
    "card": "cards",
    "frame": "frames",
}

# Item type -> text fields inside item["data"]
TRANSLATABLE_FIELDS = {
    "sticky_note": ["content"],
    "text": ["content"],
    "shape": ["content"],
    "card": ["title", "description"],
    "frame": ["title"],
}


@dataclass
class TranslationTarget:
    item_id: str
    item_type: str
    field: str
    source_text: str
    is_html: bool


class ApiError(RuntimeError):
    pass


class MiroWritePermissionError(ApiError):
    pass


class MiroPreflightCleanupError(ApiError):
    pass


def extract_board_id(value: str) -> str:
    """
    Accepts either a raw Miro board id like:
      uXjVABCDEF=
    or a Miro board URL like:
      https://miro.com/app/board/uXjVABCDEF=/
    """
    value = value.strip()

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        match = re.search(r"/app/board/([^/]+)", parsed.path)
        if not match:
            raise ValueError(f"Could not extract board id from URL: {value}")
        return unquote(match.group(1))

    return value


def looks_like_html(text: str) -> bool:
    return bool(re.search(r"</?[a-zA-Z][^>]*>", text))


def strip_html_for_empty_check(text: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", text)
    no_entities = (
        no_tags.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return re.sub(r"\s+", " ", no_entities).strip()


def is_probably_translatable(text: Optional[str]) -> bool:
    if not text:
        return False

    plain = strip_html_for_empty_check(text)

    if not plain:
        return False

    # Skip pure numbers, dates, arrows, punctuation, etc.
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]", plain):
        return False

    return True


def request_json(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    max_retries: int = 5,
    **kwargs: Any,
) -> Dict[str, Any]:
    for attempt in range(max_retries):
        response = requests.request(method, url, headers=headers, timeout=60, **kwargs)

        if response.status_code in (429, 500, 502, 503, 504):
            wait_seconds = min(2 ** attempt, 30)
            print(
                f"Temporary API issue {response.status_code}. Retrying in {wait_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
            continue

        if not response.ok:
            try:
                body = response.json()
            except Exception:
                body = response.text

            raise ApiError(
                f"{method} {url} failed with HTTP {response.status_code}: {body}"
            )

        if not response.text:
            return {}

        return response.json()

    raise ApiError(f"{method} {url} failed after {max_retries} retries")


def miro_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def copy_board(
    miro_token: str,
    source_board_id: str,
    clone_name: str,
    *,
    target_team_id: Optional[str],
) -> Dict[str, Any]:
    encoded_board_id = quote(source_board_id, safe="=")

    url = f"{MIRO_API_BASE}/boards"
    params = {"copy_from": encoded_board_id}
    payload = {"name": clone_name}
    if target_team_id:
        payload["teamId"] = target_team_id

    return request_json(
        "PUT",
        url,
        headers=miro_headers(miro_token),
        params=params,
        json=payload,
    )


def get_all_items(miro_token: str, board_id: str) -> List[Dict[str, Any]]:
    encoded_board_id = quote(board_id, safe="=")
    url = f"{MIRO_API_BASE}/boards/{encoded_board_id}/items"

    all_items: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params = {"limit": 50}
        if cursor:
            params["cursor"] = cursor

        payload = request_json(
            "GET",
            url,
            headers=miro_headers(miro_token),
            params=params,
        )

        all_items.extend(payload.get("data", []))

        cursor = payload.get("cursor")
        if not cursor:
            break

    return all_items


def collect_translation_targets(items: List[Dict[str, Any]]) -> List[TranslationTarget]:
    targets: List[TranslationTarget] = []

    for item in items:
        item_id = item.get("id")
        item_type = item.get("type")
        data = item.get("data") or {}

        if not item_id or not item_type:
            continue

        fields = TRANSLATABLE_FIELDS.get(item_type, [])
        for field in fields:
            value = data.get(field)

            if not isinstance(value, str):
                continue

            if not is_probably_translatable(value):
                continue

            targets.append(
                TranslationTarget(
                    item_id=item_id,
                    item_type=item_type,
                    field=field,
                    source_text=value,
                    is_html=looks_like_html(value),
                )
            )

    return targets


def chunk_list(values: List[Any], batch_size: int) -> Iterable[List[Any]]:
    if batch_size < 1:
        raise ValueError("--translation-batch-size must be at least 1")

    for index in range(0, len(values), batch_size):
        yield values[index : index + batch_size]


@dataclass
class Ct2TranslatorBackend:
    translator: Any
    tokenizer: Any
    tokenizer_model: str
    ct2_model_dir: str
    batch_size: int


@dataclass
class TranslationStats:
    cached_translations_used: int = 0
    newly_generated_translations: int = 0
    completed_fields: int = 0


@dataclass
class HtmlTextReplacement:
    node: Any
    prefix: str
    source_text: str
    suffix: str


@dataclass
class PreparedTranslationTarget:
    target: TranslationTarget
    soup: Optional[Any]
    html_replacements: List[HtmlTextReplacement]
    plain_source_text: Optional[str]


def load_translation_cache() -> Dict[str, str]:
    if not TRANSLATION_CACHE_FILE.exists():
        return {}

    try:
        with TRANSLATION_CACHE_FILE.open("r", encoding="utf-8") as cache_file:
            cache = json.load(cache_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: Could not read {TRANSLATION_CACHE_FILE}: {exc}. "
            "Starting with an empty translation cache.",
            file=sys.stderr,
        )
        return {}

    if not isinstance(cache, dict):
        print(
            f"WARNING: {TRANSLATION_CACHE_FILE} does not contain a JSON object. "
            "Starting with an empty translation cache.",
            file=sys.stderr,
        )
        return {}

    return {str(key): str(value) for key, value in cache.items()}


def save_translation_cache(cache: Dict[str, str]) -> None:
    tmp_path = TRANSLATION_CACHE_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as cache_file:
        json.dump(cache, cache_file, ensure_ascii=False, indent=2, sort_keys=True)
        cache_file.write("\n")
    tmp_path.replace(TRANSLATION_CACHE_FILE)


def make_translation_cache_key(
    text: str,
    tokenizer_model: str,
    ct2_model_dir: str,
) -> str:
    return json.dumps(
        {
            "source_text": text,
            "tokenizer_model": tokenizer_model,
            "ct2_model_dir": str(Path(ct2_model_dir)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def create_ct2_translator(
    ct2_model_dir: str,
    tokenizer_model: str,
    device: str,
    compute_type: str,
) -> Tuple[Any, Any]:
    model_path = Path(ct2_model_dir)
    if not model_path.is_dir():
        raise RuntimeError(
            f"Missing converted CTranslate2 model directory: {ct2_model_dir}\n"
            "Create it with:\n"
            f"  {CT2_CONVERSION_COMMAND}"
        )

    try:
        import ctranslate2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python package 'ctranslate2'. Install dependencies with:\n"
            f"  {PYTHON_DEPENDENCIES_COMMAND}"
        ) from exc

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python package 'transformers'. Install dependencies with:\n"
            f"  {PYTHON_DEPENDENCIES_COMMAND}"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "Could not load the Hugging Face tokenizer. Make sure sentencepiece "
            "and sacremoses are installed:\n"
            f"  {PYTHON_DEPENDENCIES_COMMAND}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not load tokenizer '{tokenizer_model}'. If this machine is "
            "offline, download/cache the tokenizer first or use a local tokenizer "
            "directory."
        ) from exc

    translator = ctranslate2.Translator(
        str(model_path),
        device=device,
        compute_type=compute_type,
    )

    return translator, tokenizer


def translate_plain_batch_with_ct2(
    texts: List[str],
    translator: Any,
    tokenizer: Any,
    batch_size: int,
) -> List[str]:
    translations: List[str] = []

    for batch in chunk_list(texts, batch_size):
        source_batches = [
            tokenizer.convert_ids_to_tokens(
                tokenizer.encode(text, add_special_tokens=True)
            )
            for text in batch
        ]
        results = translator.translate_batch(source_batches, beam_size=1)
        if len(results) != len(batch):
            raise RuntimeError(
                f"CTranslate2 returned {len(results)} translations for "
                f"{len(batch)} input texts"
            )

        for result in results:
            output_tokens = result.hypotheses[0]
            output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
            translations.append(
                tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            )

    return translations


def split_surrounding_whitespace(text: str) -> Tuple[str, str, str]:
    prefix_match = re.match(r"\s*", text)
    suffix_match = re.search(r"\s*$", text)

    prefix = prefix_match.group(0) if prefix_match else ""
    suffix = suffix_match.group(0) if suffix_match else ""
    core_start = len(prefix)
    core_end = len(text) - len(suffix)

    if core_start >= core_end:
        return text, "", ""

    return prefix, text[core_start:core_end], suffix


def import_beautifulsoup() -> Tuple[Any, Any]:
    try:
        from bs4 import BeautifulSoup, NavigableString
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python package 'beautifulsoup4'. Install dependencies with:\n"
            f"  {PYTHON_DEPENDENCIES_COMMAND}"
        ) from exc

    return BeautifulSoup, NavigableString


def translate_html_preserving_tags_with_ct2(
    html: str,
    translator: Any,
    tokenizer: Any,
    batch_size: int,
) -> str:
    BeautifulSoup, NavigableString = import_beautifulsoup()
    soup = BeautifulSoup(html, "html.parser")
    replacements: List[HtmlTextReplacement] = []

    for node in soup.find_all(string=True):
        if not isinstance(node, NavigableString):
            continue

        original_text = str(node)
        prefix, core_text, suffix = split_surrounding_whitespace(original_text)
        if not core_text:
            continue

        replacements.append(
            HtmlTextReplacement(
                node=node,
                prefix=prefix,
                source_text=core_text,
                suffix=suffix,
            )
        )

    translated_texts = translate_plain_batch_with_ct2(
        [replacement.source_text for replacement in replacements],
        translator,
        tokenizer,
        batch_size,
    )

    for replacement, translated_text in zip(replacements, translated_texts):
        replacement.node.replace_with(
            replacement.prefix + translated_text + replacement.suffix
        )

    return str(soup)


def prepare_translation_target(target: TranslationTarget) -> PreparedTranslationTarget:
    if not target.is_html:
        return PreparedTranslationTarget(
            target=target,
            soup=None,
            html_replacements=[],
            plain_source_text=target.source_text,
        )

    BeautifulSoup, NavigableString = import_beautifulsoup()
    soup = BeautifulSoup(target.source_text, "html.parser")
    replacements: List[HtmlTextReplacement] = []

    for node in soup.find_all(string=True):
        if not isinstance(node, NavigableString):
            continue

        original_text = str(node)
        prefix, core_text, suffix = split_surrounding_whitespace(original_text)
        if not core_text:
            continue

        replacements.append(
            HtmlTextReplacement(
                node=node,
                prefix=prefix,
                source_text=core_text,
                suffix=suffix,
            )
        )

    return PreparedTranslationTarget(
        target=target,
        soup=soup,
        html_replacements=replacements,
        plain_source_text=None,
    )


def get_cached_or_translate_texts(
    texts: List[str],
    backend: Ct2TranslatorBackend,
    cache: Dict[str, str],
    stats: TranslationStats,
) -> Dict[str, str]:
    translated_by_text: Dict[str, str] = {}
    texts_to_translate: List[str] = []
    seen_missing: set[str] = set()

    for text in texts:
        cache_key = make_translation_cache_key(
            text,
            backend.tokenizer_model,
            backend.ct2_model_dir,
        )

        if cache_key in cache:
            translated_by_text[text] = cache[cache_key]
            stats.cached_translations_used += 1
            continue

        if text not in seen_missing:
            texts_to_translate.append(text)
            seen_missing.add(text)

    if not texts_to_translate:
        return translated_by_text

    generated_since_save = 0
    for batch in chunk_list(texts_to_translate, backend.batch_size):
        translated_batch = translate_plain_batch_with_ct2(
            batch,
            backend.translator,
            backend.tokenizer,
            backend.batch_size,
        )

        for source_text, translated_text in zip(batch, translated_batch):
            translated_by_text[source_text] = translated_text
            cache_key = make_translation_cache_key(
                source_text,
                backend.tokenizer_model,
                backend.ct2_model_dir,
            )
            cache[cache_key] = translated_text

        stats.newly_generated_translations += len(batch)
        generated_since_save += len(batch)

        if generated_since_save >= TRANSLATION_CACHE_SAVE_INTERVAL:
            save_translation_cache(cache)
            generated_since_save = 0

    return translated_by_text


def translate_targets_with_ct2(
    targets: List[TranslationTarget],
    args: argparse.Namespace,
) -> Dict[Tuple[str, str, str], str]:
    """
    Returns:
      (item_id, item_type, field) -> translated_text
    """
    translator, tokenizer = create_ct2_translator(
        ct2_model_dir=args.ct2_model_dir,
        tokenizer_model=args.hf_tokenizer_model,
        device=args.ct2_device,
        compute_type=args.ct2_compute_type,
    )
    backend = Ct2TranslatorBackend(
        translator=translator,
        tokenizer=tokenizer,
        tokenizer_model=args.hf_tokenizer_model,
        ct2_model_dir=args.ct2_model_dir,
        batch_size=args.translation_batch_size,
    )

    print(f"Loading translation cache: {TRANSLATION_CACHE_FILE}")
    cache = load_translation_cache()
    stats = TranslationStats()
    prepared_targets = [prepare_translation_target(target) for target in targets]

    texts_to_translate: List[str] = []
    for prepared in prepared_targets:
        if prepared.plain_source_text is not None:
            texts_to_translate.append(prepared.plain_source_text)
            continue

        texts_to_translate.extend(
            replacement.source_text for replacement in prepared.html_replacements
        )

    print(
        f"Translating {len(targets)} fields locally with CTranslate2 "
        f"({len(set(texts_to_translate))} unique text units)..."
    )

    translated_by_text = get_cached_or_translate_texts(
        texts=texts_to_translate,
        backend=backend,
        cache=cache,
        stats=stats,
    )

    result: Dict[Tuple[str, str, str], str] = {}

    for prepared in prepared_targets:
        target = prepared.target

        if prepared.plain_source_text is not None:
            translated_text = translated_by_text.get(
                prepared.plain_source_text,
                prepared.plain_source_text,
            )
        else:
            for replacement in prepared.html_replacements:
                translated_text_node = translated_by_text.get(
                    replacement.source_text,
                    replacement.source_text,
                )
                replacement.node.replace_with(
                    replacement.prefix + translated_text_node + replacement.suffix
                )
            translated_text = str(prepared.soup)

        result[(target.item_id, target.item_type, target.field)] = translated_text
        stats.completed_fields += 1

        if stats.completed_fields % 100 == 0 or stats.completed_fields == len(targets):
            print(f"  {stats.completed_fields}/{len(targets)} fields translated")

    save_translation_cache(cache)
    print(f"Cached translations used: {stats.cached_translations_used}")
    print(f"Newly generated translations: {stats.newly_generated_translations}")

    return result


def patch_miro_item(
    miro_token: str,
    board_id: str,
    item_id: str,
    item_type: str,
    data_update: Dict[str, str],
    *,
    try_flat_fallback: bool = True,
) -> None:
    endpoint = MIRO_UPDATE_ENDPOINTS[item_type]
    encoded_board_id = quote(board_id, safe="=")
    encoded_item_id = quote(item_id, safe="=")

    url = f"{MIRO_API_BASE}/boards/{encoded_board_id}/{endpoint}/{encoded_item_id}"

    headers = miro_headers(miro_token)

    # Current v2 docs model item data under "data".
    payload = {"data": data_update}

    response = requests.patch(url, headers=headers, json=payload, timeout=60)

    if response.ok:
        return

    # Some Miro migration/reference examples show flat payloads.
    # This fallback makes the script more tolerant if your tenant/API behavior differs.
    if try_flat_fallback and response.status_code == 400:
        fallback_response = requests.patch(
            url,
            headers=headers,
            json=data_update,
            timeout=60,
        )
        if fallback_response.ok:
            return

        try:
            fallback_body = fallback_response.json()
        except Exception:
            fallback_body = fallback_response.text

        raise ApiError(
            f"PATCH fallback also failed for {item_type} {item_id}: "
            f"HTTP {fallback_response.status_code}: {fallback_body}"
        )

    try:
        body = response.json()
    except Exception:
        body = response.text

    if response.status_code == 403 and is_miro_write_permission_error(body):
        raise MiroWritePermissionError(
            "Miro refused item updates on the cloned board. The access token can "
            "copy/read the board, but its Miro user or OAuth app is not authorized "
            "to edit widgets on that board. Open the clone with the same Miro user "
            "that authorized the token and confirm it can manually edit items; if "
            "needed, reauthorize the app with board write permissions or copy the "
            "board into a team where the app is installed."
        )

    raise ApiError(
        f"PATCH failed for {item_type} {item_id}: HTTP {response.status_code}: {body}"
    )


def create_miro_preflight_shape(miro_token: str, board_id: str) -> str:
    encoded_board_id = quote(board_id, safe="=")
    url = f"{MIRO_API_BASE}/boards/{encoded_board_id}/shapes"
    payload = {
        "data": {
            "content": "__miro_translation_permission_check__",
            "shape": "rectangle",
        },
        "position": {
            "x": -100000,
            "y": -100000,
        },
        "geometry": {
            "width": 10,
            "height": 10,
        },
    }

    response = requests.post(
        url,
        headers=miro_headers(miro_token),
        json=payload,
        timeout=60,
    )

    if response.ok:
        body = response.json()
        item_id = body.get("id")
        if not item_id:
            raise ApiError(f"Miro preflight shape response did not contain id: {body}")
        return item_id

    try:
        body = response.json()
    except Exception:
        body = response.text

    if response.status_code == 403 and is_miro_write_permission_error(body):
        raise MiroWritePermissionError(
            "Miro refused creating a test item on the cloned board. The access "
            "token can copy/read the board, but its Miro user or OAuth app is not "
            "authorized to edit widgets on that board. Reauthorize the app with "
            "board write permissions or copy the board into a team where the app "
            "is installed and the token user is an editor."
        )

    raise ApiError(
        f"Creating Miro preflight shape failed with HTTP {response.status_code}: {body}"
    )


def delete_miro_item(miro_token: str, board_id: str, item_id: str) -> None:
    encoded_board_id = quote(board_id, safe="=")
    encoded_item_id = quote(item_id, safe="=")
    url = f"{MIRO_API_BASE}/boards/{encoded_board_id}/items/{encoded_item_id}"

    response = requests.delete(url, headers=miro_headers(miro_token), timeout=60)

    if response.ok:
        return

    try:
        body = response.json()
    except Exception:
        body = response.text

    raise MiroPreflightCleanupError(
        f"Deleting Miro preflight item {item_id} failed with "
        f"HTTP {response.status_code}: {body}"
    )


def is_miro_write_permission_error(body: Any) -> bool:
    if not isinstance(body, dict):
        return False

    message = str(body.get("message", "")).lower()
    code = str(body.get("code", "")).lower()

    return code == "insufficientpermissions" and (
        "update widgets" in message or "authorized" in message
    )


def verify_miro_write_access(
    miro_token: str,
    board_id: str,
) -> None:
    """
    Performs a cheap create/update/delete cycle before reading all board items or
    translating locally. This catches widget write permission problems before
    local translation starts.
    """
    print("Checking Miro write access on cloned board...")
    preflight_item_id: Optional[str] = None

    try:
        preflight_item_id = create_miro_preflight_shape(
            miro_token=miro_token,
            board_id=board_id,
        )
        patch_miro_item(
            miro_token=miro_token,
            board_id=board_id,
            item_id=preflight_item_id,
            item_type="shape",
            data_update={"content": "__miro_translation_permission_check_ok__"},
            try_flat_fallback=False,
        )
    finally:
        if preflight_item_id:
            original_error = sys.exc_info()[0]
            try:
                delete_miro_item(
                    miro_token=miro_token,
                    board_id=board_id,
                    item_id=preflight_item_id,
                )
            except MiroPreflightCleanupError as exc:
                if original_error is None:
                    raise
                print(f"WARNING: {exc}", file=sys.stderr)


def apply_translations_to_miro(
    miro_token: str,
    board_id: str,
    translations: Dict[Tuple[str, str, str], str],
    *,
    dry_run: bool,
) -> int:
    grouped: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)

    for (item_id, item_type, field), translated_text in translations.items():
        grouped[(item_id, item_type)][field] = translated_text

    print(f"Updating {len(grouped)} Miro items...")

    updated = 0
    for (item_id, item_type), data_update in grouped.items():
        if item_type not in MIRO_UPDATE_ENDPOINTS:
            continue

        if dry_run:
            print(f"[DRY RUN] Would update {item_type} {item_id}: {list(data_update)}")
            updated += 1
            continue

        patch_miro_item(
            miro_token=miro_token,
            board_id=board_id,
            item_id=item_id,
            item_type=item_type,
            data_update=data_update,
        )
        updated += 1

        if updated % 25 == 0:
            print(f"  {updated}/{len(grouped)} items updated")

    return updated


def make_default_clone_name(prefix: str) -> str:
    # Miro board names have a documented max length of 60 chars.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    name = f"{prefix} {timestamp}"
    return name[:60]


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Copy a Miro board and translate supported text items in the copy."
    )
    parser.add_argument(
        "--source-board",
        required=True,
        help="Miro board id or full Miro board URL",
    )
    parser.add_argument(
        "--clone-name",
        default=None,
        help='Name for the translated clone, e.g. "[EN] Product Workshop"',
    )
    parser.add_argument(
        "--clone-prefix",
        default="[EN]",
        help='Used if --clone-name is omitted. Default: "[EN]"',
    )
    parser.add_argument(
        "--target-team-id",
        default=None,
        help="Optional Miro team id where the cloned board should be created.",
    )
    parser.add_argument(
        "--source-lang",
        default="DE",
        help=(
            'Source language label kept for CLI compatibility. The default CT2 '
            'model translates German to English.'
        ),
    )
    parser.add_argument(
        "--target-lang",
        default="EN-US",
        help=(
            'Target language label kept for CLI compatibility. The default CT2 '
            'model translates German to English.'
        ),
    )
    parser.add_argument(
        "--translator",
        default="ct2",
        choices=["ct2"],
        help='Translation backend. Default: "ct2".',
    )
    parser.add_argument(
        "--ct2-model-dir",
        default=DEFAULT_CT2_MODEL_DIR,
        help=f"Converted CTranslate2 model directory. Default: {DEFAULT_CT2_MODEL_DIR}",
    )
    parser.add_argument(
        "--hf-tokenizer-model",
        default=DEFAULT_HF_TOKENIZER_MODEL,
        help=(
            "Hugging Face tokenizer model name or local tokenizer directory. "
            f"Default: {DEFAULT_HF_TOKENIZER_MODEL}"
        ),
    )
    parser.add_argument(
        "--ct2-device",
        default="cpu",
        help='CTranslate2 device, e.g. "cpu" or "cuda". Default: "cpu".',
    )
    parser.add_argument(
        "--ct2-compute-type",
        default="int8",
        help='CTranslate2 compute type, e.g. "int8", "float32", or "float16".',
    )
    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=32,
        help="Number of plain text units translated per CTranslate2 batch.",
    )
    parser.add_argument(
        "--sleep-after-copy",
        type=float,
        default=3.0,
        help="Seconds to wait after copying the board before reading items.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Create the clone, run the write preflight, and translate locally, "
            "but do not write translated fields back."
        ),
    )

    args = parser.parse_args()

    miro_token = os.getenv("MIRO_ACCESS_TOKEN")

    if not miro_token:
        print("Missing MIRO_ACCESS_TOKEN in environment or .env", file=sys.stderr)
        return 2

    if args.translation_batch_size < 1:
        print("--translation-batch-size must be at least 1", file=sys.stderr)
        return 2

    source_board_id = extract_board_id(args.source_board)
    clone_name = args.clone_name or make_default_clone_name(args.clone_prefix)

    print(f"Source board: {source_board_id}")
    print(f"Creating clone: {clone_name}")

    clone = copy_board(
        miro_token=miro_token,
        source_board_id=source_board_id,
        clone_name=clone_name,
        target_team_id=args.target_team_id,
    )

    clone_board_id = clone.get("id")
    if not clone_board_id:
        raise ApiError(f"Miro copy response did not contain board id: {clone}")

    print(f"Clone board id: {clone_board_id}")

    view_link = (
        clone.get("viewLink")
        or clone.get("links", {}).get("self")
        or clone.get("links", {}).get("related")
    )
    if view_link:
        print(f"Clone link: {view_link}")

    if args.sleep_after_copy > 0:
        print(f"Waiting {args.sleep_after_copy}s for board copy to settle...")
        time.sleep(args.sleep_after_copy)

    print("Reading items from cloned board...")
    items = get_all_items(miro_token=miro_token, board_id=clone_board_id)

    print(f"Found {len(items)} items in clone.")

    targets = collect_translation_targets(items)

    counts_by_type = defaultdict(int)
    for target in targets:
        counts_by_type[target.item_type] += 1

    print(f"Found {len(targets)} translatable text fields.")
    for item_type, count in sorted(counts_by_type.items()):
        print(f"  {item_type}: {count}")

    if not targets:
        print("No translatable fields found.")
        return 0

    verify_miro_write_access(
        miro_token=miro_token,
        board_id=clone_board_id,
    )

    if args.translator == "ct2":
        translations = translate_targets_with_ct2(targets=targets, args=args)
    else:
        raise ValueError(f"Unsupported translator backend: {args.translator}")

    updated = apply_translations_to_miro(
        miro_token=miro_token,
        board_id=clone_board_id,
        translations=translations,
        dry_run=args.dry_run,
    )

    print()
    print("Done.")
    print(f"Translated fields: {len(translations)}")
    print(f"Updated Miro items: {updated}")
    print(f"English clone board id: {clone_board_id}")

    if view_link:
        print(f"English clone link: {view_link}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
