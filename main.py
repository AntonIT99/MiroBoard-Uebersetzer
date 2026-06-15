#!/usr/bin/env python3

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import requests
from dotenv import load_dotenv


MIRO_API_BASE = "https://api.miro.com/v2"

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


def deepl_headers(auth_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"DeepL-Auth-Key {auth_key}",
        "Content-Type": "application/json",
    }


def copy_board(miro_token: str, source_board_id: str, clone_name: str) -> Dict[str, Any]:
    encoded_board_id = quote(source_board_id, safe="=")

    url = f"{MIRO_API_BASE}/boards"
    params = {"copy_from": encoded_board_id}
    payload = {"name": clone_name}

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


def chunk_targets(
    targets: List[TranslationTarget],
    max_payload_bytes: int = 100_000,
) -> Iterable[List[TranslationTarget]]:
    """
    DeepL has a request body size limit. This keeps batches safely below it.
    """
    batch: List[TranslationTarget] = []
    current_size = 0

    for target in targets:
        text_size = len(target.source_text.encode("utf-8")) + 200

        if batch and current_size + text_size > max_payload_bytes:
            yield batch
            batch = []
            current_size = 0

        batch.append(target)
        current_size += text_size

    if batch:
        yield batch


def get_deepl_url(auth_key: str) -> str:
    explicit_url = os.getenv("DEEPL_API_URL")
    if explicit_url:
        return explicit_url

    # DeepL Free API keys usually end with ":fx".
    if auth_key.endswith(":fx"):
        return "https://api-free.deepl.com/v2/translate"

    return "https://api.deepl.com/v2/translate"


def translate_batch(
    deepl_auth_key: str,
    targets: List[TranslationTarget],
    *,
    source_lang: Optional[str],
    target_lang: str,
    tag_handling_html: bool,
) -> List[str]:
    url = get_deepl_url(deepl_auth_key)

    body: Dict[str, Any] = {
        "text": [target.source_text for target in targets],
        "target_lang": target_lang,
    }

    if source_lang:
        body["source_lang"] = source_lang

    if tag_handling_html:
        body["tag_handling"] = "html"

    payload = request_json(
        "POST",
        url,
        headers=deepl_headers(deepl_auth_key),
        json=body,
    )

    translations = payload.get("translations", [])
    if len(translations) != len(targets):
        raise ApiError(
            f"DeepL returned {len(translations)} translations for {len(targets)} texts"
        )

    return [entry["text"] for entry in translations]


def translate_targets(
    deepl_auth_key: str,
    targets: List[TranslationTarget],
    *,
    source_lang: Optional[str],
    target_lang: str,
) -> Dict[Tuple[str, str, str], str]:
    """
    Returns:
      (item_id, item_type, field) -> translated_text
    """
    result: Dict[Tuple[str, str, str], str] = {}

    html_targets = [t for t in targets if t.is_html]
    plain_targets = [t for t in targets if not t.is_html]

    for label, group, tag_handling_html in [
        ("HTML/Rich Text", html_targets, True),
        ("Plain Text", plain_targets, False),
    ]:
        if not group:
            continue

        print(f"Translating {len(group)} {label} fields...")

        done = 0
        for batch in chunk_targets(group):
            translations = translate_batch(
                deepl_auth_key,
                batch,
                source_lang=source_lang,
                target_lang=target_lang,
                tag_handling_html=tag_handling_html,
            )

            for target, translated in zip(batch, translations):
                result[(target.item_id, target.item_type, target.field)] = translated

            done += len(batch)
            print(f"  {done}/{len(group)} translated")

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

    raise ApiError(
        f"PATCH failed for {item_type} {item_id}: HTTP {response.status_code}: {body}"
    )


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
        "--source-lang",
        default="DE",
        help='DeepL source language, e.g. "DE". Use empty string for auto-detect.',
    )
    parser.add_argument(
        "--target-lang",
        default="EN-US",
        help='DeepL target language, e.g. "EN-US" or "EN-GB".',
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
        help="Create the clone and translate, but do not write translations back.",
    )

    args = parser.parse_args()

    miro_token = os.getenv("MIRO_ACCESS_TOKEN")
    deepl_auth_key = os.getenv("DEEPL_AUTH_KEY")

    if not miro_token:
        print("Missing MIRO_ACCESS_TOKEN in environment or .env", file=sys.stderr)
        return 2

    if not deepl_auth_key:
        print("Missing DEEPL_AUTH_KEY in environment or .env", file=sys.stderr)
        return 2

    source_board_id = extract_board_id(args.source_board)
    clone_name = args.clone_name or make_default_clone_name(args.clone_prefix)
    source_lang = args.source_lang.strip() or None

    print(f"Source board: {source_board_id}")
    print(f"Creating clone: {clone_name}")

    clone = copy_board(
        miro_token=miro_token,
        source_board_id=source_board_id,
        clone_name=clone_name,
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

    translations = translate_targets(
        deepl_auth_key=deepl_auth_key,
        targets=targets,
        source_lang=source_lang,
        target_lang=args.target_lang,
    )

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