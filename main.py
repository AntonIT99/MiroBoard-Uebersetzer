import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import requests
from dotenv import load_dotenv


MIRO_API_BASE = "https://api.miro.com/v2"
DEFAULT_CT2_MODEL_DIR = "models/nllb-200-distilled-1.3B-ct2"
DEFAULT_HF_TOKENIZER_MODEL = "facebook/nllb-200-distilled-1.3B"
DEFAULT_CT2_MODEL_FAMILY = "nllb"
DEFAULT_SOURCE_LANG_CODE = "deu_Latn"
DEFAULT_TARGET_LANG_CODE = "eng_Latn"
DEFAULT_QUALITY_REVIEW_CT2_MODEL_DIR = "models/nllb-200-3.3B-ct2"
DEFAULT_QUALITY_REVIEW_HF_TOKENIZER_MODEL = "facebook/nllb-200-3.3B"
TRANSLATION_CACHE_FILE = Path("translation_cache_ct2_de_en.json")
DEFAULT_GLOSSARY_FILE = "translation_glossary_de_en.json"
CT2_CONVERSION_COMMAND = (
    "ct2-transformers-converter --model facebook/nllb-200-distilled-1.3B "
    "--output_dir models/nllb-200-distilled-1.3B-ct2 "
    "--quantization float16 --force"
)
QUALITY_REVIEW_CT2_CONVERSION_COMMAND = (
    "ct2-transformers-converter --model facebook/nllb-200-3.3B "
    "--output_dir models/nllb-200-3.3B-ct2 --quantization float16 --force"
)
PYTHON_DEPENDENCIES_COMMAND = (
    "pip install ctranslate2 torch transformers sentencepiece sacremoses beautifulsoup4 "
    "requests python-dotenv"
)
CUDA_DLL_HELP = (
    "CTranslate2 could not load the CUDA runtime library 'cublas64_12.dll'. "
    "Install the NVIDIA CUDA Toolkit 12.x and make sure its 'bin' directory is "
    "available in PATH, for example "
    "'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin'. "
    "If CUDA is installed but started from PyCharm or another shell, restart the "
    "application so it picks up the updated PATH. "
    "As a fallback, run with --ct2-device cpu --ct2-compute-type int8."
)
TRANSLATION_CACHE_SAVE_INTERVAL = 500
SYNC_STATE_VERSION = 1
POSITION_PRECISION = 1
GEOMETRY_PRECISION = 1
LOW_MAPPING_QUALITY_THRESHOLD = 0.5

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


@dataclass
class GlossaryEntry:
    source: str
    target: str
    case_sensitive: bool = False
    whole_word: bool = True


@dataclass
class Glossary:
    path: Path
    entries: List[GlossaryEntry]
    enabled: bool


@dataclass
class SyncReport:
    mode: str
    source_board_id: str
    clone_board_id: str
    sync_state_file: Optional[Path] = None
    translatable_source_fields: int = 0
    mapped_items_updated: int = 0
    new_clone_items_created: int = 0
    stale_clone_items_detected: int = 0
    stale_clone_items_deleted: int = 0
    unsupported_items_skipped: int = 0
    unmapped_source_items: int = 0
    ambiguous_items: int = 0
    updated_miro_items: int = 0
    cached_translations_used: int = 0
    higher_quality_cached_translations_used: int = 0
    manually_imported_clone_translations: int = 0
    newly_generated_translations: int = 0
    glossary_exact_overrides: int = 0
    glossary_post_replacements: int = 0
    validation_rejected_translations: int = 0
    round_trip_checks: int = 0
    round_trip_disagreements: int = 0
    quality_review_candidates: int = 0
    quality_review_cached_translations_used: int = 0
    quality_review_higher_quality_cached_translations_used: int = 0
    quality_review_newly_generated_translations: int = 0
    quality_review_retranslations: int = 0
    quality_review_retranslations_rejected: int = 0


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


def load_glossary(path: Path, disabled: bool) -> Glossary:
    if disabled:
        print("Glossary disabled.")
        return Glossary(path=path, entries=[], enabled=False)

    print(f"Glossary file: {path}")
    if not path.exists():
        print(f"Glossary file not found. Continuing without glossary: {path}")
        return Glossary(path=path, entries=[], enabled=False)

    try:
        with path.open("r", encoding="utf-8") as glossary_file:
            payload = json.load(glossary_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid glossary JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read glossary file {path}: {exc}") from exc

    entries = normalize_glossary_payload(payload, path)
    entries.sort(key=lambda entry: len(entry.source), reverse=True)
    print(f"Loaded glossary entries: {len(entries)}")
    return Glossary(path=path, entries=entries, enabled=bool(entries))


def normalize_glossary_payload(payload: Any, path: Path) -> List[GlossaryEntry]:
    entries: List[GlossaryEntry] = []

    if isinstance(payload, dict):
        for source, target in payload.items():
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(
                    f"Glossary map entries in {path} must be string -> string"
                )
            entries.append(GlossaryEntry(source=source, target=target))
        return entries

    if isinstance(payload, list):
        for index, raw_entry in enumerate(payload):
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Glossary entry #{index + 1} in {path} is not an object")

            source = raw_entry.get("source")
            target = raw_entry.get("target")
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(
                    f"Glossary entry #{index + 1} in {path} needs string source/target"
                )

            entries.append(
                GlossaryEntry(
                    source=source,
                    target=target,
                    case_sensitive=bool(raw_entry.get("case_sensitive", False)),
                    whole_word=bool(raw_entry.get("whole_word", True)),
                )
            )
        return entries

    raise ValueError(
        f"Glossary file {path} must contain either an object map or a list of entries"
    )


def exact_glossary_override(text: str, glossary: Glossary) -> Optional[str]:
    if not glossary.enabled:
        return None

    prefix, core_text, suffix = split_surrounding_whitespace(text)
    if not core_text:
        return None

    for entry in glossary.entries:
        if entry.case_sensitive:
            matches = core_text == entry.source
        else:
            matches = core_text.casefold() == entry.source.casefold()

        if matches:
            return prefix + entry.target + suffix

    return None


def apply_glossary_postprocessing(
    text: str,
    glossary: Glossary,
) -> Tuple[str, int]:
    if not glossary.enabled or not text:
        return text, 0

    result = text
    replacements = 0
    for entry in glossary.entries:
        flags = 0 if entry.case_sensitive else re.IGNORECASE
        source_pattern = re.escape(entry.source)
        if entry.whole_word:
            source_pattern = rf"(?<!\w){source_pattern}(?!\w)"

        result, count = re.subn(source_pattern, entry.target, result, flags=flags)
        replacements += count

    return result, replacements


@dataclass
class Ct2TranslatorBackend:
    translator: Any
    tokenizer: Any
    tokenizer_model: str
    ct2_model_dir: str
    model_family: str
    source_lang_code: str
    target_lang_code: str
    beam_size: int
    batch_size: int
    max_input_tokens: int
    preserve_special_symbols: bool


def create_ct2_backend(args: argparse.Namespace) -> Ct2TranslatorBackend:
    model_family = resolve_ct2_model_family(
        args.ct2_model_family,
        args.hf_tokenizer_model,
        args.ct2_model_dir,
    )
    translator, tokenizer = create_ct2_translator(
        ct2_model_dir=args.ct2_model_dir,
        tokenizer_model=args.hf_tokenizer_model,
        model_family=model_family,
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
        device=args.ct2_device,
        compute_type=args.ct2_compute_type,
    )
    return Ct2TranslatorBackend(
        translator=translator,
        tokenizer=tokenizer,
        tokenizer_model=args.hf_tokenizer_model,
        ct2_model_dir=args.ct2_model_dir,
        model_family=model_family,
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
        beam_size=args.beam_size,
        batch_size=args.translation_batch_size,
        max_input_tokens=args.max_input_tokens,
        preserve_special_symbols=args.preserve_special_symbols,
    )


def create_quality_review_ct2_backend(args: argparse.Namespace) -> Ct2TranslatorBackend:
    review_device = args.quality_review_ct2_device or args.ct2_device
    review_compute_type = args.quality_review_ct2_compute_type or args.ct2_compute_type
    model_family = resolve_ct2_model_family(
        args.quality_review_ct2_model_family,
        args.quality_review_hf_tokenizer_model,
        args.quality_review_ct2_model_dir,
    )
    translator, tokenizer = create_ct2_translator(
        ct2_model_dir=args.quality_review_ct2_model_dir,
        tokenizer_model=args.quality_review_hf_tokenizer_model,
        model_family=model_family,
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
        device=review_device,
        compute_type=review_compute_type,
    )
    return Ct2TranslatorBackend(
        translator=translator,
        tokenizer=tokenizer,
        tokenizer_model=args.quality_review_hf_tokenizer_model,
        ct2_model_dir=args.quality_review_ct2_model_dir,
        model_family=model_family,
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
        beam_size=args.quality_review_beam_size,
        batch_size=args.quality_review_batch_size,
        max_input_tokens=args.quality_review_max_input_tokens,
        preserve_special_symbols=args.preserve_special_symbols,
    )


@dataclass
class TranslationStats:
    cached_translations_used: int = 0
    higher_quality_cached_translations_used: int = 0
    manually_imported_clone_translations: int = 0
    newly_generated_translations: int = 0
    completed_fields: int = 0
    glossary_exact_overrides: int = 0
    glossary_post_replacements: int = 0
    validation_rejected_translations: int = 0
    round_trip_checks: int = 0
    round_trip_disagreements: int = 0
    quality_review_candidates: int = 0
    quality_review_cached_translations_used: int = 0
    quality_review_higher_quality_cached_translations_used: int = 0
    quality_review_newly_generated_translations: int = 0
    quality_review_retranslations: int = 0
    quality_review_retranslations_rejected: int = 0


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
    backend: Ct2TranslatorBackend,
) -> str:
    return json.dumps(
        {
            "beam_size": backend.beam_size,
            "ct2_model_dir": str(Path(backend.ct2_model_dir)),
            "max_input_tokens": backend.max_input_tokens,
            "model_family": backend.model_family,
            "preserve_special_symbols": backend.preserve_special_symbols,
            "source_lang_code": backend.source_lang_code,
            "source_text": text,
            "symbol_protection_version": 1,
            "target_lang_code": backend.target_lang_code,
            "tokenizer_model": backend.tokenizer_model,
        },
        ensure_ascii=False,
        sort_keys=True,
    )



@dataclass(frozen=True)
class TranslationCacheCandidate:
    cache_key: str
    translation: str
    metadata: Dict[str, Any]
    model_quality: Tuple[float, int]
    insertion_order: int


TranslationCacheIndex = Dict[
    Tuple[str, str, str, str, bool, int],
    List[TranslationCacheCandidate],
]


def parse_translation_cache_key(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Parse a serialized cache key.

    Invalid or legacy keys that are not JSON objects are ignored by the
    model-aware fallback logic, but they remain untouched in the cache file.
    """
    try:
        metadata = json.loads(cache_key)
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(metadata, dict):
        return None
    if not isinstance(metadata.get("source_text"), str):
        return None
    return metadata


def model_quality_from_hint(model_hint: str) -> Optional[Tuple[float, int]]:
    """
    Infer a comparable model-quality score from names such as
    'nllb-200-distilled-1.3B' and 'nllb-200-3.3B'.

    The first tuple element is the approximate parameter count in billions.
    For equal parameter counts, a non-distilled model ranks above a distilled
    one. Unknown model sizes are deliberately not compared.
    """
    normalized_hint = model_hint.lower()
    sizes_in_billions: List[float] = []

    for raw_size, raw_unit in re.findall(
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*([bm])\b",
        normalized_hint,
        flags=re.IGNORECASE,
    ):
        size = float(raw_size)
        if raw_unit.lower() == "m":
            size /= 1000.0
        sizes_in_billions.append(size)

    if not sizes_in_billions:
        return None

    non_distilled_bonus = 0 if "distilled" in normalized_hint else 1
    return max(sizes_in_billions), non_distilled_bonus


def model_quality_from_cache_metadata(
    metadata: Dict[str, Any],
) -> Optional[Tuple[float, int]]:
    model_hint = " ".join(
        str(metadata.get(field, ""))
        for field in ("tokenizer_model", "ct2_model_dir")
    )
    return model_quality_from_hint(model_hint)


def model_quality_from_backend(
    backend: Ct2TranslatorBackend,
) -> Optional[Tuple[float, int]]:
    return model_quality_from_hint(
        f"{backend.tokenizer_model} {backend.ct2_model_dir}"
    )


def cache_compatibility_identity(
    *,
    source_text: str,
    source_lang_code: str,
    target_lang_code: str,
    model_family: str,
    preserve_special_symbols: bool,
    symbol_protection_version: int = 1,
) -> Tuple[str, str, str, str, bool, int]:
    """
    Fields that must agree before one model's cached translation may substitute
    for another model's result.

    Beam size and token limits are generation settings and are not part of this
    identity. Language direction, model family and symbol handling must match.
    """
    return (
        source_text,
        source_lang_code,
        target_lang_code,
        model_family,
        preserve_special_symbols,
        symbol_protection_version,
    )


def build_translation_cache_index(
    cache: Dict[str, str],
) -> TranslationCacheIndex:
    index: TranslationCacheIndex = defaultdict(list)

    for insertion_order, (cache_key, translation) in enumerate(cache.items()):
        metadata = parse_translation_cache_key(cache_key)
        if metadata is None:
            continue

        quality = model_quality_from_cache_metadata(metadata)
        if quality is None:
            continue

        source_text = metadata.get("source_text")
        source_lang_code = metadata.get("source_lang_code")
        target_lang_code = metadata.get("target_lang_code")
        model_family = metadata.get("model_family")
        preserve_special_symbols = metadata.get("preserve_special_symbols")
        symbol_protection_version = metadata.get("symbol_protection_version", 1)

        if not all(
            isinstance(value, str)
            for value in (
                source_text,
                source_lang_code,
                target_lang_code,
                model_family,
            )
        ):
            continue
        if not isinstance(preserve_special_symbols, bool):
            continue
        if not isinstance(symbol_protection_version, int):
            continue

        identity = cache_compatibility_identity(
            source_text=source_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            model_family=model_family,
            preserve_special_symbols=preserve_special_symbols,
            symbol_protection_version=symbol_protection_version,
        )
        index[identity].append(
            TranslationCacheCandidate(
                cache_key=cache_key,
                translation=translation,
                metadata=metadata,
                model_quality=quality,
                insertion_order=insertion_order,
            )
        )

    return index


def find_higher_quality_cached_translation(
    text: str,
    backend: Ct2TranslatorBackend,
    cache_index: TranslationCacheIndex,
) -> Optional[TranslationCacheCandidate]:
    """
    Return the best compatible cached translation produced by a model that is
    strictly stronger than the currently selected backend.

    This lookup runs before the exact current-model cache lookup. Consequently,
    an existing weak-model cache entry cannot hide a stronger cached result.
    """
    current_quality = model_quality_from_backend(backend)
    if current_quality is None:
        return None

    identity = cache_compatibility_identity(
        source_text=text,
        source_lang_code=backend.source_lang_code,
        target_lang_code=backend.target_lang_code,
        model_family=backend.model_family,
        preserve_special_symbols=backend.preserve_special_symbols,
    )

    stronger_candidates = [
        candidate
        for candidate in cache_index.get(identity, [])
        if candidate.model_quality > current_quality
        and isinstance(candidate.translation, str)
        and candidate.translation != ""
    ]
    if not stronger_candidates:
        return None

    def candidate_priority(
        candidate: TranslationCacheCandidate,
    ) -> Tuple[Tuple[float, int], int, int, int]:
        metadata = candidate.metadata
        beam_size = metadata.get("beam_size")
        max_input_tokens = metadata.get("max_input_tokens")
        return (
            candidate.model_quality,
            beam_size if isinstance(beam_size, int) else 0,
            max_input_tokens if isinstance(max_input_tokens, int) else 0,
            candidate.insertion_order,
        )

    return max(stronger_candidates, key=candidate_priority)


def resolve_ct2_model_family(
    requested_family: str,
    tokenizer_model: str,
    ct2_model_dir: str,
) -> str:
    if requested_family != "auto":
        return requested_family

    model_hint = f"{tokenizer_model} {ct2_model_dir}".lower()
    if "nllb" in model_hint:
        return "nllb"
    return "marian"


def create_ct2_translator(
    ct2_model_dir: str,
    tokenizer_model: str,
    model_family: str,
    source_lang_code: str,
    target_lang_code: str,
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

    if device.lower() == "cuda":
        configure_cuda_dll_search_path()

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
        tokenizer_kwargs = {}
        if model_family == "nllb":
            tokenizer_kwargs["src_lang"] = source_lang_code
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, **tokenizer_kwargs)
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

    if model_family == "nllb":
        known_language_codes = (
            getattr(tokenizer, "lang_code_to_id", None)
            or getattr(tokenizer, "added_tokens_encoder", {})
        )
        for language_code, label in (
            (source_lang_code, "--source-lang-code"),
            (target_lang_code, "--target-lang-code"),
        ):
            if known_language_codes and language_code not in known_language_codes:
                raise RuntimeError(
                    f"Unknown NLLB language code for {label}: {language_code}"
                )

    try:
        translator = ctranslate2.Translator(
            str(model_path),
            device=device,
            compute_type=compute_type,
        )
    except RuntimeError as exc:
        if is_cuda_runtime_load_error(exc):
            raise RuntimeError(CUDA_DLL_HELP) from exc
        raise

    return translator, tokenizer


def configure_cuda_dll_search_path() -> None:
    cuda_bin_dir = find_cuda_bin_dir()
    if not cuda_bin_dir:
        return

    cuda_bin = str(cuda_bin_dir)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if cuda_bin not in path_entries:
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_bin)


def find_cuda_bin_dir() -> Optional[Path]:
    candidates: List[Path] = []

    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        value = os.getenv(env_var)
        if value:
            candidates.append(Path(value) / "bin")

    cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if cuda_root.is_dir():
        candidates.extend(
            sorted(
                [path / "bin" for path in cuda_root.glob("v12.*") if path.is_dir()],
                reverse=True,
            )
        )

    for candidate in candidates:
        if (candidate / "cublas64_12.dll").is_file():
            print(f"Using CUDA DLL directory: {candidate}")
            return candidate

    return None


def is_protected_symbol_char(value: str) -> bool:
    codepoint = ord(value)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or 0x2B00 <= codepoint <= 0x2BFF
        or codepoint in (0x200D, 0x20E3, 0xFE0E, 0xFE0F)
    )


def split_text_by_protected_symbols(text: str) -> List[Tuple[bool, str]]:
    segments: List[Tuple[bool, str]] = []
    current: List[str] = []
    current_protected: Optional[bool] = None
    index = 0

    def flush() -> None:
        nonlocal current, current_protected
        if current:
            segments.append((bool(current_protected), "".join(current)))
            current = []
        current_protected = None

    while index < len(text):
        char = text[index]
        if (
            char in "0123456789#*"
            and index + 1 < len(text)
            and text[index + 1] in ("\ufe0f", "\u20e3")
        ):
            flush()
            keycap_chars = [char]
            index += 1
            while index < len(text) and text[index] in ("\ufe0f", "\u20e3"):
                keycap_chars.append(text[index])
                index += 1
            segments.append((True, "".join(keycap_chars)))
            continue

        protected = is_protected_symbol_char(char)
        if current_protected is not None and protected != current_protected:
            flush()
        current_protected = protected
        current.append(char)
        index += 1

    flush()
    return segments


def decode_ct2_tokens(tokenizer: Any, output_tokens: List[str]) -> str:
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def token_count_for_text(text: str, backend: Ct2TranslatorBackend) -> int:
    return len(backend.tokenizer.encode(text, add_special_tokens=True))


def split_text_by_token_limit(text: str, backend: Ct2TranslatorBackend) -> List[str]:
    if backend.max_input_tokens < 1:
        return [text]

    if token_count_for_text(text, backend) <= backend.max_input_tokens:
        return [text]

    sentence_units = re.findall(r".+?(?:[.!?;:](?:\s+|$)|\n+|$)", text, flags=re.S)
    if not sentence_units or "".join(sentence_units) != text:
        sentence_units = re.findall(r"\S+\s*", text)

    chunks: List[str] = []
    current = ""

    def append_current() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for unit in sentence_units:
        candidate = current + unit
        if candidate and token_count_for_text(candidate, backend) <= backend.max_input_tokens:
            current = candidate
            continue

        append_current()
        if token_count_for_text(unit, backend) <= backend.max_input_tokens:
            current = unit
            continue

        word_units = re.findall(r"\S+\s*", unit)
        for word_unit in word_units:
            candidate = current + word_unit
            if candidate and token_count_for_text(candidate, backend) <= backend.max_input_tokens:
                current = candidate
            else:
                append_current()
                current = word_unit

    append_current()
    return chunks or [text]


def encode_text_for_backend(text: str, backend: Ct2TranslatorBackend) -> List[str]:
    if backend.model_family == "nllb" and hasattr(backend.tokenizer, "src_lang"):
        backend.tokenizer.src_lang = backend.source_lang_code

    return backend.tokenizer.convert_ids_to_tokens(
        backend.tokenizer.encode(text, add_special_tokens=True)
    )


def translate_plain_core_batch_with_ct2(
    texts: List[str],
    backend: Ct2TranslatorBackend,
) -> List[str]:
    translation_inputs: List[str] = []
    chunk_counts: List[int] = []

    for text in texts:
        chunks = split_text_by_token_limit(text, backend)
        chunk_counts.append(len(chunks))
        translation_inputs.extend(chunks)

    chunk_translations: List[str] = []

    for batch in chunk_list(translation_inputs, backend.batch_size):
        source_batches = [encode_text_for_backend(text, backend) for text in batch]
        translate_kwargs: Dict[str, Any] = {"beam_size": backend.beam_size}
        if backend.model_family == "nllb":
            translate_kwargs["target_prefix"] = [
                [backend.target_lang_code] for _ in source_batches
            ]
        try:
            results = backend.translator.translate_batch(source_batches, **translate_kwargs)
        except RuntimeError as exc:
            if is_cuda_runtime_load_error(exc):
                raise RuntimeError(CUDA_DLL_HELP) from exc
            raise

        if len(results) != len(batch):
            raise RuntimeError(
                f"CTranslate2 returned {len(results)} translations for "
                f"{len(batch)} input texts"
            )

        for result in results:
            output_tokens = result.hypotheses[0]
            if backend.model_family == "nllb" and output_tokens:
                output_tokens = output_tokens[1:]
            chunk_translations.append(decode_ct2_tokens(backend.tokenizer, output_tokens))

    translations: List[str] = []
    chunk_index = 0
    for count in chunk_counts:
        translations.append(" ".join(chunk_translations[chunk_index : chunk_index + count]))
        chunk_index += count
    return translations


def translate_plain_batch_with_ct2(
    texts: List[str],
    backend: Ct2TranslatorBackend,
) -> List[str]:
    if not backend.preserve_special_symbols:
        return translate_plain_core_batch_with_ct2(texts, backend)

    prepared_texts: List[Tuple[Optional[str], List[Tuple[bool, str]]]] = []
    cores_to_translate: List[str] = []

    for text in texts:
        segments = split_text_by_protected_symbols(text)
        if not any(is_protected for is_protected, _ in segments):
            prepared_texts.append((text, []))
            cores_to_translate.append(text)
            continue

        prepared_texts.append((None, segments))
        for is_protected, segment in segments:
            if is_protected or not is_probably_translatable(segment):
                continue

            _, core_text, _ = split_surrounding_whitespace(segment)
            if core_text:
                cores_to_translate.append(core_text)

    translated_cores = iter(translate_plain_core_batch_with_ct2(cores_to_translate, backend))
    translations: List[str] = []
    for original_text, segments in prepared_texts:
        if original_text is not None:
            translations.append(next(translated_cores))
            continue

        translated_parts: List[str] = []
        for is_protected, segment in segments:
            if is_protected or not is_probably_translatable(segment):
                translated_parts.append(segment)
                continue

            prefix, core_text, suffix = split_surrounding_whitespace(segment)
            if not core_text:
                translated_parts.append(segment)
                continue

            translated_parts.append(prefix + next(translated_cores) + suffix)
        translations.append("".join(translated_parts))

    return translations


def normalized_text_for_similarity(text: str) -> str:
    text = strip_html_for_empty_check(text).lower()
    text = re.sub(r"[^\wäöüßà-öø-ÿ]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(left: str, right: str) -> float:
    normalized_left = normalized_text_for_similarity(left)
    normalized_right = normalized_text_for_similarity(right)
    if not normalized_left and not normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]+", strip_html_for_empty_check(text)))


def has_german_signal(text: str) -> bool:
    plain = normalized_text_for_similarity(text)
    if re.search(r"[äöüß]", plain):
        return True

    german_markers = {
        "aber",
        "auch",
        "auf",
        "das",
        "der",
        "die",
        "ein",
        "eine",
        "für",
        "ist",
        "mit",
        "nicht",
        "oder",
        "und",
        "werden",
        "zu",
    }
    words = set(plain.split())
    return bool(words & german_markers)


def protected_symbol_sequence(text: str) -> List[str]:
    return [
        segment
        for is_protected, segment in split_text_by_protected_symbols(text)
        if is_protected
    ]


def validate_translation_candidate(source_text: str, translated_text: str) -> List[str]:
    reasons: List[str] = []
    source_plain = strip_html_for_empty_check(source_text)
    translated_plain = strip_html_for_empty_check(translated_text)

    if source_plain and not translated_plain:
        reasons.append("empty_translation")

    if source_plain and re.search(r"[A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]", source_plain):
        if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", translated_plain):
            reasons.append("missing_translated_words")

    if protected_symbol_sequence(source_text) != protected_symbol_sequence(translated_text):
        reasons.append("protected_symbol_mismatch")

    if "?" not in source_text and "?" in translated_text:
        reasons.append("added_question")

    if not source_plain.lstrip().startswith(("-", "–", "—")) and translated_plain.lstrip().startswith(
        ("-", "–", "—")
    ):
        reasons.append("added_dialogue_marker")

    if has_german_signal(source_text) and text_similarity(source_text, translated_text) > 0.92:
        reasons.append("likely_untranslated")

    if len(source_plain) >= 20 and translated_plain:
        length_ratio = len(translated_plain) / max(len(source_plain), 1)
        if length_ratio < 0.35:
            reasons.append("translation_too_short")
        elif length_ratio > 2.8:
            reasons.append("translation_too_long")

    return reasons


def should_accept_review_translation(
    source_text: str,
    primary_translation: str,
    review_translation: str,
) -> bool:
    if validate_translation_candidate(source_text, review_translation):
        return False

    source_words = word_count(source_text)
    if source_words <= 3:
        primary_words = word_count(primary_translation)
        review_words = word_count(review_translation)
        if review_words >= max(primary_words + 1, source_words * 2 + 1):
            return False

    return True


def is_complex_text(text: str, args: argparse.Namespace) -> bool:
    plain = strip_html_for_empty_check(text)
    punctuation_count = len(re.findall(r"[,;:()\[\]{}]", plain))
    sentence_count = len(re.findall(r"[.!?]+(?:\s+|$)", plain))
    return (
        len(plain) >= args.quality_review_long_text_chars
        or word_count(plain) >= args.quality_review_long_text_words
        or punctuation_count >= args.quality_review_complex_punctuation
        or sentence_count >= args.quality_review_complex_sentences
    )


def is_low_resource_review_pair(source_lang_code: str, target_lang_code: str) -> bool:
    high_resource_pairs = {
        ("deu_Latn", "eng_Latn"),
        ("eng_Latn", "deu_Latn"),
    }
    return (source_lang_code, target_lang_code) not in high_resource_pairs


def make_backend_for_language_pair(
    backend: Ct2TranslatorBackend,
    source_lang_code: str,
    target_lang_code: str,
    *,
    batch_size: Optional[int] = None,
) -> Ct2TranslatorBackend:
    return Ct2TranslatorBackend(
        translator=backend.translator,
        tokenizer=backend.tokenizer,
        tokenizer_model=backend.tokenizer_model,
        ct2_model_dir=backend.ct2_model_dir,
        model_family=backend.model_family,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
        beam_size=backend.beam_size,
        batch_size=batch_size or backend.batch_size,
        max_input_tokens=backend.max_input_tokens,
        preserve_special_symbols=backend.preserve_special_symbols,
    )


def get_backend_translations_with_cache(
    texts: List[str],
    backend: Ct2TranslatorBackend,
    cache: Dict[str, str],
    stats: TranslationStats,
    *,
    cached_stat: Optional[str],
    generated_stat: Optional[str],
    higher_quality_cached_stat: Optional[str] = None,
    prefer_higher_quality_cache: bool = True,
) -> Dict[str, str]:
    translated_by_text: Dict[str, str] = {}
    texts_to_translate: List[str] = []
    seen_missing: set[str] = set()
    cache_index = (
        build_translation_cache_index(cache)
        if prefer_higher_quality_cache
        else {}
    )

    for text in texts:
        stronger_candidate = (
            find_higher_quality_cached_translation(text, backend, cache_index)
            if prefer_higher_quality_cache
            else None
        )
        if stronger_candidate is not None:
            translated_by_text[text] = stronger_candidate.translation
            if cached_stat:
                setattr(stats, cached_stat, getattr(stats, cached_stat) + 1)
            if higher_quality_cached_stat:
                setattr(
                    stats,
                    higher_quality_cached_stat,
                    getattr(stats, higher_quality_cached_stat) + 1,
                )
            continue

        cache_key = make_translation_cache_key(text, backend)
        if cache_key in cache:
            translated_by_text[text] = cache[cache_key]
            if cached_stat:
                setattr(stats, cached_stat, getattr(stats, cached_stat) + 1)
            continue

        if text not in seen_missing:
            texts_to_translate.append(text)
            seen_missing.add(text)

    for batch in chunk_list(texts_to_translate, backend.batch_size):
        translated_batch = translate_plain_batch_with_ct2(batch, backend)
        for source_text, translated_text in zip(batch, translated_batch):
            cache_key = make_translation_cache_key(source_text, backend)
            cache[cache_key] = translated_text
            translated_by_text[source_text] = translated_text

        if generated_stat:
            setattr(stats, generated_stat, getattr(stats, generated_stat) + len(batch))

    return translated_by_text

def quality_review_retranslations(
    texts: List[str],
    translated_by_text: Dict[str, str],
    primary_backend: Ct2TranslatorBackend,
    review_backend: Optional[Ct2TranslatorBackend],
    cache: Dict[str, str],
    stats: TranslationStats,
    glossary: Glossary,
    args: argparse.Namespace,
) -> Dict[str, str]:
    if args.quality_review_mode == "off" or not texts:
        return translated_by_text

    candidate_reasons: Dict[str, List[str]] = {}

    for source_text in texts:
        translated_text = translated_by_text.get(source_text, "")
        reasons = validate_translation_candidate(source_text, translated_text)
        if reasons:
            stats.validation_rejected_translations += 1

        source_word_count = word_count(source_text)
        if source_word_count <= args.quality_review_short_text_words:
            reasons.append("short_expression")
        if is_complex_text(source_text, args):
            reasons.append("long_or_complex_text")
        if is_low_resource_review_pair(
            primary_backend.source_lang_code,
            primary_backend.target_lang_code,
        ):
            reasons.append("low_resource_language_pair")

        if args.quality_review_mode == "all":
            reasons.append("review_all")

        if reasons:
            candidate_reasons[source_text] = sorted(set(reasons))

    if args.quality_review_round_trip:
        roundtrip_sources = [
            text
            for text in texts
            if text in translated_by_text and is_probably_translatable(translated_by_text[text])
        ]
        if roundtrip_sources:
            roundtrip_backend = make_backend_for_language_pair(
                primary_backend,
                primary_backend.target_lang_code,
                primary_backend.source_lang_code,
                batch_size=args.translation_batch_size,
            )
            translated_values = [translated_by_text[text] for text in roundtrip_sources]
            roundtrip_by_translation = get_backend_translations_with_cache(
                translated_values,
                roundtrip_backend,
                cache,
                stats,
                cached_stat=None,
                generated_stat=None,
                prefer_higher_quality_cache=args.prefer_higher_quality_cache,
            )
            stats.round_trip_checks += len(roundtrip_sources)
            for source_text in roundtrip_sources:
                roundtrip_text = roundtrip_by_translation.get(translated_by_text[source_text], "")
                if (
                    text_similarity(source_text, roundtrip_text)
                    < args.quality_review_round_trip_threshold
                ):
                    stats.round_trip_disagreements += 1
                    candidate_reasons.setdefault(source_text, []).append(
                        "round_trip_disagreement"
                    )

    if not candidate_reasons:
        return translated_by_text

    if review_backend is None:
        print(
            "Loading secondary quality-review backend "
            f"({args.quality_review_ct2_model_dir})..."
        )
        review_backend = create_quality_review_ct2_backend(args)

    print(
        "Quality review with secondary model: "
        f"{len(candidate_reasons)} suspicious translations."
    )
    stats.quality_review_candidates += len(candidate_reasons)
    reviewed_by_text = get_backend_translations_with_cache(
        list(candidate_reasons),
        review_backend,
        cache,
        stats,
        cached_stat="quality_review_cached_translations_used",
        generated_stat="quality_review_newly_generated_translations",
        higher_quality_cached_stat=(
            "quality_review_higher_quality_cached_translations_used"
        ),
        prefer_higher_quality_cache=args.prefer_higher_quality_cache,
    )

    reviewed_translated_by_text = dict(translated_by_text)
    for source_text, reviewed_translation in reviewed_by_text.items():
        primary_translation = translated_by_text.get(source_text, "")
        if not should_accept_review_translation(
            source_text,
            primary_translation,
            reviewed_translation,
        ):
            stats.quality_review_retranslations_rejected += 1
            continue

        processed_text, replacement_count = apply_glossary_postprocessing(
            reviewed_translation,
            glossary,
        )
        reviewed_translated_by_text[source_text] = processed_text
        stats.glossary_post_replacements += replacement_count
        stats.quality_review_retranslations += 1

    return reviewed_translated_by_text


def verify_ct2_backend_available(args: argparse.Namespace) -> Ct2TranslatorBackend:
    print(
        "Checking local translation backend "
        f"({args.ct2_device}, {args.ct2_compute_type})..."
    )
    backend = create_ct2_backend(args)
    translate_plain_batch_with_ct2(
        ["Hallo Welt"],
        backend,
    )
    print("Local translation backend is ready.")
    return backend


def is_cuda_runtime_load_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "cublas64_12.dll" in message
        or "cublas" in message and "not found" in message
        or "cuda" in message and "cannot be loaded" in message
    )


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
        Ct2TranslatorBackend(
            translator=translator,
            tokenizer=tokenizer,
            tokenizer_model="<direct>",
            ct2_model_dir="<direct>",
            model_family="marian",
            source_lang_code=DEFAULT_SOURCE_LANG_CODE,
            target_lang_code=DEFAULT_TARGET_LANG_CODE,
            beam_size=4,
            batch_size=batch_size,
            max_input_tokens=512,
            preserve_special_symbols=True,
        ),
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
    glossary: Glossary,
    args: argparse.Namespace,
) -> Dict[str, str]:
    translated_by_text: Dict[str, str] = {}
    texts_to_translate: List[str] = []
    reviewable_texts: List[str] = []
    seen_missing: set[str] = set()
    cache_index = (
        build_translation_cache_index(cache)
        if args.prefer_higher_quality_cache
        else {}
    )

    for text in texts:
        exact_override = exact_glossary_override(text, glossary)
        if exact_override is not None:
            translated_by_text[text] = exact_override
            stats.glossary_exact_overrides += 1
            continue

        reviewable_texts.append(text)

        stronger_candidate = (
            find_higher_quality_cached_translation(text, backend, cache_index)
            if args.prefer_higher_quality_cache
            else None
        )
        if stronger_candidate is not None:
            translated_text, replacement_count = apply_glossary_postprocessing(
                stronger_candidate.translation,
                glossary,
            )
            translated_by_text[text] = translated_text
            stats.cached_translations_used += 1
            stats.higher_quality_cached_translations_used += 1
            stats.glossary_post_replacements += replacement_count
            continue

        cache_key = make_translation_cache_key(
            text,
            backend,
        )

        if cache_key in cache:
            translated_text, replacement_count = apply_glossary_postprocessing(
                cache[cache_key],
                glossary,
            )
            translated_by_text[text] = translated_text
            stats.cached_translations_used += 1
            stats.glossary_post_replacements += replacement_count
            continue

        if text not in seen_missing:
            texts_to_translate.append(text)
            seen_missing.add(text)

    generated_since_save = 0
    for batch in chunk_list(texts_to_translate, backend.batch_size):
        translated_batch = translate_plain_batch_with_ct2(
            batch,
            backend,
        )

        for source_text, translated_text in zip(batch, translated_batch):
            cache_key = make_translation_cache_key(
                source_text,
                backend,
            )
            cache[cache_key] = translated_text
            processed_text, replacement_count = apply_glossary_postprocessing(
                translated_text,
                glossary,
            )
            translated_by_text[source_text] = processed_text
            stats.glossary_post_replacements += replacement_count

        stats.newly_generated_translations += len(batch)
        generated_since_save += len(batch)

        if generated_since_save >= TRANSLATION_CACHE_SAVE_INTERVAL:
            save_translation_cache(cache)
            generated_since_save = 0

    if args.quality_review_mode != "off" and reviewable_texts:
        translated_by_text = quality_review_retranslations(
            texts=reviewable_texts,
            translated_by_text=translated_by_text,
            primary_backend=backend,
            review_backend=None,
            cache=cache,
            stats=stats,
            glossary=glossary,
            args=args,
        )

    return translated_by_text


def translate_targets_with_ct2(
    targets: List[TranslationTarget],
    args: argparse.Namespace,
    backend: Optional[Ct2TranslatorBackend] = None,
) -> Tuple[Dict[Tuple[str, str, str], str], TranslationStats]:
    """
    Returns:
      (item_id, item_type, field) -> translated_text
    """
    if backend is None:
        backend = create_ct2_backend(args)

    print(f"Loading translation cache: {TRANSLATION_CACHE_FILE}")
    cache = load_translation_cache()
    glossary = load_glossary(Path(args.glossary_file), args.disable_glossary)
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
        glossary=glossary,
        args=args,
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
    print(
        "Higher-quality cached translations preferred: "
        f"{stats.higher_quality_cached_translations_used}"
    )
    print(f"Newly generated translations: {stats.newly_generated_translations}")
    print(f"Glossary exact overrides: {stats.glossary_exact_overrides}")
    print(f"Glossary post-processing replacements: {stats.glossary_post_replacements}")
    print(f"Validation-rejected translations: {stats.validation_rejected_translations}")
    print(f"Round-trip checks: {stats.round_trip_checks}")
    print(f"Round-trip disagreements: {stats.round_trip_disagreements}")
    print(f"Quality-review candidates: {stats.quality_review_candidates}")
    print(
        "Quality-review cached translations used: "
        f"{stats.quality_review_cached_translations_used}"
    )
    print(
        "Quality-review higher-quality cached translations preferred: "
        f"{stats.quality_review_higher_quality_cached_translations_used}"
    )
    print(
        "Quality-review newly generated translations: "
        f"{stats.quality_review_newly_generated_translations}"
    )
    print(f"Quality-review retranslations applied: {stats.quality_review_retranslations}")
    print(
        "Quality-review retranslations rejected: "
        f"{stats.quality_review_retranslations_rejected}"
    )

    return result, stats


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


def create_miro_item(
    miro_token: str,
    board_id: str,
    item_type: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    if item_type not in MIRO_UPDATE_ENDPOINTS:
        raise ApiError(f"Creating Miro item type is not supported: {item_type}")

    endpoint = MIRO_UPDATE_ENDPOINTS[item_type]
    encoded_board_id = quote(board_id, safe="=")
    url = f"{MIRO_API_BASE}/boards/{encoded_board_id}/{endpoint}"
    return request_json(
        "POST",
        url,
        headers=miro_headers(miro_token),
        json=payload,
    )


def patch_miro_item_payload(
    miro_token: str,
    board_id: str,
    item_id: str,
    item_type: str,
    payload: Dict[str, Any],
) -> None:
    if item_type not in MIRO_UPDATE_ENDPOINTS:
        raise ApiError(f"Updating Miro item type is not supported: {item_type}")

    endpoint = MIRO_UPDATE_ENDPOINTS[item_type]
    encoded_board_id = quote(board_id, safe="=")
    encoded_item_id = quote(item_id, safe="=")
    url = f"{MIRO_API_BASE}/boards/{encoded_board_id}/{endpoint}/{encoded_item_id}"

    response = requests.patch(
        url,
        headers=miro_headers(miro_token),
        json=payload,
        timeout=60,
    )
    if response.ok:
        return

    try:
        body = response.json()
    except Exception:
        body = response.text

    raise ApiError(
        f"PATCH payload failed for {item_type} {item_id}: "
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


def sanitize_board_id_for_filename(board_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", board_id).strip("_")


def default_sync_state_file(source_board_id: str, clone_board_id: str) -> Path:
    source = sanitize_board_id_for_filename(source_board_id)
    clone = sanitize_board_id_for_filename(clone_board_id)
    return Path(f"miro_sync_state_{source}__{clone}.json")


def get_sync_state_path(
    args: argparse.Namespace,
    source_board_id: str,
    clone_board_id: str,
) -> Path:
    if args.sync_state_file:
        return Path(args.sync_state_file)
    return default_sync_state_file(source_board_id, clone_board_id)


def load_sync_state(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Sync-state file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid sync-state JSON in {path}: {exc}") from exc

    if not isinstance(state, dict) or state.get("version") != SYNC_STATE_VERSION:
        raise ValueError(f"Unsupported sync-state format in {path}")

    state.setdefault("items", {})
    return state


def save_sync_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, ensure_ascii=False, indent=2, sort_keys=True)
        state_file.write("\n")
    tmp_path.replace(path)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def current_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def supported_item(item: Dict[str, Any]) -> bool:
    return item.get("type") in TRANSLATABLE_FIELDS


def item_by_id(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["id"]: item for item in items if item.get("id")}


def rounded_number(value: Any, precision: int) -> Optional[float]:
    if isinstance(value, (int, float)):
        return round(float(value), precision)
    return None


def rounded_object_values(value: Any, precision: int) -> Tuple[Tuple[str, Optional[float]], ...]:
    if not isinstance(value, dict):
        return ()
    return tuple(
        sorted((key, rounded_number(raw_value, precision)) for key, raw_value in value.items())
    )


def stable_json(value: Any) -> str:
    try:
        return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return "{}"


def supported_text_values(item: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    item_type = item.get("type")
    data = item.get("data") or {}
    values: List[Tuple[str, str]] = []
    for field in TRANSLATABLE_FIELDS.get(item_type, []):
        value = data.get(field)
        if isinstance(value, str):
            values.append((field, value))
    return tuple(values)


def item_fingerprint_for_copy_mapping(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        item.get("type"),
        rounded_object_values(item.get("position"), POSITION_PRECISION),
        rounded_object_values(item.get("geometry"), GEOMETRY_PRECISION),
        supported_text_values(item),
        stable_json(item.get("style")),
    )


def item_fingerprint_for_position_mapping(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        item.get("type"),
        rounded_object_values(item.get("position"), POSITION_PRECISION),
        rounded_object_values(item.get("geometry"), GEOMETRY_PRECISION),
        stable_json(item.get("style")),
    )


def build_mapping_by_fingerprint(
    source_items: List[Dict[str, Any]],
    clone_items: List[Dict[str, Any]],
    *,
    include_text: bool,
) -> Tuple[Dict[str, str], int, int, int]:
    fingerprint_fn = (
        item_fingerprint_for_copy_mapping
        if include_text
        else item_fingerprint_for_position_mapping
    )
    clone_by_fingerprint: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for clone_item in clone_items:
        if supported_item(clone_item):
            clone_by_fingerprint[fingerprint_fn(clone_item)].append(clone_item)

    mapping: Dict[str, str] = {}
    unmatched = 0
    ambiguous = 0

    for source_item in source_items:
        if not supported_item(source_item) or not source_item.get("id"):
            continue

        candidates = clone_by_fingerprint.get(fingerprint_fn(source_item), [])
        if len(candidates) == 1 and candidates[0].get("id"):
            mapping[source_item["id"]] = candidates[0]["id"]
        elif len(candidates) > 1:
            ambiguous += 1
        else:
            unmatched += 1

    clone_ids = set(mapping.values())
    unmatched_clone = sum(
        1
        for clone_item in clone_items
        if supported_item(clone_item) and clone_item.get("id") not in clone_ids
    )
    return mapping, unmatched, unmatched_clone, ambiguous


def build_source_to_clone_mapping_after_copy(
    source_items: List[Dict[str, Any]],
    clone_items: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], int, int, int]:
    return build_mapping_by_fingerprint(source_items, clone_items, include_text=True)


def make_empty_sync_state(source_board_id: str, clone_board_id: str) -> Dict[str, Any]:
    now = current_timestamp()
    return {
        "version": SYNC_STATE_VERSION,
        "source_board_id": source_board_id,
        "clone_board_id": clone_board_id,
        "created_at": now,
        "updated_at": now,
        "items": {},
    }


def sync_state_entry_for_item(
    source_item: Dict[str, Any],
    clone_item_id: str,
    *,
    translated_hashes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    item_type = source_item.get("type")
    data = source_item.get("data") or {}
    source_hashes: Dict[str, str] = {}
    for field in TRANSLATABLE_FIELDS.get(item_type, []):
        value = data.get(field)
        if isinstance(value, str):
            source_hashes[field] = hash_text(value)

    return {
        "clone_item_id": clone_item_id,
        "item_type": item_type,
        "last_source_text_hash_by_field": source_hashes,
        "last_translated_text_hash_by_field": translated_hashes or {},
        "last_seen": current_timestamp(),
    }


def create_sync_state_from_mapping(
    source_board_id: str,
    clone_board_id: str,
    source_items: List[Dict[str, Any]],
    mapping: Dict[str, str],
) -> Dict[str, Any]:
    state = make_empty_sync_state(source_board_id, clone_board_id)
    source_lookup = item_by_id(source_items)
    for source_item_id, clone_item_id in mapping.items():
        source_item = source_lookup.get(source_item_id)
        if source_item:
            state["items"][source_item_id] = sync_state_entry_for_item(
                source_item,
                clone_item_id,
            )
    return state


def initialize_sync_state_from_existing_clone(
    source_board_id: str,
    clone_board_id: str,
    source_items: List[Dict[str, Any]],
    clone_items: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], int, int, int]:
    mapping, unmatched_source, unmatched_clone, ambiguous = build_mapping_by_fingerprint(
        source_items,
        clone_items,
        include_text=False,
    )

    supported_source_count = sum(1 for item in source_items if supported_item(item))
    quality = len(mapping) / supported_source_count if supported_source_count else 1.0
    print("Sync-state initialization report:")
    print(f"  Mapped items: {len(mapping)}")
    print(f"  Unmatched source items: {unmatched_source}")
    print(f"  Unmatched clone items: {unmatched_clone}")
    print(f"  Ambiguous source items: {ambiguous}")
    print(f"  Mapping quality: {quality:.1%}")

    if quality < LOW_MAPPING_QUALITY_THRESHOLD and not args.force_initialize_sync_state:
        raise RuntimeError(
            "Sync-state initialization mapping quality is suspiciously low. "
            "Review board layout and rerun with --force-initialize-sync-state if "
            "you accept this best-effort mapping."
        )

    state = create_sync_state_from_mapping(
        source_board_id,
        clone_board_id,
        source_items,
        mapping,
    )
    return state, unmatched_source, unmatched_clone, ambiguous


def writable_position_from_source(position: Any) -> Dict[str, Any]:
    if not isinstance(position, dict):
        return {}

    return {
        key: position[key]
        for key in ("x", "y", "origin")
        if key in position and position[key] is not None
    }


def writable_geometry_from_source(
    item_type: Optional[str],
    geometry: Any,
) -> Dict[str, Any]:
    if not isinstance(geometry, dict):
        return {}

    result = {
        key: geometry[key]
        for key in ("width", "height", "rotation")
        if key in geometry and geometry[key] is not None
    }

    if item_type == "sticky_note" and "width" in result and "height" in result:
        result.pop("height")
    elif item_type == "text":
        result.pop("height", None)

    return result


def build_item_payload_from_source(
    source_item: Dict[str, Any],
    translated_fields: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    item_type = source_item.get("type")
    data = dict(source_item.get("data") or {})
    for field, value in (translated_fields or {}).items():
        data[field] = value

    if data:
        payload["data"] = data

    position = writable_position_from_source(source_item.get("position"))
    if position:
        payload["position"] = position

    geometry = writable_geometry_from_source(item_type, source_item.get("geometry"))
    if geometry:
        payload["geometry"] = geometry

    style = source_item.get("style")
    if isinstance(style, dict) and style:
        payload["style"] = style

    return payload


def translated_fields_for_item(
    item_id: str,
    item_type: str,
    translations: Dict[Tuple[str, str, str], str],
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for field in TRANSLATABLE_FIELDS.get(item_type, []):
        key = (item_id, item_type, field)
        if key in translations:
            result[field] = translations[key]
    return result


def update_sync_state_entry_hashes(
    state: Dict[str, Any],
    source_item: Dict[str, Any],
    clone_item_id: str,
    translated_fields: Dict[str, str],
) -> None:
    existing_entry = state.get("items", {}).get(source_item["id"]) or {}
    translated_hashes = dict(
        existing_entry.get("last_translated_text_hash_by_field") or {}
    )
    translated_hashes.update(
        {
            field: hash_text(value)
            for field, value in translated_fields.items()
        }
    )
    state["items"][source_item["id"]] = sync_state_entry_for_item(
        source_item,
        clone_item_id,
        translated_hashes=translated_hashes,
    )


def create_clone_item_from_source_item(
    miro_token: str,
    clone_board_id: str,
    source_item: Dict[str, Any],
    translated_fields: Dict[str, str],
    *,
    dry_run: bool,
) -> Optional[str]:
    source_item_id = source_item.get("id", "<unknown>")
    item_type = source_item.get("type")
    if item_type not in MIRO_UPDATE_ENDPOINTS:
        print(f"WARNING: Cannot create unsupported item type {item_type} ({source_item_id})")
        return None

    payload = build_item_payload_from_source(source_item, translated_fields)
    if dry_run:
        print(f"[DRY RUN] Would create {item_type} from source item {source_item_id}")
        return "__dry_run_clone_item__"

    try:
        created_item = create_miro_item(
            miro_token=miro_token,
            board_id=clone_board_id,
            item_type=item_type,
            payload=payload,
        )
    except Exception as exc:
        print(
            f"WARNING: Creating clone item for source {source_item_id} failed: {exc}",
            file=sys.stderr,
        )
        return None

    clone_item_id = created_item.get("id")
    if not clone_item_id:
        print(
            f"WARNING: Created {item_type} from source {source_item_id}, "
            "but Miro response did not include an id.",
            file=sys.stderr,
        )
        return None

    return clone_item_id


def update_clone_item_from_source_item(
    miro_token: str,
    clone_board_id: str,
    source_item: Dict[str, Any],
    clone_item_id: str,
    translated_fields: Dict[str, str],
    *,
    update_layout: bool,
    dry_run: bool,
) -> bool:
    item_type = source_item.get("type")
    if item_type not in MIRO_UPDATE_ENDPOINTS:
        print(f"WARNING: Cannot update unsupported item type {item_type}")
        return False

    if dry_run:
        layout_text = " with layout" if update_layout else ""
        print(f"[DRY RUN] Would update {item_type} {clone_item_id}{layout_text}")
        return True

    if update_layout:
        payload = build_item_payload_from_source(source_item, translated_fields)
        try:
            patch_miro_item_payload(
                miro_token=miro_token,
                board_id=clone_board_id,
                item_id=clone_item_id,
                item_type=item_type,
                payload=payload,
            )
            return True
        except Exception as exc:
            print(
                f"WARNING: Layout update failed for {item_type} {clone_item_id}; "
                f"falling back to text-only update: {exc}",
                file=sys.stderr,
            )

    if not translated_fields:
        if update_layout:
            print(
                f"WARNING: No translated fields available for {item_type} {clone_item_id}; "
                "skipping text-only fallback.",
                file=sys.stderr,
            )
        return False

    patch_miro_item(
        miro_token=miro_token,
        board_id=clone_board_id,
        item_id=clone_item_id,
        item_type=item_type,
        data_update=translated_fields,
    )
    return True


def delete_clone_item(
    miro_token: str,
    clone_board_id: str,
    clone_item_id: str,
    *,
    dry_run: bool,
) -> bool:
    if dry_run:
        print(f"[DRY RUN] Would delete clone item {clone_item_id}")
        return True

    try:
        delete_miro_item(miro_token, clone_board_id, clone_item_id)
        return True
    except Exception as exc:
        print(f"WARNING: Deleting clone item {clone_item_id} failed: {exc}", file=sys.stderr)
        return False


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

    total = len(grouped)
    print(f"Translation phase complete. Updating {total} Miro items...", flush=True)

    updated = 0
    started_at = time.monotonic()

    for planned_index, ((item_id, item_type), data_update) in enumerate(
        grouped.items(),
        start=1,
    ):
        if item_type not in MIRO_UPDATE_ENDPOINTS:
            continue

        if (
            planned_index == 1
            or planned_index % 10 == 0
            or planned_index == total
        ):
            elapsed = time.monotonic() - started_at
            print(
                f"  Starting item {planned_index}/{total} "
                f"({item_type}); elapsed {elapsed:.1f}s",
                flush=True,
            )

        if dry_run:
            print(
                f"[DRY RUN] Would update {item_type} {item_id}: "
                f"{list(data_update)}"
            )
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

        if updated % 10 == 0 or updated == total:
            elapsed = time.monotonic() - started_at
            rate = updated / elapsed if elapsed > 0 else 0.0
            remaining = total - updated
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"  {updated}/{total} items updated "
                f"({rate:.2f}/s, ETA {eta:.0f}s)",
                flush=True,
            )

    return updated


def make_default_clone_name(prefix: str) -> str:
    # Miro board names have a documented max length of 60 chars.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    name = f"{prefix} {timestamp}"
    return name[:60]


def print_item_counts(targets: List[TranslationTarget]) -> None:
    counts_by_type = defaultdict(int)
    for target in targets:
        counts_by_type[target.item_type] += 1

    print(f"Found {len(targets)} translatable text fields.")
    for item_type, count in sorted(counts_by_type.items()):
        print(f"  {item_type}: {count}")


def count_unsupported_items(items: List[Dict[str, Any]]) -> Counter:
    return Counter(
        item.get("type", "<unknown>")
        for item in items
        if item.get("type") not in TRANSLATABLE_FIELDS
    )


def print_unsupported_item_counts(items: List[Dict[str, Any]]) -> int:
    counts = count_unsupported_items(items)
    total = sum(counts.values())
    if total:
        print(f"Unsupported items skipped: {total}")
        for item_type, count in sorted(counts.items()):
            print(f"  {item_type}: {count}")
    return total


def print_run_summary(report: SyncReport) -> None:
    print()
    print("Summary")
    print(f"Mode: {report.mode}")
    print(f"Source board id: {report.source_board_id}")
    print(f"Clone board id: {report.clone_board_id}")
    print(f"Translatable source fields: {report.translatable_source_fields}")
    print(f"Mapped items updated: {report.mapped_items_updated}")
    print(f"New clone items created: {report.new_clone_items_created}")
    print(f"Stale clone items detected: {report.stale_clone_items_detected}")
    print(f"Stale clone items deleted: {report.stale_clone_items_deleted}")
    print(f"Unsupported items skipped: {report.unsupported_items_skipped}")
    print(f"Cached translations used: {report.cached_translations_used}")
    print(
        "Higher-quality cached translations preferred: "
        f"{report.higher_quality_cached_translations_used}"
    )
    print(
        "Manual clone translations imported into cache: "
        f"{report.manually_imported_clone_translations}"
    )
    print(f"Newly generated translations: {report.newly_generated_translations}")
    print(f"Glossary exact overrides: {report.glossary_exact_overrides}")
    print(f"Glossary post-processing replacements: {report.glossary_post_replacements}")
    print(f"Validation-rejected translations: {report.validation_rejected_translations}")
    print(f"Round-trip checks: {report.round_trip_checks}")
    print(f"Round-trip disagreements: {report.round_trip_disagreements}")
    print(f"Quality-review candidates: {report.quality_review_candidates}")
    print(
        "Quality-review cached translations used: "
        f"{report.quality_review_cached_translations_used}"
    )
    print(
        "Quality-review higher-quality cached translations preferred: "
        f"{report.quality_review_higher_quality_cached_translations_used}"
    )
    print(
        "Quality-review newly generated translations: "
        f"{report.quality_review_newly_generated_translations}"
    )
    print(f"Quality-review retranslations applied: {report.quality_review_retranslations}")
    print(
        "Quality-review retranslations rejected: "
        f"{report.quality_review_retranslations_rejected}"
    )
    if report.sync_state_file:
        print(f"Sync-state file: {report.sync_state_file}")


def update_report_from_translation_stats(
    report: SyncReport,
    stats: TranslationStats,
) -> None:
    report.cached_translations_used = stats.cached_translations_used
    report.higher_quality_cached_translations_used = (
        stats.higher_quality_cached_translations_used
    )
    report.newly_generated_translations = stats.newly_generated_translations
    report.glossary_exact_overrides = stats.glossary_exact_overrides
    report.glossary_post_replacements = stats.glossary_post_replacements
    report.validation_rejected_translations = stats.validation_rejected_translations
    report.round_trip_checks = stats.round_trip_checks
    report.round_trip_disagreements = stats.round_trip_disagreements
    report.quality_review_candidates = stats.quality_review_candidates
    report.quality_review_cached_translations_used = (
        stats.quality_review_cached_translations_used
    )
    report.quality_review_higher_quality_cached_translations_used = (
        stats.quality_review_higher_quality_cached_translations_used
    )
    report.quality_review_newly_generated_translations = (
        stats.quality_review_newly_generated_translations
    )
    report.quality_review_retranslations = stats.quality_review_retranslations
    report.quality_review_retranslations_rejected = (
        stats.quality_review_retranslations_rejected
    )


def run_rebuild_mode(
    args: argparse.Namespace,
    miro_token: str,
    translation_backend: Ct2TranslatorBackend,
) -> int:
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

    print("Reading items from source board...")
    source_items = get_all_items(miro_token=miro_token, board_id=source_board_id)
    print(f"Found {len(source_items)} items in source.")

    print("Reading items from cloned board...")
    clone_items = get_all_items(miro_token=miro_token, board_id=clone_board_id)
    print(f"Found {len(clone_items)} items in clone.")

    sync_state_path = get_sync_state_path(args, source_board_id, clone_board_id)
    mapping, unmatched_source, unmatched_clone, ambiguous = (
        build_source_to_clone_mapping_after_copy(source_items, clone_items)
    )
    print(f"Mapped source items to cloned items: {len(mapping)}")
    if unmatched_source or unmatched_clone or ambiguous:
        print(
            "WARNING: Sync-state mapping was incomplete: "
            f"{unmatched_source} unmatched source, {unmatched_clone} unmatched clone, "
            f"{ambiguous} ambiguous source items.",
            file=sys.stderr,
        )

    sync_state = create_sync_state_from_mapping(
        source_board_id,
        clone_board_id,
        source_items,
        mapping,
    )
    save_sync_state(sync_state_path, sync_state)
    print(f"Saved sync-state file: {sync_state_path}")

    targets = collect_translation_targets(clone_items)
    print_item_counts(targets)

    report = SyncReport(
        mode="rebuild",
        source_board_id=source_board_id,
        clone_board_id=clone_board_id,
        sync_state_file=sync_state_path,
        translatable_source_fields=len(targets),
        unsupported_items_skipped=print_unsupported_item_counts(source_items),
        unmapped_source_items=unmatched_source,
        ambiguous_items=ambiguous,
    )

    if not targets:
        print("No translatable fields found.")
        print_run_summary(report)
        return 0

    if args.dry_run:
        print("[DRY RUN] Skipping Miro write-access preflight item create/update/delete.")
    else:
        verify_miro_write_access(miro_token=miro_token, board_id=clone_board_id)

    translations, stats = translate_targets_with_ct2(
        targets=targets,
        args=args,
        backend=translation_backend,
    )
    update_report_from_translation_stats(report, stats)

    updated = apply_translations_to_miro(
        miro_token=miro_token,
        board_id=clone_board_id,
        translations=translations,
        dry_run=args.dry_run,
    )
    report.updated_miro_items = updated
    report.mapped_items_updated = updated

    source_lookup = item_by_id(source_items)
    for source_item_id, entry in sync_state.get("items", {}).items():
        source_item = source_lookup.get(source_item_id)
        if not source_item:
            continue
        clone_item_id = entry.get("clone_item_id")
        item_type = source_item.get("type")
        if not clone_item_id or not item_type:
            continue
        translated_fields = translated_fields_for_item(
            clone_item_id,
            item_type,
            translations,
        )
        update_sync_state_entry_hashes(
            sync_state,
            source_item,
            clone_item_id,
            translated_fields,
        )

    sync_state["updated_at"] = current_timestamp()
    save_sync_state(sync_state_path, sync_state)

    print()
    print("Done.")
    print(f"English clone board id: {clone_board_id}")
    if view_link:
        print(f"English clone link: {view_link}")
    print_run_summary(report)
    return 0


def build_update_translation_targets(
    source_items: List[Dict[str, Any]],
    clone_items: List[Dict[str, Any]],
    sync_state: Dict[str, Any],
    *,
    update_layout: bool,
) -> Tuple[
    List[TranslationTarget],
    List[Tuple[Dict[str, Any], str]],
    List[Dict[str, Any]],
    List[str],
    int,
]:
    """
    Build the update plan.

    Every mapped translatable source field is resolved through the cache again,
    even when the source text itself is unchanged. This is required because the
    desired translation can change independently of the source text, for
    example after:

      * replacing the translation model,
      * adding a stronger-model cache entry,
      * manually correcting the cache, or
      * changing the glossary.

    Network writes are filtered later by comparing the resolved translation
    against the text currently present on the clone board. Therefore, unchanged
    clone items still incur no PATCH request.
    """
    source_lookup = item_by_id(source_items)
    clone_lookup = item_by_id(clone_items)

    targets: List[TranslationTarget] = []
    mapped_items: List[Tuple[Dict[str, Any], str]] = []
    stale_source_ids: List[str] = []
    mapped_items_with_unchanged_source = 0

    for source_item_id, entry in sync_state.get("items", {}).items():
        source_item = source_lookup.get(source_item_id)
        clone_item_id = entry.get("clone_item_id")
        if not source_item:
            stale_source_ids.append(source_item_id)
            continue
        if not clone_item_id or clone_item_id not in clone_lookup:
            print(
                f"WARNING: Mapped clone item missing for source {source_item_id}: "
                f"{clone_item_id}",
                file=sys.stderr,
            )
            continue
        if not supported_item(source_item):
            continue

        data = source_item.get("data") or {}
        item_type = source_item.get("type")
        previous_hashes = entry.get("last_source_text_hash_by_field") or {}

        translatable_fields: List[Tuple[str, str]] = []
        source_changed = False

        for field in TRANSLATABLE_FIELDS.get(item_type, []):
            value = data.get(field)
            if not isinstance(value, str) or not is_probably_translatable(value):
                continue

            translatable_fields.append((field, value))
            if previous_hashes.get(field) != hash_text(value):
                source_changed = True

        # Layout-only items still need scheduling with --update-layout.
        if translatable_fields or update_layout:
            mapped_items.append((source_item, clone_item_id))

        if translatable_fields and not source_changed:
            mapped_items_with_unchanged_source += 1

        for field, value in translatable_fields:
            targets.append(
                TranslationTarget(
                    item_id=clone_item_id,
                    item_type=item_type,
                    field=field,
                    source_text=value,
                    is_html=looks_like_html(value),
                )
            )

    mapped_source_ids = set(sync_state.get("items", {}).keys())
    new_source_items = [
        item
        for item in source_items
        if supported_item(item) and item.get("id") not in mapped_source_ids
    ]

    for source_item in new_source_items:
        data = source_item.get("data") or {}
        item_type = source_item.get("type")
        source_item_id = source_item.get("id")
        if not source_item_id:
            continue

        for field in TRANSLATABLE_FIELDS.get(item_type, []):
            value = data.get(field)
            if isinstance(value, str) and is_probably_translatable(value):
                targets.append(
                    TranslationTarget(
                        item_id=source_item_id,
                        item_type=item_type,
                        field=field,
                        source_text=value,
                        is_html=looks_like_html(value),
                    )
                )

    return (
        targets,
        mapped_items,
        new_source_items,
        stale_source_ids,
        mapped_items_with_unchanged_source,
    )


def translated_fields_needing_text_update(
    clone_item: Dict[str, Any],
    item_type: str,
    desired_fields: Dict[str, str],
) -> Dict[str, str]:
    """
    Return only translated fields whose desired value differs from the value
    currently stored on the clone board.
    """
    clone_data = clone_item.get("data") or {}
    return {
        field: desired_text
        for field, desired_text in desired_fields.items()
        if clone_data.get(field) != desired_text
    }


def make_quality_review_cache_backend(
    args: argparse.Namespace,
) -> Ct2TranslatorBackend:
    """Build review-backend metadata without loading the model."""
    return Ct2TranslatorBackend(
        translator=None,
        tokenizer=None,
        tokenizer_model=args.quality_review_hf_tokenizer_model,
        ct2_model_dir=args.quality_review_ct2_model_dir,
        model_family=resolve_ct2_model_family(
            args.quality_review_ct2_model_family,
            args.quality_review_hf_tokenizer_model,
            args.quality_review_ct2_model_dir,
        ),
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
        beam_size=args.quality_review_beam_size,
        batch_size=args.quality_review_batch_size,
        max_input_tokens=args.quality_review_max_input_tokens,
        preserve_special_symbols=args.preserve_special_symbols,
    )


def cache_units_from_manual_clone_field(
    source_text: str,
    clone_text: str,
) -> Optional[List[Tuple[str, str]]]:
    """
    Return source/cache-unit pairs for a manually edited clone field.

    Plain fields produce one pair. HTML fields are paired by text-node order and
    are skipped when the source and clone structures do not match safely.
    """
    if not looks_like_html(source_text):
        return [(source_text, clone_text)]

    BeautifulSoup, NavigableString = import_beautifulsoup()
    source_soup = BeautifulSoup(source_text, "html.parser")
    clone_soup = BeautifulSoup(clone_text, "html.parser")

    def units(soup: Any, source_side: bool) -> List[str]:
        result: List[str] = []
        for node in soup.find_all(string=True):
            if not isinstance(node, NavigableString):
                continue
            _, core_text, _ = split_surrounding_whitespace(str(node))
            if not core_text:
                continue
            if source_side and not is_probably_translatable(core_text):
                continue
            result.append(core_text)
        return result

    source_units = units(source_soup, True)
    clone_units = units(clone_soup, False)
    if len(source_units) != len(clone_units):
        return None
    return list(zip(source_units, clone_units))


def update_all_compatible_cache_entries(
    cache: Dict[str, str],
    *,
    source_text: str,
    manual_translation: str,
    primary_backend: Ct2TranslatorBackend,
    review_backend: Ct2TranslatorBackend,
) -> int:
    """
    Store one manual translation under current primary/review keys and every
    compatible historical key. This prevents a stronger old cache entry from
    overriding the manual edit on the next run.
    """
    identity = cache_compatibility_identity(
        source_text=source_text,
        source_lang_code=primary_backend.source_lang_code,
        target_lang_code=primary_backend.target_lang_code,
        model_family=primary_backend.model_family,
        preserve_special_symbols=primary_backend.preserve_special_symbols,
    )

    keys_to_update: set[str] = {
        make_translation_cache_key(source_text, primary_backend),
        make_translation_cache_key(source_text, review_backend),
    }

    for cache_key in list(cache):
        metadata = parse_translation_cache_key(cache_key)
        if metadata is None:
            continue
        candidate_identity = (
            metadata.get("source_text"),
            metadata.get("source_lang_code"),
            metadata.get("target_lang_code"),
            metadata.get("model_family"),
            metadata.get("preserve_special_symbols"),
            metadata.get("symbol_protection_version", 1),
        )
        if candidate_identity == identity:
            keys_to_update.add(cache_key)

    changed = 0
    for cache_key in keys_to_update:
        if cache.get(cache_key) != manual_translation:
            cache[cache_key] = manual_translation
            changed += 1
    return changed


def import_manual_clone_translations(
    *,
    source_items: List[Dict[str, Any]],
    clone_items: List[Dict[str, Any]],
    sync_state: Dict[str, Any],
    primary_backend: Ct2TranslatorBackend,
    args: argparse.Namespace,
) -> Tuple[int, int, int, int]:
    """
    Import clone fields that changed since the last successful sync.

    Only fields with a previous translated hash are considered. This prevents a
    newly initialized or incomplete sync-state from importing the entire clone
    as if every field had been manually edited.
    """
    source_lookup = item_by_id(source_items)
    clone_lookup = item_by_id(clone_items)
    review_backend = make_quality_review_cache_backend(args)
    cache = load_translation_cache()

    imported_fields = 0
    imported_units = 0
    skipped_missing_hash = 0
    skipped_html_mismatch = 0

    for source_item_id, state_entry in sync_state.get("items", {}).items():
        source_item = source_lookup.get(source_item_id)
        clone_item_id = state_entry.get("clone_item_id")
        clone_item = clone_lookup.get(clone_item_id)
        if not source_item or not clone_item:
            continue

        item_type = source_item.get("type")
        source_data = source_item.get("data") or {}
        clone_data = clone_item.get("data") or {}
        translated_hashes = dict(
            state_entry.get("last_translated_text_hash_by_field") or {}
        )

        for field in TRANSLATABLE_FIELDS.get(item_type, []):
            source_value = source_data.get(field)
            clone_value = clone_data.get(field)
            if (
                not isinstance(source_value, str)
                or not isinstance(clone_value, str)
                or not is_probably_translatable(source_value)
            ):
                continue

            previous_hash = translated_hashes.get(field)
            if not previous_hash:
                skipped_missing_hash += 1
                continue
            if hash_text(clone_value) == previous_hash:
                continue

            units = cache_units_from_manual_clone_field(source_value, clone_value)
            if units is None:
                skipped_html_mismatch += 1
                print(
                    "WARNING: Manual HTML translation could not be imported for "
                    f"{item_type} {clone_item_id} field {field}: text-node counts "
                    "do not match.",
                    file=sys.stderr,
                )
                continue

            for source_unit, manual_translation in units:
                update_all_compatible_cache_entries(
                    cache,
                    source_text=source_unit,
                    manual_translation=manual_translation,
                    primary_backend=primary_backend,
                    review_backend=review_backend,
                )
                imported_units += 1

            translated_hashes[field] = hash_text(clone_value)
            state_entry["last_translated_text_hash_by_field"] = translated_hashes
            imported_fields += 1

    if imported_fields:
        save_translation_cache(cache)

    return (
        imported_fields,
        imported_units,
        skipped_missing_hash,
        skipped_html_mismatch,
    )


def run_update_existing_clone_mode(
    args: argparse.Namespace,
    miro_token: str,
    translation_backend: Ct2TranslatorBackend,
) -> int:
    source_board_id = extract_board_id(args.source_board)
    if not args.clone_board:
        raise ValueError("--clone-board is required with --update-existing-clone")
    clone_board_id = extract_board_id(args.clone_board)
    sync_state_path = get_sync_state_path(args, source_board_id, clone_board_id)

    print(f"Source board: {source_board_id}")
    print(f"Existing clone board: {clone_board_id}")
    if args.update_layout:
        print("Update mode: text content and layout.")
    else:
        print("Update mode: text content only. Use --update-layout to sync layout.")

    print("Reading items from source board...")
    source_items = get_all_items(miro_token=miro_token, board_id=source_board_id)
    print(f"Found {len(source_items)} items in source.")

    print("Reading items from existing clone board...")
    clone_items = get_all_items(miro_token=miro_token, board_id=clone_board_id)
    print(f"Found {len(clone_items)} items in clone.")

    report = SyncReport(
        mode="update-existing-clone",
        source_board_id=source_board_id,
        clone_board_id=clone_board_id,
        sync_state_file=sync_state_path,
        unsupported_items_skipped=print_unsupported_item_counts(source_items),
    )

    if sync_state_path.exists():
        sync_state = load_sync_state(sync_state_path)
    else:
        if not args.initialize_sync_state:
            raise FileNotFoundError(
                "Update mode needs a sync-state file. Run rebuild mode once, "
                "provide --sync-state-file, or rerun with --initialize-sync-state "
                "to create a best-effort mapping for an existing translated clone."
            )

        sync_state, unmatched_source, unmatched_clone, ambiguous = (
            initialize_sync_state_from_existing_clone(
                source_board_id,
                clone_board_id,
                source_items,
                clone_items,
                args,
            )
        )
        report.unmapped_source_items = unmatched_source
        report.ambiguous_items = ambiguous
        save_sync_state(sync_state_path, sync_state)
        print(f"Saved initialized sync-state file: {sync_state_path}")
        print("Initialization complete. Review the state file, then rerun update mode.")
        print_run_summary(report)
        return 0

    if sync_state.get("source_board_id") != source_board_id:
        raise ValueError("Sync-state source_board_id does not match --source-board")
    if sync_state.get("clone_board_id") != clone_board_id:
        raise ValueError("Sync-state clone_board_id does not match --clone-board")

    if args.import_manual_clone_translations:
        print(
            "Importing manual translation changes from the clone into the cache...",
            flush=True,
        )
        (
            imported_fields,
            imported_units,
            skipped_missing_hash,
            skipped_html_mismatch,
        ) = import_manual_clone_translations(
            source_items=source_items,
            clone_items=clone_items,
            sync_state=sync_state,
            primary_backend=translation_backend,
            args=args,
        )
        report.manually_imported_clone_translations = imported_fields
        print(
            f"Manual clone fields imported: {imported_fields} "
            f"({imported_units} cache text units)",
            flush=True,
        )
        if skipped_missing_hash:
            print(
                "Fields not imported because no previous translated hash was "
                f"available: {skipped_missing_hash}",
                flush=True,
            )
        if skipped_html_mismatch:
            print(
                "Manual HTML fields skipped because their source/clone text-node "
                f"structures differed: {skipped_html_mismatch}",
                flush=True,
            )
        if imported_fields and not args.dry_run:
            save_sync_state(sync_state_path, sync_state)
            print(
                f"Updated sync-state after manual import: {sync_state_path}",
                flush=True,
            )

    (
        targets,
        mapped_items,
        new_source_items,
        stale_source_ids,
        mapped_items_with_unchanged_source,
    ) = build_update_translation_targets(
        source_items,
        clone_items,
        sync_state,
        update_layout=args.update_layout,
    )
    report.translatable_source_fields = len(targets)
    report.stale_clone_items_detected = len(stale_source_ids)

    print_item_counts(targets)
    if not args.update_layout:
        print(
            "Mapped items with unchanged source text: "
            f"{mapped_items_with_unchanged_source}",
            flush=True,
        )
        print(
            "Their cached/desired translations will still be checked against "
            "the clone; only actual differences will be written.",
            flush=True,
        )
    print(
        "Mapped items whose translations will be checked: "
        f"{len(mapped_items)}",
        flush=True,
    )
    if stale_source_ids:
        print(f"Stale mapped source items detected: {len(stale_source_ids)}")
    if new_source_items:
        print(f"New supported source items without mapping: {len(new_source_items)}")

    if args.dry_run:
        print("[DRY RUN] Skipping Miro write-access preflight item create/update/delete.")
    else:
        verify_miro_write_access(miro_token=miro_token, board_id=clone_board_id)

    if targets:
        translations, stats = translate_targets_with_ct2(
            targets=targets,
            args=args,
            backend=translation_backend,
        )
        update_report_from_translation_stats(report, stats)
    else:
        translations = {}

    clone_lookup = item_by_id(clone_items)
    mapped_updates: List[
        Tuple[Dict[str, Any], str, Dict[str, str]]
    ] = []
    mapped_items_already_current = 0
    mapped_items_without_translatable_text = 0

    print(
        "Comparing resolved translations with the current clone text...",
        flush=True,
    )

    for source_item, clone_item_id in mapped_items:
        item_type = source_item.get("type")
        desired_fields = translated_fields_for_item(
            clone_item_id,
            item_type,
            translations,
        )

        if args.update_layout:
            # Layout synchronization intentionally processes every mapped item.
            mapped_updates.append(
                (source_item, clone_item_id, desired_fields)
            )
            continue

        if not desired_fields:
            mapped_items_without_translatable_text += 1
            continue

        clone_item = clone_lookup.get(clone_item_id)
        if clone_item is None:
            continue

        fields_to_update = translated_fields_needing_text_update(
            clone_item,
            item_type,
            desired_fields,
        )

        if fields_to_update:
            mapped_updates.append(
                (source_item, clone_item_id, fields_to_update)
            )
        else:
            mapped_items_already_current += 1
            if not args.dry_run:
                # The clone already contains the desired result, so recording
                # the current hashes is safe without issuing a network write.
                update_sync_state_entry_hashes(
                    sync_state,
                    source_item,
                    clone_item_id,
                    desired_fields,
                )

    print(
        "Mapped items already matching the desired translation: "
        f"{mapped_items_already_current}",
        flush=True,
    )
    if mapped_items_without_translatable_text:
        print(
            "Mapped items without translatable text: "
            f"{mapped_items_without_translatable_text}",
            flush=True,
        )
    print(
        "Mapped items requiring a Miro text update: "
        f"{len(mapped_updates)}",
        flush=True,
    )

    mapped_total = len(mapped_updates)
    if mapped_total:
        print(
            f"Updating {mapped_total} mapped Miro items...",
            flush=True,
        )

    mapped_update_started_at = time.monotonic()
    for mapped_index, (
        source_item,
        clone_item_id,
        translated_fields,
    ) in enumerate(
        mapped_updates,
        start=1,
    ):
        item_type = source_item.get("type")
        if (
            mapped_index == 1
            or mapped_index % 10 == 0
            or mapped_index == mapped_total
        ):
            elapsed = time.monotonic() - mapped_update_started_at
            print(
                f"  Starting mapped item {mapped_index}/{mapped_total} "
                f"({item_type}); elapsed {elapsed:.1f}s",
                flush=True,
            )

        if update_clone_item_from_source_item(
            miro_token,
            clone_board_id,
            source_item,
            clone_item_id,
            translated_fields,
            update_layout=args.update_layout,
            dry_run=args.dry_run,
        ):
            report.mapped_items_updated += 1
            if not args.dry_run:
                # Hash all desired fields, not just the subset sent in this
                # PATCH, so the sync-state remains complete for multi-field
                # items such as cards.
                all_desired_fields = translated_fields_for_item(
                    clone_item_id,
                    item_type,
                    translations,
                )
                update_sync_state_entry_hashes(
                    sync_state,
                    source_item,
                    clone_item_id,
                    all_desired_fields,
                )

        if mapped_index % 10 == 0 or mapped_index == mapped_total:
            elapsed = time.monotonic() - mapped_update_started_at
            rate = mapped_index / elapsed if elapsed > 0 else 0.0
            remaining = mapped_total - mapped_index
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"  {mapped_index}/{mapped_total} mapped items processed "
                f"({rate:.2f}/s, ETA {eta:.0f}s)",
                flush=True,
            )

    new_total = len(new_source_items)
    if new_total:
        print(f"Creating {new_total} new clone items...", flush=True)

    for new_index, source_item in enumerate(new_source_items, start=1):
        if new_index == 1 or new_index % 10 == 0 or new_index == new_total:
            print(
                f"  Creating new item {new_index}/{new_total} "
                f"({source_item.get('type')})",
                flush=True,
            )

        source_item_id = source_item.get("id")
        translated_fields = translated_fields_for_item(
            source_item_id,
            source_item.get("type"),
            translations,
        )
        clone_item_id = create_clone_item_from_source_item(
            miro_token,
            clone_board_id,
            source_item,
            translated_fields,
            dry_run=args.dry_run,
        )
        if clone_item_id:
            report.new_clone_items_created += 1
            if not args.dry_run:
                update_sync_state_entry_hashes(
                    sync_state,
                    source_item,
                    clone_item_id,
                    translated_fields,
                )

    if stale_source_ids:
        for stale_source_id in stale_source_ids:
            entry = sync_state.get("items", {}).get(stale_source_id) or {}
            clone_item_id = entry.get("clone_item_id")
            if not clone_item_id:
                continue
            if args.delete_missing_items:
                if delete_clone_item(
                    miro_token,
                    clone_board_id,
                    clone_item_id,
                    dry_run=args.dry_run,
                ):
                    report.stale_clone_items_deleted += 1
                    if not args.dry_run:
                        sync_state["items"].pop(stale_source_id, None)
            else:
                print(
                    f"Stale source item {stale_source_id} maps to clone item "
                    f"{clone_item_id}; not deleting without --delete-missing-items."
                )

    if not args.dry_run:
        sync_state["updated_at"] = current_timestamp()
        save_sync_state(sync_state_path, sync_state)

    print()
    print("Done.")
    print_run_summary(report)
    return 0


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
        "--update-existing-clone",
        action="store_true",
        help="Update an existing translated clone instead of creating a new clone.",
    )
    parser.add_argument(
        "--clone-board",
        default=None,
        help="Existing translated clone board id or URL, required for update mode.",
    )
    parser.add_argument(
        "--sync-state-file",
        default=None,
        help="Path to the sync-state JSON file. Defaults to a deterministic filename.",
    )
    parser.add_argument(
        "--initialize-sync-state",
        action="store_true",
        help="Initialize sync-state for an existing translated clone and exit.",
    )
    parser.add_argument(
        "--force-initialize-sync-state",
        action="store_true",
        help="Allow low-confidence sync-state initialization.",
    )
    parser.add_argument(
        "--delete-missing-items",
        action="store_true",
        help="Delete mapped clone items whose source items no longer exist.",
    )
    parser.add_argument(
        "--import-manual-clone-translations",
        action="store_true",
        help=(
            "With --update-existing-clone, detect English clone fields manually "
            "edited since the last successful sync and import them into all "
            "compatible primary/review cache entries before normal translation "
            "resolution. The manual clone text therefore wins instead of being "
            "overwritten. Only fields with a known previous translated hash are "
            "imported."
        ),
    )
    parser.add_argument(
        "--update-layout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also update position/geometry/style in update mode when supported. "
            "Default: false."
        ),
    )
    parser.add_argument(
        "--text-only-update",
        dest="update_layout",
        action="store_false",
        help="Update only translated text in update mode. Alias for --no-update-layout.",
    )
    parser.add_argument(
        "--sync-supported-items-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only sync supported translatable item types. Default: true.",
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
        "--ct2-model-family",
        default=DEFAULT_CT2_MODEL_FAMILY,
        choices=["auto", "marian", "nllb"],
        help=(
            "CTranslate2 model family. Use nllb for facebook/nllb models and "
            "marian for Helsinki/OPUS-MT models. Default: nllb."
        ),
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
        "--source-lang-code",
        default=DEFAULT_SOURCE_LANG_CODE,
        help="Tokenizer source language code for multilingual models. Default: deu_Latn.",
    )
    parser.add_argument(
        "--target-lang-code",
        default=DEFAULT_TARGET_LANG_CODE,
        help="Tokenizer target language code for multilingual models. Default: eng_Latn.",
    )
    parser.add_argument(
        "--ct2-device",
        default="cpu",
        help='CTranslate2 device, e.g. "cpu" or "cuda". Default: "cpu".',
    )
    parser.add_argument(
        "--ct2-compute-type",
        default=None,
        help=(
            'CTranslate2 compute type, e.g. "int8", "int8_float16", '
            '"float32", or "float16". Default: float16 on CUDA, int8 on CPU.'
        ),
    )
    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=16,
        help="Number of plain text units translated per CTranslate2 batch. Default: 16.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam search size for translation quality/speed tradeoff. Default: 4.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=512,
        help="Maximum input tokens per translation chunk. Default: 512.",
    )
    parser.add_argument(
        "--preserve-special-symbols",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Protect emojis, pictograms, dingbats, and similar symbols from the "
            "translator and reinsert them unchanged. Default: true."
        ),
    )
    parser.add_argument(
        "--prefer-higher-quality-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse a compatible cache entry produced by a strictly stronger "
            "model before translating with the selected backend. A cached NLLB "
            "3.3B result therefore takes precedence over generation and over an "
            "older exact NLLB 1.3B cache entry. Default: true."
        ),
    )
    parser.add_argument(
        "--quality-review-mode",
        default="suspicious",
        choices=["off", "suspicious", "all"],
        help=(
            "Use the secondary quality model to replace suspicious translations, "
            "all translations, or none. Default: suspicious."
        ),
    )
    parser.add_argument(
        "--quality-review-ct2-model-dir",
        default=DEFAULT_QUALITY_REVIEW_CT2_MODEL_DIR,
        help=(
            "Converted CTranslate2 directory for the secondary quality model. "
            f"Default: {DEFAULT_QUALITY_REVIEW_CT2_MODEL_DIR}"
        ),
    )
    parser.add_argument(
        "--quality-review-hf-tokenizer-model",
        default=DEFAULT_QUALITY_REVIEW_HF_TOKENIZER_MODEL,
        help=(
            "Hugging Face tokenizer model for the secondary quality model. "
            f"Default: {DEFAULT_QUALITY_REVIEW_HF_TOKENIZER_MODEL}"
        ),
    )
    parser.add_argument(
        "--quality-review-ct2-model-family",
        default="nllb",
        choices=["auto", "marian", "nllb"],
        help="Secondary quality model family. Default: nllb.",
    )
    parser.add_argument(
        "--quality-review-ct2-device",
        default=None,
        help="Secondary quality model device. Default: inherit --ct2-device.",
    )
    parser.add_argument(
        "--quality-review-ct2-compute-type",
        default=None,
        help="Secondary quality model compute type. Default: inherit --ct2-compute-type.",
    )
    parser.add_argument(
        "--quality-review-batch-size",
        type=int,
        default=2,
        help="Secondary quality model batch size. Default: 2.",
    )
    parser.add_argument(
        "--quality-review-beam-size",
        type=int,
        default=4,
        help="Secondary quality model beam size. Default: 4.",
    )
    parser.add_argument(
        "--quality-review-max-input-tokens",
        type=int,
        default=512,
        help="Secondary quality model maximum input tokens per chunk. Default: 512.",
    )
    parser.add_argument(
        "--quality-review-round-trip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NLLB round-trip disagreement as a quality-review signal. Default: true.",
    )
    parser.add_argument(
        "--quality-review-round-trip-threshold",
        type=float,
        default=0.55,
        help="Minimum source vs round-trip similarity before review. Default: 0.55.",
    )
    parser.add_argument(
        "--quality-review-short-text-words",
        type=int,
        default=3,
        help="Review texts with this many words or fewer. Default: 3.",
    )
    parser.add_argument(
        "--quality-review-long-text-chars",
        type=int,
        default=240,
        help="Review texts at or above this character count. Default: 240.",
    )
    parser.add_argument(
        "--quality-review-long-text-words",
        type=int,
        default=35,
        help="Review texts at or above this word count. Default: 35.",
    )
    parser.add_argument(
        "--quality-review-complex-punctuation",
        type=int,
        default=3,
        help="Review texts with this many complex punctuation marks. Default: 3.",
    )
    parser.add_argument(
        "--quality-review-complex-sentences",
        type=int,
        default=2,
        help="Review texts with this many sentences or more. Default: 2.",
    )
    parser.add_argument(
        "--glossary-file",
        default=DEFAULT_GLOSSARY_FILE,
        help=f"Glossary JSON file. Default: {DEFAULT_GLOSSARY_FILE}",
    )
    parser.add_argument(
        "--disable-glossary",
        action="store_true",
        help="Disable glossary exact overrides and post-processing.",
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

    if args.ct2_compute_type is None:
        args.ct2_compute_type = (
            "float16" if args.ct2_device.lower() == "cuda" else "int8"
        )
    if args.quality_review_ct2_device is None:
        args.quality_review_ct2_device = args.ct2_device
    if args.quality_review_ct2_compute_type is None:
        args.quality_review_ct2_compute_type = args.ct2_compute_type

    miro_token = os.getenv("MIRO_ACCESS_TOKEN")

    if not miro_token:
        print("Missing MIRO_ACCESS_TOKEN in environment or .env", file=sys.stderr)
        return 2

    if args.translation_batch_size < 1:
        print("--translation-batch-size must be at least 1", file=sys.stderr)
        return 2

    if args.beam_size < 1:
        print("--beam-size must be at least 1", file=sys.stderr)
        return 2

    if args.max_input_tokens < 1:
        print("--max-input-tokens must be at least 1", file=sys.stderr)
        return 2

    if args.quality_review_batch_size < 1:
        print("--quality-review-batch-size must be at least 1", file=sys.stderr)
        return 2

    if args.quality_review_beam_size < 1:
        print("--quality-review-beam-size must be at least 1", file=sys.stderr)
        return 2

    if args.quality_review_max_input_tokens < 1:
        print("--quality-review-max-input-tokens must be at least 1", file=sys.stderr)
        return 2

    if not 0 <= args.quality_review_round_trip_threshold <= 1:
        print(
            "--quality-review-round-trip-threshold must be between 0 and 1",
            file=sys.stderr,
        )
        return 2

    if (
        args.quality_review_mode != "off"
        and not args.initialize_sync_state
        and not Path(args.quality_review_ct2_model_dir).is_dir()
    ):
        print(
            "Missing converted secondary quality-review model directory: "
            f"{args.quality_review_ct2_model_dir}\n"
            "Create it with:\n"
            f"  {QUALITY_REVIEW_CT2_CONVERSION_COMMAND}\n"
            "Or disable secondary review with --quality-review-mode off.",
            file=sys.stderr,
        )
        return 2

    if args.update_existing_clone and not args.clone_board:
        print("--clone-board is required with --update-existing-clone", file=sys.stderr)
        return 2

    if args.update_existing_clone and not args.initialize_sync_state:
        source_board_id = extract_board_id(args.source_board)
        clone_board_id = extract_board_id(args.clone_board)
        sync_state_path = get_sync_state_path(args, source_board_id, clone_board_id)
        if not sync_state_path.exists():
            print(
                "Update mode needs a sync-state file. Run rebuild mode once, "
                "provide --sync-state-file, or rerun with --initialize-sync-state "
                "to create a best-effort mapping for an existing translated clone.\n"
                f"Expected sync-state file: {sync_state_path}",
                file=sys.stderr,
            )
            return 2

    if not args.sync_supported_items_only:
        print(
            "WARNING: Unsupported Miro item types are still skipped for safety; "
            "--no-sync-supported-items-only is accepted for compatibility only.",
            file=sys.stderr,
        )

    translation_backend: Optional[Ct2TranslatorBackend] = None
    if args.translator == "ct2":
        translation_backend = verify_ct2_backend_available(args)
    else:
        raise ValueError(f"Unsupported translator backend: {args.translator}")

    if args.update_existing_clone:
        return run_update_existing_clone_mode(args, miro_token, translation_backend)

    return run_rebuild_mode(args, miro_token, translation_backend)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
