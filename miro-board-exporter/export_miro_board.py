from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import img2pdf
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat
from playwright.sync_api import Browser, Error as PlaywrightError, Page, Playwright, TimeoutError as PlaywrightTimeoutError, sync_playwright


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
ZOOM_RE = re.compile(r"^\s*(<?)\s*(\d+(?:[.,]\d+)?)\s*%\s*$")


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class CaptureConfig:
    stability_interval_ms: int
    stability_max_wait_ms: int
    stability_required_matches: int
    stability_difference_threshold: float
    blank_retry_count: int
    blank_retry_wait_ms: int
    fast_mode: bool
    post_move_wait_ms: int
    content_capture_wait_ms: int
    probe_jpeg_quality: int
    save_blank_tiles: bool


@dataclass(frozen=True)
class BlankFilterConfig:
    enabled: bool
    move_blank_tiles_to_subfolder: bool
    blank_subfolder: str
    blur_radius: float
    difference_threshold: int
    minimum_content_fraction: float
    minimum_contrast_stddev: float


@dataclass(frozen=True)
class BoundsConfig:
    detect_from_fit_overview: bool
    background_difference_threshold: int
    fit_padding_px: int
    first_tile_margin_px: int
    right_bottom_margin_px: int
    occupancy_skip_enabled: bool
    occupancy_padding_fit_px: int
    save_fit_overview: bool
    alignment_correction_attempts: int
    alignment_tolerance_px: int
    anchor_search_enabled: bool
    anchor_search_step_fraction_x: float
    anchor_search_step_fraction_y: float
    anchor_search_max_steps_x: int
    anchor_search_max_steps_y: int
    save_anchor_debug: bool


@dataclass(frozen=True)
class NavigationConfig:
    auto_calibrate_pan: bool
    calibration_fraction: float
    pan_scale_fallback_x: float
    pan_scale_fallback_y: float
    pan_scale_min: float
    pan_scale_max: float
    step_fraction_x: float
    step_fraction_y: float
    precise_navigation: bool
    movement_tolerance_px: float
    max_correction_attempts: int
    max_measured_step_fraction: float
    command_safety_factor: float
    measurement_wait_ms: int
    adaptive_scale: bool
    reanchor_each_occupied_row: bool
    open_loop_for_grid_moves: bool


@dataclass(frozen=True)
class AbsoluteNavigationConfig:
    enabled: bool
    calibrate_scale: bool
    calibration_drag_fraction: float
    calibration_probe_width: int
    center_content_before_scale_calibration: bool
    prefer_horizontal_scale: bool
    minimum_overlap_fraction_x: float
    minimum_overlap_fraction_y: float
    fit_reset_wait_ms: int
    zoom_step_wait_ms: int
    post_zoom_wait_ms: int
    save_tile_plan: bool


@dataclass
class NavigationState:
    scale_x: float
    scale_y: float


@dataclass(frozen=True)
class Config:
    cdp_url: str
    output_root: Path
    target_zoom_percent: float
    default_max_tiles: int
    ask_resolution_each_run: bool
    overlap_px: int
    render_wait_ms: int
    capture: CaptureConfig
    blank_filter: BlankFilterConfig
    bounds: BoundsConfig
    navigation: NavigationConfig
    absolute_navigation: AbsoluteNavigationConfig
    coverage_safety_factor: float
    crop: Crop
    create_preview: bool
    preview_scale: float
    create_pdf: bool
    pdf_filename: str
    pdf_include_blank_tiles: bool
    pdf_image_format: str
    pdf_jpeg_quality: int
    pdf_image_scale: float
    max_tiles_without_extra_confirmation: int


@dataclass(frozen=True)
class Grid:
    estimated_width: float
    estimated_height: float
    step_x: int
    step_y: int
    cols: int
    rows: int

    @property
    def tile_count(self) -> int:
        return self.cols * self.rows


@dataclass(frozen=True)
class ContentBounds:
    left_css: float
    top_css: float
    right_css: float
    bottom_css: float

    @property
    def width_css(self) -> float:
        return max(1.0, self.right_css - self.left_css)

    @property
    def height_css(self) -> float:
        return max(1.0, self.bottom_css - self.top_css)


@dataclass
class FitOverview:
    bounds: ContentBounds
    mask: Image.Image
    mask_scale_x: float
    mask_scale_y: float
    background_rgb: tuple[int, int, int]
    screenshot_bytes: bytes


def load_config() -> Config:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Konfigurationsdatei fehlt: {CONFIG_PATH}")

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    crop_raw = raw.get("crop", {})
    output_root = Path(raw.get("output_root", "miro-export"))
    if not output_root.is_absolute():
        output_root = BASE_DIR / output_root

    capture_raw = raw.get("capture", {})
    blank_raw = raw.get("blank_filter", {})
    bounds_raw = raw.get("bounds", {})
    navigation_raw = raw.get("navigation", {})
    absolute_raw = raw.get("absolute_navigation", {})

    config = Config(
        cdp_url=str(raw.get("cdp_url", "http://127.0.0.1:9222")),
        output_root=output_root,
        target_zoom_percent=float(raw.get("target_zoom_percent", 100)),
        default_max_tiles=int(raw.get("default_max_tiles", 120)),
        ask_resolution_each_run=bool(raw.get("ask_resolution_each_run", True)),
        overlap_px=int(raw.get("overlap_px", 160)),
        render_wait_ms=int(raw.get("render_wait_ms", 1800)),
        capture=CaptureConfig(
            stability_interval_ms=int(capture_raw.get("stability_interval_ms", 700)),
            stability_max_wait_ms=int(capture_raw.get("stability_max_wait_ms", 8000)),
            stability_required_matches=int(capture_raw.get("stability_required_matches", 2)),
            stability_difference_threshold=float(capture_raw.get("stability_difference_threshold", 1.8)),
            blank_retry_count=int(capture_raw.get("blank_retry_count", 2)),
            blank_retry_wait_ms=int(capture_raw.get("blank_retry_wait_ms", 2500)),
            fast_mode=bool(capture_raw.get("fast_mode", True)),
            post_move_wait_ms=int(capture_raw.get("post_move_wait_ms", 120)),
            content_capture_wait_ms=int(capture_raw.get("content_capture_wait_ms", 80)),
            probe_jpeg_quality=int(capture_raw.get("probe_jpeg_quality", 35)),
            save_blank_tiles=bool(capture_raw.get("save_blank_tiles", False)),
        ),
        blank_filter=BlankFilterConfig(
            enabled=bool(blank_raw.get("enabled", True)),
            move_blank_tiles_to_subfolder=bool(blank_raw.get("move_blank_tiles_to_subfolder", True)),
            blank_subfolder=str(blank_raw.get("blank_subfolder", "_blank_tiles")).strip() or "_blank_tiles",
            blur_radius=float(blank_raw.get("blur_radius", 2.0)),
            difference_threshold=int(blank_raw.get("difference_threshold", 18)),
            minimum_content_fraction=float(blank_raw.get("minimum_content_fraction", 0.0015)),
            minimum_contrast_stddev=float(blank_raw.get("minimum_contrast_stddev", 3.0)),
        ),
        bounds=BoundsConfig(
            detect_from_fit_overview=bool(bounds_raw.get("detect_from_fit_overview", True)),
            background_difference_threshold=int(bounds_raw.get("background_difference_threshold", 20)),
            fit_padding_px=int(bounds_raw.get("fit_padding_px", 3)),
            first_tile_margin_px=int(bounds_raw.get("first_tile_margin_px", 48)),
            right_bottom_margin_px=int(bounds_raw.get("right_bottom_margin_px", 48)),
            occupancy_skip_enabled=bool(bounds_raw.get("occupancy_skip_enabled", True)),
            occupancy_padding_fit_px=int(bounds_raw.get("occupancy_padding_fit_px", 4)),
            save_fit_overview=bool(bounds_raw.get("save_fit_overview", True)),
            alignment_correction_attempts=int(bounds_raw.get("alignment_correction_attempts", 3)),
            alignment_tolerance_px=int(bounds_raw.get("alignment_tolerance_px", 10)),
            anchor_search_enabled=bool(bounds_raw.get("anchor_search_enabled", True)),
            anchor_search_step_fraction_x=float(bounds_raw.get("anchor_search_step_fraction_x", 0.30)),
            anchor_search_step_fraction_y=float(bounds_raw.get("anchor_search_step_fraction_y", 0.30)),
            anchor_search_max_steps_x=int(bounds_raw.get("anchor_search_max_steps_x", 3)),
            anchor_search_max_steps_y=int(bounds_raw.get("anchor_search_max_steps_y", 10)),
            save_anchor_debug=bool(bounds_raw.get("save_anchor_debug", True)),
        ),
        navigation=NavigationConfig(
            auto_calibrate_pan=bool(navigation_raw.get("auto_calibrate_pan", True)),
            calibration_fraction=float(navigation_raw.get("calibration_fraction", 0.22)),
            pan_scale_fallback_x=float(
                navigation_raw.get("pan_scale_fallback_x", navigation_raw.get("pan_scale_fallback", 0.45))
            ),
            pan_scale_fallback_y=float(
                navigation_raw.get("pan_scale_fallback_y", navigation_raw.get("pan_scale_fallback", 0.45))
            ),
            pan_scale_min=float(navigation_raw.get("pan_scale_min", 0.10)),
            pan_scale_max=float(navigation_raw.get("pan_scale_max", 1.50)),
            step_fraction_x=float(navigation_raw.get("step_fraction_x", 0.48)),
            step_fraction_y=float(navigation_raw.get("step_fraction_y", 0.48)),
            precise_navigation=bool(navigation_raw.get("precise_navigation", True)),
            movement_tolerance_px=float(navigation_raw.get("movement_tolerance_px", 18.0)),
            max_correction_attempts=int(navigation_raw.get("max_correction_attempts", 3)),
            max_measured_step_fraction=float(navigation_raw.get("max_measured_step_fraction", 0.38)),
            command_safety_factor=float(navigation_raw.get("command_safety_factor", 0.92)),
            measurement_wait_ms=int(navigation_raw.get("measurement_wait_ms", 450)),
            adaptive_scale=bool(navigation_raw.get("adaptive_scale", True)),
            reanchor_each_occupied_row=bool(navigation_raw.get("reanchor_each_occupied_row", True)),
            open_loop_for_grid_moves=bool(navigation_raw.get("open_loop_for_grid_moves", True)),
        ),
        absolute_navigation=AbsoluteNavigationConfig(
            enabled=bool(absolute_raw.get("enabled", True)),
            calibrate_scale=bool(absolute_raw.get("calibrate_scale", True)),
            calibration_drag_fraction=float(absolute_raw.get("calibration_drag_fraction", 0.30)),
            calibration_probe_width=int(absolute_raw.get("calibration_probe_width", 1800)),
            center_content_before_scale_calibration=bool(
                absolute_raw.get("center_content_before_scale_calibration", True)
            ),
            prefer_horizontal_scale=bool(absolute_raw.get("prefer_horizontal_scale", True)),
            minimum_overlap_fraction_x=float(
                absolute_raw.get("minimum_overlap_fraction_x", 0.65)
            ),
            minimum_overlap_fraction_y=float(
                absolute_raw.get("minimum_overlap_fraction_y", 0.65)
            ),
            fit_reset_wait_ms=int(absolute_raw.get("fit_reset_wait_ms", 260)),
            zoom_step_wait_ms=int(absolute_raw.get("zoom_step_wait_ms", 65)),
            post_zoom_wait_ms=int(absolute_raw.get("post_zoom_wait_ms", 90)),
            save_tile_plan=bool(absolute_raw.get("save_tile_plan", True)),
        ),
        coverage_safety_factor=float(raw.get("coverage_safety_factor", 1.15)),
        crop=Crop(
            left=int(crop_raw.get("left", 90)),
            top=int(crop_raw.get("top", 65)),
            right=int(crop_raw.get("right", 30)),
            bottom=int(crop_raw.get("bottom", 85)),
        ),
        create_preview=bool(raw.get("create_preview", True)),
        preview_scale=float(raw.get("preview_scale", 0.12)),
        create_pdf=bool(raw.get("create_pdf", True)),
        pdf_filename=str(raw.get("pdf_filename", "miro-board-export.pdf")).strip(),
        pdf_include_blank_tiles=bool(raw.get("pdf_include_blank_tiles", False)),
        pdf_image_format=str(raw.get("pdf_image_format", "png")).strip().lower(),
        pdf_jpeg_quality=int(raw.get("pdf_jpeg_quality", 85)),
        pdf_image_scale=float(raw.get("pdf_image_scale", 1.0)),
        max_tiles_without_extra_confirmation=int(raw.get("max_tiles_without_extra_confirmation", 300)),
    )

    if not 1 <= config.target_zoom_percent <= 400:
        raise ValueError("target_zoom_percent muss zwischen 1 und 400 liegen.")
    if config.default_max_tiles < 1:
        raise ValueError("default_max_tiles muss mindestens 1 sein.")
    if config.overlap_px < 0:
        raise ValueError("overlap_px darf nicht negativ sein.")
    if not 1.0 <= config.coverage_safety_factor <= 2.0:
        raise ValueError("coverage_safety_factor muss zwischen 1.0 und 2.0 liegen.")
    if config.capture.stability_interval_ms < 100:
        raise ValueError("capture.stability_interval_ms muss mindestens 100 sein.")
    if config.capture.stability_max_wait_ms < config.capture.stability_interval_ms:
        raise ValueError("capture.stability_max_wait_ms muss mindestens so groß wie stability_interval_ms sein.")
    if config.capture.stability_required_matches < 1:
        raise ValueError("capture.stability_required_matches muss mindestens 1 sein.")
    if config.capture.blank_retry_count < 0:
        raise ValueError("capture.blank_retry_count darf nicht negativ sein.")
    if config.capture.post_move_wait_ms < 0 or config.capture.content_capture_wait_ms < 0:
        raise ValueError("capture.post_move_wait_ms/content_capture_wait_ms dürfen nicht negativ sein.")
    if not 1 <= config.capture.probe_jpeg_quality <= 100:
        raise ValueError("capture.probe_jpeg_quality muss zwischen 1 und 100 liegen.")
    if not 0 <= config.bounds.background_difference_threshold <= 255:
        raise ValueError("bounds.background_difference_threshold muss zwischen 0 und 255 liegen.")
    if config.bounds.fit_padding_px < 0 or config.bounds.occupancy_padding_fit_px < 0:
        raise ValueError("bounds padding values dürfen nicht negativ sein.")
    if config.bounds.first_tile_margin_px < 0 or config.bounds.right_bottom_margin_px < 0:
        raise ValueError("bounds margins dürfen nicht negativ sein.")
    if not 0 <= config.bounds.alignment_correction_attempts <= 5:
        raise ValueError("bounds.alignment_correction_attempts muss zwischen 0 und 5 liegen.")
    if not 0.0 <= config.blank_filter.minimum_content_fraction <= 1.0:
        raise ValueError("blank_filter.minimum_content_fraction muss zwischen 0 und 1 liegen.")
    if not 0 <= config.blank_filter.difference_threshold <= 255:
        raise ValueError("blank_filter.difference_threshold muss zwischen 0 und 255 liegen.")
    if Path(config.blank_filter.blank_subfolder).name != config.blank_filter.blank_subfolder:
        raise ValueError("blank_filter.blank_subfolder darf nur ein Ordnername sein.")
    if not 0.10 <= config.navigation.calibration_fraction <= 0.70:
        raise ValueError("navigation.calibration_fraction muss zwischen 0.10 und 0.70 liegen.")
    if not 0.05 <= config.navigation.pan_scale_min <= config.navigation.pan_scale_max <= 3.0:
        raise ValueError("navigation.pan_scale_min/max sind ungültig.")
    for label, fallback in (
        ("pan_scale_fallback_x", config.navigation.pan_scale_fallback_x),
        ("pan_scale_fallback_y", config.navigation.pan_scale_fallback_y),
    ):
        if not config.navigation.pan_scale_min <= fallback <= config.navigation.pan_scale_max:
            raise ValueError(f"navigation.{label} muss zwischen pan_scale_min und pan_scale_max liegen.")
    if not 0.20 <= config.navigation.step_fraction_x <= 0.95:
        raise ValueError("navigation.step_fraction_x muss zwischen 0.20 und 0.95 liegen.")
    if not 0.20 <= config.navigation.step_fraction_y <= 0.95:
        raise ValueError("navigation.step_fraction_y muss zwischen 0.20 und 0.95 liegen.")
    if config.navigation.movement_tolerance_px < 1:
        raise ValueError("navigation.movement_tolerance_px muss mindestens 1 sein.")
    if not 0 <= config.navigation.max_correction_attempts <= 8:
        raise ValueError("navigation.max_correction_attempts muss zwischen 0 und 8 liegen.")
    if not 0.15 <= config.navigation.max_measured_step_fraction <= 0.60:
        raise ValueError("navigation.max_measured_step_fraction muss zwischen 0.15 und 0.60 liegen.")
    if not 0.50 <= config.navigation.command_safety_factor <= 1.0:
        raise ValueError("navigation.command_safety_factor muss zwischen 0.50 und 1.0 liegen.")
    if config.navigation.measurement_wait_ms < 100:
        raise ValueError("navigation.measurement_wait_ms muss mindestens 100 sein.")
    if not 0.20 <= config.absolute_navigation.calibration_drag_fraction <= 0.80:
        raise ValueError("absolute_navigation.calibration_drag_fraction muss zwischen 0.20 und 0.80 liegen.")
    if not 500 <= config.absolute_navigation.calibration_probe_width <= 2400:
        raise ValueError("absolute_navigation.calibration_probe_width muss zwischen 500 und 2400 liegen.")
    if not 0.20 <= config.absolute_navigation.minimum_overlap_fraction_x <= 0.85:
        raise ValueError(
            "absolute_navigation.minimum_overlap_fraction_x muss zwischen 0.20 und 0.85 liegen."
        )
    if not 0.20 <= config.absolute_navigation.minimum_overlap_fraction_y <= 0.85:
        raise ValueError(
            "absolute_navigation.minimum_overlap_fraction_y muss zwischen 0.20 und 0.85 liegen."
        )
    if config.absolute_navigation.fit_reset_wait_ms < 80:
        raise ValueError("absolute_navigation.fit_reset_wait_ms muss mindestens 80 sein.")
    if config.absolute_navigation.zoom_step_wait_ms < 20:
        raise ValueError("absolute_navigation.zoom_step_wait_ms muss mindestens 20 sein.")
    if config.absolute_navigation.post_zoom_wait_ms < 0:
        raise ValueError("absolute_navigation.post_zoom_wait_ms darf nicht negativ sein.")
    if not 0.01 <= config.preview_scale <= 1.0:
        raise ValueError("preview_scale muss zwischen 0.01 und 1.0 liegen.")
    if config.create_pdf:
        if not config.pdf_filename:
            raise ValueError("pdf_filename darf nicht leer sein.")
        if Path(config.pdf_filename).name != config.pdf_filename:
            raise ValueError("pdf_filename darf nur ein Dateiname ohne Ordner sein.")
        if not config.pdf_filename.lower().endswith(".pdf"):
            raise ValueError("pdf_filename muss auf .pdf enden.")
        if config.pdf_image_format not in {"png", "jpeg", "jpg"}:
            raise ValueError("pdf_image_format muss png oder jpeg sein.")
        if not 1 <= config.pdf_jpeg_quality <= 100:
            raise ValueError("pdf_jpeg_quality muss zwischen 1 und 100 liegen.")
        if not 0.10 <= config.pdf_image_scale <= 1.0:
            raise ValueError("pdf_image_scale muss zwischen 0.10 und 1.0 liegen.")
    return config


def wait_ms(milliseconds: int) -> None:
    time.sleep(max(0, milliseconds) / 1000.0)


def connect_to_chrome(playwright: Playwright, cdp_url: str) -> Browser:
    try:
        return playwright.chromium.connect_over_cdp(cdp_url, timeout=10_000)
    except Exception as exc:
        raise RuntimeError(
            "Keine Verbindung zu Chrome möglich. Starte zuerst "
            "1_start_miro_chrome.bat und lasse das Fenster geöffnet."
        ) from exc


def choose_miro_page(browser: Browser) -> Page:
    pages: list[Page] = []
    for context in browser.contexts:
        pages.extend(context.pages)

    miro_pages = [page for page in pages if "miro.com" in page.url.lower()]
    board_pages = [page for page in miro_pages if "/app/board/" in page.url.lower()]

    candidates = board_pages or miro_pages
    if not candidates:
        raise RuntimeError(
            "Kein Miro-Tab gefunden. Öffne im separaten Chrome-Fenster dein Board und starte den Export erneut."
        )

    if len(candidates) == 1:
        return candidates[0]

    print("\nMehrere Miro-Tabs gefunden:")
    for index, page in enumerate(candidates, start=1):
        try:
            title = page.title()
        except Exception:
            title = "(Titel nicht lesbar)"
        print(f"  {index}: {title} — {page.url}")

    while True:
        answer = input(f"Tab auswählen [1-{len(candidates)}]: ").strip()
        try:
            selected = int(answer)
            if 1 <= selected <= len(candidates):
                return candidates[selected - 1]
        except ValueError:
            pass
        print("Ungültige Auswahl.")


def viewport_size(page: Page) -> tuple[int, int]:
    result = page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
    width = int(result["width"])
    height = int(result["height"])
    if width < 800 or height < 600:
        raise RuntimeError(
            f"Das Chrome-Fenster ist zu klein ({width}×{height}). Maximiere es und starte erneut."
        )
    return width, height


def focus_board(page: Page, width: int, height: int) -> None:
    page.bring_to_front()
    page.keyboard.press("Escape")
    page.mouse.click(int(width * 0.50), int(height * 0.50))
    wait_ms(250)


def read_visible_zoom(page: Page) -> tuple[float | None, str | None]:
    script = r"""
    () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const pattern = /^\s*<?\s*\d+(?:[.,]\d+)?\s*%\s*$/;
      const nodes = Array.from(document.querySelectorAll('button,[role="button"],span,div'));
      const candidates = [];
      for (const element of nodes) {
        const text = (element.innerText || element.textContent || '').trim();
        if (!pattern.test(text)) continue;
        if (element.children.length > 6) continue;
        const rect = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        if (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) continue;
        if (rect.width <= 0 || rect.height <= 0 || rect.width > 220 || rect.height > 110) continue;
        if (rect.right < width * 0.45 || rect.bottom < height * 0.50) continue;
        candidates.push({ text, x: rect.x, y: rect.y, right: rect.right, bottom: rect.bottom });
      }
      candidates.sort((a, b) => (b.bottom + b.right) - (a.bottom + a.right));
      return candidates.length ? candidates[0].text : null;
    }
    """
    try:
        text = page.evaluate(script)
    except Exception:
        return None, None
    if not text:
        return None, None

    match = ZOOM_RE.match(str(text))
    if not match:
        return None, str(text)
    less_than, number = match.groups()
    value = float(number.replace(",", "."))
    if less_than:
        return None, str(text)
    return value, str(text)


def ask_zoom_fallback(label: str, detected_text: str | None = None) -> float:
    hint = f" (Anzeige erkannt: {detected_text})" if detected_text else ""
    while True:
        answer = input(
            f"{label} konnte nicht automatisch gelesen werden{hint}. "
            "Bitte den Prozentwert unten rechts eingeben, z. B. 4 oder 4,5: "
        ).strip()
        try:
            value = float(answer.replace(",", "."))
            if 0 < value <= 400:
                return value
        except ValueError:
            pass
        print("Bitte eine gültige positive Zahl eingeben.")


def zoom_to_fit(page: Page, width: int, height: int, render_wait_ms: int) -> float:
    focus_board(page, width, height)
    page.keyboard.press("Alt+Digit1")
    wait_ms(max(render_wait_ms, 2200))
    zoom, text = read_visible_zoom(page)
    return zoom if zoom is not None else ask_zoom_fallback("Der Zoom-to-fit-Wert", text)


def zoom_to_100(page: Page, width: int, height: int, render_wait_ms: int) -> float:
    focus_board(page, width, height)
    page.keyboard.press("Control+Digit0")
    wait_ms(max(render_wait_ms, 1400))
    zoom, _ = read_visible_zoom(page)
    return zoom or 100.0


def set_target_zoom(page: Page, width: int, height: int, target: float, render_wait_ms: int) -> float:
    current = zoom_to_100(page, width, height, render_wait_ms)
    if abs(target - current) < 0.5:
        return current

    best_value = current
    best_distance = abs(current - target)
    previous = current

    for _ in range(40):
        if current > target:
            page.keyboard.press("Control+Minus")
        else:
            page.keyboard.press("Control+Shift+Equal")
        wait_ms(350)

        detected, _ = read_visible_zoom(page)
        if detected is None or abs(detected - previous) < 0.01:
            break
        current = detected
        distance = abs(current - target)
        if distance < best_distance:
            best_value = current
            best_distance = distance
        if (previous - target) * (current - target) <= 0:
            break
        previous = current

    if abs(current - best_value) > 0.01:
        zoom_to_100(page, width, height, render_wait_ms)
        direction_key = "Control+Minus" if best_value < 100 else "Control+Shift+Equal"
        for _ in range(40):
            detected, _ = read_visible_zoom(page)
            detected = detected or 100.0
            if abs(detected - best_value) < 0.01:
                current = detected
                break
            page.keyboard.press(direction_key)
            wait_ms(250)

    detected, text = read_visible_zoom(page)
    if detected is not None:
        return detected
    return ask_zoom_fallback("Der tatsächliche Export-Zoom", text)



def _dominant_background_rgb(image: Image.Image) -> tuple[int, int, int]:
    sample = image.convert("RGB")
    if sample.width > 900:
        new_height = max(1, round(sample.height * 900 / sample.width))
        sample = sample.resize((900, new_height), Image.Resampling.BILINEAR)
    quantized = sample.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
    histogram = quantized.histogram()
    dominant_index = max(range(len(histogram)), key=histogram.__getitem__)
    palette = quantized.getpalette() or []
    offset = dominant_index * 3
    background = tuple(palette[offset:offset + 3])
    return background if len(background) == 3 else (0, 0, 0)


def create_fit_overview(
    page: Page,
    clip: dict[str, float],
    clip_width_css: int,
    clip_height_css: int,
    bounds_config: BoundsConfig,
) -> FitOverview:
    """Capture the complete board at zoom-to-fit and detect its real content bounds."""
    data = page.screenshot(
        clip=clip,
        type="png",
        animations="disabled",
        caret="hide",
        timeout=60_000,
    )
    from io import BytesIO

    with Image.open(BytesIO(data)) as source:
        original = source.convert("RGB")
        background = _dominant_background_rgb(original)
        max_width = 1400
        if original.width > max_width:
            mask_height = max(1, round(original.height * max_width / original.width))
            working = original.resize((max_width, mask_height), Image.Resampling.BILINEAR)
        else:
            working = original.copy()

    background_image = Image.new("RGB", working.size, background)
    difference = ImageChops.difference(working, background_image)
    # Max-channel difference preserves thin white table lines and coloured text.
    channels = difference.split()
    max_channel = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    mask = max_channel.point(
        lambda value: 255 if value > bounds_config.background_difference_threshold else 0,
        mode="1",
    ).convert("L")
    # Slight dilation protects thin elements and makes occupancy planning conservative.
    mask = mask.filter(ImageFilter.MaxFilter(3))
    bbox = mask.getbbox()
    if bbox is None:
        raise RuntimeError(
            "Auf der Zoom-to-fit-Aufnahme konnte kein Board-Inhalt erkannt werden. "
            "Prüfe, ob das Board geladen ist, oder reduziere "
            "bounds.background_difference_threshold in config.json."
        )

    scale_x = mask.width / max(1, clip_width_css)
    scale_y = mask.height / max(1, clip_height_css)
    padding_x = round(bounds_config.fit_padding_px * scale_x)
    padding_y = round(bounds_config.fit_padding_px * scale_y)
    left = max(0, bbox[0] - padding_x)
    top = max(0, bbox[1] - padding_y)
    right = min(mask.width, bbox[2] + padding_x)
    bottom = min(mask.height, bbox[3] + padding_y)

    bounds = ContentBounds(
        left_css=left / scale_x,
        top_css=top / scale_y,
        right_css=right / scale_x,
        bottom_css=bottom / scale_y,
    )
    return FitOverview(
        bounds=bounds,
        mask=mask,
        mask_scale_x=scale_x,
        mask_scale_y=scale_y,
        background_rgb=background,
        screenshot_bytes=data,
    )


def tile_intersects_overview_content(
    overview: FitOverview,
    row: int,
    col: int,
    grid: Grid,
    fit_zoom: float,
    export_zoom: float,
    clip_width_css: int,
    clip_height_css: int,
    bounds_config: BoundsConfig,
) -> bool:
    """Use the zoom-to-fit mask to avoid visiting obviously empty grid cells."""
    if not bounds_config.occupancy_skip_enabled:
        return True
    scale = export_zoom / fit_zoom
    if scale <= 0:
        return True

    # Board coordinates are relative to the detected top-left content edge.
    board_left_export = col * grid.step_x - bounds_config.first_tile_margin_px
    board_top_export = row * grid.step_y - bounds_config.first_tile_margin_px
    left_fit_css = overview.bounds.left_css + board_left_export / scale
    top_fit_css = overview.bounds.top_css + board_top_export / scale
    right_fit_css = left_fit_css + clip_width_css / scale
    bottom_fit_css = top_fit_css + clip_height_css / scale

    padding = bounds_config.occupancy_padding_fit_px
    left_fit_css -= padding
    top_fit_css -= padding
    right_fit_css += padding
    bottom_fit_css += padding

    x0 = max(0, math.floor(left_fit_css * overview.mask_scale_x))
    y0 = max(0, math.floor(top_fit_css * overview.mask_scale_y))
    x1 = min(overview.mask.width, math.ceil(right_fit_css * overview.mask_scale_x))
    y1 = min(overview.mask.height, math.ceil(bottom_fit_css * overview.mask_scale_y))
    if x1 <= x0 or y1 <= y0:
        return False
    return overview.mask.crop((x0, y0, x1, y1)).getbbox() is not None


def plan_candidate_cells(
    overview: FitOverview,
    grid: Grid,
    fit_zoom: float,
    export_zoom: float,
    clip_width_css: int,
    clip_height_css: int,
    bounds_config: BoundsConfig,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for row in range(grid.rows):
        for col in range(grid.cols):
            if tile_intersects_overview_content(
                overview,
                row,
                col,
                grid,
                fit_zoom,
                export_zoom,
                clip_width_css,
                clip_height_css,
                bounds_config,
            ):
                cells.append((row, col))
    return cells


def predicted_top_left_camera_move(
    overview: FitOverview,
    fit_zoom: float,
    export_zoom: float,
    zoom_anchor_x_in_clip_css: float,
    zoom_anchor_y_in_clip_css: float,
    margin_px: int,
) -> tuple[float, float]:
    """Calculate the camera movement needed to put content near the first tile's top-left."""
    scale = export_zoom / fit_zoom
    current_left = (
        zoom_anchor_x_in_clip_css
        + (overview.bounds.left_css - zoom_anchor_x_in_clip_css) * scale
    )
    current_top = (
        zoom_anchor_y_in_clip_css
        + (overview.bounds.top_css - zoom_anchor_y_in_clip_css) * scale
    )
    # Positive camera movement shifts visible board content left/up.
    return current_left - margin_px, current_top - margin_px


def visible_content_bounds(
    data: bytes,
    clip_width_css: int,
    clip_height_css: int,
    threshold: int,
) -> ContentBounds | None:
    from io import BytesIO

    with Image.open(BytesIO(data)) as source:
        image = source.convert("RGB")
        background = _dominant_background_rgb(image)
        bg = Image.new("RGB", image.size, background)
        difference = ImageChops.difference(image, bg)
        channels = difference.split()
        maximum = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
        mask = maximum.point(lambda value: 255 if value > threshold else 0, mode="1")
        bbox = mask.getbbox()
        if bbox is None:
            return None
        sx = image.width / max(1, clip_width_css)
        sy = image.height / max(1, clip_height_css)
        return ContentBounds(
            left_css=bbox[0] / sx,
            top_css=bbox[1] / sy,
            right_css=bbox[2] / sx,
            bottom_css=bbox[3] / sy,
        )


def correct_first_tile_alignment(
    page: Page,
    clip: dict[str, float],
    width: int,
    height: int,
    clip_width_css: int,
    clip_height_css: int,
    config: Config,
    navigation_state: NavigationState,
) -> None:
    margin = config.bounds.first_tile_margin_px
    for attempt in range(config.bounds.alignment_correction_attempts):
        wait_ms(config.capture.post_move_wait_ms)
        probe = page.screenshot(
            clip=clip,
            type="jpeg",
            quality=config.capture.probe_jpeg_quality,
            animations="disabled",
            caret="hide",
            timeout=60_000,
        )
        detected = visible_content_bounds(
            probe,
            clip_width_css,
            clip_height_css,
            config.bounds.background_difference_threshold,
        )
        if detected is None:
            return
        error_x = detected.left_css - margin
        error_y = detected.top_css - margin
        if (
            abs(error_x) <= config.bounds.alignment_tolerance_px
            and abs(error_y) <= config.bounds.alignment_tolerance_px
        ):
            return
        print(
            f"  Startausrichtung korrigieren: X {error_x:+.0f}px, Y {error_y:+.0f}px "
            f"({attempt + 1}/{config.bounds.alignment_correction_attempts})"
        )
        move_camera_precise(
            page,
            error_x,
            error_y,
            clip,
            width,
            height,
            config.navigation,
            navigation_state,
        )



def move_camera_open_loop(
    page: Page,
    camera_dx: float,
    camera_dy: float,
    width: int,
    height: int,
    state: NavigationState,
) -> None:
    """Move using the calibrated drag scale without screenshot-based correction.

    Visual registration is unreliable on Miro's completely black empty canvas. For
    long jumps through empty areas, open-loop movement is more stable than trying to
    infer displacement from two identical screenshots.
    """
    if abs(camera_dx) >= 0.5:
        move_camera(
            page,
            camera_dx,
            0,
            width,
            height,
            pan_scale=state.scale_x,
        )
    if abs(camera_dy) >= 0.5:
        move_camera(
            page,
            0,
            camera_dy,
            width,
            height,
            pan_scale=state.scale_y,
        )


def _capture_visible_bounds(
    page: Page,
    clip: dict[str, float],
    clip_width_css: int,
    clip_height_css: int,
    config: Config,
) -> tuple[bytes, ContentBounds | None]:
    wait_ms(config.capture.post_move_wait_ms)
    data = page.screenshot(
        clip=clip,
        type="jpeg",
        quality=max(35, config.capture.probe_jpeg_quality),
        animations="disabled",
        caret="hide",
        timeout=60_000,
    )
    return data, visible_content_bounds(
        data,
        clip_width_css,
        clip_height_css,
        config.bounds.background_difference_threshold,
    )


def search_and_align_top_left_anchor(
    page: Page,
    clip: dict[str, float],
    width: int,
    height: int,
    clip_width_css: int,
    clip_height_css: int,
    config: Config,
    navigation_state: NavigationState,
    output_dir: Path | None = None,
    debug_label: str = "anchor",
) -> bool:
    """Find visible content around the predicted origin and align it to the margin.

    The old implementation stopped immediately when the predicted viewport was
    completely black. On sparse boards that can put the real first row several
    logical rows later. This routine actively searches around the prediction before
    alignment. Search moves deliberately use calibrated open-loop panning because
    blank screenshots cannot be registered reliably.
    """
    margin = config.bounds.first_tile_margin_px
    probe, detected = _capture_visible_bounds(
        page, clip, clip_width_css, clip_height_css, config
    )

    if detected is None and config.bounds.anchor_search_enabled:
        step_x = max(120.0, clip_width_css * config.bounds.anchor_search_step_fraction_x)
        step_y = max(100.0, clip_height_css * config.bounds.anchor_search_step_fraction_y)

        # The most common failure is that the prediction is above the board, so
        # search downward first. Then search the opposite directions from the
        # original predicted position.
        searches = (
            ("unten", 0.0, step_y, config.bounds.anchor_search_max_steps_y),
            ("oben", 0.0, -step_y, config.bounds.anchor_search_max_steps_y),
            ("rechts", step_x, 0.0, config.bounds.anchor_search_max_steps_x),
            ("links", -step_x, 0.0, config.bounds.anchor_search_max_steps_x),
        )

        current_offset_x = 0.0
        current_offset_y = 0.0
        for label, delta_x, delta_y, maximum_steps in searches:
            # Return to the original prediction before trying another direction.
            if abs(current_offset_x) >= 0.5 or abs(current_offset_y) >= 0.5:
                move_camera_open_loop(
                    page,
                    -current_offset_x,
                    -current_offset_y,
                    width,
                    height,
                    navigation_state,
                )
                current_offset_x = 0.0
                current_offset_y = 0.0

            for step_index in range(1, maximum_steps + 1):
                move_camera_open_loop(
                    page,
                    delta_x,
                    delta_y,
                    width,
                    height,
                    navigation_state,
                )
                current_offset_x += delta_x
                current_offset_y += delta_y
                probe, detected = _capture_visible_bounds(
                    page, clip, clip_width_css, clip_height_css, config
                )
                if detected is not None:
                    print(
                        f"  Oberer linker Anker nach Suche {label} gefunden "
                        f"(Schritt {step_index})."
                    )
                    break
            if detected is not None:
                break

    if detected is None:
        print(
            "  WARNUNG: In der Umgebung der berechneten Startposition wurde kein "
            "Inhalt erkannt. Verwende die berechnete Position unverändert."
        )
        if output_dir is not None and config.bounds.save_anchor_debug:
            (output_dir / f"_{debug_label}_not_found.jpg").write_bytes(probe)
        return False

    # Align repeatedly. These corrections happen while content is visible, so the
    # closed-loop movement estimator has usable visual detail.
    for attempt in range(max(1, config.bounds.alignment_correction_attempts)):
        error_x = detected.left_css - margin
        error_y = detected.top_css - margin
        if (
            abs(error_x) <= config.bounds.alignment_tolerance_px
            and abs(error_y) <= config.bounds.alignment_tolerance_px
        ):
            break
        print(
            f"  Anker ausrichten: X {error_x:+.0f}px, Y {error_y:+.0f}px "
            f"({attempt + 1}/{max(1, config.bounds.alignment_correction_attempts)})"
        )
        move_camera_precise(
            page,
            error_x,
            error_y,
            clip,
            width,
            height,
            config.navigation,
            navigation_state,
        )
        probe, detected = _capture_visible_bounds(
            page, clip, clip_width_css, clip_height_css, config
        )
        if detected is None:
            break

    if output_dir is not None and config.bounds.save_anchor_debug:
        (output_dir / f"_{debug_label}.jpg").write_bytes(probe)
    return True


def restore_top_left_anchor(
    page: Page,
    fit_overview: FitOverview,
    fit_zoom: float,
    export_zoom: float,
    clip: dict[str, float],
    width: int,
    height: int,
    clip_width_css: int,
    clip_height_css: int,
    config: Config,
    navigation_state: NavigationState,
    output_dir: Path | None = None,
    debug_label: str = "anchor",
) -> None:
    """Reset to a reproducible global top-left origin at the export zoom."""
    current_fit_zoom = zoom_to_fit(page, width, height, config.render_wait_ms)
    # Miro may round the displayed fit zoom, but the original overview geometry is
    # still the best stable reference for the same board and viewport.
    actual_zoom = set_target_zoom(
        page, width, height, export_zoom, config.render_wait_ms
    )
    initial_camera_x, initial_camera_y = predicted_top_left_camera_move(
        fit_overview,
        current_fit_zoom if current_fit_zoom > 0 else fit_zoom,
        actual_zoom,
        width / 2.0 - config.crop.left,
        height / 2.0 - config.crop.top,
        config.bounds.first_tile_margin_px,
    )
    move_camera_open_loop(
        page,
        initial_camera_x,
        initial_camera_y,
        width,
        height,
        navigation_state,
    )
    search_and_align_top_left_anchor(
        page,
        clip,
        width,
        height,
        clip_width_css,
        clip_height_css,
        config,
        navigation_state,
        output_dir=output_dir,
        debug_label=debug_label,
    )


def calculate_grid(
    fit_zoom: float,
    export_zoom: float,
    content_width_fit_css: float,
    content_height_fit_css: float,
    clip_width: int,
    clip_height: int,
    step_fraction_x: float,
    step_fraction_y: float,
    first_tile_margin_px: int,
    right_bottom_margin_px: int,
    minimum_overlap_fraction_x: float = 0.0,
    minimum_overlap_fraction_y: float = 0.0,
) -> Grid:
    if fit_zoom <= 0 or export_zoom <= 0:
        raise ValueError("Zoomwerte müssen positiv sein.")

    scale = export_zoom / fit_zoom
    estimated_width = (
        content_width_fit_css * scale
        + first_tile_margin_px
        + right_bottom_margin_px
    )
    estimated_height = (
        content_height_fit_css * scale
        + first_tile_margin_px
        + right_bottom_margin_px
    )
    requested_step_x = max(160, min(clip_width - 80, round(clip_width * step_fraction_x)))
    requested_step_y = max(140, min(clip_height - 80, round(clip_height * step_fraction_y)))
    overlap_limited_step_x = max(160, round(clip_width * (1.0 - minimum_overlap_fraction_x)))
    overlap_limited_step_y = max(140, round(clip_height * (1.0 - minimum_overlap_fraction_y)))
    step_x = min(requested_step_x, overlap_limited_step_x)
    step_y = min(requested_step_y, overlap_limited_step_y)
    cols = max(1, math.ceil(max(0.0, estimated_width - clip_width) / step_x) + 1)
    rows = max(1, math.ceil(max(0.0, estimated_height - clip_height) / step_y) + 1)
    return Grid(
        estimated_width=estimated_width,
        estimated_height=estimated_height,
        step_x=step_x,
        step_y=step_y,
        cols=cols,
        rows=rows,
    )


def estimate_zoom_for_tile_limit(
    fit_zoom: float,
    max_tiles: int,
    content_width_fit_css: float,
    content_height_fit_css: float,
    clip_width: int,
    clip_height: int,
    step_fraction_x: float,
    step_fraction_y: float,
    first_tile_margin_px: int,
    right_bottom_margin_px: int,
) -> tuple[float, Grid]:
    """Findet den höchsten theoretischen Zoom, der das Bilderlimit einhält."""
    minimum_zoom = max(1.0, min(fit_zoom, 400.0))
    minimum_grid = calculate_grid(
        fit_zoom,
        minimum_zoom,
        content_width_fit_css,
        content_height_fit_css,
        clip_width,
        clip_height,
        step_fraction_x,
        step_fraction_y,
        first_tile_margin_px,
        right_bottom_margin_px,
    )
    if minimum_grid.tile_count > max_tiles:
        return minimum_zoom, minimum_grid

    low = minimum_zoom
    high = 400.0
    best_grid = minimum_grid
    for _ in range(60):
        middle = (low + high) / 2.0
        grid = calculate_grid(
            fit_zoom,
            middle,
            content_width_fit_css,
            content_height_fit_css,
            clip_width,
            clip_height,
            step_fraction_x,
            step_fraction_y,
            first_tile_margin_px,
            right_bottom_margin_px,
        )
        if grid.tile_count <= max_tiles:
            low = middle
            best_grid = grid
        else:
            high = middle
    return low, best_grid


def ask_resolution(
    config: Config,
    fit_zoom: float,
    content_bounds: ContentBounds,
    clip_width: int,
    clip_height: int,
) -> tuple[float, int | None, str]:
    if not config.ask_resolution_each_run:
        return config.target_zoom_percent, None, "config_zoom"

    print("\nDetailstufe festlegen:")
    print("  1. Miro-Zoom in Prozent direkt vorgeben (Standard)")
    print("  2. Maximale Anzahl Bilder vorgeben")

    while True:
        mode = input("Auswahl [1]: ").strip() or "1"
        if mode == "1":
            while True:
                raw = input(
                    f"Gewünschter Miro-Zoom in Prozent [{config.target_zoom_percent:g}]: "
                ).strip()
                try:
                    target_zoom = (
                        float(raw.replace(",", "."))
                        if raw
                        else config.target_zoom_percent
                    )
                    if 1 <= target_zoom <= 400:
                        print(
                            "Hinweis: Miro verwendet feste Zoomstufen. "
                            "Das Programm wählt die erreichbare Stufe, die dem Wert am nächsten liegt."
                        )
                        return target_zoom, None, "manual_zoom"
                except ValueError:
                    pass
                print("Bitte einen Wert zwischen 1 und 400 eingeben.")

        if mode == "2":
            while True:
                raw = input(f"Maximal gewünschte Bilder [{config.default_max_tiles}]: ").strip()
                try:
                    max_tiles = int(raw) if raw else config.default_max_tiles
                    if max_tiles >= 1:
                        break
                except ValueError:
                    pass
                print("Bitte eine ganze Zahl ab 1 eingeben.")

            target_zoom, predicted_grid = estimate_zoom_for_tile_limit(
                fit_zoom,
                max_tiles,
                content_bounds.width_css,
                content_bounds.height_css,
                clip_width,
                clip_height,
                config.navigation.step_fraction_x,
                config.navigation.step_fraction_y,
                config.bounds.first_tile_margin_px,
                config.bounds.right_bottom_margin_px,
            )
            print(
                f"Automatisch berechneter Ziel-Zoom: ungefähr {target_zoom:.1f}% "
                f"({predicted_grid.cols} × {predicted_grid.rows} = "
                f"{predicted_grid.tile_count} Bilder theoretisch)"
            )
            if predicted_grid.tile_count > max_tiles:
                print(
                    "Hinweis: Selbst bei der kleinsten vorgesehenen Zoomstufe kann das "
                    f"Limit nicht ganz eingehalten werden ({predicted_grid.tile_count} Bilder)."
                )
            return target_zoom, max_tiles, "max_tiles"

        print("Bitte 1 oder 2 eingeben.")

def reduce_zoom_to_tile_limit(
    page: Page,
    current_zoom: float,
    max_tiles: int,
    fit_zoom: float,
    content_width_fit_css: float,
    content_height_fit_css: float,
    clip_width: int,
    clip_height: int,
    step_fraction_x: float,
    step_fraction_y: float,
    first_tile_margin_px: int,
    right_bottom_margin_px: int,
    render_wait_ms: int,
) -> tuple[float, Grid]:
    grid = calculate_grid(
        fit_zoom,
        current_zoom,
        content_width_fit_css,
        content_height_fit_css,
        clip_width,
        clip_height,
        step_fraction_x,
        step_fraction_y,
        first_tile_margin_px,
        right_bottom_margin_px,
    )

    attempts = 0
    while grid.tile_count > max_tiles and current_zoom > 1.0 and attempts < 40:
        print(
            f"Die erreichbare Miro-Zoomstufe ergäbe {grid.tile_count} Bilder. "
            "Reduziere automatisch um eine Zoomstufe ..."
        )
        previous_zoom = current_zoom
        page.keyboard.press("Control+Minus")
        wait_ms(max(500, min(render_wait_ms, 1200)))
        detected, text = read_visible_zoom(page)
        if detected is None:
            detected = ask_zoom_fallback("Der nach dem Verkleinern erreichte Zoom", text)
        current_zoom = detected
        if abs(current_zoom - previous_zoom) < 0.01:
            break
        grid = calculate_grid(
            fit_zoom,
            current_zoom,
            content_width_fit_css,
            content_height_fit_css,
            clip_width,
            clip_height,
            step_fraction_x,
            step_fraction_y,
            first_tile_margin_px,
            right_bottom_margin_px,
        )
        attempts += 1

    return current_zoom, grid


def drag_canvas(
    page: Page,
    drag_dx: float,
    drag_dy: float,
    width: int,
    height: int,
    pan_scale: float = 1.0,
) -> None:
    # Miro does not always translate an automated drag 1:1 into camera motion.
    # pan_scale compensates for this. Values below 1 shorten every drag.
    drag_dx *= pan_scale
    drag_dy *= pan_scale
    max_dx = max(180.0, width * 0.28)
    max_dy = max(140.0, height * 0.24)
    parts = max(1, math.ceil(max(abs(drag_dx) / max_dx, abs(drag_dy) / max_dy)))
    part_dx = drag_dx / parts
    part_dy = drag_dy / parts

    for _ in range(parts):
        start_x = width * 0.52
        start_y = height * 0.50
        end_x = start_x + part_dx
        end_y = start_y + part_dy

        page.keyboard.press("Escape")
        page.keyboard.down("Space")
        try:
            page.mouse.move(start_x, start_y)
            page.mouse.down()
            page.mouse.move(end_x, end_y, steps=12)
            page.mouse.up()
        finally:
            page.keyboard.up("Space")
        wait_ms(45)


def move_camera(
    page: Page,
    camera_dx: float,
    camera_dy: float,
    width: int,
    height: int,
    pan_scale: float = 1.0,
) -> None:
    # Kamera nach rechts/unten bewegen = Canvas nach links/oben ziehen.
    drag_canvas(page, -camera_dx, -camera_dy, width, height, pan_scale=pan_scale)


def _prepare_translation_probe(data: bytes, max_width: int = 360) -> Image.Image:
    """Create a compact edge image for estimating viewport translation."""
    from io import BytesIO

    with Image.open(BytesIO(data)) as source:
        image = source.convert("L")
        if image.width > max_width:
            new_height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, new_height), Image.Resampling.BILINEAR)
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.FIND_EDGES)
        if image.width > 16 and image.height > 16:
            image = image.crop((5, 5, image.width - 5, image.height - 5))
        return image.copy()


def estimate_axis_shift_css(
    before_bytes: bytes,
    after_bytes: bytes,
    requested_shift_css: float,
    clip_width_css: int,
    clip_height_css: int,
    axis: str,
    max_probe_width: int = 360,
) -> tuple[float | None, float | None]:
    """Estimate camera translation along one axis from two overlapping screenshots.

    Positive camera movement moves visible board content left/up. The search is
    constrained to the requested direction, which avoids false sign inversions.
    """
    before = _prepare_translation_probe(before_bytes, max_width=max_probe_width)
    after = _prepare_translation_probe(after_bytes, max_width=max_probe_width)
    if before.size != after.size:
        after = after.resize(before.size, Image.Resampling.BILINEAR)

    width, height = before.size
    if width < 90 or height < 60 or axis not in {"x", "y"}:
        return None, None

    edge_energy = max(
        float(ImageStat.Stat(before).stddev[0]),
        float(ImageStat.Stat(after).stddev[0]),
    )
    if edge_energy < 1.6:
        return None, None

    sign = 1 if requested_shift_css >= 0 else -1
    css_extent = clip_width_css if axis == "x" else clip_height_css
    probe_extent = width if axis == "x" else height
    expected_probe = abs(requested_shift_css) * probe_extent / max(1, css_extent)
    minimum = max(2, int(expected_probe * 0.20))
    maximum = min(probe_extent - 28, max(minimum + 1, int(expected_probe * 2.60) + 5))
    if minimum >= maximum:
        return None, None

    best_shift: int | None = None
    best_score = float("inf")
    scores: list[float] = []
    x_margin = max(3, int(width * 0.05))
    y_margin = max(3, int(height * 0.05))

    for magnitude in range(minimum, maximum + 1):
        if axis == "x":
            overlap = width - magnitude
            if overlap < 36:
                continue
            if sign > 0:
                a = before.crop((magnitude, y_margin, width, height - y_margin))
                b = after.crop((0, y_margin, overlap, height - y_margin))
            else:
                a = before.crop((0, y_margin, overlap, height - y_margin))
                b = after.crop((magnitude, y_margin, width, height - y_margin))
        else:
            overlap = height - magnitude
            if overlap < 32:
                continue
            if sign > 0:
                a = before.crop((x_margin, magnitude, width - x_margin, height))
                b = after.crop((x_margin, 0, width - x_margin, overlap))
            else:
                a = before.crop((x_margin, 0, width - x_margin, overlap))
                b = after.crop((x_margin, magnitude, width - x_margin, height))

        difference = ImageChops.difference(a, b)
        score = float(ImageStat.Stat(difference).mean[0])
        scores.append(score)
        if score < best_score:
            best_score = score
            best_shift = magnitude

    if best_shift is None or not scores:
        return None, None

    sorted_scores = sorted(scores)
    median_score = sorted_scores[len(sorted_scores) // 2]
    if best_score > 46.0 or (median_score > 0 and best_score / median_score > 0.94):
        return None, best_score

    actual_css = best_shift * css_extent / probe_extent
    return sign * actual_css, best_score


def _navigation_probe(page: Page, clip: dict[str, float]) -> bytes:
    return page.screenshot(
        clip=clip,
        type="jpeg",
        quality=52,
        animations="disabled",
        caret="hide",
        timeout=60_000,
    )


def calibrate_pan_axis(
    page: Page,
    clip: dict[str, float],
    width: int,
    height: int,
    navigation: NavigationConfig,
    axis: str,
) -> float:
    fallback = (
        navigation.pan_scale_fallback_x if axis == "x" else navigation.pan_scale_fallback_y
    )
    if not navigation.auto_calibrate_pan:
        return fallback

    extent = float(clip["width"] if axis == "x" else clip["height"])
    requested = max(140.0, extent * navigation.calibration_fraction)
    label = "horizontal" if axis == "x" else "vertikal"
    print(f"Kalibriere {label} mit einem Testschritt von {requested:.0f} CSS-Pixeln ...")

    moved = False
    try:
        before = _navigation_probe(page, clip)
        move_camera(
            page,
            requested if axis == "x" else 0,
            requested if axis == "y" else 0,
            width,
            height,
            pan_scale=1.0,
        )
        moved = True
        wait_ms(navigation.measurement_wait_ms)
        after = _navigation_probe(page, clip)
        actual, score = estimate_axis_shift_css(
            before,
            after,
            requested,
            int(clip["width"]),
            int(clip["height"]),
            axis,
        )
    except Exception as exc:
        print(f"  Kalibrierung fehlgeschlagen ({exc}); Fallback {fallback:.3f}.")
        actual = None
        score = None
    finally:
        if moved:
            try:
                move_camera(
                    page,
                    -requested if axis == "x" else 0,
                    -requested if axis == "y" else 0,
                    width,
                    height,
                    pan_scale=1.0,
                )
                wait_ms(navigation.measurement_wait_ms)
            except Exception:
                pass

    if actual is None or abs(actual) <= 1:
        detail = f" (Score {score:.2f})" if score is not None else ""
        print(f"  Bewegung nicht sicher messbar{detail}; Fallback {fallback:.3f}.")
        return fallback

    scale = requested / abs(actual)
    scale = max(navigation.pan_scale_min, min(navigation.pan_scale_max, scale))
    print(f"  Angefordert {requested:.0f}px, gemessen {abs(actual):.0f}px; Faktor {scale:.3f}.")
    return scale


def calibrate_navigation(
    page: Page,
    clip: dict[str, float],
    width: int,
    height: int,
    navigation: NavigationConfig,
) -> NavigationState:
    if not navigation.auto_calibrate_pan:
        state = NavigationState(
            scale_x=navigation.pan_scale_fallback_x,
            scale_y=navigation.pan_scale_fallback_y,
        )
        print(
            "Pan-Kalibrierung deaktiviert; "
            f"Faktoren X={state.scale_x:.3f}, Y={state.scale_y:.3f}."
        )
        return state

    focus_board(page, width, height)
    scale_x = calibrate_pan_axis(page, clip, width, height, navigation, "x")
    scale_y = calibrate_pan_axis(page, clip, width, height, navigation, "y")
    return NavigationState(scale_x=scale_x, scale_y=scale_y)


def _precise_axis_chunk(
    page: Page,
    desired_delta: float,
    axis: str,
    clip: dict[str, float],
    width: int,
    height: int,
    navigation: NavigationConfig,
    state: NavigationState,
) -> dict[str, Any]:
    """Move one overlapping chunk and correct measured over/undershoot."""
    remaining = desired_delta
    measured_total = 0.0
    measurements = 0
    last_score: float | None = None
    max_attempts = 1 + navigation.max_correction_attempts

    for attempt in range(max_attempts):
        if abs(remaining) <= navigation.movement_tolerance_px:
            break

        before = _navigation_probe(page, clip) if navigation.precise_navigation else None
        safe_delta = remaining * navigation.command_safety_factor
        scale = state.scale_x if axis == "x" else state.scale_y
        move_camera(
            page,
            safe_delta if axis == "x" else 0,
            safe_delta if axis == "y" else 0,
            width,
            height,
            pan_scale=scale,
        )
        wait_ms(navigation.measurement_wait_ms)

        if not navigation.precise_navigation or before is None:
            measured_total += safe_delta
            remaining -= safe_delta
            continue

        after = _navigation_probe(page, clip)
        actual, last_score = estimate_axis_shift_css(
            before,
            after,
            safe_delta,
            int(clip["width"]),
            int(clip["height"]),
            axis,
        )
        if actual is None or actual * desired_delta <= 0:
            # The conservative command_safety_factor makes an unmeasurable move
            # more likely to under-shoot than to skip content.
            measured_total += safe_delta
            remaining -= safe_delta
            break

        measurements += 1
        measured_total += actual
        remaining -= actual

        if navigation.adaptive_scale and abs(actual) >= 4:
            old_scale = state.scale_x if axis == "x" else state.scale_y
            corrected = old_scale * abs(safe_delta / actual)
            corrected = max(navigation.pan_scale_min, min(navigation.pan_scale_max, corrected))
            updated = old_scale * 0.55 + corrected * 0.45
            if axis == "x":
                state.scale_x = updated
            else:
                state.scale_y = updated

    return {
        "requested_css": desired_delta,
        "measured_css": measured_total,
        "remaining_css": remaining,
        "measurements": measurements,
        "score": last_score,
    }


def move_camera_precise(
    page: Page,
    camera_dx: float,
    camera_dy: float,
    clip: dict[str, float],
    width: int,
    height: int,
    navigation: NavigationConfig,
    state: NavigationState,
) -> list[dict[str, Any]]:
    """Move in small overlapping chunks, measuring and correcting each chunk."""
    results: list[dict[str, Any]] = []
    for axis, total, extent in (
        ("x", camera_dx, float(clip["width"])),
        ("y", camera_dy, float(clip["height"])),
    ):
        if abs(total) < 0.5:
            continue
        max_chunk = max(120.0, extent * navigation.max_measured_step_fraction)
        remaining_total = total
        iterations = 0
        while abs(remaining_total) > navigation.movement_tolerance_px:
            iterations += 1
            if iterations > 500:
                raise RuntimeError(
                    f"Präzise Kamerabewegung auf Achse {axis} konvergiert nicht."
                )
            chunk = math.copysign(min(abs(remaining_total), max_chunk), remaining_total)
            result = _precise_axis_chunk(
                page,
                chunk,
                axis,
                clip,
                width,
                height,
                navigation,
                state,
            )
            results.append({"axis": axis, **result})
            achieved = chunk - float(result["remaining_css"])
            if abs(achieved) < 0.5:
                # Avoid an endless loop if the scene is completely unmeasurable.
                conservative = chunk * navigation.command_safety_factor
                remaining_total -= conservative
            else:
                remaining_total -= achieved
    return results

def _probe_image_from_bytes(data: bytes, max_width: int = 320) -> Image.Image:
    from io import BytesIO

    with Image.open(BytesIO(data)) as image:
        image = image.convert("L")
        if image.width > max_width:
            height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, height), Image.Resampling.BILINEAR)
        return image.copy()


def _probe_difference(previous: Image.Image, current: Image.Image) -> float:
    if previous.size != current.size:
        current = current.resize(previous.size, Image.Resampling.BILINEAR)
    difference = ImageChops.difference(previous, current)
    return float(ImageStat.Stat(difference).mean[0])


def wait_for_render_stability(
    page: Page,
    clip: dict[str, float],
    capture_config: CaptureConfig,
    initial_wait_ms: int,
) -> tuple[int, float | None]:
    """Wait until the visible tile stops changing.

    Miro virtualizes board content. A fixed sleep can therefore capture the canvas
    before shapes/text have been painted. We take small JPEG probes until two
    consecutive probes are sufficiently similar, or until the maximum wait expires.
    """
    wait_ms(initial_wait_ms)
    previous_bytes = page.screenshot(
        clip=clip,
        type="jpeg",
        quality=35,
        animations="disabled",
        caret="hide",
        timeout=60_000,
    )
    previous = _probe_image_from_bytes(previous_bytes)
    stable_matches = 0
    waited = initial_wait_ms
    last_difference: float | None = None

    while waited < capture_config.stability_max_wait_ms:
        wait_ms(capture_config.stability_interval_ms)
        waited += capture_config.stability_interval_ms
        current_bytes = page.screenshot(
            clip=clip,
            type="jpeg",
            quality=35,
            animations="disabled",
            caret="hide",
            timeout=60_000,
        )
        current = _probe_image_from_bytes(current_bytes)
        last_difference = _probe_difference(previous, current)
        if last_difference <= capture_config.stability_difference_threshold:
            stable_matches += 1
        else:
            stable_matches = 0
        previous = current
        if stable_matches >= capture_config.stability_required_matches:
            break

    return waited, last_difference


def analyze_blank_tile(data: bytes, config: BlankFilterConfig) -> dict[str, float | bool | tuple[int, int, int]]:
    """Classify an obvious empty Miro canvas tile.

    The detector deliberately errs on the side of keeping a tile. Blank candidates
    are moved to a subfolder instead of being deleted, so false positives remain
    recoverable.
    """
    from io import BytesIO

    with Image.open(BytesIO(data)) as source:
        image = source.convert("RGB")
        max_width = 640
        if image.width > max_width:
            height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, height), Image.Resampling.BILINEAR)
        if config.blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(config.blur_radius))

        quantized = image.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
        histogram = quantized.histogram()
        dominant_index = max(range(len(histogram)), key=histogram.__getitem__)
        palette = quantized.getpalette() or []
        offset = dominant_index * 3
        background = tuple(palette[offset:offset + 3])
        if len(background) != 3:
            background = (255, 255, 255)

        background_image = Image.new("RGB", image.size, background)
        difference = ImageChops.difference(image, background_image).convert("L")
        histogram_diff = difference.histogram()
        changed_pixels = sum(histogram_diff[config.difference_threshold + 1:])
        total_pixels = image.width * image.height
        content_fraction = changed_pixels / max(1, total_pixels)
        contrast_stddev = float(ImageStat.Stat(difference).stddev[0])

        is_blank = (
            content_fraction < config.minimum_content_fraction
            and contrast_stddev < config.minimum_contrast_stddev
        )
        return {
            "is_blank": is_blank,
            "content_fraction": content_fraction,
            "contrast_stddev": contrast_stddev,
            "background_rgb": background,
        }


def capture_tile_with_retry(
    page: Page,
    clip: dict[str, float],
    render_wait_ms: int,
    capture_config: CaptureConfig,
    blank_filter: BlankFilterConfig,
) -> tuple[bytes, dict[str, Any]]:
    """Capture a stable full-resolution PNG and retry apparent blank results."""
    last_error: Exception | None = None
    attempts = max(1, blank_filter.enabled * capture_config.blank_retry_count + 1)
    best_data: bytes | None = None
    best_analysis: dict[str, Any] | None = None

    for attempt in range(1, attempts + 1):
        try:
            waited_ms, last_difference = wait_for_render_stability(
                page,
                clip,
                capture_config,
                render_wait_ms if attempt == 1 else capture_config.blank_retry_wait_ms,
            )
            data = page.screenshot(
                clip=clip,
                type="png",
                animations="disabled",
                caret="hide",
                timeout=60_000,
            )
            analysis: dict[str, Any] = {
                "attempt": attempt,
                "waited_ms": waited_ms,
                "last_probe_difference": last_difference,
                "is_blank": False,
            }
            if blank_filter.enabled:
                analysis.update(analyze_blank_tile(data, blank_filter))

            best_data = data
            best_analysis = analysis
            if not bool(analysis.get("is_blank")):
                return data, analysis

            if attempt < attempts:
                page.bring_to_front()
                print(
                    "  Kachel wirkt leer; warte länger und versuche die Darstellung erneut "
                    f"({attempt}/{attempts - 1}) ..."
                )
        except Exception as exc:
            last_error = exc
            print(f"  Aufnahme-Versuch {attempt}/{attempts} fehlgeschlagen: {exc}")
            wait_ms(1000 * attempt)

    if best_data is not None and best_analysis is not None:
        return best_data, best_analysis
    raise RuntimeError("Screenshot konnte nicht aufgenommen werden.") from last_error



def capture_tile_fast(
    page: Page,
    clip: dict[str, float],
    capture_config: CaptureConfig,
    blank_filter: BlankFilterConfig,
) -> tuple[bytes | None, dict[str, Any]]:
    """Fast path: one cheap JPEG probe, then a PNG only when content is present."""
    wait_ms(capture_config.post_move_wait_ms)
    probe = page.screenshot(
        clip=clip,
        type="jpeg",
        quality=capture_config.probe_jpeg_quality,
        animations="disabled",
        caret="hide",
        timeout=60_000,
    )
    analysis: dict[str, Any] = {
        "mode": "fast_probe",
        "waited_ms": capture_config.post_move_wait_ms,
        "is_blank": False,
    }
    if blank_filter.enabled:
        analysis.update(analyze_blank_tile(probe, blank_filter))

    if bool(analysis.get("is_blank")) and not capture_config.save_blank_tiles:
        return None, analysis

    wait_ms(capture_config.content_capture_wait_ms)
    data = page.screenshot(
        clip=clip,
        type="png",
        animations="disabled",
        caret="hide",
        timeout=60_000,
    )
    analysis["waited_ms"] = capture_config.post_move_wait_ms + capture_config.content_capture_wait_ms
    return data, analysis


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def create_preview(
    output_dir: Path,
    tile_entries: list[dict[str, Any]],
    rows: int,
    cols: int,
    step_x_css: int,
    step_y_css: int,
    clip_width_css: int,
    preview_scale: float,
) -> Path | None:
    available = [entry for entry in tile_entries if entry.get("relative_path")]
    if not available:
        return None

    first_path = output_dir / str(available[0]["relative_path"])
    with Image.open(first_path) as first:
        tile_width_px, tile_height_px = first.size

    device_scale = tile_width_px / clip_width_css
    scale = preview_scale
    estimated_width = int(((cols - 1) * step_x_css * device_scale + tile_width_px) * scale)
    estimated_height = int(((rows - 1) * step_y_css * device_scale + tile_height_px) * scale)

    max_dimension = 30000
    if max(estimated_width, estimated_height) > max_dimension:
        scale *= max_dimension / max(estimated_width, estimated_height)
        estimated_width = int(((cols - 1) * step_x_css * device_scale + tile_width_px) * scale)
        estimated_height = int(((rows - 1) * step_y_css * device_scale + tile_height_px) * scale)

    estimated_width = max(1, estimated_width)
    estimated_height = max(1, estimated_height)
    canvas = Image.new("RGB", (estimated_width, estimated_height), "white")

    for entry in available:
        tile_path = output_dir / str(entry["relative_path"])
        if not tile_path.exists():
            continue
        with Image.open(tile_path) as tile:
            tile = tile.convert("RGB")
            new_size = (max(1, int(tile.width * scale)), max(1, int(tile.height * scale)))
            tile = tile.resize(new_size, Image.Resampling.LANCZOS)
            x = int(int(entry["col"]) * step_x_css * device_scale * scale)
            y = int(int(entry["row"]) * step_y_css * device_scale * scale)
            canvas.paste(tile, (x, y))

    preview_path = output_dir / "overview_preview.jpg"
    canvas.save(preview_path, quality=88, optimize=True)
    return preview_path



def _prepare_pdf_image(
    source_path: Path,
    target_path: Path,
    image_format: str,
    jpeg_quality: int,
    image_scale: float,
) -> None:
    """Create a PDF-ready image while keeping the original tile untouched."""
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)

        if image_scale < 1.0:
            new_size = (
                max(1, round(image.width * image_scale)),
                max(1, round(image.height * image_scale)),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        if image_format in {"jpeg", "jpg"}:
            if image.mode in {"RGBA", "LA"}:
                background = Image.new("RGB", image.size, "white")
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
                image = background
            else:
                image = image.convert("RGB")
            image.save(
                target_path,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
                progressive=False,
                subsampling="4:2:0",
            )
        else:
            image.save(target_path, format="PNG", optimize=True)


def create_pdf_from_tiles(
    output_dir: Path,
    tile_entries: list[dict[str, Any]],
    pdf_filename: str,
    include_blank_tiles: bool,
    image_format: str,
    jpeg_quality: int,
    image_scale: float,
) -> tuple[Path, int]:
    """Create one PDF page per selected tile in strict row-major capture order."""
    selected = sorted(
        (
            entry for entry in tile_entries
            if entry.get("relative_path")
            and (include_blank_tiles or not entry.get("is_blank", False))
        ),
        key=lambda entry: (int(entry["row"]), int(entry["col"])),
    )
    tile_paths = [output_dir / str(entry["relative_path"]) for entry in selected]
    missing = [path.name for path in tile_paths if not path.exists()]
    if missing:
        example = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"PDF kann nicht erstellt werden: {len(missing)} Bilddatei(en) fehlen: "
            f"{example}{suffix}"
        )
    if not tile_paths:
        raise RuntimeError("PDF kann nicht erstellt werden: Es wurden keine passenden Kacheln gefunden.")

    normalized_format = "jpeg" if image_format in {"jpeg", "jpg"} else "png"
    use_original_pngs = normalized_format == "png" and image_scale == 1.0

    pdf_path = output_dir / pdf_filename
    temporary_path = pdf_path.with_suffix(pdf_path.suffix + ".tmp")

    try:
        if use_original_pngs:
            pdf_sources = tile_paths
            with temporary_path.open("wb") as output_stream:
                img2pdf.convert([str(path) for path in pdf_sources], outputstream=output_stream)
        else:
            extension = ".jpg" if normalized_format == "jpeg" else ".png"
            with tempfile.TemporaryDirectory(prefix="miro-pdf-images-", dir=output_dir) as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                pdf_sources: list[Path] = []
                for index, source_path in enumerate(tile_paths, start=1):
                    target_path = temp_dir / f"page_{index:05d}{extension}"
                    _prepare_pdf_image(
                        source_path,
                        target_path,
                        normalized_format,
                        jpeg_quality,
                        image_scale,
                    )
                    pdf_sources.append(target_path)

                with temporary_path.open("wb") as output_stream:
                    img2pdf.convert([str(path) for path in pdf_sources], outputstream=output_stream)

        temporary_path.replace(pdf_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return pdf_path, len(tile_paths)



def zoom_to_fit_quick(
    page: Page,
    width: int,
    height: int,
    wait_ms_value: int,
) -> None:
    """Reset to Miro's reproducible fit view without spending seconds reading UI state."""
    focus_board(page, width, height)
    page.keyboard.press("Alt+Digit1")
    wait_ms(max(wait_ms_value, 220))


ZOOM_IN_COMMAND = "zoom_in"
ZOOM_OUT_COMMAND = "zoom_out"


def _normalize_zoom_command(command: str) -> str:
    if command in {
        ZOOM_IN_COMMAND,
        "Control+Shift+Equal",
        "Control+Equal",
        "Control+NumpadAdd",
    }:
        return ZOOM_IN_COMMAND
    if command in {
        ZOOM_OUT_COMMAND,
        "Control+Minus",
        "Control+NumpadSubtract",
    }:
        return ZOOM_OUT_COMMAND
    raise ValueError(f"Unbekannter Zoombefehl: {command}")


def _zoom_shortcut_candidates(command: str) -> tuple[str, ...]:
    normalized = _normalize_zoom_command(command)
    if normalized == ZOOM_IN_COMMAND:
        # Ctrl++ is Miro's documented shortcut. NumpadAdd is layout-independent
        # and works reliably on German Windows layouts. The Equal variants remain
        # as fallbacks for keyboards without a numeric keypad.
        return ("Control+NumpadAdd", "Control+Shift+Equal", "Control+Equal")
    return ("Control+NumpadSubtract", "Control+Minus")


def _inverse_zoom_key(command: str) -> str:
    normalized = _normalize_zoom_command(command)
    return ZOOM_OUT_COMMAND if normalized == ZOOM_IN_COMMAND else ZOOM_IN_COMMAND


def _poll_numeric_zoom(page: Page, timeout_ms: int = 1500) -> float | None:
    deadline = time.monotonic() + max(100, timeout_ms) / 1000.0
    last: float | None = None
    while time.monotonic() < deadline:
        detected, _ = read_visible_zoom(page)
        if detected is not None:
            last = detected
            # Two short reads avoid accepting the old label while Miro is still
            # applying a preceding zoom command.
            wait_ms(45)
            confirmed, _ = read_visible_zoom(page)
            return confirmed if confirmed is not None else last
        wait_ms(70)
    return last


def _wait_for_zoom_change(
    page: Page,
    previous_zoom: float | None,
    timeout_ms: int,
) -> float | None:
    deadline = time.monotonic() + max(200, timeout_ms) / 1000.0
    last: float | None = None
    while time.monotonic() < deadline:
        detected, _ = read_visible_zoom(page)
        if detected is not None:
            last = detected
            if previous_zoom is None or abs(detected - previous_zoom) >= 0.001:
                return detected
        wait_ms(65)
    return last


def _press_zoom_command(
    page: Page,
    command: str,
    previous_zoom: float | None,
    step_wait_ms: int,
    *,
    require_detected_change: bool = True,
) -> tuple[float | None, str]:
    """Apply exactly one Miro zoom step and wait for the UI to acknowledge it.

    The old exporter replayed many Ctrl+Plus events only 65 ms apart.  On large
    boards Miro occasionally dropped some of those events, which made a manually
    selected zoom level appear to do nothing or end at the wrong scale.  This
    helper waits for the visible percentage to change before continuing.
    """
    timeout_ms = max(900, step_wait_ms * 8)
    candidates = _zoom_shortcut_candidates(command)
    unsupported_shortcuts: list[str] = []
    attempted_shortcuts: list[str] = []
    for index, shortcut in enumerate(candidates):
        attempted_shortcuts.append(shortcut)
        try:
            page.keyboard.press(shortcut)
        except PlaywrightError as exc:
            # A typo such as the former "Control+Add" must never abort the
            # complete export. Unsupported aliases are skipped and the next
            # layout-independent candidate is tried.
            unsupported_shortcuts.append(f"{shortcut}: {exc}")
            continue
        wait_ms(max(step_wait_ms, 110))
        detected = _wait_for_zoom_change(page, previous_zoom, timeout_ms)
        if detected is not None and (
            previous_zoom is None or abs(detected - previous_zoom) >= 0.001
        ):
            return detected, shortcut

        # When the label is temporarily '<1%', no numeric change can be read.
        # Do not send another fallback shortcut immediately, because the first
        # one may already have worked.  The next loop iteration can establish a
        # numeric value once Miro reaches 1% or more.
        if previous_zoom is None and index == 0:
            return None, shortcut

    if require_detected_change:
        details = ""
        if unsupported_shortcuts:
            details = " Nicht unterstützte Tastenkürzel: " + "; ".join(unsupported_shortcuts)
        raise RuntimeError(
            "Miro hat auf keines der Zoom-Kürzel reagiert "
            f"({', '.join(attempted_shortcuts)}). Klicke einmal auf eine freie "
            "Stelle des Boards, schließe geöffnete Dialoge und starte den Export "
            f"erneut.{details}"
        )
    fallback = attempted_shortcuts[0] if attempted_shortcuts else _normalize_zoom_command(command)
    return _poll_numeric_zoom(page, timeout_ms=300), fallback


def replay_zoom_sequence(
    page: Page,
    sequence: list[str],
    step_wait_ms: int,
    expected_zoom: float | None = None,
) -> float | None:
    current = _poll_numeric_zoom(page, timeout_ms=500)
    for command in sequence:
        current, _ = _press_zoom_command(
            page,
            command,
            current,
            step_wait_ms,
            require_detected_change=current is not None,
        )

    if expected_zoom is not None:
        current = ensure_zoom_level(page, expected_zoom, step_wait_ms, current)
    return current


def reverse_zoom_sequence(page: Page, sequence: list[str], step_wait_ms: int) -> None:
    current = _poll_numeric_zoom(page, timeout_ms=500)
    for command in reversed(sequence):
        current, _ = _press_zoom_command(
            page,
            _inverse_zoom_key(command),
            current,
            step_wait_ms,
            require_detected_change=current is not None,
        )


def ensure_zoom_level(
    page: Page,
    target_zoom: float,
    step_wait_ms: int,
    current_zoom: float | None = None,
    max_steps: int = 12,
) -> float | None:
    """Correct dropped replay events until the requested discrete Miro level is reached."""
    current = current_zoom if current_zoom is not None else _poll_numeric_zoom(page, 1200)
    if current is None:
        return None

    best = current
    best_distance = abs(current - target_zoom)
    for _ in range(max_steps):
        if abs(current - target_zoom) < 0.01:
            return current
        command = ZOOM_IN_COMMAND if current < target_zoom else ZOOM_OUT_COMMAND
        previous = current
        current, _ = _press_zoom_command(page, command, previous, step_wait_ms)
        if current is None:
            return best
        distance = abs(current - target_zoom)
        if distance < best_distance:
            best = current
            best_distance = distance
        crossed = (previous - target_zoom) * (current - target_zoom) <= 0
        if crossed:
            if abs(previous - target_zoom) < abs(current - target_zoom):
                restored, _ = _press_zoom_command(
                    page,
                    _inverse_zoom_key(command),
                    current,
                    step_wait_ms,
                )
                return restored if restored is not None else previous
            return current
    return best


def build_zoom_sequence_from_fit(
    page: Page,
    width: int,
    height: int,
    fit_zoom_hint: float,
    target_zoom: float,
    absolute: AbsoluteNavigationConfig,
) -> tuple[float, list[str]]:
    """Build and verify a replayable sequence from Zoom-to-fit to manual target.

    Every zoom step is acknowledged before the next one is sent.  The resulting
    sequence is then replayed once from a fresh Alt+1 reset and its final zoom is
    verified.  This specifically fixes manual zoom selections on large boards,
    where the previous 65 ms fire-and-forget replay could lose multiple steps.
    """
    zoom_to_fit_quick(page, width, height, absolute.fit_reset_wait_ms)
    current = _poll_numeric_zoom(page, timeout_ms=1800)
    if current is None:
        current = fit_zoom_hint

    if abs(current - target_zoom) < 0.01:
        return current, []

    direction = ZOOM_IN_COMMAND if target_zoom > current else ZOOM_OUT_COMMAND
    sequence: list[str] = []
    best_sequence: list[str] = []
    best_zoom = current
    best_distance = abs(current - target_zoom)
    previous = current
    step_wait_ms = max(absolute.zoom_step_wait_ms, 120)

    for _ in range(70):
        detected, _ = _press_zoom_command(
            page,
            direction,
            previous,
            step_wait_ms,
            require_detected_change=True,
        )
        if detected is None:
            break
        current = detected
        sequence.append(direction)
        distance = abs(current - target_zoom)
        if distance < best_distance - 0.001:
            best_distance = distance
            best_zoom = current
            best_sequence = sequence.copy()

        crossed = (previous - target_zoom) * (current - target_zoom) <= 0
        if crossed or abs(current - previous) < 0.001:
            break
        previous = current

    if not best_sequence and abs(best_zoom - target_zoom) >= 0.01:
        raise RuntimeError(
            f"Der gewünschte Zoom {target_zoom:g}% konnte von Miro nicht erreicht werden."
        )

    # Return to the closest reached level when the last step overshot it.
    excess = len(sequence) - len(best_sequence)
    if excess > 0:
        current_value = current
        for _ in range(excess):
            current_value, _ = _press_zoom_command(
                page,
                _inverse_zoom_key(direction),
                current_value,
                step_wait_ms,
            )
    sequence = best_sequence

    # Verify that the same sequence works after a fresh fit reset.  This catches
    # dropped key events before hundreds of tiles are exported at the wrong zoom.
    zoom_to_fit_quick(page, width, height, absolute.fit_reset_wait_ms)
    verified = replay_zoom_sequence(
        page,
        sequence,
        step_wait_ms,
        expected_zoom=best_zoom,
    )
    if verified is None:
        raise RuntimeError(
            "Die erreichte Miro-Zoomstufe konnte nach der Verifikation nicht gelesen werden."
        )
    if abs(verified - best_zoom) > 0.01:
        raise RuntimeError(
            f"Zoom-Verifikation fehlgeschlagen: erwartet {best_zoom:g}%, "
            f"erreicht {verified:g}%."
        )

    zoom_to_fit_quick(page, width, height, absolute.fit_reset_wait_ms)
    return verified, sequence

def calculate_grid_from_scale(
    scale_x: float,
    scale_y: float,
    content_width_fit_css: float,
    content_height_fit_css: float,
    clip_width: int,
    clip_height: int,
    step_fraction_x: float,
    step_fraction_y: float,
    first_tile_margin_px: int,
    right_bottom_margin_px: int,
    minimum_overlap_fraction_x: float = 0.0,
    minimum_overlap_fraction_y: float = 0.0,
) -> Grid:
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Empirische Zoomfaktoren müssen positiv sein.")
    estimated_width = (
        content_width_fit_css * scale_x
        + first_tile_margin_px
        + right_bottom_margin_px
    )
    estimated_height = (
        content_height_fit_css * scale_y
        + first_tile_margin_px
        + right_bottom_margin_px
    )
    requested_step_x = max(160, min(clip_width - 80, round(clip_width * step_fraction_x)))
    requested_step_y = max(140, min(clip_height - 80, round(clip_height * step_fraction_y)))
    overlap_limited_step_x = max(160, round(clip_width * (1.0 - minimum_overlap_fraction_x)))
    overlap_limited_step_y = max(140, round(clip_height * (1.0 - minimum_overlap_fraction_y)))
    step_x = min(requested_step_x, overlap_limited_step_x)
    step_y = min(requested_step_y, overlap_limited_step_y)
    cols = max(1, math.ceil(max(0.0, estimated_width - clip_width) / step_x) + 1)
    rows = max(1, math.ceil(max(0.0, estimated_height - clip_height) / step_y) + 1)
    return Grid(
        estimated_width=estimated_width,
        estimated_height=estimated_height,
        step_x=step_x,
        step_y=step_y,
        cols=cols,
        rows=rows,
    )


def tile_intersects_overview_content_scaled(
    overview: FitOverview,
    row: int,
    col: int,
    grid: Grid,
    scale_x: float,
    scale_y: float,
    clip_width_css: int,
    clip_height_css: int,
    bounds_config: BoundsConfig,
) -> bool:
    if not bounds_config.occupancy_skip_enabled:
        return True
    board_left_export = col * grid.step_x - bounds_config.first_tile_margin_px
    board_top_export = row * grid.step_y - bounds_config.first_tile_margin_px
    left_fit_css = overview.bounds.left_css + board_left_export / scale_x
    top_fit_css = overview.bounds.top_css + board_top_export / scale_y
    right_fit_css = left_fit_css + clip_width_css / scale_x
    bottom_fit_css = top_fit_css + clip_height_css / scale_y

    padding = bounds_config.occupancy_padding_fit_px
    left_fit_css -= padding
    top_fit_css -= padding
    right_fit_css += padding
    bottom_fit_css += padding

    x0 = max(0, math.floor(left_fit_css * overview.mask_scale_x))
    y0 = max(0, math.floor(top_fit_css * overview.mask_scale_y))
    x1 = min(overview.mask.width, math.ceil(right_fit_css * overview.mask_scale_x))
    y1 = min(overview.mask.height, math.ceil(bottom_fit_css * overview.mask_scale_y))
    if x1 <= x0 or y1 <= y0:
        return False
    return overview.mask.crop((x0, y0, x1, y1)).getbbox() is not None


def plan_candidate_cells_scaled(
    overview: FitOverview,
    grid: Grid,
    scale_x: float,
    scale_y: float,
    clip_width_css: int,
    clip_height_css: int,
    bounds_config: BoundsConfig,
) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row in range(grid.rows)
        for col in range(grid.cols)
        if tile_intersects_overview_content_scaled(
            overview,
            row,
            col,
            grid,
            scale_x,
            scale_y,
            clip_width_css,
            clip_height_css,
            bounds_config,
        )
    ]


def _center_detected_content_at_fit(
    page: Page,
    overview: FitOverview,
    clip: dict[str, float],
    width: int,
    height: int,
    absolute: AbsoluteNavigationConfig,
) -> None:
    """Place the detected content centre near the zoom anchor before calibration.

    On very wide boards the normal Zoom-to-fit centre can lie in an almost empty
    black region.  Translation matching then locks onto repeated table patterns
    and underestimates the fit/export scale, which makes adjacent tiles jump too
    far apart.  Centering real content gives the calibration screenshots strong
    features on both zoom levels.
    """
    zoom_to_fit_quick(page, width, height, absolute.fit_reset_wait_ms)
    content_center_x = overview.bounds.left_css + overview.bounds.width_css / 2.0
    content_center_y = overview.bounds.top_css + overview.bounds.height_css / 2.0
    clip_center_x = float(clip["width"]) / 2.0
    clip_center_y = float(clip["height"]) / 2.0
    drag_canvas(
        page,
        clip_center_x - content_center_x,
        clip_center_y - content_center_y,
        width,
        height,
        pan_scale=1.0,
    )
    wait_ms(max(absolute.post_zoom_wait_ms, 100))


def _calibrate_scale_axis(
    page: Page,
    overview: FitOverview,
    clip: dict[str, float],
    width: int,
    height: int,
    nominal_scale: float,
    zoom_sequence: list[str],
    config: Config,
    axis: str,
) -> float | None:
    """Measure export/fit scale from the same camera displacement at both zooms."""
    absolute = config.absolute_navigation
    if absolute.center_content_before_scale_calibration:
        _center_detected_content_at_fit(page, overview, clip, width, height, absolute)
    else:
        zoom_to_fit_quick(page, width, height, absolute.fit_reset_wait_ms)
    before_fit = _navigation_probe(page, clip)

    replay_zoom_sequence(page, zoom_sequence, absolute.zoom_step_wait_ms)
    wait_ms(absolute.post_zoom_wait_ms)
    before_target = _navigation_probe(page, clip)

    extent = float(clip["width"] if axis == "x" else clip["height"])
    requested = max(260.0, extent * absolute.calibration_drag_fraction)
    requested = min(requested, extent * 0.70)
    move_camera(
        page,
        requested if axis == "x" else 0,
        requested if axis == "y" else 0,
        width,
        height,
        pan_scale=1.0,
    )
    wait_ms(max(absolute.post_zoom_wait_ms, 100))
    after_target = _navigation_probe(page, clip)

    target_shift, _ = estimate_axis_shift_css(
        before_target,
        after_target,
        requested,
        int(clip["width"]),
        int(clip["height"]),
        axis,
        max_probe_width=absolute.calibration_probe_width,
    )
    if target_shift is None or abs(target_shift) < 8:
        zoom_to_fit_quick(page, width, height, absolute.fit_reset_wait_ms)
        return None

    reverse_zoom_sequence(page, zoom_sequence, absolute.zoom_step_wait_ms)
    wait_ms(max(absolute.post_zoom_wait_ms, 100))
    after_fit = _navigation_probe(page, clip)

    expected_fit = target_shift / max(1.0, nominal_scale)
    fit_shift, _ = estimate_axis_shift_css(
        before_fit,
        after_fit,
        expected_fit,
        int(clip["width"]),
        int(clip["height"]),
        axis,
        max_probe_width=absolute.calibration_probe_width,
    )
    zoom_to_fit_quick(page, width, height, absolute.fit_reset_wait_ms)
    if fit_shift is None or abs(fit_shift) < 1.0:
        return None
    ratio = abs(target_shift / fit_shift)
    if ratio < 1.0 or ratio > 1000.0:
        return None
    return ratio


def calibrate_absolute_scale(
    page: Page,
    overview: FitOverview,
    clip: dict[str, float],
    width: int,
    height: int,
    fit_zoom: float,
    actual_zoom: float,
    zoom_sequence: list[str],
    config: Config,
) -> tuple[float, float]:
    nominal = max(1.0, actual_zoom / max(0.01, fit_zoom))
    if not config.absolute_navigation.calibrate_scale or not zoom_sequence:
        print(f"Empirischer Zoomfaktor deaktiviert; verwende nominell {nominal:.3f}.")
        return nominal, nominal

    print("Kalibriere den tatsächlichen Export-/Fit-Maßstab ...")
    measured_x = _calibrate_scale_axis(
        page, overview, clip, width, height, nominal, zoom_sequence, config, "x"
    )
    measured_y = _calibrate_scale_axis(
        page, overview, clip, width, height, nominal, zoom_sequence, config, "y"
    )
    measured = [value for value in (measured_x, measured_y) if value is not None]
    if not measured:
        print(f"  Maßstab nicht sicher messbar; verwende nominell {nominal:.3f}.")
        return nominal, nominal

    if config.absolute_navigation.prefer_horizontal_scale and measured_x is not None:
        # Miro zoom is isotropic.  The horizontal measurement is normally much
        # more reliable on wide boards because it contains many more features.
        # Using one common factor also prevents rows/columns from drifting apart.
        scale = measured_x
        if measured_y is not None:
            disagreement = abs(measured_x - measured_y) / max(measured_x, measured_y)
            if disagreement > 0.20:
                print(
                    f"  X/Y-Messung weicht ab ({measured_x:.3f}/{measured_y:.3f}); "
                    f"verwende wegen des breiten Boards den horizontalen Faktor {scale:.3f}."
                )
    elif len(measured) == 2:
        disagreement = abs(measured[0] - measured[1]) / max(measured)
        if disagreement <= 0.25:
            scale = sum(measured) / 2.0
        else:
            scale = min(measured, key=lambda value: abs(value - nominal))
            print(
                f"  X/Y-Messung weicht stark ab ({measured[0]:.3f}/{measured[1]:.3f}); "
                f"verwende den plausibleren Wert {scale:.3f}."
            )
    else:
        scale = measured[0]
    print(
        f"  Nominell {nominal:.3f}; gemessen X "
        f"{measured_x if measured_x is not None else 'n/a'}, Y "
        f"{measured_y if measured_y is not None else 'n/a'}; verwende {scale:.3f}."
    )
    return scale, scale


def position_tile_absolutely_from_fit(
    page: Page,
    overview: FitOverview,
    row: int,
    col: int,
    grid: Grid,
    scale_x: float,
    scale_y: float,
    actual_zoom: float,
    zoom_sequence: list[str],
    width: int,
    height: int,
    clip_width: int,
    clip_height: int,
    config: Config,
) -> None:
    """Navigate to a tile without inheriting any prior camera error.

    Every tile starts from Alt+1.  At fit zoom, the requested board point is moved
    to the keyboard zoom anchor.  Replaying the exact zoom sequence then reveals
    the requested rectangle.  No calibrated long-distance pan and no cumulative
    row drift are involved.
    """
    absolute = config.absolute_navigation
    zoom_to_fit_quick(page, width, height, absolute.fit_reset_wait_ms)

    board_center_x_export = (
        col * grid.step_x
        - config.bounds.first_tile_margin_px
        + clip_width / 2.0
    )
    board_center_y_export = (
        row * grid.step_y
        - config.bounds.first_tile_margin_px
        + clip_height / 2.0
    )
    target_fit_x = overview.bounds.left_css + board_center_x_export / scale_x
    target_fit_y = overview.bounds.top_css + board_center_y_export / scale_y

    clip_center_x = clip_width / 2.0
    clip_center_y = clip_height / 2.0
    zoom_anchor_x = width / 2.0 - config.crop.left
    zoom_anchor_y = height / 2.0 - config.crop.top

    # If zoom is anchored at the viewport centre, place the wanted tile centre a
    # tiny pre-compensated distance away so it lands at the clip centre afterwards.
    prezoom_x = zoom_anchor_x + (clip_center_x - zoom_anchor_x) / scale_x
    prezoom_y = zoom_anchor_y + (clip_center_y - zoom_anchor_y) / scale_y
    content_dx = prezoom_x - target_fit_x
    content_dy = prezoom_y - target_fit_y
    drag_canvas(page, content_dx, content_dy, width, height, pan_scale=1.0)

    reached_zoom = replay_zoom_sequence(
        page,
        zoom_sequence,
        absolute.zoom_step_wait_ms,
        expected_zoom=actual_zoom,
    )
    if reached_zoom is not None and abs(reached_zoom - actual_zoom) > 0.01:
        raise RuntimeError(
            f"Kachel r{row} c{col}: erwarteter Zoom {actual_zoom:g}%, "
            f"erreicht {reached_zoom:g}%."
        )
    wait_ms(absolute.post_zoom_wait_ms)


def save_tile_plan_preview(
    output_path: Path,
    overview: FitOverview,
    candidate_cells: list[tuple[int, int]],
    grid: Grid,
    scale_x: float,
    scale_y: float,
    clip_width_css: int,
    clip_height_css: int,
    bounds_config: BoundsConfig,
) -> None:
    from io import BytesIO

    with Image.open(BytesIO(overview.screenshot_bytes)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    sx = image.width / max(1, clip_width_css)
    sy = image.height / max(1, clip_height_css)

    # Content bounds in green.
    draw.rectangle(
        (
            overview.bounds.left_css * sx,
            overview.bounds.top_css * sy,
            overview.bounds.right_css * sx,
            overview.bounds.bottom_css * sy,
        ),
        outline=(0, 255, 0),
        width=max(1, round(image.width / 700)),
    )

    line_width = max(1, round(image.width / 1000))
    for row, col in candidate_cells:
        board_left = col * grid.step_x - bounds_config.first_tile_margin_px
        board_top = row * grid.step_y - bounds_config.first_tile_margin_px
        left = (overview.bounds.left_css + board_left / scale_x) * sx
        top = (overview.bounds.top_css + board_top / scale_y) * sy
        right = left + (clip_width_css / scale_x) * sx
        bottom = top + (clip_height_css / scale_y) * sy
        draw.rectangle((left, top, right, bottom), outline=(255, 80, 40), width=line_width)
        if right - left > 35 and bottom - top > 22:
            draw.text((left + 2, top + 2), f"r{row} c{col}", fill=(255, 255, 0))
    image.save(output_path, format="PNG")


def main() -> int:
    config = load_config()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = config.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Miro Board Exporter - Python/CDP-Version 10.2")
    print("==========================================")
    print("Jede Kachel wird absolut aus der Zoom-to-fit-Ansicht positioniert.")
    print("Dadurch gibt es keinen kumulativen Kamera- oder Reihen-Drift mehr.")

    with sync_playwright() as playwright:
        browser = connect_to_chrome(playwright, config.cdp_url)
        page = choose_miro_page(browser)
        page.bring_to_front()

        print(f"\nVerwendeter Tab: {page.url}")
        print("Vorbereitung im Chrome-Fenster:")
        print("  1. Board vollständig laden lassen.")
        print("  2. Große Seitenleisten, Dialoge, Chat und Kommentare schließen.")
        print("  3. Chrome maximieren und während des Exports nicht bedienen.")
        input("\nDanach hier ENTER drücken ... ")

        width, height = viewport_size(page)
        crop = config.crop
        clip_width = width - crop.left - crop.right
        clip_height = height - crop.top - crop.bottom
        if clip_width < 400 or clip_height < 300:
            raise RuntimeError(
                "Die Crop-Werte in config.json sind für dieses Fenster zu groß. "
                f"Nutzbarer Bereich: {clip_width}x{clip_height}."
            )
        clip = {
            "x": float(crop.left),
            "y": float(crop.top),
            "width": float(clip_width),
            "height": float(clip_height),
        }

        print("\nErmittle Zoom-to-fit und die tatsächlichen Inhaltsgrenzen ...")
        fit_zoom = zoom_to_fit(page, width, height, config.render_wait_ms)
        fit_overview = create_fit_overview(
            page, clip, clip_width, clip_height, config.bounds
        )
        if config.bounds.save_fit_overview:
            (output_dir / "_fit_overview.png").write_bytes(fit_overview.screenshot_bytes)
        print(f"Zoom to fit: {fit_zoom:g}%")
        print(
            "Erkannter Inhalt bei Zoom-to-fit: "
            f"{fit_overview.bounds.width_css:.0f}x{fit_overview.bounds.height_css:.0f} CSS-Pixel, "
            f"Start bei ({fit_overview.bounds.left_css:.0f}, {fit_overview.bounds.top_css:.0f})."
        )

        requested_zoom, desired_max_tiles, resolution_mode = ask_resolution(
            config,
            fit_zoom,
            fit_overview.bounds,
            clip_width,
            clip_height,
        )

        print(f"Bestimme die Miro-Zoomstufe nahe {requested_zoom:g}% ...")
        actual_zoom, zoom_sequence = build_zoom_sequence_from_fit(
            page,
            width,
            height,
            fit_zoom,
            requested_zoom,
            config.absolute_navigation,
        )
        if not zoom_sequence and actual_zoom > fit_zoom + 0.01:
            raise RuntimeError("Die Zielzoom-Sequenz konnte nicht bestimmt werden.")

        scale_x, scale_y = calibrate_absolute_scale(
            page,
            fit_overview,
            clip,
            width,
            height,
            fit_zoom,
            actual_zoom,
            zoom_sequence,
            config,
        )
        grid = calculate_grid_from_scale(
            scale_x,
            scale_y,
            fit_overview.bounds.width_css,
            fit_overview.bounds.height_css,
            clip_width,
            clip_height,
            config.navigation.step_fraction_x,
            config.navigation.step_fraction_y,
            config.bounds.first_tile_margin_px,
            config.bounds.right_bottom_margin_px,
            config.absolute_navigation.minimum_overlap_fraction_x,
            config.absolute_navigation.minimum_overlap_fraction_y,
        )

        # The old calculation used the rounded displayed fit percentage.  After
        # empirical calibration, reduce the final zoom if the real grid exceeds
        # the requested maximum.  Scale is proportional between normal Miro zoom
        # levels, so the expensive fit calibration does not need to be repeated.
        if desired_max_tiles is not None:
            for _ in range(4):
                if grid.tile_count <= desired_max_tiles:
                    break
                correction = math.sqrt(desired_max_tiles / grid.tile_count) * 0.97
                corrected_request = max(fit_zoom, actual_zoom * correction)
                if corrected_request >= actual_zoom * 0.99:
                    break
                previous_zoom = actual_zoom
                new_zoom, new_sequence = build_zoom_sequence_from_fit(
                    page,
                    width,
                    height,
                    fit_zoom,
                    corrected_request,
                    config.absolute_navigation,
                )
                if new_zoom >= previous_zoom - 0.01 or not new_sequence:
                    break
                proportional = new_zoom / previous_zoom
                scale_x *= proportional
                scale_y *= proportional
                actual_zoom = new_zoom
                zoom_sequence = new_sequence
                grid = calculate_grid_from_scale(
                    scale_x,
                    scale_y,
                    fit_overview.bounds.width_css,
                    fit_overview.bounds.height_css,
                    clip_width,
                    clip_height,
                    config.navigation.step_fraction_x,
                    config.navigation.step_fraction_y,
                    config.bounds.first_tile_margin_px,
                    config.bounds.right_bottom_margin_px,
                    config.absolute_navigation.minimum_overlap_fraction_x,
                    config.absolute_navigation.minimum_overlap_fraction_y,
                )

        candidate_cells = plan_candidate_cells_scaled(
            fit_overview,
            grid,
            scale_x,
            scale_y,
            clip_width,
            clip_height,
            config.bounds,
        )
        if not candidate_cells:
            candidate_cells = [
                (row, col) for row in range(grid.rows) for col in range(grid.cols)
            ]

        if config.absolute_navigation.save_tile_plan:
            try:
                save_tile_plan_preview(
                    output_dir / "_tile_plan_fit.png",
                    fit_overview,
                    candidate_cells,
                    grid,
                    scale_x,
                    scale_y,
                    clip_width,
                    clip_height,
                    config.bounds,
                )
            except Exception as exc:
                print(f"Kachelplan-Vorschau konnte nicht erstellt werden: {exc}")

        print(f"Tatsächlicher Export-Zoom: {actual_zoom:g}%")
        print(f"Empirischer Export-/Fit-Faktor: {scale_x:.3f}")
        print(
            f"Inhaltsabdeckung einschließlich Rändern: "
            f"{grid.estimated_width:.0f}x{grid.estimated_height:.0f} CSS-Pixel"
        )
        print(
            f"Kachelschritt: {grid.step_x}px horizontal / {grid.step_y}px vertikal "
            f"(Überlappung {clip_width - grid.step_x}px / {clip_height - grid.step_y}px)"
        )
        print(
            f"Geometrisches Raster: {grid.cols} Spalten x {grid.rows} Zeilen = "
            f"{grid.tile_count} Positionen"
        )
        print(
            f"Die Fit-Inhaltskarte markiert {len(candidate_cells)} Positionen als Kandidaten."
        )
        print("Reihenfolge: strikt links nach rechts, danach nächste Zeile von oben nach unten.")
        print(f"Ausgabeordner: {output_dir}")

        answer = input("\nExport starten? [J/n] ").strip().lower()
        if answer.startswith("n"):
            print("Abgebrochen.")
            return 0

        planned_capture_count = len(candidate_cells)
        if planned_capture_count > config.max_tiles_without_extra_confirmation:
            answer = input(
                f"WARNUNG: Es werden ungefähr {planned_capture_count} Positionen geprüft. "
                "Zum Fortfahren exakt EXPORT eingeben: "
            ).strip()
            if answer != "EXPORT":
                print("Abgebrochen.")
                return 0

        manifest: dict[str, Any] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "exporter_version": "10.2",
            "board_url": page.url,
            "viewport_css_px": {"width": width, "height": height},
            "clip_css_px": clip,
            "fit_zoom_percent_displayed": fit_zoom,
            "resolution_mode": resolution_mode,
            "desired_max_tiles": desired_max_tiles,
            "requested_zoom_percent": round(requested_zoom, 4),
            "actual_zoom_percent": actual_zoom,
            "empirical_export_fit_scale": {
                "x": round(scale_x, 6),
                "y": round(scale_y, 6),
            },
            "zoom_key_sequence": zoom_sequence,
            "content_bounds_at_fit_css_px": {
                "left": round(fit_overview.bounds.left_css, 2),
                "top": round(fit_overview.bounds.top_css, 2),
                "right": round(fit_overview.bounds.right_css, 2),
                "bottom": round(fit_overview.bounds.bottom_css, 2),
                "width": round(fit_overview.bounds.width_css, 2),
                "height": round(fit_overview.bounds.height_css, 2),
            },
            "fit_background_rgb": fit_overview.background_rgb,
            "first_tile_margin_css_px": config.bounds.first_tile_margin_px,
            "right_bottom_margin_css_px": config.bounds.right_bottom_margin_px,
            "estimated_content_coverage_css_px": {
                "width": round(grid.estimated_width, 2),
                "height": round(grid.estimated_height, 2),
            },
            "overlap_css_px": {
                "x": clip_width - grid.step_x,
                "y": clip_height - grid.step_y,
            },
            "step_css_px": {"x": grid.step_x, "y": grid.step_y},
            "navigation": {
                "mode": "absolute_from_fit_per_tile",
                "capture_order": "strict_row_major_left_to_right_top_to_bottom",
                "cumulative_pan": False,
                "reset_to_fit_before_every_tile": True,
                "minimum_overlap_fraction": {
                    "x": config.absolute_navigation.minimum_overlap_fraction_x,
                    "y": config.absolute_navigation.minimum_overlap_fraction_y,
                },
                "content_centered_for_scale_calibration": (
                    config.absolute_navigation.center_content_before_scale_calibration
                ),
            },
            "rows": grid.rows,
            "cols": grid.cols,
            "geometric_tile_count": grid.tile_count,
            "planned_candidate_count": planned_capture_count,
            "occupancy_skip_enabled": config.bounds.occupancy_skip_enabled,
            "status": "running",
            "files": [],
        }
        manifest_path = output_dir / "manifest.json"
        write_manifest(manifest_path, manifest)

        blank_dir = output_dir / config.blank_filter.blank_subfolder
        if config.capture.save_blank_tiles:
            blank_dir.mkdir(parents=True, exist_ok=True)

        completed = 0
        blank_count = 0
        content_count = 0

        try:
            for candidate_index, (row, col) in enumerate(candidate_cells, start=1):
                position_tile_absolutely_from_fit(
                    page,
                    fit_overview,
                    row,
                    col,
                    grid,
                    scale_x,
                    scale_y,
                    actual_zoom,
                    zoom_sequence,
                    width,
                    height,
                    clip_width,
                    clip_height,
                    config,
                )

                if config.capture.fast_mode:
                    data, capture_analysis = capture_tile_fast(
                        page, clip, config.capture, config.blank_filter
                    )
                else:
                    data, capture_analysis = capture_tile_with_retry(
                        page,
                        clip,
                        config.render_wait_ms,
                        config.capture,
                        config.blank_filter,
                    )

                is_blank = bool(capture_analysis.get("is_blank", False))
                grid_index = row * grid.cols + col + 1
                relative_path: str | None = None
                content_index: int | None = None

                if not is_blank:
                    content_count += 1
                    content_index = content_count
                    filename = f"page_{content_index:04d}_r{row:03d}_c{col:03d}.png"
                    if data is None:
                        raise RuntimeError("Inhaltskachel wurde erkannt, aber kein PNG erzeugt.")
                    filepath = output_dir / filename
                    relative_path = filename
                    filepath.write_bytes(data)
                else:
                    blank_count += 1
                    filename = f"blank_{candidate_index:04d}_r{row:03d}_c{col:03d}.png"
                    if data is not None and config.capture.save_blank_tiles:
                        filepath = blank_dir / filename
                        relative_path = str(Path(config.blank_filter.blank_subfolder) / filename)
                        filepath.write_bytes(data)

                completed += 1
                entry = {
                    "capture_index": candidate_index,
                    "content_index": content_index,
                    "grid_index": grid_index,
                    "row": row,
                    "col": col,
                    "filename": filename,
                    "relative_path": relative_path,
                    "camera_x_css": col * grid.step_x,
                    "camera_y_css": row * grid.step_y,
                    "is_blank": is_blank,
                    "capture": capture_analysis,
                }
                manifest["files"].append(entry)
                manifest["content_tile_count"] = content_count
                manifest["blank_tile_count"] = blank_count
                write_manifest(manifest_path, manifest)

                label = "LEER - übersprungen" if is_blank and data is None else ("LEER" if is_blank else "INHALT")
                print(f"[{completed}/{planned_capture_count}] {filename} - {label}")

        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            write_manifest(manifest_path, manifest)
            print("\nExport durch Strg+C unterbrochen. Bereits erzeugte Dateien bleiben erhalten.")
            return 130
        except Exception:
            manifest["status"] = "failed"
            write_manifest(manifest_path, manifest)
            raise

        manifest["status"] = "complete"
        write_manifest(manifest_path, manifest)

        pdf_path: Path | None = None
        if config.create_pdf:
            print(f"\nErzeuge PDF mit einer {config.pdf_image_format.upper()}-Kachel pro Seite ...")
            try:
                pdf_path, pdf_page_count = create_pdf_from_tiles(
                    output_dir,
                    manifest["files"],
                    config.pdf_filename,
                    config.pdf_include_blank_tiles,
                    config.pdf_image_format,
                    config.pdf_jpeg_quality,
                    config.pdf_image_scale,
                )
                manifest["pdf"] = {
                    "filename": pdf_path.name,
                    "page_count": pdf_page_count,
                    "page_order": "strict_row_major",
                    "one_image_per_page": True,
                    "embedded_image_format": config.pdf_image_format,
                    "jpeg_quality": config.pdf_jpeg_quality if config.pdf_image_format in {"jpeg", "jpg"} else None,
                    "image_scale": config.pdf_image_scale,
                    "include_blank_tiles": config.pdf_include_blank_tiles,
                }
                write_manifest(manifest_path, manifest)
                print(f"PDF erstellt: {pdf_path.name} ({pdf_page_count} Seiten)")
            except Exception as exc:
                manifest["status"] = "complete_with_pdf_error"
                manifest["pdf_error"] = str(exc)
                write_manifest(manifest_path, manifest)
                print(f"PDF konnte nicht erstellt werden: {exc}")

        preview_path: Path | None = None
        if config.create_preview:
            print("\nErzeuge verkleinerte Übersicht ...")
            try:
                preview_path = create_preview(
                    output_dir,
                    manifest["files"],
                    grid.rows,
                    grid.cols,
                    grid.step_x,
                    grid.step_y,
                    clip_width,
                    config.preview_scale,
                )
            except Exception as exc:
                print(f"Übersicht konnte nicht erstellt werden: {exc}")

        print("\nFERTIG")
        print(f"Geometrische Rasterpositionen: {grid.tile_count}")
        print(f"Tatsächlich angefahrene Kandidaten: {planned_capture_count}")
        print(f"Kacheln mit Inhalt: {content_count}")
        print(f"Als leer erkannte Kandidaten: {blank_count}")
        print(f"Bilder: {output_dir}")
        print(f"Manifest: {manifest_path}")
        if pdf_path:
            print(f"PDF: {pdf_path}")
        if preview_path:
            print(f"Übersicht: {preview_path}")
        if config.absolute_navigation.save_tile_plan:
            print(f"Kachelplan: {output_dir / '_tile_plan_fit.png'}")
        print("Das separate Chrome-Fenster bleibt geöffnet.")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlaywrightTimeoutError as exc:
        print(f"\nPLAYWRIGHT-TIMEOUT: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        raise SystemExit(1)
