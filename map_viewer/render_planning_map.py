#!/usr/bin/env python3
"""Render saved game phases on the nicer planning-map artwork.

This renderer is intentionally raster-first. The planning map SVG does not
contain one named fill path per province, so we recover province regions from
the visible borders by flood-filling from the printed province labels.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_viewer.render_map import (  # noqa: E402
    BADGE_COLORS,
    POWER_ORDER,
    center_counts,
    data_uri,
    find_phase,
    load_game,
    logo_path_for_model,
    model_map,
    short_model_name,
    split_label,
)

ASSET_DIR = Path(__file__).resolve().parent / "assets"
PLANNING_SVG = ASSET_DIR / "Planning Map (v2).svg"
STANDARD_COORDS = REPO_ROOT / "diplomacy" / "maps" / "standard_coords.json"

POWER_FILL = {
    "AUSTRIA": (184, 86, 77),
    "ENGLAND": (77, 143, 194),
    "FRANCE": (66, 183, 183),
    "GERMANY": (72, 72, 72),
    "ITALY": (99, 184, 79),
    "RUSSIA": (181, 110, 189),
    "TURKEY": (183, 169, 71),
}

PROVINCE_ALIASES = {
    "LIV": "LVP",
}

BADGE_POSITIONS = {
    "ENGLAND": (1180, 1730),
    "FRANCE": (1680, 2580),
    "GERMANY": (2850, 2250),
    "ITALY": (3220, 3450),
    "AUSTRIA": (3900, 2960),
    "RUSSIA": (5050, 1700),
    "TURKEY": (5080, 3520),
}


def svg_local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def find_group(root: ET.Element, group_id: str) -> ET.Element:
    for element in root.iter():
        if svg_local(element.tag) == "g" and element.attrib.get("id") == group_id:
            return element
    raise SystemExit(f"Could not find group {group_id!r} in {PLANNING_SVG}.")


def extract_label_points(svg_path: Path) -> tuple[dict[str, tuple[float, float]], tuple[float, float, float, float]]:
    root = ET.parse(svg_path).getroot()
    view_box = tuple(float(part) for part in root.attrib["viewBox"].split())
    points: dict[str, tuple[float, float]] = {}

    for group_id in ("Province_text", "Water_text"):
        for text in find_group(root, group_id):
            if svg_local(text.tag) != "text":
                continue
            label = "".join(text.itertext()).strip().replace("\n", "")
            match = re.search(r"translate\(([-0-9.]+)\s+([-0-9.]+)\)", text.attrib.get("transform", ""))
            if not label or not match:
                continue
            # Text transforms are the left baseline, not the center of the
            # province. This offset lands inside the province label for this map.
            x = float(match.group(1)) + 40
            y = float(match.group(2)) - 25
            points[PROVINCE_ALIASES.get(label, label)] = (x, y)
    return points, view_box


def render_svg_to_png(svg_path: Path, output_path: Path, width: int) -> None:
    converter = shutil.which("rsvg-convert")
    if not converter:
        raise SystemExit("Cannot render planning map: rsvg-convert is not installed or not on PATH.")
    subprocess.run([converter, "-w", str(width), "-o", str(output_path), str(svg_path)], check=True)


def write_barrier_svg(source_svg: Path, output_svg: Path) -> None:
    tree = ET.parse(source_svg)
    root = tree.getroot()

    # The overlay contains province text and curved globe guide lines. Those are
    # useful visually, but they must not act as walls for flood-fill recovery.
    for parent in root.iter():
        for child in list(parent):
            if child.attrib.get("id") == "Overlay":
                parent.remove(child)

    tree.write(output_svg, encoding="unicode", xml_declaration=True)


def load_province_types() -> dict[str, str]:
    with STANDARD_COORDS.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        PROVINCE_ALIASES.get(province, province): details.get("type", "")
        for province, details in data.get("provinces", {}).items()
    }


def owner_by_location(state: dict[str, Any]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for source in ("influence", "centers"):
        for power, locations in state.get(source, {}).items():
            for location in locations:
                owners[location.split("/", 1)[0].upper()] = power
    return owners


def nearest_component(label_image: np.ndarray, x: int, y: int, radius_limit: int = 80) -> int:
    height, width = label_image.shape
    for radius in range(radius_limit + 1):
        x0, x1 = max(0, x - radius), min(width - 1, x + radius)
        y0, y1 = max(0, y - radius), min(height - 1, y + radius)
        window = label_image[y0 : y1 + 1, x0 : x1 + 1]
        values = np.unique(window[window > 0])
        if values.size:
            return max(((window == value).sum(), int(value)) for value in values)[1]
    return 0


def recolor_base_map(
    image: Image.Image,
    barrier_image: Image.Image,
    phase: dict[str, Any],
    label_points: dict[str, tuple[float, float]],
    province_types: dict[str, str],
    view_box: tuple[float, float, float, float],
) -> Image.Image:
    original = np.array(image.convert("RGB"))
    barrier_source = np.array(barrier_image.convert("RGB"))
    working = original.astype(np.float32)
    scale = image.size[0] / view_box[2]

    dark = (barrier_source[:, :, 0] < 145) & (barrier_source[:, :, 1] < 145) & (barrier_source[:, :, 2] < 145)
    barriers = ndimage.binary_dilation(dark, iterations=6)
    components, _ = ndimage.label(~barriers)

    owners = owner_by_location(phase["state"])
    water_components: set[int] = set()
    for province, (svg_x, svg_y) in label_points.items():
        if province_types.get(province) != "sea":
            continue
        component = nearest_component(components, round(svg_x * scale), round(svg_y * scale))
        if component:
            water_components.add(component)

    component_owner: dict[int, str] = {}
    for province, (svg_x, svg_y) in label_points.items():
        if province_types.get(province) == "sea":
            continue
        owner = owners.get(province)
        if not owner or province not in owners:
            continue
        component = nearest_component(components, round(svg_x * scale), round(svg_y * scale))
        if component in water_components:
            continue
        if component:
            component_owner[component] = owner

    for component, owner in component_owner.items():
        color = np.array(POWER_FILL[owner], dtype=np.float32)
        mask = components == component
        working[mask] = working[mask] * 0.33 + color * 0.67

    return Image.fromarray(np.clip(working, 0, 255).astype(np.uint8), "RGB")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_units(image: Image.Image, phase: dict[str, Any], label_points: dict[str, tuple[float, float]], view_box: tuple[float, float, float, float]) -> None:
    draw = ImageDraw.Draw(image)
    scale = image.size[0] / view_box[2]
    font = load_font(max(14, round(34 * scale)), bold=True)

    for power, units in phase["state"].get("units", {}).items():
        fill = POWER_FILL[power]
        for unit in units:
            pieces = unit.split()
            if len(pieces) < 2:
                continue
            unit_type, location = pieces[0], pieces[1].split("/", 1)[0].upper()
            point = label_points.get(location)
            if not point:
                continue
            x, y = round(point[0] * scale), round(point[1] * scale)
            w, h = round(120 * scale), round(54 * scale)
            box = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)
            draw.rounded_rectangle(box, radius=max(4, round(12 * scale)), fill=fill, outline=(20, 20, 20), width=max(2, round(6 * scale)))
            label = "A" if unit_type == "A" else "F"
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2 - round(2 * scale)), label, fill=(10, 10, 10), font=font)


def draw_header(image: Image.Image, phase: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(image)
    scale = image.size[0] / 7016
    title_font = load_font(max(18, round(72 * scale)), bold=True)
    body_font = load_font(max(12, round(42 * scale)), bold=True)
    padding = round(70 * scale)
    title = phase["name"]
    counts = center_counts(phase)
    draw.text((padding, padding), title, fill=(15, 15, 15), font=title_font)
    draw.text((padding, padding + round(86 * scale)), counts, fill=(15, 15, 15), font=body_font)


def planning_badge_svg(power: str, model_id: str) -> str:
    x, y = BADGE_POSITIONS[power]
    color = BADGE_COLORS[power]
    label_1, label_2 = split_label(short_model_name(model_id))
    logo_path = logo_path_for_model(model_id)
    logo = ""
    if logo_path:
        logo = f'<image x="{x + 48}" y="{y + 48}" width="144" height="144" href="{data_uri(logo_path)}"/>'
    second = ""
    if label_2:
        second = f'<text x="{x + 232}" y="{y + 190}" font-family="Arial, sans-serif" font-size="52" font-weight="700" fill="#1d1d1f">{label_2}</text>'
    return f"""
    <g id="ModelBadge-{power}">
      <rect x="{x}" y="{y}" width="720" height="240" rx="48" fill="#ffffff" fill-opacity="0.91" stroke="{color}" stroke-width="18"/>
      <circle cx="{x + 120}" cy="{y + 120}" r="92" fill="#ffffff" stroke="#2b2b2b" stroke-opacity="0.14" stroke-width="8"/>
      {logo}
      <text x="{x + 232}" y="{y + 82}" font-family="Arial, sans-serif" font-size="46" font-weight="850" fill="{color}">{power}</text>
      <text x="{x + 232}" y="{y + 148}" font-family="Arial, sans-serif" font-size="56" font-weight="800" fill="#1d1d1f">{label_1}</text>
      {second}
    </g>
    """


def write_svg_wrapper(base_png: Path, output_svg: Path, data: dict[str, Any], view_box: tuple[float, float, float, float]) -> None:
    encoded = base64.b64encode(base_png.read_bytes()).decode("ascii")
    width, height = view_box[2], view_box[3]
    badges = "\n".join(
        planning_badge_svg(power, model)
        for power, model in model_map(data).items()
        if power in BADGE_POSITIONS and model
    )
    output_svg.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}">
  <image x="0" y="0" width="{width:g}" height="{height:g}" href="data:image/png;base64,{encoded}"/>
  <g id="ModelBadgeLayer">{badges}</g>
</svg>
""",
        encoding="utf-8",
    )


def render(args: argparse.Namespace) -> None:
    data = load_game(args.game_json)
    phase = find_phase(data, args.phase)
    label_points, view_box = extract_label_points(args.map_svg)
    province_types = load_province_types()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        barrier_svg = temp_dir_path / "planning_barriers.svg"
        base_png = temp_dir_path / "planning_base.png"
        barrier_png = temp_dir_path / "planning_barriers.png"
        recolored_png = temp_dir_path / "planning_recolored.png"
        write_barrier_svg(args.map_svg, barrier_svg)
        render_svg_to_png(args.map_svg, base_png, args.width)
        render_svg_to_png(barrier_svg, barrier_png, args.width)
        image = recolor_base_map(
            Image.open(base_png),
            Image.open(barrier_png),
            phase,
            label_points,
            province_types,
            view_box,
        )
        if args.units:
            draw_units(image, phase, label_points, view_box)
        if args.header:
            draw_header(image, phase)
        image.save(recolored_png)

        if args.badges or args.output.suffix.lower() == ".svg":
            wrapper_svg = args.output if args.output.suffix.lower() == ".svg" else temp_dir_path / "planning_wrapper.svg"
            write_svg_wrapper(recolored_png, wrapper_svg, data, view_box)
            if args.output.suffix.lower() != ".svg":
                render_svg_to_png(wrapper_svg, args.output, args.width)
        else:
            image.save(args.output)

    print(f"Wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_json", type=Path)
    parser.add_argument("--phase", required=True, help="Phase name, for example W1904A.")
    parser.add_argument("--output", type=Path, required=True, help="Output .png or .svg path.")
    parser.add_argument("--map-svg", type=Path, default=PLANNING_SVG, help="Planning map SVG to use.")
    parser.add_argument("--width", type=int, default=2400, help="PNG output width.")
    parser.add_argument("--badges", action="store_true", help="Overlay model badges.")
    parser.add_argument("--units", action="store_true", help="Overlay simple unit markers.")
    parser.add_argument("--header", action="store_true", help="Overlay phase and supply-center counts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    render(args)


if __name__ == "__main__":
    main()
