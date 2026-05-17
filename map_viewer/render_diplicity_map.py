#!/usr/bin/env python3
"""Render saved game phases with the Diplicity React classical map data."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

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

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "diplicity"
DEFAULT_MAP_JSON = ASSET_DIR / "classical-map.json"
DEFAULT_VARIANT_JSON = ASSET_DIR / "classical-variant.json"

POWER_FILL = {
    "AUSTRIA": "#c44f42",
    "ENGLAND": "#4b92c8",
    "FRANCE": "#53c6d0",
    "GERMANY": "#55585c",
    "ITALY": "#54ad4c",
    "RUSSIA": "#b56ebd",
    "TURKEY": "#f2d23b",
}

LOCATION_ALIASES = {
    "LYO": "GOL",
    "MAO": "MID",
    "NAO": "NAT",
    "NWG": "NRG",
}

BADGE_POSITIONS = {
    "ENGLAND": (250, 500),
    "FRANCE": (360, 760),
    "GERMANY": (640, 560),
    "ITALY": (650, 1020),
    "AUSTRIA": (875, 860),
    "RUSSIA": (1075, 450),
    "TURKEY": (1080, 1030),
}


def view_box(width: float, height: float, aspect: str | None) -> tuple[float, float, float, float]:
    if not aspect:
        return 0, 0, width, height

    parts = aspect.split(":", 1)
    if len(parts) != 2:
        raise SystemExit(f"Invalid --aspect {aspect!r}; expected WIDTH:HEIGHT, for example 16:9.")
    try:
        target = float(parts[0]) / float(parts[1])
    except ValueError as exc:
        raise SystemExit(f"Invalid --aspect {aspect!r}; expected numeric WIDTH:HEIGHT.") from exc
    if target <= 0:
        raise SystemExit(f"Invalid --aspect {aspect!r}; aspect ratio must be positive.")

    current = width / height
    if current < target:
        adjusted_width = height * target
        return -(adjusted_width - width) / 2, 0, adjusted_width, height

    adjusted_height = width / target
    return 0, -(adjusted_height - height) / 2, width, adjusted_height


def fmt_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dip_id(location: str) -> str:
    normalized = location.upper().replace("_", "/")
    normalized = LOCATION_ALIASES.get(normalized, normalized)
    return normalized.lower()


def owner_by_location(state: dict[str, Any]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for source in ("influence", "centers"):
        for power, locations in state.get(source, {}).items():
            for location in locations:
                owners[dip_id(location)] = power
    return owners


def rgba(hex_color: str, opacity: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {opacity})"


def style_attr(styles: dict[str, Any] | None, *, skip: set[str] | None = None) -> str:
    if not styles:
        return ""
    skip = skip or set()
    pairs = []
    names = {
        "fill": "fill",
        "stroke": "stroke",
        "strokeWidth": "stroke-width",
        "fillOpacity": "fill-opacity",
        "strokeDasharray": "stroke-dasharray",
        "filter": "filter",
        "fontSize": "font-size",
        "fontFamily": "font-family",
        "fontWeight": "font-weight",
        "fontStyle": "font-style",
        "letterSpacing": "letter-spacing",
    }
    for key, value in styles.items():
        if key in skip or value is None or key not in names:
            continue
        pairs.append(f'{names[key]}="{escape(str(value))}"')
    return " ".join(pairs)


def province_types(variant: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {province["id"]: province for province in variant.get("provinces", [])}


def province_owner(province: dict[str, Any], owners: dict[str, str], types: dict[str, dict[str, Any]]) -> str | None:
    province_id = province["id"]
    parent = types.get(province_id, {}).get("parent")
    return owners.get(province_id) or (owners.get(parent) if parent else None)


def draw_text(text: dict[str, Any]) -> str:
    attrs = style_attr(text.get("styles"))
    transform = f' transform="{escape(text["transform"])}"' if text.get("transform") else ""
    tspans = text.get("tspans") or [{"value": text.get("value", ""), "x": text["point"]["x"], "y": text["point"]["y"]}]
    body = "".join(
        f'<tspan x="{span["x"]}" y="{span["y"]}">{escape(str(span.get("value", "")))}</tspan>'
        for span in tspans
    )
    return f"<text {attrs}{transform}>{body}</text>"


def draw_unit(unit: str, power: str, provinces: dict[str, dict[str, Any]]) -> str:
    pieces = unit.split()
    if len(pieces) < 2:
        return ""
    unit_type = pieces[0]
    location = dip_id(pieces[1])
    province = provinces.get(location) or provinces.get(location.split("/", 1)[0])
    if not province:
        return ""
    center = province["center"]
    x, y = center["x"] - 10, center["y"] - 10
    fill = POWER_FILL[power]
    label = "A" if unit_type == "A" else "F"
    return f"""
    <g class="unit unit-{power.lower()}">
      <circle cx="{x:g}" cy="{y:g}" r="12" fill="{fill}" stroke="#111" stroke-width="2.2"/>
      <text x="{x:g}" y="{y + 5:g}" font-family="Arial, sans-serif" font-size="15" font-weight="800" text-anchor="middle" fill="#111">{label}</text>
    </g>
    """


def badge_svg(power: str, model_id: str, dx: int = 0, dy: int = 0) -> str:
    x, y = BADGE_POSITIONS[power]
    x += dx
    y += dy
    color = BADGE_COLORS[power]
    label_1, label_2 = split_label(short_model_name(model_id))
    logo_path = logo_path_for_model(model_id)
    logo = ""
    if logo_path:
        logo = f'<image x="{x + 11}" y="{y + 15}" width="44" height="44" preserveAspectRatio="xMidYMid meet" href="{data_uri(logo_path)}"/>'
    second = ""
    if label_2:
        second = f'<text x="{x + 68}" y="{y + 53}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#1d1d1f">{escape(label_2)}</text>'
    return f"""
    <g class="model-badge" id="ModelBadge-{power}">
      <rect x="{x}" y="{y}" width="210" height="74" rx="13" fill="#ffffff" fill-opacity="0.92" stroke="{color}" stroke-width="4"/>
      <circle cx="{x + 33}" cy="{y + 37}" r="27" fill="#ffffff" stroke="#2b2b2b" stroke-opacity="0.15" stroke-width="2"/>
      {logo}
      <text x="{x + 68}" y="{y + 24}" font-family="Arial, sans-serif" font-size="13" font-weight="850" fill="{color}">{power}</text>
      <text x="{x + 68}" y="{y + 41}" font-family="Arial, sans-serif" font-size="15" font-weight="800" fill="#1d1d1f">{escape(label_1)}</text>
      {second}
    </g>
    """


def render_svg(
    data: dict[str, Any],
    phase: dict[str, Any],
    map_data: dict[str, Any],
    variant: dict[str, Any],
    include_badges: bool,
    include_units: bool,
    include_header: bool,
    badge_dx: int,
    badge_dy: int,
    aspect: str | None,
) -> str:
    types = province_types(variant)
    owners = owner_by_location(phase["state"])
    provinces_by_id = {province["id"]: province for province in map_data["provinces"]}

    min_x, min_y, box_width, box_height = view_box(map_data["width"], map_data["height"], aspect)
    view_box_text = " ".join(fmt_number(value) for value in (min_x, min_y, box_width, box_height))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box_text}">',
        "<defs>",
        map_data.get("svgDefs", ""),
        """
        <pattern patternTransform="rotate(35)" height="16" width="16" patternUnits="userSpaceOnUse" id="impassableStripes">
          <line stroke-width="18" stroke-opacity="0.1" stroke="#000000" y2="16" x2="0" y1="0" x1="0"/>
        </pattern>
        """,
        "</defs>",
        '<rect width="100%" height="100%" fill="#f7f3ea"/>',
    ]

    for element in map_data.get("backgroundElements", []):
        parts.append(f'<path d="{element["d"]}" {style_attr(element.get("styles"))}/>')

    for province in map_data["provinces"]:
        province_type = types.get(province["id"], {}).get("type")
        owner = province_owner(province, owners, types)
        fill = "transparent"
        if owner and province_type != "sea":
            fill = rgba(POWER_FILL[owner], 0.72)
        parts.append(
            f'<path id="{province["id"]}" d="{province["path"]["d"]}" fill="{fill}" '
            f'stroke="none" stroke-width="1"/>'
        )
        if province.get("supplyCenter"):
            center = province["supplyCenter"]
            center_owner = province_owner(province, {k: v for k, v in owner_by_location({"centers": phase["state"].get("centers", {})}).items()}, types)
            stroke = POWER_FILL[center_owner] if center_owner else "#111"
            parts.append(
                f'<circle cx="{center["x"]}" cy="{center["y"]}" r="7" fill="#fff" '
                f'stroke="{stroke}" stroke-width="3" opacity="0.95"/>'
            )

    for province in map_data["provinces"]:
        for text in province.get("text", []) or []:
            parts.append(draw_text(text))

    for element in map_data.get("borders", []):
        parts.append(f'<path d="{element["d"]}" {style_attr(element.get("styles"))}/>')

    for element in map_data.get("impassableProvinces", []):
        parts.append(
            f'<path d="{element["d"]}" fill="url(#impassableStripes)" stroke="#111" stroke-width="1"/>'
        )

    if include_units:
        for power, units in phase["state"].get("units", {}).items():
            for unit in units:
                parts.append(draw_unit(unit, power, provinces_by_id))

    if include_header:
        parts.append(
            f'<text x="22" y="38" font-family="Arial, sans-serif" font-size="28" font-weight="850" fill="#111">{phase["name"]}</text>'
        )
        parts.append(
            f'<text x="22" y="62" font-family="Arial, sans-serif" font-size="16" font-weight="750" fill="#111">{escape(center_counts(phase))}</text>'
        )

    if include_badges:
        models = model_map(data)
        parts.append('<g id="ModelBadgeLayer">')
        for power in POWER_ORDER:
            model = models.get(power)
            if model:
                parts.append(badge_svg(power, model, badge_dx, badge_dy))
        parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def render(args: argparse.Namespace) -> None:
    data = load_game(args.game_json)
    phase = find_phase(data, args.phase)
    map_data = load_json(args.map_json)
    variant = load_json(args.variant_json)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg = render_svg(
        data,
        phase,
        map_data,
        variant,
        args.badges,
        args.units,
        args.header,
        args.badge_dx,
        args.badge_dy,
        args.aspect,
    )
    svg_path = args.output if args.output.suffix.lower() == ".svg" else args.output.with_suffix(".svg")
    svg_path.write_text(svg, encoding="utf-8")

    if args.output.suffix.lower() == ".png":
        converter = shutil.which("rsvg-convert")
        if not converter:
            raise SystemExit("Cannot create PNG: rsvg-convert is not installed or not on PATH.")
        command = [converter, "-w", str(args.width)]
        if args.height:
            command.extend(["-h", str(args.height)])
        command.extend(["-o", str(args.output), str(svg_path)])
        subprocess.run(command, check=True)
        if not args.keep_svg:
            svg_path.unlink()

    print(f"Wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_json", type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-json", type=Path, default=DEFAULT_MAP_JSON)
    parser.add_argument("--variant-json", type=Path, default=DEFAULT_VARIANT_JSON)
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, help="PNG output height. Useful with --aspect 16:9.")
    parser.add_argument("--aspect", help="SVG viewBox aspect ratio, for example 16:9. Pads or crops without stretching.")
    parser.add_argument("--badges", action="store_true")
    parser.add_argument("--badge-dx", type=int, default=-35, help="Global badge x offset in map units.")
    parser.add_argument("--badge-dy", type=int, default=35, help="Global badge y offset in map units.")
    parser.add_argument("--units", action="store_true")
    parser.add_argument("--header", action="store_true")
    parser.add_argument("--keep-svg", action="store_true")
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
