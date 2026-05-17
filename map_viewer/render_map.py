#!/usr/bin/env python3
"""Render board snapshots from saved AI Diplomacy game histories."""

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

from diplomacy.engine.game import Game

POWER_ORDER = ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")

LOGO_DIR = Path(__file__).resolve().parent / "assets" / "logos"
BADGE_POSITIONS = {
    "ENGLAND": (330, 575),
    "FRANCE": (445, 725),
    "GERMANY": (715, 640),
    "ITALY": (780, 925),
    "AUSTRIA": (930, 740),
    "RUSSIA": (1220, 475),
    "TURKEY": (1240, 930),
}
BADGE_COLORS = {
    "AUSTRIA": "#b75b51",
    "ENGLAND": "#4b8fc2",
    "FRANCE": "#42b7b7",
    "GERMANY": "#4b4b4b",
    "ITALY": "#63b84f",
    "RUSSIA": "#b56ebd",
    "TURKEY": "#b7a947",
}
PROVIDER_LOGOS = {
    "anthropic": "claude_symbol.svg",
    "deepseek": "deepseek_symbol.svg",
    "google/gemini": "gemini_symbol.svg",
    "google/gemma": "gemma_original.png",
    "openai": "openai_symbol.svg",
    "qwen": "qwen_symbol.svg",
    "x-ai": "grok_symbol.svg",
}


def load_game(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_phase(data: dict[str, Any], phase_name: str) -> dict[str, Any]:
    phase_name = phase_name.upper()
    for phase in data["phases"]:
        if phase["name"].upper() == phase_name:
            return phase
    available = ", ".join(phase["name"] for phase in data["phases"])
    raise SystemExit(f"Unknown phase {phase_name!r}. Available phases: {available}")


def center_counts(phase: dict[str, Any]) -> str:
    centers = phase["state"].get("centers", {})
    return " ".join(f"{power[:3]}:{len(centers.get(power, []))}" for power in POWER_ORDER)


def model_map(data: dict[str, Any]) -> dict[str, str]:
    state_agents = data.get("phases", [{}])[0].get("state_agents") or {}
    models = {
        power: agent.get("model_id", "")
        for power, agent in state_agents.items()
        if isinstance(agent, dict) and agent.get("model_id")
    }
    if models:
        return models
    return data.get("power_model_map") or {}


def short_model_name(model_id: str) -> str:
    model_id = model_id.replace("openrouter:", "")
    exact = {
        "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
        "anthropic/claude-opus-4.6": "Claude Opus 4.6",
        "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
        "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
        "google/gemma-4-31b-it": "Gemma 4 31B",
        "openai/gpt-5.4": "GPT-5.4",
        "openai/gpt-oss-120b": "GPT-OSS 120B",
        "qwen/qwen3.5-27b": "Qwen 3.5 27B",
        "qwen/qwen3.6-plus": "Qwen 3.6 Plus",
        "x-ai/grok-4.1-fast": "Grok 4.1 Fast",
    }
    if model_id in exact:
        return exact[model_id]

    if "/" in model_id:
        provider, model = model_id.split("/", 1)
    else:
        provider, model = "", model_id

    replacements = {
        "anthropic": "Claude",
        "deepseek": "DeepSeek",
        "google": "Google",
        "openai": "OpenAI",
        "qwen": "Qwen",
        "x-ai": "Grok",
    }
    prefix = replacements.get(provider, provider)
    cleaned = (
        model.replace("claude-", "")
        .replace("gemini-", "Gemini ")
        .replace("gemma-", "Gemma ")
        .replace("gpt-", "GPT-")
        .replace("qwen", "Qwen ")
        .replace("grok-", "Grok ")
        .replace("deepseek-", "DeepSeek ")
        .replace("-it", "")
        .replace("-", " ")
    )
    cleaned = " ".join(part.upper() if part in {"oss"} else part.capitalize() for part in cleaned.split())
    if prefix and not cleaned.lower().startswith(prefix.lower()):
        return f"{prefix} {cleaned}".strip()
    return cleaned.strip()


def logo_path_for_model(model_id: str) -> Path | None:
    normalized = model_id.replace("openrouter:", "")
    key = None
    if normalized.startswith("google/gemini"):
        key = "google/gemini"
    elif normalized.startswith("google/gemma"):
        key = "google/gemma"
    else:
        key = normalized.split("/", 1)[0]
    file_name = PROVIDER_LOGOS.get(key)
    if not file_name:
        return None
    path = LOGO_DIR / file_name
    return path if path.exists() else None


def data_uri(path: Path) -> str:
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/svg+xml"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def split_label(label: str) -> tuple[str, str]:
    if len(label) <= 16:
        return label, ""
    words = label.split()
    first, second = [], []
    for word in words:
        target = first if len(" ".join(first + [word])) <= 16 else second
        target.append(word)
    return " ".join(first), " ".join(second)


def badge_svg(power: str, model_id: str) -> str:
    x, y = BADGE_POSITIONS[power]
    color = BADGE_COLORS[power]
    label_1, label_2 = split_label(short_model_name(model_id))
    logo_path = logo_path_for_model(model_id)
    logo = ""
    if logo_path:
        logo = f'<image x="{x + 12}" y="{y + 12}" width="36" height="36" href="{data_uri(logo_path)}"/>'
    else:
        logo = (
            f'<text x="{x + 30}" y="{y + 37}" text-anchor="middle" '
            f'font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700">{escape(power[:2])}</text>'
        )

    second_line = ""
    if label_2:
        second_line = (
            f'<text x="{x + 58}" y="{y + 48}" font-family="Inter, Arial, sans-serif" '
            f'font-size="12" font-weight="650" fill="#1d1d1f">{escape(label_2)}</text>'
        )

    return f"""
    <g class="model-badge" id="ModelBadge-{power}">
      <rect x="{x}" y="{y}" width="178" height="60" rx="12" fill="#ffffff" fill-opacity="0.90" stroke="{color}" stroke-width="4"/>
      <circle cx="{x + 30}" cy="{y + 30}" r="23" fill="#ffffff" stroke="#2b2b2b" stroke-opacity="0.14" stroke-width="2"/>
      {logo}
      <text x="{x + 58}" y="{y + 22}" font-family="Inter, Arial, sans-serif" font-size="11" font-weight="850" fill="{color}">{power}</text>
      <text x="{x + 58}" y="{y + 37}" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="750" fill="#1d1d1f">{escape(label_1)}</text>
      {second_line}
    </g>
    """


def add_model_badges(svg: str, models: dict[str, str]) -> str:
    badges = "\n".join(
        badge_svg(power, models[power])
        for power in POWER_ORDER
        if power in models and models[power]
    )
    if not badges:
        return svg
    return svg.replace("</svg>", f'<g id="ModelBadgeLayer">\n{badges}\n</g>\n</svg>')


def list_phases(args: argparse.Namespace) -> None:
    data = load_game(args.game_json)
    print("phase\t" + "\t".join(power[:3] for power in POWER_ORDER))
    for phase in data["phases"]:
        centers = phase["state"].get("centers", {})
        counts = [str(len(centers.get(power, []))) for power in POWER_ORDER]
        print(f"{phase['name']}\t" + "\t".join(counts))


def render_phase(args: argparse.Namespace) -> None:
    data = load_game(args.game_json)
    phase = find_phase(data, args.phase)

    game = Game(map_name=data.get("map", "standard"), rules=data.get("rules", []))
    game.set_state(phase["state"])
    game.note = f"{phase['name']}  {center_counts(phase)}"

    if args.orders:
        for power, orders in phase.get("orders", {}).items():
            game.set_orders(power, orders or [])

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = game.render(incl_orders=args.orders, incl_abbrev=args.abbrev)
    if args.badges:
        rendered = add_model_badges(rendered, model_map(data))
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output_path}")

    if args.png:
        converter = shutil.which("rsvg-convert")
        if not converter:
            raise SystemExit("Cannot create PNG: rsvg-convert is not installed or not on PATH.")
        png_path = output_path.with_suffix(".png")
        subprocess.run(
            [converter, "-w", str(args.width), "-o", str(png_path), str(output_path)],
            check=True,
        )
        print(f"Wrote {png_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List phases and supply-center counts.")
    list_parser.add_argument("game_json", type=Path)
    list_parser.set_defaults(func=list_phases)

    render_parser = subparsers.add_parser("render", help="Render one phase to SVG.")
    render_parser.add_argument("game_json", type=Path)
    render_parser.add_argument("--phase", required=True, help="Phase name, for example W1905A.")
    render_parser.add_argument("--output", type=Path, required=True, help="SVG output path.")
    render_parser.add_argument("--orders", action="store_true", help="Draw orders for the selected phase.")
    render_parser.add_argument("--abbrev", action="store_true", help="Show province abbreviations.")
    render_parser.add_argument("--badges", action="store_true", help="Overlay power/model logo badges.")
    render_parser.add_argument("--png", action="store_true", help="Also write a PNG next to the SVG.")
    render_parser.add_argument("--width", type=int, default=1800, help="PNG width in pixels.")
    render_parser.set_defaults(func=render_phase)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
