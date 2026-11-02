"""Shared noise-level tagging for the RAD pipeline.

Default noise_level=0 keeps every original path and filename unchanged.
Nonzero levels append _{tag} such as _1pct, _5pct, _10pct, _25pct.
"""

from __future__ import annotations

import argparse
import os
import re


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NOISE_LEVELS = (0.01, 0.05, 0.10, 0.25)


def parse_noise_level(value) -> float:
    if value is None or value is False:
        return 0.0
    text = str(value).strip().lower().replace("%", "")
    if text in {"", "0", "0.0", "none", "clean"}:
        return 0.0
    if text.endswith("pct"):
        return float(text[:-3]) / 100.0
    number = float(text)
    if number > 1.0 + 1e-12:
        number = number / 100.0
    if number < 0:
        raise ValueError("noise_level must be >= 0, got {}".format(value))
    return float(number)


def noise_tag(noise_level) -> str:
    level = parse_noise_level(noise_level)
    if level <= 1e-15:
        return ""
    percent = level * 100.0
    rounded = int(round(percent))
    if abs(percent - rounded) < 1e-8:
        return "{}pct".format(rounded)
    compact = "{:g}".format(percent).replace(".", "p")
    return "{}pct".format(compact)


def known_noise_tags():
    tags = [noise_tag(level) for level in DEFAULT_NOISE_LEVELS]
    return tuple(tag for tag in tags if tag)


def add_noise_suffix(name: str, tag: str) -> str:
    if not tag:
        return name
    root, ext = os.path.splitext(name)
    if ext:
        return "{}_{}{}".format(root, tag, ext)
    return "{}_{}".format(name, tag)


def case_stem(case_id, tag: str = "") -> str:
    stem = "case_{:03d}".format(int(case_id))
    return add_noise_suffix(stem, tag)


def has_noise_tag(name: str, tag: str) -> bool:
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    if tag:
        return stem.endswith("_" + tag)
    return not any(stem.endswith("_" + known) for known in known_noise_tags())


def case_npz_regex(tag: str):
    if not tag:
        return re.compile(r"^case_\d{3}\.npz$")
    return re.compile(r"^case_\d{3}_" + re.escape(tag) + r"\.npz$")


def model_best_regex(tag: str):
    if not tag:
        return re.compile(r"^case_\d{3}_best\.pt$")
    return re.compile(r"^case_\d{3}_" + re.escape(tag) + r"_best\.pt$")


def add_noise_argument(parser: argparse.ArgumentParser, **kwargs):
    parser.add_argument(
        "--noise-level",
        default=kwargs.get("default", 0),
        help=(
            "Observation noise relative to std(C). 0 keeps the original clean filenames. "
            "Use 0.01/0.05/0.10/0.25 or 1pct/5pct/10pct/25pct."
        ),
    )
    return parser
