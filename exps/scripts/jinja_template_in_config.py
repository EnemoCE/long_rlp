#!/usr/bin/env python3
# update_template_yaml.py
"""Utility that injects the chosen Jinja chat template into ``rpt_trainer.yaml``.

The launch script calls this before training so ``actor_rollout_ref.model``
contains the intended ``custom_chat_template``. Keeping this logic in a small
script avoids depending on ad hoc edits in the shell layer and makes the
template update reproducible.

This tool updates only the YAML template field. It does not touch dataset
preprocessing, reward wiring, or trainer behavior.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def find_default_jinja() -> Path:
    """Resolve the default Jinja template from the current working directory."""
    exact = Path("jinja/rpt_template.jinja")
    if exact.exists():
        return exact

    cands = sorted(Path(".").glob("jinja/*.jinja"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        raise FileNotFoundError(
            "Не найден Jinja-файл: custom_template.jinja или любой *.jinja в текущей директории."
        )
    raise FileNotFoundError(
        f"Найдено несколько *.jinja ({', '.join(p.name for p in cands)}). "
        "Укажи нужный через --jinja."
    )



def update_with_ruamel(yaml_path: Path, jinja_text: str) -> None:
    """Update ``custom_chat_template`` using ``ruamel.yaml`` while preserving style."""
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.scalarstring import LiteralScalarString
    except Exception as e:
        raise ImportError(
            "ruamel.yaml не установлен. Установи: pip install ruamel.yaml"
        ) from e

    if not jinja_text.endswith("\n"):
        jinja_text += "\n"

    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)

    data = y.load(read_text(yaml_path))
    if data is None:
        data = {}

    actor_rollout_ref = data.setdefault("actor_rollout_ref", {})
    model = actor_rollout_ref.setdefault("model", {})
    model["custom_chat_template"] = LiteralScalarString(jinja_text)

    with yaml_path.open("w", encoding="utf-8") as f:
        y.dump(data, f)


def detect_existing_block_indent(lines: list[str], start: int, key_indent: str) -> str:
    """
    Пытаемся понять, каким отступом был записан старый block scalar.
    Берём отступ первой непустой строки после ключа, если он глубже key_indent.
    Иначе используем key_indent + 2 пробела.
    """
    for k in range(start, len(lines)):
        ln = lines[k]
        if ln.strip() == "":
            continue
        m = re.match(r"^(\s+)\S", ln)
        if m and len(m.group(1)) > len(key_indent):
            return m.group(1)
        break
    return key_indent + "  "


def update_with_text_replace(yaml_path: Path, jinja_text: str) -> None:
    """Fallback updater that rewrites the YAML block scalar without extra deps."""
    src = read_text(yaml_path)
    lines = src.splitlines(keepends=True)

    key_re = re.compile(r"^(\s*)custom_chat_template\s*:(.*)$")
    idx = None
    key_indent = ""
    for i, ln in enumerate(lines):
        m = key_re.match(ln)
        if m:
            idx = i
            key_indent = m.group(1)
            break

    if idx is None:
        raise ValueError("Не нашла в YAML ключ `custom_chat_template:`.")

    # определяем, каким отступом был старый блок (если он вообще был)
    old_block_indent = detect_existing_block_indent(lines, idx + 1, key_indent)

    # Переписываем ключ на literal block
    lines[idx] = f"{key_indent}custom_chat_template: |\n"

    # Удаляем старое значение (если было): строки, которые относятся к блоку
    j = idx + 1
    while j < len(lines):
        ln = lines[j]

        # ВАЖНО: пустые строки считаем частью блока (часто без отступов в реальных файлах)
        if ln.strip() == "":
            j += 1
            continue

        # Непустая строка: если с отступом блока — это часть значения
        if ln.startswith(old_block_indent):
            j += 1
            continue

        # иначе блок закончился
        break

    # Вставляем новый блок с тем же отступом, что был у старого (или +2 пробела по умолчанию)
    if not jinja_text.endswith("\n"):
        jinja_text += "\n"

    new_block_indent = old_block_indent  # сохраняем стиль файла
    ins = [f"{new_block_indent}{jl}\n" for jl in jinja_text.splitlines()]
    # splitlines() съедает финальный \n -> добавим пустую строку если последняя строка была пустой
    # (в нашем случае jinja_text уже с \n, и splitlines() корректно вернёт последнюю строку без \n)
    lines[idx + 1 : j] = ins

    write_text(yaml_path, "".join(lines))


def main() -> int:
    """CLI entry point for rewriting ``config/rpt_trainer.yaml`` template text."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="config/rpt_trainer.yaml")
    ap.add_argument("--jinja", default=None)
    ap.add_argument(
        "--mode",
        choices=("strict", "text"),
        default="strict",
        help="strict: только ruamel.yaml (дефолт). text: грубый фоллбэк без зависимостей.",
    )
    ap.add_argument("--backup", action="store_true", help="Сохранить .bak перед записью")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        print(f"ERROR: YAML файл не найден: {yaml_path.resolve()}", file=sys.stderr)
        return 2

    jinja_path = Path(args.jinja) if args.jinja else find_default_jinja()
    if not jinja_path.exists():
        print(f"ERROR: Jinja файл не найден: {jinja_path.resolve()}", file=sys.stderr)
        return 2

    if args.backup:
        bak = yaml_path.with_suffix(yaml_path.suffix + ".bak")
        write_text(bak, read_text(yaml_path))

    jinja_text = read_text(jinja_path)

    try:
        if args.mode == "strict":
            update_with_ruamel(yaml_path, jinja_text)
            print(f"OK(strict): обновила {yaml_path} из {jinja_path} через ruamel.yaml")
        else:
            update_with_text_replace(yaml_path, jinja_text)
            print(f"OK(text): обновила {yaml_path} из {jinja_path} текстовой заменой")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
