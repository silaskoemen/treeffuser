from __future__ import annotations

import argparse
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_path = Path(args.output) if args.output else default_output_path(config_path)
    from benchmarks.harness import run_benchmark

    run_benchmark(config=config, output_path=output_path)
    print(f"Wrote benchmark results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Treeffuser development benchmarks.")
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML config.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_yaml(path)

    with path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level.")
    return config


def default_output_path(config_path: Path) -> Path:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("benchmarks") / "results" / "raw" / f"{config_path.stem}_{timestamp}.csv"


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    lines = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if line:
            lines.append((len(line) - len(line.lstrip(" ")), line.strip()))

    if not lines:
        return {}

    def parse_block(index: int, indent: int):
        if index >= len(lines):
            return {}, index
        if lines[index][1].startswith("- "):
            return parse_list(index, indent)
        return parse_dict(index, indent)

    def parse_dict(index: int, indent: int):
        result = {}
        while index < len(lines):
            current_indent, text = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation in {path}: {text}")
            if text.startswith("- "):
                break

            key, sep, value_text = text.partition(":")
            if sep == "":
                raise ValueError(f"Expected key/value pair in {path}: {text}")
            index += 1
            value_text = value_text.strip()
            if value_text:
                result[key] = parse_scalar(value_text)
            else:
                result[key], index = parse_block(index, indent + 2)
        return result, index

    def parse_list(index: int, indent: int):
        result = []
        while index < len(lines):
            current_indent, text = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation in {path}: {text}")
            if not text.startswith("- "):
                break

            item_text = text[2:].strip()
            index += 1
            if not item_text:
                item, index = parse_block(index, indent + 2)
            elif ":" in item_text:
                key, _, value_text = item_text.partition(":")
                item = {key: parse_scalar(value_text.strip())}
                if index < len(lines) and lines[index][0] > indent:
                    continuation, index = parse_block(index, indent + 2)
                    if not isinstance(continuation, dict):
                        raise ValueError(f"Expected mapping continuation in {path}: {text}")
                    item.update(continuation)
            else:
                item = parse_scalar(item_text)
            result.append(item)
        return result, index

    def parse_scalar(text: str):
        if text in {"true", "True"}:
            return True
        if text in {"false", "False"}:
            return False
        if text in {"null", "None", "~"}:
            return None
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(part.strip()) for part in inner.split(",")]
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
        return text.strip("\"'")

    config, next_index = parse_block(0, lines[0][0])
    if next_index != len(lines):
        raise ValueError(f"Could not parse all of {path}.")
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level.")
    return config


if __name__ == "__main__":
    main()
