"""Prepare the project data and load the views used by the notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from source_preprocessing import SourceTables


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTICE_PATH = Path("recall_audit/data")
SOURCE_VIEW_PATH = Path("preprocessing_output")
NOTICES = "dataset_corrected.json"
NOTICE_SUMMARY = "reconciliation_summary.json"
NOTICE_LEDGER = "source_row_audit.jsonl"
LEGACY_HANDOFF = "project_input_records.json"


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    sha256: str
    records: list[dict]


@dataclass
class PreparationResult:
    source_tables: SourceTables
    notice_summary: dict | None


def _records(value, path):
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a nonempty list of recall records in {path}.")
    return value


def load_source(path=None):
    """Read the raw RECALL-MM file once and preserve the hash of that exact input."""
    if path is None:
        directories = (PROJECT_ROOT, Path.cwd(), PROJECT_ROOT / "recall_study_2026",
                       PROJECT_ROOT.parent / "recall_study_2026", Path.home() / "Downloads")
        candidates = (directory / name for directory in directories
                      for name in ("dataset.json", "dataset(1).json"))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError("Set DATA_PATH or place the original RECALL-MM dataset.json in the project root.")
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    return SourceSnapshot(path, hashlib.sha256(data).hexdigest(), _records(json.loads(data), path))


def _prepared_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Build the data with DataPreprocessing.ipynb or project_data.py.")
    return path


def load_notices(output_root=PROJECT_ROOT):
    """Return the main analysis view with one record per official recall notice."""
    path = _prepared_file(Path(output_root) / NOTICE_PATH / NOTICES)
    records = _records(json.loads(path.read_text(encoding="utf-8")), path)
    numbers = [row.get("recall_number") for row in records]
    if any(not isinstance(number, str) or not number for number in numbers) or len(set(numbers)) != len(records):
        raise ValueError(f"Expected unique official recall numbers in {path}.")
    return records


def load_notice_summary(output_root=PROJECT_ROOT):
    path = _prepared_file(Path(output_root) / NOTICE_PATH / NOTICE_SUMMARY)
    return json.loads(path.read_text(encoding="utf-8"))


def load_notice_audit(output_root=PROJECT_ROOT):
    summary = load_notice_summary(output_root)
    path = _prepared_file(Path(output_root) / NOTICE_PATH / NOTICE_LEDGER)
    with path.open(encoding="utf-8") as handle:
        ledger = [json.loads(line) for line in handle]
    return summary, ledger


def load_legacy_rows(path=None):
    """Return the source-row features used by the historical consensus experiment."""
    if path is None:
        path = PROJECT_ROOT / SOURCE_VIEW_PATH / LEGACY_HANDOFF
    path = _prepared_file(Path(path).expanduser())
    return _records(json.loads(path.read_text(encoding="utf-8")), path)


def prepare_project(source, cpsc_api=None, cpsc_listing=None, *, output_root=PROJECT_ROOT):
    """Prepare source views and, with both official snapshots, rebuild the notice dataset.

    With output_root=None, return the quality tables and optional reconciliation summary
    for preview without writing files.
    """
    from reconcile import build_notice_bundle, write_notice_bundle
    from source_preprocessing import prepare_source_rows, write_source_tables

    if (cpsc_api is None) != (cpsc_listing is None):
        raise ValueError("Supply both the CPSC API JSON and listing CSV to rebuild the notice data.")

    source_tables = prepare_source_rows(source.records)
    notice_bundle = None
    if cpsc_api is not None:
        notice_bundle = build_notice_bundle(source.records, source.sha256,
                                            Path(cpsc_api).expanduser(), Path(cpsc_listing).expanduser())

    if output_root is not None:
        output_root = Path(output_root).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        # Build every requested output before replacing any existing file.
        with TemporaryDirectory(prefix=".prepare-recalls-", dir=output_root) as temporary:
            staging = Path(temporary)
            write_source_tables(staging / SOURCE_VIEW_PATH, source_tables)
            if notice_bundle is not None:
                write_notice_bundle(staging / NOTICE_PATH, *notice_bundle)
            for file in sorted(staging.rglob("*")):
                if file.is_file():
                    target = output_root / file.relative_to(staging)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    file.replace(target)

    return PreparationResult(source_tables, notice_bundle[1] if notice_bundle is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original RECALL-MM dataset.json")
    parser.add_argument("cpsc_api", type=Path, help="CPSC historical API JSON snapshot")
    parser.add_argument("cpsc_listing", type=Path, help="CPSC recall listing CSV snapshot")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT,
                        help="Root directory for both sets of prepared outputs")
    args = parser.parse_args()

    source = load_source(args.source)
    prepared = prepare_project(source, args.cpsc_api, args.cpsc_listing,
                               output_root=args.output_root)
    count = prepared.notice_summary["corrected_announcements"]
    print(f"Source snapshot: {source.path}")
    print(f"Prepared {count:,} official notices at {args.output_root / NOTICE_PATH / NOTICES}.")
    print(f"Source-row quality and legacy views: {args.output_root / SOURCE_VIEW_PATH}")


if __name__ == "__main__":
    main()
