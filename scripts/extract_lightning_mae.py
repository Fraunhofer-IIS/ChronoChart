"""Extract final validation errors from PyTorch Lightning TensorBoard logs.

The script reads run settings from each version's ``hparams.yaml`` (or
``hparam.yaml``), reads scalar values from the TensorBoard event file, and
prints rows sorted by K. Time is inferred from each run's dataset unless
``--seconds-per-k`` is supplied.

Example:
    poetry run python scripts/extract_lightning_mae.py \
        --log-dir /var/tmp/localization/lightning_logs \
        --output reports/mae.csv \
        --html reports/mae.html \
        --plot reports/mae.png

The log directory shown above is the default. Output paths are optional and
are written relative to the current working directory unless absolute paths
are provided.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import re
import sys
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_LOG_DIR = Path("/var/tmp/localization/lightning_logs")
DEFAULT_METRIC = "Error/Euclidean"
K_PATTERN = re.compile(r"^\s*K\s*:\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*$", re.MULTILINE)
SCALAR_PATTERN = re.compile(r"^\s*(?P<key>[A-Za-z0-9_]+):\s*(?P<value>[^#\n]+)", re.MULTILINE)


def read_k(version_dir: Path) -> int | float | None:
    """Read the top-level K value from a Lightning hyperparameter file."""
    for filename in ("hparams.yaml", "hparam.yaml"):
        path = version_dir / filename
        if not path.is_file():
            continue
        match = K_PATTERN.search(path.read_text(encoding="utf-8"))
        if match:
            value = float(match.group("value"))
            return int(value) if value.is_integer() else value
    return None


def read_hparam(version_dir: Path, key: str) -> str:
    """Read a simple scalar hyperparameter without requiring YAML unsafe tags."""
    for filename in ("hparams.yaml", "hparam.yaml"):
        path = version_dir / filename
        if not path.is_file():
            continue
        for match in SCALAR_PATTERN.finditer(path.read_text(encoding="utf-8")):
            if match.group("key") == key:
                value = match.group("value").strip().strip("\"'")
                return "" if value.lower() in {"null", "none"} else value
    return ""


def seconds_per_k_from_timestamps(timestamp_file: Path) -> float:
    """Calculate the project's implicit seconds-per-K from timestamp data."""
    import torch

    timestamps = torch.load(timestamp_file, map_location="cpu", weights_only=True)
    values = timestamps.detach().cpu().numpy().reshape(-1)
    if len(values) == 0:
        raise ValueError("timestamp file contains no values")
    span = float(values.max() - values.min())
    if span <= 0:
        raise ValueError("timestamp file has no positive time span")
    return span / len(values)


def dataset_for_run(version_dir: Path) -> tuple[str, str]:
    """Resolve a dataset name from run model_name or preset config fields."""
    model_name = read_hparam(version_dir, "model_name").strip()
    config_name = read_hparam(version_dir, "config").strip()
    aliases = {
        "dichasus": "dichasus", "config_dichasus": "dichasus",
        "5g": "5G", "config_5g": "5G", "passive": "passive",
        "config_passive": "passive",
    }
    for value in (model_name, config_name):
        dataset = aliases.get(value.lower())
        if dataset:
            return dataset, config_name or model_name
    return model_name or "unknown", config_name or model_name


def seconds_per_k_for_dataset(dataset: str, timestamp_override: Path | None = None) -> float:
    """Derive seconds per K from timestamps belonging to the selected dataset."""
    if timestamp_override is not None:
        return seconds_per_k_from_timestamps(timestamp_override)
    from src.localization.config.paths import PathConfig

    paths = PathConfig()
    if dataset == "dichasus":
        return seconds_per_k_from_timestamps(paths.pt_dir / "timestamps.pt")
    if dataset == "5G":
        import numpy as np
        import pandas as pd

        values = np.asarray(pd.read_pickle(paths.fiveg_train)[2]).reshape(-1)
        if values.size == 0:
            raise ValueError("5G training data contains no timestamps")
        span = float(values.max() - values.min())
        if span <= 0:
            raise ValueError("5G training timestamps have no positive time span")
        return span / len(values)
    raise ValueError(
        f"No timestamp source is configured for dataset {dataset!r}; "
        "use --seconds-per-k to provide the conversion."
    )


def find_event_file(version_dir: Path) -> Path | None:
    """Return the newest TensorBoard event file for a run."""
    event_files = sorted(version_dir.glob("events.out.tfevents.*"))
    return event_files[-1] if event_files else None


def final_metric(version_dir: Path, metric: str) -> dict[str, Any]:
    """Read the final metric event and its corresponding logged epoch."""
    event_file = find_event_file(version_dir)
    result: dict[str, Any] = {
        "event_file": str(event_file) if event_file else "",
        "metric": metric,
        "step": "",
        "epoch": "",
        "value": "",
        "status": "ok",
    }
    if event_file is None:
        result["status"] = "missing event file"
        return result

    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    if metric not in scalar_tags:
        result["status"] = "missing metric"
        return result

    values = accumulator.Scalars(metric)
    if not values:
        result["status"] = "empty metric"
        return result

    final = values[-1]
    result["step"] = final.step
    result["value"] = final.value

    # Validation error is logged before/at the same global step as the epoch
    # scalar. This recovers the checkpoint epoch for runs with partial logs.
    if "epoch" in scalar_tags:
        epochs = [event for event in accumulator.Scalars("epoch") if event.step <= final.step]
        if epochs:
            result["epoch"] = epochs[-1].value
    return result


def collect(
    log_dir: Path,
    metric: str,
    min_k: float,
    max_k: float,
    seconds_per_k: float | None,
    timestamp_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect and sort runs whose K is within the requested inclusive range."""
    rows: list[dict[str, Any]] = []
    timing_by_dataset: dict[str, float] = {}
    for version_dir in sorted(log_dir.glob("version_*")):
        if not version_dir.is_dir():
            continue
        k = read_k(version_dir)
        if k is None or not min_k <= k <= max_k:
            continue
        dataset, config = dataset_for_run(version_dir)
        row_seconds_per_k = seconds_per_k
        if row_seconds_per_k is None:
            if dataset not in timing_by_dataset:
                timing_by_dataset[dataset] = seconds_per_k_for_dataset(dataset, timestamp_file)
            row_seconds_per_k = timing_by_dataset[dataset]
        row: dict[str, Any] = {
            "version": version_dir.name,
            "K": k,
            "dataset": dataset,
            "config": config,
            "time_seconds": k * row_seconds_per_k,
        }
        row.update(final_metric(version_dir, metric))
        rows.append(row)
    return sorted(rows, key=lambda row: (row["K"], row["version"]))


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    """Write collected rows to a CSV file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "K", "value", "time_seconds", "version", "dataset", "config", "metric",
        "epoch", "step", "status", "event_file",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            exported = dict(row)
            exported["time_seconds"] = f"{row['time_seconds']:.3f}"
            if row["status"] == "ok":
                exported["value"] = f"{row['value']:.3f}"
            writer.writerow(exported)


REPORT_TITLE = "Comparison of ChronoChart Methods Across Different Triplet Window Sizes"


def write_html(rows: list[dict[str, Any]], output: Path, plot_path: Path | None = None) -> None:
    """Write a styled K-sorted HTML table and optionally embed its plot."""
    output.parent.mkdir(parents=True, exist_ok=True)
    valid_rows = [row for row in rows if row["status"] == "ok"]
    best_row = min(valid_rows, key=lambda row: row["value"]) if valid_rows else None
    best_mae = f"{best_row['value']:.3f}" if best_row else "—"
    best_k = str(best_row["K"]) if best_row else "—"
    dataset_label = html.escape(str(rows[0]["dataset"])) if rows else ""
    table_rows = []
    for row in rows:
        mae = f"{row['value']:.3f}" if row["status"] == "ok" else ""
        row_class = " class=\"best\"" if best_row is row else ""
        table_rows.append(
            f"<tr{row_class}>"
            f"<td>{html.escape(str(row['K']))}</td>"
            f"<td>{html.escape(mae)}</td>"
            f"<td>{row['time_seconds']:.3f}</td>"
            "</tr>"
        )
    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>""" + REPORT_TITLE + """</title>
<style>
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh; color: #172033;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
  background: linear-gradient(135deg, #eef4ff 0%, #f9fbff 48%, #eefaf7 100%);
}
.shell { max-width: 900px; margin: 0 auto; padding: 48px 22px 64px; }
.hero { margin-bottom: 28px; }
.eyebrow { margin: 0 0 8px; color: #4969c6; font-size: .78rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
h1 { margin: 0; font-size: clamp(2rem, 5vw, 3.4rem); letter-spacing: -.05em; line-height: 1; }
.subtitle { margin: 14px 0 0; color: #667085; font-size: 1rem; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 26px 0; }
.card, .panel { border: 1px solid rgba(133, 151, 190, .22); border-radius: 18px; background: rgba(255,255,255,.78); box-shadow: 0 12px 35px rgba(49, 72, 120, .08); }
.card { padding: 18px 20px; }
.label { color: #7b8497; font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.value { margin-top: 6px; color: #213d92; font-size: 1.55rem; font-weight: 800; }
.panel { overflow: hidden; padding: 8px; }
.chart { margin-top: 18px; padding: 18px; }
.chart img { display: block; width: 100%; height: auto; border-radius: 12px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: separate; border-spacing: 0; font-variant-numeric: tabular-nums; }
th, td { padding: 15px 18px; text-align: right; }
th { position: sticky; top: 0; color: #52617d; background: #f3f6fc; font-size: .78rem; letter-spacing: .09em; text-transform: uppercase; }
th:first-child { border-radius: 12px 0 0 12px; }
th:last-child { border-radius: 0 12px 12px 0; }
tbody tr { transition: background .18s ease, transform .18s ease; }
tbody tr:nth-child(even) { background: #f8faff; }
tbody tr:hover { background: #eaf0ff; }
td { border-bottom: 1px solid #edf0f6; color: #344054; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr.best { background: #e6f8ee; }
tbody tr.best td:first-child { color: #087443; font-weight: 800; }
.footer { margin-top: 16px; color: #7b8497; font-size: .82rem; text-align: right; }
@media (max-width: 620px) {
  .shell { padding: 30px 14px 44px; }
  .cards { grid-template-columns: 1fr; }
  th, td { padding: 12px 13px; }
}
</style>
</head>
<body>
<main class="shell">
<section class="hero">
<p class="eyebrow">Model evaluation</p>
<h1>""" + REPORT_TITLE + """</h1>
<p class="subtitle">""" + dataset_label + """ dataset · validation error sorted by temporal window size</p>
</section>
<section class="cards">
<div class="card"><div class="label">Runs</div><div class="value">""" + str(len(rows)) + """</div></div>
<div class="card"><div class="label">Best MAE</div><div class="value">""" + best_mae + """</div></div>
<div class="card"><div class="label">Best K</div><div class="value">""" + best_k + """</div></div>
</section>
<section class="panel">
<div class="table-wrap">
<table>
<thead><tr><th>k</th><th>mae</th><th>time</th></tr></thead>
<tbody>
""" + "\n".join(table_rows) + """
</tbody>
</table>
</div>
</section>
""" + (
    f'<section class="panel chart"><img src="data:image/png;base64,{base64.b64encode(plot_path.read_bytes()).decode("ascii")}" alt="{html.escape(REPORT_TITLE)}"></section>'
    if plot_path and plot_path.is_file() else ""
) + """
<p class="footer">Time is shown in seconds · best MAE highlighted in green</p>
</main>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print a compact, sorted result table."""
    headers = ["K", "value", "time_seconds", "version", "dataset", "config", "epoch", "status"]
    print("\t".join(headers))
    for row in rows:
        values = [
            f"{row['time_seconds']:.3f}" if header == "time_seconds" else str(row[header])
            for header in headers
        ]
        print("\t".join(values))


def plot_mae_vs_time(rows: list[dict[str, Any]], output: Path, metric: str) -> None:
    """Create a MAE-versus-time plot, labeling points with K."""
    import matplotlib.pyplot as plt

    valid_rows = [row for row in rows if row["status"] == "ok"]
    if not valid_rows:
        raise ValueError("no valid metric values available for plotting")
    figure, axis = plt.subplots(figsize=(10, 6))
    x = [row["time_seconds"] for row in valid_rows]
    y = [row["value"] for row in valid_rows]
    axis.plot(x, y, marker="o", linewidth=1.5)
    for row in valid_rows:
        axis.annotate(str(row["K"]), (row["time_seconds"], row["value"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("MAE")
    axis.set_title(REPORT_TITLE)
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--min-k", type=float, default=0)
    parser.add_argument("--max-k", type=float, default=180)
    parser.add_argument("--output", type=Path, help="Optional CSV table output path.")
    parser.add_argument("--html", type=Path, help="Optional HTML table output path.")
    parser.add_argument("--plot", type=Path, help="Optional MAE-versus-K PNG output path.")
    parser.add_argument(
        "--timestamp-file",
        type=Path,
        help="Override timestamp source for every run (normally inferred from its dataset).",
    )
    parser.add_argument(
        "--seconds-per-k",
        type=float,
        help="Override derived time conversion; useful for another dataset.",
    )
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        print(f"Log directory does not exist: {args.log_dir}", file=sys.stderr)
        return 2

    seconds_per_k = args.seconds_per_k
    if seconds_per_k is not None and seconds_per_k <= 0:
        print("seconds-per-k must be positive", file=sys.stderr)
        return 2

    try:
        rows = collect(args.log_dir, args.metric, args.min_k, args.max_k, seconds_per_k, args.timestamp_file)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"Cannot derive dataset timing: {exc}", file=sys.stderr)
        return 2
    print_table(rows)
    if args.output:
        write_csv(rows, args.output)
        print(f"\nWrote {len(rows)} rows to {args.output}", file=sys.stderr)
    if args.plot:
        plot_mae_vs_time(rows, args.plot, args.metric)
        print(f"Wrote plot to {args.plot}", file=sys.stderr)
    if args.html:
        write_html(rows, args.html, args.plot)
        print(f"Wrote HTML table to {args.html}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
