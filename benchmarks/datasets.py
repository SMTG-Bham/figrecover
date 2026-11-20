"""Pinned public chart images, axis annotations and published reference scores."""

import hashlib
import json
import shutil
import zipfile
from urllib.request import urlopen

import pyarrow.parquet as pq

HUB = "https://huggingface.co/datasets"
CHARTX = f"{HUB}/InternScience/ChartX/resolve/31d1c7bd971a5209a45988fb6d706e1767176b6a"
PLOTQA = f"{HUB}/achang/plot_qa/resolve/e9da2cdb45c671f3faf63449e6c329f0c3281aa0"
REFERENCE = (
    f"{HUB}/tommycarstensen/plotpick-validation/resolve/"
    "21de08a9a704b0c6daa084c6f6ca16e8a4d9600b"
)
FILES = {
    "chartx.zip": (
        f"{CHARTX}/ChartX_png.zip",
        "dd7ff9533c32ea48ada51875ffcc906f957b9ed53b6fb02979ee25f3cc8ad9aa",
    ),
    "plotqa.parquet": (
        f"{PLOTQA}/data/test-00000-of-00003-bce20d6f9e9401f1.parquet",
        "24b3af6f4792cb3dc86fab00107326e2174af7ff54b30f0f92d0b4d4815f2920",
    ),
    "plotqa-axes.json": (
        "https://drive.usercontent.google.com/download?id=1ikiPqkDgxNilYsU5hbK03T4_x2eDmopP&export=download&confirm=t",
        "2c6a0f22b53eb08d3df606ce4b2aacecb8fc0106abc43611c9ca5dbaedda9df4",
    ),
    "chartx-results.jsonl": (
        f"{REFERENCE}/chartx/validation.jsonl",
        "f40fde03090a337af46101f83f3c1727ff8ac03576f7e66b528157f3b22faefa",
    ),
    "plotqa-results.jsonl": (
        f"{REFERENCE}/plotqa/test.jsonl",
        "3cf46cbdbe96c2e261b64744a911330c799f93fa4e40c139bf218c299b94ee79",
    ),
}


def fetch(cache):
    """Verify every input, including the original PlotQA tick annotations."""
    cache.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in FILES.items():
        target = cache / name
        temp = target.with_suffix(".download")
        source = target
        if not target.exists():
            with urlopen(url, timeout=120) as response, temp.open("wb") as out:  # noqa: S310
                shutil.copyfileobj(response, out)
            source = temp
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Unexpected SHA-256 for {source}")
        if source != target:
            temp.replace(target)


def reference(cache, dataset):
    """Read all paired results on chart types supported by figrecover."""
    with (cache / f"{dataset}-results.jsonl").open() as stream:
        rows = map(json.loads, stream)
        if dataset == "chartx":
            return [
                r for r in rows if r["chart_type"] in {"line_chart", "line_chart_num"}
            ]
        return [r for r in rows if r["chart_kind"] == "line or dot-line"]


def table(text):
    """Parse the tabular ChartX reference with either literal or escaped tabs."""
    return [
        [v.strip() for v in line.split("\t")]
        for line in text.replace(r"\n", "\n").replace(r"\t", "\t").splitlines()
        if line.strip()
    ]


def midpoint(box, coord):
    """Centre of an annotation along one image axis."""
    return box[coord] + box[{"x": "w", "y": "h"}[coord]] / 2


def plotqa_case(record, annotation):
    """Build calibration from ticks and marker keys from the legend."""
    models = annotation["models"]
    if (
        models[0]["y"] != record["truth_values"]
        or models[0]["name"] != record["series_name"]
    ):
        raise ValueError(f"Reference series differs for {record['item_id']}")
    info = annotation["general_figure_info"]
    height = info["figure_info"]["bbox"]["bbox"]["h"]
    axes = {}
    for axis in ("x", "y"):
        ticks = info[f"{axis}_axis"]["major_ticks"]
        axes[axis] = {
            value: midpoint(box, axis) if axis == "x" else height - midpoint(box, axis)
            for value, box in zip(ticks["values"], ticks["bboxes"], strict=True)
        }
    x, y = axes["x"], axes["y"]
    left, right, low, high = min(x), max(x), min(y), max(y)
    positions = [(left, low), (right, low), (left, high)]
    hints = [arg for a, b in positions for arg in ("-p", f"{a},{b}")]
    hints += [arg for a, b in positions for arg in ("-l", f"{x[a]},{y[b]}")]
    xticks = info["x_axis"]
    labels = dict(
        zip(
            xticks["major_ticks"]["values"],
            xticks["major_labels"]["values"],
            strict=True,
        )
    )
    keys = [item["preview"]["bbox"] for item in info["legend"]["items"]]
    markers = [
        arg
        for box in keys
        for arg in ("-m", f"{midpoint(box, 'x')},{height - midpoint(box, 'y')}")
    ]
    return {
        "id": record["item_id"],
        "kind": "PlotQA dot-line",
        "n": len(models),
        "labels": [labels[v] for v in models[0]["x"]],
        "ordinal": models[0]["x"],
        "ys": [m["y"] for m in models],
        "inks": [m["color"] for m in models],
        "span": high - low,
        "truth": record["truth_values"],
        "hints": hints,
        "markers": markers,
        "size": [info["figure_info"]["bbox"]["bbox"]["w"], height],
    }


def cases(cache):
    """Extract paired charts and match their values to the reference."""
    fetch(cache)
    charts = {r["item_id"]: r for r in reference(cache, "chartx")}
    images = cache / "images"
    images.mkdir(exist_ok=True)
    cases = []
    with zipfile.ZipFile(cache / "chartx.zip") as archive:
        for name, r in charts.items():
            (images / f"{name}.png").write_bytes(
                archive.read("ChartX_png/" + r["chartx_image"])
            )
            header, *rows = table(r["ground_truth_csv"])
            columns = list(zip(*rows, strict=True))
            cases.append(
                {
                    "id": name,
                    "kind": "ChartX " + r["chart_type"],
                    "n": len(header) - 1,
                    "labels": list(columns[0]),
                    "ys": [list(col) for col in columns[1:]],
                    "truth": r["truth_numbers"],
                }
            )
    plots = {r["plotqa_test_index"]: r for r in reference(cache, "plotqa")}
    # The parquet shard orders PNG filenames lexically (0, 1, 10, 100, ...).
    annotations = sorted(
        json.loads((cache / "plotqa-axes.json").read_bytes()),
        key=lambda row: str(row["image_index"]),
    )
    offset = 0
    for batch in pq.ParquetFile(cache / "plotqa.parquet").iter_batches(
        batch_size=32, columns=["image"]
    ):
        for i, row in enumerate(batch.to_pylist(), offset):
            if i in plots:
                case = plotqa_case(plots[i], annotations[i])
                data = row["image"]["bytes"]
                size = [int.from_bytes(data[i : i + 4], "big") for i in (16, 20)]
                if size != case["size"]:
                    raise ValueError(f"Annotation size differs for {case['id']}")
                (images / f"{case['id']}.png").write_bytes(data)
                cases.append(case)
        offset += len(batch)
        if offset > max(plots):
            break
    return cases
