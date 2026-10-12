"""Loading defect datasets (CSV or ARFF) and the bundled PROMISE releases.

A defect dataset is a table with one row per software module (here, a Java class),
numeric code metrics as features, and a target saying whether the module had bugs.
In the PROMISE files the target ``bug`` is a bug count; any count > 0 means "defective".
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "promise"

TARGET_NAMES = ("bug", "bugs", "defects", "defect", "defective", "buggy", "faulty", "fault",
                "faults", "class", "label", "target", "problems", "error", "errors")
ID_NAMES = ("name", "module", "file", "filename", "class_name", "classname", "path", "id")
TRUE_WORDS = {"true", "yes", "y", "t", "1", "buggy", "defective", "bug", "faulty", "fault"}
FALSE_WORDS = {"false", "no", "n", "f", "0", "clean", "non-defective", "nondefective", "ok"}

# One release per project: the latest one whose share of defective classes lies between
# 5 % and 70 % (jedit-4.3 has 2 % defective classes, log4j-1.2 92 %, xalan-2.7 99 % and
# xerces-1.4 74 %: such releases make the classification problem degenerate).
BENCHMARK = ["ant-1.7", "camel-1.6", "ivy-2.0", "jedit-4.2", "log4j-1.1", "lucene-2.4",
             "poi-3.0", "synapse-1.2", "velocity-1.6", "xalan-2.6", "xerces-1.3"]


@dataclass
class Dataset:
    """Feature matrix, binary target and metadata of a defect dataset."""
    name: str
    X: np.ndarray                 # (n_modules, n_features), float, may contain NaN
    y: np.ndarray                 # (n_modules,), 0 = clean, 1 = defective
    features: List[str]
    modules: List[str]            # module names (or row numbers when absent)
    loc: Optional[np.ndarray] = None   # lines of code, used for effort-aware metrics
    target: str = "bug"
    notes: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def defect_rate(self) -> float:
        return float(self.y.mean()) if self.n else 0.0

    def summary(self) -> dict:
        return {"name": self.name, "modules": self.n, "features": len(self.features),
                "defective": int(self.y.sum()), "defect_rate": round(self.defect_rate, 4),
                "target": self.target, "has_loc": self.loc is not None, "notes": self.notes}


# ---------------------------------------------------------------------------- parsing
def _parse_arff(text: str):
    """Minimal ARFF reader: numeric, real, integer and nominal attributes, '?' = missing."""
    names, kinds, rows, in_data = [], [], [], False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if not in_data:
            if low.startswith("@attribute"):
                m = re.match(r"@attribute\s+('([^']*)'|\"([^\"]*)\"|(\S+))\s+(.*)$", line, re.I)
                if not m:
                    raise ValueError(f"cannot read attribute line: {line}")
                names.append(m.group(2) or m.group(3) or m.group(4))
                spec = m.group(5).strip()
                kinds.append("nominal" if spec.startswith("{") else spec.lower())
            elif low.startswith("@data"):
                in_data = True
            continue
        if line.startswith("{"):
            raise ValueError("sparse ARFF data is not supported")
        rows.append(next(csv.reader([line], skipinitialspace=True, quotechar="'")))
    if not names:
        raise ValueError("no @attribute found: is this really an ARFF file?")
    return names, rows


def _parse_csv(text: str):
    sample = text[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delim = dialect.delimiter
    except csv.Error:
        delim = ","
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = [r for r in reader if r and any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("the file needs a header line and at least one data line")
    return [c.strip() for c in rows[0]], rows[1:]


def _to_float(v: str) -> float:
    v = v.strip().strip("'\"")
    if v in ("", "?", "NA", "NaN", "nan", "null", "None"):
        return np.nan
    return float(v)


def _binarize(values: Sequence[str], column: str) -> np.ndarray:
    clean = [str(v).strip().strip("'\"").lower() for v in values]
    try:
        nums = np.array([float(v) for v in clean])
        return (nums > 0).astype(int)
    except ValueError:
        pass
    out = []
    for v in clean:
        if v in TRUE_WORDS:
            out.append(1)
        elif v in FALSE_WORDS:
            out.append(0)
        else:
            raise ValueError(f"cannot read target value '{v}' in column '{column}'")
    return np.array(out, dtype=int)


def detect_target(columns: Sequence[str]) -> str:
    low = [c.lower() for c in columns]
    for name in TARGET_NAMES:
        if name in low:
            return columns[low.index(name)]
    return columns[-1]


def from_table(columns: List[str], rows: List[List[str]], name: str = "dataset",
               target: Optional[str] = None) -> Dataset:
    """Build a Dataset from raw string cells (header + rows)."""
    width = len(columns)
    rows = [r for r in rows if len(r) == width]
    if not rows:
        raise ValueError("no complete data row (check the separator)")
    target = target or detect_target(columns)
    if target not in columns:
        raise ValueError(f"target column '{target}' not found")
    t = columns.index(target)
    y = _binarize([r[t] for r in rows], target)
    if len(np.unique(y)) < 2:
        raise ValueError("the target has a single class: there is nothing to predict")

    notes, features, cols, module_col = [], [], [], None
    for j, c in enumerate(columns):
        if j == t:
            continue
        try:
            values = np.array([_to_float(r[j]) for r in rows])
        except ValueError:
            if module_col is None and (c.lower() in ID_NAMES or j == 0):
                module_col = j
            else:
                notes.append(f"non-numeric column '{c}' ignored")
            continue
        if c.lower() in ID_NAMES and module_col is None:
            module_col = j
            continue
        if np.all(np.isnan(values)):
            notes.append(f"empty column '{c}' ignored")
            continue
        features.append(c)
        cols.append(values)
    if not features:
        raise ValueError("no numeric feature column found")
    X = np.column_stack(cols).astype(float)
    modules = ([r[module_col].strip() for r in rows] if module_col is not None
               else [f"row {i + 1}" for i in range(len(rows))])
    low = [f.lower() for f in features]
    loc = X[:, low.index("loc")].copy() if "loc" in low else None
    if np.isnan(X).any():
        notes.append(f"{int(np.isnan(X).sum())} missing values (filled with the training median)")
    return Dataset(name=name, X=X, y=y, features=features, modules=modules, loc=loc,
                   target=target, notes=notes)


def load(source: Union[str, Path, bytes], name: Optional[str] = None,
         target: Optional[str] = None) -> Dataset:
    """Load a CSV or ARFF file from a path or raw bytes."""
    if isinstance(source, (str, Path)) and not isinstance(source, bytes):
        path = Path(source)
        text = path.read_text(encoding="utf-8", errors="replace")
        name = name or path.stem
    else:
        text = source.decode("utf-8", errors="replace")
        name = name or "uploaded"
    head = "\n".join(l for l in text.lstrip().splitlines()[:50] if not l.startswith("%")).lower()
    if "@relation" in head or "@attribute" in head:
        columns, rows = _parse_arff(text)
    else:
        columns, rows = _parse_csv(text)
    return from_table(columns, rows, name=name, target=target)


# ---------------------------------------------------------------------------- PROMISE
def release_path(release: str) -> Path:
    project = release.rsplit("-", 1)[0]
    path = DATA_DIR / project / f"{release}.csv"
    if not path.exists():
        raise FileNotFoundError(f"unknown release '{release}' (see list_releases())")
    return path


def list_releases() -> List[str]:
    def key(p: Path):
        proj, ver = p.stem.rsplit("-", 1)
        return proj, [int(x) for x in ver.split(".")]
    return [p.stem for p in sorted(DATA_DIR.glob("*/*.csv"), key=key)]


def load_promise(release: str) -> Dataset:
    return load(release_path(release), name=release)


def previous_releases(release: str) -> List[str]:
    project = release.rsplit("-", 1)[0]
    ordered = [r for r in list_releases() if r.rsplit("-", 1)[0] == project]
    return ordered[:ordered.index(release)]
