"""Credit card default risk: checks the 23 inputs, cleans them exactly as before training,
and scores them with the tuned CatBoost model from Module 8.

Plain Python with no third-party packages, so the same file runs in two places:
the web page runs it in the visitor's browser (Pyodide), and the GitHub Actions
tests run it with ordinary Python before every deployment.
"""
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
INFO = json.loads((HERE / "model" / "model_info.json").read_text(encoding="utf-8"))

# The Module 8 model, exported by CatBoost as plain Python code (see tools/export_model.py).
_spec = importlib.util.spec_from_file_location("catboost_model", HERE / "model" / "catboost_model.py")
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)

FEATURES = INFO["features"]                 # the 23 inputs, in the model's column order
NAMES = [f["name"] for f in FEATURES]
CATEGORY_NAMES = [f["name"] for f in FEATURES if f["kind"] == "category"]  # SEX, EDUCATION, MARRIAGE
NUMBER_NAMES = [n for n in NAMES if n not in CATEGORY_NAMES]                # the other 20 inputs
THRESHOLD = INFO["threshold"]  # a risk at or above this is flagged (the F1-best threshold from Module 8)


class InputError(ValueError):
    """One or more inputs are missing or invalid; `problems` lists every one of them."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = problems


def _show(x):
    """1000.0 -> '1,000'; 2.5 -> '2.5'."""
    return f"{int(x):,}" if x == int(x) else f"{x:,}"


def check(record):
    """Validate and clean one client. Returns (values, notes).

    values: the 23 inputs as numbers, after the same cleaning as in training.
    notes:  messages about cleaned codes or values outside the training range.
    Raises InputError if anything is missing or invalid.
    """
    values, notes, problems = {}, [], []
    unknown = sorted(set(record) - set(NAMES))
    if unknown:
        problems.append(f"unknown input name(s): {', '.join(unknown)}")
    for f in FEATURES:
        name, label = f["name"], f["label"]
        raw = record.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            problems.append(f"{label}: missing")
            continue
        try:
            if isinstance(raw, bool):
                raise ValueError
            x = float(raw)
        except (TypeError, ValueError):
            problems.append(f"{label}: '{raw}' is not a number")
            continue
        if not math.isfinite(x):
            problems.append(f"{label}: must be a finite number")
            continue
        if f.get("whole") and x != int(x):
            problems.append(f"{label}: must be a whole number")
            continue

        if "codes" in f:  # categories and repayment status take a fixed set of codes
            merged = f.get("merged_codes", {})
            allowed = sorted({int(c) for c in f["codes"]} | {int(c) for c in merged})
            if int(x) not in allowed:
                problems.append(f"{label}: {_show(x)} is not a valid code ({', '.join(map(str, allowed))})")
                continue
            # DATA CLEANING - the same fixed rule as before training: undocumented codes -> "others".
            if str(int(x)) in merged:
                new = merged[str(int(x))]
                notes.append(f"{label}: undocumented code {int(x)} was treated as {new} "
                             f"({f['codes'][str(new)]}), the same cleaning as before training.")
                x = float(new)
            if "training_codes" in f and int(x) not in f["training_codes"]:
                notes.append(f"{label}: code {int(x)} never occurs for this month in the training data, "
                             "so this estimate is less certain.")
        else:
            low, high = f["valid"].get("min"), f["valid"].get("max")
            if low is not None and x < low:
                problems.append(f"{label}: must be at least {_show(low)}")
                continue
            if high is not None and x > high:
                problems.append(f"{label}: must be at most {_show(high)}")
                continue
            seen_low, seen_high = f["training_range"]
            if not seen_low <= x <= seen_high:
                notes.append(f"{label}: {_show(x)} is outside the range seen in training "
                             f"({_show(seen_low)} to {_show(seen_high)}), so this estimate is less certain.")
        values[name] = x
    if problems:
        raise InputError(problems)
    return values, notes


def score(values):
    """Risk of default next month (0 to 1) for 23 cleaned values."""
    numbers = [values[n] for n in NUMBER_NAMES]
    categories = [str(int(values[n])) for n in CATEGORY_NAMES]  # CatBoost reads category codes as text
    return 1.0 / (1.0 + math.exp(-_model.apply_catboost_model(numbers, categories)))


def predict(record):
    """Check, clean and score one client: {'risk', 'flagged', 'threshold', 'notes'}."""
    values, notes = check(record)
    risk = score(values)
    return {"risk": risk, "flagged": risk >= THRESHOLD, "threshold": THRESHOLD, "notes": notes}


def predict_json(text):
    """JSON text in, JSON text out: the bridge that the web page calls through Pyodide."""
    try:
        return json.dumps({"ok": True, **predict(json.loads(text))})
    except InputError as err:
        return json.dumps({"ok": False, "problems": err.problems})
