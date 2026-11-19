"""The files deployed to the Hugging Face Space: what is deployed, its configuration and references."""
import base64
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request

import pytest

import predictor
from conftest import REPO, SPACE

# Exactly these files are deployed: the deploy job uploads the space/ folder and nothing else.
DEPLOYED = {"README.md", "index.html", "style.css", "app.js", "predictor.py",
            "model/model_info.json", "model/catboost_model.py"}
HUB_FILE_LIMIT = 10 * 1024 * 1024  # a git push to the Hub needs Git LFS or Xet above 10 MB
COLORS = {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}  # allowed by the Hub


def read(name):
    return (SPACE / name).read_text(encoding="utf-8")


def front_matter():
    text = read("README.md")
    assert text.startswith("---\n")
    block = text.split("---\n")[1]
    return dict(line.split(": ", 1) for line in block.strip().splitlines())


def test_only_the_app_files_are_deployed():
    files = {p.relative_to(SPACE).as_posix() for p in SPACE.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    assert files == DEPLOYED


def test_files_are_small_enough_for_the_hub():
    for name in DEPLOYED:
        assert (SPACE / name).stat().st_size < HUB_FILE_LIMIT, name


def test_space_configuration():
    config = front_matter()
    assert config["sdk"] == "static"  # free for every account (Gradio and Docker Spaces need a paid plan)
    assert (SPACE / config["app_file"]).is_file()
    assert config["colorFrom"] in COLORS and config["colorTo"] in COLORS
    assert config["title"] and config["emoji"]
    assert len(config["short_description"]) <= 60


def test_page_loads_files_that_exist():
    html, js = read("index.html"), read("app.js")
    assert set(re.findall(r'(?:src|href)="(?!https?:)([^"]+)"', html)) == {"style.css", "app.js"}
    for call, name in (('fetchText("model/model_info.json"', "model/model_info.json"),
                       ('fetchText("predictor.py"', "predictor.py"),
                       ("fetchText(`model/catboost_model.py?", "model/catboost_model.py")):
        assert call in js and (SPACE / name).is_file(), name


def test_pyodide_version_is_pinned_once():
    html, js = read("index.html"), read("app.js")
    script = re.search(r'<script src="(https://cdn\.jsdelivr\.net/pyodide/v[\d.]+/full/)pyodide\.js"', html)
    index = re.search(r'const PYODIDE_INDEX = "([^"]+)";', js)
    assert script and index and script.group(1) == index.group(1)
    assert re.search(r'integrity="sha384-[A-Za-z0-9+/]+={0,2}"', html)


@pytest.mark.network
def test_pyodide_matches_its_hash_and_our_python_version():
    html = read("index.html")
    url = re.search(r'<script src="(https://[^"]+/pyodide\.js)"', html).group(1)
    integrity = re.search(r'integrity="(sha384-[^"]+)"', html).group(1)
    try:
        script = urllib.request.urlopen(url, timeout=30).read()
        lock = json.loads(urllib.request.urlopen(url.replace("pyodide.js", "pyodide-lock.json"), timeout=30).read())
    except urllib.error.URLError as err:
        pytest.skip(f"no network: {err}")
    assert "sha384-" + base64.b64encode(hashlib.sha384(script).digest()).decode() == integrity
    # The tests run on the same Python version (major.minor) as the browser does.
    assert lock["info"]["python"].split(".")[:2] == [str(v) for v in sys.version_info[:2]]


@pytest.mark.parametrize("card", [SPACE / "README.md", REPO / "README.md"])
def test_readme_numbers_match_the_model(card):
    text = card.read_text(encoding="utf-8")
    info, results = predictor.INFO, predictor.INFO["test_results"]

    def apa(x):
        return f"{x:.3f}".removeprefix("0")

    for expected in (f"{info['model']['trees']} trees", f"depth {info['model']['depth']}",
                     f"learning rate {info['model']['learning_rate']}", f"{100 * info['threshold']:.1f}%",
                     f"ROC-AUC {apa(results['roc_auc'])}", f"precision {apa(results['at_threshold']['precision'])}",
                     f"recall {apa(results['at_threshold']['recall'])}", f"F1 {apa(results['at_threshold']['f1'])}"):
        assert expected in text, f"{card.name}: {expected}"
