"""Check that the live Hugging Face Space serves exactly the files in space/.

    python tools/check_live_space.py OWNER/SPACE_NAME [--timeout SECONDS]

The last GitHub Actions job runs this after every deployment. A static Space is served from
https://<subdomain>.static.hf.space/, and the subdomain comes from the Hub API. The Space can take
a little while to update, so the check repeats until every file matches or the time runs out.
Only the Python standard library is used.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parents[1] / "space"
SERVED = ["index.html", "style.css", "app.js", "predictor.py", "model/model_info.json", "model/catboost_model.py"]


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "space-deploy-check", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", help="the Space ID, for example someone/credit-default-risk")
    parser.add_argument("--timeout", type=int, default=600, help="seconds to keep trying (default 600)")
    args = parser.parse_args()

    api = json.loads(download(f"https://huggingface.co/api/spaces/{args.space}"))
    host = f"https://{api['subdomain']}.static.hf.space"
    expected = {name: hashlib.sha256((SPACE_DIR / name).read_bytes()).hexdigest() for name in SERVED}
    print(f"Checking {host} (Space {args.space}, sdk {api.get('sdk')})")

    deadline = time.monotonic() + args.timeout
    while True:
        waiting = []
        for name in SERVED:
            try:
                served = download(f"{host}/{name}?check={time.time_ns()}")  # the query skips any cache
            except urllib.error.URLError as err:
                waiting.append(f"{name} ({err})")
                continue
            if hashlib.sha256(served).hexdigest() != expected[name]:
                waiting.append(name)
        if not waiting:
            print(f"OK: the live Space serves all {len(SERVED)} files of this commit, byte for byte.")
            return 0
        if time.monotonic() > deadline:
            print(f"FAILED after {args.timeout} s; still different or missing: {', '.join(waiting)}")
            return 1
        print(f"Not updated yet: {', '.join(waiting)}. Trying again in 15 s.")
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
