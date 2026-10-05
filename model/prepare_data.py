"""Download a slice of TinyStories.

TinyStories (Eldan & Li, 2023) is a set of short stories written in the
vocabulary of a young child. Because the language is so simple, a model with
about a million parameters can learn to continue it sensibly.

We take the first TRAIN_BYTES of the training file with an HTTP range request
and the whole (small) validation file.
"""

from pathlib import Path

import requests

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN_URL = f"{BASE}/TinyStoriesV2-GPT4-train.txt"
VALID_URL = f"{BASE}/TinyStoriesV2-GPT4-valid.txt"
TRAIN_BYTES = 200 * 1024 * 1024
EOT = "<|endoftext|>"

DATA = Path(__file__).parent / "data"


def download(url: str, dest: Path, max_bytes: int | None = None) -> None:
    if dest.exists():
        print(f"{dest.name}: already downloaded")
        return
    headers = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {}
    tmp = dest.with_suffix(".part")
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    print(f"{dest.name}: {dest.stat().st_size / 1e6:.0f} MB")


def trim_to_whole_stories(path: Path) -> None:
    """A byte slice ends mid-story; cut back to the last complete one."""
    raw = path.read_bytes()
    end = raw.rfind(EOT.encode())
    if end != -1 and end + len(EOT) != len(raw):
        path.write_bytes(raw[: end + len(EOT)])


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    download(TRAIN_URL, DATA / "train.txt", TRAIN_BYTES)
    trim_to_whole_stories(DATA / "train.txt")
    download(VALID_URL, DATA / "valid.txt")
