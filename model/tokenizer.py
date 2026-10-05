"""Train a byte-level BPE vocabulary and turn the text into token IDs.

Byte-pair encoding starts with the 256 possible bytes as its vocabulary and
then repeatedly merges the most frequent adjacent pair into a new token until
the vocabulary reaches VOCAB_SIZE. Common words end up as one token; rare
words are spelled out of smaller pieces. This is the same scheme GPT-2 uses,
with a much smaller vocabulary.
"""

import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 4096
EOT = "<|endoftext|>"

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "—": "-", "–": "-"})


def clean(text: str) -> list[str]:
    """Split into stories and tidy them, so typed text matches trained text."""
    stories = []
    for story in text.split(EOT):
        story = story.translate(_QUOTES)
        story = re.sub(r"[ \t]+\n", "\n", story)
        story = re.sub(r"\n{2,}", "\n", story)
        story = re.sub(r" {2,}", " ", story).strip()
        if story:
            stories.append(story)
    return stories


def train_tokenizer(stories: list[str]) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(stories, trainer)
    return tok


def encode_stories(tok: Tokenizer, stories: list[str]) -> np.ndarray:
    """Every story is followed by the end-of-text token."""
    eot = tok.token_to_id(EOT)
    ids: list[int] = []
    for enc in tok.encode_batch(stories):
        ids.extend(enc.ids)
        ids.append(eot)
    return np.array(ids, dtype=np.uint16)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    train_stories = clean((DATA / "train.txt").read_text(encoding="utf-8", errors="ignore"))
    valid_stories = clean((DATA / "valid.txt").read_text(encoding="utf-8"))
    print(f"{len(train_stories):,} training stories, {len(valid_stories):,} validation stories")

    tok = train_tokenizer(train_stories)
    tok.save(str(OUT / "tokenizer.json"))

    for name, stories in [("train", train_stories), ("val", valid_stories)]:
        ids = encode_stories(tok, stories)
        ids.tofile(OUT / f"{name}.bin")
        print(f"{name}: {len(ids):,} tokens")

    sample = "Once upon a time, there was a little girl named Lily."
    print([tok.id_to_token(i) for i in tok.encode(sample).ids])
