"""Package the trained model for the website.

Writes to site/public/model/:
    manifest.json        config, tensor layout, checkpoint list, training log
    weights.bin          the final weights as float16
    checkpoints/*.bin    earlier snapshots, same layout
    tokenizer.json       vocabulary and merge rules
    extras.json          token counts, a 2D map of the embeddings, neuron examples

and to site/tests/fixtures/: reference outputs from PyTorch that the
TypeScript engine must reproduce.

Weights are rounded to float16 *before* the reference outputs are computed, so
the browser and PyTorch run exactly the same numbers.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import EOT, clean

HERE = Path(__file__).parent
OUT = HERE / "out"
SITE = HERE.parent / "site"
PUBLIC = SITE / "public" / "model"
FIXTURES = SITE / "tests" / "fixtures"

FIXTURE_TEXT = "Once upon a time, there was a little girl named Lily. She loved to play with her"


def load(path: Path) -> tuple[GPT, int]:
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    # Round to float16 and back, so this model is bit-for-bit what the browser loads.
    state = {k: v.to(torch.float16).to(torch.float32) for k, v in ckpt["state"].items()}
    model.load_state_dict(state)
    return model.eval(), ckpt["step"]


def write_weights(model: GPT, dest: Path) -> list[dict]:
    layout, offset, chunks = [], 0, []
    for name, tensor in model.state_dict().items():
        arr = tensor.numpy().astype("<f2")
        layout.append({"name": name, "shape": list(arr.shape), "offset": offset})
        offset += arr.size
        chunks.append(arr.tobytes())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"".join(chunks))
    return layout


def export_tokenizer() -> dict:
    raw = json.loads((OUT / "tokenizer.json").read_text())
    vocab = raw["model"]["vocab"]
    tokens = [None] * len(vocab)
    for tok, i in vocab.items():
        tokens[i] = tok
    merges = [m if isinstance(m, list) else m.split(" ") for m in raw["model"]["merges"]]
    data = {"tokens": tokens, "merges": merges, "eot": vocab[EOT]}
    (PUBLIC / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False))
    return data


def r(t: torch.Tensor, digits: int = 5):
    return np.round(t.squeeze(0).numpy().astype(np.float64), digits).tolist()


def export_fixtures(model: GPT, tok) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    ids = tok.encode(FIXTURE_TEXT).ids
    idx = torch.tensor([ids])
    trace: dict = {}
    with torch.no_grad():
        model(idx, trace=trace)
        variants = {
            "no_pos": model(idx, no_pos=True)[0],
            "ablate_head_1_2": model(idx, ablate_heads=[(1, 2)])[0],
            "ablate_mlp_0": model(idx, ablate_mlp=[0])[0],
        }
    keys = ["resid_pre", "q", "k", "v", "attn", "z", "attn_out", "resid_mid", "mlp_pre", "mlp_act", "mlp_out", "resid_post"]
    fixture = {
        "text": FIXTURE_TEXT,
        "ids": ids,
        "tok_emb": r(trace["tok_emb"]),
        "embed": r(trace["embed"]),
        "layers": [{k: r(layer[k]) for k in keys} for layer in trace["layers"]],
        "final": r(trace["final"]),
        "logits": r(trace["logits"], 4),
        "variants": {k: r(v[:, -1], 4) for k, v in variants.items()},
    }
    (FIXTURES / "trace.json").write_text(json.dumps(fixture))

    stories = clean((HERE / "data" / "valid.txt").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    cases = []
    for story in rng.choice(stories, size=990, replace=False):
        sentence = story[: rng.integers(20, 200)]
        cases.append({"text": sentence, "ids": tok.encode(sentence).ids})
    for text in ["", " ", "Hello", "  two  spaces", "don't can't I'll", "naïve café 123", "emoji 🙂 ok", "A\nB\n\nC", "x" * 50, "Zyzzyva qwrtp!!"]:
        cases.append({"text": text, "ids": tok.encode(text).ids})
    (FIXTURES / "tokenizer_cases.json").write_text(json.dumps(cases, ensure_ascii=False))


def embedding_map(model: GPT, tokens: list[str], counts: np.ndarray, n: int = 600) -> list[dict]:
    """2D PCA of the embeddings of the most common whole-word tokens."""
    is_word = np.array([t.startswith("Ġ") and t[1:].isalpha() and len(t) > 3 for t in tokens])
    order = [i for i in np.argsort(-counts) if is_word[i]][:n]
    emb = model.wte.weight.detach()[order]
    emb = emb - emb.mean(0)
    _, _, v = torch.pca_lowrank(emb, q=2, center=False)
    xy = (emb @ v[:, :2]).numpy()
    return [{"id": int(i), "x": round(float(x), 3), "y": round(float(y), 3)} for i, (x, y) in zip(order, xy)]


@torch.no_grad()
def neuron_examples(model: GPT, val: np.ndarray, windows: int = 3000, top: int = 4, ctx: int = 8) -> list:
    """For every MLP neuron, the places in real text where it fired most."""
    cfg = model.cfg
    T = cfg.block_size
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(val) - T, size=windows)
    best_val = [torch.full((top, 4 * cfg.d_model), -1e9) for _ in range(cfg.n_layer)]
    best_pos = [torch.zeros((top, 4 * cfg.d_model), dtype=torch.long) for _ in range(cfg.n_layer)]
    for b in range(0, windows, 250):
        batch = starts[b : b + 250]
        idx = torch.from_numpy(np.stack([val[s : s + T] for s in batch]).astype(np.int64))
        # Where each activation sits in the validation file.
        where = torch.from_numpy(batch)[:, None] + torch.arange(T)[None, :]
        trace: dict = {}
        model(idx, trace=trace)
        for l, layer in enumerate(trace["layers"]):
            act = layer["mlp_act"].reshape(-1, 4 * cfg.d_model)
            vals = torch.cat([best_val[l], act])
            pos = torch.cat([best_pos[l], where.reshape(-1, 1).expand(-1, 4 * cfg.d_model)])
            top_vals, top_idx = vals.topk(top, dim=0)
            best_val[l], best_pos[l] = top_vals, pos.gather(0, top_idx)
    out = []
    for l in range(cfg.n_layer):
        layer_out = []
        for n in range(4 * cfg.d_model):
            examples = []
            for j in range(top):
                p = int(best_pos[l][j, n])
                examples.append({"a": round(float(best_val[l][j, n]), 2), "ctx": val[max(0, p - ctx) : p + 1].tolist()})
            layer_out.append(examples)
        out.append(layer_out)
    return out


def main():
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--final", type=int, default=None, help="step of the checkpoint to ship (default: the latest)")
    args = p.parse_args()

    ckpts = sorted((OUT / "checkpoints").glob("step_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    if args.final is not None:
        ckpts = [c for c in ckpts if int(c.stem.split("_")[1]) <= args.final]
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for old in (PUBLIC / "checkpoints").glob("*.bin"):
        old.unlink()

    log = json.loads((OUT / "trainlog.json").read_text())
    val_at = {e["step"]: e["val"] for e in log}

    checkpoints = []
    for path in ckpts:
        model, step = load(path)
        write_weights(model, PUBLIC / "checkpoints" / f"step_{step}.bin")
        checkpoints.append({"step": step, "file": f"checkpoints/step_{step}.bin", "val": val_at.get(step)})

    model, step = load(ckpts[-1])
    layout = write_weights(model, PUBLIC / "weights.bin")
    tok_data = export_tokenizer()
    tok = Tokenizer.from_file(str(OUT / "tokenizer.json"))

    manifest = {
        "config": vars(model.cfg),
        "params": model.num_params(),
        "step": step,
        "tensors": layout,
        "checkpoints": checkpoints,
        "trainlog": [e for e in log if e["step"] <= step],
    }
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest))

    train = np.memmap(OUT / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(OUT / "val.bin", dtype=np.uint16, mode="r")
    counts = np.bincount(train, minlength=model.cfg.vocab_size)
    extras = {
        "counts": counts.tolist(),
        "map": embedding_map(model, tok_data["tokens"], counts),
        "neurons": neuron_examples(model, val),
    }
    (PUBLIC / "extras.json").write_text(json.dumps(extras))

    export_fixtures(model, tok)
    size = sum(f.stat().st_size for f in PUBLIC.rglob("*") if f.is_file())
    print(f"exported step {step}: {model.num_params():,} parameters, {size / 1e6:.1f} MB in {PUBLIC}")


if __name__ == "__main__":
    main()
