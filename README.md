# One Sentence, All the Way Through

An interactive course that teaches transformers from scratch. You type one
sentence and follow it through a real GPT running in your browser, from text to
the predicted next word. Every number shown is computed live from your sentence.

**Read it online: https://heera-ai.github.io/one-sentence-transformer/**

- `site/` is the course: a static website (Svelte + TypeScript).
- `model/` is the GPT behind it: PyTorch code to train it and package it for the site.
- `docs/design.md` is the design the project was built from.

## Read the course

```sh
cd site
pnpm install
pnpm dev          # open the address it prints
```

`pnpm build` writes a static site to `site/dist` that can be hosted anywhere.

## Retrain the model

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). On an Apple M4 the
whole pipeline takes about an hour, most of it training.

```sh
cd model
uv sync
uv run python prepare_data.py     # download 200 MB of TinyStories
uv run python tokenizer.py        # learn the vocabulary, encode the text
uv run pytest                     # check the model is wired correctly
uv run python train.py            # train (--n_layer, --d_model, --steps ... to change it)
uv run python export.py           # write weights and test fixtures into site/
```

After `export.py`, run the site tests: the browser engine must still reproduce
PyTorch's numbers for the new weights.

## Tests

```sh
cd model && uv run pytest         # model: shapes, causal mask, can overfit a batch
cd site  && pnpm test             # engine: tokenizer and every stage match PyTorch
cd site  && pnpm e2e              # every chapter in a real browser, desktop and phone
```

## How the site runs the model

`site/src/engine/forward.ts` is the whole transformer in about 150 lines of
plain TypeScript, mirroring `model/model.py`. It returns every intermediate
result (embeddings, attention weights, neuron activations, scores), and every
figure in the course reads from that one trace. Switches for removing position
information, silencing a head, or skipping an MLP are part of the engine, which
is what makes the "change it and see" figures real.

## Credits

Training data: [TinyStories](https://arxiv.org/abs/2305.07759) by Ronen Eldan
and Yuanzhi Li. The model follows the GPT-2 architecture.
