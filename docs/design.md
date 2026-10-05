# Plan: "One Sentence, All the Way Through" — an interactive transformer course

## Context

The goal is a learning resource that teaches transformers from scratch so well that the learner never forgets how they work and can go on to build and scale their own models.

Decisions already made with the user:

- **Learner**: a programmer (Python, school algebra) with no machine learning background.
- **Format**: an interactive website that runs in any browser with no install.
- **Live model**: our own tiny GPT, trained by us and shipped with the site, so every visual shows real numbers.

The project folder is empty and is not a git repository. The machine is an Apple M4 with 16 GB RAM, with Node 24, pnpm, uv, Python 3.13 and PyTorch 2.11 (MPS available), which is enough to train the model locally.

## The core idea

The learner types one sentence, and that same sentence stays with them through every chapter. Each chapter opens up one stage of the machine and shows what that stage did to *their* sentence, using the real numbers from a real model running in the page.

Three things make it stick:

1. **One mental picture, repeated.** Every word gets its own lane (the residual stream). In each layer, words first *talk* (attention: read from earlier words), then *think* (MLP: process alone). The last lane *votes* on the next word. A mini-map of this picture is pinned on screen in every chapter with the current stage lit up.
2. **Everything can be poked.** Nothing is a canned animation. The learner can switch off position information, knock out an attention head, change the temperature, or scrub through training time, and the prediction changes because the model is really re-run.
3. **Recall, not just reading.** Each chapter ends with "predict before you reveal" questions and adds one line to a cumulative summary card. The last chapter asks the learner to rebuild the whole pipeline from memory.

Every chapter follows the same five beats: intuition → live visual → the maths → the code → check yourself.

## The model

A standard decoder-only GPT, small enough to show in full:

| Setting | Value |
|---|---|
| Layers | 4 |
| Heads per layer | 4 |
| Vector size (`d_model`) | 128 |
| Context length | 64 tokens |
| Vocabulary | 4,096 byte-level BPE tokens, trained by us |
| Other | pre-norm, GELU MLP (4×), learned positions, tied input/output embeddings |
| Size | about 1.3M parameters, about 3 MB to download |

- **Training data**: a slice (about 100 MB) of the TinyStories dataset, which is simple English written with a small vocabulary. That is why a model this small can predict sensibly.
- **Training**: PyTorch on the Mac's GPU (MPS). My estimate is 30–60 minutes; this is not measured yet.
- **Quality gate**: if next-word predictions on simple sentences look poor after training, step up to 6 layers and `d_model` 192 (about 3.4M parameters, about 7 MB). The training script takes these as settings, so this is a re-run and not a rewrite.
- **Checkpoints**: about 8 snapshots are saved during training (step 0 through the final step) for the training chapter.

## The chapters

| # | Chapter | What the learner does |
|---|---|---|
| 0 | The game | Plays "guess the next word" against the model. Learns that this single task is all the model does. |
| 1 | Tokens | Watches their sentence split into pieces and IDs. Steps through BPE merges to see how the vocabulary was learned. |
| 2 | Embeddings | Sees each ID become a vector. Explores nearest neighbours in the real embedding table; learns dot product as similarity. |
| 3 | Position | Switches position information off and watches predictions break, then sees how position vectors are added. |
| 4 | Attention, one head | Works through query, key and value by hand: scores → causal mask → softmax → weighted sum. Clicks any word to see what it reads from. |
| 5 | Many heads, one stream | Sees heads run in parallel and write into the lane. Knocks out a head and watches the prediction change. |
| 6 | The MLP | Sees each word "think alone". Inspects individual neurons and the tokens that fire them most. |
| 7 | The full block, stacked | Adds layer norm and residual connections. Decodes the lane after every layer to watch the prediction take shape. |
| 8 | From numbers to a word | Unembedding, softmax, temperature, top-k and top-p. Runs the generation loop one token at a time. |
| 9 | Training | Loss as "surprise", every position as a training example, gradients and AdamW. Scrubs a time slider across the saved checkpoints and watches noise turn into language. |
| 10 | Build it | The complete PyTorch model (about 200 lines) and training script, annotated, with every line linked back to its chapter. This is the exact code that produced the model in the page. |
| 11 | Scale it | A calculator: set layers, width, vocabulary and context, and see parameters, memory and compute. Then what large modern models change (rotary positions, RMSNorm, SwiGLU, KV cache, mixed precision, gradient accumulation, quantization) and why. |

## How it is built

```
gpt/
  model/                 Python (uv)
    prepare_data.py      download and slice TinyStories
    tokenizer.py         train the BPE vocabulary (Hugging Face `tokenizers`)
    model.py             the GPT — also the code shown in chapter 10
    train.py             training loop, saves checkpoints
    export.py            writes weights, tokenizer and a reference trace for the site
    tests/
  site/                  Vite + TypeScript + Svelte 5
    src/engine/          tokenizer.ts, ops.ts, forward.ts, sample.ts
    src/viz/             shared visual parts (lanes, heatmap, vector bars, mini-map)
    src/chapters/        one folder per chapter
    public/model/        weights.bin, manifest.json, tokenizer.json, checkpoints/
  docs/                  the design spec
```

Key technical choices:

- **The engine is hand-written TypeScript**, with no ML library. At 1.3M parameters, plain typed arrays are fast enough. `forward.ts` returns a full *trace*: embeddings, each layer's lane contents, every head's attention weights, queries/keys/values, MLP activations and final scores. Every visual reads from this one trace.
- **Switches are part of the engine**: no positions, head knocked out, stop after layer N. These power the "poke it" interactions.
- **Weights** ship as one float16 binary file plus a JSON manifest of names, shapes and offsets, decoded to `Float32Array` on load.
- **Visuals** use SVG and Canvas with `d3-scale`/`d3-shape` only. Maths is rendered with KaTeX, code highlighted with Shiki.
- **Output is a static site**, so it can be hosted anywhere.
- A few things are precomputed at export time and not in the browser: the 2D embedding map (PCA) and each neuron's top-firing tokens.

## Build order

Each stage ends with something that works on its own.

1. **Set-up**: `git init`, save this design as `docs/design.md`, scaffold `model/` and `site/`.
2. **Model pipeline**: data, tokenizer, model, training, export. Ends with trained weights and a reference trace on disk.
3. **Browser engine**: tokenizer and forward pass in TypeScript, proven to match PyTorch.
4. **Site shell + chapters 0–3**: layout, pinned mini-map, the shared sentence, summary card.
5. **Chapters 4–8**: attention through sampling. This is the heart of the course and gets the most design time.
6. **Chapters 9–11**: training time slider, build-it, scale-it.
7. **Polish**: visual design pass (using the `frontend-design` skill), keyboard and screen-reader access, phone layout, a line-by-line fact check of every chapter.

## Verification

- **Model**: unit tests for tensor shapes and the causal mask (changing a later token must not change earlier outputs); the model must overfit a single small batch; validation loss is reported at the end of training.
- **Engine matches PyTorch**: `export.py` writes a full trace for a fixed sentence; a Vitest test asserts the TypeScript trace matches within 1e-3 at every stage.
- **Tokenizer matches**: 1,000 sentences are encoded in both Python and TypeScript and must give identical IDs.
- **Site**: Playwright opens every chapter, types a sentence, uses each main control and checks for no console errors and that the prediction updates.
- **By hand**: run `pnpm dev`, go through all chapters start to finish with two or three different sentences, on desktop and phone width.
- **Facts**: every claim in the chapter text is checked against the code in `model.py` and, for chapter 11, against the original papers.
