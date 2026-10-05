<script lang="ts">
  import { chapters } from '../lib/chapters'
  import { app } from '../lib/state.svelte'
  import { big, bytes } from '../lib/util'
  import Code from '../viz/Code.svelte'
  import Quiz from '../viz/Quiz.svelte'
  import Tex from '../viz/Tex.svelte'

  const manifest = app.manifest!
  const mine = manifest.config

  // --- The calculator ---
  const presets = [
    { name: 'The model in this page', L: mine.n_layer, D: mine.d_model, H: mine.n_head, V: mine.vocab_size, C: mine.block_size },
    { name: 'GPT-2 small (2019)', L: 12, D: 768, H: 12, V: 50257, C: 1024 },
    { name: 'GPT-2 XL (2019)', L: 48, D: 1600, H: 25, V: 50257, C: 1024 },
    { name: 'GPT-3 (2020)', L: 96, D: 12288, H: 96, V: 50257, C: 2048 },
  ]
  let L = $state(mine.n_layer)
  let D = $state(mine.d_model)
  let H = $state(mine.n_head)
  let V = $state(mine.vocab_size)
  let C = $state(mine.block_size)
  let logTokens = $state(Math.log10(manifest.step * 128 * mine.block_size))
  function load(p: (typeof presets)[number]) {
    L = p.L
    D = p.D
    H = p.H
    V = p.V
    C = p.C
  }
  const ok = $derived(L > 0 && D > 0 && H > 0 && V > 0 && C > 0)
  const emb = $derived(V * D + C * D)
  const attn = $derived(L * (4 * D * D + 4 * D))
  const mlp = $derived(L * (8 * D * D + 5 * D))
  const norms = $derived((2 * L + 1) * 2 * D)
  const total = $derived(emb + attn + mlp + norms)
  const tokens = $derived(10 ** logTokens)
  const flops = $derived(6 * (attn + mlp) * tokens)
  const GPU = 1e14 // useful operations per second: roughly one data-centre GPU of the A100 class in practice
  const gpuDays = $derived(flops / GPU / 86400)
  const kv = $derived(2 * L * C * D * 2)
  const parts = $derived([
    { name: 'Tables', n: emb, c: 'var(--ink)' },
    { name: 'Attention', n: attn, c: 'var(--talk)' },
    { name: 'MLPs', n: mlp, c: 'var(--think-fill)' },
  ])
  function duration(days: number): string {
    if (days < 1 / 24) return `${Math.max(1, Math.round(days * 1440))} GPU-minutes`
    if (days < 1) return `${(days * 24).toFixed(1)} GPU-hours`
    if (days < 365) return `${days.toFixed(days < 10 ? 1 : 0)} GPU-days`
    return `${big(days / 365)} GPU-years`
  }

  // --- Final recall: put the pipeline in order ---
  const steps = [
    'Cut the text into tokens and look up each one’s ID',
    'Swap each ID for its row in the embedding table',
    'Add the vector for its position',
    'Normalise a copy of the stream; attention reads from earlier tokens and adds what it found',
    'Normalise a copy again; the MLP works on each token alone and adds its result',
    'Do the previous two steps again for every layer',
    'Normalise the last token’s stream and take its dot product with every token’s embedding',
    'Softmax turns those scores into probabilities',
    'Draw one token, add it to the text, and start again',
  ]
  const shuffled = [3, 7, 0, 5, 8, 2, 6, 1, 4]
  let placed = $state<number[]>([])
  let wrong = $state(-1)
  function place(i: number) {
    if (i === placed.length) {
      placed = [...placed, i]
      wrong = -1
    } else wrong = i
  }
</script>

<p>
  The model in this page has {big(manifest.params)} weights. Openly published models today run from a few billion to several hundred billion, and the commercial ones you talk to
  every day are believed to be at least that large. It is fair to ask how much of what you have learned carries over.
</p>
<p>
  Nearly all of it. A frontier model is the machine in the diagram above the chapter list: tokens, vectors, layers of talk and think, a vote. What changes is five numbers, a handful of substitutions
  for individual parts, and a great deal of engineering to make the arithmetic affordable. This chapter goes through all three.
</p>

<h2>The five numbers</h2>
<div class="fig">
  <p class="fig-title">Size a model</p>
  <div class="controls">
    {#each presets as p}<button class="btn quiet" onclick={() => load(p)}>{p.name}</button>{/each}
  </div>
  <div class="inputs">
    <label>Layers <input type="number" min="1" max="200" bind:value={L} /></label>
    <label>Width (d_model) <input type="number" min="8" max="20000" step="8" bind:value={D} /></label>
    <label>Heads per layer <input type="number" min="1" max="128" bind:value={H} /></label>
    <label>Vocabulary <input type="number" min="256" max="300000" bind:value={V} /></label>
    <label>Context length <input type="number" min="8" max="2000000" bind:value={C} /></label>
  </div>
  {#if ok}
    <p class="bigline">{total.toLocaleString('en')} weights <span class="muted">({big(total)})</span></p>
    <div class="split" role="img" aria-label="Share of weights by part">
      {#each parts as p}<span style:width="{(p.n / total) * 100}%" style:background={p.c}></span>{/each}
    </div>
    <p class="legend">{#each parts as p}<span><i style:background={p.c}></i>{p.name} {((p.n / total) * 100).toFixed(1)}%</span>{/each}</p>
    <table class="plain">
      <tbody>
        <tr><th>To store the weights</th><td>{bytes(total * 4)} at 32 bits each · {bytes(total * 2)} at 16 bits · {bytes(total / 2)} squeezed to 4 bits</td></tr>
        <tr><th>Vector per head</th><td>{D} ÷ {H} = {(D / H).toFixed(D % H ? 1 : 0)} numbers{D % H ? ' (the width must divide evenly by the heads)' : ''}</td></tr>
        <tr><th>Attention scores for a full context</th><td>{big(C * C)} per head, {big(L * H * C * C)} across the model: this grows with the <em>square</em> of the context</td></tr>
        <tr><th>KV cache for a full context</th><td>{bytes(kv)} at 16 bits (keys and values for {L} layers × {C.toLocaleString('en')} tokens × {D})</td></tr>
      </tbody>
    </table>
    <label class="tokens">Training tokens <input type="range" min="6" max="13" step="0.1" bind:value={logTokens} /> <span class="num">{big(tokens)}</span></label>
    <table class="plain">
      <tbody>
        <tr><th>Tokens per weight</th><td>{(tokens / total).toFixed(tokens / total < 10 ? 1 : 0)}</td></tr>
        <tr><th>Arithmetic to train</th><td>about {flops.toExponential(1).replace('e+', ' × 10^')} operations, or roughly {duration(gpuDays)}</td></tr>
      </tbody>
    </table>
    <p class="fig-note">
      The training estimate uses the rule of thumb “6 operations per weight per token” (2 for the forward pass, 4 for the backward pass) and a GPU that manages 10¹⁴ useful operations a
      second, about one A100-class card in practice. Treat it as an order of magnitude.
    </p>
  {:else}
    <p class="muted">Enter a positive number in every box.</p>
  {/if}
</div>

<p>The count is one formula, and you can derive every term from chapter 10:</p>
<Tex block tex={'\\text{weights} \\approx \\underbrace{V d + C d}_{\\text{tables}} + L\\,(\\underbrace{4d^2}_{\\text{attention}} + \\underbrace{8d^2}_{\\text{MLP}}) = (V + C)\\,d + 12\\,L\\,d^2'} />
<p>
  Load the GPT-2 small preset: 124,439,808, which is exactly the published size. GPT-3 comes out near its published 175 billion. The same architecture, the same formula; only the five
  numbers moved. Three things to take from playing with it:
</p>
<ul>
  <li><strong>Width is the expensive dial.</strong> Weights grow with width squared and only in proportion to depth.</li>
  <li><strong>Context length costs almost no weights</strong> but a great deal of computation and memory, because every token attends to every earlier token.</li>
  <li><strong>The vocabulary matters less and less.</strong> It is a third of this model and a rounding error in a large one.</li>
</ul>

<h2>What modern models swap out</h2>
<p>
  The model you studied is essentially GPT-2, from 2019. Models since then have kept the structure and replaced a few parts with better-behaved versions. You can now understand each change
  in a sentence.
</p>
<div class="scroll-x">
  <table class="plain wide">
    <thead><tr><th>Part</th><th>Here (GPT-2 style)</th><th>Common now</th><th>Why</th></tr></thead>
    <tbody>
      <tr>
        <td><a href="#/position">Position</a></td><td>A learned vector per position, added at the start</td><td><strong>Rotary embeddings (RoPE)</strong>: queries and keys are rotated by an angle set by their position</td>
        <td>After rotation, how well a query matches a key depends on their contents and on how far apart they are, not on where in the text they sit. Distance is what language mostly cares about, and there is no table to run out of.</td>
      </tr>
      <tr>
        <td><a href="#/block">Normalisation</a></td><td>LayerNorm: subtract the mean, divide by the spread</td><td><strong>RMSNorm</strong>: only divide by the root-mean-square size</td>
        <td>Dropping the mean subtraction is cheaper and works as well.</td>
      </tr>
      <tr>
        <td><a href="#/mlp">MLP</a></td><td>Expand, GELU, shrink</td><td><strong>SwiGLU</strong>: two expansions, one of which gates the other by multiplication</td>
        <td>Slightly better quality for the same number of weights.</td>
      </tr>
      <tr>
        <td><a href="#/heads">Attention</a></td><td>Every head has its own keys and values</td><td><strong>Grouped-query attention</strong>: several query heads share one set of keys and values</td>
        <td>Shrinks the KV cache, which is the main memory cost when generating long texts.</td>
      </tr>
      <tr>
        <td><a href="#/tokens">Tokens</a></td><td>{mine.vocab_size.toLocaleString('en')} BPE tokens</td><td>Roughly 100,000 to 250,000</td>
        <td>Covers many languages and code with reasonably short sequences.</td>
      </tr>
      <tr>
        <td><a href="#/vote">Context</a></td><td>{mine.block_size} tokens</td><td>A hundred thousand to over a million</td>
        <td>Whole documents and codebases fit. Made affordable by the engineering below.</td>
      </tr>
    </tbody>
  </table>
</div>
<p>
  Not one of these changes the picture. There are still lanes, they still talk and then think, and the last one still votes.
</p>

<h2>Making the arithmetic affordable</h2>
<p>Scaling up is mostly a fight against memory and time. These are the standard tools, roughly in the order you will need them as your models grow.</p>

<h3>Training</h3>
<ul>
  <li>
    <strong>Mixed precision.</strong> Do most arithmetic in 16-bit numbers and keep 32-bit only where precision matters. About half the memory, and usually much faster on a GPU. One
    line: <code>torch.autocast</code>.
  </li>
  <li>
    <strong>Fused attention.</strong> The <code>T × T</code> score matrix never needs to exist in full. <code>F.scaled_dot_product_attention</code> can compute the same result in pieces
    (the FlashAttention method), saving most of attention's memory. It replaces the four middle lines of our <code>Attention.forward</code>.
  </li>
  <li>
    <strong>Gradient accumulation.</strong> If the batch you want does not fit in memory, run several small batches, add up their gradients, and only then take a step. The result is the
    same as one large batch.
  </li>
  <li>
    <strong>Compilation.</strong> <code>torch.compile(model)</code> fuses operations and removes Python overhead.
  </li>
  <li>
    <strong>Many GPUs.</strong> Each GPU processes a different part of the batch with its own copy of the model; the gradients are averaged before each step. Beyond that, models too
    large for one GPU are split across several.
  </li>
  <li>
    <strong>Data that streams.</strong> The token file is read straight from disk as needed (our <code>train.py</code> already does this with <code>np.memmap</code>), so the dataset can be
    far larger than memory.
  </li>
  <li><strong>Checkpoints.</strong> Long runs fail. Save the weights and the optimizer's state regularly, so a crash costs minutes and not days.</li>
</ul>
<Code
  caption="The training step from chapter 9, with mixed precision and gradient accumulation"
  code={`
accum = 4                                            # 4 small batches act as one large one
for step in range(steps):
    for micro in range(accum):
        x, y = get_batch(train_data, batch_size, block_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / accum).backward()                    # gradients add up across the small batches
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
`}
/>

<h3>Running the model</h3>
<ul>
  <li>
    <strong>KV cache</strong> (<a href="#/vote">chapter 8</a>). Keep every lane's keys and values, and compute only the newest lane for each new token. Without it, writing <em>n</em> tokens
    costs on the order of <em>n</em>² lane computations; with it, <em>n</em>.
  </li>
  <li>
    <strong>Quantisation.</strong> Store each weight in 8 or 4 bits in place of 16. The model gets slightly worse and two to four times smaller, which is what lets large models run on a
    laptop or phone.
  </li>
  <li><strong>Batching.</strong> Serve many users' requests in one pass through the model, since a GPU is much more efficient on large batches.</li>
</ul>

<h2>How much model, how much data?</h2>
<p>
  The most important empirical fact about these models is how predictable they are. Across many orders of magnitude, the loss falls smoothly as you add weights, data and computation,
  following what are called <strong>scaling laws</strong>. Nothing special happens at any size; there is no trick that large models have and small ones lack. This predictability is what
  made it rational to spend so much on larger models.
</p>
<p>
  The laws also say how to split a budget. If the model is too small for its data, it cannot absorb what it is shown; too large, and computation is wasted on weights that the data cannot
  fill. A much-cited 2022 study (known as “Chinchilla”) put the balance for their setting at about <strong>20 training tokens per weight</strong>. The model in this page saw
  {big(manifest.step * 128 * mine.block_size)} tokens for {big(manifest.params)} weights, about {Math.round((manifest.step * 128 * mine.block_size) / manifest.params)} per weight: far
  more data than it could use, which is fine for a model meant to be small.
</p>

<h2>A path from here</h2>
<p>If you want to build your own models, climb in steps. Each one is a real project and each teaches something the previous one could not.</p>
<ol>
  <li>
    <strong>Retrain this model</strong> on your laptop and change one thing at a time (the exercises in <a href="#/build">chapter 10</a>). Cost: minutes to an hour.
  </li>
  <li>
    <strong>10 to 30 million weights on all of TinyStories.</strong> Use the 6-layer, wider settings and the whole dataset. The stories become clearly coherent. A single consumer GPU,
    a few hours to a day.
  </li>
  <li>
    <strong>Modernise it.</strong> Swap in RoPE, RMSNorm and fused attention, one at a time, checking the loss after each. You will then have read and changed every line of a current
    architecture.
  </li>
  <li>
    <strong>Reproduce GPT-2 small</strong> (124 million weights) on a web-text dataset. This needs mixed precision, gradient accumulation, several GPUs and checkpointing: for 300
    billion training tokens the calculator above gives about 18 GPU-days, and in practice you should budget a GPU-month. It is small enough to train yourself and large enough to write general English, which makes it the standard rite of passage.
  </li>
  <li>
    <strong>Beyond that</strong>, the model code barely changes. The work becomes data: collecting it, cleaning it, and evaluating what the model learned from it.
  </li>
</ol>

<h2>The last test</h2>
<p>
  The course promised you would not forget how a transformer works. Check it now, from memory: click the steps in the order they happen, starting from a sentence of text.
</p>
<div class="fig">
  <p class="fig-title">{placed.length === steps.length ? 'That is a transformer, start to finish.' : `Step ${placed.length + 1} of ${steps.length}: what happens next?`}</p>
  {#if placed.length}
    <ol class="done">
      {#each placed as i}<li>{steps[i]}</li>{/each}
    </ol>
  {/if}
  <div class="pool">
    {#each shuffled as i}
      {#if !placed.includes(i)}
        <button class="card" class:wrong={wrong === i} onclick={() => place(i)}>{steps[i]}</button>
      {/if}
    {/each}
  </div>
  <p class="readout" role="status">
    {#if wrong >= 0}Not yet. Something has to happen before that. Look at the diagram above the chapter list.
    {:else if placed.length === steps.length}You rebuilt the whole pipeline.
    {:else}Click the step that comes next.{/if}
  </p>
  {#if placed.length}<button class="btn quiet" onclick={() => ((placed = []), (wrong = -1))}>Start over</button>{/if}
</div>

<h2>Everything, on one card</h2>
<ol class="card-list">
  {#each chapters.slice(0, -1) as c}
    <li><span><a href="#/{c.slug}">{c.title}.</a> {c.takeaway}</span></li>
  {/each}
</ol>

<h2>Where to read next</h2>
<ul>
  <li><a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a> (Vaswani and others, 2017). The paper that introduced the transformer. You can now read it.</li>
  <li><a href="https://github.com/karpathy/nanoGPT">nanoGPT</a> (Andrej Karpathy). A compact GPT-2 training codebase; the natural next thing to run after this one.</li>
  <li><a href="https://arxiv.org/abs/2305.07759">TinyStories</a> (Eldan and Li, 2023). The dataset used here, and a study of how small a useful language model can be.</li>
  <li>
    <a href="https://transformer-circuits.pub/2021/framework/index.html">A Mathematical Framework for Transformer Circuits</a> (Elhage and others, 2021). Where the “stream” way of
    thinking comes from, and the start of the research on what heads and neurons do.
  </li>
  <li><a href="https://arxiv.org/abs/2203.15556">Training Compute-Optimal Large Language Models</a> (Hoffmann and others, 2022). The “Chinchilla” scaling study.</li>
</ul>

<h2>Check yourself</h2>
<Quiz
  question="You want a model that can read documents ten times longer. What grows the most?"
  options={[
    { text: 'The number of weights', why: 'Context length adds almost no weights (none at all with rotary positions).' },
    { text: 'The attention computation and the KV cache', correct: true, why: 'Attention compares every token with every earlier one, so its cost grows about a hundredfold for ten times the length, and the cache grows tenfold.' },
    { text: 'The vocabulary', why: 'The vocabulary is unrelated to how long the input is.' },
  ]}
/>
<Quiz
  question="What is the main difference between the model in this page and a frontier model?"
  options={[
    { text: 'A fundamentally different algorithm', why: 'The algorithm is the one you just learned.' },
    { text: 'Scale: the same structure with far larger numbers, more data, and refinements to a few parts', correct: true, why: 'Tokens, vectors, attention, MLPs, a stream, a vote, trained by next-token prediction. The rest is size and engineering.' },
    { text: 'Large models do not use attention', why: 'Attention is the core of all of them.' },
  ]}
/>

<style>
  .inputs {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
    gap: 0.7rem;
    margin: 0.6rem 0 1.2rem;
  }
  .inputs label {
    display: grid;
    gap: 0.2rem;
    font-size: 0.84rem;
    font-weight: 600;
  }
  .inputs input {
    font-family: var(--mono);
    font-size: 0.95rem;
    border: 1.5px solid var(--line);
    border-radius: 8px;
    padding: 0.4rem 0.6rem;
    width: 100%;
  }
  .bigline {
    font-size: 1.7rem;
    font-weight: 720;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem;
    font-variant-numeric: tabular-nums;
  }
  .bigline .muted {
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 0;
  }
  .split {
    display: flex;
    height: 16px;
    border-radius: 4px;
    overflow: hidden;
    max-width: 40rem;
  }
  .split span {
    transition: width 0.3s;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem 1.2rem;
    font-size: 0.82rem;
    margin: 0.4rem 0 1rem;
  }
  .legend i {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 0.35rem;
  }
  .tokens {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.7rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  .tokens input {
    width: 16rem;
    max-width: 100%;
  }
  table.wide {
    min-width: 44rem;
    max-width: 62rem;
    font-size: 0.9rem;
  }
  .fig table.plain th {
    white-space: nowrap;
    font-weight: 600;
  }
  .done {
    max-width: none;
    margin-bottom: 1rem;
  }
  .done li {
    font-weight: 560;
  }
  .pool {
    display: grid;
    gap: 6px;
    max-width: 46rem;
  }
  .card {
    text-align: left;
    background: #edf1f5;
    border: 1.5px solid transparent;
    border-radius: 9px;
    padding: 0.55rem 0.85rem;
    font-family: var(--display);
    font-size: 0.95rem;
  }
  .card:hover {
    border-color: var(--ink);
  }
  .card.wrong {
    border-color: var(--neg);
    background: #fae8ef;
  }
  .card-list {
    list-style: none;
    padding: 0;
    counter-reset: c -1;
    max-width: 46rem;
  }
  .card-list li {
    counter-increment: c;
    display: grid;
    grid-template-columns: 2rem 1fr;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--line);
    font-family: var(--display);
    font-size: 0.98rem;
    line-height: 1.4;
  }
  .card-list li::before {
    content: counter(c);
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--ink-3);
    padding-top: 0.15rem;
  }
  .card-list li a {
    font-weight: 650;
    color: var(--ink);
  }
</style>
