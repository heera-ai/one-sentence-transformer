<script lang="ts">
  import { row } from '../engine/ops'
  import { fmt } from '../lib/color'
  import { app } from '../lib/state.svelte'
  import { cosine } from '../lib/util'
  import Bars from '../viz/Bars.svelte'
  import Cells from '../viz/Cells.svelte'
  import Chips from '../viz/Chips.svelte'
  import Code from '../viz/Code.svelte'
  import Quiz from '../viz/Quiz.svelte'
  import Rows from '../viz/Rows.svelte'
  import Tex from '../viz/Tex.svelte'

  const { d_model: D, vocab_size: V } = app.model!.config
  const wte = app.model!.w['wte.weight']
  const trace = $derived(app.trace!)
  const id = $derived(app.ids[app.pos])
  const vec = $derived(row(wte, id, D))
  let hover = $state('')

  // --- Two arrows and their dot product ---
  let angle = $state(40)
  let len = $state(1.6)
  const a = [2, 0]
  const b = $derived([len * Math.cos((angle * Math.PI) / 180), len * Math.sin((angle * Math.PI) / 180)])
  const ab = $derived(a[0] * b[0] + a[1] * b[1])

  // --- Nearest neighbours in the real table ---
  let probe = $state('')
  const probeIds = $derived(probe.trim() ? app.tok!.encode(' ' + probe.trim()) : [])
  const target = $derived(probeIds.length === 1 ? probeIds[0] : id)
  const neighbours = $derived.by(() => {
    const t = row(wte, target, D)
    const counts = app.extras?.counts
    const sims: { id: number; s: number }[] = []
    for (let i = 0; i < V; i++) {
      if (i === target || (counts && counts[i] < 200)) continue // skip tokens too rare to have a settled vector
      sims.push({ id: i, s: cosine(t, row(wte, i, D)) })
    }
    return sims.sort((x, y) => y.s - x.s).slice(0, 10)
  })

  // --- The 2D map ---
  const map = $derived(app.extras?.map ?? [])
  const box = $derived.by(() => {
    const xs = map.map((p) => p.x)
    const ys = map.map((p) => p.y)
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) }
  })
  const W = 880
  const H = 520
  const px = (x: number) => 30 + ((x - box.x0) / (box.x1 - box.x0)) * (W - 60)
  const py = (y: number) => 20 + ((y - box.y0) / (box.y1 - box.y0)) * (H - 40)
  const inSentence = $derived(new Set(app.ids))
  const near = $derived(new Set(neighbours.map((n) => n.id)))
  let mapHover = $state(-1)
</script>

<p>
  The model now holds a list of ID numbers. But an ID is only a label. Token {app.ids[0]} is not “less than” token {app.ids[0] + 1} in any useful sense. To compute with tokens, the model needs a
  description of each one, written in numbers.
</p>
<p>
  So the model keeps a big table with one row per token: {V.toLocaleString('en')} rows, each a list of {D} numbers. A list of numbers like this is called a <strong>vector</strong>, and the row
  for a token is its <strong>embedding</strong>. The first thing the model does is swap each ID for its row.
</p>

<div class="fig">
  <p class="fig-title">Look up a token's vector</p>
  <p class="fig-note">Pick a token. Its ID selects one row of the table. <span style="color:var(--pos)">Teal</span> is positive, <span style="color:var(--neg)">rose</span> is negative, pale is near zero.</p>
  <Chips ids={app.ids} selected={app.pos} onselect={(i) => (app.focus = i)} sub={(i) => String(app.ids[i])} />
  <p class="sub">Row {id} of the table: the vector for <span class="mono">{app.label(id)}</span></p>
  <div class="scroll-x">
    <Cells data={vec} rows={1} cols={D} cw={6} ch={34} label="Embedding vector of the selected token" onhover={(_, c, v) => (hover = `dimension ${c}: ${fmt(v, 3)}`)} onleave={() => (hover = '')} />
  </div>
  <p class="readout">{hover || `${D} numbers. First five: ${Array.from(vec.slice(0, 5), (v) => fmt(v, 2)).join('  ')} …`}</p>
</div>

<p>
  Nobody chose these numbers. They began random and were adjusted during training, along with every other weight. No single number has a meaning you can name, such as “is an animal”. What
  carries meaning is the vector as a whole: <strong>where it points</strong>.
</p>

<div class="fig">
  <p class="fig-title">Your whole sentence, as vectors</p>
  <p class="fig-note">One row per token. This grid, {app.ids.length} rows by {D} columns, is what moves through the rest of the model. If a token appears twice, its two rows are identical for now.</p>
  <Rows ids={app.ids} data={trace.tokEmb} cols={D} selected={app.pos} onselect={(i) => (app.focus = i)} name="Token embeddings for the sentence" />
</div>

<h2>Direction, and how to measure it</h2>
<p>
  Think of a vector as an arrow from the origin. With two numbers it is an arrow on paper; with {D} numbers it is an arrow in {D}-dimensional space, which nobody can picture, but the
  arithmetic is identical. The one operation you need is the <strong>dot product</strong>: multiply the two vectors number by number and add up the results.
</p>
<Tex block tex={'a \\cdot b = a_1 b_1 + a_2 b_2 + \\dots + a_n b_n'} />
<p>It gives a single number that says how much two arrows agree.</p>

<div class="fig">
  <p class="fig-title">Turn one arrow and watch the dot product</p>
  <div class="dot">
    <svg viewBox="-3 -3 6 6" role="img" aria-label="Two arrows from the origin">
      <circle r="2" class="guide" /><line x1="-3" x2="3" class="axis" /><line y1="-3" y2="3" class="axis" />
      <line x2={a[0]} y2="0" class="arrow a" /><line x2={b[0]} y2={-b[1]} class="arrow b" />
      <text x={a[0] + 0.15} y="0.35">a</text><text x={b[0] * 1.12} y={-b[1] * 1.12}>b</text>
    </svg>
    <div>
      <div class="controls">
        <label>Angle <input type="range" min="0" max="360" bind:value={angle} /> <span class="num">{angle}°</span></label>
        <label>Length of b <input type="range" min="0.3" max="2.6" step="0.1" bind:value={len} /> <span class="num">{len.toFixed(1)}</span></label>
      </div>
      <p class="mono calc">
        a = ({a[0].toFixed(1)}, {a[1].toFixed(1)})<br />
        b = ({b[0].toFixed(2)}, {b[1].toFixed(2)})<br />
        a · b = {a[0].toFixed(1)}×{b[0].toFixed(2)} + {a[1].toFixed(1)}×{b[1].toFixed(2)} = <strong style:color={ab >= 0 ? 'var(--pos)' : 'var(--neg)'}>{fmt(ab)}</strong>
      </p>
      <p>
        {#if Math.abs(ab) < 0.25}At a right angle the dot product is about zero: the arrows have nothing in common.
        {:else if ab > 0}Pointing the same way gives a positive number, and it grows with how well they line up and how long they are.
        {:else}Pointing opposite ways gives a negative number.{/if}
      </p>
    </div>
  </div>
</div>

<p>
  Hold on to the dot product. It is the workhorse of the whole model: attention uses it to decide which tokens are relevant to each other, every neuron in the MLP is one, and the final
  vote is {V.toLocaleString('en')} of them.
</p>
<p>
  If you divide out the lengths you get the <strong>cosine similarity</strong>, which runs from −1 (opposite) through 0 (unrelated) to +1 (same direction) and measures direction alone.
</p>

<h2>Similar tokens point the same way</h2>
<p>
  Here is the surprising part. Nobody told the model which words are alike, yet after training, tokens that are used in similar ways end up with similar vectors. Check it on the real table:
</p>

<div class="fig">
  <p class="fig-title">Nearest neighbours of <span class="mono">{app.label(target)}</span></p>
  <div class="controls">
    <label>Try a word <input type="text" bind:value={probe} placeholder="e.g. happy, dog, ran" spellcheck="false" /></label>
    {#if probeIds.length > 1}<span class="muted">That is {probeIds.length} tokens, so there is no single vector for it. Showing the selected token of your sentence.</span>{/if}
  </div>
  <Bars
    name="Tokens with the most similar embedding"
    color="var(--pos)"
    max={1}
    items={neighbours.map((n) => ({ label: app.label(n.id), value: Math.max(0, n.s), text: n.s.toFixed(2) }))}
    onpick={(i) => (probe = app.tok!.text(neighbours[i].id).trim())}
  />
  <p class="readout">Cosine similarity to every other token's vector; the ten highest. Click one to jump to it.</p>
</div>

<p>
  Why would this happen? Because the model is trained only to predict the next token, and words that can be swapped for each other, <em>happy</em> and <em>glad</em>, <em>mom</em> and
  <em>dad</em>, call for nearly the same predictions. Giving them nearby vectors is the cheapest way to treat them alike. Try a name, a colour, and a verb in the past tense.
</p>

{#if map.length}
  <div class="fig">
    <p class="fig-title">A flat map of the {map.length} most common words</p>
    <p class="fig-note">
      The {D} dimensions are squashed down to the two that spread the words out most. This throws away most of the detail, so treat it as a rough sketch. Dark dots are in your sentence; teal
      dots are the neighbours listed above.
    </p>
    <div class="scroll-x">
      <svg viewBox="0 0 {W} {H}" class="map" role="img" aria-label="Scatter plot of word embeddings in two dimensions" onmouseleave={() => (mapHover = -1)}>
        {#each map as p, i}
          {@const hot = inSentence.has(p.id) ? 'in' : near.has(p.id) || p.id === target ? 'near' : ''}
          <circle cx={px(p.x)} cy={py(p.y)} r={hot ? 5 : 3} class={hot} onmouseenter={() => (mapHover = i)} role="presentation" />
          {#if hot || i < 45 || i === mapHover}
            <text x={px(p.x) + 6} y={py(p.y) + 4} class={hot} class:hover={i === mapHover}>{app.tok!.text(p.id).trim()}</text>
          {/if}
        {/each}
      </svg>
    </div>
    <p class="readout">{mapHover >= 0 ? app.tok!.text(map[mapHover].id).trim() : 'Hover over a dot to read the word.'}</p>
  </div>
{/if}

<h2>In maths and code</h2>
<p>
  The table is a matrix <Tex tex={'E'} /> with {V.toLocaleString('en')} rows and {D} columns. The embedding of the token at position <Tex tex={'t'} /> is simply the row its ID points at:
</p>
<Tex block tex={'x_t = E[\\,\\text{id}_t\\,]'} />
<p>
  That table alone holds {V.toLocaleString('en')} × {D} = {(V * D).toLocaleString('en')} weights, about {Math.round((V * D * 100) / app.manifest!.params)}% of this small model. In PyTorch it is
  one line:
</p>
<Code
  code={`
import torch.nn as nn

wte = nn.Embedding(4096, 128)      # a 4096 x 128 table of learnable numbers
ids = torch.tensor([[${app.ids.slice(0, 6).join(', ')}]])   # shape (batch=1, T=${Math.min(6, app.ids.length)})
x = wte(ids)                       # shape (1, ${Math.min(6, app.ids.length)}, 128): one row per token
`}
/>

<h2>Check yourself</h2>
<Quiz
  question="Two vectors have a dot product close to zero. What does that say?"
  options={[
    { text: 'They point in opposite directions', why: 'Opposite directions give a large negative dot product.' },
    { text: 'They are at about a right angle: unrelated', correct: true, why: 'Zero means no agreement either way. (It can also mean one of them is very short.)' },
    { text: 'They are the same vector', why: 'A vector’s dot product with itself is its length squared, which is positive.' },
  ]}
/>
<Quiz
  question="How did “happy” and “glad” end up with similar vectors?"
  options={[
    { text: 'Someone supplied a dictionary of synonyms', why: 'The model was given nothing but stories.' },
    { text: 'They are spelled similarly', why: 'The model never sees spelling, only IDs.' },
    { text: 'They appear in similar contexts, so similar vectors help predict what follows', correct: true, why: 'Training only rewards good next-token predictions. Similar vectors for interchangeable words is a side effect of that.' },
  ]}
/>

<style>
  .sub {
    font-weight: 600;
    margin: 1rem 0 0.4rem;
  }
  .dot {
    display: grid;
    grid-template-columns: 240px 1fr;
    gap: 1.5rem;
    align-items: center;
  }
  @media (max-width: 640px) {
    .dot {
      grid-template-columns: 1fr;
    }
  }
  .dot svg {
    width: 240px;
    height: 240px;
  }
  .dot svg text {
    font-size: 0.34px;
    font-family: var(--mono);
    font-weight: 700;
  }
  .guide {
    fill: none;
    stroke: var(--line-soft);
    stroke-width: 0.02;
  }
  .axis {
    stroke: var(--line);
    stroke-width: 0.02;
  }
  .arrow {
    stroke-width: 0.09;
    stroke-linecap: round;
  }
  .arrow.a {
    stroke: var(--ink);
  }
  .arrow.b {
    stroke: var(--talk);
  }
  .calc {
    line-height: 1.8;
    font-size: 0.86rem;
  }
  input[type='text'] {
    font-family: var(--mono);
    border: 1.5px solid var(--line);
    border-radius: 8px;
    padding: 0.35rem 0.6rem;
    width: 13rem;
  }
  .map {
    width: 100%;
    min-width: 640px;
    background: #f6f8fa;
    border-radius: 8px;
  }
  .map circle {
    fill: #aab4c5;
  }
  .map circle.in {
    fill: var(--ink);
  }
  .map circle.near {
    fill: var(--pos);
  }
  .map text {
    font-family: var(--display);
    font-size: 11px;
    fill: var(--ink-3);
    pointer-events: none;
  }
  .map text.in {
    fill: var(--ink);
    font-weight: 700;
    font-size: 13px;
  }
  .map text.near {
    fill: var(--pos);
    font-weight: 700;
    font-size: 13px;
  }
  .map text.hover {
    fill: var(--ink);
    font-weight: 700;
    font-size: 13px;
  }
</style>
