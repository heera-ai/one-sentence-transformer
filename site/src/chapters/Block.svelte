<script lang="ts">
  import { lens } from '../engine/forward'
  import { layerNorm, norm, row, softmax } from '../engine/ops'
  import { fmt, pct } from '../lib/color'
  import { modelPy, section } from '../lib/source'
  import { app, candidates } from '../lib/state.svelte'
  import { mean } from '../lib/util'
  import Bars from '../viz/Bars.svelte'
  import Cells from '../viz/Cells.svelte'
  import Chips from '../viz/Chips.svelte'
  import Code from '../viz/Code.svelte'
  import Quiz from '../viz/Quiz.svelte'
  import Tex from '../viz/Tex.svelte'

  const model = app.model!
  const { d_model: D, n_head: H, n_layer: L } = model.config
  const trace = $derived(app.trace!)
  const T = $derived(trace.T)
  const t = $derived(app.pos)
  let hover = $state('')

  // --- Layer norm on the selected token's stream ---
  let layer = $state(L - 1)
  const raw = $derived(row(trace.layers[layer].residPre, t, D))
  const normed = $derived(layerNorm(raw.slice(), model.w[`blocks.${layer}.ln1.weight`], model.w[`blocks.${layer}.ln1.bias`], 1, D))
  const std = (x: ArrayLike<number>) => {
    const m = mean(x)
    return Math.sqrt(mean(Array.from(x, (v) => (v - m) ** 2)))
  }
  const stages = $derived([trace.embed, ...trace.layers.map((l) => l.residPost)])
  const stageName = (s: number) => (s === 0 ? 'Start (token + position)' : `After layer ${s}`)

  // --- The lanes: who reads whom, in every layer ---
  const colW = $derived(Math.max(30, Math.min(64, 760 / T)))
  const rowH = 74
  const strength = (l: number, i: number, j: number) => {
    let m = 0
    const a = trace.layers[l].attn
    for (let h = 0; h < H; h++) m = Math.max(m, a[(h * T + i) * T + j])
    return m
  }
  const THRESH = 0.25
  // Walk back from the selected token at the top layer: which earlier lanes fed it?
  const paths = $derived.by(() => {
    const edges: { l: number; i: number; j: number; w: number; hot: boolean }[] = []
    let live = new Set([t])
    const hot = new Set<string>()
    for (let l = L - 1; l >= 0; l--) {
      const nextLive = new Set(live)
      for (const i of live)
        for (let j = 0; j < i; j++)
          if (strength(l, i, j) >= THRESH) {
            hot.add(`${l}.${i}.${j}`)
            nextLive.add(j)
          }
      live = nextLive
    }
    for (let l = 0; l < L; l++)
      for (let i = 1; i < T; i++)
        for (let j = 0; j < i; j++) {
          const w = strength(l, i, j)
          if (w >= THRESH) edges.push({ l, i, j, w, hot: hot.has(`${l}.${i}.${j}`) })
        }
    return edges.sort((a, b) => Number(a.hot) - Number(b.hot))
  })
  const cx = (i: number) => 48 + colW * (i + 0.5)
  const cy = (s: number) => 46 + rowH * s

  // --- Reading the stream early ---
  const peek = $derived(stages.map((s) => lens(model, s, t)))
  const finalTop = $derived(candidates(peek[L], 1)[0])
  const finalProb = $derived(peek.map((lg) => softmax(lg)[finalTop.id]))
</script>

<p>
  You now have both halves of a layer: <span class="talk">talk</span> (attention) and <span class="think">think</span> (the MLP). Two small pieces of plumbing hold them together, and
  then the layer is simply repeated. This chapter assembles the whole thing.
</p>

<h2>Plumbing, part one: keep the numbers in range</h2>
<p>
  Every head and every MLP adds something to the stream, so the stream's numbers tend to grow and drift as they travel upwards. Parts that expect inputs of a certain size would be thrown
  off. So, just before the stream is read by attention or by the MLP, a copy of it is <strong>normalised</strong>: shifted so that its {D} numbers average zero, and scaled so that their
  spread is one. This is <strong>layer normalisation</strong>.
</p>

<div class="fig">
  <p class="fig-title">Normalising the stream of <span class="mono">{app.label(app.ids[t])}</span> before layer {layer + 1} reads it</p>
  <div class="controls">
    <label>Layer <select bind:value={layer}>{#each Array(L) as _, l}<option value={l}>{l + 1}</option>{/each}</select></label>
  </div>
  <Chips ids={app.ids} selected={t} onselect={(i) => (app.focus = i)} />
  <div class="scroll-x">
    <div class="sum">
      <div class="line"><span class="what">the stream itself</span><Cells data={raw} rows={1} cols={D} ch={24} max={3} label="Stream before normalising" onhover={(_, c, v) => (hover = `stream, dimension ${c}: ${fmt(v, 3)}`)} onleave={() => (hover = '')} /><span class="num">average {fmt(mean(raw))} · spread {std(raw).toFixed(2)}</span></div>
      <div class="line"><span class="what">the copy that is read</span><Cells data={normed} rows={1} cols={D} ch={24} max={3} label="Stream after normalising" onhover={(_, c, v) => (hover = `normalised, dimension ${c}: ${fmt(v, 3)}`)} onleave={() => (hover = '')} /><span class="num">average {fmt(mean(normed))} · spread {std(normed).toFixed(2)}</span></div>
    </div>
  </div>
  <p class="readout">{hover || 'Both rows use the same colour scale. Try layer 1 and the last layer: the stream grows, the normalised copy does not.'}</p>
  <p class="sub">Length of this token's stream as it moves through the model</p>
  <Bars name="Stream length by stage" items={stages.map((s, n) => ({ label: n === 0 ? 'start' : `layer ${n}`, value: norm(row(s, t, D)), text: norm(row(s, t, D)).toFixed(1) }))} />
</div>

<Tex block tex={'\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\text{mean}(x)}{\\sqrt{\\text{var}(x) + \\epsilon}} + \\beta'} />
<p>
  The mean and variance are taken over the {D} numbers of one token. <Tex tex={'\\gamma'} /> and <Tex tex={'\\beta'} /> are learned: a gain and an offset for each of the {D} dimensions, so
  the model can undo the normalisation where that helps (which is why the “spread” above is not exactly 1). <Tex tex={'\\epsilon'} /> is a tiny number that prevents division by zero.
</p>
<p>
  Notice what is normalised: the <em>copy</em> that gets read. The stream itself is left alone and keeps accumulating. This arrangement is called <strong>pre-norm</strong>. The original 2017
  transformer normalised the stream itself after each addition (post-norm); GPT-2 moved it to before, because deep stacks train much more reliably that way, and nearly every model since has
  followed.
</p>

<h2>Plumbing, part two: always add</h2>
<p>
  You met this in chapter 5: results are added to the stream, never written over it. These additions are called <strong>residual connections</strong>, and there is a second reason for
  them beyond keeping information alive. A model learns by sending a correction signal backwards from the output to every weight (<a href="#/training">chapter 9</a>). In a deep stack
  without these connections, that signal fades to nothing on the way down. The stream gives it a direct road from the top of the model to the bottom.
</p>

<h2>One layer, in full</h2>
<div class="fig">
  <svg viewBox="0 0 640 200" class="diagram" role="img" aria-label="One layer: the stream passes through, with two side branches. First branch: normalise, attention, add. Second branch: normalise, MLP, add.">
    <line x1="20" y1="150" x2="620" y2="150" class="stream" marker-end="url(#tip)" />
    <text x="20" y="176" class="lbl">stream in</text><text x="620" y="176" class="lbl" text-anchor="end">stream out, to the next layer</text>
    <path d="M90 150 V60 H132" class="branch" /><rect x="132" y="42" width="78" height="36" rx="8" class="box" /><text x="171" y="65" class="in">normalise</text>
    <line x1="210" y1="60" x2="232" y2="60" class="branch" /><rect x="232" y="42" width="84" height="36" rx="8" class="box talkbox" /><text x="274" y="65" class="in on">attention</text>
    <path d="M316 60 H340 V138" class="branch" /><circle cx="340" cy="150" r="12" class="plus" /><text x="340" y="155.5" class="in">+</text>
    <path d="M380 150 V60 H412" class="branch" /><rect x="412" y="42" width="78" height="36" rx="8" class="box" /><text x="451" y="65" class="in">normalise</text>
    <line x1="490" y1="60" x2="512" y2="60" class="branch" /><rect x="512" y="42" width="56" height="36" rx="8" class="box thinkbox" /><text x="540" y="65" class="in">MLP</text>
    <path d="M568 60 H590 V138" class="branch" /><circle cx="590" cy="150" r="12" class="plus" /><text x="590" y="155.5" class="in">+</text>
    <text x="224" y="26" class="lbl talk">talk: tokens exchange information</text><text x="451" y="26" class="lbl think">think: each token alone</text>
    <defs><marker id="tip" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8z" fill="#16213a" /></marker></defs>
  </svg>
</div>
<Code caption="model/model.py" code={section(modelPy, 'class Block', 'class GPT')} />
<p>Strip away the bookkeeping for this page and the layer is two lines:</p>
<Code code={`
x = x + self.attn(self.ln1(x))    # talk
x = x + self.mlp(self.ln2(x))     # think
`} />

<h2>Stack it</h2>
<p>
  The model in this page stacks {L} of these layers; GPT-2 stacks 12 to 48, and the largest models around a hundred. Every layer has the same shape but its own weights. Stacking is what
  lets tokens build on each other's work: in layer 1 a token can only read raw tokens, but in layer 2 it reads streams that already contain what layer 1 gathered, and so on up.
</p>

<div class="fig">
  <p class="fig-title">Your sentence in its lanes</p>
  <p class="fig-note">
    Each column is a token's stream, flowing down through the {L} layers. A curve from one lane to another means some head in that layer read strongly (over {THRESH * 100}%) from that
    earlier token. Squares are the MLPs. The <span class="talk">blue</span> curves trace everything that could have reached the selected token by the last layer. Click a token to trace it.
  </p>
  <div class="scroll-x">
    <svg width={48 + colW * T} height={cy(L) + 40} class="lanes" role="img" aria-label="Flow of information between token streams across layers">
      {#each paths as e}
        <path
          d="M{cx(e.j)} {cy(e.l)} C {cx(e.j)} {cy(e.l) + rowH * 0.3}, {cx(e.i)} {cy(e.l) + rowH * 0.18}, {cx(e.i)} {cy(e.l) + rowH * 0.48}"
          class:hot={e.hot}
          style:opacity={e.hot ? 0.35 + 0.65 * e.w : 0.5}
          style:stroke-width={e.hot ? 1 + 2.5 * e.w : 1}
        />
      {/each}
      {#each app.ids as id, i}
        <line x1={cx(i)} y1="30" x2={cx(i)} y2={cy(L) + (i === T - 1 ? 24 : 4)} class="lane" class:sel={i === t} />
        {#each Array(L) as _, l}
          <circle cx={cx(i)} cy={cy(l) + rowH * 0.48} r="3.5" class="dot" />
          <rect x={cx(i) - 5} y={cy(l) + rowH * 0.7} width="10" height="10" rx="2" class="sq" />
        {/each}
        <text x={cx(i)} y="18" class="tok" class:sel={i === t} role="button" tabindex="0" onclick={() => (app.focus = i)} onkeydown={(e) => e.key === 'Enter' && (app.focus = i)}>{app.label(id).slice(0, Math.floor(colW / 7))}</text>
      {/each}
      {#each Array(L) as _, l}<text x="4" y={cy(l) + rowH * 0.3} class="lay">layer {l + 1}</text>{/each}
      <text x={cx(T - 1)} y={cy(L) + 38} class="tok sel">vote</text>
    </svg>
  </div>
</div>

<p>
  This is the picture from above the chapter list, drawn with your sentence's real numbers. Only the last lane votes on the next token, but look at how much of the sentence can reach it:
  directly, and through other lanes that did their own reading in earlier layers.
</p>

<h2>Watching a prediction form</h2>
<p>
  The model only produces a prediction from the stream after the last layer. But since every layer merely adds to the same stream, nothing stops us from taking the stream earlier and running
  it through the final voting step anyway, to see what the model “would say so far”.
</p>

<div class="fig">
  <p class="fig-title">What follows <span class="mono">{app.label(app.ids[t])}</span>, read out after each layer</p>
  <Chips ids={app.ids} selected={t} onselect={(i) => (app.focus = i)} />
  <div class="scroll-x">
    <table class="plain peek">
      <thead><tr><th>Stream taken</th><th>Top three guesses at that point</th><th>Chance given to the final answer, <span class="mono">{app.label(finalTop.id)}</span></th></tr></thead>
      <tbody>
        {#each peek as lg, s}
          <tr class:final={s === L}>
            <td>{stageName(s)}</td>
            <td>{#each candidates(lg, 3) as c}<span class="guess" class:hit={c.id === finalTop.id}>{app.label(c.id)} <span class="num">{pct(c.p)}</span></span>{/each}</td>
            <td><span class="bar"><span style:width="{finalProb[s] * 100}%"></span></span> <span class="num">{pct(finalProb[s])}</span></td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>

<p>
  The first row nearly always guesses the token itself. That is weight tying at work: the stream starts out as the token's own embedding, and the vote asks which embedding the stream
  most resembles. From there the early rows are rough guesses, and the final answer gains ground layer by layer. Each layer makes an edit to
  the stream, and the prediction is the sum of the edits.
</p>
<div class="aside">
  <p>
    Read the early rows with care. The model was trained only to be read at the end, so earlier streams are not guaranteed to be in a readable form. This technique is known as the
    <em>logit lens</em>; it is a useful approximation, not a precise instrument.
  </p>
</div>

<h2>Check yourself</h2>
<Quiz
  question="In this model, what does layer normalisation change?"
  options={[
    { text: 'The stream itself, permanently', why: 'That is post-norm, the 2017 design. This model, like GPT-2, uses pre-norm.' },
    { text: 'Only the copy of the stream that attention or the MLP reads', correct: true, why: 'The stream keeps accumulating untouched. Each reader gets a freshly normalised copy.' },
    { text: 'The attention weights', why: 'Attention weights are normalised by softmax, a different mechanism.' },
  ]}
/>
<Quiz
  question="Why can a token in layer 3 know about a word it never attended to directly?"
  options={[
    { text: 'It cannot', why: 'It can, through an intermediate lane.' },
    { text: 'It attended to a token whose stream had already absorbed that word in an earlier layer', correct: true, why: 'Lane A reads lane B in layer 1; lane C reads lane A in layer 2. C now carries something of B. Depth lets information travel in several hops.' },
    { text: 'The MLP reads other lanes', why: 'The MLP never looks outside its own lane.' },
  ]}
/>

<style>
  select {
    font-family: var(--mono);
    border: 1.5px solid var(--line);
    border-radius: 7px;
    padding: 0.2rem 0.4rem;
    background: var(--paper);
  }
  .sum {
    display: grid;
    gap: 6px;
    margin-top: 1rem;
    width: max-content;
  }
  .line {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .what {
    width: 9.5rem;
    flex: none;
    text-align: right;
    font-size: 0.84rem;
    font-weight: 600;
  }
  .sub {
    font-weight: 620;
    margin: 1.2rem 0 0.5rem;
  }
  .diagram {
    width: 100%;
    max-width: 640px;
    display: block;
  }
  .stream {
    stroke: var(--ink);
    stroke-width: 4;
  }
  .branch {
    fill: none;
    stroke: var(--ink-3);
    stroke-width: 1.5;
  }
  .box {
    fill: #edf1f5;
    stroke: var(--ink-3);
  }
  .talkbox {
    fill: var(--talk);
    stroke: var(--talk);
  }
  .thinkbox {
    fill: var(--think-fill);
    stroke: var(--think-fill);
  }
  .plus {
    fill: var(--paper);
    stroke: var(--ink);
    stroke-width: 2;
  }
  .diagram text {
    font-family: var(--display);
    font-size: 13px;
  }
  .diagram .in {
    text-anchor: middle;
    font-weight: 600;
    fill: var(--ink);
  }
  .diagram .in.on {
    fill: #fff;
  }
  .diagram .lbl {
    font-size: 12px;
    fill: var(--ink-2);
  }
  .diagram .lbl.talk {
    fill: var(--talk);
    text-anchor: middle;
  }
  .diagram .lbl.think {
    fill: var(--think);
    text-anchor: middle;
  }
  .lanes path {
    fill: none;
    stroke: #b4bdcc;
  }
  .lanes path.hot {
    stroke: var(--talk);
  }
  .lanes .lane {
    stroke: var(--line);
    stroke-width: 2;
  }
  .lanes .lane.sel {
    stroke: var(--ink);
  }
  .lanes .dot {
    fill: var(--talk);
  }
  .lanes .sq {
    fill: var(--think-fill);
  }
  .lanes .tok {
    font-family: var(--mono);
    font-size: 11px;
    text-anchor: middle;
    fill: var(--ink-2);
    cursor: pointer;
  }
  .lanes .tok.sel {
    fill: var(--ink);
    font-weight: 700;
  }
  .lanes .lay {
    font-family: var(--display);
    font-size: 10.5px;
    fill: var(--ink-3);
  }
  .peek {
    margin-top: 1rem;
    min-width: 36rem;
  }
  .peek tr.final td {
    font-weight: 650;
  }
  .guess {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.8rem;
    background: #edf1f5;
    border-radius: 6px;
    padding: 0.12rem 0.4rem;
    margin-right: 0.3rem;
    white-space: pre;
  }
  .guess.hit {
    background: var(--think-soft);
    font-weight: 700;
  }
  .bar {
    display: inline-block;
    width: 9rem;
    height: 12px;
    background: #edf1f5;
    border-radius: 3px;
    vertical-align: middle;
    overflow: hidden;
  }
  .bar span {
    display: block;
    height: 100%;
    background: var(--ink);
  }
</style>
