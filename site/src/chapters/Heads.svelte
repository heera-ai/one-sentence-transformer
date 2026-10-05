<script lang="ts">
  import { forward, headWrite, logitsAt } from '../engine/forward'
  import { norm, row } from '../engine/ops'
  import { fmt, pct, talk } from '../lib/color'
  import { modelPy, section } from '../lib/source'
  import { app, candidates } from '../lib/state.svelte'
  import Cells from '../viz/Cells.svelte'
  import Chips from '../viz/Chips.svelte'
  import Code from '../viz/Code.svelte'
  import Next from '../viz/Next.svelte'
  import Quiz from '../viz/Quiz.svelte'

  const model = app.model!
  const { d_model: D, n_head: H, n_layer: L, vocab_size: V } = model.config
  const dHead = D / H
  let layer = $state(0)
  let head = $state(0)

  const trace = $derived(app.trace!)
  const T = $derived(trace.T)
  const i = $derived(app.pos)
  const size = $derived(Math.max(4, Math.min(11, Math.floor(132 / T))))
  const pat = (l: number, h: number) => trace.layers[l].attn.subarray(h * T * T, (h + 1) * T * T)
  const weights = $derived(pat(layer, head).subarray(i * T, (i + 1) * T))

  // Describe each head by what it does on this sentence, measured rather than assumed.
  function habit(l: number, h: number): string {
    if (T < 3) return ''
    const a = pat(l, h)
    let prev = 0
    let self = 0
    let first = 0
    for (let r = 1; r < T; r++) {
      prev += a[r * T + r - 1]
      self += a[r * T + r]
      first += a[r * T]
    }
    const n = T - 1
    const best = Math.max(prev, self, first) / n
    if (best < 0.3) return 'spread out'
    const what = prev >= self && prev >= first ? 'previous token' : self >= first ? 'itself' : 'first token'
    return `${what} ${Math.round(best * 100)}%`
  }
  // The head that most consistently reads the token just before the reader.
  const prevHead = $derived.by(() => {
    let best = { l: 0, h: 0, w: 0 }
    if (T < 3) return best
    for (let l = 0; l < L; l++)
      for (let h = 0; h < H; h++) {
        const a = pat(l, h)
        let w = 0
        for (let r = 1; r < T; r++) w += a[r * T + r - 1]
        w /= T - 1
        if (w > best.w) best = { l, h, w }
      }
    return best
  })

  // --- What each head writes into the stream ---
  let hover = $state('')
  const writes = $derived(Array.from({ length: H }, (_, h) => headWrite(model, trace, layer, h, i)))
  const before = $derived(row(trace.layers[layer].residPre, i, D))
  const after = $derived(row(trace.layers[layer].residMid, i, D))
  const scale = $derived(Math.max(...Array.from(after, Math.abs), 1e-6))

  // --- Silencing heads ---
  let off = $state<string[]>([])
  const key = (l: number, h: number) => `${l}.${h}`
  const toggle = (l: number, h: number) => (off = off.includes(key(l, h)) ? off.filter((x) => x !== key(l, h)) : [...off, key(l, h)])
  const ablated = $derived(forward(model, app.ids, { ablateHeads: off.map((s) => s.split('.').map(Number) as [number, number]) }))
  const normal = $derived(app.next(app.last, 6))
  const shared = $derived(Math.max(0.05, normal[0]?.p ?? 0, candidates(logitsAt(ablated, app.last, V), 1)[0]?.p ?? 0))

  let effects = $state<number[] | null>(null)
  $effect(() => {
    app.ids
    effects = null
  })
  function testAll() {
    const top = normal[0]
    const out: number[] = []
    for (let l = 0; l < L; l++)
      for (let h = 0; h < H; h++) {
        const t = forward(model, app.ids, { ablateHeads: [[l, h]], lastOnly: true })
        const p = candidates(logitsAt(t, app.last, V), V).find((c) => c.id === top.id)!.p
        out.push(p - top.p)
      }
    effects = out
  }
  const worst = $derived(effects ? Math.max(...effects.map(Math.abs), 1e-6) : 1)
</script>

<p>
  One attention head can look for one kind of thing at a time. But a token may need several things at once: the word just before it, the subject of the sentence, whether a quotation is
  open. So a layer runs several heads <strong>in parallel</strong>, each with its own query, key and value matrices, each free to look for something different.
</p>
<p>
  This model has {H} heads in each of its {L} layers, so {L * H} in all. To keep the cost the same as one big head, each head works with small vectors: {D} ÷ {H} = {dHead} numbers.
</p>

<div class="fig">
  <p class="fig-title">All {L * H} attention patterns for your sentence</p>
  <p class="fig-note">
    Each square is one head: row = the token reading, column = the token being read. The caption says where the head puts most of its weight on this sentence, on average. Click a head to
    inspect it.
  </p>
  <div class="scroll-x">
    <div class="grid" style:grid-template-columns="auto repeat({H}, max-content)">
      {#each Array(L) as _, l}
        <span class="lname">Layer {l + 1}</span>
        {#each Array(H) as _, h}
          <button class="mini" class:sel={l === layer && h === head} onclick={() => ((layer = l), (head = h))} aria-label="Layer {l + 1}, head {h + 1}: {habit(l, h)}">
            <Cells data={pat(l, h)} rows={T} cols={T} cw={size} ch={size} gap={T > 20 ? 0 : 1} mode="talk" max={1} label="" />
            <span class="cap">{habit(l, h) || `head ${h + 1}`}</span>
          </button>
        {/each}
      {/each}
    </div>
  </div>
  <p class="sub">Layer {layer + 1}, head {head + 1}: what <span class="mono">{app.label(app.ids[i])}</span> reads</p>
  <Chips ids={app.ids} selected={i} onselect={(n) => (app.focus = n)} weights={weights} tint={talk} dim={(n) => n > i} sub={(n) => (n <= i ? pct(weights[n]) : '')} />
</div>

<p>
  Three shapes are worth learning to spot. A stripe just below the main diagonal is a head that reads the <strong>previous token</strong>. A dark main diagonal is a head that reads
  <strong>itself</strong>. A dark first column is a head resting on the <strong>first token</strong>.
  {#if prevHead.w > 0.3}
    On your sentence the clearest previous-token head is layer {prevHead.l + 1}, head {prevHead.h + 1}, which puts {Math.round(prevHead.w * 100)}% of its weight there on average.
  {/if}
</p>
<p>
  In large models, researchers have found many heads with crisp jobs that can be named. In a model this small the heads are softer: most mix several habits and shift from sentence to
  sentence. Edit your sentence and watch which heads react.
</p>
<div class="aside">
  <p>
    A caution. A pattern shows <em>where</em> a head reads from. It does not show <em>what</em> the head does with what it read; that depends on the value and output matrices. Reading
    intentions into attention pictures is a well-known trap.
  </p>
</div>

<h2>The stream</h2>
<p>
  What happens to the result of each head? The four heads' results are placed side by side to make one vector of {D} numbers, passed through one more matrix, and then
  <strong>added</strong> to the token's vector. Added, not swapped in. This matters enough to deserve a name.
</p>
<p>
  From here on, think of each token's lane as a <strong>stream</strong> (researchers say <em>residual stream</em>): a running vector that starts as token + position and flows up through the
  model. Nothing ever replaces it. Every head and every MLP reads from the stream and <strong>writes an addition</strong> back into it.
</p>

<div class="fig">
  <p class="fig-title">What layer {layer + 1}'s heads add to the stream of <span class="mono">{app.label(app.ids[i])}</span></p>
  <div class="controls">
    <label>Layer <select bind:value={layer}>{#each Array(L) as _, l}<option value={l}>{l + 1}</option>{/each}</select></label>
    <span class="muted">All rows share one colour scale. The size is the vector's length.</span>
  </div>
  <div class="scroll-x">
    <div class="sum">
      <div class="line"><span class="what">stream before</span><Cells data={before} rows={1} cols={D} ch={22} max={scale} label="Stream before attention" onhover={(_, c, x) => (hover = `stream before, dimension ${c}: ${fmt(x, 3)}`)} onleave={() => (hover = '')} /><span class="num">{norm(before).toFixed(2)}</span></div>
      {#each writes as w, h}
        <div class="line"><span class="what talk">+ head {h + 1}</span><Cells data={w} rows={1} cols={D} ch={22} max={scale} label="Head {h + 1} write" onhover={(_, c, x) => (hover = `head ${h + 1} adds, dimension ${c}: ${fmt(x, 3)}`)} onleave={() => (hover = '')} /><span class="num">{norm(w).toFixed(2)}</span></div>
      {/each}
      <div class="line total"><span class="what">= stream after</span><Cells data={after} rows={1} cols={D} ch={22} max={scale} label="Stream after attention" onhover={(_, c, x) => (hover = `stream after, dimension ${c}: ${fmt(x, 3)}`)} onleave={() => (hover = '')} /><span class="num">{norm(after).toFixed(2)}</span></div>
    </div>
  </div>
  <p class="readout">{hover || 'A small constant offset (the bias) is also added; it is left out of the picture.'}</p>
</div>

<p>Adding has two big consequences, and both come back later:</p>
<ul>
  <li>
    <strong>Information survives.</strong> What the token and position vectors wrote at the start is still in the stream at the top, unless something deliberately cancels it. Later layers
    can read what any earlier layer wrote.
  </li>
  <li>
    <strong>Each part's contribution can be measured.</strong> Because the final stream is a sum, you can ask “how much did this one head change the answer?” Do that now.
  </li>
</ul>

<h2>Switch heads off</h2>
<p>Silence a head, so that it adds nothing, and the model is re-run on your sentence. If a head matters for this prediction, the answer moves.</p>

<div class="fig">
  <p class="fig-title">Silence heads and compare the prediction</p>
  <div class="scroll-x">
    <div class="grid switches" style:grid-template-columns="auto repeat({H}, max-content)">
      {#each Array(L) as _, l}
        <span class="lname">Layer {l + 1}</span>
        {#each Array(H) as _, h}
          {@const e = effects?.[l * H + h]}
          <button class="sw" class:off={off.includes(key(l, h))} aria-pressed={off.includes(key(l, h))} onclick={() => toggle(l, h)} style:--e={e === undefined ? 0 : Math.abs(e) / worst}>
            head {h + 1}
            {#if e !== undefined}<span class="num">{e >= 0 ? '+' : '−'}{Math.abs(e * 100).toFixed(1)}</span>{/if}
          </button>
        {/each}
      {/each}
    </div>
  </div>
  <div class="controls" style="margin-top:.9rem">
    <button class="btn" onclick={testAll}>Test each head on its own</button>
    <button class="btn quiet" disabled={!off.length} onclick={() => (off = [])}>Switch all back on</button>
  </div>
  {#if effects}
    <p class="fig-note">
      The number on each head is how many percentage points the probability of <span class="mono">{app.label(normal[0].id)}</span> moves when that head alone is silenced. Darker means a
      bigger effect.
    </p>
  {/if}
  <div class="two">
    <div>
      <p class="sub">All heads on</p>
      <Next items={normal} max={shared} />
    </div>
    <div>
      <p class="sub">{off.length ? `${off.length} silenced` : 'Silence a head to compare'}</p>
      <Next items={candidates(logitsAt(ablated, app.last, V), 6)} color="var(--talk)" mark={normal[0]?.id} name="Next tokens with heads silenced" max={shared} />
    </div>
  </div>
</div>

<p>
  You will usually find that a few heads matter a great deal for a given prediction and most barely matter. Which ones matter changes with the sentence. This kind of experiment, removing a
  part and measuring the damage, is the basic tool of the research field that tries to work out what the parts of a trained model do.
</p>

<h2>In maths and code</h2>
<p>
  In code, nobody loops over heads. One matrix of width {D} produces all the queries at once; the result is then cut into {H} slices of {dHead}, and the attention formula runs on all slices
  in a single batched operation. This is the attention module of the model in this page, exactly as trained:
</p>
<Code caption="model/model.py" code={section(modelPy, 'class Attention', 'class MLP')} />
<p>
  Read <code>split</code> first: it reshapes <code>(B, T, {D})</code> into <code>(B, {H}, T, {dHead})</code>, so that everything after it treats the heads as a batch. The last
  <code>reshape</code> puts the heads back side by side, and <code>self.out</code> is the matrix that writes into the stream. (<code>trace</code> and <code>ablate_heads</code> exist only
  to feed this page.)
</p>

<h2>Check yourself</h2>
<Quiz
  question={`Why does each head use ${dHead}-number vectors and not ${D}?`}
  options={[
    { text: 'Smaller vectors are more accurate', why: 'They are not; it is a budget decision.' },
    { text: `So that ${H} heads together cost about the same as one head of ${D}`, correct: true, why: `${H} × ${dHead} = ${D}. The layer trades one wide search for several narrow ones run in parallel.` },
    { text: 'The embedding table requires it', why: `The embedding size is ${D}; the head size is a separate choice.` },
  ]}
/>
<Quiz
  question="A head's output is added to the stream. What would be lost if it replaced the stream?"
  options={[
    { text: 'Nothing important', why: 'Quite a lot would be lost.' },
    { text: 'Everything earlier stages wrote, including which token this is', correct: true, why: 'Adding keeps the token, its position and every earlier layer’s work available to everything that comes later.' },
    { text: 'Only the position information', why: 'Position would go, but so would the token itself and all earlier results.' },
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
  .grid {
    display: grid;
    gap: 8px 10px;
    align-items: center;
    width: max-content;
  }
  .lname {
    font-size: 0.8rem;
    font-weight: 600;
    padding-right: 0.4rem;
    white-space: nowrap;
  }
  .mini {
    background: none;
    border: 1.5px solid var(--line-soft);
    border-radius: 8px;
    padding: 6px;
    display: grid;
    gap: 4px;
    justify-items: center;
  }
  .mini:hover {
    border-color: var(--ink-3);
  }
  .mini.sel {
    border-color: var(--ink);
    box-shadow: 0 2px 0 var(--ink);
  }
  .cap {
    font-size: 0.68rem;
    color: var(--ink-2);
    white-space: nowrap;
  }
  .sub {
    font-weight: 620;
    margin: 1.2rem 0 0.5rem;
  }
  .sum {
    display: grid;
    gap: 5px;
    width: max-content;
  }
  .line {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .what {
    width: 7.5rem;
    flex: none;
    text-align: right;
    font-size: 0.84rem;
    font-weight: 600;
  }
  .line.total .what {
    font-weight: 750;
  }
  .sw {
    font-family: var(--display);
    font-size: 0.82rem;
    font-weight: 600;
    border: 1.5px solid var(--talk);
    color: var(--ink);
    background: color-mix(in srgb, var(--talk) calc(var(--e) * 45%), white);
    border-radius: 8px;
    padding: 0.4rem 0.6rem;
    min-width: 5.6rem;
    display: grid;
    gap: 1px;
  }
  .sw.off {
    border-style: dashed;
    border-color: var(--ink-3);
    color: var(--ink-3);
    background: #edf1f5;
    text-decoration: line-through;
  }
  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-top: 0.5rem;
  }
  @media (max-width: 720px) {
    .two {
      grid-template-columns: 1fr;
    }
  }
</style>
