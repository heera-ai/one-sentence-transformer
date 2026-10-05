<script lang="ts">
  import { forward, logitsAt } from '../engine/forward'
  import { row, topK } from '../engine/ops'
  import { fmt } from '../lib/color'
  import { modelPy, section } from '../lib/source'
  import { app, candidates } from '../lib/state.svelte'
  import Cells from '../viz/Cells.svelte'
  import Chips from '../viz/Chips.svelte'
  import Code from '../viz/Code.svelte'
  import Next from '../viz/Next.svelte'
  import Quiz from '../viz/Quiz.svelte'
  import Tex from '../viz/Tex.svelte'

  const model = app.model!
  const { d_model: D, n_layer: L, vocab_size: V } = model.config
  const N = 4 * D // neurons per layer
  const COLS = 32
  let layer = $state(L - 1)
  let neuron = $state(-1)
  let hover = $state('')

  const trace = $derived(app.trace!)
  const t = $derived(app.pos)
  const lt = $derived(trace.layers[layer])
  const pre = $derived(row(lt.mlpPre, t, N))
  const act = $derived(row(lt.mlpAct, t, N))
  const out = $derived(row(lt.mlpOut, t, D))
  const busiest = $derived(topK(act, 8))
  const n = $derived(neuron < 0 ? busiest[0] : neuron)
  const activeCount = $derived(Array.from(act).filter((a) => a > 0.1).length)
  const negative = $derived(Math.round((Array.from(pre).filter((a) => a < 0).length / N) * 100))
  $effect(() => {
    layer
    app.pos
    app.ids
    neuron = -1
  })

  // Tokens this neuron pushes up when it fires: its output direction compared with every token's embedding.
  const pushes = $derived.by(() => {
    const W = model.w[`blocks.${layer}.mlp.proj.weight`] // [D, N]
    const dir = new Float32Array(D)
    for (let o = 0; o < D; o++) dir[o] = W[o * N + n]
    const wte = model.w['wte.weight']
    const scores = new Float32Array(V)
    for (let v = 0; v < V; v++) {
      let s = 0
      for (let d = 0; d < D; d++) s += wte[v * D + d] * dir[d]
      scores[v] = s
    }
    return topK(scores, 8)
  })
  const examples = $derived(app.extras?.neurons[layer][n] ?? [])

  // --- GELU ---
  const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
  const X0 = -4
  const X1 = 4
  const gx = (x: number) => ((x - X0) / (X1 - X0)) * 400
  const gy = (y: number) => 150 - ((y + 0.5) / 4.5) * 150
  const curve = Array.from({ length: 81 }, (_, i) => X0 + i * 0.1)
    .map((x, i) => `${i ? 'L' : 'M'}${gx(x).toFixed(1)} ${gy(gelu(x)).toFixed(1)}`)
    .join(' ')
  const relu = `M${gx(X0)} ${gy(0)} L${gx(0)} ${gy(0)} L${gx(X1)} ${gy(X1)}`
  const markX = $derived(Math.max(X0, Math.min(X1, pre[n])))

  // --- Skipping the MLP ---
  let skip = $state(false)
  const skipped = $derived(forward(model, app.ids, { ablateMlp: [layer] }))
</script>

<p>
  After attention, each token's stream holds a mixture: its own information plus what it gathered from other tokens. But gathering is all attention can do. A weighted average can move
  information around; it cannot work anything out from it. If one head brought “the subject is a dog” and another brought “the verb is missing”, something still has to combine the two into
  “a verb that dogs do comes next”.
</p>
<p>
  That is the job of the second half of every layer, the <strong class="think">MLP</strong> (multi-layer perceptron). It is where each token <strong>thinks alone</strong>: it looks only at its
  own stream, with no access to any other token.
</p>

<h2>Wide, bend, narrow</h2>
<p>The MLP does three things to a token's vector:</p>
<ol>
  <li><strong>Expand.</strong> A matrix turns the {D} numbers into {N}. Each of those {N} outputs is called a <strong>neuron</strong>.</li>
  <li><strong>Bend.</strong> Each neuron's number goes through a simple curve that keeps positives and nearly erases negatives.</li>
  <li><strong>Shrink.</strong> A second matrix turns the {N} numbers back into {D}, and the result is added to the stream.</li>
</ol>

<div class="fig">
  <p class="fig-title">The {N} neurons of layer {layer + 1}, for the token <span class="mono">{app.label(app.ids[t])}</span></p>
  <div class="controls">
    <label>Layer <select bind:value={layer}>{#each Array(L) as _, l}<option value={l}>{l + 1}</option>{/each}</select></label>
  </div>
  <Chips ids={app.ids} selected={t} onselect={(i) => (app.focus = i)} />
  <div class="two">
    <div>
      <p class="sub">Before the bend</p>
      <div class="scroll-x">
        <Cells data={pre} rows={N / COLS} cols={COLS} cw={11} ch={11} label="Neuron values before the activation function" markRow={Math.floor(n / COLS)} markCol={n % COLS} onhover={(r, c, v) => (hover = `neuron ${r * COLS + c}: ${fmt(v)} before the bend`)} onleave={() => (hover = '')} onpick={(r, c) => (neuron = r * COLS + c)} />
      </div>
    </div>
    <div>
      <p class="sub">After the bend</p>
      <div class="scroll-x">
        <Cells data={act} rows={N / COLS} cols={COLS} cw={11} ch={11} mode="think" label="Neuron values after the activation function" markRow={Math.floor(n / COLS)} markCol={n % COLS} onhover={(r, c, v) => (hover = `neuron ${r * COLS + c}: ${v.toFixed(2)} after the bend`)} onleave={() => (hover = '')} onpick={(r, c) => (neuron = r * COLS + c)} />
      </div>
    </div>
  </div>
  <p class="readout">{hover || `${activeCount} of ${N} neurons are clearly active (above 0.1) for this token. Click any neuron to inspect it.`}</p>
  <p class="sub">What the MLP then adds to this token's stream ({D} numbers)</p>
  <div class="scroll-x">
    <Cells data={out} rows={1} cols={D} cw={6} ch={22} label="MLP output" onhover={(_, c, v) => (hover = `MLP adds ${fmt(v, 3)} to dimension ${c}`)} onleave={() => (hover = '')} />
  </div>
</div>

<p>
  Compare the two grids. On the left, {negative}% of the neurons are negative (rose). On the right those have gone almost blank: for any one token, most neurons are switched off and a
  minority do the work. A different token lights up a different set. Select other tokens and watch the pattern change.
</p>

<h2>A neuron is a detector</h2>
<p>
  Each neuron owns one row of the first matrix, a vector of {D} weights, plus one extra number, its bias. Its value is the <strong>dot product</strong> of that row with the token's stream,
  plus the bias:
</p>
<Tex block tex={'\\text{neuron}_n = \\text{bend}\\,(\\,w_n \\cdot x + b_n\\,)'} />
<p>
  You know what a dot product measures: agreement in direction. So a neuron fires when the stream points the way its weights point, and stays silent otherwise. It is a detector for one
  pattern in the stream. When it fires, it adds its own fixed vector (a column of the second matrix) to the stream, scaled by how strongly it fired: “I found my pattern, so here is what
  follows from it.”
</p>

<div class="fig">
  <p class="fig-title">Layer {layer + 1}, neuron {n}</p>
  <div class="controls">
    <span>Most active for this token:</span>
    {#each busiest as b}
      <button class="pick" class:sel={b === n} onclick={() => (neuron = b)}>{b} <span class="num">{act[b].toFixed(1)}</span></button>
    {/each}
  </div>
  <p>On <span class="mono">{app.label(app.ids[t])}</span> it reads <span class="num">{fmt(pre[n])}</span> before the bend and <span class="num">{act[n].toFixed(2)}</span> after.</p>
  <div class="two">
    <div>
      <p class="sub">Where it fired hardest in real stories</p>
      {#if examples.length}
        {#each examples as ex}
          <p class="ex">
            <span class="num">{ex.a.toFixed(1)}</span>
            <span class="mono">{app.tok!.decode(ex.ctx.slice(0, -1)).replace(/\n/g, ' ')}<mark>{app.tok!.text(ex.ctx[ex.ctx.length - 1]).replace(/\n/g, '⏎')}</mark></span>
          </p>
        {/each}
        <p class="fig-note">From 192,000 tokens of stories the model was not trained on. The highlighted token is where the neuron fired.</p>
      {:else}<p class="muted">Loading examples…</p>{/if}
    </div>
    <div>
      <p class="sub">Tokens it pushes towards when it fires</p>
      <p class="mono pushes">{#each pushes as id}<span>{app.label(id)}</span>{/each}</p>
      <p class="fig-note">
        Found by comparing the neuron's output vector with every token's embedding. This is the neuron's direct effect on the final vote; later layers can also react to it, so it is most
        reliable in the last layer.
      </p>
    </div>
  </div>
</div>

<p>
  Some neurons are easy to read: they fire on names, or after “was”, or at the end of a sentence. Many are not. A neuron often responds to several unrelated things, because the model has
  more patterns to track than it has neurons and so packs several into each. Do not expect every neuron to make sense.
</p>

<h2>Why the bend matters</h2>
<p>
  The curve in step 2 is the smallest part of the model and the one it could least do without. Without it, expand-then-shrink would be two matrix multiplications in a row, and two matrix
  multiplications always collapse into one. The MLP could then only stretch and rotate the stream; it could never respond to one pattern and ignore another. The bend is what lets a
  neuron say “only if”, and “only if” is what computing something new requires.
</p>

<div class="fig">
  <p class="fig-title">The bend: GELU</p>
  <div class="gelu">
    <svg viewBox="-10 -10 420 175" role="img" aria-label="Graph of the GELU function compared with ReLU">
      <line x1="0" x2="400" y1={gy(0)} y2={gy(0)} class="axis" /><line x1={gx(0)} x2={gx(0)} y1="0" y2="150" class="axis" />
      <path d={relu} class="relu" /><path d={curve} class="curve" />
      <circle cx={gx(markX)} cy={gy(gelu(markX))} r="5" class="mark" />
      <text x="396" y={gy(0) + 14} text-anchor="end">input</text><text x={gx(0) + 6} y="10">output</text>
    </svg>
    <p>
      The solid curve is GELU, used by this model and by GPT-2. Large positive inputs pass through almost unchanged; negative inputs are pushed to almost zero. The dashed line is the
      simpler ReLU, “negative becomes zero”, which does the same job with a sharp corner. The dot is neuron {n} on your selected token: in
      <span class="num">{fmt(pre[n])}</span>, out <span class="num">{act[n].toFixed(2)}</span>.
    </p>
  </div>
</div>

<h2>What happens without it</h2>
<div class="fig">
  <p class="fig-title">Skip layer {layer + 1}'s MLP</p>
  <div class="controls">
    <label><input type="checkbox" bind:checked={skip} /> Skip the MLP in layer {layer + 1}</label>
    <label>Layer <select bind:value={layer}>{#each Array(L) as _, l}<option value={l}>{l + 1}</option>{/each}</select></label>
  </div>
  <Next items={skip ? candidates(logitsAt(skipped, app.last, V), 6) : app.next(app.last, 6)} color={skip ? 'var(--think-fill)' : 'var(--ink)'} mark={app.next(app.last, 1)[0]?.id} />
  <p class="readout">{skip ? 'The highlighted row is the normal top prediction.' : 'Tick the box to remove this MLP and re-run the model.'}</p>
</div>

<p>
  The MLPs are also where most of the weights live. In each layer, the four attention matrices hold 4 × {D}² = {(4 * D * D).toLocaleString('en')} weights, and the two MLP matrices hold 8
  × {D}² = {(8 * D * D).toLocaleString('en')}: twice as many. A common, simplified way to think about the division of labour is that attention decides <em>what to bring together</em>, and
  the MLPs hold much of <em>what the model knows</em>.
</p>

<h2>In code</h2>
<Code caption="model/model.py" code={section(modelPy, 'class MLP', 'class Block')} />
<p>
  The factor of 4 between {D} and {N} is a convention from the original transformer paper that has stuck. Note that <code>forward</code> receives all tokens at once, yet no line mixes them:
  every operation works on each token's vector separately.
</p>

<h2>Check yourself</h2>
<Quiz
  question="Inside the MLP, how much does one token learn about the others?"
  options={[
    { text: 'Nothing. Each token is processed alone', correct: true, why: 'Only attention moves information between tokens. The MLP works on whatever attention has already brought into this token’s stream.' },
    { text: 'It sees the tokens before it', why: 'That describes attention. The MLP has no access to other lanes at all.' },
    { text: 'It sees the whole sentence', why: 'No stage sees later tokens, and the MLP sees no other token at all.' },
  ]}
/>
<Quiz
  question="Remove the bend (GELU) from every MLP. What is the MLP now?"
  options={[
    { text: 'The same, but a bit less accurate', why: 'It is a much bigger loss than that.' },
    { text: 'A single matrix multiplication, however many layers it had', correct: true, why: 'Multiplying by one matrix and then another equals multiplying by their product. Without something non-linear in between, extra layers add nothing.' },
    { text: 'It stops working because of negative numbers', why: 'Negative numbers are fine. The loss is the ability to express anything but straight-line relationships.' },
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
  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-top: 0.8rem;
  }
  @media (max-width: 860px) {
    .two {
      grid-template-columns: 1fr;
    }
  }
  .sub {
    font-weight: 620;
    margin: 0.6rem 0 0.4rem;
  }
  .pick {
    font-family: var(--mono);
    font-size: 0.8rem;
    background: var(--think-soft);
    border: 1.5px solid transparent;
    border-radius: 7px;
    padding: 0.2rem 0.5rem;
  }
  .pick.sel {
    border-color: var(--ink);
  }
  .ex {
    display: grid;
    grid-template-columns: 2.6rem 1fr;
    gap: 0.5rem;
    margin: 0 0 0.35rem;
    font-size: 0.8rem;
    align-items: baseline;
  }
  .ex .mono {
    white-space: pre-wrap;
  }
  mark {
    background: var(--think-fill);
    border-radius: 3px;
    font-weight: 700;
    color: var(--ink);
  }
  .pushes {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .pushes span {
    background: #edf1f5;
    border-radius: 6px;
    padding: 0.2rem 0.45rem;
    white-space: pre;
  }
  .gelu {
    display: grid;
    grid-template-columns: minmax(240px, 420px) 1fr;
    gap: 1.5rem;
    align-items: center;
  }
  @media (max-width: 720px) {
    .gelu {
      grid-template-columns: 1fr;
    }
  }
  .gelu svg text {
    font-size: 11px;
    fill: var(--ink-2);
    font-family: var(--display);
  }
  .axis {
    stroke: var(--line);
  }
  .curve {
    fill: none;
    stroke: var(--think);
    stroke-width: 2.5;
  }
  .relu {
    fill: none;
    stroke: var(--ink-3);
    stroke-dasharray: 4 4;
    stroke-width: 1.5;
  }
  .mark {
    fill: var(--ink);
  }
</style>
