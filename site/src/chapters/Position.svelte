<script lang="ts">
  import { forward, logitsAt, losses } from '../engine/forward'
  import { row } from '../engine/ops'
  import { fmt } from '../lib/color'
  import { app, candidates } from '../lib/state.svelte'
  import { cosine, mean } from '../lib/util'
  import Cells from '../viz/Cells.svelte'
  import Chips from '../viz/Chips.svelte'
  import Code from '../viz/Code.svelte'
  import Next from '../viz/Next.svelte'
  import Quiz from '../viz/Quiz.svelte'
  import Tex from '../viz/Tex.svelte'

  const model = app.model!
  const { d_model: D, vocab_size: V, block_size: B } = model.config
  const wpe = model.w['wpe.weight']
  const trace = $derived(app.trace!)
  const blind = $derived(forward(model, app.ids, { noPos: true }))
  const blindTop = $derived(candidates(logitsAt(blind, app.last, V), 1)[0]?.p ?? 0)
  const shared = $derived(Math.max(0.05, app.next(app.last, 1)[0]?.p ?? 0, blindTop))
  const withLoss = $derived(mean(losses(trace, V)))
  const blindLoss = $derived(mean(losses(blind, V)))

  let hover = $state('')
  const t = $derived(app.pos)

  // How alike are the position vectors of any two positions?
  const posSim = new Float32Array(B * B)
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) posSim[i * B + j] = cosine(row(wpe, i, D), row(wpe, j, D))
  let simHover = $state('')
</script>

<p>
  Each token now has a vector, but something has been lost. The table lookup gives <span class="mono">␣dog</span> the same vector wherever it appears. And the next stage, attention, treats
  its input as a bag of vectors: it has no built-in idea of which came first. Yet order is most of the meaning. “The dog chased the cat” and “the cat chased the dog” use the same words.
</p>
<p>See how much this model depends on knowing the order. Below, the same model is run twice on your sentence: normally, and with the position information removed.</p>

<div class="fig">
  <p class="fig-title">Your sentence, with and without position</p>
  <div class="two">
    <div>
      <p class="sub">With position</p>
      <Next items={app.next(app.last, 6)} max={shared} />
    </div>
    <div>
      <p class="sub">Position removed</p>
      <Next items={candidates(logitsAt(blind, app.last, V), 6)} color="var(--neg)" name="Most likely next tokens without position" max={shared} />
    </div>
  </div>
  {#if app.ids.length > 1}
    <p class="readout">
      Average surprise at each token of your sentence (lower is better): {withLoss.toFixed(2)} with position, {blindLoss.toFixed(2)} without.
    </p>
  {/if}
</div>

<div class="aside">
  <p>
    The model does not become completely blind to order. Each token can only look backwards (you will see why in the next chapter), so a token near the end sees more tokens than one near
    the start, and that leaks a little order information. But this model was trained with position vectors and leans on them heavily.
  </p>
</div>

<h2>The fix: add a position vector</h2>
<p>
  The model keeps a second, smaller table with one row per <em>position</em>: a vector for “first token”, one for “second token”, and so on up to position {B}, the longest input this
  model can read. The position vector is simply added, number by number, to the token vector.
</p>

<div class="fig">
  <p class="fig-title">Token vector + position vector</p>
  <Chips ids={app.ids} selected={t} onselect={(i) => (app.focus = i)} sub={(i) => `pos ${i}`} />
  <div class="sum scroll-x">
    <div class="line"><span class="what">token <span class="mono">{app.label(app.ids[t])}</span></span><Cells data={row(trace.tokEmb, t, D)} rows={1} cols={D} ch={26} max={0.6} label="Token vector" onhover={(_, c, v) => (hover = `token vector, dimension ${c}: ${fmt(v, 3)}`)} onleave={() => (hover = '')} /></div>
    <div class="line"><span class="what">+ position {t}</span><Cells data={row(trace.posEmb, t, D)} rows={1} cols={D} ch={26} max={0.6} label="Position vector" onhover={(_, c, v) => (hover = `position vector, dimension ${c}: ${fmt(v, 3)}`)} onleave={() => (hover = '')} /></div>
    <div class="line total"><span class="what">= what enters layer 1</span><Cells data={row(trace.embed, t, D)} rows={1} cols={D} ch={26} max={0.6} label="Sum of token and position vectors" onhover={(_, c, v) => (hover = `sum, dimension ${c}: ${fmt(v, 3)}`)} onleave={() => (hover = '')} /></div>
  </div>
  <p class="readout">{hover || 'All three rows use the same colour scale. Select another token to change the position.'}</p>
</div>

<p>
  Adding might look destructive: how can the model tell the token and the position apart afterwards? With {D} dimensions there is plenty of room. Training pushes token information and position
  information into different directions, and later stages can read whichever one they need with a dot product.
</p>

<h2>What the position table learned</h2>
<p>Like the token table, the position vectors started random. Compare every position's vector with every other's:</p>

<div class="fig">
  <p class="fig-title">How similar is position <em>i</em> to position <em>j</em>?</p>
  <p class="fig-note">Row and column are positions 0 to {B - 1}. <span style="color:var(--pos)">Teal</span> means the two position vectors point the same way.</p>
  <div class="scroll-x">
    <Cells data={posSim} rows={B} cols={B} cw={6} ch={6} max={1} label="Cosine similarity between position vectors" onhover={(r, c, v) => (simHover = `position ${r} and position ${c}: similarity ${fmt(v)}`)} onleave={() => (simHover = '')} />
  </div>
  <p class="readout">{simHover || 'Hover to compare two positions.'}</p>
</div>

<p>
  The teal band along the diagonal says that neighbouring positions have similar vectors, and positions far apart have different ones. Nobody built that in. The model worked out that
  position 7 should resemble position 8 more than position 40, because that is useful for predicting text.
</p>

<h2>Other ways to tell the model about order</h2>
<table class="plain">
  <thead><tr><th>Method</th><th>How it works</th><th>Used by</th></tr></thead>
  <tbody>
    <tr><td>Learned table</td><td>One trainable vector per position, added to the token vector. Cannot go past the table's length.</td><td>GPT-2, and this model</td></tr>
    <tr><td>Sinusoidal</td><td>The same idea, but the vectors are fixed sine and cosine waves of different wavelengths; nothing to train.</td><td>The original 2017 transformer</td></tr>
    <tr>
      <td>Rotary (RoPE)</td>
      <td>Nothing is added. Inside attention, vectors are rotated by an angle that depends on position, so what matters is how far apart two tokens are.</td>
      <td>Most current large models</td>
    </tr>
  </tbody>
</table>
<p>All three solve the same problem you just saw. <a href="#/scale">Chapter 12</a> returns to rotary positions.</p>

<h2>In maths and code</h2>
<Tex block tex={'x_t = E[\\,\\text{id}_t\\,] + P[\\,t\\,]'} />
<Code
  code={`
wte = nn.Embedding(4096, 128)     # token table
wpe = nn.Embedding(64, 128)       # position table: block_size rows

T = ids.shape[1]
x = wte(ids) + wpe(torch.arange(T))   # (1, T, 128); the position row is added to every sequence
`}
/>

<h2>Check yourself</h2>
<Quiz
  question="The word “dog” appears at position 2 and again at position 9. After this stage, are the two vectors the same?"
  options={[
    { text: 'Yes, same token, same vector', why: 'That was true after the lookup, but not after adding position.' },
    { text: 'No. Same token vector, but different position vectors were added', correct: true, why: 'The two lanes start out different, and from now on the model can tell them apart.' },
  ]}
/>
<Quiz
  question={`This model's position table has ${B} rows. What happens with a ${B + 1}-token input?`}
  options={[
    { text: 'It works, slightly less well', why: `There is simply no row for position ${B}.` },
    { text: 'It cannot be read; only the most recent tokens can be kept', correct: true, why: `A learned table has a hard limit, called the context length or block size. This page keeps the last ${B} tokens of your sentence for that reason.` },
  ]}
/>

<style>
  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
  }
  @media (max-width: 720px) {
    .two {
      grid-template-columns: 1fr;
    }
  }
  .sub {
    font-weight: 620;
    margin: 0 0 0.5rem;
  }
  .sum {
    margin-top: 1rem;
    display: grid;
    gap: 6px;
  }
  .line {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .what {
    width: 11rem;
    flex: none;
    text-align: right;
    font-size: 0.86rem;
    white-space: nowrap;
  }
  .line.total .what {
    font-weight: 700;
  }
</style>
