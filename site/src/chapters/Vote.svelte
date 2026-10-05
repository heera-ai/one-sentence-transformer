<script lang="ts">
  import { forward, logitsAt } from '../engine/forward'
  import { row, softmax, topK } from '../engine/ops'
  import { distribution, draw } from '../engine/sample'
  import { fmt, pct } from '../lib/color'
  import { modelPy, section } from '../lib/source'
  import { app } from '../lib/state.svelte'
  import Bars from '../viz/Bars.svelte'
  import Cells from '../viz/Cells.svelte'
  import Code from '../viz/Code.svelte'
  import Quiz from '../viz/Quiz.svelte'
  import Tex from '../viz/Tex.svelte'

  const model = app.model!
  const { d_model: D, vocab_size: V, block_size: B } = model.config
  const trace = $derived(app.trace!)
  const final = $derived(row(trace.final, app.last, D))
  const logits = $derived(logitsAt(trace, app.last, V))
  const order = $derived(topK(logits, V))
  let hover = $state('')

  let temperature = $state(1)
  let k = $state(0) // 0 = off
  let p = $state(1)
  const opts = $derived({ temperature, topK: k || undefined, topP: p })
  const base = $derived(softmax(logits))
  const dist = $derived(distribution(logits, opts))
  const kept = $derived(Array.from(dist).filter((x) => x > 0).length)
  // "How many equally likely options is this like?"
  const choices = $derived(Math.exp(-Array.from(dist).reduce((s, x) => s + (x > 0 ? x * Math.log(x) : 0), 0)))

  // --- Writing ---
  let written = $state<number[]>([])
  let busy = $state(false)
  $effect(() => {
    app.ids
    written = []
  })
  function step(): boolean {
    const ids = [...app.ids, ...written].slice(-B)
    const t = forward(model, ids, { lastOnly: true })
    const id = draw(distribution(logitsAt(t, ids.length - 1, V), opts))
    if (id === app.tok!.eot) return false
    written = [...written, id]
    return true
  }
  async function write(n: number) {
    busy = true
    for (let i = 0; i < n && busy; i++) {
      if (!step()) break
      await new Promise((r) => setTimeout(r, 45))
    }
    busy = false
  }
</script>

<p>
  After the last layer, the model holds one finished stream per token. To predict what comes after your sentence, only one of them is needed: the <strong>last</strong> lane. Through
  attention, it has had the chance to collect whatever it needed from all the others.
</p>

<h2>From a vector to {V.toLocaleString('en')} scores</h2>
<p>
  The last lane is {D} numbers. The answer has to be one score per token in the vocabulary. The step between them is the operation you have used all along: after one final layer
  normalisation, take the <strong>dot product</strong> of the stream with a vector for each token. Tokens whose vector points the same way as the stream get a high score. These raw scores
  are called <strong>logits</strong>.
</p>
<p>
  Which vectors? In this model, as in GPT-2, they are the rows of the <em>embedding table</em> from chapter 2, used a second time. This is called <strong>weight tying</strong>. So the
  whole model can be read as: turn tokens into vectors, edit those vectors for {model.config.n_layer} layers, then ask “which token's vector does the result now look like?”
</p>

<div class="fig">
  <p class="fig-title">The last lane, and the scores it produces</p>
  <div class="scroll-x">
    <Cells data={final} rows={1} cols={D} cw={6} ch={30} label="Final vector of the last token" onhover={(_, c, v) => (hover = `dimension ${c}: ${fmt(v, 3)}`)} onleave={() => (hover = '')} />
  </div>
  <p class="readout">{hover || `The finished stream of “${app.label(app.ids[app.last])}”, after the final layer norm.`}</p>
  <div class="two">
    <div>
      <p class="sub">Highest scores</p>
      <Bars name="Highest logits" signed items={order.slice(0, 7).map((id) => ({ label: app.label(id), value: logits[id], text: fmt(logits[id], 1) }))} />
    </div>
    <div>
      <p class="sub">Lowest scores</p>
      <Bars name="Lowest logits" signed max={Math.abs(logits[order[0]])} items={order.slice(-5).map((id) => ({ label: app.label(id), value: logits[id], text: fmt(logits[id], 1) }))} />
    </div>
  </div>
</div>

<h2>From scores to probabilities</h2>
<p>
  Scores can be any size, positive or negative. Probabilities must be positive and add up to 1. You already know the tool that makes that conversion: <strong>softmax</strong>, from chapter
  4. There it turned match scores into attention weights; here it turns logits into the probability of each next token.
</p>
<p>
  With one addition. Before softmax, every score is divided by a number called the <strong>temperature</strong>. It is not part of the model; it is a dial for whoever runs it.
</p>
<Tex block tex={'p_i = \\frac{e^{\\,z_i / T}}{\\sum_j e^{\\,z_j / T}}'} />

<div class="fig">
  <p class="fig-title">Shape the probabilities</p>
  <div class="dials">
    <label>Temperature <input type="range" min="0" max="2" step="0.05" bind:value={temperature} /> <span class="num">{temperature.toFixed(2)}</span></label>
    <label>Top-k <input type="range" min="0" max="50" step="1" bind:value={k} /> <span class="num">{k === 0 ? 'off' : k}</span></label>
    <label>Top-p <input type="range" min="0.05" max="1" step="0.05" bind:value={p} /> <span class="num">{p >= 1 ? 'off' : p.toFixed(2)}</span></label>
    <button class="btn quiet" onclick={() => ((temperature = 1), (k = 0), (p = 1))}>Reset</button>
  </div>
  <Bars
    name="Probability of each next token"
    max={Math.max(0.05, dist[order[0]], base[order[0]])}
    items={order.slice(0, 14).map((id) => ({ label: app.label(id), value: dist[id], text: pct(dist[id]), faded: dist[id] === 0, note: dist[id] === 0 ? 'cut' : `was ${pct(base[id])}` }))}
  />
  <p class="readout">
    {kept.toLocaleString('en')} of {V.toLocaleString('en')} tokens can be chosen. The spread is like choosing between {choices.toFixed(1)} equally likely options.
  </p>
</div>

<ul>
  <li>
    <strong>Temperature below 1</strong> stretches the gaps between scores, so the leader takes more. At 0 the top token is always chosen (called <em>greedy</em>). Safe, but repetitive.
  </li>
  <li><strong>Temperature above 1</strong> shrinks the gaps, so unlikely tokens get a real chance. More surprising, and sooner or later, nonsense.</li>
  <li>
    <strong>Top-k</strong> keeps only the <em>k</em> most likely tokens and discards the rest. <strong>Top-p</strong> keeps the smallest group of top tokens whose probabilities add up to
    <em>p</em>, so it keeps few when the model is sure and many when it is not. Both exist to cut off the long tail of thousands of very unlikely tokens, which together are likely enough
    to derail a text now and then.
  </li>
</ul>

<h2>Pick one, add it, go again</h2>
<p>
  The last step is not arithmetic at all: one token is <strong>drawn at random</strong> according to those probabilities, like spinning a wheel where each token's slice is as wide as its
  probability. The token is added to the end of the text, and the whole model runs again on the longer text. You saw this loop in chapter 0. Now you know everything that happens inside
  each turn of it.
</p>

<div class="fig">
  <p class="fig-title">Let it write</p>
  <p class="fig-note">Uses the dials above. Run it twice with the same settings and you get different stories, because of the random draw. Set temperature to 0 and it is the same every time.</p>
  <p class="story"><span>{app.tok!.decode(app.ids)}</span><mark>{app.tok!.decode(written)}</mark></p>
  <div class="controls" style="margin-bottom:0">
    <button class="btn" disabled={busy} onclick={() => write(1)}>Write one token</button>
    <button class="btn" disabled={busy} onclick={() => write(40)}>Write 40</button>
    {#if busy}<button class="btn quiet" onclick={() => (busy = false)}>Stop</button>{/if}
    <button class="btn quiet" disabled={busy || !written.length} onclick={() => (written = [])}>Clear</button>
    <button class="btn quiet" disabled={busy || !written.length} onclick={() => app.setText(app.text + app.tok!.decode(written))}>Make this my sentence</button>
  </div>
</div>

<p>Three practical details of the loop:</p>
<ul>
  <li>
    <strong>Knowing when to stop.</strong> The training stories were separated by a special end-of-text token. The model learned to predict it where a story should end; when it is drawn,
    writing stops.
  </li>
  <li>
    <strong>Running out of room.</strong> This model can read {B} tokens. Once the text is longer, only the most recent {B} are fed in, and it forgets the beginning. The limit is called the
    <strong>context window</strong>; in large models it runs from a hundred thousand tokens to over a million.
  </li>
  <li>
    <strong>Not repeating work.</strong> Because of the causal mask, adding a token changes nothing in the earlier lanes. Real systems therefore save each lane's keys and values and compute
    only the new lane on each turn. This is the <strong>KV cache</strong>, and it is why a chatbot's reply streams out quickly even though each token formally needs the whole text.
  </li>
</ul>

<h2>In code</h2>
<Code caption="model/model.py" code={section(modelPy, '@torch.no_grad()', 'def num_params')} />
<p>And the lines that produce the scores, from the model's <code>forward</code>:</p>
<Code code={`
x = self.ln_f(x)                  # one last layer norm
logits = x @ self.wte.weight.T    # (B, T, vocab): dot product with every token's embedding
`} />

<h2>Check yourself</h2>
<Quiz
  question="You set the temperature to 0.2. What happens to the probabilities?"
  options={[
    { text: 'They become more equal', why: 'That is what a high temperature does.' },
    { text: 'The most likely tokens become even more likely', correct: true, why: 'Dividing by 0.2 multiplies every gap between scores by 5, and softmax turns bigger gaps into a bigger lead.' },
    { text: 'The model computes different scores', why: 'The model’s scores are untouched. Temperature is applied afterwards.' },
  ]}
/>
<Quiz
  question="The same prompt gives a chatbot different answers on different tries. Where does the difference come from?"
  options={[
    { text: 'The weights change between runs', why: 'The weights are fixed once training ends.' },
    { text: 'The random draw from the probabilities', correct: true, why: 'The model itself is deterministic: same text in, same probabilities out. The only randomness is in picking a token, and one different pick changes everything after it.' },
    { text: 'The tokenizer is random', why: 'The tokenizer always gives the same IDs for the same text.' },
  ]}
/>
<Quiz
  question="Why does the KV cache give exactly the same result as recomputing everything?"
  options={[
    { text: 'It does not; it is an approximation', why: 'It is exact.' },
    { text: 'Earlier lanes cannot see later tokens, so they never change', correct: true, why: 'The causal mask guarantees that a lane depends only on tokens up to its own position. Once computed, it stays valid for as long as the text only grows.' },
  ]}
/>

<style>
  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-top: 0.6rem;
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
  .dials {
    display: grid;
    gap: 0.4rem;
    margin-bottom: 1.1rem;
    justify-items: start;
  }
  .dials label {
    display: grid;
    grid-template-columns: 7rem 14rem 3rem;
    gap: 0.6rem;
    align-items: center;
  }
  @media (max-width: 520px) {
    .dials label {
      grid-template-columns: 6rem 1fr 3rem;
      width: 100%;
    }
  }
  .story {
    font-family: var(--body);
    font-size: 1.1rem;
    line-height: 1.6;
    white-space: pre-wrap;
  }
  mark {
    background: var(--think-soft);
    color: var(--ink);
    border-radius: 3px;
  }
</style>
