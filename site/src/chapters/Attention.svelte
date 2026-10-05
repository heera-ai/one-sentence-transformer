<script lang="ts">
  import { softmax } from '../engine/ops'
  import { fmt, pct } from '../lib/color'
  import { app } from '../lib/state.svelte'
  import { columns } from '../lib/util'
  import Bars from '../viz/Bars.svelte'
  import Cells from '../viz/Cells.svelte'
  import Chips from '../viz/Chips.svelte'
  import Code from '../viz/Code.svelte'
  import Quiz from '../viz/Quiz.svelte'
  import Rows from '../viz/Rows.svelte'
  import Tex from '../viz/Tex.svelte'

  const { d_model: D, n_head: H, n_layer: L } = app.model!.config
  const dHead = D / H
  let layer = $state(0)
  let head = $state(0)

  const trace = $derived(app.trace!)
  const T = $derived(trace.T)
  const i = $derived(app.pos) // the token doing the reading
  const lt = $derived(trace.layers[layer])
  const q = $derived(columns(lt.q, T, D, head * dHead, dHead))
  const k = $derived(columns(lt.k, T, D, head * dHead, dHead))
  const v = $derived(columns(lt.v, T, D, head * dHead, dHead))
  const z = $derived(columns(lt.z, T, D, head * dHead, dHead))
  const scores = $derived(lt.scores.subarray((head * T + i) * T, (head * T + i + 1) * T))
  const weights = $derived(lt.attn.subarray((head * T + i) * T, (head * T + i + 1) * T))
  const pattern = $derived(lt.attn.subarray(head * T * T, (head + 1) * T * T))
  const strongest = $derived.by(() => {
    let best = 0
    for (let j = 1; j <= i; j++) if (weights[j] > weights[best]) best = j
    return best
  })
  const visible = $derived(app.ids.slice(0, i + 1))
  let patHover = $state('')
  let hover = $state('')

  // --- Softmax by hand ---
  let s = $state([2.0, 1.0, 0.0])
  const sm = $derived(softmax(s))
  const names = ['A', 'B', 'C']
</script>

<p>
  This is the chapter the transformer is famous for. So far each token's lane holds only its own token and position. But the meaning of a word depends on the words around it. In “the
  park was big”, <em>big</em> describes the park; to predict what follows, the lane for <em>big</em> needs to know that. <strong class="talk">Attention</strong> is how a lane pulls in
  information from other lanes.
</p>
<p>The idea in one sentence: <strong>each token asks a question, every earlier token offers an answer, and the token takes mostly the answers that match its question.</strong></p>

<h2>Three vectors per token</h2>
<p>For attention, every token's vector is turned into three new, smaller vectors, each by its own learned matrix:</p>
<ul>
  <li>a <strong>query</strong>: what am I looking for?</li>
  <li>a <strong>key</strong>: what do I contain, as seen by someone searching?</li>
  <li>a <strong>value</strong>: what I will hand over if I am chosen.</li>
</ul>
<p>
  It is a soft version of looking something up. In a dictionary, your query must equal a key exactly and you get that one value. Here, the query is compared with every key, each gets a match
  score, and you receive a blend of all the values, weighted by how well each key matched.
</p>

<div class="fig">
  <p class="fig-title">Follow one token as it reads</p>
  <div class="controls">
    <label>Layer <select bind:value={layer}>{#each Array(L) as _, l}<option value={l}>{l + 1}</option>{/each}</select></label>
    <label>Head <select bind:value={head}>{#each Array(H) as _, h}<option value={h}>{h + 1}</option>{/each}</select></label>
    <span class="muted">A layer has {H} of these running side by side, called heads. One is enough for now.</span>
  </div>
  <p class="fig-note">Pick the token that is doing the reading. Tokens after it are hidden from it.</p>
  <Chips ids={app.ids} selected={i} onselect={(n) => (app.focus = n)} dim={(n) => n > i} />

  <div class="step">
    <p class="step-title"><span class="n">1</span> Its query, and the keys of every token it can see</p>
    <p class="fig-note">Each is {dHead} numbers.</p>
    <div class="scroll-x">
      <div class="line"><span class="what">query of <span class="mono">{app.label(app.ids[i])}</span></span><Cells data={q.subarray(i * dHead, (i + 1) * dHead)} rows={1} cols={dHead} cw={14} ch={22} label="Query vector" onhover={(_, c, x) => (hover = `query, dimension ${c}: ${fmt(x, 3)}`)} onleave={() => (hover = '')} /></div>
    </div>
    <Rows ids={visible} data={k.subarray(0, (i + 1) * dHead)} cols={dHead} cw={14} ch={18} name="Key vectors" unit="key dimension" />
  </div>

  <div class="step">
    <p class="step-title"><span class="n">2</span> Score = query · key</p>
    <p class="fig-note">
      One dot product per visible token, divided by √{dHead} ≈ {Math.sqrt(dHead).toFixed(2)}. A high score means “this key matches what I am looking for”.
    </p>
    <Bars name="Match scores" signed items={visible.map((id, j) => ({ label: app.label(id), value: scores[j], text: fmt(scores[j]), mark: j === strongest }))} />
  </div>

  <div class="step">
    <p class="step-title"><span class="n">3</span> Softmax turns scores into weights that add up to 100%</p>
    <Bars name="Attention weights" color="var(--talk)" max={1} items={visible.map((id, j) => ({ label: app.label(id), value: weights[j], text: pct(weights[j]), mark: j === strongest }))} />
    <p class="fig-note" style="margin-top:.8rem">
      In this head, <span class="mono">{app.label(app.ids[i])}</span> reads most from <span class="mono">{app.label(app.ids[strongest])}</span>
      {strongest === i ? '(itself)' : `(${i - strongest} back)`}, at {pct(weights[strongest])}.
    </p>
  </div>

  <div class="step">
    <p class="step-title"><span class="n">4</span> Blend the values using those weights</p>
    <p class="fig-note">Each row is a token's value vector. The result is their weighted average: mostly the values of the tokens with the largest weights.</p>
    <Rows ids={visible} data={v.subarray(0, (i + 1) * dHead)} cols={dHead} cw={14} ch={18} name="Value vectors" unit="value dimension" />
    <div class="scroll-x">
      <div class="line total"><span class="what">what this head collected</span><Cells data={z.subarray(i * dHead, (i + 1) * dHead)} rows={1} cols={dHead} cw={14} ch={22} label="Weighted average of values" onhover={(_, c, x) => (hover = `result, dimension ${c}: ${fmt(x, 3)}`)} onleave={() => (hover = '')} /></div>
    </div>
    <p class="readout">{hover}</p>
  </div>
</div>

<p>
  That is all of attention: <strong>match, weigh, blend</strong>. The result is added to the token's lane, so the lane now carries information taken from other tokens. The individual numbers
  inside queries, keys and values have no names; what matters is which keys a query lines up with.
</p>

<h2>Softmax, the step that turns scores into shares</h2>
<p>
  You will meet softmax twice in a transformer: here, and at the very end where scores become probabilities. It takes any list of numbers and returns positive numbers that add up to 1. It
  does this by raising <Tex tex={'e'} /> to each score, then dividing by the total:
</p>
<Tex block tex={'\\text{softmax}(s)_j = \\frac{e^{s_j}}{\\sum_k e^{s_k}}'} />

<div class="fig">
  <p class="fig-title">Softmax by hand</p>
  <p class="fig-note">Move a score and watch the shares. What counts is the gap between scores, not their size: add 5 to all three and nothing changes.</p>
  <div class="sm">
    {#each s as _, n}
      <label>Score {names[n]} <input type="range" min="-4" max="6" step="0.1" bind:value={s[n]} /> <span class="num">{fmt(s[n], 1)}</span></label>
    {/each}
  </div>
  <Bars name="Softmax output" color="var(--talk)" max={1} items={names.map((nm, n) => ({ label: nm, value: sm[n], text: pct(sm[n]) }))} />
  <p class="readout">A gap of 1 between two scores makes one share e ≈ 2.72 times the other. A gap of 3 makes it about 20 times.</p>
</div>

<p>
  Because of the exponential, softmax is “winner takes most”: a modest lead in score becomes a large lead in weight. That is also the reason for dividing scores by √{dHead}. A dot product of
  vectors with more dimensions naturally has a wider spread, which would make softmax pick a single winner almost every time. Dividing by the square root of the number of dimensions keeps
  the spread steady.
</p>

<h2>No looking ahead</h2>
<p>
  In the figure above, tokens after the reader were greyed out. That is deliberate, and it is called the <strong>causal mask</strong>. The model's job is to predict the next token; if a
  token could read the ones after it, it would simply copy the answer. So before softmax, the score for every later token is set to −∞, which softmax turns into a weight of exactly zero.
</p>
<p>Here is the same head for every reader at once. Each row is one reader and adds up to 100%. The blank upper triangle is the mask.</p>

<div class="fig">
  <p class="fig-title">The full attention pattern of layer {layer + 1}, head {head + 1}</p>
  <p class="fig-note">Row = the token reading. Column = the token being read. Click a row to follow that reader above.</p>
  <div class="scroll-x">
    <div class="pat">
      <div class="labels">
        {#each app.ids as id, r}<span class:sel={r === i} style:height="{Math.max(14, Math.min(26, 420 / T)) + 1}px">{app.label(id)}</span>{/each}
      </div>
      <Cells
        data={pattern}
        rows={T}
        cols={T}
        cw={Math.max(14, Math.min(26, 420 / T))}
        ch={Math.max(14, Math.min(26, 420 / T))}
        mode="talk"
        max={1}
        label="Attention pattern"
        markRow={i}
        onhover={(r, c, x) => (patHover = c > r ? `${app.label(app.ids[r])} cannot see ${app.label(app.ids[c])}: it comes later` : `${app.label(app.ids[r])} reads ${app.label(app.ids[c])}: ${pct(x)}`)}
        onleave={() => (patHover = '')}
        onpick={(r) => (app.focus = r)}
      />
    </div>
  </div>
  <p class="readout">{patHover || 'Hover over a cell.'}</p>
</div>

<p>
  The mask has a useful side effect. Since no token can see later ones, what the model computes for token 3 does not change when tokens 4, 5 and 6 are added. So one pass over a sentence gives
  a next-token prediction at <em>every</em> position at once. Training depends on this, as you will see in <a href="#/training">chapter 9</a>.
</p>
<p>Try the other layers and heads with the selectors. Different heads have very different patterns; the next chapter is about why.</p>

<h2>In maths and code</h2>
<p>
  Stack the queries, keys and values of all tokens into matrices <Tex tex={'Q'} />, <Tex tex={'K'} />, <Tex tex={'V'} />, one row per token. Then steps 2 to 4 for every reader at once are
  one line, the most quoted formula in the field:
</p>
<Tex block tex={'\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d}} + M\\right)V'} />
<p>
  <Tex tex={'QK^{\\top}'} /> is every query dotted with every key. <Tex tex={'M'} /> is the mask: 0 where reading is allowed, −∞ where it is not. <Tex tex={'d'} /> is the number of dimensions
  in a query, {dHead} here.
</p>
<Code
  caption="One attention head"
  code={`
import math, torch, torch.nn as nn, torch.nn.functional as F

d_model, d_head = 128, 32
query = nn.Linear(d_model, d_head)
key   = nn.Linear(d_model, d_head)
value = nn.Linear(d_model, d_head)

def attend(x):                                   # x: (T, d_model), one row per token
    T = x.shape[0]
    q, k, v = query(x), key(x), value(x)         # each (T, d_head)
    scores = q @ k.T / math.sqrt(d_head)         # (T, T): every query · every key
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))   # hide the future
    weights = F.softmax(scores, dim=-1)          # each row sums to 1
    return weights @ v                           # (T, d_head): blended values
`}
/>

<h2>Check yourself</h2>
<Quiz
  question="A token's query matches one earlier key far better than all the others. What does the token receive?"
  options={[
    { text: 'That token’s key', why: 'Keys are only used for matching. They are never passed on.' },
    { text: 'Almost exactly that token’s value', correct: true, why: 'Softmax gives that token nearly all the weight, so the blend is nearly its value alone.' },
    { text: 'An equal mix of all values', why: 'An equal mix happens when all scores are the same.' },
  ]}
/>
<Quiz
  question="Why is the score for later tokens set to −∞ and not to 0?"
  options={[
    { text: 'A score of 0 still earns a share after softmax', correct: true, why: 'e⁰ = 1, a real share of the weight. Only e^(−∞) = 0 removes a token completely.' },
    { text: 'It makes the computation faster', why: 'Speed is not the reason; correctness is.' },
    { text: 'Zero would cause a division by zero', why: 'Nothing is divided by the scores.' },
  ]}
/>
<Quiz
  question="You add a word to the end of your sentence. What changes for the first token's lane?"
  options={[
    { text: 'Nothing', correct: true, why: 'The first token can only read itself, whatever comes after. Every lane depends only on tokens up to and including its own.' },
    { text: 'It is recomputed with the new word included', why: 'The causal mask prevents exactly that.' },
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
  .step {
    border-top: 1px solid var(--line-soft);
    margin-top: 1.3rem;
    padding-top: 1rem;
  }
  .step-title {
    font-weight: 650;
    margin: 0 0 0.3rem;
  }
  .n {
    display: inline-grid;
    place-items: center;
    width: 1.5rem;
    height: 1.5rem;
    border-radius: 50%;
    background: var(--talk);
    color: #fff;
    font-size: 0.8rem;
    margin-right: 0.35rem;
  }
  .line {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  .what {
    min-width: 5.5rem;
    text-align: right;
    font-size: 0.8rem;
    white-space: nowrap;
    flex: none;
  }
  .line.total {
    margin-top: 8px;
  }
  .line.total .what {
    font-weight: 700;
  }
  .sm {
    display: grid;
    gap: 0.3rem;
    margin-bottom: 1rem;
  }
  .sm label {
    display: flex;
    gap: 0.6rem;
    align-items: center;
  }
  .pat {
    display: flex;
    gap: 8px;
  }
  .labels {
    display: flex;
    flex-direction: column;
    min-width: 5.5rem;
    flex: none;
  }
  .labels span {
    font-family: var(--mono);
    font-size: 0.76rem;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    white-space: pre;
  }
  .labels .sel {
    font-weight: 700;
  }
</style>
