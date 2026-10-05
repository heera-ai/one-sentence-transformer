<script lang="ts">
  import { forward, logitsAt, losses } from '../engine/forward'
  import type { Model } from '../engine/model'
  import { softmax, topK } from '../engine/ops'
  import { fmt, pct, think } from '../lib/color'
  import { section, trainPy } from '../lib/source'
  import { app, candidates } from '../lib/state.svelte'
  import { mean } from '../lib/util'
  import Bars from '../viz/Bars.svelte'
  import Cells from '../viz/Cells.svelte'
  import Chips from '../viz/Chips.svelte'
  import Code from '../viz/Code.svelte'
  import Next from '../viz/Next.svelte'
  import Quiz from '../viz/Quiz.svelte'
  import Tex from '../viz/Tex.svelte'

  const manifest = app.manifest!
  const { vocab_size: V, block_size: B, n_head: H } = manifest.config
  const trace = $derived(app.trace!)
  const T = $derived(trace.T)
  const uniform = Math.log(V)

  // --- Surprise at each token of the sentence ---
  const loss = $derived(losses(trace, V))
  const probs = $derived(loss.map((l) => Math.exp(-l)))
  let pSlider = $state(0.25)

  // --- One weight, one hill ---
  const hill = (w: number) => 0.35 * (w - 1.4) ** 2 + 0.45 * Math.sin(2.6 * w) + 1.2
  const slope = (w: number) => 0.7 * (w - 1.4) + 0.45 * 2.6 * Math.cos(2.6 * w)
  let w = $state(-2.2)
  let lr = $state(0.3)
  let trail = $state<number[]>([])
  const hx = (x: number) => ((x + 3) / 7) * 460
  const hy = (y: number) => 190 - (y / 8) * 190
  const hillPath = Array.from({ length: 141 }, (_, i) => -3 + i * 0.05)
    .map((x, i) => `${i ? 'L' : 'M'}${hx(x).toFixed(1)} ${hy(hill(x)).toFixed(1)}`)
    .join(' ')
  function descend() {
    trail = [...trail, w].slice(-12)
    w = Math.max(-3, Math.min(4, w - lr * slope(w)))
  }

  // --- The push on the scores ---
  const at = $derived(Math.min(app.pos, T - 2))
  const push = $derived.by(() => {
    if (T < 2) return []
    const p = softmax(logitsAt(trace, at, V))
    const target = app.ids[at + 1]
    const ids = [target, ...topK(p, 6).filter((id) => id !== target).slice(0, 5)]
    return ids.map((id) => ({ id, g: p[id] - (id === target ? 1 : 0), target: id === target }))
  })

  // --- The time machine ---
  const cks = manifest.checkpoints
  let ckIndex = $state(cks.length - 1)
  let snap = $state.raw<Model>(app.model!)
  let loading = $state(false)
  $effect(() => {
    const step = cks[ckIndex].step
    loading = true
    app.snapshot(step).then((m) => {
      if (cks[ckIndex].step === step) {
        snap = m
        loading = false
      }
    })
  })
  const snapTrace = $derived(forward(snap, app.ids))
  const snapLoss = $derived(mean(losses(snapTrace, V)))
  const story = $derived.by(() => {
    let ids = [...app.ids]
    const out: number[] = []
    for (let i = 0; i < 24; i++) {
      const ctx = ids.slice(-B)
      const lg = logitsAt(forward(snap, ctx, { lastOnly: true }), ctx.length - 1, V)
      let best = 0
      for (let v = 1; v < V; v++) if (lg[v] > lg[best]) best = v
      if (best === app.tok!.eot) break
      ids.push(best)
      out.push(best)
    }
    return app.tok!.decode(out)
  })
  const size = $derived(Math.max(4, Math.min(10, Math.floor(110 / T))))

  // Loss curve, with steps on a stretched scale so the early part is visible.
  const log = manifest.trainlog
  const lx = (step: number) => 40 + (Math.log10(step + 30) - Math.log10(30)) / (Math.log10(manifest.step + 30) - Math.log10(30)) * 500
  const ly = (v: number) => 12 + (1 - (v - 1.5) / (uniform + 0.3 - 1.5)) * 180
  const line = (key: 'train' | 'val') => log.map((e, i) => `${i ? 'L' : 'M'}${lx(e.step).toFixed(1)} ${ly(e[key]).toFixed(1)}`).join(' ')
  const ticks = [0, 100, 1000, 10000, manifest.step].filter((s, i, a) => s <= manifest.step && a.indexOf(s) === i)
  const finalLog = log[log.length - 1]
  const lrMax = Math.max(...log.map((e) => e.lr))
  const lrLine = log.map((e, i) => `${i ? 'L' : 'M'}${(40 + (e.step / manifest.step) * 500).toFixed(1)} ${(70 - (e.lr / lrMax) * 56).toFixed(1)}`).join(' ')
</script>

<p>
  Every chapter so far described what the model does with its weights. This one answers the remaining question: where do the {manifest.params.toLocaleString('en')} weights come from?
  Nobody sets them. They start as random numbers, and the model is then shown text and corrected, over and over, until its predictions are good. That process is <strong>training</strong>.
</p>
<p>It needs just three ideas: a way to score a prediction, a way to tell which direction each weight should move, and patience.</p>

<h2>Idea one: measure the surprise</h2>
<p>
  Take any real text. At each position we know what actually came next, so we can look up the probability the model gave to that token. A good model gave it a high probability. The
  standard score is the <strong>loss</strong>:
</p>
<Tex block tex={'\\text{loss} = -\\ln\\big(\\,p(\\text{the token that actually came next})\\,\\big)'} />
<p>
  Think of it as <strong>surprise</strong>. If the model said 100%, the surprise is 0. If it said 50%, it is 0.69. As the probability approaches zero, the surprise grows without limit:
  being confidently wrong is punished hardest.
</p>

<div class="fig">
  <p class="fig-title">Probability in, surprise out</p>
  <div class="controls">
    <label>Probability given to the right token <input type="range" min="0.001" max="1" step="0.001" bind:value={pSlider} /> <span class="num">{pct(pSlider)}</span></label>
    <span>Surprise: <strong class="num">{(-Math.log(pSlider)).toFixed(2)}</strong></span>
  </div>
  <p class="fig-note">
    A model that knows nothing and spreads its bets evenly over all {V.toLocaleString('en')} tokens gives each one {pct(1 / V)}, a surprise of ln({V.toLocaleString('en')}) =
    {uniform.toFixed(2)}. That is the starting line.
  </p>
</div>

<p>Here is the trained model's surprise at each token of your own sentence:</p>
{#if T > 1}
  <div class="fig">
    <p class="fig-title">How surprised the model was by each token you typed</p>
    <p class="fig-note">Under each token: the probability the model gave it, having read only the tokens before it. Darker means more surprised. The first token has nothing before it, so it is not scored.</p>
    <Chips ids={app.ids.slice(1)} weights={loss.map((l) => Math.min(1, l / uniform))} tint={think} sub={(i) => pct(probs[i])} selected={at} onselect={(i) => (app.focus = i)} name="Tokens tinted by surprise" />
    <p class="readout">Average surprise over these {T - 1} predictions: {mean(loss).toFixed(2)}. That average is the loss for this sentence.</p>
  </div>
{/if}

<p>
  Look at what that figure implies. One sentence of {T} tokens gave {T - 1} separate predictions to score, and they all came from a <em>single</em> run of the model. This is the pay-off of
  the causal mask: because each lane sees only what came before it, each lane is an honest prediction of its next token. Training gets one lesson per token, not one per sentence.
</p>

<h2>Idea two: find out which way is downhill</h2>
<p>
  The loss is one number, and it depends on every weight. Imagine a landscape with one direction per weight, where height is the loss. Training means walking downhill. To take a step you
  need to know, for each weight, “if I increased this a tiny bit, would the loss go up or down, and how steeply?” That slope, for every weight at once, is the
  <strong>gradient</strong>.
</p>
<p>With a single weight the landscape is a curve you can see:</p>

<div class="fig">
  <p class="fig-title">Gradient descent with one weight</p>
  <div class="controls">
    <button class="btn" onclick={descend}>Take a step</button>
    <button class="btn quiet" onclick={() => ((w = -2.2), (trail = []))}>Start again</button>
    <label>Step size <input type="range" min="0.02" max="2.2" step="0.02" bind:value={lr} /> <span class="num">{lr.toFixed(2)}</span></label>
  </div>
  <svg viewBox="0 0 460 200" class="hill" role="img" aria-label="A loss curve with a ball that moves downhill">
    <path d={hillPath} />
    {#each trail as x, i}<circle cx={hx(x)} cy={hy(hill(x))} r="4" class="ghost" style:opacity={0.15 + (0.5 * i) / trail.length} />{/each}
    <line x1={hx(w)} y1={hy(hill(w))} x2={hx(w) - Math.sign(slope(w)) * Math.min(70, Math.abs(slope(w)) * lr * 66)} y2={hy(hill(w))} class="arrow" />
    <circle cx={hx(w)} cy={hy(hill(w))} r="7" class="ball" />
    <text x="456" y="194" text-anchor="end">value of the weight</text><text x="4" y="12">loss</text>
  </svg>
  <p class="readout">weight {fmt(w)} · loss {hill(w).toFixed(2)} · slope {fmt(slope(w))} → next step moves it by {fmt(-lr * slope(w))}</p>
</div>

<p>
  The rule is: <strong>new weight = old weight − step size × slope</strong>. The step size is called the <strong>learning rate</strong>. Try a small one: safe but slow. Try a large one:
  the ball overshoots the valley and bounces, or flies off. Notice, too, that the ball can settle in a dip that is not the lowest point. With millions of weights this turns out to matter less
  than you would fear, but choosing the learning rate matters a great deal.
</p>

<h3>Where the slopes come from</h3>
<p>
  For the scores at the very top of the model the slope has a remarkably simple form. For each token, it is the probability the model gave, minus 1 if that token was the right answer:
</p>
<Tex block tex={'\\frac{\\partial\\,\\text{loss}}{\\partial\\,\\text{score}_i} = p_i - [\\,i \\text{ is the right token}\\,]'} />

{#if push.length}
  <div class="fig">
    <p class="fig-title">The correction after <span class="mono">{app.label(app.ids[at])}</span>, where the real next token was <span class="mono">{app.label(app.ids[at + 1])}</span></p>
    <p class="fig-note">
      Weights move against the slope. So a negative slope (rose) means “raise this score”, and a positive one (teal) means “lower it”. Select another token in the figure above to
      change the position.
    </p>
    <Bars name="Slope of the loss for each score" signed max={1} items={push.map((x) => ({ label: app.label(x.id), value: x.g, text: fmt(x.g), mark: x.target, note: x.target ? 'the right token: raise' : 'lower' }))} />
  </div>
{/if}

<p>
  So the right token's score is pushed up by exactly as much as the model fell short, and every wrong token is pushed down in proportion to the probability wasted on it. If the model was
  already right, all the pushes are near zero and nothing changes.
</p>
<p>
  From there, the slope for every other weight is found by working backwards through the model one operation at a time, using the chain rule from calculus: the scores came from the last
  stream, which came from an MLP and a head, which came from the layer below, and so on down to the embedding table. This is <strong>backpropagation</strong>. You do not write it by hand;
  PyTorch records every operation on the way forward and replays them in reverse when you call <code>loss.backward()</code>. This is also where the residual connections from chapter 7 earn
  their keep: the additions give the correction a direct path to the lowest layers.
</p>

<h2>Idea three: repeat, a lot</h2>
<p>One training step for this model is:</p>
<ol>
  <li>Pick 128 random stretches of {B} tokens from the stories. That is 128 × {B} = {(128 * B).toLocaleString('en')} predictions.</li>
  <li>Run the model forwards and average the surprise over all of them.</li>
  <li>Run backwards to get the gradient: one slope for each of the {manifest.params.toLocaleString('en')} weights.</li>
  <li>Move every weight a little, downhill.</li>
</ol>
<p>
  That was done {manifest.step.toLocaleString('en')} times. No step contains any insight; each one makes the model slightly less wrong on one batch. Everything the model can do is the
  accumulated residue of those small corrections.
</p>
<p>Two refinements are used by practically everyone, including here:</p>
<ul>
  <li>
    <strong>AdamW</strong> in place of the plain rule. It keeps a running average of each weight's recent slopes and of their typical size, and uses them to give every weight its own
    effective step size, which makes training far less touchy. The “W” is weight decay: a gentle pull of the weights towards zero that discourages extreme values.
  </li>
  <li>
    <strong>A learning-rate schedule.</strong> The step size ramps up from near zero over the first 500 steps, because the first gradients from random weights are erratic, and then glides down
    so the model can settle.
  </li>
</ul>
<div class="fig">
  <p class="fig-title">The learning rate used for this model</p>
  <svg viewBox="0 0 560 92" class="curve" role="img" aria-label="Learning rate over training steps: a short ramp up, then a smooth decline">
    <line x1="40" x2="540" y1="70" y2="70" class="axis" />
    <path d={lrLine} class="lr" />
    <text x="40" y="86">step 0</text><text x="540" y="86" text-anchor="end">step {manifest.step.toLocaleString('en')}</text><text x="60" y="34">peak {lrMax}</text>
  </svg>
</div>

<h2>Watch it learn</h2>
<p>
  Snapshots of the weights were saved during training. Drag the slider to load the model as it was at that moment, and see what it then thought of your sentence.
</p>

<div class="fig">
  <p class="fig-title">The model after {cks[ckIndex].step.toLocaleString('en')} training steps {loading ? '(loading…)' : ''}</p>
  <div class="controls">
    <label>Training step <input type="range" min="0" max={cks.length - 1} step="1" bind:value={ckIndex} class="wide" /></label>
    <span class="steps">{#each cks as c, i}<button class="stepbtn" class:sel={i === ckIndex} onclick={() => (ckIndex = i)}>{c.step.toLocaleString('en')}</button>{/each}</span>
  </div>
  <svg viewBox="0 0 560 220" class="curve" role="img" aria-label="Loss during training, falling from {uniform.toFixed(1)} to {finalLog.val.toFixed(2)}">
    <line x1="40" x2="540" y1={ly(uniform)} y2={ly(uniform)} class="guide" /><text x="536" y={ly(uniform) - 4} text-anchor="end">uniform guessing: {uniform.toFixed(2)}</text>
    {#each ticks as s}<line x1={lx(s)} x2={lx(s)} y1="192" y2="197" class="axis" /><text x={lx(s)} y="210" text-anchor="middle">{s.toLocaleString('en')}</text>{/each}
    {#each [2, 4, 6, 8] as v}<text x="34" y={ly(v) + 4} text-anchor="end">{v}</text><line x1="40" x2="540" y1={ly(v)} y2={ly(v)} class="grid" />{/each}
    <path d={line('train')} class="train" /><path d={line('val')} class="val" />
    <line x1={lx(cks[ckIndex].step)} x2={lx(cks[ckIndex].step)} y1="8" y2="192" class="now" />
    {#if cks[ckIndex].val}<circle cx={lx(cks[ckIndex].step)} cy={ly(cks[ckIndex].val!)} r="5" class="nowdot" />{/if}
  </svg>
  <p class="readout">
    Loss on unseen stories at this step: {cks[ckIndex].val?.toFixed(2) ?? '–'}.
    {#if T > 1}Average surprise at your sentence: {snapLoss.toFixed(2)}.{/if}
    Solid line: stories it trained on. Dashed: stories held back. The step axis is stretched at the start.
  </p>
  <div class="two">
    <div>
      <p class="sub">What it expected after your sentence</p>
      <Next items={candidates(logitsAt(snapTrace, T - 1, V), 6)} mark={app.next(app.last, 1)[0]?.id} />
    </div>
    <div>
      <p class="sub">How it continued your sentence (always taking the top token)</p>
      <p class="story">{app.text}<mark>{story}</mark></p>
    </div>
  </div>
  <p class="sub">Layer 1's attention patterns at this step</p>
  <div class="heads">
    {#each Array(H) as _, h}
      <Cells data={snapTrace.layers[0].attn.subarray(h * T * T, (h + 1) * T * T)} rows={T} cols={T} cw={size} ch={size} gap={T > 20 ? 0 : 1} mode="talk" max={1} label="Layer 1, head {h + 1} attention pattern" />
    {/each}
  </div>
</div>

<p>Step through the snapshots in order and you can see the model learn in stages:</p>
<ul>
  <li><strong>Step 0:</strong> random weights. Every token gets about the same tiny probability, the loss sits at the uniform-guessing line, and the attention patterns are an even smear.</li>
  <li>
    <strong>Step 50:</strong> the fastest gain is also the dullest. The model has learned which tokens are common at all, such as “.”, “the” and “,”, and predicts them everywhere,
    whatever the sentence says.
  </li>
  <li>
    <strong>Steps 200 to 1,000:</strong> the predictions start to depend on the text. Real words appear in a plausible order, then short grammatical sentences, though they wander and
    repeat. Layer 1's attention patterns sharpen from a smear into definite shapes.
  </li>
  <li>
    <strong>Step 3,000 onwards:</strong> the curve flattens, but the writing keeps improving: continuations stay on the subject of your sentence for longer. Two thirds of the training
    time went into the last, small part of the drop.
  </li>
</ul>

<h2>Has it learned, or memorised?</h2>
<p>
  The honest test of a model is text it has never seen. That is the dashed line: stories held back from training. Here the two lines end almost together ({finalLog.train.toFixed(2)} on
  training stories, {finalLog.val.toFixed(2)} on held-back ones), which says the model learned patterns that carry over. If the solid line kept falling while the dashed one turned
  upwards, the model would be <strong>overfitting</strong>: memorising its training text. That happens when the model is too large for its data. This one is tiny and its data plentiful, so it
  does not.
</p>
<p>
  A final loss of {finalLog.val.toFixed(2)} means that on average the model gives the true next token a probability of about e<sup>−{finalLog.val.toFixed(2)}</sup> ≈ {pct(
    Math.exp(-finalLog.val),
  )}. That may sound low, but remember your own score in chapter 0: language is unpredictable, and nobody can do much better on text where many continuations are equally good.
</p>

<h2>In code</h2>
<p>Getting a batch. The targets are the inputs shifted along by one token: for every position, the answer is simply the token after it.</p>
<Code caption="model/train.py" code={section(trainPy, 'def get_batch', 'def lr_at')} />
<p>And the loop, with logging and snapshots removed:</p>
<Code
  code={`
model = GPT(GPTConfig())
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

for step in range(30_000):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)              # warm-up, then cosine decay

    x, y = get_batch(train_data, 128, 64, device)
    logits, loss = model(x, y)                 # forward: predictions, and average surprise

    optimizer.zero_grad()                      # clear the previous step's gradients
    loss.backward()                            # backward: a slope for every weight
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # cap freak gradients
    optimizer.step()                           # move every weight a little, downhill
`}
/>
<p>
  The loss inside the model is one call, <code>F.cross_entropy(logits, targets)</code>, which applies softmax and takes −ln of the right token's probability in a single numerically safe
  operation. “Cross-entropy” is the formal name for the surprise score.
</p>

<h2>Check yourself</h2>
<Quiz
  question={`An untrained model with a ${V.toLocaleString('en')}-token vocabulary should start with a loss of about…`}
  options={[
    { text: '0', why: 'Zero would mean it predicts every token with certainty.' },
    { text: `ln(${V.toLocaleString('en')}) ≈ ${uniform.toFixed(1)}`, correct: true, why: 'Random weights give roughly equal probability to everything. Checking for this number at step 0 is a standard test that a new model is wired up correctly.' },
    { text: String(V), why: 'The loss is the logarithm of how many options the model is effectively choosing between.' },
  ]}
/>
<Quiz
  question="The model gave the correct next token 90%. What does the correction look like?"
  options={[
    { text: 'A large push on every score', why: 'The pushes are proportional to the error, and the error is small.' },
    { text: 'A small push: raise the right score by 0.1, lower the others slightly', correct: true, why: 'The slope for the right token is 0.9 − 1 = −0.1, and the wrong tokens share +0.1 between them. Nearly right means nearly no change.' },
    { text: 'No push at all', why: 'Only at exactly 100% is there nothing to correct.' },
  ]}
/>
<Quiz
  question="Why does one 64-token stretch of text provide 64 training examples and not just one?"
  options={[
    { text: 'The text is run through the model 64 times', why: 'One run is enough.' },
    { text: 'Every position predicts its own next token in the same run, without seeing it', correct: true, why: 'The causal mask means position 10’s output depends only on tokens 0 to 10, so it is a fair prediction of token 11, and likewise for every other position.' },
  ]}
/>

<style>
  .hill {
    width: 100%;
    max-width: 460px;
    display: block;
  }
  .hill path {
    fill: none;
    stroke: var(--ink);
    stroke-width: 2;
  }
  .hill .ball {
    fill: var(--think-fill);
    stroke: var(--ink);
    stroke-width: 2;
  }
  .hill .ghost {
    fill: var(--ink-3);
  }
  .hill .arrow {
    stroke: var(--neg);
    stroke-width: 3;
    stroke-linecap: round;
  }
  .hill text,
  .curve text {
    font-family: var(--display);
    font-size: 11px;
    fill: var(--ink-2);
  }
  .curve {
    width: 100%;
    max-width: 620px;
    display: block;
  }
  .axis {
    stroke: var(--ink-3);
  }
  .grid {
    stroke: var(--line-soft);
  }
  .guide {
    stroke: var(--neg);
    stroke-dasharray: 3 4;
  }
  .lr {
    fill: none;
    stroke: var(--ink);
    stroke-width: 2;
  }
  .train {
    fill: none;
    stroke: var(--ink);
    stroke-width: 2;
  }
  .val {
    fill: none;
    stroke: var(--talk);
    stroke-width: 2;
    stroke-dasharray: 5 4;
  }
  .now {
    stroke: var(--think-fill);
    stroke-width: 2;
  }
  .nowdot {
    fill: var(--think-fill);
    stroke: var(--ink);
    stroke-width: 1.5;
  }
  .wide {
    width: 14rem;
  }
  .steps {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .stepbtn {
    font-family: var(--mono);
    font-size: 0.76rem;
    background: #edf1f5;
    border: 1.5px solid transparent;
    border-radius: 6px;
    padding: 0.15rem 0.45rem;
  }
  .stepbtn.sel {
    border-color: var(--ink);
    background: var(--think-soft);
  }
  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-top: 0.8rem;
  }
  @media (max-width: 760px) {
    .two {
      grid-template-columns: 1fr;
    }
  }
  .sub {
    font-weight: 620;
    margin: 0.8rem 0 0.5rem;
  }
  .story {
    font-family: var(--body);
    font-size: 1rem;
    line-height: 1.55;
    white-space: pre-wrap;
  }
  mark {
    background: var(--think-soft);
    color: var(--ink);
    border-radius: 3px;
  }
  .heads {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }
</style>
