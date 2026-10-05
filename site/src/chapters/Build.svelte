<script lang="ts">
  import { modelPy, section } from '../lib/source'
  import { app } from '../lib/state.svelte'
  import Code from '../viz/Code.svelte'
  import Quiz from '../viz/Quiz.svelte'

  const manifest = app.manifest!
  const { d_model: D, n_head: H, n_layer: L, vocab_size: V, block_size: B } = manifest.config
  const T = $derived(app.ids.length)

  // Count the weights straight from the file that was loaded.
  const count = (test: (name: string) => boolean) => manifest.tensors.filter((t) => test(t.name)).reduce((n, t) => n + t.shape.reduce((a, b) => a * b, 1), 0)
  const groups = [
    { name: 'Token table (also used for the final vote)', where: 'embeddings', n: count((n) => n === 'wte.weight'), formula: `${V} × ${D}` },
    { name: 'Position table', where: 'position', n: count((n) => n === 'wpe.weight'), formula: `${B} × ${D}` },
    { name: `Attention: query, key, value, output (${L} layers)`, where: 'attention', n: count((n) => n.includes('.attn.')), formula: `${L} × 4 × (${D}² + ${D})` },
    { name: `MLP: expand and shrink (${L} layers)`, where: 'mlp', n: count((n) => n.includes('.mlp.')), formula: `${L} × (8 × ${D}² + 5 × ${D})` },
    { name: `Layer norms (${2 * L + 1} of them)`, where: 'block', n: count((n) => n.includes('ln')), formula: `${2 * L + 1} × 2 × ${D}` },
  ]

  const shapes = $derived([
    ['idx', `(1, ${T})`, 'token IDs', 'tokens'],
    ['wte(idx) + wpe(…)', `(1, ${T}, ${D})`, 'one vector per token: the stream', 'position'],
    ['q, k, v', `(1, ${H}, ${T}, ${D / H})`, 'split into heads', 'heads'],
    ['scores, weights', `(1, ${H}, ${T}, ${T})`, 'every token against every token, per head', 'attention'],
    ['attention output', `(1, ${T}, ${D})`, 'heads glued back; added to the stream', 'heads'],
    ['MLP hidden', `(1, ${T}, ${4 * D})`, 'the neurons', 'mlp'],
    ['MLP output', `(1, ${T}, ${D})`, 'added to the stream', 'mlp'],
    ['logits', `(1, ${T}, ${V})`, 'a score for every token, at every position', 'vote'],
  ])
</script>

<p>
  You have now seen every part of a GPT. This chapter puts the parts in one file. It is the actual file that produced the model running in this page: {modelPy.split('\n').length} lines
  including comments, with nothing hidden in a library beyond matrix arithmetic and automatic gradients.
</p>
<p>Read it top to bottom. Each section links back to the chapter that explains it. If any line is unclear, that chapter is where to look.</p>

<h2>The settings</h2>
<p>Five numbers define the size of the model. Everything in <a href="#/scale">chapter 12</a> is about turning these up.</p>
<Code caption="model/model.py" code={section(modelPy, '@dataclass', 'class Attention')} />

<h2>Talk: attention</h2>
<p>
  <a href="#/attention">Chapter 4</a> explains queries, keys, values, the mask and softmax; <a href="#/heads">chapter 5</a> explains the split into heads and the write to the stream.
</p>
<Code caption="model/model.py" code={section(modelPy, 'class Attention', 'class MLP')} />

<h2>Think: the MLP</h2>
<p><a href="#/mlp">Chapter 6</a>. Expand to four times the width, bend with GELU, shrink back.</p>
<Code caption="model/model.py" code={section(modelPy, 'class MLP', 'class Block')} />

<h2>One layer</h2>
<p><a href="#/block">Chapter 7</a>. Normalise a copy, talk, add; normalise a copy, think, add.</p>
<Code caption="model/model.py" code={section(modelPy, 'class Block', 'class GPT')} />

<h2>The whole model</h2>
<p>
  The two tables from <a href="#/embeddings">chapters 2</a> and <a href="#/position">3</a>, the stack of layers, and the vote from <a href="#/vote">chapter 8</a>. The loss at the end is
  the surprise score from <a href="#/training">chapter 9</a>.
</p>
<Code caption="model/model.py" code={section(modelPy, 'class GPT', '@torch.no_grad()')} />
<p>
  One thing here is new: <code>_init_weights</code>. The random starting weights are drawn small (a standard deviation of 0.02), and the matrices that write into the stream are drawn
  smaller still, scaled down by the number of layers. With many layers all adding to the same stream, this keeps the stream from starting out too large. It follows GPT-2.
</p>

<h2>Writing text</h2>
<p><a href="#/vote">Chapter 8</a>: take the last position's scores, apply temperature and top-k, draw a token, append, repeat.</p>
<Code caption="model/model.py" code={section(modelPy, '@torch.no_grad()', 'def num_params')} />

<h2>Follow the shapes</h2>
<p>
  The most useful habit when writing model code is to know the shape of every tensor. Here are the shapes for your sentence, {T} tokens, as a batch of one. If a shape in your own code
  differs from what you expect, the bug is on that line.
</p>
<div class="scroll-x">
  <table class="plain">
    <thead><tr><th>Tensor</th><th>Shape</th><th>What it holds</th><th>Chapter</th></tr></thead>
    <tbody>
      {#each shapes as [name, shape, what, slug]}
        <tr><td class="mono">{name}</td><td class="mono">{shape}</td><td>{what}</td><td><a href="#/{slug}">{slug}</a></td></tr>
      {/each}
    </tbody>
  </table>
</div>

<h2>Where the weights are</h2>
<div class="scroll-x">
  <table class="plain">
    <thead><tr><th>Part</th><th>Count</th><th>Weights</th><th>Share</th></tr></thead>
    <tbody>
      {#each groups as g}
        <tr><td><a href="#/{g.where}">{g.name}</a></td><td class="mono">{g.formula}</td><td class="num">{g.n.toLocaleString('en')}</td><td class="num">{((g.n / manifest.params) * 100).toFixed(1)}%</td></tr>
      {/each}
      <tr class="total"><td>Total</td><td></td><td class="num">{manifest.params.toLocaleString('en')}</td><td class="num">100%</td></tr>
    </tbody>
  </table>
</div>
<p>
  In a model this small the token table is a large share. As models grow, the table grows in proportion to the width, but the layers grow with the width <em>squared</em> and with the
  depth, so in large models nearly all weights are in the attention and MLP matrices.
</p>

<h2>Run it yourself</h2>
<p>
  The project this page came from has a <code>model</code> folder with everything. You need Python and <a href="https://docs.astral.sh/uv/">uv</a>. A laptop is enough: on an Apple M4 the
  training run took about {Math.round((manifest.trainlog.at(-1)?.secs ?? 0) / 60)} minutes.
</p>
<Code
  code={`
# in the model/ folder
uv sync                           # install PyTorch and the tokenizer library
uv run python prepare_data.py     # download 200 MB of TinyStories
uv run python tokenizer.py        # learn the 4,096-token vocabulary, encode the text
uv run pytest                     # six checks that the model is wired correctly
uv run python train.py            # train; prints the loss every 500 steps
uv run python export.py           # package the weights for this website
`}
/>

<h2>How to know your build is right</h2>
<p>A model with a bug usually still runs and still learns something, just worse. These checks catch the common mistakes, and each takes seconds. They are in <code>model/tests</code>.</p>
<ol>
  <li>
    <strong>The first loss is ln(vocabulary size).</strong> Before any training, the loss should be close to ln({V.toLocaleString('en')}) = {Math.log(V).toFixed(2)}. Much higher means the
    initial weights are too large; a different number often means the targets are misaligned.
  </li>
  <li>
    <strong>The future is hidden.</strong> Change the last token of an input. The outputs at all earlier positions must not change by even the smallest amount. If they do, the mask is wrong,
    and the model will “learn” by cheating and then fail when generating.
  </li>
  <li>
    <strong>It can memorise one batch.</strong> Train on the same few sequences over and over. The loss should fall to nearly zero. If it cannot memorise four sequences, something in the
    forward or backward path is broken.
  </li>
  <li><strong>The shapes match the table above</strong> at every step.</li>
</ol>

<h3>Mistakes almost everyone makes once</h3>
<ul>
  <li>Targets not shifted by one, so the model learns to copy its input. The loss drops to zero suspiciously fast.</li>
  <li>Softmax taken over the wrong dimension. Attention must sum to 1 across the tokens being read (<code>dim=-1</code>).</li>
  <li>Masking with 0 where it should be −∞, so future tokens still get weight.</li>
  <li>Forgetting <code>optimizer.zero_grad()</code>, so gradients pile up from step to step.</li>
  <li>Mixing up <code>(B, T, heads, d)</code> and <code>(B, heads, T, d)</code> when reshaping. A <code>view</code> with the wrong order runs without error and silently scrambles tokens.</li>
</ul>

<h2>Exercises</h2>
<p>Understanding sticks when you change something and predict the result before you look. In rough order of difficulty:</p>
<ol>
  <li>Train with <code>--n_layer 1</code>, then <code>--n_layer 8</code>. Predict the final loss of each before you run it.</li>
  <li>Train without the position table (pass <code>no_pos=True</code> to the model call in <code>train.py</code>). How much worse does it get? Why not completely broken?</li>
  <li>Replace <code>F.gelu</code> with <code>F.relu</code>. Then remove it altogether, and connect the result to chapter 6.</li>
  <li>Train on a different text: your own writing, code, song lyrics. You will need to rerun <code>tokenizer.py</code>. What does a 1.3-million-weight model manage?</li>
  <li>Close this page and write <code>model.py</code> again from an empty file, using only the diagram above the chapter list. This is the one that makes it permanent.</li>
</ol>

<h2>Check yourself</h2>
<Quiz
  question="Before any training, your new model reports a loss of 2.1 on a 4,096-token vocabulary. What should you think?"
  options={[
    { text: 'Great, it is already good', why: 'An untrained model cannot be good. Something is leaking the answer.' },
    { text: 'There is a bug: it is probably seeing the answers', correct: true, why: `Random weights should give about ${Math.log(V).toFixed(1)}. A much lower loss at step 0 nearly always means the targets or the mask are wrong.` },
    { text: 'The learning rate is too high', why: 'No learning has happened yet at step 0.' },
  ]}
/>
<Quiz
  question="You double d_model and leave everything else alone. Roughly what happens to the number of weights in the layers?"
  options={[
    { text: 'It doubles', why: 'The embedding tables double, but the layer matrices are width × width.' },
    { text: 'It quadruples', correct: true, why: `Each layer has about 12 × d_model² weights, so doubling the width multiplies that by four.` },
    { text: 'It stays the same', why: 'Every matrix in the model has d_model as at least one of its sides.' },
  ]}
/>

<style>
  tr.total td {
    font-weight: 700;
    border-bottom: 0;
  }
</style>
