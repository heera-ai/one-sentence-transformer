<script lang="ts">
  import { forward, logitsAt } from '../engine/forward'
  import { pct } from '../lib/color'
  import { app, candidates } from '../lib/state.svelte'
  import Cells from '../viz/Cells.svelte'
  import Code from '../viz/Code.svelte'
  import Next from '../viz/Next.svelte'
  import Quiz from '../viz/Quiz.svelte'

  const model = app.model!
  const { n_head: H, n_layer: L, vocab_size: V, block_size: B } = model.config
  const trace = $derived(app.trace!)

  // Keep the pictures readable: at most the last 12 tokens of the sentence.
  const src = $derived(app.ids.slice(-12))
  const S = $derived(src.length)
  // Something for a decoder to be "writing": the model's own top continuation.
  const out = $derived.by(() => {
    const ids = [...app.ids]
    const made: number[] = []
    for (let i = 0; i < 5; i++) {
      const ctx = ids.slice(-B)
      const next = candidates(logitsAt(forward(model, ctx, { lastOnly: true }), ctx.length - 1, V), 1)[0].id
      if (next === app.tok!.eot) break
      ids.push(next)
      made.push(next)
    }
    return made
  })
  const O = $derived(out.length)

  type Family = 'decoder' | 'encoder' | 'both'
  let family = $state<Family>('decoder')
  const tri = (n: number) => Float32Array.from({ length: n * n }, (_, i) => (i % n <= Math.floor(i / n) ? 1 : 0))
  const full = (r: number, c: number) => new Float32Array(r * c).fill(1)
  let hover = $state('')
  const say = (reader: number, read: number, allowed: number, how: string) => `${app.label(reader)} ${allowed ? 'may read' : 'may not read'} ${app.label(read)}${allowed ? '' : ': it comes later'}  ·  ${how}`

  // --- Take the mask off the real model ---
  let layer = $state(0)
  let head = $state(0)
  let masked = $state(true)
  const open = $derived(forward(model, app.ids, { noMask: true }))
  const shown = $derived(masked ? trace : open)
  const T = $derived(trace.T)
  const size = $derived(Math.max(8, Math.min(24, Math.floor(380 / T))))
  const pattern = $derived(shown.layers[layer].attn.subarray(head * T * T, (head + 1) * T * T))
  const normal = $derived(app.next(app.last, 5))
  const openNext = $derived(candidates(logitsAt(open, app.last, V), 5))
  const scale = $derived(Math.max(0.05, normal[0]?.p ?? 0, openNext[0]?.p ?? 0))
  let patHover = $state('')
</script>

{#snippet matrix(title: string, note: string, rows: number[], cols: number[], data: Float32Array, how: string)}
  <div class="mat">
    <p class="sub">{title}</p>
    <p class="fig-note">{note}</p>
    <div class="scroll-x">
      <div class="pat">
        <div class="labels">{#each rows as id}<span style:height="19px">{app.label(id)}</span>{/each}</div>
        <Cells {data} rows={rows.length} cols={cols.length} cw={18} ch={18} mode="talk" max={1.4} label={title} onhover={(r, c, v) => (hover = say(rows[r], cols[c], v, how))} onleave={() => (hover = '')} />
      </div>
    </div>
  </div>
{/snippet}

<p>
  The model you built is a GPT, and a GPT is one member of a family of three. The word “transformer” covers all of them. They are made of the same parts you already know, and the
  differences come down to two questions:
</p>
<ol>
  <li>May a token read the tokens <em>after</em> it? (Is there a causal mask?)</li>
  <li>Is there one stack of layers, or two that are connected?</li>
</ol>

<div class="scroll-x">
<table class="plain" style="min-width:40rem">
  <thead><tr><th>Family</th><th>Reads later tokens?</th><th>Stacks</th><th>Trained to</th><th>Examples</th></tr></thead>
  <tbody>
    <tr><td><strong>Decoder-only</strong></td><td>No</td><td>One</td><td>Predict the next token</td><td>GPT, Claude, Llama, the model in this page</td></tr>
    <tr><td><strong>Encoder-only</strong></td><td>Yes</td><td>One</td><td>Fill in hidden tokens</td><td>BERT</td></tr>
    <tr><td><strong>Encoder + decoder</strong></td><td>Encoder yes, decoder no</td><td>Two</td><td>Predict the next token of an output, given a whole input</td><td>The original 2017 transformer, T5, Whisper</td></tr>
  </tbody>
</table>
</div>

<h2>Who may read whom</h2>
<p>
  Everything about the three families can be read off one kind of picture: a grid where each row is a token doing the reading and each column a token being read. A filled cell means
  attention is allowed there. You saw the real version of this grid in chapter 4.
</p>

<div class="fig">
  <p class="fig-title">What attention is allowed to see</p>
  <div class="controls" role="group" aria-label="Family">
    {#each [['decoder', 'Decoder-only (GPT)'], ['encoder', 'Encoder-only (BERT)'], ['both', 'Encoder + decoder']] as [key, name]}
      <button class="btn" class:quiet={family !== key} aria-pressed={family === key} onclick={() => (family = key as Family)}>{name}</button>
    {/each}
  </div>
  {#if family === 'decoder'}
    {@render matrix('Self-attention, with the causal mask', 'Each token reads itself and earlier tokens. This is every layer of the model in this page.', src, src, tri(S), 'decoder self-attention')}
  {:else if family === 'encoder'}
    {@render matrix('Self-attention, no mask', 'Every token reads every token, in both directions.', src, src, full(S, S), 'encoder self-attention')}
  {:else}
    <div class="three">
      {@render matrix('1. Encoder: self-attention, no mask', 'The input text. Every token reads the whole input.', src, src, full(S, S), 'encoder self-attention')}
      {@render matrix('2. Decoder: self-attention, masked', 'The output being written. Each token reads only earlier output.', out, out, tri(O), 'decoder self-attention')}
      {@render matrix('3. Decoder: cross-attention', 'Each output token (row) reads the encoder’s finished vectors for the whole input (columns).', out, src, full(O, S), 'cross-attention')}
    </div>
    <p class="fig-note">The output tokens here are this model's own continuation of your sentence, standing in for a translation or a summary.</p>
  {/if}
  <p class="readout">{hover || 'Hover over a cell.'}</p>
</div>

<h2>Encoder-only: take the mask away</h2>
<p>
  An encoder layer is the layer from chapter 7 with one line removed: the causal mask. Queries, keys, values, heads, the stream, the MLP and the normalisation are all the same.
  Without the mask, every token's final vector reflects the whole sentence, words before <em>and</em> after it. That is better for understanding a text: to know what “bank” means in “I
  sat on the bank of the river”, it helps to read to the end.
</p>
<p>
  The price is that next-token prediction stops making sense, since each token can see its own answer. So an encoder is trained differently: some tokens of the input are hidden (BERT
  selects 15% of them), and the model has to say what was there, using the words on both sides. And because there is no “next token” output, an encoder cannot write text. It turns text
  into vectors, which are then used to classify it, to search with it, or to compare two texts.
</p>
<p>The mask is not a detail that can be flipped afterwards. Try removing it from the model in this page, which was trained with it:</p>

<div class="fig">
  <p class="fig-title">The model in this page, with its mask switched off</p>
  <div class="controls">
    <label><input type="checkbox" bind:checked={masked} /> Causal mask on</label>
    <label>Layer <select bind:value={layer}>{#each Array(L) as _, l}<option value={l}>{l + 1}</option>{/each}</select></label>
    <label>Head <select bind:value={head}>{#each Array(H) as _, h}<option value={h}>{h + 1}</option>{/each}</select></label>
  </div>
  <div class="two">
    <div>
      <p class="sub">Attention pattern {masked ? 'as trained' : 'with the future visible'}</p>
      <div class="scroll-x">
        <div class="pat">
          <div class="labels">{#each app.ids as id}<span style:height="{size + 1}px">{app.label(id)}</span>{/each}</div>
          <Cells data={pattern} rows={T} cols={T} cw={size} ch={size} mode="talk" max={1} label="Attention pattern" onhover={(r, c, v) => (patHover = `${app.label(app.ids[r])} reads ${app.label(app.ids[c])}: ${pct(v)}`)} onleave={() => (patHover = '')} />
        </div>
      </div>
      <p class="readout">{patHover || 'Row = the token reading. Column = the token being read.'}</p>
    </div>
    <div>
      <p class="sub">Its prediction after your sentence</p>
      <Next items={masked ? normal : openNext} max={scale} color={masked ? 'var(--ink)' : 'var(--neg)'} mark={normal[0]?.id} />
    </div>
  </div>
</div>

<p>
  With the mask off, the upper triangle fills in: tokens now put weight on words that come after them. The prediction shifts, sometimes slightly and sometimes a lot, depending on the
  sentence. It does not collapse, because the last lane could already see every token; what has changed is what the earlier lanes hand to it. They are now built from information the
  model never had during training. This is not what a real encoder looks like inside: a real encoder is trained without the mask from the start and learns to use both directions well.
  What the figure shows is only that the mask is part of what the weights were fitted to.
</p>

<h2>Encoder + decoder: two stacks and a bridge</h2>
<p>
  The original transformer was built for translation, and it has both. The <strong>encoder</strong> reads the whole input sentence, with no mask, and produces one finished vector per input
  token. The <strong>decoder</strong> writes the output one token at a time, with the causal mask, exactly as a GPT does.
</p>
<p>
  The two are joined by the only part of the family you have not met: <strong>cross-attention</strong>. Each decoder layer has three steps where a GPT layer has two: self-attention over
  the output so far, then cross-attention, then the MLP. Cross-attention is ordinary attention with one change in where its inputs come from:
</p>
<ul>
  <li>the <strong>query</strong> comes from the decoder's stream: “I am writing this word; what do I need from the input?”</li>
  <li>the <strong>keys and values</strong> come from the encoder's finished vectors.</li>
</ul>
<p>Match, weigh, blend, add to the stream: nothing else is different. There is no mask on it, because the whole input is known before writing begins.</p>
<Code
  caption="Cross-attention: the same arithmetic as chapter 4, with two sources"
  code={`
def cross_attend(x_dec, enc_out):                # (T_out, d) and (T_in, d)
    q = query(x_dec)                             # queries from the decoder's stream
    k, v = key(enc_out), value(enc_out)          # keys and values from the encoder's output
    scores = q @ k.T / math.sqrt(d_head)         # (T_out, T_in): no mask
    weights = F.softmax(scores, dim=-1)
    return weights @ v                           # what each output token took from the input

# one decoder layer in an encoder-decoder model
x = x + self_attention(ln1(x))                   # masked: read the output so far
x = x + cross_attend(ln2(x), enc_out)            # read the input
x = x + mlp(ln3(x))
`}
/>

<h2>So what is a GPT?</h2>
<p>
  Take the original transformer, delete the encoder, and delete cross-attention from the decoder, since there is no longer anything for it to read. What is left is a stack of masked
  self-attention and MLPs: the model in this page. That history is why it is called <strong>decoder-only</strong>.
</p>
<p>
  The name causes confusion, so note what follows from it: an encoder layer and a GPT layer are the <em>same code</em>. The only differences are the mask and the training task.
</p>
<p>
  Without an encoder, a GPT handles “input” and “output” by putting both in one sequence. To translate, the text to be translated is simply the start of the sequence, and the translation is
  its continuation; ordinary self-attention lets the output tokens read the input tokens, which does the job cross-attention used to do.
</p>

<h2>Why the big models are decoder-only</h2>
<ul>
  <li><strong>One task covers everything.</strong> Answering, translating, summarising and writing code can all be posed as “continue this text”, so one model and one training method serve them all.</li>
  <li><strong>Every token is a training example</strong> (chapter 9). A masked-token model learns only from the tokens that were hidden.</li>
  <li><strong>Generation is cheap</strong>, thanks to the KV cache (chapter 8), which depends on the causal mask.</li>
  <li><strong>It is simpler:</strong> one stack, one kind of layer.</li>
</ul>
<p>
  The other two are far from obsolete. Encoder-only models are the standard tool for turning text into vectors for search and classification, where reading in both directions gives
  better vectors at a fraction of the size. Encoder + decoder models remain common where the input and output are different kinds of thing, such as speech in and text out. And the same
  encoder design, applied to patches of an image in place of tokens, is how transformers read pictures.
</p>

<h2>Check yourself</h2>
<Quiz
  question="What is the difference between an encoder layer and the GPT layer you built?"
  options={[
    { text: 'The encoder has no MLP', why: 'Both have attention followed by an MLP.' },
    { text: 'The encoder has no causal mask', correct: true, why: 'Same attention, same MLP, same stream. The encoder lets every token read every other token; the GPT layer hides later ones.' },
    { text: 'The encoder uses a different kind of attention', why: 'It is the same query, key and value arithmetic.' },
  ]}
/>
<Quiz
  question="In cross-attention, where do the keys and values come from?"
  options={[
    { text: 'The decoder’s own stream', why: 'That is the decoder’s self-attention. Only the queries come from the decoder in cross-attention.' },
    { text: 'The encoder’s finished vectors', correct: true, why: 'The decoder asks the question; the encoder’s output supplies what can be matched and collected.' },
    { text: 'The embedding table', why: 'They come from vectors that have been through the whole encoder, not from raw embeddings.' },
  ]}
/>
<Quiz
  question="Why can an encoder-only model not write text the way a GPT does?"
  options={[
    { text: 'It is too small', why: 'Size is not the issue.' },
    { text: 'It was never trained to predict a next token; every token sees the whole input', correct: true, why: 'It is trained to fill in hidden tokens using both sides. There is no position whose job is “what comes after all this”.' },
    { text: 'It has no embedding table', why: 'It has one, like every transformer.' },
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
  .sub {
    font-weight: 620;
    margin: 0.6rem 0 0.2rem;
  }
  .three {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr));
    gap: 1.2rem;
  }
  .three > :global(*),
  .mat {
    min-width: 0;
  }
  .pat {
    display: flex;
    gap: 8px;
  }
  .labels {
    display: flex;
    flex-direction: column;
    min-width: 4.5rem;
    flex: none;
  }
  .labels span {
    font-family: var(--mono);
    font-size: 0.74rem;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    white-space: pre;
  }
  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
  }
  @media (max-width: 760px) {
    .two {
      grid-template-columns: 1fr;
    }
  }
</style>
