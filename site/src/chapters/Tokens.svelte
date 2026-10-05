<script lang="ts">
  import { Tokenizer } from '../engine/tokenizer'
  import { app } from '../lib/state.svelte'
  import Chips from '../viz/Chips.svelte'
  import Code from '../viz/Code.svelte'
  import Quiz from '../viz/Quiz.svelte'

  const tok = app.tok!
  const show = Tokenizer.displayPart

  // --- How one chunk is split, merge by merge ---
  let word = $state(' unbelievable')
  let step = $state(0)
  const steps = $derived(tok.mergeSteps(word || ' '))
  const at = $derived(Math.min(step, steps.length - 1))
  $effect(() => {
    word
    step = 0
  })

  // --- A toy BPE trainer on a tiny text. Uses characters where the real one uses bytes. ---
  const RHYME = 'the cat sat on the mat. the fat cat sat on the hat. that cat is the fattest cat.'
  let source = $state(RHYME)
  let merges = $state<[string, string][]>([])
  const words = $derived.by(() => {
    let ws = tok.chunks(source).map((c) => Array.from(c))
    for (const [a, b] of merges) ws = ws.map((w) => mergePair(w, a, b))
    return ws
  })
  const pairCounts = $derived.by(() => {
    const counts = new Map<string, { a: string; b: string; n: number }>()
    for (const w of words)
      for (let i = 0; i < w.length - 1; i++) {
        const key = w[i] + '\u0000' + w[i + 1]
        const e = counts.get(key) ?? { a: w[i], b: w[i + 1], n: 0 }
        e.n++
        counts.set(key, e)
      }
    return [...counts.values()].sort((x, y) => y.n - x.n)
  })
  const tokenCount = $derived(words.reduce((n, w) => n + w.length, 0))
  const vocab = $derived(new Set(words.flat()).size)

  function mergePair(w: string[], a: string, b: string): string[] {
    const out: string[] = []
    for (let i = 0; i < w.length; i++) {
      if (i < w.length - 1 && w[i] === a && w[i + 1] === b) {
        out.push(a + b)
        i++
      } else out.push(w[i])
    }
    return out
  }
  function mergeTop(n: number) {
    for (let i = 0; i < n; i++) {
      const top = pairCounts[0]
      if (!top || top.n < 2) return
      merges = [...merges, [top.a, top.b]]
    }
  }
  const vis = (s: string) => s.replace(/ /g, '␣')
</script>

<p>A model works on numbers, so the first job is to turn text into numbers. Here is your sentence after that step.</p>

<div class="fig">
  <p class="fig-title">Your sentence is {app.text.length} characters and {app.ids.length} tokens</p>
  <p class="fig-note">Each chip is one token. The number under it is its ID, which is all the model receives. <span class="mono">␣</span> marks a space.</p>
  <Chips ids={app.ids} sub={(i) => String(app.ids[i])} name="Tokens with their IDs" />
  <p class="readout">The model's input: [{app.ids.join(', ')}]</p>
</div>

<p>
  Notice three things. A space belongs to the word that follows it, so <span class="mono">␣dog</span> and <span class="mono">dog</span> are different tokens. Common words are a single token. And
  if you type an unusual word, try <em>unbelievable</em> or your own name, it gets cut into several pieces.
</p>

<h2>Why pieces, and not letters or whole words?</h2>
<p>There are two obvious ways to cut text, and both have a problem.</p>
<ul>
  <li>
    <strong>One token per letter.</strong> The vocabulary is tiny, but sentences become very long, and the model has to spend effort learning that <em>d-o-g</em> is a thing.
  </li>
  <li>
    <strong>One token per word.</strong> Sentences are short, but the vocabulary is endless. Any word that was not in the list, a typo, a new name, cannot be read at all.
  </li>
</ul>
<p>
  The fix used by nearly every modern model is in between: a fixed list of <strong>sub-word pieces</strong>. Frequent words get their own token, rare words are spelled from smaller pieces, and
  since single characters are always in the list, nothing is ever unreadable. This model's list has {tok.vocabSize.toLocaleString('en')} entries. GPT-2's has 50,257, and current large
  models use roughly 100,000 to 250,000.
</p>

<h2>How a word gets split</h2>
<p>
  The tokenizer holds an ordered list of <strong>merge rules</strong>, each saying “these two pieces, side by side, become one piece”. To split a word it starts with single characters and
  keeps applying whichever available rule comes earliest in the list, until none applies.
</p>

<div class="fig">
  <p class="fig-title">Watch the merges happen</p>
  <div class="controls">
    <label>Word <input type="text" bind:value={word} spellcheck="false" /></label>
    <button class="btn quiet" disabled={at === 0} onclick={() => (step = at - 1)}>Back</button>
    <button class="btn" disabled={at >= steps.length - 1} onclick={() => (step = at + 1)}>Apply next merge</button>
    <button class="btn quiet" disabled={at >= steps.length - 1} onclick={() => (step = steps.length - 1)}>Finish</button>
  </div>
  <p class="fig-note">Keep the leading space if the word comes after another word; it is part of the token.</p>
  <div class="parts">
    {#each steps[at].parts as part}<span class="part">{show(part)}</span>{/each}
  </div>
  <p class="readout">
    {#if at === 0}Start: {steps[0].parts.length} single characters.
    {:else}Merge {at} of {steps.length - 1}: <b>{show(steps[at].merged![0])}</b> + <b>{show(steps[at].merged![1])}</b> → <b>{show(steps[at].merged!.join(''))}</b>. Now {steps[at].parts.length}
      {steps[at].parts.length === 1 ? 'token' : 'tokens'}.{/if}
  </p>
</div>

<div class="aside">
  <p>
    Strictly, the starting units are <strong>bytes</strong>, not characters. An English letter is one byte, so you will not see the difference here, but an emoji or an accented letter is
    several bytes. Starting from the 256 possible bytes is what guarantees any text at all can be tokenized.
  </p>
</div>

<h2>Where the merge rules come from</h2>
<p>
  They are learned from text by counting, with an algorithm called <strong>byte-pair encoding</strong> (BPE). Start with single characters. Find the pair of neighbours that occurs most often.
  Glue it into a new token. Repeat until the vocabulary is the size you want. Try it on a small text:
</p>

<div class="fig">
  <p class="fig-title">Train a tokenizer by hand</p>
  <label class="block">Training text <textarea rows="2" bind:value={source} oninput={() => (merges = [])} spellcheck="false"></textarea></label>
  <div class="controls">
    <button class="btn" disabled={!pairCounts[0] || pairCounts[0].n < 2} onclick={() => mergeTop(1)}>Merge the most common pair</button>
    <button class="btn quiet" disabled={!pairCounts[0] || pairCounts[0].n < 2} onclick={() => mergeTop(5)}>Do 5 merges</button>
    <button class="btn quiet" disabled={merges.length === 0} onclick={() => (merges = [])}>Reset</button>
  </div>
  <div class="two">
    <div>
      <p class="sub">The text, as tokens ({tokenCount} tokens, {vocab} different ones)</p>
      <div class="parts small">
        {#each words as w}{#each w as part}<span class="part">{vis(part)}</span>{/each}{/each}
      </div>
    </div>
    <div>
      <p class="sub">Most common neighbouring pairs right now</p>
      <table class="plain pairs">
        <tbody>
          {#each pairCounts.slice(0, 5) as p, i}
            <tr class:top={i === 0}><td class="mono">{vis(p.a)} + {vis(p.b)}</td><td class="num">{p.n}×</td></tr>
          {/each}
        </tbody>
      </table>
      {#if merges.length}
        <p class="sub">Rules learned so far</p>
        <p class="mono rules">{merges.map(([a, b]) => `${vis(a)}+${vis(b)}`).join('   ')}</p>
      {/if}
    </div>
  </div>
</div>

<p>
  Each merge makes the text shorter and the vocabulary one entry larger. Stop early and you have a small vocabulary and long sequences; keep going and common words become single tokens.
  The real tokenizer for this model ran this procedure on about 200 MB of stories until it had {tok.vocabSize.toLocaleString('en')} tokens. Its first rules were:
</p>
<p class="mono rules">
  {#each tok.merges.slice(0, 10) as [a, b]}<span>{show(a)} + {show(b)} → {show(a + b)}</span>{/each}
</p>
<p>and its last, by which point whole words were being assembled:</p>
<p class="mono rules">
  {#each tok.merges.slice(-6) as [a, b]}<span>{show(a)} + {show(b)} → {show(a + b)}</span>{/each}
</p>

<h2>In code</h2>
<p>The training loop is short. This version is slow but complete:</p>
<Code
  caption="Byte-pair encoding: learning the merge rules"
  code={`
from collections import Counter

def train_bpe(words: list[list[str]], num_merges: int) -> list[tuple[str, str]]:
    """words: each word as a list of single characters."""
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for w in words:
            pairs.update(zip(w, w[1:]))          # count every neighbouring pair
        if not pairs:
            break
        a, b = pairs.most_common(1)[0][0]        # the most frequent pair
        merges.append((a, b))
        words = [merge(w, a, b) for w in words]  # glue it together everywhere
    return merges

def merge(w: list[str], a: str, b: str) -> list[str]:
    out, i = [], 0
    while i < len(w):
        if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(w[i]); i += 1
    return out
`}
/>
<p>
  In practice you use a fast library. This model's tokenizer was trained with Hugging Face <code>tokenizers</code> in <code>model/tokenizer.py</code>, and the page you are reading runs its own
  copy of the encoder, tested to give identical IDs.
</p>

<h2>Check yourself</h2>
<Quiz
  question="You type a word the tokenizer has never seen, like “zorbified”. What happens?"
  options={[
    { text: 'It is replaced by a special “unknown” token', why: 'Older word-level systems did this. Byte-level BPE never needs to.' },
    { text: 'It is split into smaller pieces that are in the vocabulary', correct: true, why: 'In the worst case it falls all the way back to single bytes, which are always in the vocabulary.' },
    { text: 'The model refuses the input', why: 'Any text can be tokenized.' },
  ]}
/>
<Quiz
  question="What does the model itself receive?"
  options={[
    { text: 'The letters of each token', why: 'The model never sees letters. It cannot spell unless it learns spelling indirectly.' },
    { text: 'A list of ID numbers', correct: true, why: 'Just integers. Everything the model knows about a token it has to learn and store in that token’s vector, which is the next chapter.' },
    { text: 'The meaning of each word', why: 'Meaning is not given; it is learned, starting in the next chapter.' },
  ]}
/>

<style>
  .parts {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    min-height: 2.4rem;
  }
  .part {
    font-family: var(--mono);
    font-size: 1rem;
    background: var(--talk-soft);
    border-radius: 6px;
    padding: 0.3rem 0.5rem;
    white-space: pre;
  }
  .parts.small .part {
    font-size: 0.8rem;
    padding: 0.18rem 0.32rem;
  }
  input[type='text'] {
    font-family: var(--mono);
    border: 1.5px solid var(--line);
    border-radius: 8px;
    padding: 0.35rem 0.6rem;
    width: 12rem;
    white-space: pre;
  }
  .block {
    display: block;
    font-weight: 600;
  }
  textarea {
    display: block;
    width: 100%;
    font-family: var(--mono);
    font-size: 0.86rem;
    border: 1.5px solid var(--line);
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    margin: 0.3rem 0 0.8rem;
    resize: vertical;
  }
  .two {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 1.5rem;
  }
  @media (max-width: 720px) {
    .two {
      grid-template-columns: 1fr;
    }
  }
  .sub {
    font-weight: 600;
    font-size: 0.86rem;
    margin: 0 0 0.4rem;
  }
  .pairs {
    margin-bottom: 0.8rem;
  }
  .pairs td {
    padding: 0.2rem 1rem 0.2rem 0.4rem;
  }
  .pairs tr.top {
    background: var(--think-soft);
    font-weight: 700;
  }
  .rules {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem 1.2rem;
    font-size: 0.82rem;
    white-space: pre-wrap;
    max-width: 52rem;
  }
</style>
