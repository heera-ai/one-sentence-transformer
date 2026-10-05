<script lang="ts">
  import { forward, logitsAt } from '../engine/forward'
  import { pct } from '../lib/color'
  import { app, candidates } from '../lib/state.svelte'
  import { big } from '../lib/util'
  import Chips from '../viz/Chips.svelte'
  import Next from '../viz/Next.svelte'
  import Quiz from '../viz/Quiz.svelte'

  const cfg = app.model!.config
  const START = 'The little dog ran to the park and saw a big'

  function addTop(n: number) {
    for (let i = 0; i < n; i++) {
      const top = app.next(app.last, 1)[0]
      if (!top || top.id === app.tok!.eot) break
      app.append(top.id)
    }
  }

  // --- The guessing game: a short story the model has never seen. ---
  const STORY =
    'Once upon a time, there was a little rabbit named Ben. Ben loved to hop in the garden. One day, he found a big red ball under a tree. He was very happy and wanted to show it to his mom.'
  const storyIds = app.tok!.encode(STORY)
  const storyTrace = forward(app.model!, storyIds)
  const spots: number[] = []
  for (let t = 6; t < storyIds.length - 1 && spots.length < 8; t++) {
    const wholeWord = /^ [a-z]+$/.test(app.tok!.text(storyIds[t + 1]))
    if (wholeWord && (spots.length === 0 || t - spots[spots.length - 1] >= 4)) spots.push(t)
  }

  let round = $state(0)
  let guess = $state('')
  let revealed = $state(false)
  let you = $state(0)
  let machine = $state(0)
  let youRight = $state(false)

  const t = $derived(spots[Math.min(round, spots.length - 1)])
  const actual = $derived(storyIds[t + 1])
  const guesses = $derived(candidates(logitsAt(storyTrace, t, cfg.vocab_size), 5))
  const actualP = $derived(candidates(logitsAt(storyTrace, t, cfg.vocab_size), cfg.vocab_size).find((c) => c.id === actual)!.p)
  const done = $derived(round >= spots.length)

  function reveal(e: Event) {
    e.preventDefault()
    if (!guess.trim()) return
    youRight = guess.trim().toLowerCase() === app.tok!.text(actual).trim().toLowerCase()
    if (youRight) you++
    if (guesses[0].id === actual) machine++
    revealed = true
  }
  function nextRound() {
    round++
    guess = ''
    revealed = false
  }
  function restart() {
    round = 0
    you = 0
    machine = 0
    guess = ''
    revealed = false
  }
</script>

<h1>One sentence, all the way through</h1>

<p class="lead">
  A transformer is the kind of program behind ChatGPT, Claude and Gemini. In this course you will type one sentence and follow it through a real transformer, stage by stage, until it comes out
  the other end as a prediction of the next word. Every number you see is computed live in this page, from your sentence.
</p>

<div class="fig">
  <p class="fig-title">What the model expects after your sentence</p>
  <p class="fig-note">Edit the sentence in the bar above, or click a guess to add it. Keep adding and the model writes a story.</p>
  <Chips ids={app.ids} name="Your sentence as tokens" />
  <div style="height:1rem"></div>
  <Next items={app.next(app.last, 8)} onpick={(id) => app.append(id)} />
  <div class="controls" style="margin-top:1rem;margin-bottom:0">
    <button class="btn" onclick={() => addTop(1)}>Add the top guess</button>
    <button class="btn quiet" onclick={() => addTop(10)}>Add 10 in a row</button>
    <button class="btn quiet" onclick={() => app.setText(START)}>Start over</button>
  </div>
</div>

<p>
  That is the whole trick. The model looks at the text so far and gives every piece of text it knows a probability of coming next. Pick one, add it to the end, and ask again. A chatbot's
  long answer is this loop run a few hundred times.
</p>
<p>
  Nobody wrote rules for it. The model in this page is {big(app.manifest!.params)} numbers, called <strong>weights</strong>, that started out random and were adjusted, bit by bit, until
  the predictions got good. The large models you have used work the same way, with many thousands of times more weights.
</p>

<h2>Play the same game yourself</h2>
<p>To feel what the model is doing, do its job. Here is a story the model has never seen. Guess the next word, then see how the model did at the same spot.</p>

<div class="fig">
  {#if !done}
    <p class="fig-title">Round {round + 1} of {spots.length}</p>
    <p class="story">{app.tok!.decode(storyIds.slice(0, t + 1))}<span class="blank">{revealed ? app.tok!.text(actual) : ' ____'}</span></p>
    {#if !revealed}
      <form class="controls" onsubmit={reveal}>
        <label for="guess">Your guess</label>
        <input id="guess" type="text" bind:value={guess} autocomplete="off" spellcheck="false" />
        <button class="btn" disabled={!guess.trim()}>Reveal</button>
      </form>
    {:else}
      <p role="status">
        {youRight ? 'You got it.' : `You said “${guess.trim()}”.`}
        The story continued with <span class="mono">{app.label(actual)}</span>. The model gave that word {pct(actualP)}. Its five best guesses:
      </p>
      <Next items={guesses} mark={actual} />
      <div class="controls" style="margin:1rem 0 0">
        <button class="btn" onclick={nextRound}>{round + 1 < spots.length ? 'Next round' : 'See the score'}</button>
      </div>
    {/if}
  {:else}
    <p class="fig-title">Final score: you {you}, the model {machine}, out of {spots.length}</p>
    <p>
      The model scores a point when its single top guess is the real word. Notice that neither of you could be right every time: after “he found a big red”, many words fit. That is why the model
      does not output one word. It outputs a probability for every option, and a good model puts high probability on the words that really do tend to follow.
    </p>
    <button class="btn quiet" onclick={restart}>Play again</button>
  {/if}
  <p class="readout">You {you} · the model {machine}</p>
</div>

<h2>The picture to remember</h2>
<p>
  The diagram above the chapter list is the whole machine, and it stays there for the entire course. Each chapter opens one part of it.
</p>
<ol>
  <li><strong>Tokens.</strong> The text is cut into pieces, and each piece becomes an ID number.</li>
  <li><strong>Vectors.</strong> Each ID is swapped for a list of {cfg.d_model} numbers, and its position in the sentence is mixed in. From here on, every token has its own lane.</li>
  <li>
    <strong><span class="talk">Talk</span>, then <span class="think">think</span>.</strong> In a layer, each lane first collects information from the lanes before it (attention), then works
    on what it collected, alone (the MLP). This model does that {cfg.n_layer} times.
  </li>
  <li><strong>Vote.</strong> The last lane's numbers are turned into a probability for every possible next token.</li>
</ol>
<p>If you remember only one line: <strong>tokens become vectors; vectors talk, then think, several times; the last one votes.</strong></p>

<h2>About the model in this page</h2>
<p>
  It has {cfg.n_layer} layers and {app.manifest!.params.toLocaleString('en')} weights, and it knows {cfg.vocab_size.toLocaleString('en')} tokens. It was trained on
  <a href="https://arxiv.org/abs/2305.07759">TinyStories</a>, a collection of short stories written with the vocabulary of a young child. That is why something this small can write sensible
  sentences, and also why it knows nothing else. Ask it to continue “The capital of France is” and it will fail, because it has never read about France. Keep your sentence in story-land:
  dogs, parks, moms, toys, being happy or sad.
</p>
<p>
  The structure, though, is the real thing. It is the same design as GPT-2, and you will build it yourself in <a href="#/build">chapter 10</a>.
</p>

<h2>Check yourself</h2>
<Quiz
  question="What does the model produce each time it runs?"
  options={[
    { text: 'The single best next word', why: 'It never commits to one word itself. Choosing happens afterwards, outside the model.' },
    { text: 'A probability for every token it knows', correct: true, why: `Each run produces ${cfg.vocab_size.toLocaleString('en')} probabilities that add up to 1. Picking one of them is a separate step.` },
    { text: 'A whole sentence at once', why: 'Sentences come from running the model again and again, adding one token each time.' },
  ]}
/>
<Quiz
  question="How does the model write a long answer?"
  options={[
    { text: 'It plans the whole answer, then prints it', why: 'There is no separate planning step in the machinery.' },
    { text: 'It predicts one token, adds it to the text, and runs again', correct: true, why: 'Predict, append, repeat. Each new token is predicted from everything written so far, including the model’s own earlier output.' },
    { text: 'It looks up a stored answer', why: 'Nothing is stored as text. The weights are numbers, not sentences.' },
  ]}
/>

<style>
  .lead {
    font-size: 1.22rem;
    line-height: 1.55;
    margin-top: 1.2rem;
  }
  .story {
    font-family: var(--body);
    font-size: 1.15rem;
    line-height: 1.6;
  }
  .blank {
    font-weight: 700;
    background: var(--think-soft);
    border-radius: 4px;
    white-space: pre;
  }
  input {
    font-family: var(--mono);
    border: 1.5px solid var(--line);
    border-radius: 9px;
    padding: 0.45rem 0.7rem;
    width: 11rem;
  }
</style>
