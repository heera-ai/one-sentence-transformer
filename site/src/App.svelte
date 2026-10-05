<script lang="ts">
  import { onMount, type Component } from 'svelte'
  import Attention from './chapters/Attention.svelte'
  import Block from './chapters/Block.svelte'
  import Build from './chapters/Build.svelte'
  import Family from './chapters/Family.svelte'
  import Embeddings from './chapters/Embeddings.svelte'
  import Game from './chapters/Game.svelte'
  import Heads from './chapters/Heads.svelte'
  import Mlp from './chapters/Mlp.svelte'
  import Position from './chapters/Position.svelte'
  import Scale from './chapters/Scale.svelte'
  import Tokens from './chapters/Tokens.svelte'
  import Training from './chapters/Training.svelte'
  import Vote from './chapters/Vote.svelte'
  import { chapters, indexOf } from './lib/chapters'
  import { app } from './lib/state.svelte'
  import Machine from './viz/Machine.svelte'
  import SentenceBar from './viz/SentenceBar.svelte'

  const pages: Record<string, Component> = {
    game: Game,
    tokens: Tokens,
    embeddings: Embeddings,
    position: Position,
    attention: Attention,
    heads: Heads,
    mlp: Mlp,
    block: Block,
    vote: Vote,
    training: Training,
    build: Build,
    family: Family,
    scale: Scale,
  }

  let menuOpen = $state(false)
  const index = $derived(indexOf(app.slug))
  const chapter = $derived(chapters[index])
  const Page = $derived(pages[app.slug])

  function route() {
    app.go(location.hash.replace(/^#\/?/, '') || 'game')
    menuOpen = false
    window.scrollTo(0, 0)
    document.title = `${chapters[indexOf(app.slug)].title} · One Sentence, All the Way Through`
  }

  onMount(() => {
    route()
    app.load()
    window.addEventListener('hashchange', route)
    return () => window.removeEventListener('hashchange', route)
  })
</script>

<div class="layout">
  <nav class="rail" class:open={menuOpen} aria-label="Chapters">
    <a class="brand" href="#/game">One sentence,<br />all the way through</a>
    <Machine stage={chapter.stage} />
    <ol>
      {#each chapters as c, i}
        <li>
          <a href="#/{c.slug}" class:current={c.slug === app.slug} aria-current={c.slug === app.slug ? 'page' : undefined}>
            <span class="n">{i}</span>
            <span>{c.title}</span>
            {#if app.visited.includes(c.slug) && c.slug !== app.slug}<span class="seen" aria-label="visited">✓</span>{/if}
          </a>
        </li>
      {/each}
    </ol>
  </nav>

  <div class="main">
    <header class="sticky">
      <SentenceBar onmenu={() => (menuOpen = !menuOpen)} />
    </header>

    <main>
      {#if app.error}
        <div class="state">
          <h2>The model did not load</h2>
          <p>{app.error}. Check your connection and reload the page.</p>
          <button class="btn" onclick={() => location.reload()}>Reload</button>
        </div>
      {:else if !app.ready}
        <div class="state"><p>Loading the model (about 3 MB)…</p></div>
      {:else}
        {#key app.slug}
          <article>
            {#if index > 0}
              <p class="kicker">Chapter {index} of {chapters.length - 1}</p>
              <h1>{chapter.title}</h1>
            {/if}
            <Page />

            <section class="takeaway" aria-label="Takeaway">
              <h2>Keep this</h2>
              <p>{chapter.takeaway}</p>
            </section>

            <footer>
              {#if index > 0}
                <a class="btn quiet" href="#/{chapters[index - 1].slug}">Back: {chapters[index - 1].title}</a>
              {/if}
              {#if index < chapters.length - 1}
                <a class="btn" href="#/{chapters[index + 1].slug}">Next: {chapters[index + 1].title}</a>
              {/if}
            </footer>
          </article>
        {/key}
      {/if}
    </main>
  </div>
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: var(--rail) minmax(0, 1fr);
    min-height: 100vh;
  }
  .rail {
    position: sticky;
    top: 0;
    align-self: start;
    height: 100vh;
    overflow-y: auto;
    padding: 1.4rem 1.1rem 2rem 1.4rem;
    border-right: 1px solid var(--line);
    font-family: var(--display);
  }
  .brand {
    display: block;
    font-weight: 750;
    font-size: 1.18rem;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: var(--ink);
    text-decoration: none;
    margin-bottom: 1.1rem;
    font-variation-settings: 'wdth' 85;
  }
  ol {
    list-style: none;
    padding: 0;
    margin: 1rem 0 0;
    max-width: none;
  }
  li {
    margin: 0;
  }
  ol a {
    display: grid;
    grid-template-columns: 1.5rem 1fr auto;
    align-items: baseline;
    padding: 0.3rem 0.5rem;
    border-radius: 7px;
    color: var(--ink-2);
    text-decoration: none;
    font-size: 0.93rem;
    line-height: 1.3;
  }
  ol a:hover {
    background: #dfe5ec;
    color: var(--ink);
  }
  ol a.current {
    background: var(--ink);
    color: #fff;
  }
  .n {
    font-family: var(--mono);
    font-size: 0.74rem;
    opacity: 0.7;
  }
  .seen {
    font-size: 0.75rem;
    color: var(--pos);
  }

  .main {
    min-width: 0;
  }
  .sticky {
    position: sticky;
    top: 0;
    z-index: 5;
    background: color-mix(in srgb, var(--bg) 92%, transparent);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line);
    padding: 0.7rem clamp(1rem, 4vw, 3.5rem);
  }
  main {
    padding: 2.4rem clamp(1rem, 4vw, 3.5rem) 5rem;
  }
  .kicker {
    font-family: var(--display);
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--ink-2);
    margin: 0 0 0.3rem;
  }
  article > :global(h1) {
    margin-bottom: 1.4rem;
  }
  .state {
    font-family: var(--display);
    padding: 3rem 0;
  }
  .takeaway {
    margin-top: 3.5rem;
    border-top: 3px solid var(--ink);
    padding-top: 1rem;
    max-width: var(--measure);
  }
  .takeaway h2 {
    font-size: 1rem;
    margin: 0 0 0.4rem;
    letter-spacing: 0;
  }
  .takeaway p {
    font-family: var(--display);
    font-size: 1.3rem;
    line-height: 1.32;
    font-weight: 560;
    letter-spacing: -0.01em;
  }
  footer {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin-top: 2rem;
  }
  footer a {
    text-decoration: none;
  }

  @media (max-width: 900px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    .rail {
      display: none;
      position: fixed;
      inset: 0;
      z-index: 10;
      background: var(--bg);
      border: 0;
      padding-top: 4.2rem;
    }
    .rail.open {
      display: block;
    }
    .sticky {
      z-index: 11;
    }
  }
</style>
