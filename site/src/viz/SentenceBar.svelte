<script lang="ts">
  // Always on screen: the learner's sentence, and what the model expects next.
  import { pct } from '../lib/color'
  import { app } from '../lib/state.svelte'

  let { onmenu }: { onmenu: () => void } = $props()
  const top = $derived(app.next(app.last, 1)[0])
</script>

<div class="bar">
  <button class="menu" onclick={onmenu} aria-label="Show chapters">☰</button>
  <label for="sentence">Your sentence</label>
  <input
    id="sentence"
    type="text"
    value={app.text}
    oninput={(e) => app.setText(e.currentTarget.value)}
    spellcheck="false"
    autocomplete="off"
    placeholder="Type the start of a simple story"
  />
  <div class="out" aria-live="polite">
    {#if top}
      <span class="muted">next</span>
      <span class="tok">{app.label(top.id)}</span>
      <span class="num">{pct(top.p)}</span>
    {:else if app.ready}
      <span class="muted">Type something to begin</span>
    {/if}
  </div>
</div>
{#if app.truncated > 0}
  <p class="warn">The model reads at most {app.model?.config.block_size} tokens, so it is using only the last {app.ids.length} of your {app.allIds.length}.</p>
{/if}

<style>
  .bar {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    font-family: var(--display);
    font-size: 0.92rem;
  }
  label {
    font-weight: 620;
    white-space: nowrap;
  }
  input {
    font-family: var(--mono);
    font-size: 0.92rem;
    border: 1.5px solid var(--line);
    border-radius: 9px;
    padding: 0.5rem 0.75rem;
    background: var(--paper);
    flex: 1;
    min-width: 0;
  }
  input:focus {
    border-color: var(--ink);
    outline: none;
    box-shadow: 0 2px 0 var(--ink);
  }
  .out {
    display: flex;
    gap: 0.5rem;
    align-items: baseline;
    white-space: nowrap;
    min-width: 9rem;
  }
  .tok {
    font-family: var(--mono);
    font-weight: 700;
    background: var(--ink);
    color: #fff;
    border-radius: 6px;
    padding: 0.15rem 0.45rem;
    white-space: pre;
  }
  .menu {
    display: none;
    background: none;
    border: 1.5px solid var(--line);
    border-radius: 8px;
    padding: 0.3rem 0.6rem;
    font-size: 1.05rem;
    line-height: 1;
  }
  .warn {
    font-family: var(--display);
    font-size: 0.84rem;
    color: var(--think);
    margin: 0.4rem 0 0;
    max-width: none;
  }
  @media (max-width: 900px) {
    label {
      position: absolute;
      left: -999rem;
    }
    .menu {
      display: block;
    }
    .out {
      min-width: 0;
    }
    .out .muted {
      display: none;
    }
  }
</style>
