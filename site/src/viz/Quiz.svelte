<script lang="ts">
  // Commit to an answer first; the explanation appears only afterwards.
  export interface Option {
    text: string
    correct?: boolean
    why: string
  }
  let { question, options }: { question: string; options: Option[] } = $props()
  let picked = $state(-1)
</script>

<div class="quiz">
  <p class="q">{question}</p>
  <div class="opts">
    {#each options as o, i}
      <button class="opt" class:right={picked >= 0 && o.correct} class:wrong={picked === i && !o.correct} disabled={picked >= 0} onclick={() => (picked = i)}>
        {o.text}
      </button>
    {/each}
  </div>
  {#if picked >= 0}
    <p class="why" role="status">
      <strong>{options[picked].correct ? 'Yes.' : 'Not quite.'}</strong>
      {options[picked].why}
      {#if !options[picked].correct}{options.find((o) => o.correct)?.why}{/if}
      <button class="again" onclick={() => (picked = -1)}>Try again</button>
    </p>
  {/if}
</div>

<style>
  .quiz {
    border-top: 1px solid var(--line);
    padding: 1.1rem 0 0.4rem;
    max-width: var(--measure);
    font-family: var(--display);
    font-size: 0.98rem;
    line-height: 1.45;
  }
  .q {
    font-weight: 620;
    margin-bottom: 0.7rem;
  }
  .opts {
    display: grid;
    gap: 6px;
  }
  .opt {
    text-align: left;
    background: var(--paper);
    border: 1.5px solid var(--line);
    border-radius: 9px;
    padding: 0.55rem 0.85rem;
  }
  .opt:hover:not(:disabled) {
    border-color: var(--ink);
  }
  .opt:disabled {
    cursor: default;
    color: var(--ink-2);
  }
  .opt.right {
    border-color: var(--pos);
    background: #e2f4f2;
    color: var(--ink);
  }
  .opt.wrong {
    border-color: var(--neg);
    background: #fae8ef;
    color: var(--ink);
  }
  .why {
    margin: 0.8rem 0 0;
    color: var(--ink);
  }
  .again {
    background: none;
    border: 0;
    padding: 0;
    margin-left: 0.4rem;
    color: var(--talk);
    text-decoration: underline;
  }
</style>
