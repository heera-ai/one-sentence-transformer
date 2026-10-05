<script lang="ts">
  // The sentence as a row of token chips. Optionally selectable and tinted.
  import { inkOn, talk } from '../lib/color'
  import { app } from '../lib/state.svelte'

  interface Props {
    ids: number[]
    selected?: number
    onselect?: (i: number) => void
    weights?: ArrayLike<number> | null // tint each chip by a value in 0..1
    tint?: (t: number) => string
    sub?: (i: number) => string // small text under each chip
    dim?: (i: number) => boolean
    name?: string
  }
  let { ids, selected = -1, onselect, weights = null, tint = talk, sub, dim, name = 'Tokens' }: Props = $props()
</script>

<div class="chips" role={onselect ? 'listbox' : 'list'} aria-label={name}>
  {#each ids as id, i}
    {@const w = weights ? weights[i] : 0}
    <svelte:element
      this={onselect ? 'button' : 'span'}
      class="chip"
      class:selected={i === selected}
      class:dim={dim?.(i)}
      role={onselect ? 'option' : 'listitem'}
      aria-selected={onselect ? i === selected : undefined}
      style:background={weights ? tint(w) : undefined}
      style:color={weights ? inkOn(w) : undefined}
      onclick={onselect ? () => onselect(i) : undefined}
    >
      <span class="t">{app.label(id)}</span>
      {#if sub}<span class="s">{sub(i)}</span>{/if}
    </svelte:element>
  {/each}
</div>

<style>
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .chip {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    font-family: var(--mono);
    font-size: 0.86rem;
    line-height: 1.2;
    background: #edf1f5;
    border: 1.5px solid transparent;
    border-radius: 7px;
    padding: 0.32rem 0.5rem;
    white-space: pre;
  }
  button.chip:hover {
    border-color: var(--ink-3);
  }
  .chip.selected {
    border-color: var(--ink);
    box-shadow: 0 2px 0 var(--ink);
  }
  .chip.dim {
    opacity: 0.32;
  }
  .s {
    font-size: 0.68rem;
    opacity: 0.75;
    margin-top: 0.15rem;
  }
</style>
