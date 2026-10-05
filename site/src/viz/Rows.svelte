<script lang="ts">
  // One row of cells per token, with the token's text beside it.
  import { fmt } from '../lib/color'
  import { app } from '../lib/state.svelte'
  import Cells from './Cells.svelte'

  interface Props {
    ids: number[]
    data: ArrayLike<number>
    cols: number
    cw?: number
    ch?: number
    max?: number
    mode?: 'diverge' | 'talk' | 'think'
    selected?: number
    onselect?: (row: number) => void
    unit?: string
    name: string
    dimRow?: (i: number) => boolean
  }
  let { ids, data, cols, cw = 5, ch = 20, max, mode = 'diverge', selected = -1, onselect, unit = 'dimension', name, dimRow }: Props = $props()
  let hover = $state('')
</script>

<div class="scroll-x">
  <div class="rows">
    <div class="labels">
      {#each ids as id, i}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <svelte:element
          this={onselect ? 'button' : 'span'}
          class="label"
          class:selected={i === selected}
          class:dim={dimRow?.(i)}
          style:height="{ch + 1}px"
          onclick={onselect ? () => onselect(i) : undefined}>{app.label(id)}</svelte:element
        >
      {/each}
    </div>
    <Cells
      {data}
      rows={ids.length}
      {cols}
      {cw}
      {ch}
      {max}
      {mode}
      label={name}
      markRow={selected}
      onhover={(r, c, v) => (hover = `${app.label(ids[r])}  ·  ${unit} ${c}  ·  ${Number.isFinite(v) ? fmt(v, 3) : 'hidden'}`)}
      onleave={() => (hover = '')}
      onpick={onselect ? (r) => onselect(r) : undefined}
    />
  </div>
</div>
<p class="readout" aria-live="off">{hover || 'Hover over a cell to read its value.'}</p>

<style>
  .rows {
    display: flex;
    gap: 8px;
    align-items: flex-start;
  }
  .labels {
    display: flex;
    flex-direction: column;
    flex: none;
    min-width: 5.5rem;
  }
  .label {
    font-family: var(--mono);
    font-size: 0.78rem;
    text-align: right;
    white-space: pre;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    background: none;
    border: 0;
    padding: 0 4px;
    border-radius: 4px;
  }
  button.label:hover {
    background: #edf1f5;
  }
  .label.selected {
    font-weight: 700;
    background: #dfe5ec;
  }
  .label.dim {
    opacity: 0.35;
  }
</style>
