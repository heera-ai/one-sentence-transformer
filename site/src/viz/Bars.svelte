<script lang="ts">
  // Horizontal bars: one row per item, e.g. candidate next tokens.
  export interface Bar {
    label: string
    value: number
    text?: string // shown at the right; defaults to the value
    note?: string
    mark?: boolean
    faded?: boolean
  }
  interface Props {
    items: Bar[]
    max?: number
    color?: string
    onpick?: (i: number) => void
    signed?: boolean
    name: string
  }
  let { items, max, color = 'var(--ink)', onpick, signed = false, name }: Props = $props()
  const top = $derived(max ?? Math.max(1e-9, ...items.map((b) => Math.abs(b.value))))
</script>

<div class="bars" role="group" aria-label={name}>
  {#each items as b, i}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <svelte:element this={onpick ? 'button' : 'div'} class="row" class:mark={b.mark} class:faded={b.faded} onclick={onpick ? () => onpick(i) : undefined}>
      <span class="label">{b.label}</span>
      <span class="track" class:signed>
        {#if signed}
          <span
            class="fill"
            style:left={b.value >= 0 ? '50%' : `${50 - (Math.abs(b.value) / top) * 50}%`}
            style:width="{(Math.abs(b.value) / top) * 50}%"
            style:background={b.value >= 0 ? 'var(--pos)' : 'var(--neg)'}
          ></span>
          <span class="zero"></span>
        {:else}
          <span class="fill" style:width="{Math.min(100, (b.value / top) * 100)}%" style:background={color}></span>
        {/if}
      </span>
      <span class="val">{b.text ?? b.value.toFixed(2)}</span>
      {#if b.note}<span class="note">{b.note}</span>{/if}
    </svelte:element>
  {/each}
</div>

<style>
  .bars {
    display: grid;
    gap: 3px;
    max-width: 34rem;
  }
  .row {
    display: grid;
    grid-template-columns: 7.5rem 1fr 4rem auto;
    align-items: center;
    gap: 0.6rem;
    background: none;
    border: 0;
    padding: 2px 4px;
    border-radius: 6px;
    text-align: left;
    width: 100%;
  }
  button.row:hover {
    background: #edf1f5;
  }
  .row.mark .label {
    font-weight: 700;
  }
  .row.mark {
    background: var(--think-soft);
  }
  .row.faded {
    opacity: 0.35;
  }
  .label {
    font-family: var(--mono);
    font-size: 0.84rem;
    white-space: pre;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .track {
    position: relative;
    height: 14px;
    background: #edf1f5;
    border-radius: 3px;
    overflow: hidden;
  }
  .fill {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    border-radius: 3px;
    transition: width 0.25s ease, left 0.25s ease;
  }
  .zero {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background: var(--ink-3);
  }
  .val {
    font-family: var(--mono);
    font-size: 0.78rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .note {
    font-size: 0.78rem;
    color: var(--ink-2);
  }
</style>
