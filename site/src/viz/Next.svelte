<script lang="ts">
  // The model's top guesses for the next token, as bars.
  import type { Candidate } from '../lib/state.svelte'
  import { pct } from '../lib/color'
  import { app } from '../lib/state.svelte'
  import Bars from './Bars.svelte'

  interface Props {
    items: Candidate[]
    onpick?: (id: number) => void
    mark?: number // a token ID to highlight
    color?: string
    name?: string
    max?: number // share one scale between two charts
  }
  let { items, onpick, mark = -1, color = 'var(--ink)', name = 'Most likely next tokens', max }: Props = $props()
</script>

<Bars
  {name}
  {color}
  max={max ?? Math.max(0.05, items[0]?.p ?? 1)}
  items={items.map((c) => ({ label: app.label(c.id), value: c.p, text: pct(c.p), mark: c.id === mark }))}
  onpick={onpick ? (i) => onpick(items[i].id) : undefined}
/>
