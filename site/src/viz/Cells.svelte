<script lang="ts">
  // A grid of coloured cells drawn on a canvas: one cell per number.
  import { diverge, talk, think } from '../lib/color'

  interface Props {
    data: ArrayLike<number>
    rows: number
    cols: number
    cw?: number
    ch?: number
    gap?: number
    max?: number
    mode?: 'diverge' | 'talk' | 'think'
    label: string
    onhover?: (row: number, col: number, value: number) => void
    onleave?: () => void
    onpick?: (row: number, col: number) => void
    markRow?: number
    markCol?: number
  }
  let { data, rows, cols, cw = 5, ch = 22, gap = 1, max, mode = 'diverge', label, onhover, onleave, onpick, markRow = -1, markCol = -1 }: Props = $props()

  let canvas: HTMLCanvasElement
  const width = $derived(cols * (cw + gap) - gap)
  const height = $derived(rows * (ch + gap) - gap)
  const scale = $derived.by(() => {
    if (max) return max
    let m = 1e-9
    for (let i = 0; i < rows * cols; i++) if (Number.isFinite(data[i])) m = Math.max(m, Math.abs(data[i]))
    return m
  })

  $effect(() => {
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[r * cols + c]
        if (!Number.isFinite(v)) {
          ctx.fillStyle = '#edf1f5'
        } else if (mode === 'diverge') ctx.fillStyle = diverge(v, scale)
        else ctx.fillStyle = (mode === 'talk' ? talk : think)(Math.max(0, v) / scale)
        ctx.fillRect(c * (cw + gap), r * (ch + gap), cw, ch)
      }
    }
    ctx.strokeStyle = '#16213a'
    ctx.lineWidth = 2
    if (markRow >= 0 && markCol >= 0) ctx.strokeRect(markCol * (cw + gap) + 1, markRow * (ch + gap) + 1, cw - 2, ch - 2)
    else if (markRow >= 0) ctx.strokeRect(1, markRow * (ch + gap) + 1, width - 2, ch - 2)
    else if (markCol >= 0) ctx.strokeRect(markCol * (cw + gap) + 1, 1, cw - 2, height - 2)
  })

  function cell(e: MouseEvent): [number, number] {
    const box = canvas.getBoundingClientRect()
    const c = Math.min(cols - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * cols)))
    const r = Math.min(rows - 1, Math.max(0, Math.floor(((e.clientY - box.top) / box.height) * rows)))
    return [r, c]
  }
</script>

<!-- svelte-ignore a11y_no_interactive_element_to_noninteractive_role -->
<canvas
  bind:this={canvas}
  role="img"
  aria-label={label}
  style:width="{width}px"
  style:height="{height}px"
  style:cursor={onpick ? 'pointer' : onhover ? 'crosshair' : 'default'}
  onmousemove={(e) => {
    const [r, c] = cell(e)
    onhover?.(r, c, data[r * cols + c])
  }}
  onmouseleave={() => onleave?.()}
  onclick={(e) => {
    const [r, c] = cell(e)
    onpick?.(r, c)
  }}
></canvas>

<style>
  canvas {
    display: block;
    border-radius: 3px;
    flex: none;
  }
</style>
