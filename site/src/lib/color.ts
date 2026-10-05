// Number -> colour. Teal is positive, rose is negative, white is zero.
const POS = [11, 143, 134]
const NEG = [201, 60, 110]
const TALK = [68, 80, 216]
const THINK = [226, 150, 20]

function mix(to: number[], t: number): string {
  const u = Math.max(0, Math.min(1, t))
  const c = to.map((x) => Math.round(255 + (x - 255) * u))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

export function diverge(v: number, max: number): string {
  const t = Math.sqrt(Math.min(1, Math.abs(v) / max)) // sqrt keeps small values visible
  return mix(v >= 0 ? POS : NEG, t)
}

export function talk(p: number): string {
  return mix(TALK, p)
}

export function think(p: number): string {
  return mix(THINK, p)
}

// Readable text colour on top of a tinted background of strength t (0..1).
export function inkOn(t: number): string {
  return t > 0.55 ? '#fff' : '#16213a'
}

export function fmt(v: number, digits = 2): string {
  return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(digits)
}

export function pct(p: number): string {
  if (p >= 0.0995) return (p * 100).toFixed(0) + '%'
  if (p >= 0.001) return (p * 100).toFixed(1) + '%'
  return '<0.1%'
}
