import { dot, norm } from '../engine/ops'

// Columns [from, from + width) of a [T, D] matrix, as a [T, width] matrix.
export function columns(x: Float32Array, T: number, D: number, from: number, width: number): Float32Array {
  const out = new Float32Array(T * width)
  for (let t = 0; t < T; t++) out.set(x.subarray(t * D + from, t * D + from + width), t * width)
  return out
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return dot(a, b) / (norm(a) * norm(b) + 1e-12)
}

export function mean(xs: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < xs.length; i++) s += xs[i]
  return xs.length ? s / xs.length : 0
}

// A small seeded random number generator, so "random" demos can be replayed.
export function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function big(n: number): string {
  if (n >= 1e12) return (n / 1e12).toFixed(n >= 1e13 ? 0 : 1) + ' trillion'
  if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + ' billion'
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + ' million'
  if (n >= 1e4) return Math.round(n / 1e3) + ' thousand'
  return Math.round(n).toLocaleString('en')
}

export function bytes(n: number): string {
  if (n >= 1e12) return (n / 1e12).toFixed(1) + ' TB'
  if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB'
  if (n >= 1e6) return (n / 1e6).toFixed(1) + ' MB'
  return (n / 1e3).toFixed(0) + ' kB'
}
