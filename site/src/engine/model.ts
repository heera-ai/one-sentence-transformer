import type { TokenizerData } from './tokenizer'

export interface Config {
  vocab_size: number
  block_size: number
  n_layer: number
  n_head: number
  d_model: number
}

export interface TensorInfo {
  name: string
  shape: number[]
  offset: number
}

export interface Checkpoint {
  step: number
  file: string
  val: number | null
}

export interface LogEntry {
  step: number
  train: number
  val: number
  lr: number
  secs?: number
}

export interface Manifest {
  config: Config
  params: number
  step: number
  tensors: TensorInfo[]
  checkpoints: Checkpoint[]
  trainlog: LogEntry[]
}

export interface NeuronExample {
  a: number
  ctx: number[]
}

export interface Extras {
  counts: number[]
  map: { id: number; x: number; y: number }[]
  neurons: NeuronExample[][][]
}

export interface Model {
  config: Config
  step: number
  w: Record<string, Float32Array>
}

// Weights are stored as 16-bit floats to halve the download.
export function decodeFloat16(buffer: ArrayBuffer): Float32Array {
  const half = new Uint16Array(buffer)
  const out = new Float32Array(half.length)
  for (let i = 0; i < half.length; i++) {
    const h = half[i]
    const sign = h & 0x8000 ? -1 : 1
    const exp = (h >> 10) & 0x1f
    const frac = h & 0x3ff
    if (exp === 0) out[i] = sign * 2 ** -14 * (frac / 1024)
    else if (exp === 31) out[i] = frac ? NaN : sign * Infinity
    else out[i] = sign * 2 ** (exp - 15) * (1 + frac / 1024)
  }
  return out
}

export function buildModel(manifest: Manifest, buffer: ArrayBuffer, step = manifest.step): Model {
  const all = decodeFloat16(buffer)
  const w: Record<string, Float32Array> = {}
  for (const t of manifest.tensors) {
    const size = t.shape.reduce((a, b) => a * b, 1)
    w[t.name] = all.subarray(t.offset, t.offset + size)
  }
  return { config: manifest.config, step, w }
}

const BASE = `${import.meta.env?.BASE_URL ?? '/'}model/`

export async function fetchJson<T>(file: string): Promise<T> {
  const res = await fetch(BASE + file)
  if (!res.ok) throw new Error(`Could not load ${file} (${res.status})`)
  return res.json()
}

export async function fetchModel(manifest: Manifest, file = 'weights.bin', step = manifest.step): Promise<Model> {
  const res = await fetch(BASE + file)
  if (!res.ok) throw new Error(`Could not load ${file} (${res.status})`)
  return buildModel(manifest, await res.arrayBuffer(), step)
}

export type { TokenizerData }
