import { softmax } from './ops'

export interface SampleOptions {
  temperature: number
  topK?: number // keep only the K most likely tokens
  topP?: number // keep the smallest set of tokens whose probability adds up to P
}

// The probabilities the next token is actually drawn from.
export function distribution(logits: ArrayLike<number>, opts: SampleOptions): Float32Array {
  const n = logits.length
  if (opts.temperature <= 0) {
    // Temperature 0 means "always take the top one".
    let best = 0
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i
    const out = new Float32Array(n)
    out[best] = 1
    return out
  }
  const scaled = new Float32Array(n)
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / opts.temperature
  let probs = softmax(scaled)

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b] - probs[a])
  let keep = n
  if (opts.topK && opts.topK < keep) keep = opts.topK
  if (opts.topP && opts.topP < 1) {
    let total = 0
    let count = 0
    while (count < n && total < opts.topP) total += probs[order[count++]]
    keep = Math.min(keep, count)
  }
  if (keep < n) {
    const filtered = new Float32Array(n)
    let sum = 0
    for (let r = 0; r < keep; r++) sum += probs[order[r]]
    for (let r = 0; r < keep; r++) filtered[order[r]] = probs[order[r]] / sum
    probs = filtered
  }
  return probs
}

export function draw(probs: ArrayLike<number>, random: () => number = Math.random): number {
  let r = random()
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i]
    if (r <= 0) return i
  }
  return probs.length - 1
}
