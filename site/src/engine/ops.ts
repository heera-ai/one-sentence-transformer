// The handful of operations a transformer is made of, on flat arrays.
// A "matrix" of T rows and D columns is one Float32Array of length T * D.

export type Vec = Float32Array

// y = W x + b for each of the T rows of x. W has shape [nOut, nIn].
export function linear(x: Vec, W: Vec, b: Vec | null, T: number, nIn: number, nOut: number): Vec {
  const y = new Float32Array(T * nOut)
  for (let t = 0; t < T; t++) {
    const xo = t * nIn
    for (let o = 0; o < nOut; o++) {
      let sum = b ? b[o] : 0
      const wo = o * nIn
      for (let i = 0; i < nIn; i++) sum += W[wo + i] * x[xo + i]
      y[t * nOut + o] = sum
    }
  }
  return y
}

// Shift each row to mean 0 and scale it to variance 1, then apply a learned
// gain and offset per column.
export function layerNorm(x: Vec, gain: Vec, bias: Vec, T: number, D: number): Vec {
  const y = new Float32Array(T * D)
  for (let t = 0; t < T; t++) {
    const o = t * D
    let mean = 0
    for (let i = 0; i < D; i++) mean += x[o + i]
    mean /= D
    let variance = 0
    for (let i = 0; i < D; i++) variance += (x[o + i] - mean) ** 2
    const scale = 1 / Math.sqrt(variance / D + 1e-5)
    for (let i = 0; i < D; i++) y[o + i] = (x[o + i] - mean) * scale * gain[i] + bias[i]
  }
  return y
}

// GELU, in the tanh form GPT-2 uses: roughly "pass positives, squash negatives".
const GELU_C = Math.sqrt(2 / Math.PI)
export function gelu(x: Vec): Vec {
  const y = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) {
    const v = x[i]
    y[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)))
  }
  return y
}

// Turn any list of scores into positive numbers that sum to 1.
export function softmax(scores: ArrayLike<number>): Float32Array {
  let max = -Infinity
  for (let i = 0; i < scores.length; i++) if (scores[i] > max) max = scores[i]
  const out = new Float32Array(scores.length)
  let sum = 0
  for (let i = 0; i < scores.length; i++) {
    out[i] = Math.exp(scores[i] - max)
    sum += out[i]
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum
  return out
}

export function add(a: Vec, b: Vec): Vec {
  const y = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i]
  return y
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>, n = a.length, ao = 0, bo = 0): number {
  let s = 0
  for (let i = 0; i < n; i++) s += a[ao + i] * b[bo + i]
  return s
}

export function norm(a: ArrayLike<number>): number {
  return Math.sqrt(dot(a, a))
}

export function row(x: Vec, t: number, D: number): Vec {
  return x.subarray(t * D, (t + 1) * D)
}

// Indices of the k largest values, largest first.
export function topK(values: ArrayLike<number>, k: number): number[] {
  const idx = Array.from({ length: values.length }, (_, i) => i)
  idx.sort((a, b) => values[b] - values[a])
  return idx.slice(0, k)
}
