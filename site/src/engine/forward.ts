// The whole transformer, run forwards, keeping every intermediate result.
//
//   token IDs -> vectors -> [ talk (attention), think (MLP) ] x n_layer -> scores
//
// This mirrors model/model.py line for line.

import type { Model } from './model'
import { add, gelu, layerNorm, linear, softmax, type Vec } from './ops'

export interface LayerTrace {
  residPre: Vec // [T, D] the stream entering this layer
  q: Vec // [T, D]; head h owns columns h*dHead .. (h+1)*dHead
  k: Vec
  v: Vec
  scores: Vec // [H, T, T] query·key, before softmax (-Infinity where masked)
  attn: Vec // [H, T, T] each row sums to 1
  z: Vec // [T, D] the weighted average of values, heads side by side
  attnOut: Vec // [T, D] what attention adds to the stream
  residMid: Vec
  mlpPre: Vec // [T, 4D]
  mlpAct: Vec // [T, 4D]
  mlpOut: Vec // [T, D] what the MLP adds to the stream
  residPost: Vec
}

export interface Trace {
  ids: number[]
  T: number
  tokEmb: Vec // [T, D]
  posEmb: Vec // [T, D]
  embed: Vec // [T, D]
  layers: LayerTrace[]
  final: Vec // [T, D] after the last layer norm
  logits: Vec // [T, V]
}

export interface Options {
  noPos?: boolean // leave out the position vectors
  ablateHeads?: [number, number][] // [layer, head] pairs to silence
  ablateMlp?: number[] // layers whose MLP is skipped
  noMask?: boolean // let positions look at the future (the model was never trained this way)
  lastOnly?: boolean // compute scores for the last position only (faster when writing text)
}

export function forward(model: Model, ids: number[], opts: Options = {}): Trace {
  const { d_model: D, n_head: H, n_layer: L, block_size, vocab_size: V } = model.config
  const w = model.w
  const T = ids.length
  if (T === 0 || T > block_size) throw new Error(`Need between 1 and ${block_size} tokens, got ${T}`)
  const dHead = D / H

  // 1. Look up one vector per token, and one per position.
  const tokEmb = new Float32Array(T * D)
  const posEmb = new Float32Array(T * D)
  for (let t = 0; t < T; t++) {
    tokEmb.set(w['wte.weight'].subarray(ids[t] * D, (ids[t] + 1) * D), t * D)
    posEmb.set(w['wpe.weight'].subarray(t * D, (t + 1) * D), t * D)
  }
  const embed = opts.noPos ? tokEmb.slice() : add(tokEmb, posEmb)

  let x = embed
  const layers: LayerTrace[] = []
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`
    const residPre = x

    // 2. Talk. Each position builds a query, a key and a value.
    const h1 = layerNorm(x, w[p + 'ln1.weight'], w[p + 'ln1.bias'], T, D)
    const q = linear(h1, w[p + 'attn.query.weight'], w[p + 'attn.query.bias'], T, D, D)
    const k = linear(h1, w[p + 'attn.key.weight'], w[p + 'attn.key.bias'], T, D, D)
    const v = linear(h1, w[p + 'attn.value.weight'], w[p + 'attn.value.bias'], T, D, D)

    const scores = new Float32Array(H * T * T)
    const attn = new Float32Array(H * T * T)
    const z = new Float32Array(T * D)
    const scale = 1 / Math.sqrt(dHead)
    for (let h = 0; h < H; h++) {
      const silenced = opts.ablateHeads?.some(([al, ah]) => al === l && ah === h)
      const col = h * dHead
      for (let i = 0; i < T; i++) {
        const rowAt = (h * T + i) * T
        // Score = this position's query · every earlier position's key.
        for (let j = 0; j < T; j++) {
          if (j > i && !opts.noMask) {
            scores[rowAt + j] = -Infinity
            continue
          }
          let s = 0
          for (let d = 0; d < dHead; d++) s += q[i * D + col + d] * k[j * D + col + d]
          scores[rowAt + j] = s * scale
        }
        const weights = softmax(scores.subarray(rowAt, rowAt + T))
        attn.set(weights, rowAt)
        if (silenced) continue
        // Collect the values, weighted by attention.
        for (let j = 0; j < T; j++) {
          const wgt = weights[j]
          if (wgt === 0) continue
          for (let d = 0; d < dHead; d++) z[i * D + col + d] += wgt * v[j * D + col + d]
        }
      }
    }
    const attnOut = linear(z, w[p + 'attn.out.weight'], w[p + 'attn.out.bias'], T, D, D)
    const residMid = add(x, attnOut)

    // 3. Think. Each position is processed on its own.
    const h2 = layerNorm(residMid, w[p + 'ln2.weight'], w[p + 'ln2.bias'], T, D)
    const mlpPre = linear(h2, w[p + 'mlp.fc.weight'], w[p + 'mlp.fc.bias'], T, D, 4 * D)
    const mlpAct = gelu(mlpPre)
    const mlpOut = linear(mlpAct, w[p + 'mlp.proj.weight'], w[p + 'mlp.proj.bias'], T, 4 * D, D)
    const residPost = opts.ablateMlp?.includes(l) ? residMid : add(residMid, mlpOut)

    layers.push({ residPre, q, k, v, scores, attn, z, attnOut, residMid, mlpPre, mlpAct, mlpOut, residPost })
    x = residPost
  }

  // 4. Vote. Compare the final vector with every token's embedding.
  const final = layerNorm(x, w['ln_f.weight'], w['ln_f.bias'], T, D)
  let logits: Vec
  if (opts.lastOnly) {
    logits = new Float32Array(T * V)
    logits.set(linear(final.subarray((T - 1) * D), w['wte.weight'], null, 1, D, V), (T - 1) * V)
  } else logits = linear(final, w['wte.weight'], null, T, D, V)

  return { ids, T, tokEmb, posEmb, embed, layers, final, logits }
}

// Scores for the next token as they would be if the model stopped here:
// apply the final layer norm and the output matrix to any point in the stream.
export function lens(model: Model, stream: Vec, t: number): Float32Array {
  const { d_model: D, vocab_size: V } = model.config
  const x = stream.slice(t * D, (t + 1) * D)
  const final = layerNorm(x, model.w['ln_f.weight'], model.w['ln_f.bias'], 1, D)
  return linear(final, model.w['wte.weight'], null, 1, D, V)
}

// What one head alone adds to the stream at position t (its slice of the output matrix).
export function headWrite(model: Model, trace: Trace, l: number, h: number, t: number): Float32Array {
  const { d_model: D, n_head: H } = model.config
  const dHead = D / H
  const W = model.w[`blocks.${l}.attn.out.weight`]
  const z = trace.layers[l].z
  const out = new Float32Array(D)
  for (let o = 0; o < D; o++) {
    let s = 0
    for (let d = 0; d < dHead; d++) s += W[o * D + h * dHead + d] * z[t * D + h * dHead + d]
    out[o] = s
  }
  return out
}

export function logitsAt(trace: Trace, t: number, V: number): Float32Array {
  return trace.logits.subarray(t * V, (t + 1) * V)
}

// How surprised the model was by each actual next token: -ln(probability it gave).
export function losses(trace: Trace, V: number): number[] {
  const out: number[] = []
  for (let t = 0; t < trace.T - 1; t++) {
    const p = softmax(logitsAt(trace, t, V))
    out.push(-Math.log(Math.max(p[trace.ids[t + 1]], 1e-12)))
  }
  return out
}
