import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { forward, headWrite, lens, losses, type Trace } from '../src/engine/forward'
import { buildModel, type Manifest } from '../src/engine/model'
import { softmax } from '../src/engine/ops'
import { distribution } from '../src/engine/sample'
import { Tokenizer } from '../src/engine/tokenizer'

const read = (p: string) => readFileSync(new URL(p, import.meta.url))
const manifest: Manifest = JSON.parse(read('../public/model/manifest.json').toString())
const bin = read('../public/model/weights.bin')
const model = buildModel(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength) as ArrayBuffer)
const tok = new Tokenizer(JSON.parse(read('../public/model/tokenizer.json').toString()))
const fx = JSON.parse(read('./fixtures/trace.json').toString())
const cases = JSON.parse(read('./fixtures/tokenizer_cases.json').toString())

const { d_model: D, n_head: H, vocab_size: V } = model.config
const dHead = D / H
const TOL = 1e-3

function maxDiff(got: ArrayLike<number>, want: number[]): number {
  expect(got.length).toBe(want.length)
  let worst = 0
  for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs(got[i] - want[i]))
  return worst
}

// PyTorch stores q, k, v, z as [head, position, dHead]; the engine as [position, D].
function byHead(x: Float32Array, T: number): number[] {
  const out: number[] = []
  for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) for (let d = 0; d < dHead; d++) out.push(x[t * D + h * dHead + d])
  return out
}

describe('tokenizer', () => {
  it('matches the Python tokenizer on 1,000 texts', () => {
    expect(cases.length).toBe(1000)
    for (const c of cases) expect(tok.encode(c.text), JSON.stringify(c.text)).toEqual(c.ids)
  })
  it('decodes back to the original text', () => {
    for (const c of cases) expect(tok.decode(c.ids)).toBe(c.text)
  })
  it('handles the end-of-text marker', () => {
    expect(tok.encode('a<|endoftext|>b')).toEqual([...tok.encode('a'), tok.eot, ...tok.encode('b')])
  })
})

describe('forward pass matches PyTorch', () => {
  const trace: Trace = forward(model, fx.ids)
  const T = fx.ids.length

  it('tokenizes the fixture sentence identically', () => {
    expect(tok.encode(fx.text)).toEqual(fx.ids)
  })
  it('embeddings', () => {
    expect(maxDiff(trace.tokEmb, fx.tok_emb.flat())).toBeLessThan(TOL)
    expect(maxDiff(trace.embed, fx.embed.flat())).toBeLessThan(TOL)
  })
  it('every stage of every layer', () => {
    fx.layers.forEach((want: any, l: number) => {
      const got = trace.layers[l]
      const check = (name: string, g: ArrayLike<number>, w: number[]) => expect(maxDiff(g, w), `layer ${l} ${name}`).toBeLessThan(TOL)
      check('resid_pre', got.residPre, want.resid_pre.flat())
      check('q', byHead(got.q, T), want.q.flat(2))
      check('k', byHead(got.k, T), want.k.flat(2))
      check('v', byHead(got.v, T), want.v.flat(2))
      check('attn', got.attn, want.attn.flat(2))
      check('z', byHead(got.z, T), want.z.flat(2))
      check('attn_out', got.attnOut, want.attn_out.flat())
      check('resid_mid', got.residMid, want.resid_mid.flat())
      check('mlp_pre', got.mlpPre, want.mlp_pre.flat())
      check('mlp_act', got.mlpAct, want.mlp_act.flat())
      check('mlp_out', got.mlpOut, want.mlp_out.flat())
      check('resid_post', got.residPost, want.resid_post.flat())
    })
  })
  it('final vectors and scores', () => {
    expect(maxDiff(trace.final, fx.final.flat())).toBeLessThan(TOL)
    expect(maxDiff(trace.logits, fx.logits.flat())).toBeLessThan(TOL)
  })
  it('switches: no positions, silenced head, skipped MLP', () => {
    const last = (t: Trace) => t.logits.subarray((T - 1) * V, T * V)
    expect(maxDiff(last(forward(model, fx.ids, { noPos: true })), fx.variants.no_pos)).toBeLessThan(TOL)
    expect(maxDiff(last(forward(model, fx.ids, { ablateHeads: [[1, 2]] })), fx.variants.ablate_head_1_2)).toBeLessThan(TOL)
    expect(maxDiff(last(forward(model, fx.ids, { ablateMlp: [0] })), fx.variants.ablate_mlp_0)).toBeLessThan(TOL)
  })
  it('the lens on the last stream equals the real scores', () => {
    const got = lens(model, trace.layers[trace.layers.length - 1].residPost, T - 1)
    expect(maxDiff(got, fx.logits[T - 1])).toBeLessThan(TOL)
  })
  it('head writes plus the bias add up to the attention output', () => {
    const l = 2
    const t = T - 1
    const sum = new Float32Array(model.w[`blocks.${l}.attn.out.bias`])
    for (let h = 0; h < H; h++) headWrite(model, trace, l, h, t).forEach((x, i) => (sum[i] += x))
    expect(maxDiff(sum, Array.from(trace.layers[l].attnOut.subarray(t * D, (t + 1) * D)))).toBeLessThan(1e-4)
  })
  it('earlier positions do not change when later tokens do', () => {
    const changed = [...fx.ids]
    changed[T - 1] = (changed[T - 1] + 1) % V
    const other = forward(model, changed)
    expect(maxDiff(other.logits.subarray(0, (T - 1) * V), Array.from(trace.logits.subarray(0, (T - 1) * V)))).toBeLessThan(1e-6)
  })
  it('without the mask, tokens read later tokens and rows still sum to 1', () => {
    const open = forward(model, fx.ids, { noMask: true })
    const a = open.layers[0].attn
    let upper = 0
    for (let j = 1; j < T; j++) upper += a[j] // head 0, first token reading later ones
    expect(upper).toBeGreaterThan(0)
    let sum = 0
    for (let j = 0; j < T; j++) sum += a[j]
    expect(sum).toBeCloseTo(1, 4)
  })
  it('scoring only the last position gives the same last scores', () => {
    const fast = forward(model, fx.ids, { lastOnly: true })
    expect(maxDiff(fast.logits.subarray((T - 1) * V), Array.from(trace.logits.subarray((T - 1) * V)))).toBeLessThan(1e-6)
  })
  it('gives one loss per predicted token', () => {
    const ls = losses(trace, V)
    expect(ls.length).toBe(T - 1)
    ls.forEach((x) => expect(x).toBeGreaterThan(0))
  })
})

describe('sampling', () => {
  const logits = [2, 1, 0, -1]
  it('temperature 1 is plain softmax', () => {
    expect(maxDiff(distribution(logits, { temperature: 1 }), Array.from(softmax(logits)))).toBeLessThan(1e-6)
  })
  it('temperature 0 picks the top token', () => {
    expect(Array.from(distribution(logits, { temperature: 0 }))).toEqual([1, 0, 0, 0])
  })
  it('top-k keeps k tokens and renormalises', () => {
    const p = distribution(logits, { temperature: 1, topK: 2 })
    expect(p[2]).toBe(0)
    expect(p[0] + p[1]).toBeCloseTo(1, 5)
  })
  it('top-p keeps the smallest set reaching p', () => {
    const p = distribution(logits, { temperature: 1, topP: 0.7 })
    expect(Array.from(p).filter((x) => x > 0).length).toBe(2)
  })
})
