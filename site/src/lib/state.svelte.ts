import { forward, logitsAt, type Trace } from '../engine/forward'
import { fetchJson, fetchModel, type Extras, type Manifest, type Model } from '../engine/model'
import { softmax, topK } from '../engine/ops'
import { Tokenizer, type TokenizerData } from '../engine/tokenizer'
import { chapters } from './chapters'

const DEFAULT_TEXT = 'The little dog ran to the park and saw a big'

function stored(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback
  } catch {
    return fallback
  }
}
function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    /* private mode: carry on without saving */
  }
}

export interface Candidate {
  id: number
  p: number
  logit: number
}

export function candidates(logits: ArrayLike<number>, k: number): Candidate[] {
  const p = softmax(logits)
  return topK(p, k).map((id) => ({ id, p: p[id], logit: logits[id] }))
}

class App {
  text = $state(stored('sentence', DEFAULT_TEXT))
  focus = $state(-1) // the position being inspected; -1 means "the last one"
  slug = $state('game')
  visited = $state<string[]>(JSON.parse(stored('visited', '[]')))
  error = $state('')

  model = $state.raw<Model | null>(null)
  manifest = $state.raw<Manifest | null>(null)
  tok = $state.raw<Tokenizer | null>(null)
  extras = $state.raw<Extras | null>(null)
  private snapshots = new Map<number, Model>()

  ready = $derived(this.model !== null && this.tok !== null)
  allIds = $derived(this.tok ? this.tok.encode(this.text) : [])
  // The model can read at most block_size tokens; keep the most recent ones.
  ids = $derived(this.model ? this.allIds.slice(-this.model.config.block_size) : this.allIds)
  truncated = $derived(this.allIds.length - this.ids.length)
  trace = $derived<Trace | null>(this.model && this.ids.length ? forward(this.model, this.ids) : null)
  pos = $derived(this.focus < 0 || this.focus >= this.ids.length ? this.ids.length - 1 : this.focus)
  last = $derived(this.ids.length - 1)

  async load() {
    try {
      const [manifest, tokData] = await Promise.all([fetchJson<Manifest>('manifest.json'), fetchJson<TokenizerData>('tokenizer.json')])
      this.manifest = manifest
      this.tok = new Tokenizer(tokData)
      this.model = await fetchModel(manifest)
      this.snapshots.set(manifest.step, this.model)
      fetchJson<Extras>('extras.json').then((e) => (this.extras = e))
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e)
    }
  }

  async snapshot(step: number): Promise<Model> {
    const hit = this.snapshots.get(step)
    if (hit) return hit
    const ck = this.manifest!.checkpoints.find((c) => c.step === step)!
    const model = await fetchModel(this.manifest!, ck.file, step)
    this.snapshots.set(step, model)
    return model
  }

  setText(text: string) {
    this.text = text
    this.focus = -1
    store('sentence', text)
  }

  append(id: number) {
    this.setText(this.text + this.tok!.text(id))
  }

  go(slug: string) {
    if (!chapters.some((c) => c.slug === slug)) slug = 'game'
    this.slug = slug
    if (!this.visited.includes(slug)) {
      this.visited = [...this.visited, slug]
      store('visited', JSON.stringify(this.visited))
    }
  }

  // The model's top guesses for what follows position t.
  next(t = this.last, k = 8): Candidate[] {
    if (!this.trace || !this.model) return []
    return candidates(logitsAt(this.trace, t, this.model.config.vocab_size), k)
  }

  label(id: number): string {
    return this.tok ? this.tok.label(id) : ''
  }
}

export const app = new App()
