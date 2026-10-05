// Byte-level BPE, the scheme GPT-2 uses. This must produce exactly the same
// IDs as the Python tokenizer that prepared the training data.

export interface TokenizerData {
  tokens: string[]
  merges: [string, string][]
  eot: number
}

const EOT_TEXT = '<|endoftext|>'

// Split text into chunks that merges may not cross: a word with its leading
// space, a number, a run of punctuation, or whitespace.
const CHUNKS = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

// Every byte gets a printable stand-in character, so tokens can be stored as
// ordinary strings. A space, for example, is shown as "Ġ".
const BYTE_TO_CHAR: string[] = []
const CHAR_TO_BYTE = new Map<string, number>()
{
  const printable = (b: number) => (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)
  let extra = 0
  for (let b = 0; b < 256; b++) {
    const ch = String.fromCharCode(printable(b) ? b : 256 + extra++)
    BYTE_TO_CHAR.push(ch)
    CHAR_TO_BYTE.set(ch, b)
  }
}

export interface MergeStep {
  parts: string[]
  merged: [string, string] | null
}

export class Tokenizer {
  readonly tokens: string[]
  readonly eot: number
  readonly merges: [string, string][]
  private ids = new Map<string, number>()
  private ranks = new Map<string, number>()
  private cache = new Map<string, string[]>()
  private encoder = new TextEncoder()
  private decoder = new TextDecoder('utf-8')

  constructor(data: TokenizerData) {
    this.tokens = data.tokens
    this.eot = data.eot
    this.merges = data.merges
    data.tokens.forEach((t, i) => this.ids.set(t, i))
    data.merges.forEach(([a, b], i) => this.ranks.set(a + ' ' + b, i))
  }

  get vocabSize(): number {
    return this.tokens.length
  }

  private toByteChars(chunk: string): string[] {
    return Array.from(this.encoder.encode(chunk), (b) => BYTE_TO_CHAR[b])
  }

  // Start from single bytes and keep applying the earliest-learned merge
  // that is available, until none applies. Returns every intermediate step.
  mergeSteps(chunk: string): MergeStep[] {
    let parts = this.toByteChars(chunk)
    const steps: MergeStep[] = [{ parts, merged: null }]
    while (parts.length > 1) {
      let best = Infinity
      let at = -1
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.ranks.get(parts[i] + ' ' + parts[i + 1])
        if (rank !== undefined && rank < best) {
          best = rank
          at = i
        }
      }
      if (at === -1) break
      const [a, b] = [parts[at], parts[at + 1]]
      const next: string[] = []
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          next.push(a + b)
          i++
        } else next.push(parts[i])
      }
      parts = next
      steps.push({ parts, merged: [a, b] })
    }
    return steps
  }

  private bpe(chunk: string): string[] {
    let out = this.cache.get(chunk)
    if (!out) {
      const steps = this.mergeSteps(chunk)
      out = steps[steps.length - 1].parts
      this.cache.set(chunk, out)
    }
    return out
  }

  chunks(text: string): string[] {
    return text.match(CHUNKS) ?? []
  }

  encode(text: string): number[] {
    const ids: number[] = []
    text.split(EOT_TEXT).forEach((piece, i) => {
      if (i > 0) ids.push(this.eot)
      for (const chunk of this.chunks(piece)) {
        for (const part of this.bpe(chunk)) ids.push(this.ids.get(part)!)
      }
    })
    return ids
  }

  bytes(id: number): Uint8Array {
    if (id === this.eot) return this.encoder.encode(EOT_TEXT)
    return Uint8Array.from(Array.from(this.tokens[id]), (ch) => CHAR_TO_BYTE.get(ch)!)
  }

  decode(ids: number[]): string {
    const all: number[] = []
    for (const id of ids) all.push(...this.bytes(id))
    return this.decoder.decode(Uint8Array.from(all))
  }

  // The text of one token. A token can be a fragment of a multi-byte
  // character, in which case it has no readable text of its own.
  text(id: number): string {
    return this.decoder.decode(this.bytes(id))
  }

  // Text for display: spaces and newlines made visible.
  label(id: number): string {
    return this.text(id).replace(/ /g, '␣').replace(/\n/g, '⏎')
  }

  static displayPart(part: string): string {
    const bytes = Uint8Array.from(Array.from(part), (ch) => CHAR_TO_BYTE.get(ch)!)
    return new TextDecoder().decode(bytes).replace(/ /g, '␣').replace(/\n/g, '⏎')
  }
}
