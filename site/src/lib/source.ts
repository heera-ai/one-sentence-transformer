// The real Python source of the model, so the course shows the code that actually ran.
import modelPy from '../../../model/model.py?raw'
import trainPy from '../../../model/train.py?raw'

export { modelPy, trainPy }

// The lines from the one starting with `from` up to (not including) the one starting with `to`.
export function section(src: string, from: string, to?: string): string {
  const lines = src.split('\n')
  const start = lines.findIndex((l) => l.trimStart().startsWith(from))
  let end = to ? lines.findIndex((l, i) => i > start && l.trimStart().startsWith(to)) : lines.length
  if (start < 0) return ''
  if (end < 0) end = lines.length
  const picked = lines.slice(start, end)
  while (picked.length && !picked[picked.length - 1].trim()) picked.pop()
  const indent = Math.min(...picked.filter((l) => l.trim()).map((l) => l.length - l.trimStart().length))
  return picked.map((l) => l.slice(indent)).join('\n')
}
