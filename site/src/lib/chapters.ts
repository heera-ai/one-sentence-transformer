// The course, in order. `stage` says which part of the machine a chapter opens up.
export type Stage = 'whole' | 'tokens' | 'vectors' | 'position' | 'talk' | 'think' | 'block' | 'vote' | 'train'

export interface ChapterInfo {
  slug: string
  title: string
  stage: Stage
  takeaway: string
}

export const chapters: ChapterInfo[] = [
  { slug: 'game', title: 'The game', stage: 'whole', takeaway: 'A language model does one thing: given some text, it gives every possible next token a probability.' },
  { slug: 'tokens', title: 'Tokens', stage: 'tokens', takeaway: 'Text is cut into tokens from a fixed vocabulary, and each token is just an ID number.' },
  { slug: 'embeddings', title: 'Embeddings', stage: 'vectors', takeaway: 'Each ID looks up a learned vector. Similar tokens point in similar directions; the dot product measures that.' },
  { slug: 'position', title: 'Position', stage: 'position', takeaway: 'Attention cannot see order by itself, so a position vector is added to each token vector.' },
  { slug: 'attention', title: 'Attention, one head', stage: 'talk', takeaway: 'Talk: a query is matched against earlier keys; softmax turns the scores into weights; the weighted values are collected.' },
  { slug: 'heads', title: 'Many heads, one stream', stage: 'talk', takeaway: 'Several heads look for different things at once, and each adds what it found to the token’s running vector, the stream.' },
  { slug: 'mlp', title: 'The MLP', stage: 'think', takeaway: 'Think: each position alone passes through a wide layer of neurons that detect patterns and write new information to the stream.' },
  { slug: 'block', title: 'The full block, stacked', stage: 'block', takeaway: 'A layer is normalise, talk, add, normalise, think, add. Stack layers and the prediction sharpens step by step.' },
  { slug: 'vote', title: 'From numbers to a word', stage: 'vote', takeaway: 'The last position’s final vector is scored against every token; softmax gives probabilities; one token is sampled and appended.' },
  { slug: 'training', title: 'Training', stage: 'train', takeaway: 'Training measures surprise at every position and nudges every weight downhill, millions of times.' },
  { slug: 'build', title: 'Build it', stage: 'whole', takeaway: 'The whole model is about 150 lines of PyTorch; nothing in it is beyond what you have already seen.' },
  { slug: 'family', title: 'The other transformers', stage: 'whole', takeaway: 'A GPT is the decoder alone. Remove the causal mask and the same layer is an encoder; add a second attention that reads another stack and you have the original encoder-decoder.' },
  { slug: 'scale', title: 'Scale it', stage: 'whole', takeaway: 'Big models are the same machine with larger numbers, plus engineering to make those numbers affordable.' },
]

export const indexOf = (slug: string) => chapters.findIndex((c) => c.slug === slug)
