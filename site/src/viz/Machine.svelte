<script lang="ts">
  // The one picture of the whole course: every token has a lane; in each
  // layer the lanes talk, then think; the last lane votes.
  import type { Stage } from '../lib/chapters'

  let { stage }: { stage: Stage } = $props()
  const on = (...s: Stage[]) => stage === 'whole' || stage === 'train' || s.includes(stage)
  const lanes = [34, 64, 94]
</script>

<svg viewBox="0 0 232 318" role="img" aria-label="Diagram of the model: text becomes tokens, tokens become vectors, position is added, then four layers of talk and think, then a vote for the next token.">
  <!-- lanes -->
  {#each lanes as x, i}
    <line x1={x} y1="44" x2={x} y2={i === 2 ? 262 : 246} class="lane" class:faint={i < 2} />
  {/each}

  <a href="#/tokens" class:on={on('tokens')}>
    <title>Tokens</title>
    {#each lanes as x}<rect x={x - 11} y="30" width="22" height="14" rx="4" class="solid" />{/each}
    <text x="122" y="41">tokens</text>
  </a>

  <a href="#/embeddings" class:on={on('vectors')}>
    <title>Embeddings</title>
    {#each lanes as x}
      {#each [0, 1, 2, 3] as c}<rect x={x - 4} y={62 + c * 6} width="8" height="5" class="cell c{(c + x) % 3}" />{/each}
    {/each}
    <text x="122" y="78">vectors</text>
  </a>

  <a href="#/position" class:on={on('position')}>
    <title>Position</title>
    {#each lanes as x, i}
      <circle cx={x} cy="108" r="8" class="ring" />
      <text x={x} y="111.5" class="tiny">{i + 1}</text>
    {/each}
    <text x="122" y="112">+ position</text>
  </a>

  <a href="#/block" class:on={on('block', 'talk', 'think')}>
    <title>The full block</title>
    <rect x="14" y="130" width="198" height="104" rx="10" class="box" />
    <text x="204" y="145" class="times">× 4 layers</text>
  </a>

  <a href="#/attention" class="talk" class:on={on('talk', 'block')}>
    <title>Attention</title>
    <path d="M34 150 C 34 172, 64 160, 64 180" />
    <path d="M34 150 C 34 178, 94 158, 94 180" />
    <path d="M64 150 C 64 172, 94 162, 94 180" />
    {#each lanes as x}<circle cx={x} cy="180" r="4.5" />{/each}
    <text x="122" y="176">talk</text>
  </a>

  <a href="#/mlp" class="think" class:on={on('think', 'block')}>
    <title>The MLP</title>
    {#each lanes as x}<rect x={x - 7} y="202" width="14" height="14" rx="3" />{/each}
    <text x="122" y="213">think</text>
  </a>

  <a href="#/vote" class:on={on('vote')}>
    <title>From numbers to a word</title>
    <rect x="80" y="262" width="7" height="22" class="solid" />
    <rect x="90" y="272" width="7" height="12" class="solid" />
    <rect x="100" y="278" width="7" height="6" class="solid" />
    <text x="122" y="278">vote</text>
    <text x="94" y="304" class="word">next token</text>
  </a>

  <a href="#/training" class="loop" class:on={stage === 'train'}>
    <title>Training</title>
    <path d="M58 300 C 4 300, 2 250, 6 150 C 8 80, 6 30, 16 26" marker-end="url(#arrow)" />
  </a>
  <defs>
    <marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8z" fill="context-stroke" /></marker>
  </defs>
</svg>

<style>
  svg {
    width: 100%;
    max-width: 232px;
    display: block;
  }
  text {
    font-family: var(--display);
    font-size: 12.5px;
    font-weight: 560;
    fill: var(--ink-3);
  }
  .tiny {
    font-size: 9px;
    text-anchor: middle;
    font-weight: 700;
  }
  .times {
    font-size: 10.5px;
    text-anchor: end;
  }
  .word {
    text-anchor: middle;
    font-size: 11px;
  }
  .lane {
    stroke: var(--line);
    stroke-width: 2;
  }
  .lane.faint {
    stroke-dasharray: none;
  }
  a {
    text-decoration: none;
    cursor: pointer;
  }
  a .solid,
  a .ring,
  a .cell,
  a circle,
  a rect {
    fill: #c3ccd9;
    transition: fill 0.2s;
  }
  a .ring {
    fill: var(--bg);
    stroke: #c3ccd9;
    stroke-width: 2;
  }
  a .box {
    fill: none;
    stroke: #c3ccd9;
    stroke-width: 1.5;
    stroke-dasharray: 4 4;
  }
  a path {
    fill: none;
    stroke: #c3ccd9;
    stroke-width: 2;
    transition: stroke 0.2s;
  }
  a.on text {
    fill: var(--ink);
  }
  a.on .solid {
    fill: var(--ink);
  }
  a.on .ring {
    stroke: var(--ink);
  }
  a.on .c0 {
    fill: var(--pos);
  }
  a.on .c1 {
    fill: var(--neg);
  }
  a.on .c2 {
    fill: #7fc4be;
  }
  a.on .box {
    stroke: var(--ink-2);
  }
  a.talk.on path {
    stroke: var(--talk);
  }
  a.talk.on circle {
    fill: var(--talk);
  }
  a.talk.on text {
    fill: var(--talk);
  }
  a.think.on rect {
    fill: var(--think-fill);
  }
  a.think.on text {
    fill: var(--think);
  }
  a.loop path {
    stroke: transparent;
    stroke-dasharray: 3 4;
  }
  a.loop.on path {
    stroke: var(--ink);
  }
  a:hover text {
    fill: var(--ink);
    text-decoration: underline;
  }
  a:focus-visible {
    outline: none;
  }
  a:focus-visible text {
    fill: var(--talk);
    text-decoration: underline;
  }
</style>
