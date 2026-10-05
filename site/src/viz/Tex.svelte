<script lang="ts">
  import katex from 'katex'
  let { tex, block = false }: { tex: string; block?: boolean } = $props()
  const html = $derived(katex.renderToString(tex, { displayMode: block, throwOnError: false }))
</script>

{#if block}<div class="tex">{@html html}</div>{:else}<span>{@html html}</span>{/if}

<style>
  .tex {
    overflow-x: auto;
    margin: 0.6rem 0 1.3rem;
    max-width: var(--measure);
    font-size: 1.02rem;
  }
</style>
