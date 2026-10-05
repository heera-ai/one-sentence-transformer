<script lang="ts">
  import hljs from 'highlight.js/lib/core'
  import python from 'highlight.js/lib/languages/python'
  hljs.registerLanguage('python', python)

  let { code, caption = '' }: { code: string; caption?: string } = $props()
  const html = $derived(hljs.highlight(code.replace(/^\n+|\s+$/g, ''), { language: 'python' }).value)
</script>

<figure class="code">
  {#if caption}<figcaption>{caption}</figcaption>{/if}
  <pre><code>{@html html}</code></pre>
</figure>

<style>
  .code {
    margin: 1.2rem 0 1.8rem;
    max-width: 52rem;
    background: #16213a;
    color: #e6ebf5;
    border-radius: 12px;
    overflow: hidden;
  }
  figcaption {
    font-family: var(--display);
    font-size: 0.85rem;
    padding: 0.55rem 1.1rem;
    background: #223055;
    color: #c5cee3;
  }
  pre {
    margin: 0;
    padding: 1rem 1.1rem;
    overflow-x: auto;
    font-size: 0.82rem;
    line-height: 1.6;
    tab-size: 4;
  }
  code {
    font-family: var(--mono);
    font-size: inherit;
  }
  .code :global(.hljs-keyword),
  .code :global(.hljs-literal) {
    color: #a9b3ff;
  }
  .code :global(.hljs-string) {
    color: #8fdcc9;
  }
  .code :global(.hljs-comment) {
    color: #8b97b5;
    font-style: italic;
  }
  .code :global(.hljs-number) {
    color: #f6c667;
  }
  .code :global(.hljs-title),
  .code :global(.hljs-built_in) {
    color: #f3a6c4;
  }
</style>
