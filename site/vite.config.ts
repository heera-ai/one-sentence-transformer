import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig } from 'vite'

export default defineConfig({
  base: './',
  plugins: [svelte()],
  // Chapter 10 shows the real Python source from ../model.
  server: { fs: { allow: ['..'] } },
  test: { include: ['tests/**/*.test.ts'] },
})
