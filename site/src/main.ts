import '@fontsource-variable/bricolage-grotesque'
import '@fontsource-variable/literata'
import '@fontsource-variable/literata/wght-italic.css'
import '@fontsource-variable/jetbrains-mono'
import 'katex/dist/katex.min.css'
import { mount } from 'svelte'
import './app.css'
import App from './App.svelte'

export default mount(App, { target: document.getElementById('app')! })
