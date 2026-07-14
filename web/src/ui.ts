// Small UI helpers: tooltip, slider ticks, badges, gallery cards.

import { displayToken } from './grid'
import type { CellInfo } from './grid'
import type { Prompt } from './data'

export function showTooltip(el: HTMLElement, cellInfo: CellInfo, evt: MouseEvent, tokens: string[]): void {
  const { layer, pos, cell } = cellInfo
  const rows = cell.top
    .map(
      ([tok, p]) =>
        `<tr><td>${escapeHtml(displayToken(tok))}</td><td>${(p * 100).toFixed(p >= 0.1 ? 1 : 2)}%</td></tr>`
    )
    .join('')
  el.innerHTML = `<div class="tt-head">${layer === 0 ? 'embedding' : `layer ${layer}`} · after “${escapeHtml(
    displayToken(tokens[pos] ?? '')
  )}”</div><table>${rows}</table>`
  el.hidden = false
  const pad = 12
  const w = el.offsetWidth
  const h = el.offsetHeight
  let x = evt.clientX + pad
  let y = evt.clientY + pad
  if (x + w > innerWidth - 4) x = evt.clientX - w - pad
  if (y + h > innerHeight - 4) y = evt.clientY - h - pad
  el.style.left = `${x}px`
  el.style.top = `${y}px`
}

export function hideTooltip(el: HTMLElement): void {
  el.hidden = true
}

/** Tooltip for the acquisition-map view: the cell's final answer and when it first became top-1. */
export function showAcqTooltip(
  el: HTMLElement,
  cellInfo: CellInfo,
  evt: MouseEvent,
  tokens: string[],
  firstStep: number
): void {
  const { layer, pos, cell } = cellInfo
  const [tok, p] = cell.top[0]
  el.innerHTML = `<div class="tt-head">${layer === 0 ? 'embedding' : `layer ${layer}`} · after “${escapeHtml(
    displayToken(tokens[pos] ?? '')
  )}”</div><table><tr><td>final top-1</td><td>${escapeHtml(displayToken(tok))} (${(p * 100).toFixed(1)}%)</td></tr><tr><td>first top-1 at</td><td>step ${firstStep.toLocaleString()}</td></tr></table>`
  el.hidden = false
  const pad = 12
  let x = evt.clientX + pad
  let y = evt.clientY + pad
  if (x + el.offsetWidth > innerWidth - 4) x = evt.clientX - el.offsetWidth - pad
  if (y + el.offsetHeight > innerHeight - 4) y = evt.clientY - el.offsetHeight - pad
  el.style.left = `${x}px`
  el.style.top = `${y}px`
}

export function buildSliderTicks(container: HTMLElement, steps: number[], liveSteps: number[], maxStep: number): void {
  container.replaceChildren()
  const live = new Set(liveSteps)
  const xmax = Math.log10(maxStep + 1)
  for (const s of steps) {
    const t = document.createElement('div')
    t.className = live.has(s) ? 'tick live' : 'tick'
    t.style.left = `${(Math.log10(s + 1) / xmax) * 100}%`
    t.title = `step ${s.toLocaleString()}${live.has(s) ? ' (live-capable)' : ''}`
    container.appendChild(t)
  }
}

export function setBadge(el: HTMLElement, mode: string): void {
  const labels: Record<string, [string, string]> = {
    precomputed: ['precomputed', ''],
    wasm: ['live · WASM', ''],
    webgpu: ['live · WebGPU', ''],
    server: ['live · server', ''],
    'precomputed-only': ['precomputed only', 'warn'],
  }
  const [text, cls] = labels[mode] ?? [mode, '']
  el.textContent = text
  el.className = `chip badge ${cls}`.trim()
}

export interface StoryCard {
  tag: string
  title: string
  desc: string
  text: string // the prompt the card demonstrates
  pin: string
  step: number
}

/** Curated example prompts (mirrors PROMPTS in lenslapse/precompute_lens.py) — offered as
 * live-probe suggestions for models that ship no precomputed shards. */
export const EXAMPLE_TEXTS = [
  'The capital of Japan is the city of',
  'The Eiffel Tower is located in the city of',
  'Water is made of hydrogen and',
  'The first president of the United States was George',
  'The opposite of hot is',
  'Paris is to France as Tokyo is to',
  'Two plus two equals',
  '3 + 4 =',
  'The quick brown fox jumps over the lazy',
  'Once upon a',
  'Thank you very',
  'The keys to the cabinet',
  'def add(a, b):\n    return a +',
  'import numpy as',
  'The DNA molecule has the shape of a double',
  'Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly',
]

const STORY_CARDS: StoryCard[] = [
  {
    tag: 'fact acquisition',
    title: 'When does the model learn “Tokyo”?',
    desc: 'Scrub the slider: for “The capital of Japan is the city of”, watch Tokyo surface in the last layers after thousands of steps — then dip and recover. Acquisition is not monotone.',
    text: 'The capital of Japan is the city of',
    pin: 'lastLayerLastPos',
    step: 8000,
  },
  {
    tag: 'early-training bias',
    title: 'At first, everything is “the”',
    desc: 'At early steps the lens predicts high-frequency tokens (“the”, “,”) at every layer and position — before position-specific structure emerges.',
    text: 'The quick brown fox jumps over the lazy',
    pin: 'lastLayerLastPos',
    step: 512,
  },
  {
    tag: 'layer division of labor',
    title: 'Facts crystallize in the deep layers',
    desc: 'Late in training, early layers still guess frequent tokens while deeper layers assemble the answer — the classic logit-lens picture, now with a time axis.',
    text: 'The Eiffel Tower is located in the city of',
    pin: 'lastLayerLastPos',
    step: 143000,
  },
]

export function buildGallery(container: HTMLElement, onSelect: (card: StoryCard) => void | Promise<void>): void {
  // cards apply to whichever model is selected: precomputed when its shards carry the prompt,
  // live-probed otherwise (main.ts resolves that per click)
  container.replaceChildren()
  for (const card of STORY_CARDS) {
    const btn = document.createElement('button')
    btn.className = 'story-card'
    btn.innerHTML = `<div class="story-tag">${card.tag}</div><div class="story-title">${card.title}</div><div class="story-desc">${card.desc}</div>`
    btn.addEventListener('click', () => onSelect(card))
    container.appendChild(btn)
  }
}

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
