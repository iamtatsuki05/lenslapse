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
  match: (p: Prompt) => boolean
  pin: string
  step: number
}

const STORY_CARDS: StoryCard[] = [
  {
    tag: 'fact acquisition',
    title: 'When does the model learn “Tokyo”?',
    desc: 'Scrub the slider: for “The capital of Japan is the city of”, watch Tokyo surface in the last layers after thousands of steps — then dip and recover. Acquisition is not monotone.',
    match: (p) => p.text.startsWith('The capital of Japan'),
    pin: 'lastLayerLastPos',
    step: 8000,
  },
  {
    tag: 'early-training bias',
    title: 'At first, everything is “the”',
    desc: 'At early steps the lens predicts high-frequency tokens (“the”, “,”) at every layer and position — before position-specific structure emerges.',
    match: (p) => p.text.startsWith('The quick brown fox'),
    pin: 'lastLayerLastPos',
    step: 512,
  },
  {
    tag: 'layer division of labor',
    title: 'Facts crystallize in the deep layers',
    desc: 'Late in training, early layers still guess frequent tokens while deeper layers assemble the answer — the classic logit-lens picture, now with a time axis.',
    match: (p) => p.text.startsWith('The Eiffel Tower'),
    pin: 'lastLayerLastPos',
    step: 143000,
  },
]

export function buildGallery(
  container: HTMLElement,
  prompts: Prompt[],
  onSelect: (p: Prompt, card: StoryCard) => void | Promise<void>
): void {
  container.replaceChildren()
  for (const card of STORY_CARDS) {
    const p = prompts.find(card.match)
    if (!p) continue
    const btn = document.createElement('button')
    btn.className = 'story-card'
    btn.innerHTML = `<div class="story-tag">${card.tag}</div><div class="story-title">${card.title}</div><div class="story-desc">${card.desc}</div>`
    btn.addEventListener('click', () => onSelect(p, card))
    container.appendChild(btn)
  }
}

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
