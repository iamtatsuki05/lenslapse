// Small UI helpers: tooltip, slider ticks, badges, gallery cards.

import { logStepFrac } from './data'
import { displayToken, layerLabel } from './grid'
import type { CellInfo } from './grid'
import type { Prompt } from './data'

/** The head line every tooltip variant shares: which layer, after which token. */
function tooltipHead(layer: number, token: string): string {
  return `<div class="tt-head">${layerLabel(layer)} · after “${escapeHtml(displayToken(token))}”</div>`
}

/** Place a tooltip at the pointer, clamped to stay fully inside the viewport. Must run after
 * `el.innerHTML` is set (it reads offsetWidth/Height, which depend on the current content). */
function positionTooltip(el: HTMLElement, evt: MouseEvent): void {
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

export function showTooltip(el: HTMLElement, cellInfo: CellInfo, evt: MouseEvent, tokens: string[]): void {
  const { layer, pos, cell } = cellInfo
  const maxP = cell.top[0]?.[1] || 1
  const rows = cell.top
    .map(
      ([tok, p]) =>
        `<tr><td>${escapeHtml(displayToken(tok))}</td><td>${(p * 100).toFixed(p >= 0.1 ? 1 : 2)}%</td>` +
        `<td class="tt-bar"><i style="width:${Math.max(2, (p / maxP) * 100)}%"></i></td></tr>`
    )
    .join('')
  el.innerHTML = `${tooltipHead(layer, tokens[pos] ?? '')}<table>${rows}</table>`
  positionTooltip(el, evt)
}

export function hideTooltip(el: HTMLElement): void {
  el.hidden = true
}

/** Tooltip for the diff view: the cell's top-1 at the reference vs the current checkpoint. */
export function showDiffTooltip(
  el: HTMLElement,
  cellInfo: CellInfo,
  evt: MouseEvent,
  tokens: string[],
  diff: { refStep: number; curStep: number; ref: [string, number, number]; cur: [string, number, number]; change: number }
): void {
  const { layer, pos } = cellInfo
  const row = (label: string, [tok, p]: [string, number, number]) =>
    `<tr><td>${label}</td><td>${escapeHtml(displayToken(tok))}</td><td>${(p * 100).toFixed(1)}%</td></tr>`
  el.innerHTML = `${tooltipHead(layer, tokens[pos] ?? '')}<table>${row(`step ${diff.refStep.toLocaleString()}`, diff.ref)}${row(
    `step ${diff.curStep.toLocaleString()}`,
    diff.cur
  )}<tr><td>top-10 turnover</td><td colspan="2">${(diff.change * 100).toFixed(0)}%</td></tr></table>`
  positionTooltip(el, evt)
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
  el.innerHTML = `${tooltipHead(layer, tokens[pos] ?? '')}<table><tr><td>final top-1</td><td>${escapeHtml(displayToken(tok))} (${(p * 100).toFixed(1)}%)</td></tr><tr><td>first top-1 at</td><td>step ${firstStep.toLocaleString()}</td></tr></table>`
  positionTooltip(el, evt)
}

export function buildSliderTicks(container: HTMLElement, steps: number[], liveSteps: number[], maxStep: number): void {
  container.replaceChildren()
  const live = new Set(liveSteps)
  for (const s of steps) {
    const t = document.createElement('div')
    t.className = live.has(s) ? 'tick live' : 'tick'
    // logStepFrac's step-0 guard matters here too: a single-checkpoint model registered on the
    // probe server has steps=[0], which would otherwise emit left:"NaN%"
    t.style.left = `${logStepFrac(s, maxStep) * 100}%`
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

/** Curated example prompts (mirrors PROMPTS in src/lenslapse/precompute_lens.py) — offered as
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

export const STORY_CARDS: StoryCard[] = [
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
