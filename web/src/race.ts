// Top-k "bar race" for the pinned cell: rows are keyed by token id and reused across renders,
// so CSS transitions animate rank swaps and probability changes as the training step advances.

import { displayToken } from './grid'
import type { TopEntry } from './data'

export const RACE_ROW_H = 21

export function renderRace(container: HTMLElement, top: TopEntry[], opts: { goldId?: number; limit?: number } = {}): void {
  const rows = top.slice(0, opts.limit ?? 8)
  if (!rows.length) {
    container.hidden = true
    container.replaceChildren()
    return
  }
  container.hidden = false
  container.style.height = `${rows.length * RACE_ROW_H}px`
  const maxP = Math.max(rows[0][1], 1e-9)
  const seen = new Set<string>()
  rows.forEach(([tok, p, id], rank) => {
    const key = String(id)
    seen.add(key)
    let row = container.querySelector<HTMLElement>(`.race-row[data-id="${key}"]`)
    if (!row) {
      row = document.createElement('div')
      row.className = 'race-row'
      row.dataset.id = key
      row.innerHTML = '<span class="race-bar"></span><span class="race-tok"></span><span class="race-p"></span>'
      container.appendChild(row)
    }
    row.classList.toggle('gold', id === opts.goldId)
    row.style.top = `${rank * RACE_ROW_H}px`
    row.querySelector<HTMLElement>('.race-bar')!.style.width = `${Math.max(1.5, (p / maxP) * 100)}%`
    row.querySelector<HTMLElement>('.race-tok')!.textContent = displayToken(tok)
    row.querySelector<HTMLElement>('.race-p')!.textContent = `${(p * 100).toFixed(p >= 0.001 ? 1 : 2)}%`
  })
  for (const el of [...container.querySelectorAll<HTMLElement>('.race-row')]) {
    if (!seen.has(el.dataset.id!)) el.remove()
  }
}
