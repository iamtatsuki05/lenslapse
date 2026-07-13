// Trajectory panel: token probability vs training step (log x) for the pinned cell, as inline SVG.

import { displayToken } from './grid.js'

const SERIES_COLORS = ['#4a5bd4', '#d4494a', '#1f9d6f', '#c97b0f', '#8b5cf6']
const M = { top: 14, right: 12, bottom: 34, left: 40 }

const xlog = (s) => Math.log10(s + 1)

export function renderTrajectory(svg, series, steps, currentStep, { goldId } = {}) {
  const W = svg.clientWidth || 360
  const H = svg.clientHeight || 240
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`)
  svg.replaceChildren()
  if (!series.length || !steps.length) return

  const xMax = xlog(steps.at(-1))
  const X = (s) => M.left + (xlog(s) / xMax) * (W - M.left - M.right)
  const Y = (p) => M.top + (1 - p) * (H - M.top - M.bottom)

  const mk = (tag, attrs, text) => {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag)
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v)
    if (text != null) el.textContent = text
    svg.appendChild(el)
    return el
  }

  // axes + gridlines
  const axisColor = 'color-mix(in oklab, currentColor 30%, transparent)'
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    mk('line', { x1: M.left, x2: W - M.right, y1: Y(p), y2: Y(p), stroke: axisColor, 'stroke-width': p === 0 ? 1 : 0.4 })
    mk('text', { x: M.left - 6, y: Y(p) + 3.5, 'text-anchor': 'end', 'font-size': 9.5, fill: 'currentColor', opacity: 0.65 }, p.toFixed(2))
  }
  for (const s of [0, 10, 100, 1000, 10000, 100000]) {
    if (s > steps.at(-1)) break
    mk('line', { x1: X(s), x2: X(s), y1: M.top, y2: H - M.bottom, stroke: axisColor, 'stroke-width': 0.4 })
    mk(
      'text',
      { x: X(s), y: H - M.bottom + 13, 'text-anchor': 'middle', 'font-size': 9.5, fill: 'currentColor', opacity: 0.65 },
      s >= 1000 ? `${s / 1000}k` : String(s)
    )
  }
  mk(
    'text',
    { x: (M.left + W - M.right) / 2, y: H - 4, 'text-anchor': 'middle', 'font-size': 10, fill: 'currentColor', opacity: 0.75 },
    'training step (log scale)'
  )

  // current-step rule
  mk('line', {
    x1: X(currentStep),
    x2: X(currentStep),
    y1: M.top,
    y2: H - M.bottom,
    stroke: 'currentColor',
    'stroke-dasharray': '3 3',
    opacity: 0.6,
  })

  series.slice(0, SERIES_COLORS.length).forEach((s, i) => {
    const color = SERIES_COLORS[i % SERIES_COLORS.length]
    const d = s.points.map(([st, p], j) => `${j ? 'L' : 'M'}${X(st).toFixed(1)},${Y(p).toFixed(1)}`).join('')
    mk('path', { d, fill: 'none', stroke: color, 'stroke-width': s.id === goldId ? 2.6 : 1.7 })
    for (const [st, p] of s.points) mk('circle', { cx: X(st), cy: Y(p), r: 2.1, fill: color })
    const last = s.points.at(-1)
    mk(
      'text',
      {
        x: Math.min(X(last[0]) + 4, W - M.right - 2),
        y: Y(last[1]) - 5,
        'font-size': 10.5,
        'font-family': 'ui-monospace, monospace',
        fill: color,
        'font-weight': s.id === goldId ? 700 : 400,
      },
      displayToken(s.token) + (s.id === goldId ? ' ★' : '')
    )
  })
}
