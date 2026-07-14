// Trajectory panel: token probability vs training step (log x) for the pinned cell, as inline SVG.

import { displayToken } from './grid'
import type { TrajectorySeries } from './data'

const SERIES_COLORS = ['#4a5bd4', '#d4494a', '#1f9d6f', '#c97b0f', '#8b5cf6']
const M = { top: 14, right: 12, bottom: 34, left: 40 }

const xlog = (s: number) => Math.log10(s + 1)

/** Stable token-id -> color map, so the trajectory, layer profile, and labels stay consistent. */
export function assignSeriesColors(series: { id: number }[]): Map<number, string> {
  const map = new Map<number, string>()
  series.slice(0, SERIES_COLORS.length).forEach((s, i) => map.set(s.id, SERIES_COLORS[i]))
  return map
}

export function renderTrajectory(
  svg: SVGSVGElement,
  series: TrajectorySeries[],
  steps: number[],
  currentStep: number,
  { goldId, colors }: { goldId?: number; colors?: Map<number, string> } = {}
): void {
  const W = svg.clientWidth || 360
  const H = svg.clientHeight || 240
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`)
  svg.replaceChildren()
  if (!series.length || !steps.length) return

  const xMax = xlog(steps.at(-1)!)
  const X = (s: number) => M.left + (xlog(s) / xMax) * (W - M.left - M.right)
  const Y = (p: number) => M.top + (1 - p) * (H - M.top - M.bottom)

  const mk = (tag: string, attrs: Record<string, string | number>, text?: string | null) => {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag)
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v))
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
    if (s > steps.at(-1)!) break
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
    const color = colors?.get(s.id) ?? SERIES_COLORS[i % SERIES_COLORS.length]
    const d = s.points.map(([st, p], j) => `${j ? 'L' : 'M'}${X(st).toFixed(1)},${Y(p).toFixed(1)}`).join('')
    mk('path', { d, fill: 'none', stroke: color, 'stroke-width': s.id === goldId ? 2.6 : 1.7 })
    for (const [st, p] of s.points) mk('circle', { cx: X(st), cy: Y(p), r: 2.1, fill: color })
    const last = s.points.at(-1)!
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

/** The classic logit-lens view: p vs layer at one position and step (linear x, emb..L_n). */
export function renderLayerProfile(
  svg: SVGSVGElement,
  series: TrajectorySeries[], // points: [[layer, p, rank], ...]
  layers: number,
  pinnedLayer: number,
  { goldId, colors }: { goldId?: number; colors?: Map<number, string> } = {}
): void {
  const W = svg.clientWidth || 360
  const H = svg.clientHeight || 130
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`)
  svg.replaceChildren()
  if (!series.length || layers < 2) return

  const B = { top: 8, right: 12, bottom: 26, left: 40 }
  const X = (li: number) => B.left + (li / (layers - 1)) * (W - B.left - B.right)
  const yMax = Math.max(0.05, ...series.flatMap((s) => s.points.map((p) => p[1])))
  const Y = (p: number) => B.top + (1 - p / yMax) * (H - B.top - B.bottom)

  const mk = (tag: string, attrs: Record<string, string | number>, text?: string | null) => {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag)
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v))
    if (text != null) el.textContent = text
    svg.appendChild(el)
    return el
  }

  const axisColor = 'color-mix(in oklab, currentColor 30%, transparent)'
  for (const p of [0, yMax]) {
    mk('line', { x1: B.left, x2: W - B.right, y1: Y(p), y2: Y(p), stroke: axisColor, 'stroke-width': p === 0 ? 1 : 0.4 })
    mk(
      'text',
      { x: B.left - 6, y: Y(p) + 3.5, 'text-anchor': 'end', 'font-size': 9, fill: 'currentColor', opacity: 0.65 },
      p === 0 ? '0' : yMax.toFixed(yMax >= 0.1 ? 2 : 3)
    )
  }
  const every = layers > 14 ? 4 : layers > 8 ? 2 : 1
  for (let li = 0; li < layers; li++) {
    if (li % every && li !== layers - 1) continue
    mk(
      'text',
      { x: X(li), y: H - B.bottom + 12, 'text-anchor': 'middle', 'font-size': 9, fill: 'currentColor', opacity: 0.65 },
      li === 0 ? 'emb' : `L${li}`
    )
  }
  mk(
    'text',
    { x: (B.left + W - B.right) / 2, y: H - 3, 'text-anchor': 'middle', 'font-size': 10, fill: 'currentColor', opacity: 0.75 },
    'layer (lens read-out depth)'
  )
  // pinned-layer rule ties this chart to the outlined grid cell
  mk('line', {
    x1: X(pinnedLayer),
    x2: X(pinnedLayer),
    y1: B.top,
    y2: H - B.bottom,
    stroke: 'currentColor',
    'stroke-dasharray': '3 3',
    opacity: 0.6,
  })

  series.slice(0, SERIES_COLORS.length).forEach((s, i) => {
    const color = colors?.get(s.id) ?? SERIES_COLORS[i % SERIES_COLORS.length]
    const d = s.points.map(([li, p], j) => `${j ? 'L' : 'M'}${X(li).toFixed(1)},${Y(p).toFixed(1)}`).join('')
    mk('path', { d, fill: 'none', stroke: color, 'stroke-width': s.id === goldId ? 2.4 : 1.5 })
    for (const [li, p] of s.points) mk('circle', { cx: X(li), cy: Y(p), r: 1.8, fill: color })
  })
}
