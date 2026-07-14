// One-click figure export: composes the current view (title block, lens grid, trajectory,
// permalink footer) into a publication-friendly image and saves it as PNG or PDF.
// Rendering is forced to a light theme on a white background at 3x resolution.

import { displayToken } from './grid'
import type { LensGrid, PinnedCell } from './grid'

export interface FigureMeta {
  model: string
  prompt: string
  step: number
  pinned: PinnedCell | null
  permalink: string
  /** non-default grid views (acquisition map, diff) must say so — the colors mean something else */
  view?: string
}

export interface ExportView {
  grid: LensGrid
  trajSvg: SVGSVGElement | null
  meta: FigureMeta
}

const SCALE = 3
const PAD = 28
const TITLE_H = 58
const FOOTER_H = 30
const GAP = 22

/** Rasterize the trajectory SVG (which relies on currentColor) at the export scale. */
async function svgToImage(svg: SVGSVGElement, scale: number) {
  const clone = svg.cloneNode(true) as SVGSVGElement
  const w = svg.clientWidth || 360
  const h = svg.clientHeight || 240
  clone.setAttribute('width', String(w))
  clone.setAttribute('height', String(h))
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.style.color = '#1b1c20' // resolve currentColor for a light background
  clone.style.fontFamily = 'Helvetica, Arial, sans-serif'
  const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml' })
  const url = URL.createObjectURL(blob)
  try {
    const img = new Image()
    await new Promise((resolve, reject) => {
      img.onload = resolve
      img.onerror = reject
      img.src = url
    })
    const canvas = document.createElement('canvas')
    canvas.width = w * scale
    canvas.height = h * scale
    const ctx = canvas.getContext('2d')!
    ctx.scale(scale, scale)
    ctx.drawImage(img, 0, 0, w, h)
    return { canvas, w, h }
  } finally {
    URL.revokeObjectURL(url)
  }
}

/**
 * Compose the figure. `view` = { grid: LensGrid, trajSvg: SVGElement|null, meta: {model, prompt,
 * step, pinned, permalink} }. Returns a canvas at SCALE resolution.
 */
export async function composeFigure({ grid, trajSvg, meta }: ExportView) {
  const g = grid.size()
  const traj = trajSvg && trajSvg.childNodes.length > 0 ? await svgToImage(trajSvg, SCALE) : null

  const contentW = Math.max(g.w, traj ? traj.w : 0)
  const trajH = traj ? traj.h + GAP : 0
  const W = contentW + PAD * 2
  const H = TITLE_H + g.h + trajH + FOOTER_H + PAD * 2

  const canvas = document.createElement('canvas')
  canvas.width = W * SCALE
  canvas.height = H * SCALE
  const ctx = canvas.getContext('2d')!
  ctx.scale(SCALE, SCALE)

  // background + hairline frame
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, W, H)
  ctx.strokeStyle = '#d8dae0'
  ctx.lineWidth = 1
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1)

  // title block
  ctx.fillStyle = '#15161a'
  ctx.font = '600 15px Helvetica, Arial, sans-serif'
  ctx.textBaseline = 'alphabetic'
  ctx.fillText(
    `LensLapse · ${meta.model} · training step ${meta.step.toLocaleString()}${meta.view ? ` · ${meta.view}` : ''}`,
    PAD,
    PAD + 6
  )
  ctx.font = '13px Helvetica, Arial, sans-serif'
  ctx.fillStyle = '#4a4d57'
  const promptLine = `“${meta.prompt}”${meta.pinned ? `   (pinned: ${meta.pinned.layer === 0 ? 'embedding' : `layer ${meta.pinned.layer}`}, position ${meta.pinned.pos})` : ''}`
  ctx.fillText(clipText(ctx, promptLine, contentW), PAD, PAD + 28)

  // lens grid
  ctx.save()
  ctx.translate(PAD, PAD + TITLE_H)
  grid.renderTo(ctx, { dark: false })
  ctx.restore()

  // trajectory
  if (traj) {
    ctx.drawImage(traj.canvas, PAD, PAD + TITLE_H + g.h + GAP, traj.w, traj.h)
  }

  // footer: permalink + attribution
  ctx.font = '11px ui-monospace, Menlo, monospace'
  ctx.fillStyle = '#7a7d88'
  ctx.fillText(clipText(ctx, meta.permalink, contentW - 120), PAD, H - PAD + 8)
  ctx.textAlign = 'right'
  ctx.fillText(new Date().toISOString().slice(0, 10), W - PAD, H - PAD + 8)
  ctx.textAlign = 'left'

  return { canvas, w: W, h: H }
}

function clipText(ctx: CanvasRenderingContext2D, text: string, maxW: number): string {
  if (ctx.measureText(text).width <= maxW) return text
  let t = text
  while (t.length > 1 && ctx.measureText(`${t}…`).width > maxW) t = t.slice(0, -1)
  return `${t}…`
}

function figureBasename(meta: FigureMeta): string {
  const slug = meta.prompt
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40)
  return `lenslapse_${meta.model.replace(/[^a-z0-9]+/gi, '-')}_step${meta.step}_${slug || 'view'}`
}

function download(url: string, filename: string): void {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
}

export async function exportPng(view: ExportView): Promise<void> {
  const { canvas } = await composeFigure(view)
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
  const url = URL.createObjectURL(blob!)
  try {
    download(url, `${figureBasename(view.meta)}.png`)
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 5000)
  }
}

export async function exportPdf(view: ExportView): Promise<void> {
  const { canvas, w, h } = await composeFigure(view)
  const { jsPDF } = await import('jspdf') // lazy: keeps the main bundle lean
  const orientation = w >= h ? 'landscape' : 'portrait'
  const pdf = new jsPDF({ orientation, unit: 'pt', format: [w, h] })
  pdf.addImage(canvas.toDataURL('image/png'), 'PNG', 0, 0, w, h)
  pdf.save(`${figureBasename(view.meta)}.pdf`)
}

export function displayPrompt(tokens: string[]): string {
  return tokens.map(displayToken).join('').replace(/␣/g, ' ').trim()
}
