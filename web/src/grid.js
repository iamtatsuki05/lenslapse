// Canvas lens grid: rows = embedding + transformer layers (bottom-up), cols = token positions.

import { cssColor, probColor, textColorOn } from './color.js'

const CELL_W = 78
const CELL_H = 27
const LABEL_W = 46
const HEADER_H = 26

export class LensGrid {
  constructor(canvas, { onHover, onPin }) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    this.grid = null
    this.tokens = []
    this.pinned = null // {layer, pos}
    this.onHover = onHover
    this.onPin = onPin
    this.dark = matchMedia('(prefers-color-scheme: dark)')
    this.dark.addEventListener('change', () => this.render())

    canvas.addEventListener('mousemove', (e) => {
      const cell = this.hit(e)
      this.onHover?.(cell, e)
    })
    canvas.addEventListener('mouseleave', () => this.onHover?.(null))
    canvas.addEventListener('click', (e) => {
      const cell = this.hit(e)
      if (cell) {
        this.pinned = this.pinned && this.pinned.layer === cell.layer && this.pinned.pos === cell.pos ? null : cell
        this.render()
        this.onPin?.(this.pinned)
      }
    })
  }

  setData(grid, tokens, pinned = this.pinned) {
    this.grid = grid
    this.tokens = tokens
    this.pinned = pinned
    this.render()
  }

  hit(e) {
    if (!this.grid) return null
    const rect = this.canvas.getBoundingClientRect()
    const x = e.clientX - rect.left - LABEL_W
    const y = e.clientY - rect.top - HEADER_H
    if (x < 0 || y < 0) return null
    const pos = Math.floor(x / CELL_W)
    const row = Math.floor(y / CELL_H)
    const layer = this.grid.layers - 1 - row // top row = deepest layer
    if (pos >= this.grid.positions || layer < 0 || layer >= this.grid.layers) return null
    return { layer, pos, cell: this.grid.cells[layer][pos] }
  }

  render() {
    const { canvas, ctx, grid } = this
    if (!grid) return
    const w = LABEL_W + grid.positions * CELL_W
    const h = HEADER_H + grid.layers * CELL_H
    const dpr = window.devicePixelRatio || 1
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    this.renderTo(ctx, { dark: this.dark.matches })
  }

  /** Grid pixel size at scale 1 (used by the figure exporter). */
  size() {
    if (!this.grid) return { w: 0, h: 0 }
    return { w: LABEL_W + this.grid.positions * CELL_W, h: HEADER_H + this.grid.layers * CELL_H }
  }

  /** Draw the grid into an arbitrary 2D context (transform must be set by the caller). */
  renderTo(ctx, { dark = false } = {}) {
    const { grid } = this
    if (!grid) return
    ctx.font = '11px ui-monospace, monospace'
    ctx.textBaseline = 'middle'

    // header: input tokens
    ctx.fillStyle = dark ? '#9a9daa' : '#6b6e78'
    for (let t = 0; t < grid.positions; t++) {
      const label = clip(ctx, displayToken(this.tokens[t] ?? ''), CELL_W - 10)
      ctx.fillText(label, LABEL_W + t * CELL_W + 5, HEADER_H / 2)
    }

    for (let layer = 0; layer < grid.layers; layer++) {
      const row = grid.layers - 1 - layer
      const y = HEADER_H + row * CELL_H
      // row label
      ctx.fillStyle = dark ? '#9a9daa' : '#6b6e78'
      ctx.fillText(layer === 0 ? 'emb' : `L${layer}`, 6, y + CELL_H / 2)
      for (let t = 0; t < grid.positions; t++) {
        const cell = grid.cells[layer][t]
        const x = LABEL_W + t * CELL_W
        const rgb = probColor(cell.prob)
        ctx.fillStyle = cssColor(rgb)
        ctx.fillRect(x + 1, y + 1, CELL_W - 2, CELL_H - 2)
        ctx.fillStyle = textColorOn(rgb)
        ctx.fillText(clip(ctx, displayToken(cell.token), CELL_W - 10), x + 5, y + CELL_H / 2)
        if (this.pinned && this.pinned.layer === layer && this.pinned.pos === t) {
          ctx.strokeStyle = dark ? '#ff8b8b' : '#d4494a'
          ctx.lineWidth = 2.5
          ctx.strokeRect(x + 1.5, y + 1.5, CELL_W - 3, CELL_H - 3)
        }
      }
    }
  }
}

export function displayToken(tok) {
  return tok.replace(/^Ġ/, '␣').replace(/^ /, '␣').replace(/\n/g, '⏎').replace(/Ċ/g, '⏎')
}

function clip(ctx, text, maxW) {
  if (ctx.measureText(text).width <= maxW) return text
  let t = text
  while (t.length > 1 && ctx.measureText(`${t}…`).width > maxW) t = t.slice(0, -1)
  return `${t}…`
}
