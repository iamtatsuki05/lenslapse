// Canvas lens grid: rows = embedding + transformer layers (bottom-up), cols = token positions.

import { cssColor, logProb01, probColor, textColorOn } from './color'
import type { GridCell, GridData } from './data'

const CELL_W = 78
const CELL_H = 27
const LABEL_W = 46
const HEADER_H = 26

export interface PinnedCell {
  layer: number
  pos: number
}

export interface CellInfo extends PinnedCell {
  cell: GridCell
}

interface LensGridCallbacks {
  onHover?: (cell: CellInfo | null, evt?: MouseEvent) => void
  onPin?: (pinned: PinnedCell | null) => void
}

export class LensGrid {
  // `declare`: type-only field declarations — all fields are assigned in the constructor, and
  // emitting them as class fields would change the compiled output (define semantics).
  declare canvas: HTMLCanvasElement
  declare ctx: CanvasRenderingContext2D
  declare grid: GridData | null
  declare tokens: string[]
  declare pinned: PinnedCell | null
  declare onHover: LensGridCallbacks['onHover']
  declare onPin: LensGridCallbacks['onPin']
  declare dark: MediaQueryList
  declare flash: Set<string> | null // "layer:pos" cells whose top-1 just changed (scrub/play)
  declare flashTimer: number | undefined
  declare logScale: boolean // color by log10(p) instead of p — reveals early-training structure

  constructor(canvas: HTMLCanvasElement, { onHover, onPin }: LensGridCallbacks) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.grid = null
    this.tokens = []
    this.pinned = null // {layer, pos}
    this.flash = null
    this.flashTimer = undefined
    this.logScale = false
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

  setData(grid: GridData | null, tokens: string[], pinned: PinnedCell | null = this.pinned): void {
    this.grid = grid
    this.tokens = tokens
    this.pinned = pinned
    this.flash = null // stale highlights must not outline cells of a different grid
    this.render()
  }

  /** Briefly outline cells (e.g. whose top-1 changed since the previous step) — the "learning
   * events" become visible while scrubbing or playing the time-lapse. */
  flashCells(cells: Set<string>): void {
    if (!cells.size) return
    this.flash = cells
    window.clearTimeout(this.flashTimer)
    this.flashTimer = window.setTimeout(() => {
      this.flash = null
      this.render()
    }, 450)
    this.render()
  }

  hit(e: MouseEvent): CellInfo | null {
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

  render(): void {
    const { canvas, ctx, grid } = this
    if (!grid) {
      // setData(null, ...) must blank the canvas, not leave the previous model's grid visible
      canvas.width = 0
      canvas.height = 0
      canvas.style.width = '0px'
      canvas.style.height = '0px'
      return
    }
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
    // change-flash overlay lives here, NOT in renderTo: exported figures must never carry it
    if (this.flash) {
      ctx.strokeStyle = this.dark.matches ? '#ffd166' : '#e8590c'
      ctx.lineWidth = 2
      for (const key of this.flash) {
        const [layer, t] = key.split(':').map(Number)
        if (layer < 0 || layer >= grid.layers || t < 0 || t >= grid.positions) continue
        const row = grid.layers - 1 - layer
        ctx.strokeRect(LABEL_W + t * CELL_W + 1, HEADER_H + row * CELL_H + 1, CELL_W - 2, CELL_H - 2)
      }
    }
  }

  /** Grid pixel size at scale 1 (used by the figure exporter). */
  size(): { w: number; h: number } {
    if (!this.grid) return { w: 0, h: 0 }
    return { w: LABEL_W + this.grid.positions * CELL_W, h: HEADER_H + this.grid.layers * CELL_H }
  }

  /** Draw the grid into an arbitrary 2D context (transform must be set by the caller). */
  renderTo(ctx: CanvasRenderingContext2D, { dark = false }: { dark?: boolean } = {}): void {
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
        const rgb = probColor(this.logScale ? logProb01(cell.prob) : cell.prob)
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

export function displayToken(tok: string): string {
  return tok.replace(/^[Ġ▁ ]/, '␣').replace(/\n/g, '⏎').replace(/Ċ/g, '⏎')
}

function clip(ctx: CanvasRenderingContext2D, text: string, maxW: number): string {
  if (ctx.measureText(text).width <= maxW) return text
  let t = text
  while (t.length > 1 && ctx.measureText(`${t}…`).width > maxW) t = t.slice(0, -1)
  return `${t}…`
}
