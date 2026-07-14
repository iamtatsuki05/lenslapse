import { describe, expect, it } from 'vitest'
import { assignSeriesColors, renderLayerProfile, renderTrajectory } from '../src/traj'
import type { TrajectorySeries } from '../src/data'

const series = (): TrajectorySeries[] => [
  { id: 1, token: 'Ġthe', points: [[0, 0.1, 5], [100, 0.4, 1], [1000, 0.2, 3]] },
  { id: 2, token: 'ĠTokyo', points: [[0, 0.0, 20], [100, 0.05, 8], [1000, 0.3, 1]] },
]

describe('assignSeriesColors', () => {
  it('assigns a stable id -> color map, capped at the palette size', () => {
    const colors = assignSeriesColors(series())
    expect(colors.size).toBe(2)
    expect(colors.get(1)).not.toBe(colors.get(2))
    const six = Array.from({ length: 6 }, (_, i) => ({ id: i }))
    expect(assignSeriesColors(six).size).toBe(5) // SERIES_COLORS has 5 entries
  })
})

describe('renderTrajectory', () => {
  it('draws one path + one set of point circles per series, plus the current-step marker', () => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement
    renderTrajectory(svg, series(), [0, 100, 1000], 100, { goldId: 2 })
    const paths = svg.querySelectorAll('path')
    expect(paths).toHaveLength(2)
    // gold series (id=2) gets the thicker stroke
    const strokeWidths = Array.from(paths).map((p) => p.getAttribute('stroke-width'))
    expect(strokeWidths).toContain('2.6') // gold
    expect(strokeWidths).toContain('1.7') // non-gold
    expect(svg.querySelectorAll('circle')).toHaveLength(6) // 3 points x 2 series
    // dashed current-step rule
    const dashed = Array.from(svg.querySelectorAll('line')).filter((l) => l.getAttribute('stroke-dasharray'))
    expect(dashed).toHaveLength(1)
    // end-of-series labels, gold one starred
    const labels = Array.from(svg.querySelectorAll('text')).map((t) => t.textContent)
    expect(labels.some((t) => t?.includes('★'))).toBe(true)
  })

  it('renders nothing for an empty series or step list', () => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement
    renderTrajectory(svg, [], [0, 100], 0)
    expect(svg.children).toHaveLength(0)
    renderTrajectory(svg, series(), [], 0)
    expect(svg.children).toHaveLength(0)
  })
})

describe('renderLayerProfile', () => {
  it('draws one path + circles per series and the pinned-layer marker, with no end-labels', () => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement
    const profile: TrajectorySeries[] = [
      { id: 1, token: 'Ġthe', points: [[0, 0.1, 5], [1, 0.3, 2], [2, 0.2, 3]] },
      { id: 2, token: 'ĠTokyo', points: [[0, 0.0, 20], [1, 0.1, 8], [2, 0.4, 1]] },
    ]
    renderLayerProfile(svg, profile, 3, 2, { goldId: 2 })
    expect(svg.querySelectorAll('path')).toHaveLength(2)
    expect(svg.querySelectorAll('circle')).toHaveLength(6)
    // no ★ or token-label text nodes (layer profile has only axis/layer labels)
    const labels = Array.from(svg.querySelectorAll('text')).map((t) => t.textContent)
    expect(labels.some((t) => t?.includes('★'))).toBe(false)
    expect(labels.some((t) => t?.includes('Tokyo'))).toBe(false)
    const dashed = Array.from(svg.querySelectorAll('line')).filter((l) => l.getAttribute('stroke-dasharray'))
    expect(dashed).toHaveLength(1)
  })

  it('renders nothing for fewer than 2 layers (single-checkpoint models)', () => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg') as SVGSVGElement
    renderLayerProfile(svg, [{ id: 1, token: 'x', points: [[0, 0.5, 1]] }], 1, 0, {})
    expect(svg.children).toHaveLength(0)
  })
})
