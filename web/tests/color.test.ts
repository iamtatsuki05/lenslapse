import { describe, expect, it } from 'vitest'
import { cssColor, legendGradient, probColor, textColorOn } from '../src/color'

describe('probColor', () => {
  it('maps the endpoints to the first/last viridis control points', () => {
    expect(probColor(0)).toEqual([68, 1, 84])
    expect(probColor(1)).toEqual([253, 231, 37])
  })

  it('clamps probabilities outside [0, 1]', () => {
    expect(probColor(-0.5)).toEqual(probColor(0))
    expect(probColor(2)).toEqual(probColor(1))
  })

  it('interpolates to integer rgb channels', () => {
    for (const p of [0.1, 0.25, 0.5, 0.9]) {
      const rgb = probColor(p)
      expect(rgb).toHaveLength(3)
      for (const c of rgb) {
        expect(Number.isInteger(c)).toBe(true)
        expect(c).toBeGreaterThanOrEqual(0)
        expect(c).toBeLessThanOrEqual(255)
      }
    }
  })
})

describe('cssColor', () => {
  it('formats an rgb() string', () => {
    expect(cssColor([1, 2, 3])).toBe('rgb(1,2,3)')
  })
})

describe('textColorOn', () => {
  it('uses dark text on light backgrounds and light text on dark ones', () => {
    expect(textColorOn([255, 255, 255])).toBe('#111')
    expect(textColorOn([0, 0, 0])).toBe('#fff')
    expect(textColorOn(probColor(1))).toBe('#111') // bright yellow
    expect(textColorOn(probColor(0))).toBe('#fff') // dark purple
  })
})

describe('legendGradient', () => {
  it('builds an 11-stop linear gradient from 0% to 100%', () => {
    const g = legendGradient()
    expect(g.startsWith('linear-gradient(90deg, ')).toBe(true)
    expect(g.match(/rgb\(/g)).toHaveLength(11)
    expect(g).toContain(`${cssColor(probColor(0))} 0%`)
    expect(g).toContain(`${cssColor(probColor(1))} 100%`)
  })
})
