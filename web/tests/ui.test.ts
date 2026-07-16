import { describe, expect, it } from 'vitest'
import { buildSliderTicks, escapeHtml, setBadge } from '../src/ui'
import { displayToken } from '../src/grid'

describe('escapeHtml', () => {
  it('escapes &, < and >', () => {
    expect(escapeHtml('<a href="x">&amp;</a>')).toBe('&lt;a href="x"&gt;&amp;amp;&lt;/a&gt;')
    expect(escapeHtml('plain')).toBe('plain')
  })
})

describe('displayToken', () => {
  it('shows a leading space marker and newline glyphs', () => {
    expect(displayToken('Ġcat')).toBe('␣cat')
    expect(displayToken('▁cat')).toBe('␣cat')
    expect(displayToken(' cat')).toBe('␣cat')
    expect(displayToken('a\nb')).toBe('a⏎b')
    expect(displayToken('Ċ')).toBe('⏎')
    expect(displayToken('cat')).toBe('cat')
  })

  it('replaces every leading marker, not just the first (multi-space BPE, SP indentation)', () => {
    expect(displayToken('ĠĠ')).toBe('␣␣')
    expect(displayToken('▁▁▁▁if')).toBe('␣␣␣␣if')
    expect(displayToken('  cat')).toBe('␣␣cat')
    // markers past the leading run are part of the token text, left alone
    expect(displayToken('aĠb')).toBe('aĠb')
  })
})

describe('setBadge', () => {
  it('maps known modes to labels and warn styling', () => {
    const el = document.createElement('span')
    setBadge(el, 'webgpu')
    expect(el.textContent).toBe('live · WebGPU')
    expect(el.className).toBe('chip badge')
    setBadge(el, 'precomputed-only')
    expect(el.textContent).toBe('precomputed only')
    expect(el.className).toBe('chip badge warn')
  })

  it('falls back to the raw mode string for unknown modes', () => {
    const el = document.createElement('span')
    setBadge(el, 'something-new')
    expect(el.textContent).toBe('something-new')
    expect(el.className).toBe('chip badge')
  })
})

describe('buildSliderTicks', () => {
  it('renders one tick per step and marks live-capable steps', () => {
    const container = document.createElement('div')
    buildSliderTicks(container, [0, 10, 100], [10], 100)
    const ticks = [...container.children] as HTMLElement[]
    expect(ticks).toHaveLength(3)
    expect(ticks.map((t) => t.className)).toEqual(['tick', 'tick live', 'tick'])
    expect(ticks[1].title).toContain('(live-capable)')
    const lefts = ticks.map((t) => parseFloat(t.style.left))
    expect(lefts[0]).toBe(0)
    expect(lefts[2]).toBeCloseTo(100, 6)
    expect(lefts[0]).toBeLessThan(lefts[1])
    expect(lefts[1]).toBeLessThan(lefts[2])
  })

  it('emits no NaN positions when the only step is 0 (server-registered final model)', () => {
    // model_steps() returns [0] for a single-checkpoint model on the probe server: maxStep=0
    // made the log scale divide by zero and set left:"NaN%" on the lone tick
    const container = document.createElement('div')
    buildSliderTicks(container, [0], [0], 0)
    const tick = container.children[0] as HTMLElement
    expect(tick.style.left).toBe('0%')
  })
})
