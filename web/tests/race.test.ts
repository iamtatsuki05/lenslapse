import { beforeEach, describe, expect, it } from 'vitest'
import { RACE_ROW_H, renderRace } from '../src/race'
import type { TopEntry } from '../src/data'

const top = (rows: [string, number, number][]): TopEntry[] => rows

let el: HTMLElement

beforeEach(() => {
  el = document.createElement('div')
})

describe('renderRace', () => {
  it('renders rows in rank order with widths relative to the leader', () => {
    renderRace(el, top([['Ġthe', 0.5, 1], ['ĠTokyo', 0.25, 2]]), { goldId: 2 })
    const rows = [...el.querySelectorAll<HTMLElement>('.race-row')]
    expect(rows).toHaveLength(2)
    expect(rows[0].style.top).toBe('0px')
    expect(rows[1].style.top).toBe(`${RACE_ROW_H}px`)
    expect(rows[0].querySelector<HTMLElement>('.race-bar')!.style.width).toBe('100%')
    expect(rows[1].querySelector<HTMLElement>('.race-bar')!.style.width).toBe('50%')
    expect(rows[0].querySelector('.race-tok')!.textContent).toBe('␣the')
    expect(rows[1].classList.contains('gold')).toBe(true)
  })

  it('reuses row elements across renders so CSS transitions can animate rank swaps', () => {
    renderRace(el, top([['Ġthe', 0.5, 1], ['ĠTokyo', 0.25, 2]]))
    const tokyo = el.querySelector<HTMLElement>('.race-row[data-id="2"]')!
    renderRace(el, top([['ĠTokyo', 0.6, 2], ['Ġthe', 0.3, 1]]))
    expect(el.querySelector<HTMLElement>('.race-row[data-id="2"]')).toBe(tokyo) // same node, new rank
    expect(tokyo.style.top).toBe('0px')
  })

  it('removes rows that dropped out of the top-k and hides when unpinned', () => {
    renderRace(el, top([['Ġthe', 0.5, 1], ['ĠTokyo', 0.25, 2]]))
    renderRace(el, top([['Ġthe', 0.5, 1]]))
    expect(el.querySelectorAll('.race-row')).toHaveLength(1)
    renderRace(el, [])
    expect(el.hidden).toBe(true)
    expect(el.children).toHaveLength(0)
  })
})
