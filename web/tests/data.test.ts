import { describe, expect, it } from 'vitest'
import { gridFromShard, languageTag, promptLang, promptLangLabel, trajectoryFromShard } from '../src/data'
import type { ModelEntry, Prompt, Shard } from '../src/data'

// Minimal 2-layer × 2-position shard with two steps (0 and 128); step 64 is deliberately absent.
// Token id 99 has no vocab entry so it must render as '?'.
const shard: Shard = {
  vocab: { '10': 'Ġthe', '11': 'Ġcat', '12': 'Ġsat' },
  steps: {
    '0': {
      top: [
        [
          [
            [10, 0.6],
            [11, 0.3],
          ],
          [
            [11, 0.5],
            [99, 0.2],
          ],
        ],
        [
          [
            [12, 0.9],
            [10, 0.05],
          ],
          [
            [10, 0.4],
            [12, 0.3],
          ],
        ],
      ],
      tgt: {
        '11': {
          p: [
            [0.1, 0.2],
            [0.3, 0.4],
          ],
          r: [
            [5, 4],
            [3, 2],
          ],
        },
        '12': {
          p: [
            [0.01, 0.6],
            [0.02, 0.9],
          ],
          r: [
            [9, 2],
            [8, 1],
          ],
        },
      },
    },
    '128': {
      top: [
        [
          [
            [11, 0.7],
            [10, 0.1],
          ],
          [
            [12, 0.6],
            [11, 0.2],
          ],
        ],
        [
          [
            [10, 0.8],
            [12, 0.1],
          ],
          [
            [12, 0.5],
            [10, 0.2],
          ],
        ],
      ],
      tgt: {
        '11': {
          p: [
            [0.15, 0.25],
            [0.35, 0.45],
          ],
          r: [
            [6, 5],
            [4, 2],
          ],
        },
      },
    },
  },
}

const prompt: Prompt = {
  id: 0,
  text: 'the cat',
  ids: [10, 11],
  tokens: ['the', 'Ġcat'],
  targets: [
    [11, 12],
    [11, 12],
  ],
}

describe('gridFromShard', () => {
  it('builds the grid for an existing step', () => {
    const g = gridFromShard(shard, 0)
    expect(g).not.toBeNull()
    expect(g!.layers).toBe(2)
    expect(g!.positions).toBe(2)
    expect(g!.cells[0][0]).toEqual({
      token: 'Ġthe',
      prob: 0.6,
      top: [
        ['Ġthe', 0.6, 10],
        ['Ġcat', 0.3, 11],
      ],
    })
    expect(g!.cells[1][0].token).toBe('Ġsat')
    expect(g!.cells[1][0].prob).toBe(0.9)
  })

  it('returns null for a step that is not in the shard', () => {
    expect(gridFromShard(shard, 64)).toBeNull()
    expect(gridFromShard(shard, 7)).toBeNull()
  })

  it('falls back to "?" for token ids missing from the vocab', () => {
    const g = gridFromShard(shard, 0)
    expect(g!.cells[0][1].top[1]).toEqual(['?', 0.2, 99])
  })
})

describe('trajectoryFromShard', () => {
  it('collects target-token points sorted by final probability (descending)', () => {
    const series = trajectoryFromShard(shard, prompt, 1, 1, [0, 128])
    // sat's series ends at 0.9 (only step 0), cat's at 0.45 (step 128)
    expect(series.map((s) => s.id)).toEqual([12, 11])
    const cat = series[1]
    expect(cat.token).toBe('Ġcat')
    expect(cat.points).toEqual([
      [0, 0.4, 2],
      [128, 0.45, 2],
    ])
  })

  it('skips steps that are absent from the shard or missing the target id', () => {
    const series = trajectoryFromShard(shard, prompt, 1, 1, [0, 64, 128])
    const cat = series.find((s) => s.id === 11)!
    expect(cat.points.map(([step]) => step)).toEqual([0, 128]) // step 64 is not in the shard
    const sat = series.find((s) => s.id === 12)!
    expect(sat.points).toEqual([[0, 0.9, 1]]) // id 12 has no tgt entry at step 128
  })

  it('orders by the last point probability even when series lengths differ', () => {
    const series = trajectoryFromShard(shard, prompt, 1, 1, [0, 64, 128])
    // sat ends at 0.9 (step 0), cat at 0.45 (step 128) -> sat first
    expect(series.map((s) => s.id)).toEqual([12, 11])
  })

  it('returns an empty list when the position has no targets', () => {
    expect(trajectoryFromShard(shard, prompt, 1, 5, [0, 128])).toEqual([])
  })
})

describe('languageTag', () => {
  const entry = (languages?: string[], languageLabel?: string): ModelEntry => ({
    id: 'm',
    hf: 'm',
    ...(languages ? { languages } : {}),
    ...(languageLabel ? { languageLabel } : {}),
  })

  it('returns null for a model with no languages on file', () => {
    expect(languageTag(entry())).toBeNull()
  })

  it('returns null for a single-language (English-only) model', () => {
    expect(languageTag(entry(['en']))).toBeNull()
  })

  it('joins two or more known language codes by name', () => {
    expect(languageTag(entry(['zh', 'en']))).toBe('Chinese/English')
  })

  it('falls back to the raw code for an unrecognized language', () => {
    expect(languageTag(entry(['fr', 'en']))).toBe('fr/English')
  })

  it('prefers an explicit languageLabel override over the derived tag', () => {
    expect(languageTag(entry(['zh', 'en'], 'Multilingual'))).toBe('Multilingual')
  })
})

describe('promptLang', () => {
  it('detects English text with no CJK characters', () => {
    expect(promptLang('The capital of Japan is the city of')).toBe('en')
  })

  it('detects Chinese text containing CJK ideographs', () => {
    expect(promptLang('中国的首都是')).toBe('zh')
  })

  it('detects Chinese text mixed with Latin punctuation/digits', () => {
    expect(promptLang('# 计算两数之和\ndef add(a, b):\n    return a +')).toBe('zh')
  })
})

describe('promptLangLabel', () => {
  it('labels zh as 中文 and en as English', () => {
    expect(promptLangLabel('zh')).toBe('中文')
    expect(promptLangLabel('en')).toBe('English')
  })
})
