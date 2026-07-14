import { describe, expect, it } from 'vitest'
import { acquisitionMap, fmtStep, layerProfileFromShard, nearestStep } from '../src/data'
import { logProb01 } from '../src/color'
import type { Prompt, Shard } from '../src/data'

// 1 layer x 2 positions, steps 0/10/20. Cell 0: B,A,A -> final A first at step 10.
// Cell 1: A,B,A -> final A first at step 0 (the map records first appearance, not stability).
const shard: Shard = {
  vocab: { '1': 'A', '2': 'B' },
  steps: {
    '0': { top: [[[[2, 0.3]], [[1, 0.2]]]] },
    '10': { top: [[[[1, 0.4]], [[2, 0.3]]]] },
    '20': { top: [[[[1, 0.6]], [[1, 0.5]]]] },
  },
}

describe('acquisitionMap', () => {
  it('finds the first step where each cell already predicts its final answer', () => {
    const acq = acquisitionMap(shard, [0, 10, 20])!
    expect(acq.layers).toBe(1)
    expect(acq.positions).toBe(2)
    expect(acq.firstIdx[0]).toEqual([1, 0])
    expect(acq.finalTop[0][0]).toEqual(['A', 0.6, 1])
    expect(acq.finalTop[0][1]).toEqual(['A', 0.5, 1])
  })

  it('returns null when the final step is missing from the shard', () => {
    expect(acquisitionMap(shard, [0, 10, 999])).toBeNull()
  })

  it('tolerates steps absent from the shard mid-scan', () => {
    const acq = acquisitionMap(shard, [0, 5, 10, 20])!
    expect(acq.firstIdx[0]).toEqual([2, 0]) // step 5 has no entry; A first seen at index 2 (step 10)
  })
})

describe('fmtStep', () => {
  it('formats steps compactly for grid cells', () => {
    expect(fmtStep(0)).toBe('0')
    expect(fmtStep(512)).toBe('512')
    expect(fmtStep(1000)).toBe('1k')
    expect(fmtStep(2738)).toBe('2.7k')
    expect(fmtStep(143000)).toBe('143k')
  })
})

describe('layerProfileFromShard', () => {
  it('returns p-vs-layer curves for the prompt targets, sorted by final-layer probability', () => {
    const prompt = { targets: [[], [1, 2]] } as unknown as Prompt
    const withTgt: Shard = {
      vocab: { '1': 'A', '2': 'B' },
      steps: {
        '20': {
          top: [[[[1, 0.6]], [[1, 0.5]]]],
          tgt: {
            '1': { p: [[0.1, 0.5], [0.2, 0.6]], r: [[3, 1], [2, 1]] },
            '2': { p: [[0.4, 0.7], [0.1, 0.2]], r: [[1, 1], [4, 5]] },
          },
        },
      },
    }
    const series = layerProfileFromShard(withTgt, prompt, 1, 20)
    expect(series.map((s) => s.token)).toEqual(['A', 'B']) // 0.6 at the last layer beats 0.2
    expect(series[0].points).toEqual([
      [0, 0.5, 1],
      [1, 0.6, 1],
    ])
    expect(layerProfileFromShard(withTgt, prompt, 1, 999)).toEqual([])
  })
})

describe('nearestStep', () => {
  it('keeps a compared model in lockstep with the closest checkpoint', () => {
    expect(nearestStep([0, 1000, 8000, 32000], 8000)).toBe(8000)
    expect(nearestStep([0, 1000, 8000, 32000], 12000)).toBe(8000)
    expect(nearestStep([0, 1000, 8000, 32000], 999999)).toBe(32000)
  })
})

describe('logProb01', () => {
  it('maps the log axis 1e-6..1 to 0..1 and reveals near-uniform distributions', () => {
    expect(logProb01(1)).toBe(1)
    expect(logProb01(1e-6)).toBe(0)
    expect(logProb01(0)).toBe(0) // floored, never -Infinity
    expect(logProb01(2e-5)).toBeGreaterThan(0.2) // ~vocab^-1 is visibly above the floor
    expect(logProb01(1e-3)).toBeCloseTo(0.5, 5)
  })
})
