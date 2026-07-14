import { describe, expect, it, vi } from 'vitest'

// onnxruntime-web is only needed by LiveEngine methods; topkSoftmax is pure JS.
vi.mock('onnxruntime-web/webgpu', () => ({}))

import { topkSoftmax } from '../src/live'

const decode = (id: number) => `t${id}`

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits)
  const exps = logits.map((v) => Math.exp(v - max))
  const Z = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / Z)
}

describe('topkSoftmax', () => {
  const logits = [1, 3, 2, -1, 0]

  it('returns k entries sorted by probability (descending) with matching ids', () => {
    const cell = topkSoftmax(new Float32Array(logits), 0, logits.length, 3, decode)
    expect(cell.top).toHaveLength(3)
    expect(cell.top.map(([, , id]) => id)).toEqual([1, 2, 0])
    const probs = cell.top.map(([, p]) => p)
    expect(probs).toEqual([...probs].sort((a, b) => b - a))
  })

  it('matches a reference softmax and sums to ~1 when k covers the full vocab', () => {
    const cell = topkSoftmax(new Float32Array(logits), 0, logits.length, logits.length, decode)
    const expected = softmax(logits)
    for (const [, p, id] of cell.top) expect(p).toBeCloseTo(expected[id], 6)
    const sum = cell.top.reduce((a, [, p]) => a + p, 0)
    expect(sum).toBeCloseTo(1, 6)
  })

  it('exposes the argmax as the cell token/prob and decodes ids', () => {
    const cell = topkSoftmax(new Float32Array(logits), 0, logits.length, 3, decode)
    expect(cell.token).toBe('t1')
    expect(cell.prob).toBeCloseTo(softmax(logits)[1], 6)
    for (const [tok, , id] of cell.top) expect(tok).toBe(`t${id}`)
  })

  it('respects the offset into a flat [N, V] logits buffer', () => {
    const V = logits.length
    const row2 = [0, -2, 5, 1, 2]
    const data = new Float32Array([...logits, ...row2])
    const cell = topkSoftmax(data, V, V, 2, decode)
    expect(cell.top.map(([, , id]) => id)).toEqual([2, 4])
    expect(cell.prob).toBeCloseTo(softmax(row2)[2], 6)
  })
})
