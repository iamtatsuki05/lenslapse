import 'fake-indexeddb/auto'
import { describe, expect, it } from 'vitest'
import { getProbe, probeKey, putProbe } from '../src/probeStore'

describe('probeKey', () => {
  it('is unique per (model, step, text)', () => {
    const keys = [
      probeKey('pythia-70m', 0, 'The capital'),
      probeKey('pythia-70m', 128, 'The capital'),
      probeKey('pythia-160m', 0, 'The capital'),
      probeKey('pythia-70m', 0, 'The capital '),
    ]
    expect(new Set(keys).size).toBe(keys.length)
  })
})

describe('IndexedDB probe store', () => {
  it('round-trips a stored probe result', async () => {
    const key = probeKey('pythia-70m', 512, 'roundtrip prompt')
    const value = {
      tokens: ['round', 'trip'],
      grid: { layers: 1, positions: 2, cells: [[{ token: 'a', prob: 0.5, top: [['a', 0.5, 7]] }]] },
      timing: { total: 12.5, forward: 8, probe: 10 },
      backend: 'wasm',
    }
    await putProbe(key, value)
    await expect(getProbe(key)).resolves.toEqual(value)
  })

  it('resolves null for a key that was never stored', async () => {
    await expect(getProbe(probeKey('nope', 0, 'missing'))).resolves.toBeNull()
  })

  it('overwrites an existing entry for the same key', async () => {
    const key = probeKey('pythia-70m', 0, 'overwrite')
    await putProbe(key, { backend: 'wasm' })
    await putProbe(key, { backend: 'webgpu' })
    await expect(getProbe(key)).resolves.toEqual({ backend: 'webgpu' })
  })
})
