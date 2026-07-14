/**
 * @vitest-environment jsdom
 * @vitest-environment-options {"url": "https://example.com/"}
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('onnxruntime-web/webgpu', () => ({}))

async function importLive() {
  vi.resetModules()
  return await import('../src/live')
}

beforeEach(() => {
  localStorage.clear()
})

describe('resolveProbeUrl on a public (non-localhost) host', () => {
  it('never auto-detects a probe server', async () => {
    const live = await importLive()
    expect(live.resolveProbeUrl()).toBeNull()
    expect(live.probeServerOrigin()).toBeNull()
  })

  it('still honors an explicitly remembered server', async () => {
    localStorage.setItem('lenslapse-probe', 'http://opted-in:8017')
    const live = await importLive()
    expect(live.resolveProbeUrl()).toBe('http://opted-in:8017')
  })
})
