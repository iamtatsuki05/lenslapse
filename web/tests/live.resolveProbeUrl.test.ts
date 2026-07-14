// jsdom's default URL is http://localhost:3000/, so this file covers the localhost behaviors;
// the non-localhost case lives in live.resolveProbeUrl.public.test.ts (different jsdom URL).
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('onnxruntime-web/webgpu', () => ({}))

// resolveProbeUrl also runs once at module load (PROBE_URL); re-import the module per test so
// probeServerOrigin() reflects the URL/localStorage state under test.
async function importLive() {
  vi.resetModules()
  return await import('../src/live')
}

beforeEach(() => {
  localStorage.clear()
  history.replaceState(null, '', '/')
})

describe('resolveProbeUrl on localhost', () => {
  it('connects to ?probe=<url> and remembers it in localStorage', async () => {
    history.replaceState(null, '', '/?probe=http://myhost:9000')
    const live = await importLive()
    expect(live.resolveProbeUrl()).toBe('http://myhost:9000')
    expect(localStorage.getItem('lenslapse-probe')).toBe('http://myhost:9000')
    expect(live.probeServerOrigin()).toBe('http://myhost:9000')
  })

  it('forgets the saved server with ?probe=off', async () => {
    localStorage.setItem('lenslapse-probe', 'http://myhost:9000')
    history.replaceState(null, '', '/?probe=off')
    const live = await importLive()
    expect(live.resolveProbeUrl()).toBeNull()
    expect(localStorage.getItem('lenslapse-probe')).toBeNull()
    expect(live.probeServerOrigin()).toBeNull()
  })

  it('reuses the remembered server on a later visit without ?probe=', async () => {
    localStorage.setItem('lenslapse-probe', 'http://saved:8123')
    const live = await importLive()
    expect(live.resolveProbeUrl()).toBe('http://saved:8123')
    expect(live.probeServerOrigin()).toBe('http://saved:8123')
  })

  it('auto-detects the default port on localhost when nothing is configured', async () => {
    const live = await importLive()
    expect(live.resolveProbeUrl()).toBe('http://localhost:8017')
    expect(live.probeServerOrigin()).toBe('http://localhost:8017')
  })
})
