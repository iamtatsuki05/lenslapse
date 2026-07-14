// jsdom's default URL is http://localhost:3000/, so this file covers the localhost behaviors;
// the non-localhost case lives in live.resolveProbeUrl.public.test.ts (different jsdom URL).
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('onnxruntime-web/webgpu', () => ({}))

// resolveProbeCandidates also runs once at module load; re-import the module per test so
// probeServerOrigin() reflects the URL/localStorage state under test.
async function importLive() {
  vi.resetModules()
  return await import('../src/live')
}

beforeEach(() => {
  localStorage.clear()
  history.replaceState(null, '', '/')
})

describe('resolveProbeCandidates on localhost', () => {
  it('connects to ?probe=<url> and remembers it in localStorage', async () => {
    history.replaceState(null, '', '/?probe=http://myhost:9000')
    const live = await importLive()
    expect(live.resolveProbeCandidates()).toEqual({ candidates: ['http://myhost:9000'], explicit: true })
    expect(localStorage.getItem('lenslapse-probe')).toBe('http://myhost:9000')
    expect(live.probeServerOrigin()).toBe('http://myhost:9000')
  })

  it('forgets the saved server with ?probe=off', async () => {
    localStorage.setItem('lenslapse-probe', 'http://myhost:9000')
    history.replaceState(null, '', '/?probe=off')
    const live = await importLive()
    expect(live.resolveProbeCandidates()).toEqual({ candidates: [], explicit: false })
    expect(localStorage.getItem('lenslapse-probe')).toBeNull()
    expect(live.probeServerOrigin()).toBeNull()
  })

  it('reuses the remembered server on a later visit without ?probe=', async () => {
    localStorage.setItem('lenslapse-probe', 'http://saved:8123')
    const live = await importLive()
    expect(live.resolveProbeCandidates()).toEqual({ candidates: ['http://saved:8123'], explicit: true })
    expect(live.probeServerOrigin()).toBe('http://saved:8123')
  })

  it('auto-detects its own origin first, then the default port, when nothing is configured', async () => {
    const live = await importLive()
    // jsdom serves from http://localhost:3000 — the app's own origin leads (the probe server
    // can serve the app itself), with the conventional port as the fallback
    expect(live.resolveProbeCandidates()).toEqual({
      candidates: ['http://localhost:3000', 'http://localhost:8017'],
      explicit: false,
    })
    expect(live.probeServerOrigin()).toBe('http://localhost:3000')
  })

  it('deduplicates when the app is served from the default port itself', async () => {
    // simulated by an explicit param equal to the origin; the Set-dedup path is covered above
    history.replaceState(null, '', '/?probe=http://localhost:3000')
    const live = await importLive()
    expect(live.resolveProbeCandidates()).toEqual({ candidates: ['http://localhost:3000'], explicit: true })
  })
})
