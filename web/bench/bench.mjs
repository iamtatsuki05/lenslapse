// Browser benchmark harness for LensLapse: measures checkpoint load and probe latency across
// browser engines, execution providers, emulated CPU throttling, model sizes, and prompt lengths.
//
// Usage:
//   node bench/bench.mjs --base http://localhost:5199 --out ../../experiments/browser-bench.json
//
// Requires `npx playwright install chromium firefox webkit` and a running preview/dev server with
// LENSLAPSE_MODELS_DIR pointing at the converted models root.

import { writeFileSync } from 'node:fs'
import { chromium, firefox, webkit } from 'playwright'

const args = Object.fromEntries(
  process.argv.slice(2).map((a, i, all) => (a.startsWith('--') ? [a.slice(2), all[i + 1]] : null)).filter(Boolean)
)
const BASE = args.base ?? 'http://localhost:5199'
const OUT = args.out

const PROMPTS = {
  short: 'The capital of Japan is the city of', // 8 tokens
  long:
    'In the early days of natural language processing, researchers believed that grammar rules alone ' +
    'would be enough to understand human language, but decades of work have shown that', // ~32 tokens
}

const CELLS = [
  // execution-provider axis @ 70m
  { browser: 'chromium', ep: 'webgpu', model: 'pythia-70m', throttle: 1, prompt: 'short' },
  { browser: 'chromium', ep: 'wasm', model: 'pythia-70m', throttle: 1, prompt: 'short' },
  // emulated slower CPU (Chrome DevTools throttling; approximates a low-end laptop)
  { browser: 'chromium', ep: 'webgpu', model: 'pythia-70m', throttle: 4, prompt: 'short' },
  { browser: 'chromium', ep: 'wasm', model: 'pythia-70m', throttle: 4, prompt: 'short' },
  // model-size axis
  { browser: 'chromium', ep: 'webgpu', model: 'pythia-14m', throttle: 1, prompt: 'short' },
  { browser: 'chromium', ep: 'wasm', model: 'pythia-14m', throttle: 1, prompt: 'short' },
  { browser: 'chromium', ep: 'webgpu', model: 'pythia-160m', throttle: 1, prompt: 'short' },
  { browser: 'chromium', ep: 'wasm', model: 'pythia-160m', throttle: 1, prompt: 'short' },
  // browser-engine axis (WASM only: Firefox/WebKit have no ORT WebGPU EP path here)
  { browser: 'firefox', ep: 'wasm', model: 'pythia-70m', throttle: 1, prompt: 'short' },
  { browser: 'webkit', ep: 'wasm', model: 'pythia-70m', throttle: 1, prompt: 'short' },
  // prompt-length axis
  { browser: 'chromium', ep: 'webgpu', model: 'pythia-70m', throttle: 1, prompt: 'long' },
  { browser: 'chromium', ep: 'wasm', model: 'pythia-70m', throttle: 1, prompt: 'long' },
]

const LAUNCHERS = { chromium, firefox, webkit }

async function measureCell(page, cell, text) {
  return await page.evaluate(
    async ({ text }) => {
      const { state, engines, getEngine, enginesReady } = window.__lenslapse
      getEngine(state.model)
      const ok = await enginesReady.get(state.model)
      if (!ok) throw new Error('engine unavailable')
      const eng = engines.get(state.model)
      const step = eng.liveSteps().at(-1)
      const dropSessions = async () => {
        for (const pair of eng.sessions.values()) {
          try {
            await Promise.all([pair.backbone.release(), pair.lens.release()])
          } catch {}
        }
        eng.sessions.clear()
      }
      try {
        await caches.delete('lenslapse-models-v1')
      } catch {}
      await dropSessions()
      const t0 = performance.now()
      await eng.loadCheckpoint(step, () => {})
      const loadColdMs = performance.now() - t0
      await dropSessions()
      const t1 = performance.now()
      await eng.loadCheckpoint(step, () => {})
      const loadWarmMs = performance.now() - t1
      const lat = []
      for (let i = 0; i < 6; i++) {
        const r = await eng.probe(text, step, () => {})
        lat.push(r.timing.probe) // forward + lens + JS top-10 extraction
      }
      const enc = eng.tokenizer(text)
      return {
        step,
        backend: eng.backend,
        promptTokens: enc.input_ids.data.length,
        loadColdMs: Math.round(loadColdMs),
        loadWarmMs: Math.round(loadWarmMs),
        probeFirstMs: Math.round(lat[0]),
        probeWarmMeanMs: Math.round(lat.slice(1).reduce((a, b) => a + b, 0) / (lat.length - 1)),
      }
    },
    { text }
  )
}

const results = []
for (const cell of CELLS) {
  const headed = cell.browser === 'chromium' && cell.ep === 'webgpu'
  const browser = await LAUNCHERS[cell.browser].launch({ headless: !headed })
  try {
    const context = await browser.newContext()
    const page = await context.newPage()
    if (cell.browser === 'chromium' && cell.throttle > 1) {
      const cdp = await context.newCDPSession(page)
      await cdp.send('Emulation.setCPUThrottlingRate', { rate: cell.throttle })
    }
    // probe=off: the bench measures the in-browser ONNX path — never attach to a probe server
    // that happens to be running on this machine (it would replace the engine being measured).
    // fresh: bypass saved-probe replay — replayed results carry their *stored* timing, which
    // would silently report the first probe's latency as the warm mean.
    await page.goto(`${BASE}/?ep=${cell.ep}&probe=off&fresh#m=${cell.model}`)
    await page.waitForFunction(() => window.__lenslapse?.state?.model, null, { timeout: 30000 })
    const m = await measureCell(page, cell, PROMPTS[cell.prompt])
    const row = { ...cell, browserVersion: browser.version(), ...m }
    results.push(row)
    console.log(JSON.stringify(row))
  } catch (e) {
    const row = { ...cell, error: String(e).slice(0, 200) }
    results.push(row)
    console.log(JSON.stringify(row))
  } finally {
    await browser.close()
  }
}

// markdown table for the paper
const md = [
  '| browser | EP | model | CPU throttle | prompt tok | load cold/warm (ms) | probe first/warm (ms) |',
  '|---|---|---|---|---|---|---|',
  ...results.map((r) =>
    r.error
      ? `| ${r.browser} | ${r.ep} | ${r.model} | ${r.throttle}x | - | ERROR | ${r.error} |`
      : `| ${r.browser} | ${r.backend} | ${r.model} | ${r.throttle}x | ${r.promptTokens} | ${r.loadColdMs} / ${r.loadWarmMs} | ${r.probeFirstMs} / ${r.probeWarmMeanMs} |`
  ),
]
console.log(`\n${md.join('\n')}`)

if (OUT) {
  writeFileSync(OUT, JSON.stringify({ date: new Date().toISOString(), base: BASE, results }, null, 1))
  console.log(`\nwritten: ${OUT}`)
}
