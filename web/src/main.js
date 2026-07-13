import './style.css'
import { gridFromShard, loadIndex, loadModels, loadShard, trajectoryFromShard } from './data.js'
import { LensGrid } from './grid.js'
import { LiveEngine } from './live.js'
import { renderTrajectory } from './traj.js'
import { buildGallery, buildSliderTicks, hideTooltip, setBadge, showTooltip } from './ui.js'
import { legendGradient } from './color.js'

const $ = (id) => document.getElementById(id)

const state = {
  model: null, // model id from models.json
  mode: 'pre', // 'pre' (precomputed prompt) | 'live' (free-text probe)
  promptId: 0,
  liveText: '',
  stepIdx: 0,
  pinned: null, // {layer, pos}
  steps: [],
  liveResult: null, // cached last live probe {text, step, grid, tokens}
}

let catalog = null // models.json
let index = null // current model's index.json
const engines = new Map() // model id -> LiveEngine
const enginesReady = new Map() // model id -> Promise<boolean>

// generation token: bumped on every user-initiated view change so that slow async loads
// (shards, live probes) resolving out of order cannot paint a stale view
let viewGen = 0

const status = (msg) => {
  $('status-line').textContent = msg ?? ''
}

function modelInfo(id = state.model) {
  return catalog.models.find((m) => m.id === id)
}

function getEngine(id = state.model) {
  if (!engines.has(id)) {
    const eng = new LiveEngine(id, modelInfo(id).hf)
    engines.set(id, eng)
    enginesReady.set(
      id,
      eng.init(status).then((ok) => {
        if (id === state.model) refreshBadgeAndTicks()
        return ok
      })
    )
  }
  return engines.get(id)
}

function currentStep() {
  return state.steps[state.stepIdx] ?? 0
}

function currentPrompt() {
  return index.prompts.find((p) => p.id === state.promptId) ?? index.prompts[0]
}

/* ---------- rendering ---------- */

const grid = new LensGrid($('lens-canvas'), {
  onHover(cellInfo, evt) {
    if (cellInfo) showTooltip($('tooltip'), cellInfo, evt, gridTokens())
    else hideTooltip($('tooltip'))
  },
  onPin(pinned) {
    state.pinned = pinned ? { layer: pinned.layer, pos: pinned.pos } : null
    syncHash()
    refreshTrajectory()
  },
})

function gridTokens() {
  return state.mode === 'live' && state.liveResult ? state.liveResult.tokens : currentPrompt().tokens
}

async function refreshGrid() {
  if (state.mode === 'live') {
    if (!state.liveResult) return
    grid.setData(state.liveResult.grid, state.liveResult.tokens, state.pinned)
    return
  }
  const gen = viewGen
  const shard = await loadShard(state.model, state.promptId)
  if (gen !== viewGen) return
  const g = gridFromShard(shard, currentStep())
  if (!g) return
  if (state.pinned && (state.pinned.layer >= g.layers || state.pinned.pos >= g.positions)) {
    state.pinned = null
    grid.pinned = null
  }
  grid.setData(g, currentPrompt().tokens, state.pinned)
}

async function refreshTrajectory() {
  const svg = $('traj-svg')
  const sub = $('traj-subtitle')
  if (state.mode === 'live') {
    svg.replaceChildren()
    sub.textContent = 'trajectories are available for curated (precomputed) prompts'
    return
  }
  if (!state.pinned) {
    svg.replaceChildren()
    sub.textContent = 'click a cell in the grid'
    return
  }
  const gen = viewGen
  const prompt = currentPrompt()
  const shard = await loadShard(state.model, state.promptId)
  if (gen !== viewGen) return
  const { layer, pos } = state.pinned
  const series = trajectoryFromShard(shard, prompt, layer, pos, index.steps)
  sub.textContent = `${layer === 0 ? 'embedding' : `layer ${layer}`}, position ${pos} (“${prompt.tokens[pos]}” →)`
  renderTrajectory(svg, series, index.steps, currentStep(), {
    goldId: pos === prompt.tokens.length - 1 ? prompt.gold_id : undefined,
  })
}

function refreshStepUI() {
  const s = currentStep()
  $('slider-value').textContent = s.toLocaleString()
  $('step-readout').textContent = `step ${s.toLocaleString()}`
}

function refreshBadgeAndTicks() {
  const eng = engines.get(state.model)
  if (state.mode === 'pre') setBadge($('backend-badge'), eng?.available ? 'precomputed' : 'precomputed-only')
  buildSliderTicks($('slider-ticks'), state.steps, eng?.liveSteps() ?? [], state.steps.at(-1))
}

/* ---------- slider (index into steps, ticks positioned on log scale) ---------- */

function setupSliderRange() {
  const slider = $('step-slider')
  slider.max = String(state.steps.length - 1)
  slider.value = String(state.stepIdx)
  $('slider-max-label').textContent = `step ${state.steps.at(-1).toLocaleString()}`
  refreshBadgeAndTicks()
}

let liveDebounce = null
function onStepChanged() {
  viewGen++
  syncHash()
  if (state.mode === 'live') {
    // debounce: re-probing on every slider notch would queue big downloads
    clearTimeout(liveDebounce)
    liveDebounce = setTimeout(() => runLiveProbe(state.liveText), 350)
  } else {
    refreshGrid()
    refreshTrajectory()
  }
}

/* ---------- live probing ---------- */

async function runLiveProbe(text) {
  if (!text.trim()) return
  const gen = ++viewGen
  const engine = getEngine()
  await enginesReady.get(engine.modelId)
  if (gen !== viewGen) return // model/view switched while the engine was initializing
  if (!engine.available) {
    status('live probing unavailable — model host unreachable; showing precomputed prompts only')
    return
  }
  const step = nearestLiveStep(currentStep())
  $('live-btn').disabled = true
  try {
    const res = await engine.probe(text, step, status)
    if (gen !== viewGen) return // user switched views while the probe ran
    state.mode = 'live'
    state.liveText = text
    state.liveResult = { ...res, text, step }
    state.pinned = null
    setBadge($('backend-badge'), res.backend)
    status(
      `live probe @ step ${step.toLocaleString()} — forward+lens+top-k ${res.timing.probe.toFixed(0)}ms on ${res.backend} (fully in your browser)`
    )
    grid.pinned = null
    refreshGrid()
    refreshTrajectory()
    syncHash()
  } catch (e) {
    status(`live probe failed: ${e.message}`)
  } finally {
    $('live-btn').disabled = false
  }
}

function nearestLiveStep(step) {
  const ls = engines.get(state.model)?.liveSteps() ?? []
  if (!ls.length) return step
  return ls.reduce((a, b) => (Math.abs(b - step) < Math.abs(a - step) ? b : a))
}

/* ---------- permalink ---------- */

function syncHash() {
  const h = new URLSearchParams()
  h.set('m', state.model)
  if (state.mode === 'live' && state.liveText) h.set('q', state.liveText)
  else h.set('p', String(state.promptId))
  h.set('s', String(currentStep()))
  if (state.pinned) h.set('pin', `${state.pinned.layer},${state.pinned.pos}`)
  history.replaceState(null, '', `#${h.toString()}`)
}

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1))
  if (h.has('m') && catalog.models.some((m) => m.id === h.get('m'))) state.model = h.get('m')
  if (h.has('p')) {
    const pid = Number(h.get('p'))
    state.promptId = index.prompts.some((p) => p.id === pid) ? pid : 0
  }
  const s = Number(h.get('s'))
  if (h.has('s') && !Number.isNaN(s)) {
    const idx = state.steps.indexOf(s)
    state.stepIdx = idx >= 0 ? idx : state.steps.length - 1
  } else {
    state.stepIdx = state.steps.length - 1
  }
  if (h.has('q')) {
    // live-probe deep link: the pin (if any) belongs to the live grid, which does not exist yet
    state.liveText = h.get('q')
    $('live-input').value = state.liveText
  } else if (h.has('pin')) {
    const [l, p] = h.get('pin').split(',').map(Number)
    // upper bounds are re-validated in refreshGrid once the shard's dimensions are known
    if (Number.isInteger(l) && Number.isInteger(p) && l >= 0 && p >= 0) state.pinned = { layer: l, pos: p }
  }
}

/* ---------- model & prompt selection ---------- */

function rebuildPromptSelect() {
  const sel = $('prompt-select')
  sel.replaceChildren()
  for (const p of index.prompts) {
    const opt = document.createElement('option')
    opt.value = String(p.id)
    opt.textContent = p.text.replaceAll('\n', '⏎')
    sel.appendChild(opt)
  }
  sel.value = String(state.promptId)
}

async function switchModel(id) {
  const gen = ++viewGen
  clearTimeout(liveDebounce)
  const prev = state.model
  let nextIndex
  try {
    nextIndex = await loadIndex(id)
  } catch (e) {
    status(`model ${id} unavailable: ${e.message}`)
    $('model-select').value = prev
    return
  }
  if (gen !== viewGen) return // a newer switch/interaction superseded this one
  state.model = id
  index = nextIndex
  state.steps = index.steps
  state.mode = 'pre'
  state.promptId = Math.min(state.promptId, index.prompts.length - 1)
  state.stepIdx = Math.min(state.stepIdx, state.steps.length - 1)
  state.pinned = null
  grid.pinned = null
  state.liveResult = null
  rebuildPromptSelect()
  setupSliderRange()
  refreshStepUI()
  getEngine(id) // kick off engine init in the background
  refreshBadgeAndTicks()
  syncHash()
  await refreshGrid()
  await refreshTrajectory()
}

/* ---------- boot ---------- */

async function boot() {
  status('loading precomputed data…')
  try {
    catalog = await loadModels()
    state.model = catalog.default ?? catalog.models[0].id
    // hash may override the model; read it before loading the index
    const h = new URLSearchParams(location.hash.slice(1))
    if (h.has('m') && catalog.models.some((m) => m.id === h.get('m'))) state.model = h.get('m')
    index = await loadIndex(state.model)
  } catch (e) {
    status(`failed to load precomputed data: ${e.message}`)
    return
  }
  state.steps = index.steps

  const msel = $('model-select')
  for (const m of catalog.models) {
    const opt = document.createElement('option')
    opt.value = m.id
    opt.textContent = m.label ?? m.hf
    msel.appendChild(opt)
  }
  msel.value = state.model
  msel.addEventListener('change', () => switchModel(msel.value))

  const sel = $('prompt-select')
  sel.addEventListener('change', () => {
    viewGen++
    clearTimeout(liveDebounce)
    state.mode = 'pre'
    state.promptId = Number(sel.value)
    state.pinned = null
    grid.pinned = null
    refreshBadgeAndTicks()
    status('')
    syncHash()
    refreshGrid()
    refreshTrajectory()
  })

  readHash()
  rebuildPromptSelect()
  const slider = $('step-slider')
  slider.addEventListener('input', () => {
    state.stepIdx = Number(slider.value)
    refreshStepUI()
    onStepChanged()
  })
  setupSliderRange()
  refreshStepUI()
  getEngine(state.model) // engine init is non-blocking for first paint

  $('grid-legend').innerHTML =
    `<span>lens top-1 probability</span><span class="swatch" style="background:${legendGradient()}"></span><span>0 → 1</span>`

  buildGallery($('gallery-cards'), index.prompts, async (p, card) => {
    // the story cards narrate the 70M model's training run; switch to it if needed
    if (state.model !== 'pythia-70m' && catalog.models.some((m) => m.id === 'pythia-70m')) {
      $('model-select').value = 'pythia-70m'
      await switchModel('pythia-70m')
    }
    viewGen++
    clearTimeout(liveDebounce)
    state.mode = 'pre'
    state.promptId = p.id
    $('prompt-select').value = String(p.id)
    const idx = state.steps.indexOf(card.step)
    state.stepIdx = idx >= 0 ? idx : state.steps.length - 1
    $('step-slider').value = String(state.stepIdx)
    state.pinned = card.pin === 'lastLayerLastPos' ? { layer: gridLayers() - 1, pos: p.tokens.length - 1 } : null
    grid.pinned = state.pinned ? { ...state.pinned } : null
    refreshBadgeAndTicks()
    refreshStepUI()
    syncHash()
    refreshGrid()
    refreshTrajectory()
  })

  $('live-btn').addEventListener('click', () => runLiveProbe($('live-input').value))
  $('live-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !$('live-btn').disabled) runLiveProbe($('live-input').value)
  })

  $('permalink-btn').addEventListener('click', async () => {
    await navigator.clipboard.writeText(location.href)
    status('permalink copied to clipboard')
  })

  const exportView = () => ({
    grid,
    trajSvg: state.pinned && state.mode === 'pre' ? $('traj-svg') : null,
    meta: {
      model: modelInfo()?.label ?? state.model,
      prompt: state.mode === 'live' ? state.liveText : currentPrompt().text,
      step: state.mode === 'live' ? (state.liveResult?.step ?? currentStep()) : currentStep(),
      pinned: state.pinned,
      permalink: location.href,
    },
  })
  const runExport = async (fn, label) => {
    $('export-menu').removeAttribute('open')
    try {
      status(`exporting ${label}…`)
      const { exportPng, exportPdf } = await import('./export.js')
      await (fn === 'png' ? exportPng : exportPdf)(exportView())
      status(`${label} exported`)
    } catch (e) {
      status(`export failed: ${e.message}`)
    }
  }
  $('export-png').addEventListener('click', () => runExport('png', 'PNG figure'))
  $('export-pdf').addEventListener('click', () => runExport('pdf', 'PDF figure'))

  try {
    await refreshGrid()
    await refreshTrajectory()
    status('')
  } catch (e) {
    status(`failed to render precomputed data: ${e.message}`)
  }

  // deep-linked live probe
  if (state.liveText) {
    const ok = await enginesReady.get(state.model)
    if (ok) runLiveProbe(state.liveText)
  }
}

function gridLayers() {
  return grid.grid?.layers ?? 7
}

// debug/measurement hook (used by the benchmark and fidelity harnesses)
window.__lenslapse = { state, engines, grid, getEngine, enginesReady }

boot()
