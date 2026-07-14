import './style.css'
import { gridFromShard, loadIndex, loadModels, loadShard, trajectoryFromShard } from './data'
import { LensGrid } from './grid'
import { LiveEngine, fetchServerModels } from './live'
import { setupManageModels } from './manage'
import { renderTrajectory } from './traj'
import { buildGallery, buildSliderTicks, hideTooltip, setBadge, showTooltip } from './ui'
import { legendGradient } from './color'
import type { ModelCatalog, ModelEntry, ModelIndex, Prompt } from './data'
import type { PinnedCell } from './grid'
import type { ProbeResult, ServerModel } from './live'

const $ = <T extends Element = HTMLElement>(id: string) => document.getElementById(id) as unknown as T

type LiveResult = ProbeResult & { text: string; step: number }

const state = {
  model: null as string | null, // model id from models.json
  mode: 'pre' as 'pre' | 'live', // 'pre' (precomputed prompt) | 'live' (free-text probe)
  promptId: 0,
  liveText: '',
  stepIdx: 0,
  pinned: null as PinnedCell | null, // {layer, pos}
  steps: [] as number[],
  liveResult: null as LiveResult | null, // cached last live probe {text, step, grid, tokens}
}

let catalog: ModelCatalog | null = null // models.json
let index: ModelIndex | null = null // current model's index.json
const engines = new Map<string, LiveEngine>() // model id -> LiveEngine
const enginesReady = new Map<string, Promise<boolean>>() // model id -> Promise<boolean>

// generation token: bumped on every user-initiated view change so that slow async loads
// (shards, live probes) resolving out of order cannot paint a stale view
let viewGen = 0

const status = (msg?: string) => {
  $('status-line').textContent = msg ?? ''
}

function modelInfo(id: string | null = state.model): ModelEntry | undefined {
  return catalog!.models.find((m) => m.id === id)
}

function getEngine(id: string = state.model!): LiveEngine {
  if (!engines.has(id)) {
    const eng = new LiveEngine(id, modelInfo(id)!.hf)
    engines.set(id, eng)
    enginesReady.set(
      id,
      eng.init(status).then((ok) => {
        if (id === state.model) refreshBadgeAndTicks()
        return ok
      })
    )
  }
  return engines.get(id)!
}

function currentStep(): number {
  return state.steps[state.stepIdx] ?? 0
}

function currentPrompt(): Prompt {
  return index!.prompts.find((p) => p.id === state.promptId) ?? index!.prompts[0]
}

/* ---------- rendering ---------- */

const grid = new LensGrid($<HTMLCanvasElement>('lens-canvas'), {
  onHover(cellInfo, evt) {
    if (cellInfo) showTooltip($('tooltip'), cellInfo, evt!, gridTokens())
    else hideTooltip($('tooltip'))
  },
  onPin(pinned) {
    state.pinned = pinned ? { layer: pinned.layer, pos: pinned.pos } : null
    syncHash()
    refreshTrajectory()
  },
})

function gridTokens(): string[] {
  return state.mode === 'live' && state.liveResult ? state.liveResult.tokens : (currentPrompt()?.tokens ?? [])
}

async function refreshGrid(): Promise<void> {
  if (state.mode === 'live') {
    if (!state.liveResult) return
    grid.setData(state.liveResult.grid, state.liveResult.tokens, state.pinned)
    return
  }
  const gen = viewGen
  const shard = await loadShard(state.model!, state.promptId)
  if (gen !== viewGen) return
  const g = gridFromShard(shard, currentStep())
  if (!g) return
  if (state.pinned && (state.pinned.layer >= g.layers || state.pinned.pos >= g.positions)) {
    state.pinned = null
    grid.pinned = null
  }
  grid.setData(g, currentPrompt().tokens, state.pinned)
}

async function refreshTrajectory(): Promise<void> {
  const svg = $<SVGSVGElement>('traj-svg')
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
  const shard = await loadShard(state.model!, state.promptId)
  if (gen !== viewGen) return
  const { layer, pos } = state.pinned
  const series = trajectoryFromShard(shard, prompt, layer, pos, index!.steps)
  sub.textContent = `${layer === 0 ? 'embedding' : `layer ${layer}`}, position ${pos} (“${prompt.tokens[pos]}” →)`
  renderTrajectory(svg, series, index!.steps, currentStep(), {
    goldId: pos === prompt.tokens.length - 1 ? prompt.gold_id : undefined,
  })
}

function refreshStepUI(): void {
  const s = currentStep()
  $('slider-value').textContent = s.toLocaleString()
  $('step-readout').textContent = `step ${s.toLocaleString()}`
}

function refreshBadgeAndTicks(): void {
  const eng = engines.get(state.model!)
  if (state.mode === 'pre') setBadge($('backend-badge'), eng?.available ? 'precomputed' : 'precomputed-only')
  const live = eng?.server ? state.steps : (eng?.liveSteps() ?? [])
  buildSliderTicks($('slider-ticks'), state.steps, live, state.steps.at(-1)!)
}

/* ---------- slider (index into steps, ticks positioned on log scale) ---------- */

function setupSliderRange(): void {
  const slider = $<HTMLInputElement>('step-slider')
  slider.max = String(state.steps.length - 1)
  slider.value = String(state.stepIdx)
  $('slider-max-label').textContent = `step ${state.steps.at(-1)!.toLocaleString()}`
  refreshBadgeAndTicks()
}

let liveDebounce: ReturnType<typeof setTimeout> | undefined
function onStepChanged(): void {
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

let probeSeq = 0 // only the latest probe run controls the button / error status
async function runLiveProbe(text: string): Promise<void> {
  if (!text.trim()) return
  const gen = ++viewGen
  const run = ++probeSeq
  const engine = getEngine()
  await enginesReady.get(engine.modelId)
  if (gen !== viewGen) return // model/view switched while the engine was initializing
  if (!engine.available) {
    status(
      index!.prompts.length
        ? 'live probing unavailable — model host unreachable; showing precomputed prompts only'
        : 'live probing unavailable — this model needs the probe server, which is unreachable'
    )
    return
  }
  const step = nearestLiveStep(currentStep())
  $<HTMLButtonElement>('live-btn').disabled = true
  try {
    const res = await engine.probe(text, step, status)
    if (gen !== viewGen) return // user switched views while the probe ran
    state.mode = 'live'
    state.liveText = text
    state.liveResult = { ...res, text, step }
    state.pinned = null
    setBadge($('backend-badge'), res.backend)
    const where = res.backend === 'server' ? 'on the local probe server' : 'fully in your browser'
    const note = res.replayed ? ' · replayed from saved probe' : res.serverCached ? ' · from server cache' : ''
    status(
      `live probe @ step ${step.toLocaleString()} — forward+lens+top-k ${res.timing.probe.toFixed(0)}ms on ${res.backend} (${where})${note}`
    )
    grid.pinned = null
    refreshGrid()
    refreshTrajectory()
    syncHash()
  } catch (e) {
    // a stale probe's late failure must not clobber the status of the view the user is on now
    if (gen === viewGen) status(`live probe failed: ${(e as Error).message}`)
  } finally {
    if (run === probeSeq) $<HTMLButtonElement>('live-btn').disabled = false
  }
}

function nearestLiveStep(step: number): number {
  const eng = engines.get(state.model!)
  if (eng?.server) return step // the probe server can load any suite checkpoint
  const ls = eng?.liveSteps() ?? []
  if (!ls.length) return step
  return ls.reduce((a, b) => (Math.abs(b - step) < Math.abs(a - step) ? b : a))
}

/* ---------- permalink ---------- */

function syncHash(): void {
  const h = new URLSearchParams()
  h.set('m', state.model!)
  if (state.mode === 'live' && state.liveText) h.set('q', state.liveText)
  else if (index!.prompts.length) h.set('p', String(state.promptId))
  h.set('s', String(currentStep()))
  if (state.pinned) h.set('pin', `${state.pinned.layer},${state.pinned.pos}`)
  history.replaceState(null, '', `#${h.toString()}`)
}

function readHash(): void {
  const h = new URLSearchParams(location.hash.slice(1))
  if (h.has('m') && catalog!.models.some((m) => m.id === h.get('m'))) state.model = h.get('m')
  if (h.has('p')) {
    const pid = Number(h.get('p'))
    state.promptId = index!.prompts.some((p) => p.id === pid) ? pid : 0
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
    state.liveText = h.get('q')!
    $<HTMLInputElement>('live-input').value = state.liveText
  } else if (h.has('pin')) {
    const [l, p] = h.get('pin')!.split(',').map(Number)
    // upper bounds are re-validated in refreshGrid once the shard's dimensions are known
    if (Number.isInteger(l) && Number.isInteger(p) && l >= 0 && p >= 0) state.pinned = { layer: l, pos: p }
  }
}

/* ---------- model & prompt selection ---------- */

function rebuildPromptSelect(): void {
  const sel = $<HTMLSelectElement>('prompt-select')
  sel.replaceChildren()
  for (const p of index!.prompts) {
    const opt = document.createElement('option')
    opt.value = String(p.id)
    opt.textContent = p.text.replaceAll('\n', '⏎')
    sel.appendChild(opt)
  }
  sel.disabled = !index!.prompts.length
  if (index!.prompts.length) sel.value = String(state.promptId)
}

async function switchModel(id: string): Promise<void> {
  const gen = ++viewGen
  clearTimeout(liveDebounce)
  const prev = state.model
  let nextIndex: ModelIndex
  try {
    nextIndex = await modelIndex(id)
  } catch (e) {
    status(`model ${id} unavailable: ${(e as Error).message}`)
    $<HTMLSelectElement>('model-select').value = prev!
    return
  }
  if (gen !== viewGen) return // a newer switch/interaction superseded this one
  state.model = id
  index = nextIndex
  state.steps = index.steps
  state.mode = index.prompts.length ? 'pre' : 'live'
  state.promptId = Math.max(0, Math.min(state.promptId, index.prompts.length - 1))
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
  if (!index.prompts.length) {
    // server-registered model with no precomputed shards: live-only view
    grid.setData(null, [], null)
    status(`${modelInfo(id)?.label ?? id} is live-only — type a prompt below and hit Live probe`)
    $<HTMLInputElement>('live-input').focus()
  }
  await refreshGrid()
  await refreshTrajectory()
}

/** index.json for shipped models; a synthetic prompt-less index for server-registered ones. */
function modelIndex(id: string): Promise<ModelIndex> {
  const entry = modelInfo(id)
  if (entry?.serverOnly) return Promise.resolve({ steps: entry.steps ?? [0], prompts: [] })
  return loadIndex(id)
}

/** Add probe-server registry entries that the static catalog does not ship. */
function addServerEntries(serverModels: ServerModel[] | null): void {
  for (const sm of serverModels ?? []) {
    // steps can be empty when a registered local directory vanished; such a model cannot be
    // probed, so keep it out of the picker (it stays visible in the models dialog for removal)
    if (!sm.steps?.length) continue
    if (!catalog!.models.some((m) => m.id === sm.id)) {
      catalog!.models.push({ id: sm.id, hf: sm.id, label: sm.label, serverOnly: true, steps: sm.steps })
    }
  }
}

function rebuildModelSelect(): void {
  const msel = $<HTMLSelectElement>('model-select')
  msel.replaceChildren()
  for (const m of catalog!.models) {
    const opt = document.createElement('option')
    opt.value = m.id
    opt.textContent = m.serverOnly ? `${m.label ?? m.id} (server)` : (m.label ?? m.hf)
    msel.appendChild(opt)
  }
  msel.value = state.model!
}

/** Called by the models dialog after every add/remove with the fresh server registry. */
function mergeServerModels(serverModels: ServerModel[] | null): void {
  if (!serverModels) return
  catalog!.models = catalog!.models.filter((m) => !m.serverOnly || serverModels.some((s) => s.id === m.id))
  addServerEntries(serverModels)
  if (!catalog!.models.some((m) => m.id === state.model)) {
    // the current model was unregistered under us; fall back to the shipped default
    state.model = catalog!.default ?? catalog!.models[0].id
    rebuildModelSelect()
    switchModel(state.model)
    return
  }
  rebuildModelSelect()
}

/* ---------- boot ---------- */

async function boot(): Promise<void> {
  status('loading precomputed data…')
  try {
    catalog = await loadModels()
    addServerEntries(await fetchServerModels()) // server-registered models join the picker
    state.model = catalog.default ?? catalog.models[0].id
    // hash may override the model; read it before loading the index
    const h = new URLSearchParams(location.hash.slice(1))
    if (h.has('m') && catalog.models.some((m) => m.id === h.get('m'))) state.model = h.get('m')
    index = await modelIndex(state.model!)
  } catch (e) {
    status(`failed to load precomputed data: ${(e as Error).message}`)
    return
  }
  state.steps = index.steps
  if (!index.prompts.length) state.mode = 'live'

  rebuildModelSelect()
  const msel = $<HTMLSelectElement>('model-select')
  msel.addEventListener('change', () => switchModel(msel.value))
  setupManageModels(mergeServerModels)

  const sel = $<HTMLSelectElement>('prompt-select')
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
  const slider = $<HTMLInputElement>('step-slider')
  slider.addEventListener('input', () => {
    state.stepIdx = Number(slider.value)
    refreshStepUI()
    onStepChanged()
  })
  setupSliderRange()
  refreshStepUI()
  getEngine(state.model!) // engine init is non-blocking for first paint

  $('grid-legend').innerHTML =
    `<span>lens top-1 probability</span><span class="swatch" style="background:${legendGradient()}"></span><span>0 → 1</span>`

  buildGallery($('gallery-cards'), index.prompts, async (p, card) => {
    // the story cards narrate the 70M model's training run; switch to it if needed
    if (state.model !== 'pythia-70m' && catalog!.models.some((m) => m.id === 'pythia-70m')) {
      $<HTMLSelectElement>('model-select').value = 'pythia-70m'
      await switchModel('pythia-70m')
    }
    viewGen++
    clearTimeout(liveDebounce)
    state.mode = 'pre'
    state.promptId = p.id
    $<HTMLSelectElement>('prompt-select').value = String(p.id)
    const idx = state.steps.indexOf(card.step)
    state.stepIdx = idx >= 0 ? idx : state.steps.length - 1
    $<HTMLInputElement>('step-slider').value = String(state.stepIdx)
    state.pinned = card.pin === 'lastLayerLastPos' ? { layer: gridLayers() - 1, pos: p.tokens.length - 1 } : null
    grid.pinned = state.pinned ? { ...state.pinned } : null
    refreshBadgeAndTicks()
    refreshStepUI()
    syncHash()
    refreshGrid()
    refreshTrajectory()
  })

  $('live-btn').addEventListener('click', () => runLiveProbe($<HTMLInputElement>('live-input').value))
  $('live-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !$<HTMLButtonElement>('live-btn').disabled) runLiveProbe($<HTMLInputElement>('live-input').value)
  })

  $('permalink-btn').addEventListener('click', async () => {
    await navigator.clipboard.writeText(location.href)
    status('permalink copied to clipboard')
  })

  const exportView = () => ({
    grid,
    trajSvg: state.pinned && state.mode === 'pre' ? $<SVGSVGElement>('traj-svg') : null,
    meta: {
      model: modelInfo()?.label ?? state.model!,
      prompt: state.mode === 'live' ? state.liveText : currentPrompt().text,
      step: state.mode === 'live' ? (state.liveResult?.step ?? currentStep()) : currentStep(),
      pinned: state.pinned,
      permalink: location.href,
    },
  })
  const runExport = async (fn: 'png' | 'pdf', label: string) => {
    $('export-menu').removeAttribute('open')
    try {
      status(`exporting ${label}…`)
      const { exportPng, exportPdf } = await import('./export')
      await (fn === 'png' ? exportPng : exportPdf)(exportView())
      status(`${label} exported`)
    } catch (e) {
      status(`export failed: ${(e as Error).message}`)
    }
  }
  $('export-png').addEventListener('click', () => runExport('png', 'PNG figure'))
  $('export-pdf').addEventListener('click', () => runExport('pdf', 'PDF figure'))

  try {
    await refreshGrid()
    await refreshTrajectory()
    status(index.prompts.length ? '' : 'this model is live-only — type a prompt below and hit Live probe')
  } catch (e) {
    status(`failed to render precomputed data: ${(e as Error).message}`)
  }

  // deep-linked live probe
  if (state.liveText) {
    const ok = await enginesReady.get(state.model!)
    if (ok) runLiveProbe(state.liveText)
  }
}

function gridLayers(): number {
  return grid.grid?.layers ?? 7
}

declare global {
  interface Window {
    __lenslapse: {
      state: typeof state
      engines: Map<string, LiveEngine>
      grid: LensGrid
      getEngine: typeof getEngine
      enginesReady: Map<string, Promise<boolean>>
    }
  }
}

// debug/measurement hook (used by the benchmark and fidelity harnesses)
window.__lenslapse = { state, engines, grid, getEngine, enginesReady }

boot()
