import './style.css'
import {
  acquisitionMap,
  diffMap,
  fmtStep,
  gridFromShard,
  layerProfileFromShard,
  loadIndex,
  loadModels,
  loadShard,
  nearestStep,
  trajectoryFromShard,
} from './data'
import { LensGrid } from './grid'
import { LiveEngine, fetchServerModels, probeServerOrigin } from './live'
import { setupManageModels } from './manage'
import { renderRace } from './race'
import { firstContentToken, probeCliCommand, probeCurlCommand } from './snippets'
import { assignSeriesColors, renderLayerProfile, renderTrajectory } from './traj'
import {
  EXAMPLE_TEXTS,
  STORY_CARDS,
  buildGallery,
  buildSliderTicks,
  hideTooltip,
  setBadge,
  showAcqTooltip,
  showDiffTooltip,
  showTooltip,
} from './ui'
import { legendGradient } from './color'
import type { DiffMap, ModelCatalog, ModelEntry, ModelIndex, Prompt, TopEntry, TrajectorySeries } from './data'
import type { PinnedCell } from './grid'
import type { ProbeResult, ServerModel } from './live'
import type { StoryCard } from './ui'

const $ = <T extends Element = HTMLElement>(id: string) => document.getElementById(id) as unknown as T

type LiveResult = ProbeResult & { text: string; step: number }

const state = {
  model: null as string | null, // model id from models.json
  mode: 'pre' as 'pre' | 'live', // 'pre' (precomputed prompt) | 'live' (free-text probe)
  promptId: 0,
  liveText: '',
  stepIdx: 0,
  pinned: null as PinnedCell | null, // {layer, pos}
  gridView: 'top1' as 'top1' | 'acq' | 'diff', // acq = acquisition map; diff = change vs a reference step
  diffRef: null as number | null, // the frozen reference checkpoint for the diff view
  logColor: false, // color cells by log10(p) — reveals early-training structure
  compareId: null as string | null, // second model rendered in lockstep under the main grid
  extraTargets: [] as { id: number; token: string }[], // user-tracked tokens added to traces
  steps: [] as number[],
  liveResult: null as LiveResult | null, // cached last live probe {text, step, grid, tokens}
  // trajectory sweep over every checkpoint for the current live prompt
  liveSweep: null as null | {
    text: string
    pos: number // the position whose final-layer top-3 fixed the targets
    targets: { id: number; token: string }[]
    byStep: Map<number, ProbeResult>
  },
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
    if (!cellInfo) {
      hideTooltip($('tooltip'))
      return
    }
    if (state.mode === 'pre' && state.gridView === 'acq' && acqView) {
      const firstStep = acqView.steps[acqView.firstIdx[cellInfo.layer]?.[cellInfo.pos] ?? acqView.steps.length - 1]
      showAcqTooltip($('tooltip'), cellInfo, evt!, gridTokens(), firstStep)
    } else if (state.mode === 'pre' && state.gridView === 'diff' && diffView && state.diffRef !== null) {
      showDiffTooltip($('tooltip'), cellInfo, evt!, gridTokens(), {
        refStep: state.diffRef,
        curStep: currentStep(),
        ref: diffView.refTop[cellInfo.layer][cellInfo.pos],
        cur: diffView.curTop[cellInfo.layer][cellInfo.pos],
        change: diffView.change[cellInfo.layer][cellInfo.pos],
      })
    } else showTooltip($('tooltip'), cellInfo, evt!, gridTokens())
  },
  onPin(pinned) {
    state.pinned = pinned ? { layer: pinned.layer, pos: pinned.pos } : null
    syncHash()
    refreshTrajectory()
    updateRace()
  },
})

// change-flash bookkeeping: the previous step's top-1 ids, valid only for the same (model, prompt)
let prevTop1: { key: string; ids: number[][] } | null = null
let prevTop1B: { key: string; ids: number[][] } | null = null // same, for the compared model
// data backing the acquisition-map tooltips
let acqView: { steps: number[]; firstIdx: number[][] } | null = null
// data backing the diff-view tooltips
let diffView: DiffMap | null = null

// second grid for compare mode: hover works, pinning stays on the primary grid
const compareGrid = new LensGrid($<HTMLCanvasElement>('compare-canvas'), {
  onHover(cellInfo, evt) {
    if (cellInfo) showTooltip($('tooltip'), cellInfo, evt!, compareGrid.tokens)
    else hideTooltip($('tooltip'))
  },
  onPin() {
    compareGrid.pinned = null
    compareGrid.render()
  },
})

// a function call, not an inline comparison: TS control-flow narrowing would otherwise pin
// state.gridView to 'top1' across the awaits below and reject the re-check
const compareAllowed = (): boolean => state.mode === 'pre' && state.gridView === 'top1'

/** Render the compared model's grid at its checkpoint nearest to the current step. */
async function refreshCompare(): Promise<void> {
  const wrap = $('compare-wrap')
  if (!state.compareId || !compareAllowed()) {
    wrap.hidden = true
    return
  }
  const gen = viewGen
  const cmpId = state.compareId
  try {
    const cmpIndex = await loadIndex(cmpId)
    if (gen !== viewGen || cmpId !== state.compareId) return
    const prompt = cmpIndex.prompts.find((p) => p.text === currentPrompt().text)
    if (!prompt) {
      wrap.hidden = true
      status(`${modelInfo(cmpId)?.label ?? cmpId} has no precomputed data for this prompt — compare is off`)
      return
    }
    const step = nearestStep(cmpIndex.steps, currentStep())
    const shard = await loadShard(cmpId, prompt.id)
    if (gen !== viewGen || cmpId !== state.compareId) return
    // the acq toggle does not bump viewGen — re-check the entry conditions after every await,
    // or a slow first shard load could un-hide compare underneath the acquisition map
    if (!compareAllowed()) return
    const g = gridFromShard(shard, step)
    if (!g) return
    wrap.hidden = false
    $('compare-title').textContent = modelInfo(cmpId)?.label ?? cmpId
    $('compare-step').textContent = `step ${step.toLocaleString()} (nearest checkpoint)`
    compareGrid.logScale = state.logColor
    const key = `${cmpId}:${prompt.id}`
    const ids = g.cells.map((row) => row.map((c) => c.top[0][2]))
    compareGrid.setData(g, prompt.tokens, null)
    if (prevTop1B && prevTop1B.key === key && prevTop1B.ids.length === ids.length) {
      const changed = new Set<string>()
      for (let li = 0; li < ids.length; li++)
        for (let t = 0; t < ids[li].length; t++)
          if (prevTop1B.ids[li]?.[t] !== undefined && prevTop1B.ids[li][t] !== ids[li][t]) changed.add(`${li}:${t}`)
      compareGrid.flashCells(changed)
    }
    prevTop1B = { key, ids }
  } catch (e) {
    if (gen === viewGen) {
      wrap.hidden = true
      status(`compare failed: ${(e as Error).message}`)
    }
  }
}

function gridTokens(): string[] {
  return state.mode === 'live' && state.liveResult ? state.liveResult.tokens : (currentPrompt()?.tokens ?? [])
}

async function refreshGrid(): Promise<void> {
  if (state.mode === 'live') {
    $('compare-wrap').hidden = true // compare is precomputed-only — hide even before the first probe lands
    if (!state.liveResult) return
    grid.logScale = state.logColor
    grid.setData(state.liveResult.grid, state.liveResult.tokens, state.pinned)
    updateRace()
    return
  }
  const gen = viewGen
  const shard = await loadShard(state.model!, state.promptId)
  if (gen !== viewGen) return
  if (state.gridView === 'diff' && state.diffRef !== null) {
    const d = diffMap(shard, state.diffRef, currentStep())
    if (!d) return
    diffView = d
    const cells = d.curTop.map((row, li) =>
      row.map((cur, t) => ({ token: cur[0], prob: d.change[li][t], top: [cur] }))
    )
    if (state.pinned && (state.pinned.layer >= d.layers || state.pinned.pos >= d.positions)) {
      state.pinned = null
      grid.pinned = null
    }
    grid.logScale = false // the color IS the change fraction, not a probability
    grid.setData({ layers: d.layers, positions: d.positions, cells }, currentPrompt().tokens, state.pinned)
    const flippedSet = new Set<string>()
    d.flipped.forEach((row, li) => row.forEach((f, t) => f && flippedSet.add(`${li}:${t}`)))
    grid.setMarks(flippedSet)
    $('compare-wrap').hidden = true
    updateRace()
    return
  }
  if (state.gridView === 'acq') {
    const acq = acquisitionMap(shard, index!.steps)
    if (!acq) return
    acqView = { steps: index!.steps, firstIdx: acq.firstIdx }
    const span = Math.max(1, index!.steps.length - 1)
    const cells = acq.firstIdx.map((row, li) =>
      row.map((si, t) => ({
        token: fmtStep(index!.steps[si]),
        prob: si / span, // color = when the final answer arrived (early → dark, late → bright)
        top: [acq.finalTop[li][t]],
      }))
    )
    if (state.pinned && (state.pinned.layer >= acq.layers || state.pinned.pos >= acq.positions)) {
      state.pinned = null
      grid.pinned = null
    }
    grid.logScale = false // ordinal acquisition colors, not probabilities
    grid.setData({ layers: acq.layers, positions: acq.positions, cells }, currentPrompt().tokens, state.pinned)
    $('compare-wrap').hidden = true
    updateRace()
    return
  }
  const g = gridFromShard(shard, currentStep())
  if (!g) return
  if (state.pinned && (state.pinned.layer >= g.layers || state.pinned.pos >= g.positions)) {
    state.pinned = null
    grid.pinned = null
  }
  const key = `${state.model}:${state.promptId}`
  const ids = g.cells.map((row) => row.map((c) => c.top[0][2]))
  grid.logScale = state.logColor
  grid.setData(g, currentPrompt().tokens, state.pinned)
  refreshCompare()
  if (prevTop1 && prevTop1.key === key && prevTop1.ids.length === ids.length) {
    const changed = new Set<string>()
    for (let li = 0; li < ids.length; li++)
      for (let t = 0; t < ids[li].length; t++)
        if (prevTop1.ids[li]?.[t] !== undefined && prevTop1.ids[li][t] !== ids[li][t]) changed.add(`${li}:${t}`)
    grid.flashCells(changed)
  }
  prevTop1 = { key, ids }
  updateRace()
}

/** The pinned cell's top-10 as an animated leaderboard (below the trajectory chart). */
async function updateRace(): Promise<void> {
  const bars = $('race-bars')
  const cap = $('race-caption')
  const pin = state.pinned
  let top: TopEntry[] | null = null
  let goldId: number | undefined
  if (pin && state.mode === 'live') {
    top = state.liveResult?.grid.cells[pin.layer]?.[pin.pos]?.top ?? null
  } else if (pin && index!.prompts.length) {
    const gen = viewGen
    const shard = await loadShard(state.model!, state.promptId) // promise-cached: instant after first load
    if (gen !== viewGen) return
    top = gridFromShard(shard, currentStep())?.cells[pin.layer]?.[pin.pos]?.top ?? null
    const p = currentPrompt()
    if (pin.pos === p.tokens.length - 1) goldId = p.gold_id
  }
  if (!top) {
    bars.hidden = true
    cap.hidden = true
    bars.replaceChildren()
    return
  }
  cap.hidden = false
  cap.textContent = `top-10 race at the pinned cell — step ${
    state.mode === 'live' ? (state.liveResult?.step ?? currentStep()).toLocaleString() : currentStep().toLocaleString()
  }`
  renderRace(bars, top, { goldId, limit: 10 })
}

function hideLayerProfile(): void {
  $('layer-caption').hidden = true
  const svg = $<SVGSVGElement>('layer-svg')
  svg.style.display = 'none'
  svg.replaceChildren()
}

function showLayerProfile(
  series: TrajectorySeries[],
  layers: number,
  pinnedLayer: number,
  opts: { goldId?: number; colors?: Map<number, string> },
  stepLabel: number = currentStep()
): void {
  if (!series.length || layers < 2) {
    hideLayerProfile()
    return
  }
  $('layer-caption').hidden = false
  $('layer-caption').textContent = `layer profile — step ${stepLabel.toLocaleString()}`
  const svg = $<SVGSVGElement>('layer-svg')
  svg.style.display = 'block'
  renderLayerProfile(svg, series, layers, pinnedLayer, opts)
}

async function refreshTrajectory(): Promise<void> {
  const svg = $<SVGSVGElement>('traj-svg')
  const sub = $('traj-subtitle')
  const sweepBtn = $<HTMLButtonElement>('traj-sweep')
  const eng = engines.get(state.model!)
  // token tracking rides on the live-trace machinery: offer it whenever a trace is possible
  $('track-row').hidden = !eng?.available || sweepableSteps().length < 2
  if (state.mode === 'live') {
    const canSweep = !!state.liveText && !!eng?.available && sweepableSteps().length > 1
    sweepBtn.hidden = !canSweep
    const sweep = state.liveSweep
    if (sweep && sweep.text === state.liveText && sweep.byStep.size && state.pinned && state.pinned.pos === sweep.pos) {
      const steps = [...sweep.byStep.keys()].sort((a, b) => a - b)
      const { layer, pos } = state.pinned
      const series = sweep.targets
        .map(({ id, token }) => ({
          id,
          token,
          points: steps.flatMap((st) => {
            const t = sweep.byStep.get(st)!.tgt?.[String(id)]
            return t?.p[layer]?.[pos] !== undefined ? [[st, t.p[layer][pos], t.r[layer][pos]] as [number, number, number]] : []
          }),
        }))
        .filter((sr) => sr.points.length)
        .sort((a, b) => b.points[b.points.length - 1][1] - a.points[a.points.length - 1][1])
      sub.textContent = `${state.pinned.layer === 0 ? 'embedding' : `layer ${state.pinned.layer}`}, position ${pos} across ${steps.length} live-probed checkpoints`
      const colors = assignSeriesColors(series)
      renderTrajectory(svg, series, steps, currentStep(), { colors })
      // in-browser sweeps only cover live-capable checkpoints — read the nearest probed one
      const curKey = sweep.byStep.has(currentStep()) ? currentStep() : nearestStep(steps, currentStep())
      const cur = sweep.byStep.get(curKey)
      const profile = cur?.tgt
        ? sweep.targets
            .map(({ id, token }) => {
              const t = cur.tgt![String(id)]
              return t
                ? { id, token, points: t.p.map((row, li) => [li, row[pos], t.r[li][pos]] as [number, number, number]) }
                : null
            })
            .filter((s): s is NonNullable<typeof s> => s !== null)
        : []
      showLayerProfile(profile, cur?.grid.layers ?? 0, layer, { colors }, curKey)
      return
    }
    svg.replaceChildren()
    hideLayerProfile()
    sub.textContent = canSweep
      ? state.pinned
        ? sweep && sweep.text === state.liveText && state.pinned.pos !== sweep.pos
          ? 'pinned a different position — trace again to track its tokens'
          : 'press “trace across training” to probe every checkpoint'
        : 'click a cell in the grid, then trace it across training'
      : 'trajectories are available for curated (precomputed) prompts'
    return
  }
  sweepBtn.hidden = true
  if (!state.pinned) {
    svg.replaceChildren()
    hideLayerProfile()
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
  const goldId = pos === prompt.tokens.length - 1 ? prompt.gold_id : undefined
  const colors = assignSeriesColors(series)
  renderTrajectory(svg, series, index!.steps, currentStep(), { goldId, colors })
  const profile = layerProfileFromShard(shard, prompt, pos, currentStep())
  showLayerProfile(profile, profile[0]?.points.length ?? 0, layer, { goldId, colors })
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

/* ---------- time-lapse playback ---------- */

const PLAY_MS = 220 // per checkpoint: 38 steps ≈ 8 s — slow enough to read, fast enough to feel like a time-lapse
let playTimer: number | undefined

function stopPlay(): void {
  if (playTimer === undefined) return
  window.clearInterval(playTimer)
  playTimer = undefined
  const b = $<HTMLButtonElement>('play-btn')
  b.textContent = '▶'
  b.setAttribute('aria-pressed', 'false')
  syncHash() // hash writes are skipped during playback (Safari rate-limits replaceState) — record where we stopped
}

function startPlay(): void {
  if (playTimer !== undefined || state.mode !== 'pre' || state.steps.length < 2) return
  if (state.gridView === 'acq') setGridView('top1') // the map aggregates time; playing means watching steps
  if (state.stepIdx >= state.steps.length - 1) setStepIdx(0)
  const b = $<HTMLButtonElement>('play-btn')
  b.textContent = '⏸'
  b.setAttribute('aria-pressed', 'true')
  playTimer = window.setInterval(() => {
    if (state.stepIdx >= state.steps.length - 1) {
      stopPlay()
      if (KIOSK) window.setTimeout(kioskNext, 1800) // booth mode: cycle to the next story
      return
    }
    setStepIdx(state.stepIdx + 1)
  }, PLAY_MS)
}

function setStepIdx(idx: number): void {
  state.stepIdx = idx
  $<HTMLInputElement>('step-slider').value = String(idx)
  refreshStepUI()
  onStepChanged()
}

function refreshPlayControls(): void {
  const pre = state.mode === 'pre' && !!index?.prompts.length
  $<HTMLButtonElement>('play-btn').disabled = state.mode !== 'pre' || state.steps.length < 2
  // the map is meaningless for single-checkpoint models (every cell would read "step 0")
  $<HTMLButtonElement>('acq-toggle').hidden = !pre || state.steps.length < 2
  $<HTMLButtonElement>('log-toggle').hidden = !pre || state.gridView !== 'top1'
  $<HTMLSelectElement>('compare-select').hidden = !pre || state.gridView !== 'top1'
  $<HTMLButtonElement>('diff-toggle').hidden = !pre || state.steps.length < 2
  $<HTMLButtonElement>('dice-btn').hidden = !index?.prompts.length
}

/** Serendipity, Neuronpedia-style: jump to a random (prompt, step, cell). */
function randomView(): void {
  if (!index?.prompts.length) return
  stopPlay()
  if (state.gridView !== 'top1') setGridView('top1')
  viewGen++
  clearTimeout(liveDebounce)
  state.mode = 'pre'
  const prompt = index.prompts[Math.floor(Math.random() * index.prompts.length)]
  state.promptId = prompt.id
  state.extraTargets = []
  $<HTMLSelectElement>('prompt-select').value = String(prompt.id)
  state.stepIdx = Math.floor(Math.random() * state.steps.length)
  $<HTMLInputElement>('step-slider').value = String(state.stepIdx)
  state.pinned = { layer: Math.max(1, Math.floor(Math.random() * gridLayers())), pos: prompt.tokens.length - 1 }
  grid.pinned = { ...state.pinned }
  refreshBadgeAndTicks()
  refreshStepUI()
  syncHash()
  refreshGrid()
  refreshTrajectory()
  updateRace()
}

function rebuildCompareSelect(): void {
  const sel = $<HTMLSelectElement>('compare-select')
  sel.replaceChildren()
  const off = document.createElement('option')
  off.value = ''
  off.textContent = 'compare: off'
  sel.appendChild(off)
  for (const m of catalog!.models) {
    if (m.id === state.model || m.serverOnly) continue
    const opt = document.createElement('option')
    opt.value = m.id
    opt.textContent = `vs ${m.label ?? m.id}`
    sel.appendChild(opt)
  }
  sel.value = state.compareId ?? ''
}

function setGridView(view: 'top1' | 'acq' | 'diff'): void {
  state.gridView = view
  if (view !== 'diff') state.diffRef = null
  const acqBtn = $<HTMLButtonElement>('acq-toggle')
  acqBtn.classList.toggle('active', view === 'acq')
  acqBtn.textContent = view === 'acq' ? '▦ top-1 view' : '⏱ acquisition map'
  const diffBtn = $<HTMLButtonElement>('diff-toggle')
  diffBtn.classList.toggle('active', view === 'diff')
  diffBtn.textContent =
    view === 'diff' && state.diffRef !== null ? `Δ vs step ${fmtStep(state.diffRef)} ✕` : 'Δ vs this step'
  $('grid-legend').innerHTML =
    view === 'acq'
      ? `<span>final answer first becomes top-1</span><span class="swatch" style="background:${legendGradient()}"></span><span>earliest → latest checkpoint</span>`
      : view === 'diff'
        ? `<span>top-10 turnover vs step ${state.diffRef !== null ? fmtStep(state.diffRef) : '—'} (outline = top-1 flipped)</span><span class="swatch" style="background:${legendGradient()}"></span><span>unchanged → replaced</span>`
        : state.logColor
          ? `<span>lens top-1 probability (log)</span><span class="swatch" style="background:${legendGradient()}"></span><span>10⁻⁶ → 1</span>`
          : `<span>lens top-1 probability</span><span class="swatch" style="background:${legendGradient()}"></span><span>0 → 1</span>`
  prevTop1 = null
  prevTop1B = null
  if (view !== 'acq') acqView = null
  if (view !== 'diff') diffView = null
  refreshPlayControls()
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
  stopPlay()
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
    if (state.gridView !== 'top1') setGridView('top1') // acq/diff need all-step shards, which live mode lacks
    refreshPlayControls()
    if (state.liveSweep && state.liveSweep.text !== text) {
      state.liveSweep = null
      state.extraTargets = [] // tracked tokens belong to the text they were added for
    }
    state.liveText = text
    state.liveResult = { ...res, text, step }
    if (state.pinned && (state.pinned.layer >= res.grid.layers || state.pinned.pos >= res.grid.positions)) {
      state.pinned = null
    }
    setBadge($('backend-badge'), res.backend)
    const where = res.backend === 'server' ? 'on the local probe server' : 'fully in your browser'
    const note = res.replayed ? ' · replayed from saved probe' : res.serverCached ? ' · from server cache' : ''
    status(
      `live probe @ step ${step.toLocaleString()} — forward+lens+top-k ${res.timing.probe.toFixed(0)}ms on ${res.backend} (${where})${note}`
    )
    grid.pinned = state.pinned ? { ...state.pinned } : null
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

/** Steps a trajectory sweep can cover: every suite step via the server, live steps in-browser. */
function sweepableSteps(): number[] {
  const eng = engines.get(state.model!)
  if (!eng?.available) return []
  return eng.server ? state.steps : eng.liveSteps()
}

let sweepSeq = 0
/** Probe the current live prompt at every checkpoint and stream the trajectory in. */
async function runSweep(): Promise<void> {
  const gen = viewGen
  const run = ++sweepSeq
  const engine = getEngine()
  const text = state.liveText
  const steps = sweepableSteps()
  if (!text || steps.length < 2 || !state.liveResult) return
  if (!state.pinned) {
    // default to the classic lens cell: deepest layer, last position
    state.pinned = { layer: state.liveResult.grid.layers - 1, pos: state.liveResult.grid.positions - 1 }
    grid.pinned = { ...state.pinned }
    refreshGrid()
  }
  const pin = state.pinned
  const btn = $<HTMLButtonElement>('traj-sweep')
  btn.disabled = true
  try {
    // fix the tracked tokens from the FINAL checkpoint: final-layer top-3 at the pinned position
    // (the same convention the precomputed shards use)
    const last = steps[steps.length - 1]
    const finalRes = await engine.probe(text, last)
    if (gen !== viewGen || run !== sweepSeq) return
    const finalCell = finalRes.grid.cells[finalRes.grid.layers - 1][Math.min(pin.pos, finalRes.grid.positions - 1)]
    const targets = finalCell.top.slice(0, 3).map(([token, , id]) => ({ id, token }))
    for (const ex of state.extraTargets) if (!targets.some((t) => t.id === ex.id)) targets.push(ex)
    targets.splice(5) // the chart palette holds five series
    state.liveSweep = { text, pos: pin.pos, targets, byStep: new Map() }
    const ids = targets.map((t) => t.id)
    for (let i = 0; i < steps.length; i++) {
      if (gen !== viewGen || run !== sweepSeq) return
      status(`tracing “${text.slice(0, 40)}” — step ${steps[i].toLocaleString()} (${i + 1}/${steps.length})…`)
      const res = await engine.probe(text, steps[i], undefined, ids)
      if (gen !== viewGen || run !== sweepSeq) return
      state.liveSweep.byStep.set(steps[i], res)
      refreshTrajectory()
    }
    status(`traced ${steps.length} checkpoints — scrub the slider to replay any of them instantly`)
  } catch (e) {
    if (gen === viewGen) status(`trace failed: ${(e as Error).message}`)
  } finally {
    if (run === sweepSeq) btn.disabled = false
  }
}

/** Jump to a curated story on the CURRENTLY selected model: precomputed when its shards carry
 * the prompt, a live probe otherwise — at the model's nearest available step. */
async function applyStory(card: StoryCard): Promise<void> {
  stopPlay()
  if (state.gridView !== 'top1') setGridView('top1') // stories are step-specific top-1 views
  state.extraTargets = [] // a story defines its own view; tracked tokens belong to the old one
  const nearest = nearestStep(state.steps, card.step)
  state.stepIdx = state.steps.indexOf(nearest)
  $<HTMLInputElement>('step-slider').value = String(state.stepIdx)
  const p = index!.prompts.find((pr) => pr.text === card.text)
  if (!p) {
    clearTimeout(liveDebounce)
    $<HTMLInputElement>('live-input').value = card.text
    refreshStepUI()
    await runLiveProbe(card.text)
    if (card.pin === 'lastLayerLastPos' && state.liveResult?.text === card.text) {
      state.pinned = { layer: state.liveResult.grid.layers - 1, pos: state.liveResult.grid.positions - 1 }
      grid.pinned = { ...state.pinned }
      refreshGrid()
      refreshTrajectory()
    }
    return
  }
  viewGen++
  clearTimeout(liveDebounce)
  state.mode = 'pre'
  state.promptId = p.id
  $<HTMLSelectElement>('prompt-select').value = String(p.id)
  state.pinned = card.pin === 'lastLayerLastPos' ? { layer: gridLayers() - 1, pos: p.tokens.length - 1 } : null
  grid.pinned = state.pinned ? { ...state.pinned } : null
  refreshBadgeAndTicks()
  refreshStepUI()
  syncHash()
  refreshGrid()
  refreshTrajectory()
}

/** Track an arbitrary token: tokenize it, add it to the trace targets, and (re)run the sweep.
 * Works from a precomputed view too — the trace itself always runs live. */
async function trackToken(raw: string): Promise<void> {
  if (!raw) return
  const engine = getEngine()
  await enginesReady.get(engine.modelId)
  if (!engine.available) {
    status('tracking a token needs live probing, which is unavailable for this model right now')
    return
  }
  const promptText = index?.prompts.length ? currentPrompt().text : null
  const text = state.mode === 'live' && state.liveText ? state.liveText : promptText
  if (!text) {
    status('probe a prompt first, then track tokens on it')
    return
  }
  let ids: number[]
  let tokens: string[]
  try {
    ;({ ids, tokens } = await engine.tokenize(raw))
  } catch (e) {
    status(`could not tokenize “${raw}”: ${(e as Error).message}`)
    return
  }
  if (!ids.length) return
  const pick = firstContentToken(tokens)
  const id = ids[pick]
  const token = tokens[pick]
  if (ids.length > 1) status(`“${raw}” splits into ${ids.length} tokens — tracking “${token}”`)
  // ensure the live view is on `text` BEFORE registering the target: a text change inside
  // runLiveProbe clears extraTargets, which would silently drop a token added earlier
  if (state.mode !== 'live' || state.liveText !== text) {
    $<HTMLInputElement>('live-input').value = text
    await runLiveProbe(text)
    if (state.mode !== 'live' || state.liveText !== text) return // probe failed or was superseded
  }
  if (!state.extraTargets.some((t) => t.id === id)) state.extraTargets.push({ id, token })
  await runSweep()
}

/* ---------- kiosk mode (?kiosk): loop the time-lapse through the emergence stories ---------- */

const KIOSK = new URLSearchParams(location.search).has('kiosk')
// ?embed: chrome-less view for iframes (header, prompt bar, and gallery hidden via CSS)
const EMBED = new URLSearchParams(location.search).has('embed')
if (EMBED) document.body.classList.add('embed')
let kioskIdx = 0

function kioskNext(): void {
  const card = STORY_CARDS[kioskIdx % STORY_CARDS.length]
  kioskIdx++
  void applyStory(card).then(() => {
    if (state.mode !== 'pre' || state.steps.length < 2) return
    setStepIdx(0)
    startPlay()
  })
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
  if (playTimer !== undefined) return // ~5 writes/s during playback would trip Safari's replaceState rate limit
  const h = new URLSearchParams()
  h.set('m', state.model!)
  if (state.mode === 'live' && state.liveText) h.set('q', state.liveText)
  else if (index!.prompts.length) h.set('p', String(state.promptId))
  h.set('s', String(currentStep()))
  if (state.pinned) h.set('pin', `${state.pinned.layer},${state.pinned.pos}`)
  if (state.compareId) h.set('cmp', state.compareId)
  try {
    history.replaceState(null, '', `#${h.toString()}`)
  } catch {
    /* a rate-limited replaceState must never break rendering */
  }
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
  const cmp = h.get('cmp')
  if (cmp && cmp !== state.model && catalog!.models.some((m) => m.id === cmp && !m.serverOnly)) {
    state.compareId = cmp
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
  if (index!.prompts.length) {
    for (const p of index!.prompts) {
      const opt = document.createElement('option')
      opt.value = String(p.id)
      opt.textContent = p.text.replaceAll('\n', '⏎')
      sel.appendChild(opt)
    }
    sel.disabled = false
    sel.value = String(state.promptId)
    return
  }
  // live-only model: offer the curated examples as one-click live probes on THIS model
  const hint = document.createElement('option')
  hint.value = ''
  hint.textContent = 'example prompts (probed live on this model)…'
  sel.appendChild(hint)
  EXAMPLE_TEXTS.forEach((text, i) => {
    const opt = document.createElement('option')
    opt.value = `ex:${i}`
    opt.textContent = text.replaceAll('\n', '⏎')
    sel.appendChild(opt)
  })
  sel.disabled = false
  sel.value = ''
}

async function switchModel(id: string): Promise<void> {
  const gen = ++viewGen
  stopPlay()
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
  state.liveSweep = null // another model's trajectory must never render as this one's
  state.liveText = ''
  state.extraTargets = [] // tracked tokens are model-specific ids
  if (state.compareId === id) state.compareId = null
  setGridView('top1')
  refreshPlayControls()
  rebuildPromptSelect()
  rebuildCompareSelect()
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
    stopPlay()
    if (sel.value.startsWith('ex:')) {
      // live-only model: the selected example fires a live probe on the current model
      const text = EXAMPLE_TEXTS[Number(sel.value.slice(3))]
      $<HTMLInputElement>('live-input').value = text
      runLiveProbe(text)
      return
    }
    if (sel.value === '') return
    viewGen++
    clearTimeout(liveDebounce)
    state.mode = 'pre'
    state.promptId = Number(sel.value)
    state.extraTargets = []
    state.pinned = null
    grid.pinned = null
    refreshBadgeAndTicks()
    status('')
    syncHash()
    refreshGrid()
    refreshTrajectory()
  })

  const deepLinked = location.hash.length > 1
  readHash()
  rebuildPromptSelect()
  const slider = $<HTMLInputElement>('step-slider')
  slider.addEventListener('input', () => {
    stopPlay()
    state.stepIdx = Number(slider.value)
    refreshStepUI()
    onStepChanged()
  })
  setupSliderRange()
  refreshStepUI()
  getEngine(state.model!) // engine init is non-blocking for first paint

  setGridView('top1')
  refreshPlayControls()
  rebuildCompareSelect()
  $('play-btn').addEventListener('click', () => (playTimer === undefined ? startPlay() : stopPlay()))
  $('acq-toggle').addEventListener('click', () => {
    if (state.mode !== 'pre') return
    stopPlay()
    setGridView(state.gridView === 'acq' ? 'top1' : 'acq')
    refreshGrid()
  })
  $('diff-toggle').addEventListener('click', () => {
    if (state.mode !== 'pre') return
    if (state.gridView === 'diff') {
      setGridView('top1')
    } else {
      stopPlay() // freeze THIS checkpoint as the reference; scrubbing/playing then shows change vs it
      state.diffRef = currentStep()
      setGridView('diff')
      status(`diff mode: comparing against step ${state.diffRef.toLocaleString()} — scrub or play to see what changes`)
    }
    refreshGrid()
  })
  $('log-toggle').addEventListener('click', () => {
    state.logColor = !state.logColor
    $('log-toggle').classList.toggle('active', state.logColor)
    setGridView(state.gridView) // refreshes the legend text for the new scale
    refreshGrid()
  })
  const cmpSel = $<HTMLSelectElement>('compare-select')
  cmpSel.addEventListener('change', () => {
    state.compareId = cmpSel.value || null
    prevTop1B = null
    syncHash()
    refreshCompare()
  })
  $('dice-btn').addEventListener('click', () => randomView())
  $('track-btn').addEventListener('click', () => trackToken($<HTMLInputElement>('track-input').value))
  $('track-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') trackToken($<HTMLInputElement>('track-input').value)
  })
  if (EMBED) {
    const a = $<HTMLAnchorElement>('embed-link')
    a.hidden = false
    // resolve at activation time so the link opens the full app on the exact embedded view
    // (click covers keyboard activation too; handlers run before the default navigation)
    a.addEventListener('click', () => a.setAttribute('href', `./${location.hash}`))
  }
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' || e.repeat) return
    const t = e.target as HTMLElement | null
    // never steal Space from form fields or from a focused control's native activation
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return
    if (t?.closest('button, summary, a, [role="button"]')) return
    if ($<HTMLDialogElement>('models-dialog').open) return
    e.preventDefault()
    if (playTimer === undefined) startPlay()
    else stopPlay()
  })

  buildGallery($('gallery-cards'), applyStory)

  $('live-btn').addEventListener('click', () => runLiveProbe($<HTMLInputElement>('live-input').value))
  $('traj-sweep').addEventListener('click', () => runSweep())
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
    stopPlay() // the figure must show the step its metadata names, not wherever playback ran to
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
  const copySnippet = async (kind: 'cli' | 'curl') => {
    $('export-menu').removeAttribute('open')
    const text = state.mode === 'live' ? state.liveText : currentPrompt().text
    const step = state.mode === 'live' ? (state.liveResult?.step ?? currentStep()) : currentStep()
    const origin = probeServerOrigin() ?? 'http://localhost:8017'
    const cmd =
      kind === 'cli' ? probeCliCommand(state.model!, text, step) : probeCurlCommand(origin, state.model!, text, step)
    try {
      await navigator.clipboard.writeText(cmd)
      status(
        kind === 'cli'
          ? 'CLI command copied — reproduces this probe via lenslapse probe'
          : 'cURL request copied — hits the probe-server /probe API'
      )
    } catch {
      status('clipboard unavailable — the command is printed in the console instead')
      console.log(cmd)
    }
  }
  $('copy-cli').addEventListener('click', () => copySnippet('cli'))
  $('copy-curl').addEventListener('click', () => copySnippet('curl'))

  try {
    await refreshGrid()
    await refreshTrajectory()
    status(index.prompts.length ? '' : 'this model is live-only — type a prompt below and hit Live probe')
  } catch (e) {
    status(`failed to render precomputed data: ${(e as Error).message}`)
  }

  // first-load affordances (skipped for permalinks, which encode a specific view): pin the
  // classic cell so the trajectory panel is never an empty box, and play the time-lapse once
  // for first-time visitors — the product IS the motion, so show it without asking
  if (!deepLinked && state.mode === 'pre' && index.prompts.length && grid.grid) {
    state.pinned = { layer: grid.grid.layers - 1, pos: currentPrompt().tokens.length - 1 }
    grid.pinned = { ...state.pinned }
    grid.render()
    await refreshTrajectory()
    updateRace()
    let played = false
    try {
      played = localStorage.getItem('lenslapse-played') === '1'
    } catch {
      /* storage unavailable */
    }
    if ((!played || KIOSK) && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
      try {
        localStorage.setItem('lenslapse-played', '1')
      } catch {
        /* storage unavailable */
      }
      if (KIOSK) kioskNext()
      else startPlay()
    }
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
