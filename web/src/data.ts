// Precomputed-data store: models.json (catalog), per-model index.json + per-prompt shards,
// fetched lazily with promise-caching so concurrent callers share one fetch.

import { signUrl } from './auth'

export interface ModelEntry {
  id: string
  hf: string
  label?: string | null
  mode?: string
  source?: string
  serverOnly?: boolean
  steps?: number[]
  /** Languages this model is documented to handle well, e.g. ["en"] or ["zh", "en"] — drives
   * which curated example prompts are offered (see promptLang in ui.ts) and, for compare mode,
   * whether a missing precomputed prompt is worth falling back to a live probe for. Omitted
   * entirely for models with no language classification on file (custom/registered models). */
  languages?: string[]
  /** Overrides the auto-derived language tag (e.g. "Chinese/English" from `languages`) for
   * models documented as supporting many more languages than just the ones we have curated
   * examples for — BLOOM/Qwen3/Gemma-3 ship English+Chinese examples like any zh+en bilingual
   * model, but are labeled "Multilingual" since their own model cards claim dozens+ languages. */
  languageLabel?: string
}

/** The display tag for a model's language support, e.g. "Chinese/English" or "Multilingual" —
 * null for English-only (and for models with no `languages` on file), matching the existing
 * convention that English needs no tag at all. */
export function languageTag(entry: ModelEntry): string | null {
  if (entry.languageLabel) return entry.languageLabel
  if (!entry.languages || entry.languages.length < 2) return null
  const NAMES: Record<string, string> = { en: 'English', zh: 'Chinese' }
  return entry.languages.map((l) => NAMES[l] ?? l).join('/')
}

/** Best-effort language of one prompt's text: CJK ideographs mean Chinese, anything else is
 * treated as English. The only two languages any curated prompt currently comes in — good
 * enough to group a bilingual/multilingual model's examples without a per-prompt data field. */
export function promptLang(text: string): 'zh' | 'en' {
  return /[一-鿿]/.test(text) ? 'zh' : 'en'
}

/** Display name for one detected prompt language, for a dropdown optgroup label. */
export function promptLangLabel(lang: 'zh' | 'en'): string {
  return lang === 'zh' ? '中文' : 'English'
}

export interface ModelCatalog {
  models: ModelEntry[]
  default?: string
}

export interface Prompt {
  id: number
  text: string
  gold?: string
  gold_id?: number
  story?: string
  ids: number[]
  tokens: string[]
  targets: number[][]
}

export interface ModelIndex {
  model?: string
  steps: number[]
  prompts: Prompt[]
}

/** One shard step entry: top[layer][pos] = [[id, prob], ...]; tgt[id] = per-(layer,pos) prob/rank. */
export interface ShardStep {
  top: [number, number][][][]
  tgt?: Record<string, { p: number[][]; r: number[][] }>
}

export interface Shard {
  vocab: Record<string, string>
  steps: Record<string, ShardStep>
}

export type TopEntry = [string, number, number]

export interface GridCell {
  token: string
  prob: number
  top: TopEntry[]
}

/** Grid shape shared by every per-cell view: the live/precomputed lens grid, the acquisition
 * map, and the diff view all key their per-cell arrays by the same (layer, position) axes. */
export interface GridDims {
  layers: number
  positions: number
}

export interface GridData extends GridDims {
  cells: GridCell[][]
}

/** A token tracked across training: the classic final-checkpoint top-3, the gold continuation,
 * or a token the user typed into "track any token". */
export interface TokenRef {
  id: number
  token: string
}

export interface TrajectorySeries extends TokenRef {
  points: [number, number, number][]
}

/** Which of the grid's cell colors mean "probability" vs something else entirely — the
 * acquisition map colors by checkpoint order, and diff colors by turnover since a reference. */
export type GridView = 'top1' | 'acq' | 'diff'

const BASE = `${import.meta.env.BASE_URL}data/`

const cache = new Map<string, Promise<unknown>>()

function fetchJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    const p = fetch(signUrl(`${BASE}${path}`)).then((res) => {
      if (!res.ok) throw new Error(`${path} missing (${res.status})`)
      return res.json()
    })
    p.catch(() => cache.delete(path))
    cache.set(path, p)
  }
  return cache.get(path) as Promise<T>
}

/** { models: [{id, hf, label}], default: id } */
export function loadModels(): Promise<ModelCatalog> {
  return fetchJson('models.json')
}

export function loadIndex(modelId: string): Promise<ModelIndex> {
  return fetchJson(`${modelId}/index.json`)
}

export function loadShard(modelId: string, promptId: number): Promise<Shard> {
  return fetchJson(`${modelId}/p${promptId}.json`)
}

/**
 * Grid view for one (prompt, step) from a shard.
 * Returns { layers, positions, cells } with cells[layer][pos] = {token, prob, top:[[token,prob,id]...]}.
 */
export function gridFromShard(shard: Shard, step: number): GridData | null {
  const entry = shard.steps[String(step)]
  if (!entry) return null
  const vocab = shard.vocab
  const cells = entry.top.map((layerRow) =>
    layerRow.map((cell) => ({
      token: vocab[String(cell[0][0])] ?? '?',
      prob: cell[0][1],
      top: cell.map(([id, p]): TopEntry => [vocab[String(id)] ?? '?', p, id]),
    }))
  )
  return { layers: cells.length, positions: cells[0].length, cells }
}

/** The step in `steps` closest to `step` (used to keep a compared model in lockstep). */
export function nearestStep(steps: number[], step: number): number {
  return steps.reduce((a, b) => (Math.abs(b - step) < Math.abs(a - step) ? b : a))
}

/** Compact step label for grid cells: 512 → "512", 8000 → "8k", 2738 → "2.7k". */
export function fmtStep(s: number): string {
  if (s < 1000) return String(s)
  const k = s / 1000
  return `${Number.isInteger(k) ? k : Number(k.toFixed(1))}k`
}

/** Allocate a `layers`-long array of empty per-position arrays, ready for `arr[li].push(...)`
 * — the shared shape every per-cell 2D grid below is built from. */
function grid2D<T>(layers: number): T[][] {
  return Array.from({ length: layers }, () => [])
}

export interface AcquisitionMap extends GridDims {
  /** index into `steps` of the first step where the cell's FINAL top-1 is already its top-1 */
  firstIdx: number[][]
  /** the final answer per cell: [token, final prob, id] */
  finalTop: TopEntry[][]
}

/**
 * When did each cell first predict its final answer? For every (layer, pos), the earliest step
 * at which the FINAL checkpoint's top-1 token is already the cell's top-1 (the final step
 * itself trivially qualifies, so the scan always terminates).
 */
export function acquisitionMap(shard: Shard, steps: number[]): AcquisitionMap | null {
  const last = shard.steps[String(steps.at(-1))]
  if (!last) return null
  const layers = last.top.length
  const positions = last.top[0].length
  const vocab = shard.vocab
  const firstIdx = grid2D<number>(layers)
  const finalTop = grid2D<TopEntry>(layers)
  for (let li = 0; li < layers; li++) {
    for (let t = 0; t < positions; t++) {
      const [finalId, finalP] = last.top[li][t][0]
      finalTop[li].push([vocab[String(finalId)] ?? '?', finalP, finalId])
      let idx = steps.length - 1
      for (let si = 0; si < steps.length; si++) {
        const e = shard.steps[String(steps[si])]
        if (e && e.top[li][t][0][0] === finalId) {
          idx = si
          break
        }
      }
      firstIdx[li].push(idx)
    }
  }
  return { layers, positions, firstIdx, finalTop }
}

export interface DiffMap extends GridDims {
  /** per cell: did the top-1 flip between the two steps? */
  flipped: boolean[][]
  /** per cell: 1 - Jaccard(top-10 ids) — 0 = same candidate set, 1 = fully replaced */
  change: number[][]
  refTop: TopEntry[][]
  curTop: TopEntry[][]
}

/** What changed between two checkpoints, per (layer, pos) cell — computed from the shard's
 * stored top-10 lists (exact turnover of the candidate sets; probabilities via TopEntry). */
export function diffMap(shard: Shard, refStep: number, curStep: number): DiffMap | null {
  const ref = shard.steps[String(refStep)]
  const cur = shard.steps[String(curStep)]
  if (!ref || !cur) return null
  const vocab = shard.vocab
  const entry = (cell: [number, number][]): TopEntry => [vocab[String(cell[0][0])] ?? '?', cell[0][1], cell[0][0]]
  const layers = cur.top.length
  const positions = cur.top[0].length
  const flipped = grid2D<boolean>(layers)
  const change = grid2D<number>(layers)
  const refTop = grid2D<TopEntry>(layers)
  const curTop = grid2D<TopEntry>(layers)
  for (let li = 0; li < layers; li++) {
    for (let t = 0; t < positions; t++) {
      const a = ref.top[li][t]
      const b = cur.top[li][t]
      flipped[li].push(a[0][0] !== b[0][0])
      const setA = new Set(a.map(([id]) => id))
      let inter = 0
      for (const [id] of b) if (setA.has(id)) inter++
      change[li].push(1 - inter / (setA.size + b.length - inter))
      refTop[li].push(entry(a))
      curTop[li].push(entry(b))
    }
  }
  return { layers, positions, flipped, change, refTop, curTop }
}

/**
 * Layer profile at one (pos, step): the classic logit-lens curve, p vs layer, for the
 * prompt's target tokens. Returns [{id, token, points: [[layer, prob, rank], ...]}].
 */
export function layerProfileFromShard(shard: Shard, prompt: Prompt, pos: number, step: number): TrajectorySeries[] {
  const entry = shard.steps[String(step)]
  if (!entry?.tgt) return []
  const out: TrajectorySeries[] = []
  for (const id of new Set(prompt.targets[pos] ?? [])) {
    const t = entry.tgt[String(id)]
    if (!t) continue
    const points: [number, number, number][] = t.p.map((row, li) => [li, row[pos], t.r[li][pos]])
    out.push({ id, token: shard.vocab[String(id)] ?? '?', points })
  }
  out.sort((a, b) => b.points.at(-1)![1] - a.points.at(-1)![1])
  return out
}

/**
 * Trajectory of target tokens for a pinned (layer, pos) across all steps present in the shard.
 * Returns [{id, token, points: [[step, prob, rank], ...]}].
 */
export function trajectoryFromShard(
  shard: Shard,
  prompt: Prompt,
  layer: number,
  pos: number,
  steps: number[]
): TrajectorySeries[] {
  const ids = new Set(prompt.targets[pos] ?? [])
  const out: TrajectorySeries[] = []
  for (const id of ids) {
    const points: [number, number, number][] = []
    for (const s of steps) {
      const e = shard.steps[String(s)]
      const t = e?.tgt?.[String(id)]
      if (t) points.push([s, t.p[layer][pos], t.r[layer][pos]])
    }
    if (points.length) out.push({ id, token: shard.vocab[String(id)] ?? '?', points })
  }
  out.sort((a, b) => b.points.at(-1)![1] - a.points.at(-1)![1])
  return out
}
