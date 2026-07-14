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

export interface GridData {
  layers: number
  positions: number
  cells: GridCell[][]
}

export interface TrajectorySeries {
  id: number
  token: string
  points: [number, number, number][]
}

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

export interface AcquisitionMap {
  layers: number
  positions: number
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
  const firstIdx: number[][] = []
  const finalTop: TopEntry[][] = []
  for (let li = 0; li < layers; li++) {
    firstIdx.push([])
    finalTop.push([])
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
