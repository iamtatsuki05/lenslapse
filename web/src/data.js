// Precomputed-data store: models.json (catalog), per-model index.json + per-prompt shards,
// fetched lazily with promise-caching so concurrent callers share one fetch.

const BASE = `${import.meta.env.BASE_URL}data/`

const cache = new Map()

function fetchJson(path) {
  if (!cache.has(path)) {
    const p = fetch(`${BASE}${path}`).then((res) => {
      if (!res.ok) throw new Error(`${path} missing (${res.status})`)
      return res.json()
    })
    p.catch(() => cache.delete(path))
    cache.set(path, p)
  }
  return cache.get(path)
}

/** { models: [{id, hf, label}], default: id } */
export function loadModels() {
  return fetchJson('models.json')
}

export function loadIndex(modelId) {
  return fetchJson(`${modelId}/index.json`)
}

export function loadShard(modelId, promptId) {
  return fetchJson(`${modelId}/p${promptId}.json`)
}

/**
 * Grid view for one (prompt, step) from a shard.
 * Returns { layers, positions, cells } with cells[layer][pos] = {token, prob, top:[[token,prob,id]...]}.
 */
export function gridFromShard(shard, step) {
  const entry = shard.steps[String(step)]
  if (!entry) return null
  const vocab = shard.vocab
  const cells = entry.top.map((layerRow) =>
    layerRow.map((cell) => ({
      token: vocab[String(cell[0][0])] ?? '?',
      prob: cell[0][1],
      top: cell.map(([id, p]) => [vocab[String(id)] ?? '?', p, id]),
    }))
  )
  return { layers: cells.length, positions: cells[0].length, cells }
}

/**
 * Trajectory of target tokens for a pinned (layer, pos) across all steps present in the shard.
 * Returns [{id, token, points: [[step, prob, rank], ...]}].
 */
export function trajectoryFromShard(shard, prompt, layer, pos, steps) {
  const ids = new Set(prompt.targets[pos] ?? [])
  const out = []
  for (const id of ids) {
    const points = []
    for (const s of steps) {
      const e = shard.steps[String(s)]
      const t = e?.tgt?.[String(id)]
      if (t) points.push([s, t.p[layer][pos], t.r[layer][pos]])
    }
    if (points.length) out.push({ id, token: shard.vocab[String(id)] ?? '?', points })
  }
  out.sort((a, b) => b.points.at(-1)[1] - a.points.at(-1)[1])
  return out
}
