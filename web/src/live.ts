// Live in-browser probing: onnxruntime-web sessions over per-checkpoint (backbone, lens) ONNX pairs.
// Zero backend: model files come from a static host (HF Hub or same-origin), are cached via the
// Cache API, and all computation happens on the visitor's device (WebGPU when available, else WASM).
// One LiveEngine instance per model; the models root hosts one subdirectory per model id.

import * as ort from 'onnxruntime-web/webgpu'
import { signUrl } from './auth'
import { getProbe, probeKey, putProbe } from './probeStore'
import type { GridCell, GridData, TopEntry } from './data'
import type { PreTrainedTokenizer } from '@huggingface/transformers'

type StatusFn = (msg: string) => void

/** One entry of the probe server's /models registry. */
export interface ServerModel {
  id: string
  ref: string
  mode: string
  label?: string | null
  steps: number[]
  origin: string
}

/** Converted-checkpoint manifest served next to the ONNX files. */
interface Manifest {
  files?: [string, string]
  steps: { step: number; backbone_bytes?: number; lens_bytes?: number }[]
}

interface SessionPair {
  backbone: ort.InferenceSession
  lens: ort.InferenceSession
}

/** Result of one probe (server or in-browser); also the shape persisted for replay. */
export interface ProbeResult {
  tokens: string[]
  grid: GridData
  timing: { total: number; forward: number; probe: number }
  backend: string
  serverCached?: boolean
  replayed?: boolean
  /** Exact probability/rank per (layer, position) for explicitly tracked token ids. */
  tgt?: Record<string, { token: string; p: number[][]; r: number[][] }>
}

// Optional local probe server for models too heavy for the browser.
// ?probe=<url> connects and is remembered (localStorage) so later visits need no parameter;
// ?probe=off forgets it. A locally-served app additionally auto-detects its own origin (the
// probe server can serve this app itself) and then the default port — a closed local port
// refuses instantly, so this costs nothing when no server is running. Public (non-localhost)
// deployments never probe the visitor's machine without explicit opt-in.
export function resolveProbeCandidates(): { candidates: string[]; explicit: boolean } {
  const toOrigin = (u: string): string | null => {
    try {
      return new URL(u, location.href).origin
    } catch {
      return null
    }
  }
  const param = new URLSearchParams(location.search).get('probe')
  if (param === 'off') {
    try {
      localStorage.removeItem('lenslapse-probe')
    } catch {
      /* storage unavailable */
    }
    return { candidates: [], explicit: false }
  }
  if (param) {
    try {
      localStorage.setItem('lenslapse-probe', param)
    } catch {
      /* storage unavailable — still usable for this visit */
    }
    const origin = toOrigin(param)
    return { candidates: origin ? [origin] : [], explicit: true }
  }
  try {
    const saved = localStorage.getItem('lenslapse-probe')
    if (saved) {
      const origin = toOrigin(saved)
      return { candidates: origin ? [origin] : [], explicit: true }
    }
  } catch {
    /* storage unavailable */
  }
  if (['localhost', '127.0.0.1'].includes(location.hostname)) {
    return { candidates: [...new Set([location.origin, 'http://localhost:8017'])], explicit: false }
  }
  return { candidates: [], explicit: false }
}
const PROBE = resolveProbeCandidates()
let resolvedProbeOrigin: string | null = null

/** Candidate probe-server origins, most likely first (a previously reached one leads). */
function probeOrigins(): string[] {
  if (resolvedProbeOrigin) {
    return [resolvedProbeOrigin, ...PROBE.candidates.filter((c) => c !== resolvedProbeOrigin)]
  }
  return PROBE.candidates
}

/** The probe server the app is talking to (or would try first), or null when purely static. */
export function probeServerOrigin(): string | null {
  return resolvedProbeOrigin ?? PROBE.candidates[0] ?? null
}

/** Server model registry ([{id, ref, mode, label, steps, origin}]) or null if unreachable. */
export async function fetchServerModels(): Promise<ServerModel[] | null> {
  for (const origin of probeOrigins()) {
    try {
      // hard timeout: a hanging request (e.g. a private-network preflight that never settles on
      // an HTTPS page) must degrade to "no server", never stall the caller — boot awaits this
      const res = await fetch(new URL('/models', origin), { signal: AbortSignal.timeout(4000) })
      if (res.ok) {
        const models = await res.json() // parse before caching: a static 200 must not poison the origin
        resolvedProbeOrigin = origin
        return models
      }
    } catch {
      /* try the next candidate */
    }
  }
  return null
}
// ?fresh bypasses saved-probe replay (IndexedDB) and recomputes; the new result overwrites the
// saved one. Needed after re-exporting a model under the same id, where replay would be stale.
const FRESH = new URLSearchParams(location.search).has('fresh')

const APP_BASE = import.meta.env.BASE_URL
// Models-root fallback chain: ?models= URL param > same-origin models/ > HF Hub dataset repo.
const HF_DEFAULT = 'https://huggingface.co/iamtatsuki05/lenslapse-onnx/resolve/main/'

// ORT's WASM runtime resolves via import.meta.url; Vite bundles the .wasm files as hashed assets
// in both dev and build, so no wasmPaths override is needed and the site stays self-contained.

const CACHE_NAME = 'lenslapse-models-v1'
const MAX_SESSIONS = 2 // fp32-expanded weights are large (70m ≈ 280MB, 160m ≈ 650MB in memory)

async function fetchManifest(root: string, modelId: string): Promise<Manifest | null> {
  try {
    const res = await fetch(signUrl(new URL(`${modelId}/manifest.json`, root).href), {
      signal: AbortSignal.timeout(8000),
    })
    return res.ok ? await res.json() : null
  } catch {
    return null
  }
}

async function resolveManifest(modelId: string): Promise<{ manifest: Manifest; baseUrl: string } | null> {
  // every model walks the full chain in priority order (explicit param > same-origin > Hub):
  // a locally converted model lives under the dev middleware while shipped suites live on the
  // Hub, and a remembered root must never outrank an explicit ?models= override
  const chain = [new URLSearchParams(location.search).get('models'), `${APP_BASE}models/`, HF_DEFAULT]
  for (const base of chain.filter(Boolean) as string[]) {
    const root = new URL(base, location.href).href
    const manifest = await fetchManifest(root, modelId)
    if (manifest) return { manifest, baseUrl: new URL(`${modelId}/`, root).href }
  }
  return null
}

export class LiveEngine {
  // `declare`: type-only field declarations — the constructor (or a later method, for `server`
  // and `loading`) assigns them, and emitting real class fields would change the compiled output.
  declare modelId: string
  declare hfName: string
  declare manifest: Manifest | null
  declare baseUrl: string | null
  declare tokenizer: PreTrainedTokenizer | null
  declare backend: 'webgpu' | 'wasm' | 'server' | null
  declare sessions: Map<number, SessionPair>
  declare available: boolean
  declare server?: string
  declare loading?: Map<number, Promise<SessionPair>>

  constructor(modelId: string, hfName: string) {
    this.modelId = modelId
    this.hfName = hfName
    this.manifest = null
    this.baseUrl = null
    this.tokenizer = null
    this.backend = null // 'webgpu' | 'wasm'
    this.sessions = new Map() // step -> {backbone, lens}
    this.available = false
  }

  async init(onStatus?: StatusFn): Promise<boolean> {
    for (const serverOrigin of probeOrigins()) {
      let health: { ok?: boolean; models?: string[] } | null = null
      try {
        const res = await fetch(new URL('/health', serverOrigin), { signal: AbortSignal.timeout(4000) })
        health = res.ok ? await res.json() : null
      } catch {
        continue // try the next candidate
      }
      if (!health?.ok) continue
      resolvedProbeOrigin = serverOrigin
      if (health.models?.includes(this.modelId)) {
        this.server = serverOrigin
        this.backend = 'server'
        this.available = true
        return true
      }
      onStatus?.(`${this.modelId} is not registered on the probe server — falling back to in-browser inference`)
      continue // another candidate may have this model (e.g. a second server on the default port)
    }
    if (PROBE.explicit && !resolvedProbeOrigin) {
      onStatus?.(`probe server ${probeServerOrigin()} unreachable — falling back to in-browser inference`)
    }
    const resolved = await resolveManifest(this.modelId)
    if (resolved) {
      this.manifest = resolved.manifest
      this.baseUrl = resolved.baseUrl
    }
    if (!this.manifest) {
      onStatus?.('live probing unavailable (model host unreachable) — precomputed mode only')
      return false
    }
    const epOverride = new URLSearchParams(location.search).get('ep')
    this.backend =
      epOverride === 'wasm' ? 'wasm' : 'gpu' in navigator && navigator.gpu ? 'webgpu' : 'wasm'
    try {
      const { AutoTokenizer, env } = await import('@huggingface/transformers')
      env.allowRemoteModels = false
      env.allowLocalModels = true
      env.localModelPath = `${APP_BASE}tokenizer/`
      this.tokenizer = await AutoTokenizer.from_pretrained(this.hfName)
    } catch (e) {
      // init must resolve, never reject: callers await it outside their try blocks
      onStatus?.(`live probing unavailable (tokenizer failed to load: ${(e as Error).message}) — precomputed mode only`)
      return false
    }
    this.available = true
    return true
  }

  /** Tokenize with this model's own tokenizer: locally when it is loaded, else via the probe
   * server (server-backed models never ship a tokenizer to the browser). */
  async tokenize(text: string): Promise<{ ids: number[]; tokens: string[] }> {
    if (this.tokenizer) {
      const enc = this.tokenizer(text)
      const ids = Array.from(enc.input_ids.data as ArrayLike<bigint | number>, Number)
      return { ids, tokens: ids.map((id) => this.tokenizer!.decode([id])) }
    }
    if (this.server) {
      const res = await fetch(new URL('/tokenize', this.server), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: this.modelId, text }),
        signal: AbortSignal.timeout(30000),
      })
      if (!res.ok) throw new Error(`tokenize failed (${res.status})`)
      return await res.json()
    }
    throw new Error('no tokenizer available for this model')
  }

  liveSteps(): number[] {
    return this.manifest ? this.manifest.steps.map((s) => s.step) : []
  }

  async fetchCached(url: string, onStatus?: StatusFn, label?: string): Promise<ArrayBuffer> {
    // The Cache API is best-effort: it can be unavailable (private browsing) or full (quota);
    // neither should break the probe — worst case is re-downloading next visit.
    // Cache entries are keyed on the unsigned URL: the __sign token rotates daily and must not
    // fragment the cache.
    let cache: Cache | null = null
    try {
      cache = 'caches' in self ? await caches.open(CACHE_NAME) : null
      const hit = cache ? await cache.match(url) : null
      if (hit) return await hit.arrayBuffer()
    } catch {
      cache = null
    }
    onStatus?.(`downloading ${label}…`)
    const res = await fetch(signUrl(url))
    if (!res.ok) throw new Error(`fetch failed: ${url} (${res.status})`)
    const clone = cache ? res.clone() : null
    const buf = await res.arrayBuffer()
    if (cache && clone) {
      try {
        await cache.put(url, clone)
      } catch {
        /* quota exceeded — continue uncached */
      }
    }
    return buf
  }

  async loadCheckpoint(step: number, onStatus?: StatusFn): Promise<SessionPair> {
    if (this.sessions.has(step)) return this.sessions.get(step)!
    // in-flight guard: concurrent probes for the same step must share one load
    this.loading ??= new Map()
    if (this.loading.has(step)) return await this.loading.get(step)!
    const p = this._loadCheckpoint(step, onStatus)
    this.loading.set(step, p)
    try {
      return await p
    } finally {
      this.loading.delete(step)
    }
  }

  async _loadCheckpoint(step: number, onStatus?: StatusFn): Promise<SessionPair> {
    const dir = `${this.baseUrl}step${step}/`
    const [bbFile, lensFile] = this.manifest!.files ?? ['backbone.f16.onnx', 'lens.f16.onnx']
    const sizes = this.manifest!.steps.find((s) => s.step === step)
    const mb = (b?: number) => (b ? `${Math.round(b / 1e6)}MB` : '')
    const mk = async (name: string, bytes?: number) => {
      const buf = await this.fetchCached(`${dir}${name}`, onStatus, `${this.modelId} step ${step} ${name} (${mb(bytes)})`)
      // recompute providers per file: a WebGPU failure on the first file must demote the second too
      const providers = this.backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
      try {
        return await ort.InferenceSession.create(buf, { executionProviders: providers })
      } catch (e) {
        if (this.backend === 'webgpu') {
          this.backend = 'wasm'
          return await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] })
        }
        throw e
      }
    }
    onStatus?.(`loading ${this.modelId} checkpoint step ${step}…`)
    const backbone = await mk(bbFile, sizes?.backbone_bytes)
    const lens = await mk(lensFile, sizes?.lens_bytes)
    const pair = { backbone, lens }
    this.sessions.set(step, pair)
    if (this.sessions.size > MAX_SESSIONS) {
      const oldest = this.sessions.keys().next().value!
      if (oldest !== step) {
        const old = this.sessions.get(oldest)!
        this.sessions.delete(oldest)
        try {
          await Promise.all([old.backbone.release(), old.lens.release()])
        } catch {
          /* releasing is best-effort */
        }
      }
    }
    return pair
  }

  /** Single forward pass + lens over every (layer, position). Returns grid cells + timing. */
  async probe(text: string, step: number, onStatus?: StatusFn, targets?: number[]): Promise<ProbeResult> {
    // reproducibility: identical (model, step, prompt) replays the stored result — unless the
    // caller tracks target tokens the stored result does not carry yet
    const key = probeKey(this.modelId, step, text)
    const saved = FRESH ? null : await getProbe<ProbeResult>(key)
    if (saved && (!targets?.length || targets.every((id) => saved.tgt?.[String(id)]))) {
      return { ...saved, replayed: true }
    }
    const result = this.server
      ? await this.probeServer(text, step, onStatus, targets)
      : await this.probeOnnx(text, step, onStatus, targets)
    // merge tracked tokens into the stored record so earlier targets survive later probes
    const merged = saved?.tgt || result.tgt ? { ...result, tgt: { ...saved?.tgt, ...result.tgt } } : result
    await putProbe(key, merged)
    return merged
  }

  async probeServer(text: string, step: number, onStatus?: StatusFn, targets?: number[]): Promise<ProbeResult> {
    onStatus?.(`probing on ${this.server}…`)
    const res = await fetch(new URL('/probe', this.server), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.modelId, step, text, ...(targets?.length ? { targets } : {}) }),
    })
    if (!res.ok) throw new Error(`probe server: ${res.status} ${(await res.text()).slice(0, 120)}`)
    const r = await res.json()
    return {
      tokens: r.tokens,
      grid: r.grid,
      timing: { total: r.timing.total, forward: r.timing.forward, probe: r.timing.total },
      backend: 'server',
      serverCached: r.cached,
      ...(r.tgt ? { tgt: r.tgt } : {}),
    }
  }

  async probeOnnx(text: string, step: number, onStatus?: StatusFn, targets?: number[]): Promise<ProbeResult> {
    const t0 = performance.now()
    const { backbone, lens } = await this.loadCheckpoint(step, onStatus)
    const enc = this.tokenizer!(text)
    const ids = enc.input_ids.data as BigInt64Array // BigInt64Array
    const T = ids.length
    if (T === 0) throw new Error('empty prompt')
    if (T > 64) throw new Error('prompt too long (max 64 tokens for the live probe)')
    const mask = new BigInt64Array(T).fill(1n)
    const t1 = performance.now()
    const out = await backbone.run({
      input_ids: new ort.Tensor('int64', ids, [1, T]),
      attention_mask: new ort.Tensor('int64', mask, [1, T]),
    })
    const hs = out.hidden_states // [L+1, 1, T, H]
    const [L1, , , H] = hs.dims
    const lensOut = await lens.run({ hidden: new ort.Tensor('float32', hs.data as Float32Array, [L1 * T, H]) })
    const logits = lensOut.logits // [N, V]
    const V = logits.dims[1]
    const t2 = performance.now()

    const tokens = Array.from(ids, (id) => this.tokenizer!.decode([Number(id)]))
    const cells: GridCell[][] = []
    for (let li = 0; li < L1; li++) {
      const row: GridCell[] = []
      for (let t = 0; t < T; t++) {
        const off = (li * T + t) * V
        row.push(topkSoftmax(logits.data as Float32Array, off, V, 10, (id) => this.tokenizer!.decode([id])))
      }
      cells.push(row)
    }
    let tgt: ProbeResult['tgt']
    if (targets?.length) {
      tgt = {}
      for (const tid of [...new Set(targets)].sort((a, b) => a - b)) {
        const pRows: number[][] = []
        const rRows: number[][] = []
        for (let li = 0; li < L1; li++) {
          const pRow: number[] = []
          const rRow: number[] = []
          for (let t = 0; t < T; t++) {
            const off = (li * T + t) * V
            const { p, r } = targetStat(logits.data as Float32Array, off, V, tid)
            pRow.push(p)
            rRow.push(r)
          }
          pRows.push(pRow)
          rRows.push(rRow)
        }
        tgt[String(tid)] = { token: this.tokenizer!.decode([tid]), p: pRows, r: rRows }
      }
    }
    const t3 = performance.now()
    return {
      tokens,
      grid: { layers: L1, positions: T, cells },
      // forward = backbone + lens sessions; probe additionally includes the JS softmax/top-10 pass
      timing: { total: t3 - t0, forward: t2 - t1, probe: t3 - t1 },
      backend: this.backend!,
      ...(tgt ? { tgt } : {}),
    }
  }
}

/** Exact softmax probability and rank of one token id in a cell's logit row. */
export function targetStat(data: Float32Array, off: number, V: number, tid: number): { p: number; r: number } {
  let max = -Infinity
  for (let i = 0; i < V; i++) if (data[off + i] > max) max = data[off + i]
  let Z = 0
  let rank = 1
  const v = data[off + tid]
  for (let i = 0; i < V; i++) {
    Z += Math.exp(data[off + i] - max)
    if (data[off + i] > v) rank++
  }
  return { p: Math.round((Math.exp(v - max) / Z) * 1e6) / 1e6, r: rank }
}

export function topkSoftmax(
  data: Float32Array,
  off: number,
  V: number,
  k: number,
  decode: (id: number) => string
): GridCell {
  let max = -Infinity
  for (let i = 0; i < V; i++) if (data[off + i] > max) max = data[off + i]
  let Z = 0
  for (let i = 0; i < V; i++) Z += Math.exp(data[off + i] - max)
  const top: [number, number][] = [] // [{id, logit}] ascending by logit, length <= k
  for (let i = 0; i < V; i++) {
    const v = data[off + i]
    if (top.length < k) {
      top.push([i, v])
      if (top.length === k) top.sort((a, b) => a[1] - b[1])
    } else if (v > top[0][1]) {
      top[0] = [i, v]
      let j = 0
      while (j + 1 < k && top[j][1] > top[j + 1][1]) {
        ;[top[j], top[j + 1]] = [top[j + 1], top[j]]
        j++
      }
    }
  }
  top.sort((a, b) => b[1] - a[1])
  const entries = top.map(([id, v]): TopEntry => [decode(id), Math.exp(v - max) / Z, id])
  return { token: entries[0][0], prob: entries[0][1], top: entries }
}
