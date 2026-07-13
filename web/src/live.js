// Live in-browser probing: onnxruntime-web sessions over per-checkpoint (backbone, lens) ONNX pairs.
// Zero backend: model files come from a static host (HF Hub or same-origin), are cached via the
// Cache API, and all computation happens on the visitor's device (WebGPU when available, else WASM).
// One LiveEngine instance per model; the models root hosts one subdirectory per model id.

import * as ort from 'onnxruntime-web/webgpu'
import { signUrl } from './auth.js'
import { getProbe, probeKey, putProbe } from './probeStore.js'

// Optional local probe server (?probe=http://localhost:8017) for models too heavy for the browser.
const PROBE_URL = new URLSearchParams(location.search).get('probe')

/** Origin of the configured probe server, or null when the app runs purely static. */
export function probeServerOrigin() {
  try {
    return PROBE_URL ? new URL(PROBE_URL, location.href).origin : null
  } catch {
    return null
  }
}

/** Server model registry ([{id, ref, mode, label, steps, origin}]) or null if unreachable. */
export async function fetchServerModels() {
  if (!probeServerOrigin()) return null
  try {
    // hard timeout: a hanging request (e.g. a private-network preflight that never settles on
    // an HTTPS page) must degrade to "no server", never stall the caller — boot awaits this
    const res = await fetch(new URL('/models', probeServerOrigin()), { signal: AbortSignal.timeout(4000) })
    return res.ok ? await res.json() : null
  } catch {
    return null
  }
}
// ?fresh bypasses saved-probe replay (IndexedDB) and recomputes; the new result overwrites the
// saved one. Needed after re-exporting a model under the same id, where replay would be stale.
const FRESH = new URLSearchParams(location.search).has('fresh')

const APP_BASE = import.meta.env.BASE_URL
// Models-root fallback chain: ?models= URL param > same-origin models/ > HF Hub dataset repo.
const HF_DEFAULT = 'https://huggingface.co/datasets/iamtatsuki05/lenslapse-onnx/resolve/main/'

// ORT's WASM runtime resolves via import.meta.url; Vite bundles the .wasm files as hashed assets
// in both dev and build, so no wasmPaths override is needed and the site stays self-contained.

const CACHE_NAME = 'lenslapse-models-v1'
const MAX_SESSIONS = 2 // fp32-expanded weights are large (70m ≈ 280MB, 160m ≈ 650MB in memory)

let sharedRoot // first root that served a manifest wins, shared by all engines

async function fetchManifest(root, modelId) {
  try {
    const res = await fetch(signUrl(new URL(`${modelId}/manifest.json`, root).href))
    return res.ok ? await res.json() : null
  } catch {
    return null
  }
}

async function resolveManifest(modelId) {
  const param = new URLSearchParams(location.search).get('models')
  const candidates = sharedRoot
    ? [sharedRoot]
    : [param, `${APP_BASE}models/`, HF_DEFAULT].filter(Boolean).map((b) => new URL(b, location.href).href)
  for (const root of candidates) {
    const manifest = await fetchManifest(root, modelId)
    if (manifest) {
      sharedRoot = root
      return { manifest, baseUrl: new URL(`${modelId}/`, root).href }
    }
  }
  return null
}

export class LiveEngine {
  constructor(modelId, hfName) {
    this.modelId = modelId
    this.hfName = hfName
    this.manifest = null
    this.baseUrl = null
    this.tokenizer = null
    this.backend = null // 'webgpu' | 'wasm'
    this.sessions = new Map() // step -> {backbone, lens}
    this.available = false
  }

  async init(onStatus) {
    const serverOrigin = probeServerOrigin()
    if (serverOrigin) {
      try {
        const res = await fetch(new URL('/health', serverOrigin), { signal: AbortSignal.timeout(4000) })
        const health = res.ok ? await res.json() : null
        if (health?.ok && health.models?.includes(this.modelId)) {
          this.server = serverOrigin
          this.backend = 'server'
          this.available = true
          return true
        }
        if (health?.ok) {
          onStatus?.(`${this.modelId} is not registered on the probe server — falling back to in-browser inference`)
        }
      } catch {
        onStatus?.(`probe server ${PROBE_URL} unreachable — falling back to in-browser inference`)
      }
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
      onStatus?.(`live probing unavailable (tokenizer failed to load: ${e.message}) — precomputed mode only`)
      return false
    }
    this.available = true
    return true
  }

  liveSteps() {
    return this.manifest ? this.manifest.steps.map((s) => s.step) : []
  }

  async fetchCached(url, onStatus, label) {
    // The Cache API is best-effort: it can be unavailable (private browsing) or full (quota);
    // neither should break the probe — worst case is re-downloading next visit.
    // Cache entries are keyed on the unsigned URL: the __sign token rotates daily and must not
    // fragment the cache.
    let cache = null
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

  async loadCheckpoint(step, onStatus) {
    if (this.sessions.has(step)) return this.sessions.get(step)
    // in-flight guard: concurrent probes for the same step must share one load
    this.loading ??= new Map()
    if (this.loading.has(step)) return await this.loading.get(step)
    const p = this._loadCheckpoint(step, onStatus)
    this.loading.set(step, p)
    try {
      return await p
    } finally {
      this.loading.delete(step)
    }
  }

  async _loadCheckpoint(step, onStatus) {
    const dir = `${this.baseUrl}step${step}/`
    const [bbFile, lensFile] = this.manifest.files ?? ['backbone.f16.onnx', 'lens.f16.onnx']
    const sizes = this.manifest.steps.find((s) => s.step === step)
    const mb = (b) => (b ? `${Math.round(b / 1e6)}MB` : '')
    const mk = async (name, bytes) => {
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
      const oldest = this.sessions.keys().next().value
      if (oldest !== step) {
        const old = this.sessions.get(oldest)
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
  async probe(text, step, onStatus) {
    // reproducibility: identical (model, step, prompt) replays the stored result
    const key = probeKey(this.modelId, step, text)
    const saved = FRESH ? null : await getProbe(key)
    if (saved) return { ...saved, replayed: true }
    const result = this.server ? await this.probeServer(text, step, onStatus) : await this.probeOnnx(text, step, onStatus)
    await putProbe(key, result)
    return result
  }

  async probeServer(text, step, onStatus) {
    onStatus?.(`probing on ${this.server}…`)
    const res = await fetch(new URL('/probe', this.server), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.modelId, step, text }),
    })
    if (!res.ok) throw new Error(`probe server: ${res.status} ${(await res.text()).slice(0, 120)}`)
    const r = await res.json()
    return {
      tokens: r.tokens,
      grid: r.grid,
      timing: { total: r.timing.total, forward: r.timing.forward, probe: r.timing.total },
      backend: 'server',
      serverCached: r.cached,
    }
  }

  async probeOnnx(text, step, onStatus) {
    const t0 = performance.now()
    const { backbone, lens } = await this.loadCheckpoint(step, onStatus)
    const enc = this.tokenizer(text)
    const ids = enc.input_ids.data // BigInt64Array
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
    const lensOut = await lens.run({ hidden: new ort.Tensor('float32', hs.data, [L1 * T, H]) })
    const logits = lensOut.logits // [N, V]
    const V = logits.dims[1]
    const t2 = performance.now()

    const tokens = Array.from(ids, (id) => this.tokenizer.decode([Number(id)]))
    const cells = []
    for (let li = 0; li < L1; li++) {
      const row = []
      for (let t = 0; t < T; t++) {
        const off = (li * T + t) * V
        row.push(topkSoftmax(logits.data, off, V, 10, (id) => this.tokenizer.decode([id])))
      }
      cells.push(row)
    }
    const t3 = performance.now()
    return {
      tokens,
      grid: { layers: L1, positions: T, cells },
      // forward = backbone + lens sessions; probe additionally includes the JS softmax/top-10 pass
      timing: { total: t3 - t0, forward: t2 - t1, probe: t3 - t1 },
      backend: this.backend,
    }
  }
}

function topkSoftmax(data, off, V, k, decode) {
  let max = -Infinity
  for (let i = 0; i < V; i++) if (data[off + i] > max) max = data[off + i]
  let Z = 0
  for (let i = 0; i < V; i++) Z += Math.exp(data[off + i] - max)
  const top = [] // [{id, logit}] ascending by logit, length <= k
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
  const entries = top.map(([id, v]) => [decode(id), Math.exp(v - max) / Z, id])
  return { token: entries[0][0], prob: entries[0][1], top: entries }
}
