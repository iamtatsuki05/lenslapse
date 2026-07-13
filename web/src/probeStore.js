// Reproducibility store: every live probe result is persisted to IndexedDB, keyed by
// (model, step, prompt). Identical requests replay the saved result byte-for-byte, so a view
// keeps rendering the same numbers across sessions, backends, and even model-host outages.
// All operations are best-effort: storage failure must never break probing.

const DB = 'lenslapse-probes'
const STORE = 'probes'

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1)
    req.onupgradeneeded = () => req.result.createObjectStore(STORE)
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export function probeKey(modelId, step, text) {
  return `${modelId}|${step}|${text}`
}

export async function getProbe(key) {
  try {
    const db = await openDb()
    return await new Promise((resolve) => {
      const req = db.transaction(STORE).objectStore(STORE).get(key)
      req.onsuccess = () => resolve(req.result ?? null)
      req.onerror = () => resolve(null)
    })
  } catch {
    return null
  }
}

export async function putProbe(key, value) {
  try {
    const db = await openDb()
    await new Promise((resolve) => {
      const tx = db.transaction(STORE, 'readwrite')
      tx.objectStore(STORE).put(value, key)
      tx.oncomplete = resolve
      tx.onerror = resolve
    })
  } catch {
    /* best-effort */
  }
}
