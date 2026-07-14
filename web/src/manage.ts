// "Models" management dialog: register / remove probe-server models from the UI.
// Only wired up when a probe server is configured (?probe=...); the static site never shows it.

import { fetchServerModels, probeServerOrigin } from './live'
import type { ServerModel } from './live'

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T

const DEFAULT_SUITE_STEPS = '0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,64000,128000,143000'

/**
 * Wires the ⚙ models button + dialog. `onChange(serverModels)` fires after every successful
 * add/remove with the fresh server registry so the caller can rebuild its catalog.
 */
export function setupManageModels(onChange: (serverModels: ServerModel[] | null) => void): void {
  const origin = probeServerOrigin()
  if (!origin) return
  const btn = $('manage-models-btn')
  const dialog = $<HTMLDialogElement>('models-dialog')
  btn.hidden = false
  btn.addEventListener('click', async () => {
    await renderList()
    dialog.showModal()
  })
  $('models-close').addEventListener('click', () => dialog.close())

  const refInput = $<HTMLInputElement>('am-ref')
  const idInput = $<HTMLInputElement>('am-id')
  const stepsRow = $('am-steps-row')
  const error = $('am-error')

  const mode = () => document.querySelector<HTMLInputElement>('input[name="am-mode"]:checked')!.value
  for (const radio of document.querySelectorAll('input[name="am-mode"]')) {
    radio.addEventListener('change', () => {
      stepsRow.hidden = mode() !== 'suite'
    })
  }
  refInput.addEventListener('input', () => {
    // convenience prefills; both stay editable
    const ref = refInput.value.trim()
    if (!idInput.dataset.touched) {
      idInput.value = (ref.split(/[\\/]/).filter(Boolean).pop() ?? '')
        .toLowerCase()
        .replace(/[^a-z0-9._-]/g, '-')
        .replace(/^[^a-z0-9]+/, '') // ids must start alphanumeric
    }
    if (ref.startsWith('/') || ref.startsWith('./') || ref.startsWith('~')) {
      document.querySelector<HTMLInputElement>('input[name="am-mode"][value="local"]')!.checked = true
      stepsRow.hidden = true
    }
  })
  idInput.addEventListener('input', () => {
    idInput.dataset.touched = '1'
  })

  // native folder picker: the dialog opens on the machine running the probe server (the same
  // machine in the intended localhost setup), so nobody has to type an absolute path
  const browse = $<HTMLButtonElement>('am-browse')
  browse.addEventListener('click', async () => {
    browse.disabled = true
    error.textContent = ''
    const oldLabel = browse.textContent
    browse.textContent = 'see the dialog…'
    try {
      const res = await fetch(new URL('/pick-folder', origin), { signal: AbortSignal.timeout(310000) })
      if (!res.ok) {
        const detail = await errorDetail(res)
        if (res.status !== 400) error.textContent = detail // 400 = user cancelled; stay quiet
        return
      }
      const { path } = await res.json()
      refInput.value = path
      refInput.dispatchEvent(new Event('input')) // prefills the id
      // the picker always returns a directory — select local mode explicitly (the ref-prefix
      // heuristic would miss Windows paths like C:\Users\...)
      document.querySelector<HTMLInputElement>('input[name="am-mode"][value="local"]')!.checked = true
      stepsRow.hidden = true
    } catch (err) {
      error.textContent = `folder dialog failed: ${(err as Error).message}`
    } finally {
      browse.disabled = false
      browse.textContent = oldLabel
    }
  })

  let listGen = 0 // bumped per render: poll chains from replaced (detached) rows must stop

  async function renderList(): Promise<void> {
    const gen = ++listGen
    const list = $('server-model-list')
    const models = await fetchServerModels()
    list.replaceChildren()
    if (!models) {
      list.innerHTML = '<li class="hint">probe server unreachable</li>'
      return
    }
    for (const m of models) {
      const li = document.createElement('li')
      const steps = m.mode === 'suite' ? `${m.steps.length} steps` : m.mode === 'local' ? `${m.steps.length} local ckpts` : 'final'
      li.innerHTML = `<span class="mono">${escapeHtml(m.id)}</span> <span class="hint">${escapeHtml(m.ref)} · ${steps}</span>`
      if (m.origin === 'user') {
        li.appendChild(convertButton(m.id, gen))
        const rm = document.createElement('button')
        rm.className = 'btn'
        rm.textContent = 'remove'
        rm.addEventListener('click', async () => {
          rm.disabled = true
          try {
            const res = await fetch(new URL(`/models/${encodeURIComponent(m.id)}`, origin!), {
              method: 'DELETE',
              signal: AbortSignal.timeout(10000),
            })
            if (!res.ok) {
              error.textContent = `remove failed: ${await errorDetail(res)}`
              return
            }
            error.textContent = ''
            await renderList()
            onChange(await fetchServerModels())
          } catch (err) {
            error.textContent = `probe server unreachable: ${(err as Error).message}`
          } finally {
            rm.disabled = false
          }
        })
        li.appendChild(rm)
      }
      list.appendChild(li)
    }
  }

  /** "convert to ONNX" button + status; resumes progress display if a job is already running. */
  function convertButton(id: string, gen: number): HTMLElement {
    const wrap = document.createElement('span')
    wrap.className = 'convert-wrap'
    const btn = document.createElement('button')
    btn.className = 'btn'
    btn.textContent = 'convert to ONNX'
    const stat = document.createElement('span')
    stat.className = 'hint'
    wrap.append(btn, stat)

    const poll = async (): Promise<void> => {
      if (gen !== listGen || !$<HTMLDialogElement>('models-dialog').open) return // row was re-rendered or dialog closed
      try {
        const res = await fetch(new URL(`/models/${encodeURIComponent(id)}/convert`, origin!), {
          signal: AbortSignal.timeout(10000),
        })
        if (res.status === 404) return // no job — leave the button idle
        if (!res.ok) {
          setTimeout(poll, 3000) // transient server error: keep watching a possibly-running job
          return
        }
        const job = await res.json()
        if (job.status === 'running') {
          btn.disabled = true
          stat.textContent = `converting… ${job.log.at(-1) ?? ''}`
          setTimeout(poll, 3000)
        } else if (job.status === 'done') {
          btn.disabled = true
          stat.textContent = job.note ?? 'converted — rebuild/reload the app to run it in-browser'
        } else {
          btn.disabled = false
          stat.textContent = `conversion failed: ${job.log.at(-1) ?? 'see server log'}`
        }
      } catch {
        stat.textContent = 'probe server unreachable — retrying…'
        setTimeout(poll, 3000)
      }
    }
    btn.addEventListener('click', async () => {
      btn.disabled = true
      stat.textContent = 'starting…'
      try {
        const res = await fetch(new URL(`/models/${encodeURIComponent(id)}/convert`, origin!), {
          method: 'POST',
          signal: AbortSignal.timeout(10000),
        })
        if (!res.ok) {
          stat.textContent = ''
          error.textContent = `convert failed: ${await errorDetail(res)}`
          btn.disabled = false
          return
        }
        poll()
      } catch (err) {
        stat.textContent = ''
        error.textContent = `probe server unreachable: ${(err as Error).message}`
        btn.disabled = false
      }
    })
    poll() // pick up a job already in flight (e.g. dialog reopened mid-conversion)
    return wrap
  }

  $<HTMLFormElement>('add-model-form').addEventListener('submit', async (e) => {
    e.preventDefault()
    error.textContent = ''
    const body: { id: string; ref: string; mode: string; label: string | null; steps?: number[] } = {
      id: idInput.value.trim(),
      ref: refInput.value.trim(),
      mode: mode(),
      label: $<HTMLInputElement>('am-label').value.trim() || null,
    }
    if (body.mode === 'suite') {
      const parts = $<HTMLInputElement>('am-steps')
        .value.split(',')
        .map((s) => s.trim())
      // strict parse: Number('') is 0, so empty/garbage segments would silently become step 0
      if (!parts.length || parts.some((p) => !/^\d+$/.test(p))) {
        error.textContent = 'steps must be a comma-separated list of non-negative integers'
        return
      }
      body.steps = parts.map(Number)
    }
    const submit = $<HTMLButtonElement>('am-submit')
    submit.disabled = true
    submit.textContent = 'validating…'
    try {
      const res = await fetch(new URL('/models', origin), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(30000), // hub-side validation can take a few seconds
      })
      if (!res.ok) {
        error.textContent = await errorDetail(res)
        return
      }
      $<HTMLFormElement>('add-model-form').reset()
      $<HTMLInputElement>('am-steps').value = DEFAULT_SUITE_STEPS
      delete idInput.dataset.touched
      stepsRow.hidden = true
      await renderList()
      onChange(await fetchServerModels())
    } catch (err) {
      error.textContent = `probe server unreachable: ${(err as Error).message}`
    } finally {
      submit.disabled = false
      submit.textContent = 'Add model'
    }
  })
  $<HTMLInputElement>('am-steps').value = DEFAULT_SUITE_STEPS
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const detail = (await res.json()).detail
    return typeof detail === 'string' ? detail : JSON.stringify(detail)
  } catch {
    return `HTTP ${res.status}`
  }
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
