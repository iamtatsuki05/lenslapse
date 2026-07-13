// "Models" management dialog: register / remove probe-server models from the UI.
// Only wired up when a probe server is configured (?probe=...); the static site never shows it.

import { fetchServerModels, probeServerOrigin } from './live.js'

const $ = (id) => document.getElementById(id)

const DEFAULT_SUITE_STEPS = '0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,64000,128000,143000'

/**
 * Wires the ⚙ models button + dialog. `onChange(serverModels)` fires after every successful
 * add/remove with the fresh server registry so the caller can rebuild its catalog.
 */
export function setupManageModels(onChange) {
  const origin = probeServerOrigin()
  if (!origin) return
  const btn = $('manage-models-btn')
  const dialog = $('models-dialog')
  btn.hidden = false
  btn.addEventListener('click', async () => {
    await renderList()
    dialog.showModal()
  })
  $('models-close').addEventListener('click', () => dialog.close())

  const refInput = $('am-ref')
  const idInput = $('am-id')
  const stepsRow = $('am-steps-row')
  const error = $('am-error')

  const mode = () => document.querySelector('input[name="am-mode"]:checked').value
  for (const radio of document.querySelectorAll('input[name="am-mode"]')) {
    radio.addEventListener('change', () => {
      stepsRow.hidden = mode() !== 'suite'
    })
  }
  refInput.addEventListener('input', () => {
    // convenience prefills; both stay editable
    const ref = refInput.value.trim()
    if (!idInput.dataset.touched) {
      idInput.value = (ref.split('/').filter(Boolean).pop() ?? '')
        .toLowerCase()
        .replace(/[^a-z0-9._-]/g, '-')
        .replace(/^[^a-z0-9]+/, '') // ids must start alphanumeric
    }
    if (ref.startsWith('/') || ref.startsWith('./') || ref.startsWith('~')) {
      document.querySelector('input[name="am-mode"][value="local"]').checked = true
      stepsRow.hidden = true
    }
  })
  idInput.addEventListener('input', () => {
    idInput.dataset.touched = '1'
  })

  async function renderList() {
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
        const rm = document.createElement('button')
        rm.className = 'btn'
        rm.textContent = 'remove'
        rm.addEventListener('click', async () => {
          rm.disabled = true
          try {
            const res = await fetch(new URL(`/models/${encodeURIComponent(m.id)}`, origin), {
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
            error.textContent = `probe server unreachable: ${err.message}`
          } finally {
            rm.disabled = false
          }
        })
        li.appendChild(rm)
      }
      list.appendChild(li)
    }
  }

  $('add-model-form').addEventListener('submit', async (e) => {
    e.preventDefault()
    error.textContent = ''
    const body = {
      id: idInput.value.trim(),
      ref: refInput.value.trim(),
      mode: mode(),
      label: $('am-label').value.trim() || null,
    }
    if (body.mode === 'suite') {
      const parts = $('am-steps')
        .value.split(',')
        .map((s) => s.trim())
      // strict parse: Number('') is 0, so empty/garbage segments would silently become step 0
      if (!parts.length || parts.some((p) => !/^\d+$/.test(p))) {
        error.textContent = 'steps must be a comma-separated list of non-negative integers'
        return
      }
      body.steps = parts.map(Number)
    }
    const submit = $('am-submit')
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
      $('add-model-form').reset()
      $('am-steps').value = DEFAULT_SUITE_STEPS
      delete idInput.dataset.touched
      stepsRow.hidden = true
      await renderList()
      onChange(await fetchServerModels())
    } catch (err) {
      error.textContent = `probe server unreachable: ${err.message}`
    } finally {
      submit.disabled = false
      submit.textContent = 'Add model'
    }
  })
  $('am-steps').value = DEFAULT_SUITE_STEPS
}

async function errorDetail(res) {
  try {
    const detail = (await res.json()).detail
    return typeof detail === 'string' ? detail : JSON.stringify(detail)
  } catch {
    return `HTTP ${res.status}`
  }
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
