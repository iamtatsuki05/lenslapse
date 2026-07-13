// Private Hugging Face Spaces authenticate via a `__sign` JWT on the page URL, which sets a
// SameSite=Lax cookie. In embedded or cookie-restricted contexts that cookie is not sent, so
// same-origin fetches would 401 even though the server accepts `?__sign=` on any path.
// Propagating the token explicitly makes data/model fetches work wherever the app itself loaded.
// On public hosting (`__sign` absent) this is a no-op.

const SIGN = new URLSearchParams(location.search).get('__sign')

/** Append the page's __sign token to a same-origin URL (absolute or relative). */
export function signUrl(url) {
  const u = new URL(url, location.href)
  if (!SIGN || u.origin !== location.origin || u.searchParams.has('__sign')) return u.href
  u.searchParams.set('__sign', SIGN)
  return u.href
}
