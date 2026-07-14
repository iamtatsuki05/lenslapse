// Reproduce-this-view snippets: the CLI and cURL commands that fetch the exact probe the app
// is showing, for pasting into a terminal, notebook, or paper appendix.

/** Index of the first content-bearing sub-token: tokenizers that split " Kyoto" into
 * ["▁", "Kyoto"] would otherwise track the bare space, which is never what the user meant. */
export function firstContentToken(tokens: string[]): number {
  let i = 0
  while (i < tokens.length - 1 && /^[Ġ▁\s]+$/.test(tokens[i])) i++
  return i
}

/** POSIX single-quote escaping: safe for any text, including quotes and newlines. */
export function shellQuote(s: string): string {
  return `'${s.replaceAll("'", `'\\''`)}'`
}

export function probeCliCommand(model: string, text: string, step: number): string {
  return `lenslapse probe --model ${model} --step ${step} --text ${shellQuote(text)}`
}

export function probeCurlCommand(origin: string, model: string, text: string, step: number): string {
  const body = JSON.stringify({ model, step, text })
  return `curl -s ${origin}/probe -H 'content-type: application/json' -d ${shellQuote(body)}`
}
