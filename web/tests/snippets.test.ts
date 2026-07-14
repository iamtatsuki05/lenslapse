import { describe, expect, it } from 'vitest'
import { firstContentToken, probeCliCommand, probeCurlCommand, shellQuote } from '../src/snippets'

describe('firstContentToken', () => {
  it('skips leading bare space/metaspace sub-tokens', () => {
    expect(firstContentToken(['▁', 'Kyoto'])).toBe(1) // sarashina-style " Kyoto"
    expect(firstContentToken(['ĠKyoto'])).toBe(0) // BPE keeps the space attached
    expect(firstContentToken(['▁Kyoto'])).toBe(0)
    expect(firstContentToken(['▁'])).toBe(0) // never walks past the last token
  })
})

describe('shellQuote', () => {
  it('wraps in single quotes and survives embedded quotes and newlines', () => {
    expect(shellQuote('plain')).toBe(`'plain'`)
    expect(shellQuote(`it's`)).toBe(`'it'\\''s'`)
    expect(shellQuote('a\nb')).toBe(`'a\nb'`)
    expect(shellQuote('say "hi"')).toBe(`'say "hi"'`)
  })
})

describe('probe snippets', () => {
  it('builds the CLI command for the current view', () => {
    expect(probeCliCommand('pythia-70m', 'The capital of Japan is the city of', 8000)).toBe(
      `lenslapse probe --model pythia-70m --step 8000 --text 'The capital of Japan is the city of'`
    )
  })

  it('builds a cURL command whose -d payload is valid JSON', () => {
    const cmd = probeCurlCommand('http://localhost:8017', 'my-run', `it's`, 700)
    expect(cmd.startsWith(`curl -s http://localhost:8017/probe`)).toBe(true)
    const m = cmd.match(/-d '(.*)'$/s)!
    const payload = m[1].replaceAll(`'\\''`, `'`)
    expect(JSON.parse(payload)).toEqual({ model: 'my-run', step: 700, text: `it's` })
  })
})
