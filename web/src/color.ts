// Probability -> color mapping for the lens grid (viridis control points, perceptually ordered).
export type Rgb = [number, number, number]

const VIRIDIS: Rgb[] = [
  [68, 1, 84],
  [71, 44, 122],
  [59, 81, 139],
  [44, 113, 142],
  [33, 144, 141],
  [39, 173, 129],
  [92, 200, 99],
  [170, 220, 50],
  [253, 231, 37],
]

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** prob in [0,1] -> rgb array. sqrt scale so low-probability structure stays visible. */
export function probColor(p: number): Rgb {
  // a NaN prob (a live probe whose logits went NaN) would index VIRIDIS[NaN] and throw,
  // killing the whole grid render — paint the one bad cell neutral gray instead
  if (!Number.isFinite(p)) return [128, 128, 128]
  const t = Math.sqrt(Math.min(Math.max(p, 0), 1)) * (VIRIDIS.length - 1)
  const i = Math.min(Math.floor(t), VIRIDIS.length - 2)
  const f = t - i
  return [
    Math.round(lerp(VIRIDIS[i][0], VIRIDIS[i + 1][0], f)),
    Math.round(lerp(VIRIDIS[i][1], VIRIDIS[i + 1][1], f)),
    Math.round(lerp(VIRIDIS[i][2], VIRIDIS[i + 1][2], f)),
  ]
}

/** prob -> [0,1] on a log10 axis from 1e-6 to 1: near-uniform early-training distributions
 * (p ~ vocab^-1 ~ 2e-5) become visibly structured instead of uniformly dark. */
export function logProb01(p: number): number {
  const floor = 1e-6
  return (Math.log10(Math.min(Math.max(p, floor), 1)) + 6) / 6
}

export function cssColor(rgb: Rgb): string {
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`
}

/** black/white text for contrast on a given background. */
export function textColorOn(rgb: Rgb): string {
  const lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
  return lum > 140 ? '#111' : '#fff'
}

export function legendGradient(): string {
  const stops: string[] = []
  for (let i = 0; i <= 10; i++) {
    stops.push(`${cssColor(probColor(i / 10))} ${i * 10}%`)
  }
  return `linear-gradient(90deg, ${stops.join(', ')})`
}
