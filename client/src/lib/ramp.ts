import type { Legend } from '../api/types'

/** The colour for a station with no value at this frame. */
export const NO_VALUE = '#5e5c56'

const rgb = (hex: string) => [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16))

/** A station dot's colour on the scale the service sent with the field, so the dot and the field
 *  under it agree. The scale's stops, range and category bounds are the service's; this only reads them. */
export function dotColor(value: number | null, legend: Legend): string {
  if (value == null || Number.isNaN(value)) return NO_VALUE
  if (legend.kind === 'categories') {
    let color = legend.items[0].color
    for (const item of legend.items) if (item.from != null && value >= item.from) color = item.color
    return color
  }
  const { stops, min, max } = legend
  const t = Math.max(0, Math.min(1, (value - min) / (max - min))) * (stops.length - 1)
  const k = Math.min(Math.floor(t), stops.length - 2)
  const [a, b] = [rgb(stops[k]), rgb(stops[k + 1])]
  return `rgb(${a.map((c, j) => Math.round(c + (b[j] - c) * (t - k))).join(', ')})`
}
