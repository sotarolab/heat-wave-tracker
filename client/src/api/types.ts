/** The shapes service/payloads.py returns. The `contract-checker` subagent compares the two. */
export type Unit = 'F' | 'C'

export interface Manifest {
  init: string
  age_h: number | null
  badge: string
  live: boolean
  steps: number
  stations: number
  commit: string | null
}

export interface Frame { i: number; time: string; label: string }
export interface RampLegend { kind: 'ramp'; stops: string[]; min: number; max: number }
export interface CategoryLegend { kind: 'categories'; items: { label: string; color: string; from: number | null }[] }
export type Legend = RampLegend | CategoryLegend

export interface Variable { id: string; label: string; note: string; frames: Frame[]; start: number; legend: Legend }
export interface FieldMeta {
  cycle: string
  unit: Unit
  /** [[south, west], [north, east]]: the outer edges of the grid the images cover. */
  bounds: [[number, number], [number, number]]
  opacity: number
  variables: Variable[]
}

export interface Station { id: string; name: string; state: string; lat: number; lon: number }
export interface Stations {
  unit: Unit
  stations: Station[]
  /** values[variable id][station index][frame index], in `unit`; null where the grid has no value. */
  values: Record<string, (number | null)[][]>
}
