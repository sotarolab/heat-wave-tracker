/** The shapes service/payloads.py returns. The `contract-checker` subagent compares the two. */
export type Unit = 'F' | 'C'

export interface Manifest {
  init: string
  /** The cycle as the header shows it, e.g. "GFS 18Z Nov 16". */
  label: string
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

export interface ChartLine {
  label: string
  /** A CSS custom property name the page resolves, e.g. '--obs'. */
  color: string
  values: (number | null)[]
  dash?: boolean
  width?: number
  band?: { lo: (number | null)[]; up: (number | null)[] }
  /** Sparse readings (gusts): drawn as points, not joined across the gaps. */
  dots?: boolean
}
/** One variable's chart: its lines on the page's time axis, in `unit`, with hard limits where it has them. */
export interface ChartSpec { id: string; label: string; unit: string; lines: ChartLine[]; floor?: number; ceil?: number }
export interface Tile { label: string; value: string; note: string }
export interface TodayRow { metric: string; n: number; bias: number; rmse: number; interval: number | null }
export interface ModelRecord {
  version: string
  days: number
  n: number
  rmse_raw: number
  rmse_offset: number
  rmse_ml: number
  band_coverage_pct: number | null
}
/** service/station.py: one station's page, every number final and in `unit`. */
export interface StationPage {
  station: Station & { tz: string }
  unit: Unit
  /** The forecast on the page has ended: nothing current is drawn over it. */
  stale: boolean
  headline: string
  tiles: Tile[]
  /** Epoch seconds: the one time axis every line's values follow. */
  time: number[]
  now: number
  /** Temperature and feels like first, then the other variables, each only where it has a line. */
  charts: ChartSpec[]
  verification: { today: TodayRow[]; model: ModelRecord | null }
  has_learned: boolean
  /** How far ahead the same-day correction is drawn, and how many verified steps it needs. */
  same_day_hours: number
  min_steps: number
  /** Hours over which a corrected line's gap to the latest reading fades; null when no line is tied to one. */
  anchor_hours: number | null
}

export interface NetworkRow extends Station {
  observed: number | null
  observed_min_ago: number | null
  forecast: number | null
  /** Observed minus forecast, a difference in `unit`. */
  error: number | null
  high_today: number | null
  high_tomorrow: number | null
}
export interface Network {
  unit: Unit
  /** False when the forecast on the page has ended. */
  current: boolean
  /** Hours back an observation may be and still count as now. */
  window_h: number
  reporting: number
  stations: number
  mean_abs_error: number | null
  rows: NetworkRow[]
}
