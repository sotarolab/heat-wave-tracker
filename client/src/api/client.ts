import type { FieldMeta, Manifest, Network, StationPage, Stations, Unit } from './types'

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(path, { signal, headers: { accept: 'application/json' } })
  if (!r.ok) {
    let detail = r.statusText
    try {
      detail = (await r.json()).detail ?? detail
    } catch {
      /* body was not JSON */
    }
    throw new Error(detail)
  }
  return (await r.json()) as T
}

export const api = {
  manifest: (signal?: AbortSignal) => get<Manifest>('/api/manifest', signal),
  stations: (unit: Unit, signal?: AbortSignal) => get<Stations>(`/api/stations?unit=${unit}`, signal),
  station: (id: string, unit: Unit, signal?: AbortSignal) => get<StationPage>(`/api/station/${encodeURIComponent(id)}?unit=${unit}`, signal),
  network: (unit: Unit, signal?: AbortSignal) => get<Network>(`/api/network?unit=${unit}`, signal),
  fieldMeta: (unit: Unit, signal?: AbortSignal) => get<FieldMeta>(`/api/field/meta?unit=${unit}`, signal),
  /** The cycle is in the address so the browser can keep an image for as long as it likes. */
  fieldUrl: (variable: string, frame: number, cycle: string) => `/api/field/${encodeURIComponent(variable)}/${frame}.png?c=${cycle}`,
}
