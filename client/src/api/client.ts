import type { FieldMeta, Manifest, Network, StationPage, Stations, Unit } from './types'

/** Waits before the second and third attempt. A gateway between the browser and the service can
 *  drop a request the service never saw; asking again a moment later nearly always works, and a
 *  panel that gives up on the first failure stays empty until the next five-minute refresh. */
export const RETRY_MS = [400, 1200]
const GATEWAY = new Set([502, 503, 504])
const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const id = setTimeout(resolve, ms)
    signal?.addEventListener('abort', () => { clearTimeout(id); reject(signal.reason) }, { once: true })
  })

export async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    const last = attempt >= RETRY_MS.length
    let r: Response
    try {
      r = await fetch(path, { signal, headers: { accept: 'application/json' } })
    } catch (e) {
      if (last || signal?.aborted) throw e // the network is down, or the caller moved on
      await wait(RETRY_MS[attempt], signal)
      continue
    }
    if (r.ok) return (await r.json()) as T
    if (GATEWAY.has(r.status) && !last) {
      await wait(RETRY_MS[attempt], signal)
      continue
    }
    let detail = r.statusText || `HTTP ${r.status}`
    try {
      detail = (await r.json()).detail ?? detail
    } catch {
      /* body was not JSON */
    }
    throw new Error(detail)
  }
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
