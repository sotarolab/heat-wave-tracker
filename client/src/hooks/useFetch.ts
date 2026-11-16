import { useEffect, useRef, useState } from 'react'

export interface Fetched<T> {
  data: T | null
  error: string | null
  loading: boolean
}

/** Run a fetcher whenever `deps` change, aborting the previous request, and again on each
 *  change of `refresh` without aborting anything.
 *
 *  The distinction is the point. A new station or day makes the request in flight worthless,
 *  so it is cancelled. A refresh tick does not: the panel wants the same data, only newer. Before
 *  this the tick was in `deps`, so a request slower than the refresh interval was cancelled and
 *  restarted every tick and the panel said "Loading…" for as long as the server stayed slow. Now
 *  a tick that finds a request still running lets it land and waits for the next tick. */
export function useFetch<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
  refresh?: number,
): Fetched<T> {
  const [state, setState] = useState<Fetched<T>>({ data: null, error: null, loading: true })
  const inflight = useRef<AbortController | null>(null)
  const key = useRef<string | null>(null)
  useEffect(() => {
    const k = JSON.stringify(deps)
    const changed = key.current !== k
    key.current = k
    if (!changed && inflight.current) return
    inflight.current?.abort()
    const ctl = new AbortController()
    inflight.current = ctl
    setState((s) => ({ ...s, loading: true, error: null }))
    fetcher(ctl.signal)
      .then((data) => {
        if (ctl.signal.aborted) return
        if (inflight.current === ctl) inflight.current = null
        setState({ data, error: null, loading: false })
      })
      .catch((e: unknown) => {
        if (ctl.signal.aborted) return
        if (inflight.current === ctl) inflight.current = null
        // A failed refresh keeps what the panel already shows: the error is reported beside it,
        // and the content is not replaced by a loading line until the next tick succeeds.
        setState((s) => ({ data: s.data, error: e instanceof Error ? e.message : String(e), loading: false }))
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, refresh])
  // Unmount: abort and forget the request. Forgetting matters in development, where StrictMode
  // mounts, unmounts and mounts again: the second mount found the aborted controller still set,
  // took it for a request in flight, and never fetched, so every panel said "Loading…" for good.
  useEffect(() => () => { inflight.current?.abort(); inflight.current = null }, [])
  return state
}
