import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import type { Unit } from './api/types'
import { MapPane } from './components/MapPane'
import { SatellitePane } from './components/SatellitePane'
import { useFetch } from './hooks/useFetch'

const TABS = [
  { id: 'map', label: 'Map' },
  { id: 'sat', label: 'Satellite' },
] as const
type Tab = (typeof TABS)[number]['id']
/** The forecast changes every six hours and the imagery every five minutes, so five minutes it is. */
const REFRESH_MS = 5 * 60_000
const UNIT_KEY = 'us.unit'

/** The hash carries the pane, so a link to #sat opens the satellite loops. */
const readTab = (): Tab => TABS.find((t) => t.id === location.hash.slice(1).split('?')[0])?.id ?? 'map'
const readUnit = (): Unit => {
  try {
    return localStorage.getItem(UNIT_KEY) === 'C' ? 'C' : 'F'
  } catch {
    return 'F' // storage can be blocked; the page works without it
  }
}

export default function App() {
  const [tab, setTab] = useState<Tab>(readTab)
  const [unit, setUnit] = useState<Unit>(readUnit)
  const [tick, setTick] = useState(0)
  const refresh = useCallback(() => setTick((n) => n + 1), [])
  useEffect(() => {
    const id = setInterval(() => { if (!document.hidden) refresh() }, REFRESH_MS)
    return () => clearInterval(id)
  }, [refresh])
  useEffect(() => { history.replaceState(null, '', `#${tab}`) }, [tab])
  const onUnit = useCallback((u: Unit) => {
    setUnit(u)
    try { localStorage.setItem(UNIT_KEY, u) } catch { /* see readUnit */ }
  }, [])
  const manifest = useFetch((s) => api.manifest(s), [], tick)
  const m = manifest.data

  return (
    <div className="wrap">
      <header className="top">
        <div className="brand">
          <h1>US nowcast</h1>
          <span className="sub">station temperature forecasts, checked against what happened</span>
        </div>
        <div className="row">
          {m
            ? <span className={`stamp live${m.live ? '' : ' stale'}`} title="The forecast cycle on this page. It refreshes every six hours.">GFS {m.init.slice(0, 13)}Z · {m.badge}</span>
            : <span className="stamp">{manifest.error ?? 'Reading the forecast cycle…'}</span>}
          <button className="acct" onClick={refresh} title="Refetch every panel now; they also refresh every five minutes while the tab is visible">Refresh</button>
        </div>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={`tab${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      {tab === 'map' && <MapPane unit={unit} onUnit={onUnit} tick={tick} />}
      {tab === 'sat' && <SatellitePane />}
      <footer>
        Forecast: NOAA GFS. Stations: the ASOS network, by way of the Iowa Environmental Mesonet. Radar and satellite: NWS, NOAA
        STAR and the Iowa Environmental Mesonet. Not a warning product: for heat or any other hazard, check weather.gov.
      </footer>
    </div>
  )
}
