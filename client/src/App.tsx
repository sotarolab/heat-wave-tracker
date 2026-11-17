import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import type { Unit } from './api/types'
import { ForecastPane } from './components/ForecastPane'
import { MapPane } from './components/MapPane'
import { SatellitePane } from './components/SatellitePane'
import { StationsPane } from './components/StationsPane'
import { useFetch } from './hooks/useFetch'

const TABS = [
  { id: 'forecast', label: 'Forecast' },
  { id: 'map', label: 'Map' },
  { id: 'stations', label: 'Stations' },
  { id: 'sat', label: 'Satellite' },
] as const
type Tab = (typeof TABS)[number]['id']
/** The forecast changes every six hours and the imagery every five minutes, so five minutes it is. */
const REFRESH_MS = 5 * 60_000
const UNIT_KEY = 'us.unit'

const STATION_KEY = 'us.station'
const stored = (key: string): string | null => {
  try {
    return localStorage.getItem(key)
  } catch {
    return null // storage can be blocked; the page works without it
  }
}
const store = (key: string, value: string) => {
  try { localStorage.setItem(key, value) } catch { /* see stored */ }
}

/** The hash carries the pane and the station, so #forecast?s=KLAX and #map?s=KLAX open exactly that and a link to it works. */
function readHash(): { tab: Tab; station: string; picked: boolean } {
  const [raw, query] = location.hash.slice(1).split('?')
  const s = new URLSearchParams(query ?? '').get('s')
  return { tab: TABS.find((t) => t.id === raw)?.id ?? 'forecast', station: (s ?? stored(STATION_KEY) ?? 'KDCA').toUpperCase(), picked: s != null }
}
const readUnit = (): Unit => (stored(UNIT_KEY) === 'C' ? 'C' : 'F')

export default function App() {
  const [start] = useState(readHash)
  const [tab, setTab] = useState<Tab>(start.tab)
  const [station, setStation] = useState(start.station)
  const [unit, setUnit] = useState<Unit>(readUnit)
  const [tick, setTick] = useState(0)
  const refresh = useCallback(() => setTick((n) => n + 1), [])
  useEffect(() => {
    const id = setInterval(() => { if (!document.hidden) refresh() }, REFRESH_MS)
    return () => clearInterval(id)
  }, [refresh])
  // A station clicked on the map stays on the map, with its forecast drawn underneath.
  const [onMap, setOnMap] = useState<string | null>(start.tab === 'map' && start.picked ? start.station : null)
  const pickOnMap = useCallback((id: string) => { setOnMap(id); setStation(id) }, [])
  useEffect(() => {
    const pick = tab === 'forecast' ? station : tab === 'map' ? onMap : null
    history.replaceState(null, '', `#${tab}${pick ? `?s=${pick}` : ''}`)
    store(STATION_KEY, station)
  }, [tab, station, onMap])
  const openStation = useCallback((id: string) => { setStation(id); setTab('forecast') }, [])
  const onUnit = useCallback((u: Unit) => {
    setUnit(u)
    store(UNIT_KEY, u)
  }, [])
  const manifest = useFetch((s) => api.manifest(s), [], tick)
  const net = useFetch((s) => api.stations(unit, s), [unit])
  const list = [...(net.data?.stations ?? [])].sort((a, b) => a.name.localeCompare(b.name))
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
            ? <span className={`stamp live${m.live ? '' : ' stale'}`} title="The forecast cycle on this page. It refreshes every six hours.">{m.label} · {m.badge}</span>
            : <span className="stamp">{manifest.error ?? 'Reading the forecast cycle…'}</span>}
          <button className="acct" onClick={refresh} title="Refetch every panel now; they also refresh every five minutes while the tab is visible">Refresh</button>
        </div>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={`tab${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      {tab === 'forecast' && (
        <div className="context">
          <label className="picker">
            <span className="eyebrow">Station</span>
            <select value={station} onChange={(e) => setStation(e.target.value)}>
              {list.length === 0 && <option>{station}</option>}
              {list.map((s) => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
            </select>
          </label>
          <div className="seg">
            {(['F', 'C'] as const).map((u) => <button key={u} className={u === unit ? 'on' : undefined} onClick={() => onUnit(u)}>°{u}</button>)}
          </div>
        </div>
      )}
      {tab === 'forecast' && <ForecastPane station={station} unit={unit} tick={tick} />}
      {tab === 'map' && <MapPane unit={unit} onUnit={onUnit} selected={onMap} onStation={pickOnMap} tick={tick} />}
      {tab === 'map' && onMap && <ForecastPane station={onMap} unit={unit} tick={tick} />}
      {tab === 'stations' && <StationsPane unit={unit} onSelect={openStation} tick={tick} />}
      {tab === 'sat' && <SatellitePane />}
      <footer>
        Forecast: NOAA GFS; HRRR and NBM by way of Open-Meteo. Stations: the ASOS network, by way of the Iowa Environmental Mesonet. Radar and satellite: NWS, NOAA
        STAR and the Iowa Environmental Mesonet. Not a warning product: for heat or any other hazard, check weather.gov.
      </footer>
    </div>
  )
}
