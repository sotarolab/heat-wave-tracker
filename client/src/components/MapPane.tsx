import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { Legend, Unit } from '../api/types'
import { useFetch } from '../hooks/useFetch'
import { dotColor } from '../lib/ramp'
import { FieldMap } from './FieldMap'
import type { Dot, Weather } from './FieldMap'

const WEATHER: { id: Weather; label: string }[] = [
  { id: 'off', label: 'Off' }, { id: 'radar', label: 'Radar' }, { id: 'ir', label: 'Infrared' }, { id: 'vis', label: 'Visible' },
]
const NO_DOTS: Dot[] = []

function Scale({ legend, unit }: { legend: Legend; unit: Unit }) {
  if (legend.kind === 'categories') {
    return (
      <div className="swatches">
        {legend.items.map((c) => <span key={c.label}><i style={{ background: c.color }} />{c.label}{c.from != null && <span className="num"> {c.from}°{unit}+</span>}</span>)}
      </div>
    )
  }
  return (
    <div className="scale">
      <span className="stamp">{legend.min}°{unit}</span>
      <div className="ramp" style={{ background: `linear-gradient(to right, ${legend.stops.join(', ')})` }} />
      <span className="stamp">{legend.max}°{unit}</span>
    </div>
  )
}

/** The forecast over the lower 48: one GFS field at a time, stepped through the next five days, with
 *  every station on it in the same colours, and radar or satellite over it for what is there now. */
export function MapPane({ unit, onUnit, onStation, tick }: { unit: Unit; onUnit: (u: Unit) => void; onStation: (id: string) => void; tick: number }) {
  const meta = useFetch((s) => api.fieldMeta(unit, s), [unit], tick)
  const net = useFetch((s) => api.stations(unit, s), [unit], tick)
  const [varId, setVarId] = useState('t2m')
  // null follows the variable's own opening frame, so a refresh keeps a viewer of "now" on now.
  const [frame, setFrame] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const [weather, setWeather] = useState<Weather>('off')
  const [showDots, setShowDots] = useState(true)
  const [selected, setSelected] = useState<string | null>(null)

  const variable = meta.data?.variables.find((v) => v.id === varId) ?? meta.data?.variables[0]
  const count = variable?.frames.length ?? 0
  const start = variable?.start ?? 0
  const i = Math.max(0, Math.min(frame ?? start, count - 1))
  useEffect(() => {
    if (!playing || count < 2) return
    const id = setInterval(() => setFrame((f) => ((f ?? start) + 1) % count), 700)
    return () => clearInterval(id)
  }, [playing, count, start])

  // While a change of unit is in flight the map keeps the old numbers; they keep the old unit's label
  // (`shown`), and the dots wait until they are in the same unit as the scale they are coloured by.
  const shown = meta.data?.unit ?? unit
  const values = variable && net.data?.unit === shown ? net.data.values[variable.id] : undefined
  const dots = useMemo(() => {
    if (!showDots || !net.data || !values || !variable) return NO_DOTS
    return net.data.stations.map((s, k) => {
      const v = values[k]?.[i] ?? null
      return { id: s.id, lat: s.lat, lon: s.lon, color: dotColor(v, variable.legend), label: `${s.name} (${s.id}) · ${v == null ? 'no value' : `${v}°${shown}`}` }
    })
  }, [showDots, net.data, values, variable, i, shown])

  if (!meta.data || !variable) return <section className="panel"><p className="todo">{meta.error ?? 'Reading the forecast…'}</p></section>
  const pick = (id: string) => { setVarId(id); setFrame(null); setPlaying(false) }
  const k = selected && net.data ? net.data.stations.findIndex((s) => s.id === selected) : -1
  const at = k >= 0 && net.data ? net.data.stations[k] : null
  const atValue = k >= 0 ? values?.[k]?.[i] ?? null : null
  return (
    <section className="panel">
      <div className="panel-h">
        <h2>Forecast: {variable.label.toLowerCase()}</h2>
        <div className="row">
          <div className="seg">
            {meta.data.variables.map((v) => <button key={v.id} className={v === variable ? 'on' : undefined} onClick={() => pick(v.id)}>{v.label}</button>)}
          </div>
          <div className="seg">
            {(['F', 'C'] as const).map((u) => <button key={u} className={u === unit ? 'on' : undefined} onClick={() => onUnit(u)}>°{u}</button>)}
          </div>
        </div>
      </div>
      <div className="row" style={{ marginBottom: 10 }}>
        <span className="eyebrow">Now, over it</span>
        <div className="seg">
          {WEATHER.map((w) => <button key={w.id} className={w.id === weather ? 'on' : undefined} onClick={() => setWeather(w.id)}>{w.label}</button>)}
        </div>
        <label className="toggle stamp"><input type="checkbox" checked={showDots} onChange={(e) => setShowDots(e.target.checked)} /> Stations</label>
        {net.error && <span className="stamp">Stations unavailable: {net.error}</span>}
      </div>
      <FieldMap image={api.fieldUrl(variable.id, i, meta.data.cycle)} bounds={meta.data.bounds} opacity={meta.data.opacity}
        dots={dots} selected={selected} onSelect={setSelected} weather={weather} tick={tick} />
      <div className="row" style={{ marginTop: 10 }}>
        <button className="acct" disabled={count < 2} onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
        <input className="frames" type="range" min={0} max={count - 1} value={i} aria-label="Forecast time"
          onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)) }} />
        <span className="stamp">{variable.frames[i]?.label}{meta.error && ` · refresh failed: ${meta.error}`}</span>
      </div>
      <Scale legend={variable.legend} unit={shown} />
      {at && (
        <p className="headline">
          {at.name}: <span className="num">{atValue == null ? 'no value' : `${atValue}°${shown}`}</span>
          <span className="note"> {variable.label.toLowerCase()}, {variable.frames[i]?.label} </span>
          <button className="acct" onClick={() => onStation(at.id)}>Open its forecast</button>
        </p>
      )}
      <p className="note">
        {variable.label}: {variable.note}. One model, GFS, at a quarter of a degree; each station shows the grid point nearest it.
        Radar and satellite are what is there now, whatever forecast time the slider is on. Click a station for its value, then open its forecast.
      </p>
    </section>
  )
}
