import { useMemo, useState } from 'react'
import { api } from '../api/client'
import type { NetworkRow, Unit } from '../api/types'
import { useFetch } from '../hooks/useFetch'

type Key = 'name' | 'observed' | 'forecast' | 'error' | 'high_today' | 'high_tomorrow'
const COLUMNS: { key: Key; label: string }[] = [
  { key: 'name', label: 'Station' }, { key: 'observed', label: 'Observed' }, { key: 'forecast', label: 'GFS for now' },
  { key: 'error', label: 'Error' }, { key: 'high_today', label: 'High today' }, { key: 'high_tomorrow', label: 'High tomorrow' },
]
// One decimal on every row so the columns line up; the values arrive rounded to it.
const show = (v: number | null, signed = false) => (v == null ? '–' : `${signed && v > 0 ? '+' : ''}${v.toFixed(1)}`)

/** Every station in one table: what it reads now beside what GFS said for now, and the next two highs.
 *  Sorted by the size of the miss to begin with, so the stations the model has wrong are at the top. */
export function StationsPane({ unit, onSelect, tick }: { unit: Unit; onSelect: (id: string) => void; tick: number }) {
  const net = useFetch((s) => api.network(unit, s), [unit], tick)
  const [sort, setSort] = useState<{ key: Key; down: boolean }>({ key: 'error', down: true })
  const rows = useMemo(() => {
    const value = (r: NetworkRow) => (sort.key === 'name' ? r.name : sort.key === 'error' ? (r.error == null ? null : Math.abs(r.error)) : r[sort.key])
    return [...(net.data?.rows ?? [])].sort((a, b) => {
      const [x, y] = [value(a), value(b)]
      if (x == null || y == null) return x == null ? (y == null ? 0 : 1) : -1 // stations with no value go last either way
      const d = typeof x === 'string' ? x.localeCompare(y as string) : x - (y as number)
      return sort.down ? -d : d
    })
  }, [net.data, sort])

  if (!net.data) return <section className="panel"><p className="todo">{net.error ?? 'Reading the stations…'}</p></section>
  const n = net.data
  const pick = (key: Key) => setSort((s) => ({ key, down: s.key === key ? !s.down : key !== 'name' }))
  return (
    <section className="panel">
      <div className="panel-h"><h2>Stations now</h2><span className="stamp">°{n.unit}{net.error && ` · refresh failed: ${net.error}`}</span></div>
      <div className="tiles">
        <div className="tile"><span className="eyebrow">Reporting</span><b className="num">{n.reporting} of {n.stations}</b><span className="note">in the last {n.window_h} hours</span></div>
        <div className="tile"><span className="eyebrow">GFS miss now, mean</span><b className="num">{n.mean_abs_error == null ? '–' : `${n.mean_abs_error}°${n.unit}`}</b><span className="note">absolute error across stations</span></div>
      </div>
      {!n.current && <p className="note">The forecast on this page has ended, so there is nothing current to compare with the observations.</p>}
      {n.current && n.reporting === 0 && <p className="note">The observation network is not answering right now; the forecast columns are shown without it.</p>}
      <div className="scroll">
        <table className="grid">
          <thead>
            <tr>{COLUMNS.map((c) => <th key={c.key} className={`sortable${c.key === 'name' ? '' : ' r'}`} onClick={() => pick(c.key)}>{c.label}{sort.key === c.key ? (sort.down ? ' ↓' : ' ↑') : ''}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className={`pick${r.observed == null ? ' dead' : ''}`} onClick={() => onSelect(r.id)}>
                <td>{r.name} <small className="stamp">{r.id}</small></td>
                <td className="r num">{show(r.observed)}</td><td className="r num">{show(r.forecast)}</td><td className="r num">{show(r.error, true)}</td>
                <td className="r num">{show(r.high_today)}</td><td className="r num">{show(r.high_tomorrow)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        Error is observed minus forecast: positive where GFS is too cold. GFS for now is the forecast step nearest the present, at the
        grid point nearest the station. Highs are the raw GFS maximum over each station's own calendar day. Click a station for its forecast.
      </p>
    </section>
  )
}
