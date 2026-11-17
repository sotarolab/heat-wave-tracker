import { useMemo, useState } from 'react'
import { api } from '../api/client'
import type { Unit } from '../api/types'
import { useFetch } from '../hooks/useFetch'
import { Chart } from './Chart'

/** The charts with a correction on them share the large frame; the rest are read side by side under it. */
const MAIN = ['t2m', 'hi']

/** One station: what it has measured, what the models said, the forecast corrected two ways, and how
 *  each has verified. Every number and sentence is the service's; this pane lays them out. */
export function ForecastPane({ station, unit, tick }: { station: string; unit: Unit; tick: number }) {
  const f = useFetch((s) => api.station(station, unit, s), [station, unit], tick)
  const [metric, setMetric] = useState('t2m')
  const page = f.data?.station.id === station && f.data.unit === unit ? f.data : null // not the last station's reply while this one loads
  const main = useMemo(() => page?.charts.filter((c) => MAIN.includes(c.id)) ?? [], [page])
  const rest = useMemo(() => page?.charts.filter((c) => !MAIN.includes(c.id)) ?? [], [page])
  const chart = main.find((c) => c.id === metric) ?? main[0]

  if (!page || !chart) return <section className="panel"><p className="todo">{f.error ?? `Reading the forecast for ${station}…`}</p></section>
  const { today, model } = page.verification
  const deg = `°${unit}`
  return (
    <>
      <section className="panel">
        <div className="panel-h">
          <h2>{page.station.name} ({page.station.id})</h2>
          <div className="seg">
            {main.map((c) => <button key={c.id} className={c === chart ? 'on' : undefined} onClick={() => setMetric(c.id)}>{c.label}</button>)}
          </div>
        </div>
        <p className="headline">{page.headline}</p>
        <div className="tiles">
          {page.tiles.map((t) => <div className="tile" key={t.label}><span className="eyebrow">{t.label}</span><b className="num">{t.value}</b><span className="note">{t.note}</span></div>)}
        </div>
        <div className="vchart"><Chart time={page.time} lines={chart.lines} unit={chart.unit} tz={page.station.tz} now={page.now} /></div>
        <p className="note">
          The dashed grey line is GFS as issued, at the grid point nearest the station. The white line is what the station measured.
          {' '}<b>Corrected with today's observations</b> shifts GFS by its mean error so far today, for the next {page.same_day_hours} hours; the shaded band
          is a 95% prediction interval from today's residuals. <b>Learned correction</b> is a gradient-boosted model trained on the archive
          of past forecasts and observations, with a calibrated band; it is logged each cycle before the observations arrive
          {page.has_learned ? '' : ', and has no rows for this cycle'}. HRRR and NBM are the latest runs, raw, for the hours ahead.
          Times are the station's own. The dashed rule is now.{f.error && ` Refresh failed: ${f.error}.`}
        </p>
      </section>
      {rest.length > 0 && (
        <section className="panel">
          <div className="panel-h"><h2>The weather around it</h2><span className="stamp">observed, and GFS, HRRR and NBM for the hours ahead</span></div>
          <div className="pgrid">
            {rest.map((c) => (
              <div className="vchart" key={c.id}>
                <h3>{c.label}<span className="stamp">{c.unit.trim()}</span></h3>
                <Chart time={page.time} lines={c.lines} unit={c.unit} tz={page.station.tz} now={page.now} height={200} floor={c.floor} ceil={c.ceil} />
              </div>
            ))}
          </div>
          <p className="note">
            The same clock as the chart above. Dew point and humidity say how much the air can cool at night and how the heat feels;
            wind and cloud say whether the day's heating and the night's cooling get their way; pressure and precipitation place the
            station in the pattern passing over it. Precipitation is a running total: what the station measured, then each model's
            amount added on from now. Gusts are shown as points where one was reported.
          </p>
        </section>
      )}
      <section className="panel">
        <div className="panel-h"><h2>How it is verifying</h2><span className="stamp">observed minus forecast, {deg}</span></div>
        {today.length > 0 ? (
          <table className="grid">
            <thead><tr><th>Today, raw GFS at this station</th><th className="r">Steps</th><th className="r">Mean error</th><th className="r">RMSE</th><th className="r">95% interval</th></tr></thead>
            <tbody>
              {today.map((r) => (
                <tr key={r.metric}>
                  <td>{r.metric}</td><td className="r num">{r.n}</td><td className="r num">{r.bias > 0 ? '+' : ''}{r.bias}</td>
                  <td className="r num">{r.rmse}</td><td className="r num">{r.interval == null ? '–' : `±${r.interval}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="todo">{page.stale ? 'The forecast on this page has ended.' : `Too few forecast steps have verified today to score it yet: ${page.min_steps} are needed.`}</p>}
        {model ? (
          <>
            <table className="grid" style={{ marginTop: 14 }}>
              <thead><tr><th>All stations, last {model.days} days, scored after the fact</th><th className="r">Pairs</th><th className="r">RMSE</th></tr></thead>
              <tbody>
                <tr><td><i className="key" style={{ background: 'var(--raw)' }} />GFS, raw</td><td className="r num">{model.n.toLocaleString('en-US')}</td><td className="r num">{model.rmse_raw}</td></tr>
                <tr><td><i className="key" style={{ background: 'var(--series-4)' }} />Station's trailing offset</td><td className="r num">{model.n.toLocaleString('en-US')}</td><td className="r num">{model.rmse_offset}</td></tr>
                <tr className="best"><td><i className="key" style={{ background: 'var(--ml)' }} />Learned correction ({model.version})</td><td className="r num">{model.n.toLocaleString('en-US')}</td><td className="r num">{model.rmse_ml}</td></tr>
              </tbody>
            </table>
            <p className="note">
              The same forecast and observation pairs for all three rows, each observed after its forecast was logged.
              {model.band_coverage_pct != null && ` The model's band, drawn to hold 95% of outcomes, held ${model.band_coverage_pct}%.`}
            </p>
          </>
        ) : <p className="note">The scored record of the learned correction is read from the forecast archive, which is not reachable right now.</p>}
      </section>
    </>
  )
}
