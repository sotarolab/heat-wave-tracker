import { useState } from 'react'
import { stamp } from './FieldMap'

/** NOAA STAR's GOES loops and the NWS radar mosaic: animated GIFs their owners refresh every few minutes. */
const BANDS = [
  { id: 'GEOCOLOR', label: 'GeoColor', note: 'true colour by day, infrared with city lights by night' },
  { id: '13', label: 'Infrared', note: 'clean infrared, cloud top temperature: the coldest and deepest cloud is the brightest' },
  { id: '08', label: 'Water vapor', note: 'upper-level water vapor: dry air sinking under a ridge is dark, moist ascent is bright' },
  { id: 'AirMass', label: 'Air mass', note: 'the jet runs along the edge between red, dry stratospheric air and green, moist air' },
] as const
type Band = (typeof BANDS)[number]['id']

/** GOES-East sees the whole lower 48; the two Pacific sectors are sharper from GOES-West. */
const SECTORS = [
  { id: 'CONUS', label: 'Lower 48', sat: 'GOES19' }, { id: 'ne', label: 'Northeast', sat: 'GOES19' },
  { id: 'se', label: 'Southeast', sat: 'GOES19' }, { id: 'umv', label: 'Upper Midwest', sat: 'GOES19' },
  { id: 'sp', label: 'Southern Plains', sat: 'GOES19' }, { id: 'nr', label: 'Northern Rockies', sat: 'GOES19' },
  { id: 'sr', label: 'Southern Rockies', sat: 'GOES19' }, { id: 'pnw', label: 'Pacific Northwest', sat: 'GOES18' },
  { id: 'psw', label: 'Pacific Southwest', sat: 'GOES18' },
] as const
type Sector = (typeof SECTORS)[number]

const STAR = 'https://cdn.star.nesdis.noaa.gov'
const loop = (s: Sector, band: Band) =>
  s.id === 'CONUS'
    ? `${STAR}/${s.sat}/ABI/CONUS/${band}/${s.sat}-CONUS-${band}-625x375.gif`
    : `${STAR}/${s.sat}/ABI/SECTOR/${s.id}/${band}/${s.sat}-${s.id.toUpperCase()}-${band}-600x600.gif`

/** The app's refresh re-renders this pane, and the five-minute stamp in the address then fetches the newest loop. */
export function SatellitePane() {
  const [band, setBand] = useState<Band>('GEOCOLOR')
  const [sector, setSector] = useState<Sector>(SECTORS[0])
  const info = BANDS.find((b) => b.id === band)!
  const t = stamp()
  return (
    <section className="panel">
      <div className="panel-h">
        <h2>Satellite and radar</h2>
        <div className="seg">{SECTORS.map((s) => <button key={s.id} className={s === sector ? 'on' : undefined} onClick={() => setSector(s)}>{s.label}</button>)}</div>
      </div>
      <div className="sky">
        <div>
          <div className="panel-h">
            <span className="stamp">{sector.sat === 'GOES19' ? 'GOES-19 (East)' : 'GOES-18 (West)'}</span>
            <div className="seg">{BANDS.map((b) => <button key={b.id} className={b.id === band ? 'on' : undefined} onClick={() => setBand(b.id)}>{b.label}</button>)}</div>
          </div>
          <div className="sat"><img src={`${loop(sector, band)}?t=${t}`} alt={`${sector.label} ${info.label} loop`} /></div>
          <p className="note">{info.label}: {info.note}. The last few hours, updated every few minutes.</p>
        </div>
        <div>
          <div className="panel-h"><span className="stamp">NWS radar mosaic</span></div>
          <div className="sat"><img src={`https://radar.weather.gov/ridge/standard/CONUS_loop.gif?t=${t}`} alt="National radar loop" /></div>
          <p className="note">Where precipitation is falling now. The last hour or so, updated every few minutes.</p>
        </div>
      </div>
    </section>
  )
}
