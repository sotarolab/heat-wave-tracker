import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

export interface Dot { id: string; lat: number; lon: number; label: string; color: string }
export type Bounds = [[number, number], [number, number]]
export type Weather = 'off' | 'radar' | 'ir' | 'vis'
export const CONUS: Bounds = [[24, -125], [50, -66.5]]

// Esri's tile route is {z}/{y}/{x}, row before column.
const ESRI = 'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas'
const IEM = 'https://mesonet.agron.iastate.edu'

/** Radar and GOES composites are published every five minutes, so a five-minute stamp in the address
 *  is what makes a redraw fetch the new frame and lets the cache serve the old one in between. */
export const stamp = () => Math.floor(Date.now() / 300_000)

function weatherLayer(kind: Exclude<Weather, 'off'>): L.TileLayer {
  if (kind === 'radar') {
    return L.tileLayer(`${IEM}/cache/tile.py/1.0.0/nexrad-n0q-900913/{z}/{x}/{y}.png?t={t}`, {
      opacity: 0.7, attribution: 'Radar: IEM NEXRAD', maxNativeZoom: 12, pane: 'wx', t: stamp(),
    } as L.TileLayerOptions)
  }
  const [service, layers] = kind === 'ir' ? ['conus_ir', 'conus_ir_4km'] : ['conus_vis', 'conus_vis_1km']
  return L.tileLayer.wms(`${IEM}/cgi-bin/wms/goes/${service}.cgi`, {
    layers, format: 'image/png', transparent: true, uppercase: true, opacity: 0.6, attribution: 'Satellite: IEM GOES', pane: 'wx', t: stamp(),
  } as L.WMSOptions)
}

/** The national map: the forecast field as one image the service drew, radar or satellite over it,
 *  place names over those, and one dot per station on top. The map only places what it is given. */
export function FieldMap({ image, bounds, opacity, dots, selected, onSelect, weather, tick }: {
  /** Address of the field image for the frame on show, or null for a bare map. */
  image: string | null
  bounds: Bounds
  opacity: number
  dots: Dot[]
  selected: string | null
  onSelect: (id: string) => void
  weather: Weather
  /** The app's refresh counter: the weather overlay moves to the newest frame when it changes. */
  tick: number
}) {
  const host = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const dotLayer = useRef<L.LayerGroup | null>(null)
  const shown = useRef<L.ImageOverlay | null>(null)
  useEffect(() => {
    // No wheel zoom, and no drag on a phone: the map is most of the screen there and would catch every scroll past it.
    const m = L.map(host.current!, { scrollWheelZoom: false, dragging: !L.Browser.mobile, minZoom: 3, maxZoom: 10, zoomSnap: 0.25 })
    L.tileLayer(`${ESRI}/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, { maxNativeZoom: 13, attribution: 'Esri, HERE, Garmin, &copy; OpenStreetMap contributors' }).addTo(m)
    // Above the basemap (200): the field, then the weather, then place names, all under the dots (400).
    m.createPane('field').style.zIndex = '300'
    m.createPane('wx').style.zIndex = '350'
    const names = m.createPane('names')
    names.style.zIndex = '380'
    names.style.pointerEvents = 'none'
    L.tileLayer(`${ESRI}/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, { maxNativeZoom: 13, pane: 'names' }).addTo(m)
    dotLayer.current = L.layerGroup().addTo(m)
    m.fitBounds(CONUS)
    map.current = m
    return () => { m.remove(); map.current = dotLayer.current = shown.current = null }
  }, [])
  useEffect(() => {
    const m = map.current
    if (!m) return
    if (!image) {
      if (shown.current) m.removeLayer(shown.current)
      shown.current = null
      return
    }
    // The next frame loads unseen and replaces the one on screen only once it is in, so stepping
    // through frames does not flash the bare map between them.
    const next = L.imageOverlay(image, bounds, { opacity: 0, pane: 'field', interactive: false }).addTo(m)
    const swap = () => {
      next.setOpacity(opacity)
      if (shown.current && shown.current !== next) m.removeLayer(shown.current)
      shown.current = next
    }
    next.once('load', swap)
    return () => { next.off('load', swap); if (shown.current !== next) m.removeLayer(next) }
  }, [image, bounds, opacity])
  useEffect(() => {
    const m = map.current
    if (!m || weather === 'off') return
    const layer = weatherLayer(weather).addTo(m)
    return () => { m.removeLayer(layer) }
  }, [weather, tick])
  useEffect(() => {
    const group = dotLayer.current
    if (!group) return
    group.clearLayers()
    for (const d of dots) {
      const on = d.id === selected
      L.circleMarker([d.lat, d.lon], {
        radius: on ? 8 : 5, weight: on ? 2 : 1, color: on ? '#f4f1ea' : 'rgba(15, 17, 20, .8)', fillColor: d.color, fillOpacity: 1, interactive: false,
      }).addTo(group)
      // The pointer target is wider than the dot, and the label goes in as text, never as HTML.
      const tip = document.createElement('span')
      tip.textContent = d.label
      L.circleMarker([d.lat, d.lon], { radius: 11, stroke: false, fillOpacity: 0 }).bindTooltip(tip).on('click', () => onSelect(d.id)).addTo(group)
    }
  }, [dots, selected, onSelect])
  return <div className="map" ref={host} />
}
