import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import type { ChartLine } from '../api/types'

export type Line = ChartLine

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()
const shade = (hex: string, alpha: number) => `rgba(${[1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16)).join(', ')}, ${alpha})`

/** Lines over time on one axis, in the station's own time zone. The legend under the plot is the readout:
 *  it lists every line at the crosshair. A dashed rule marks `now`; drag to zoom, double click to reset.
 *  Rebuilt when its data changes, so callers pass memoised `time` and `lines`. */
export function Chart({ time, lines, unit, tz, now, height = 320 }: { time: number[]; lines: Line[]; unit: string; tz: string; now: number; height?: number }) {
  const host = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = host.current
    if (!el || time.length === 0) return
    const clock = new Intl.DateTimeFormat('en-US', { timeZone: tz, hour: 'numeric' })
    const day = new Intl.DateTimeFormat('en-US', { timeZone: tz, weekday: 'short', day: 'numeric' })
    const full = new Intl.DateTimeFormat('en-US', { timeZone: tz, weekday: 'short', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' })
    const hour = new Intl.DateTimeFormat('en-US', { timeZone: tz, hour: 'numeric', hourCycle: 'h23' })
    const axis: uPlot.Axis = { stroke: css('--dim'), font: `11px ${css('--mono')}`, grid: { stroke: css('--rule'), width: 1 }, ticks: { stroke: css('--rule'), width: 1 } }
    // Each band adds two unlabelled series after the lines; uPlot fills between them.
    const series: uPlot.Series[] = [{ label: 'Time', value: (_u, t) => (t == null ? '–' : full.format(t * 1e3)) }]
    const data: (number | null)[][] = [time]
    const bands: uPlot.Band[] = []
    for (const l of lines) {
      series.push({
        label: l.label, stroke: css(l.color), width: l.width ?? 2, dash: l.dash ? [5, 4] : undefined, spanGaps: true,
        points: { show: false }, value: (_u, v) => (v == null ? '–' : `${v}${unit}`),
      })
      data.push(l.values)
    }
    const hidden: number[] = []
    for (const l of lines) {
      if (!l.band) continue
      const at = series.length
      series.push({ stroke: 'transparent', width: 0, spanGaps: true, points: { show: false } }, { stroke: 'transparent', width: 0, spanGaps: true, points: { show: false } })
      data.push(l.band.up, l.band.lo)
      bands.push({ series: [at, at + 1], fill: shade(css(l.color), 0.16) })
      hidden.push(at, at + 1)
    }
    const u = new uPlot({
      width: el.clientWidth,
      height,
      tzDate: (t) => uPlot.tzDate(new Date(t * 1e3), tz),
      cursor: { drag: { x: true, y: false } },
      // The key beside a name is a short stroke of the line's colour, as the line itself is drawn.
      legend: { markers: { width: 0, fill: (_u, i) => (i > 0 && i <= lines.length ? css(lines[i - 1].color) : 'transparent') } },
      axes: [
        { ...axis, values: (_u, ticks) => ticks.map((t) => (hour.format(t * 1e3) === '0' || hour.format(t * 1e3) === '00' ? day.format(t * 1e3) : clock.format(t * 1e3))) },
        { ...axis, size: 54, values: (_u, ticks) => ticks.map((t) => `${t}°`) },
      ],
      series,
      bands,
      hooks: {
        draw: [(p) => {
          const x = p.valToPos(now, 'x', true)
          const { left, top, width, height: h } = p.bbox
          if (x <= left || x >= left + width) return
          p.ctx.save()
          p.ctx.strokeStyle = css('--now')
          p.ctx.lineWidth = devicePixelRatio
          p.ctx.setLineDash([3 * devicePixelRatio, 3 * devicePixelRatio])
          p.ctx.beginPath()
          p.ctx.moveTo(x, top)
          p.ctx.lineTo(x, top + h)
          p.ctx.stroke()
          p.ctx.restore()
        }],
      },
    }, data as uPlot.AlignedData, el)
    // The band's two edges are drawing aids, not lines to read: they stay out of the legend.
    const rows = u.root.querySelectorAll<HTMLElement>('.u-legend .u-series')
    for (const i of hidden) if (rows[i]) rows[i].style.display = 'none'
    const resize = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height }))
    resize.observe(el)
    return () => { resize.disconnect(); u.destroy() }
  }, [time, lines, unit, tz, now, height])
  return <div className="chart" ref={host} />
}
