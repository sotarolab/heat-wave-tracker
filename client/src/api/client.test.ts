import { afterEach, expect, test, vi } from 'vitest'
import { get, RETRY_MS } from './client'

const reply = (status: number, body: unknown = {}) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })

/** Run a request to its end with the retry waits skipped. */
async function run<T>(p: Promise<T>): Promise<T | Error> {
  const settled = p.catch((e: Error) => e)
  await vi.runAllTimersAsync()
  return settled
}

test('a gateway error is asked again, and the answer that comes is returned', async () => {
  vi.useFakeTimers()
  const fetch = vi.fn().mockResolvedValueOnce(reply(502)).mockResolvedValueOnce(reply(200, { ok: true }))
  vi.stubGlobal('fetch', fetch)
  expect(await run(get('/api/manifest'))).toEqual({ ok: true })
  expect(fetch).toHaveBeenCalledTimes(2)
})

test('it gives up after the last wait, with the reason', async () => {
  vi.useFakeTimers()
  const fetch = vi.fn().mockImplementation(() => Promise.resolve(reply(502)))
  vi.stubGlobal('fetch', fetch)
  const out = await run(get('/api/manifest'))
  expect(out).toBeInstanceOf(Error)
  expect(fetch).toHaveBeenCalledTimes(RETRY_MS.length + 1)
})

test('an answer from the service itself is not asked again', async () => {
  vi.useFakeTimers()
  const fetch = vi.fn().mockResolvedValue(reply(404, { detail: 'No such station: NOPE' }))
  vi.stubGlobal('fetch', fetch)
  const out = await run(get('/api/station/NOPE'))
  expect((out as Error).message).toBe('No such station: NOPE')
  expect(fetch).toHaveBeenCalledTimes(1)
})

test('a dropped connection is asked again', async () => {
  vi.useFakeTimers()
  const fetch = vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch')).mockResolvedValueOnce(reply(200, [1]))
  vi.stubGlobal('fetch', fetch)
  expect(await run(get('/api/stations'))).toEqual([1])
})
