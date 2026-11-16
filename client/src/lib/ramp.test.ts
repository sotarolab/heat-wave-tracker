import { expect, test } from 'vitest'
import type { Legend } from '../api/types'
import { dotColor, NO_VALUE } from './ramp'

const ramp: Legend = { kind: 'ramp', stops: ['#000000', '#808080', '#ffffff'], min: 0, max: 100 }
const risk: Legend = {
  kind: 'categories',
  items: [
    { label: 'No Elevated Risk', color: '#0ca30c', from: null },
    { label: 'Caution', color: '#f5c451', from: 80 },
    { label: 'Extreme Caution', color: '#f2994a', from: 90 },
  ],
}

test('a ramp interpolates between stops and clamps past its ends', () => {
  expect(dotColor(0, ramp)).toBe('rgb(0, 0, 0)')
  expect(dotColor(50, ramp)).toBe('rgb(128, 128, 128)')
  expect(dotColor(25, ramp)).toBe('rgb(64, 64, 64)')
  expect(dotColor(100, ramp)).toBe('rgb(255, 255, 255)')
  expect(dotColor(-40, ramp)).toBe('rgb(0, 0, 0)')
  expect(dotColor(400, ramp)).toBe('rgb(255, 255, 255)')
})

test('categories take the highest bound reached, and the first item below all of them', () => {
  expect(dotColor(79.9, risk)).toBe('#0ca30c')
  expect(dotColor(80, risk)).toBe('#f5c451')
  expect(dotColor(104, risk)).toBe('#f2994a')
})

test('a missing value has its own colour', () => {
  expect(dotColor(null, ramp)).toBe(NO_VALUE)
  expect(dotColor(Number.NaN, risk)).toBe(NO_VALUE)
})
