// gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers, hour by hour.
//
// v2 (18 Nov 2026, tuning round 2). The model, fitted at run time from the case
// file's own comparables (past big launches' hourly SullyGnome counts, 14 days
// before to 3 days after their release day):
//
//   For each hour h = 2..24, the log-ratio ln(viewers_h / viewers_1) is regressed on the
//   launch's own scale, ln(viewers_1), with a quadratic term:
//
//       ln(v_h / v_1)  ~  a_h + b_h * x + c_h * x^2,   x = ln(v_1) - 10
//
//   The scale is the whole story. Small launches (hour 1 in the hundreds) grow 10-100x
//   through the day as regions unlock; the biggest launches (hour 1 near a million) are
//   already at their peak and stay flat or fade. The quadratic term is what lets the
//   curve saturate at the top end instead of extrapolating a straight line off a cliff.
//
//   Three changes vs v1 (each validated on practice p00-p03 AND leave-one-out over
//   all 41 launch curves; v1 practice 0.2174 / LOO 0.3772 -> v2 0.2043 / 0.3677,
//   big-launch LOO 0.1690 -> 0.1352, and the gain holds in 25/30 bootstrap resamples
//   of the fit pool):
//
//   1. Ridge (lambda 5) on the slope and quadratic coefficients. With only ~40
//      comparables the OLS quadratic is jittery at the top end; the ridge shrinks it
//      slightly toward the linear fit and stabilises extrapolation past the biggest
//      observed hour-1 (875k) — exactly where GTA VI will land.
//   2. Pooled residual quantiles. With ~37 residuals per hour, the 5%/95% empirical
//      quantiles are the 2nd order statistic — pure noise. The residuals are now
//      normalised per hour by their MAD scale, pooled across all 23 hours (~850
//      points), and the pooled quantiles are rescaled per hour. Same centre, much
//      steadier tails.
//   3. Scale-conditioned spread. Big launches are far more predictable than small
//      ones: the median |residual| is 0.79 for hour-1 under 10k, 0.34 at 10k-200k
//      and 0.15 above 200k (a log|residual| ~ log(h1) slope of -0.18 measured on the
//      comparables). So the spread is multiplied by s = max(0.45,
//      exp(-0.18 * max(0, ln(v_1) - ln(20000)))): full width for small launches,
//      ~half width for a million-viewer launch. The same factor applies to the
//      thin-hour MAD fallback.
//
//   Fallbacks keep the entry valid on thin data: an hour with fewer than 8 usable
//   comparables falls back to the cross-launch median log-ratio for that hour with a
//   MAD spread (also scale-conditioned); if the case holds fewer than 5 usable
//   comparables at all, the entry falls back to the house baseline's persistence
//   forecast (spread 0.15 log units per square-root hour).
//
// Nothing is hard-coded from any one launch: the numbers are re-fitted from whatever
// comparables the case carries (38 in a practice case, 42 in the final, which adds the
// four practice launches). The program is deterministic — the same input bytes give
// the same output bytes — reads only the case file, and uses the standard library only.
//
// Run locally:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The standard normal's quantile at each level (used only by the fallback). */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The fallback's spread, in natural-log units per square-root hour (house baseline). */
const FALLBACK_SIGMA = 0.15;
/** Centre for ln(viewers_1): keeps the quadratic fit well conditioned. */
const LOG_CENTRE = 10;
/** Ridge penalty on the slope and quadratic coefficients (not the intercept). */
const RIDGE = 5;
/** Spread conditioning: residual scale falls with launch size, slope measured on the comparables. */
const SPREAD_REF = 20000;
const SPREAD_SLOPE = 0.18;
const SPREAD_FLOOR = 0.45;
/** A per-hour regression needs at least this many comparables; below it, use the median. */
const MIN_PER_HOUR = 8;
/** Below this many usable comparables overall, fall back to persistence entirely. */
const MIN_COMPARABLES = 5;

interface KnownRow {
  hour: number;
  viewers: number;
}

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  known: KnownRow[];
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** x rounded to 2 decimals (half up), as the house baseline writes it. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The median of a non-empty list of numbers. */
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** The p-quantile of a sorted list, linear interpolation on the order statistics. */
const quantileOfSorted = (s: number[], p: number): number => {
  if (s.length === 1) return s[0];
  const idx = p * (s.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
};

/** Solve a 3x3 linear system by Gaussian elimination with partial pivoting. */
const solve3 = (m: number[][], v: number[]): number[] | null => {
  const a = m.map((row) => [...row]);
  const b = [...v];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    if (Math.abs(a[piv][col]) < 1e-12) return null;
    [a[col], a[piv]] = [a[piv], a[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = a[r][col] / a[col][col];
      for (let cc = 0; cc < 3; cc++) a[r][cc] -= f * a[col][cc];
      b[r] -= f * b[col];
    }
  }
  return [b[0] / a[0][0], b[1] / a[1][1], b[2] / a[2][2]];
};

/** Ordinary least squares of y on [1, x, x^2] with a ridge penalty on the slope
 *  and quadratic coefficients; null when the system is singular. */
const olsQuadratic = (xs: number[], ys: number[]): number[] | null => {
  const s = [0, 0, 0, 0, 0];
  const t = [0, 0, 0];
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    const p = [1, x, x * x, x * x * x, x * x * x * x];
    for (let k = 0; k < 5; k++) s[k] += p[k];
    for (let k = 0; k < 3; k++) t[k] += p[k] * ys[i];
  }
  return solve3(
    [
      [s[0], s[1], s[2]],
      [s[1], s[2] + RIDGE, s[3]],
      [s[2], s[3], s[4] + RIDGE],
    ],
    t,
  );
};

/** The spread scale for a launch of hour-1 size p1: big launches are steadier. */
const spreadScale = (p1: number): number =>
  Math.max(SPREAD_FLOOR, Math.exp(-SPREAD_SLOPE * Math.max(0, Math.log(p1) - Math.log(SPREAD_REF))));

/** 3-point moving average over hours 2..24 (endpoints average their two neighbours). */
const smoothHours = (vals: Map<number, number>): Map<number, number> => {
  const out = new Map<number, number>();
  for (let h = 2; h <= 24; h++) {
    const neigh = [h - 1, h, h + 1].filter((hh) => vals.has(hh)).map((hh) => vals.get(hh) as number);
    out.set(h, neigh.reduce((a, b) => a + b, 0) / neigh.length);
  }
  return out;
};

/** A launch-day curve: viewers by hour (1..24) of the comparable's UTC release day. */
const dayCurve = (c: Comparable): Map<number, number> => {
  const day = c.day_start.slice(0, 10);
  const out = new Map<number, number>();
  for (const pt of c.hourly ?? []) {
    if (typeof pt.from !== 'string' || pt.from.slice(0, 10) !== day) continue;
    const hour = parseInt(pt.from.slice(11, 13), 10) + 1;
    if (hour >= 1 && hour <= 24 && typeof pt.viewers === 'number' && pt.viewers > 0) {
      out.set(hour, pt.viewers);
    }
  }
  return out;
};

/** The persistence fallback: every hour stays at hour 1's print, widening spread. */
const fallbackQuantiles = (p1: number, h: number, levels: number[]): number[] =>
  levels.map((_tau, j) => {
    const z = Z[j] ?? 0;
    return round2(p1 * Math.exp(z * FALLBACK_SIGMA * Math.sqrt(h - 1)));
  });

/** The forecast for one case: {hour: [q at each level]} for hours 2..24. */
const forecastCase = (c: CaseFile): Map<number, number[]> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const levels = c.levels;

  // Usable comparables: a real launch (not the target's own pre-release history)
  // with an hour-1 print and at least half of hours 2..24 present. A comparable
  // missing a few hours still contributes to the hours it has.
  const curves: Array<{ h1: number; byHour: Map<number, number> }> = [];
  for (const comp of c.comparables ?? []) {
    if (typeof comp.unlock === 'string' && comp.unlock.startsWith('Not a launch')) continue;
    if (typeof comp.name === 'string' && comp.name.includes('before release')) continue;
    const byHour = dayCurve(comp);
    const h1 = byHour.get(1);
    if (h1 === undefined || h1 <= 0) continue;
    let present = 0;
    for (let h = 2; h <= 24; h++) if (byHour.has(h)) present++;
    if (present >= 12) curves.push({ h1, byHour });
  }
  if (curves.length < MIN_COMPARABLES) {
    const out = new Map<number, number[]>();
    for (const { hour } of c.forecast) out.set(hour, fallbackQuantiles(p1, hour, levels));
    return out;
  }

  const x0 = Math.log(p1) - LOG_CENTRE;
  const sFactor = spreadScale(p1);
  const medians = new Map<number, number>();
  const offsets = new Map<number, number[]>();
  // Per-hour centred residuals and their MAD scale, for the pooled spread.
  const centredByHour = new Map<number, number[]>();
  const scaleByHour = new Map<number, number>();
  for (let h = 2; h <= 24; h++) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const { h1, byHour } of curves) {
      const v = byHour.get(h);
      if (v !== undefined && v > 0) {
        xs.push(Math.log(h1) - LOG_CENTRE);
        ys.push(Math.log(v / h1));
      }
    }
    if (xs.length >= MIN_PER_HOUR) {
      const beta = olsQuadratic(xs, ys);
      if (beta !== null) {
        const resid = ys.map((y, i) => y - (beta[0] + beta[1] * xs[i] + beta[2] * xs[i] * xs[i]));
        const shift = median(resid);
        medians.set(h, beta[0] + beta[1] * x0 + beta[2] * x0 * x0 + shift);
        const centred = resid.map((r) => r - shift);
        centredByHour.set(h, centred);
        const mid = median(centred);
        const mad = median(centred.map((r) => Math.abs(r - mid)));
        scaleByHour.set(h, Math.max(0.05, 1.4826 * mad));
        continue;
      }
    }
    // Thin hour: the cross-launch median log-ratio, with a MAD spread.
    const ratios = ys;
    const mu = median(ratios);
    const mad = median(ratios.map((r) => Math.abs(r - mu)));
    const sig = Math.max(0.1, 1.4826 * mad);
    medians.set(h, mu);
    offsets.set(h, levels.map((_tau, j) => (Z[j] ?? 0) * sig * sFactor));
  }

  // Pooled spread: normalise each regression hour's residuals by its MAD scale,
  // pool them across hours (~850 points), take the pooled quantiles, and rescale
  // per hour. Steadier tails than per-hour empirical quantiles from ~37 points.
  const pooled: number[] = [];
  for (const [h, centred] of centredByHour) {
    const sc = scaleByHour.get(h) as number;
    for (const r of centred) pooled.push(r / sc);
  }
  pooled.sort((a, b) => a - b);
  for (const [h] of centredByHour) {
    const sc = scaleByHour.get(h) as number;
    offsets.set(
      h,
      levels.map((tau) => quantileOfSorted(pooled, tau) * sc * sFactor),
    );
  }

  const smoothMed = smoothHours(medians);
  const smoothOff = new Map<number, number[]>();
  for (let j = 0; j < levels.length; j++) {
    const col = new Map<number, number>();
    for (let h = 2; h <= 24; h++) col.set(h, (offsets.get(h) as number[])[j]);
    const sm = smoothHours(col);
    for (let h = 2; h <= 24; h++) {
      const row = smoothOff.get(h) ?? new Array<number>(levels.length).fill(0);
      row[j] = sm.get(h) as number;
      smoothOff.set(h, row);
    }
  }

  const out = new Map<number, number[]>();
  for (const { hour } of c.forecast) {
    const mu = smoothMed.get(hour) as number;
    const off = smoothOff.get(hour) as number[];
    let qs = off.map((o) => round2(Math.min(1e9, Math.max(0.01, p1 * Math.exp(mu + o)))));
    for (let j = 1; j < qs.length; j++) qs[j] = Math.max(qs[j], qs[j - 1]);
    out.set(hour, qs);
  }
  return out;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const forecast = forecastCase(c);
  const hours = c.forecast.map(({ hour }) => ({ hour, q: forecast.get(hour) as number[] }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
