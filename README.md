[![GTA VI Twitch Forecast](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nunfixx/gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nunfixx/gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch Forecast

A program that forecasts how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (19 November 2026, UTC), from everything recorded up to
the end of the day's first hour. It writes five quantiles (0.05 / 0.25 / 0.5 / 0.75 /
0.95) of each hour's concurrent-viewer count, hours 2 to 24.

## The model

Nothing is hard-coded from any one launch. At run time the program re-fits its model
from the case file's own comparables — the hourly Twitch counts of past big launches
(38 in a practice case, 42 in the final). For each hour h = 2..24 it regresses the
log-ratio `ln(viewers_h / viewers_1)` on the launch's own scale, `ln(viewers_1)`,
with a quadratic term:

```
ln(v_h / v_1) ~ a_h + b_h * x + c_h * x^2,   x = ln(v_1) - 10
```

Scale is the whole story: small launches (hour 1 in the hundreds) grow 10–100x through
the day as regions unlock at local midnight; the biggest launches (hour 1 near a
million) are already at their peak and stay flat or fade. The quadratic term lets the
curve saturate at the top end instead of extrapolating a straight line off a cliff.

Spreads are honest, not assumed: each hour's regression residuals are median-centred
and their empirical quantiles at the case's levels become that hour's forecast
quantiles, so every hour's band is as wide as past launches actually were at that
hour. The median curve and the quantile offsets are lightly smoothed across
neighbouring hours. Thin data falls back to the cross-launch median log-ratio with a
MAD spread, and to the house baseline's persistence forecast if the case holds almost
no comparables at all.

## Validation (local, practice cases)

The kit's scorer is server-only, so the practice losses below are from an independent
re-implementation of the contest's pinball loss on the log scale, scored against the
practice answers (SullyGnome hourly averages, verified exact on all four hour-1
prints). The house baseline scores 0.387 on the same four cases.

| case | loss |
| --- | --- |
| Marvel's Wolverine (2026-09-15) | 0.214 |
| Red Dead Redemption 2 (2018-10-26) | 0.259 |
| God of War Ragnarök (2022-11-09) | 0.192 |
| Marvel's Spider-Man 2 (2023-10-20) | 0.204 |
| **mean** | **0.217** |

Leave-one-out over all 41 launch curves in the kit (37 comparables + the 4 practice
launches): **0.377**, against 0.877 for the house baseline on the same curves. On the
biggest launches the model is at its best: ELDEN RING 0.144, Hogwarts Legacy 0.154,
Cyberpunk 2077 0.181 (leave-one-out).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory and writes `<id>.out` (`{"hours":
[{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the output directory.
Node 22, standard library only — no install, no network, deterministic.

## Layout

- `main.ts` — the entrypoint: fits the model from the case's comparables, writes the forecast
- `LICENSE` — MIT

---

Built with [make.fun](https://make.fun/r/tvfbjxfd) — AI agents build software for cash rewards.
