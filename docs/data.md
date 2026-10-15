# Data

## Sources

| Source | Provides | Access | Required? |
|---|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | results, bookmaker odds (pre-closing + closing), rolling fixtures, second tiers | free CSV, no auth | **yes** |
| [Understat](https://understat.com) via [`soccerdata`](https://github.com/probberechts/soccerdata) | match xG, full forward season schedule | free scrape, cached | no |

Only football-data.co.uk is load-bearing. Everything else degrades: no xG
means fitting on goals, and no Understat schedule means falling back to the
rolling fixtures file.

## The processed table

`data/processed/matches.csv` is the single table everything fits on. Full
per-column metadata (type, source column, when it is missing) lives in
`data/matches_metadata.yaml`, and `tests/test_matches_metadata.py` keeps it
in sync with the loader:

| Column | Meaning |
|---|---|
| `league` | football-data division code — the primary key everywhere |
| `date` | match date |
| `home`, `away` | club names in football-data spelling (the canonical one) |
| `home_goals`, `away_goals` | full-time score |
| `b365h/d/a` | Bet365 **PRE-closing** odds |
| `b365ch/cd/ca`, `psch/pscd/psca`, `avgch/avgcd/avgca` | **CLOSING** odds: Bet365, Pinnacle, cross-book average |
| `home_xg`, `away_xg` | match xG, where available |

## Pre-closing vs closing odds

**The single most consequential distinction in this dataset.**
`B365H/D/A` are collected Friday afternoons for weekend games and Tuesday
afternoons for midweek ones — days before kickoff. The true closing prices are
the separate `C`-infixed columns.

Benchmarking against the pre-closing line flatters the model, because a stale
price is an easier target. Measured on four seasons of Premier League data
with the same model:

| Benchmark | Market log loss |
|---|---|
| Bet365 pre-closing | 1.0180 |
| Bet365 closing | 1.0124 |
| Cross-book average closing | 1.0111 |

Closing odds are used **only as an evaluation benchmark**. They are never an
input to the model, so the comparison stays honest.

## Club-name resolution

Every cross-source join goes through `data/xg.py`. Football-data spellings are
canonical; everything else maps onto them.

`normalize_team()` strips accents, punctuation and case, drops generic
organizational tokens (`fc`, `afc`, `cf`, `ac`, founding years), then applies
an alias table. `build_team_mapping()` resolves the remainder within a
league's own roster in three passes: exact canonical match, prefix/containment,
then fuzzy best-match.

Three rules that are load-bearing rather than cosmetic:

1. **The noise-token list is deliberately conservative.** An aggressive
   version collapsed "Manchester United"/"Manchester City" and "Real Madrid"/
   "Atletico Madrid" to identical strings — and the one-to-one assignment
   would then hand a club its city rival's data for a whole season. Words that
   distinguish two clubs in the same city are never noise, however generic
   they look.
2. **Many-to-one is allowed only via the exact-canonical pass.** Two spellings
   that canonicalize identically are the same club by construction. A fuzzy
   hit is a *guess*, and letting guesses share a target produces wrong data
   rather than missing data.
3. **Aliases are for genuine domain knowledge only** — nicknames and
   cross-language names no lexical rule can bridge (Wolverhampton
   Wanderers/Wolves, Athletic Club/Ath Bilbao, Cologne/Köln), plus source
   feed shorthand ("PSG", "Atletico", "Nottingham").

### Duplicate identities

`loader.canonicalize_teams()` collapses spellings sharing a canonical form
*per league*, keeping the most frequent. Frequency-based selection absorbs
future renames without a hardcoded list.

This exists because football-data.co.uk renamed two La Liga clubs mid-archive
— "Ath Madrid" → "Atl. Madrid", "Vallecano" → "Rayo Vallecano" — for exactly
one match each, so the model was fitting Atlético Madrid as a 153-match club
*plus a phantom with one match*, and the phantom looked newly promoted.

## Upstream hazards

Every data bug found in this project has been **silent**. Assume more exist.

| Hazard | Symptom | Guard |
|---|---|---|
| Mid-archive encoding change (Latin-1 → UTF-8 BOM in 2024/25) | first column becomes `﻿Div`, so `league` is null and per-league fits silently drop half the dataset | try `utf-8-sig` then `latin1`; strip BOMs; **raise** on missing required column |
| Unpublished season | HTML "300 Multiple Choices" page saved as `.csv`, parser explodes far from the cause | validate payload looks like a results CSV before writing |
| Hardcoded "current season" | live season becomes cacheable, new results silently stop arriving | `current_season_code()` computes it |
| Club renamed mid-archive | one club fitted as two teams | `canonicalize_teams()` |
| Odds column drop-out | Pinnacle goes blank mid-season | benchmark requires ≥90% holdout coverage |

**The lesson worth generalising:** degrade gracefully when data is legitimately
*absent*; fail loudly when it is *wrong*. Validate that required columns exist,
and verify joins against an independent signal rather than trusting a coverage
percentage — `merge_xg` compares both sources' scorelines, which is how it
caught a real discrepancy (Union Berlin v Bochum, 2024-12-14: 1-1 on the pitch
per Understat, awarded 0-2 by the DFB per football-data).

## Fixtures

Two sources, preferred in order:

1. **`fixtures_full.csv`** — Understat's whole-season schedule, ~370 fixtures
   per league through next May.
2. **`fixtures.csv`** — football-data's rolling near-term window. Republished
   weekly, and right after a matchweek it can contain *only already-played
   games*.

Fallback is **per league, not per file**: leagues start at different times, so
the full schedule can legitimately cover four of five while the fifth is
unpublished.

`load_fixtures()` returns only fixtures whose kickoff is still ahead. Both
sources contain played matches, and rendering those makes the model look like
it is predicting the past. Kickoffs combine `Date` + `Time` localized as
`Europe/London` (the files carry UK local time with no zone marker) and are
compared in UTC, so the boundary is right for a viewer anywhere. A missing
time falls back to 23:59 so a date-only fixture stays visible for its match
day.

## Second tiers

`E1`, `SP2`, `I2`, `D2`, `F2` are available from the same source and are
downloadable, but are **not currently used** — a promoted team's second-tier
record carries no signal for its top-flight season (`findings.md`).
