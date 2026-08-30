# Assumptions

What the model takes for granted, how wrong each one is, and what it would
cost to fix. **Add to this when you make a new assumption, and amend it when
you discover one is wrong.**

Severity is judgement, not measurement, unless a number is quoted.

## Statistical

**Goals are conditionally Poisson given team strengths.**
Mostly reasonable — goals are rare, roughly independent events. Real football
is mildly *over*-dispersed relative to Poisson (blowouts happen more than it
predicts). The `rho` correction patches only the four lowest cells; nothing
handles the upper tail. *Severity: low for W/D/L, higher for correct-score and
high over/under lines.*

**The two teams' goal counts are independent, apart from `rho`.**
`tau` is a local patch on four cells, not a correlation structure. Real
dependence (a team chasing a game, red cards, garbage time) is not captured.
*Severity: low-moderate.*

**Team strength is a single number per team per side, constant within a fit.**
No form cycles, no opponent-specific effects, no in-season trend beyond what
time decay imposes. *Severity: moderate — this is the model's core simplification.*

**Home advantage is one league-wide multiplier.**
Teams genuinely differ (stadium, travel, altitude). Per-team home advantage
with shrinkage is unimplemented (Task 3). *Severity: low-moderate.*

**Time decay is exponential in calendar days.**
Hence the summer problem: an 80-day break retains 87% of May's weight, so
pre-transfer-window form carries into a new season almost undiminished. A
season-boundary uncertainty injection would target this directly and is the
most promising unexplored idea. *Severity: **high** early in a season — see
`findings.md`.*

**Log-ratings are Gaussian about the league mean, sd 0.35.**
A modeling convenience giving partial pooling. `0.35` was chosen a priori
from the empirical spread, not tuned. The true distribution is probably
left-skewed (a few very weak teams). *Severity: low.*

**Poisson likelihood is valid for a continuous xG target.**
It is a *quasi*-likelihood: consistent for the mean, but the implied
`Var = mean` is wrong for xG, which is far less dispersed. Point estimates
are fine; **no standard error may be read off it**. *Severity: low for point
estimates, disqualifying for inference.*

**Partial xG coverage can be mixed with goals in one fit.**
Weights xG-backed and goals-backed matches as equally precise, which they are
not. `params.xg_coverage` records the mix. *Severity: low at ~100% coverage.*

## Data

**Club names are stable and resolvable across sources.**
Known false — football-data.co.uk renamed two La Liga clubs mid-archive.
`canonicalize_teams()` handles duplicates by frequency; cross-source mapping
is normalization plus in-league fuzzy matching plus aliases. A mis-mapping is
*silent* and gives a club another's data. *Severity: high if it happens,
guarded by tests and by scoreline verification in `merge_xg`.*

**Absence from the previous season means promotion.**
Also catches data gaps, league size changes, and clubs returning after two
years away. All share the property that matters (no recent record), but this
is a heuristic, not a promotion table. *Severity: low.*

**De-vigged bookmaker odds approximate fair probabilities.**
Proportional de-vigging assumes margin is spread evenly; in reality longshots
carry more, so their probabilities are mildly overstated. Applied identically
to both sides of every comparison. *Severity: low.*

**The closing line is the best freely available probability estimate.**
Well supported, and this project's own evidence agrees: closing (0.9778) beats
pre-closing (0.9806) beats the model (0.9952). *Severity: n/a — this is the
benchmark, and it holds up.*

## Structural

**Leagues are independent; no cross-league information transfers.**
Deliberate. Teams never meet, so a pooled fit would rank them on their
leagues' goal averages. The cost is that European competition and cross-league
transfers are invisible. *Severity: low for domestic markets, disqualifying
for any cross-league or UCL market.*

**A season is a single round-robin with a stable roster.**
True for the five leagues covered; false for MLS (conferences + playoffs) and
Champions League (two-legged ties, extra time, penalties). Both are deferred
as separate design problems, not config entries. *Severity: n/a within scope.*

**Cup and European fixtures don't exist.**
The model sees only league matches, so it cannot see congestion or rotation.
Any rest-days feature built from league history alone will undercount true
congestion — a documented limitation, not a bug to silently fix.
*Severity: moderate, unmeasured.*

**Squads are continuous across a season boundary.**
No transfers, no manager changes, no injuries. This is the single largest
known blind spot, and the early-season error is nearly as large for
established teams (+0.0225) as for promoted ones (+0.0241). *Severity: **high**,
partially quantified.*

**Match outcomes are the thing worth predicting.**
The model targets W/D/L and derived markets. It says nothing about in-play,
player props, or anything requiring event-level data. *Severity: n/a — scope.*

## Operational

**Free data sources remain available and stable.**
Understat is a scrape with no API contract; football-data.co.uk has already
changed encoding and club names mid-archive. Every optional source degrades
gracefully; the football-data dependency is load-bearing and has no fallback.
*Severity: moderate.*

## The meta-assumption worth stating

**That a better log loss means a better forecast in the ways that matter.**
Not automatically. The model loses to the bookmaker closing line (0.9952 vs
0.9778), and its *largest* departures from that line are its worst
predictions. So a change can improve average log loss while leaving the model
no more informative than a price anyone can look up. Treat closing-line
comparison, not self-comparison, as the standard.
