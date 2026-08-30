# Documentation

Reference docs for the model, the data it eats, and the evidence behind
every design choice. The top-level `README.md` is the user-facing guide (how
to run things); these are the *why*.

| Doc | Covers |
|---|---|
| [`architecture.md`](architecture.md) | Module map, data flow, artifacts, scripts |
| [`data.md`](data.md) | Sources, schemas, club-name resolution, upstream hazards |
| [`modeling.md`](modeling.md) | Dixon-Coles derivation, priors, continuous xG target |
| [`evaluation.md`](evaluation.md) | Backtest protocol, scoring rules, benchmark selection, significance |
| [`findings.md`](findings.md) | Every measured result, including the negative ones |
| [`assumptions.md`](assumptions.md) | Standing assumptions and where they break |
| [`references.md`](references.md) | Literature and data sources |

## Keeping these current

**These docs are part of the change, not paperwork afterwards.** A change
that alters behavior and leaves the docs describing the old behavior has
not landed - the next person (or session) will read the stale version and
trust it. This project has already been bitten repeatedly by documentation
that confidently described something untrue: odds columns labeled "closing"
that were not, a repo layout naming a package that never existed.

Update in the same commit as the code:

| If you change… | Update |
|---|---|
| A module's responsibility, or add/remove one | `architecture.md` |
| A data source, schema, or club-name mapping | `data.md` |
| The likelihood, a prior, or any model parameter | `modeling.md` |
| Backtest protocol, metric, or benchmark choice | `evaluation.md` |
| **Anything you measured** - especially if it did not work | `findings.md` |
| A standing assumption, or discover one is wrong | `assumptions.md` |

Three rules that matter more than completeness:

1. **Record negative results.** A measurement showing a feature does *not*
   help is as valuable as one showing it does, and far more likely to be
   lost. `findings.md` exists so nobody re-runs a dead experiment.
2. **Numbers carry their provenance.** Any figure quoted here should say what
   data produced it (which leagues, which seasons, how many matches) and,
   where relevant, its confidence interval. A bare number ages into a myth.
3. **Say "unverified" when it is.** An honest gap is useful; a confident
   wrong claim is worse than silence.

`tests/test_docs.py` mechanically checks that every module under
`src/hattrick/` appears in `architecture.md`, so adding a module without
documenting it fails the suite. Everything else here is on the honour system.
