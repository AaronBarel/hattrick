#!/usr/bin/env python
"""Predict outcome, total-goals and scoreline probabilities for each upcoming fixture."""
from __future__ import annotations
from hattrick.config import load_config, load_league_models, resolve_path
from hattrick.data.loader import load_fixtures
from hattrick.model.scorelines import match_probabilities


def main():
    cfg = load_config()
    models = load_league_models(cfg)
    if not models:
        raise SystemExit("No fitted league models found - run scripts/fit_model.py first.")
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    max_goals = cfg["model"]["max_goals"]

    for league in cfg["leagues"]:
        model = models.get(league["code"])
        if model is None:
            print(f"No fitted model for {league['name']} ({league['code']}) - skipping.")
            continue
        fixtures = load_fixtures(raw_dir, league["code"])
        if fixtures.empty:
            print(f"No upcoming fixtures found for {league['name']} ({league['code']}).")
            continue
        print(f"\n=== {league['name']} upcoming fixtures ===")
        for _, row in fixtures.iterrows():
            home, away = row["home"], row["away"]
            if home not in model.params.attack or away not in model.params.attack:
                print(f"{row['date'].date()} {home} vs {away}: skipped (unseen team)")
                continue
            m = match_probabilities(model, home, away, max_goals)
            res = m["result"]
            print(f"\n{row['date'].date()}  {home} vs {away}")
            print(f"  Win/Draw/Loss:  H {res['home_win']:.1%}  D {res['draw']:.1%}  A {res['away_win']:.1%}")
            print(f"  Over/Under 2.5: Over {m['over_under_2_5']['over']:.1%}  "
                  f"Under {m['over_under_2_5']['under']:.1%}")
            print(f"  BTTS:           Yes {m['btts']['yes']:.1%}  No {m['btts']['no']:.1%}")
            top = ", ".join(f"{s[0]}-{s[1]} ({p:.1%})" for s, p in m["top_scorelines"][:3])
            print(f"  Most likely scorelines: {top}")


if __name__ == "__main__":
    main()
