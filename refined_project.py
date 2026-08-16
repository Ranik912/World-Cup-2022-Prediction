"""
World Cup 2022 Prediction — Refined End-to-End Pipeline
=========================================================

What this version fixes/improves:
- Keeps draws as a real third class (Away / Draw / Home).
- Builds every pre-match feature using only information available before that match.
- Uses walk-forward/time-based validation instead of shuffled CV.
- Uses exponential recency decay and competition weights as training sample weights.
- Uses meaningful strength/form/goal features instead of arbitrary LabelEncoder IDs.
- Compares Logistic Regression, Random Forest and XGBoost.
- Selects a probabilistic ensemble using out-of-time validation.
- Produces 10,000+ World Cup Monte Carlo simulations.
- Handles group standings and knockout draws.
- Fixes Korea Republic naming and validates the 2022 groups/schedule.
- Saves model-comparison and tournament-probability outputs for the README/resume.

Run:
    python refined_project.py

The script expects these files in the same folder:
    international_matches.csv
    2022_world_cup_groups.csv
    2022_world_cup_matches.csv

Dependencies are listed in requirements.txt.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
N_SIMULATIONS = 10_000
HALF_LIFE_YEARS = 4.0
ELO_K = 32
ELO_HOME_ADVANTAGE = 55
FORM_WINDOW = 5

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LABEL_MAP = {"Away": 0, "Draw": 1, "Home": 2}
CLASS_NAMES = ["Away", "Draw", "Home"]

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "UEFA Euro": 0.95,
    "Copa America": 0.95,
    "Confederations Cup": 0.90,
    "African Cup of Nations": 0.90,
    "AFC Asian Cup": 0.90,
    "CONCACAF Championship": 0.90,
    "UEFA Nations League": 0.85,
    "Gold Cup": 0.85,
    "African Nations Championship": 0.80,
    "FIFA World Cup qualification": 0.80,
    "UEFA Euro qualification": 0.75,
    "African Cup of Nations qualification": 0.70,
    "AFC Asian Cup qualification": 0.70,
    "CONCACAF Championship qualification": 0.70,
    "Friendly": 0.30,
}

# Model features are intentionally based on information available before a match.
FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form5",
    "away_form5",
    "form_diff",
    "home_win_rate5",
    "away_win_rate5",
    "win_rate_diff",
    "home_gf5",
    "away_gf5",
    "gf_diff",
    "home_ga5",
    "away_ga5",
    "ga_diff",
    "home_gd5",
    "away_gd5",
    "gd_diff",
    "home_stadium",
    "tournament_weight_feature",
    "abs_elo_diff",
    "elo_diff_sq",
    "form_diff_sq",
    "gd_diff_sq",
]


def normalize_team(name: str) -> str:
    aliases = {
        "South Korea": "Korea Republic",
        "Korea South": "Korea Republic",
        "USA": "United States",
        "IR Iran": "Iran",
        "Czechia": "Czech Republic",
    }
    return aliases.get(str(name).strip(), str(name).strip())


def tournament_weight(tournament: str) -> float:
    tournament = str(tournament)
    if tournament in TOURNAMENT_WEIGHTS:
        return TOURNAMENT_WEIGHTS[tournament]
    t = tournament.lower()
    if "qualification" in t:
        return 0.65
    if "cup" in t or "championship" in t:
        return 0.70
    return 0.50


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist = pd.read_csv(BASE_DIR / "international_matches.csv")
    groups = pd.read_csv(BASE_DIR / "2022_world_cup_groups.csv")
    wc_matches = pd.read_csv(BASE_DIR / "2022_world_cup_matches.csv")

    hist["Home Team"] = hist["Home Team"].map(normalize_team)
    hist["Away Team"] = hist["Away Team"].map(normalize_team)

    # format='mixed' is important because the source CSV contains both '-' and '/'.
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce", format="mixed")
    hist = hist.dropna(
        subset=["Date", "Home Team", "Away Team", "Home Goals", "Away Goals"]
    ).copy()
    hist = hist[hist["Date"].dt.year >= 1950].copy()
    hist = hist.sort_values(["Date", "ID"]).reset_index(drop=True)

    hist["Result"] = np.select(
        [
            hist["Home Goals"] > hist["Away Goals"],
            hist["Home Goals"] < hist["Away Goals"],
        ],
        ["Home", "Away"],
        default="Draw",
    )
    hist["Tournament Weight"] = hist["Tournament"].map(tournament_weight)

    groups["Team"] = groups["Team"].map(normalize_team)
    wc_matches["Home Team"] = wc_matches["Home Team"].map(normalize_team)
    wc_matches["Away Team"] = wc_matches["Away Team"].map(normalize_team)
    wc_matches["Date"] = pd.to_datetime(wc_matches["Date"], format="mixed")

    validate_world_cup_data(groups, wc_matches)
    return hist, groups, wc_matches


def validate_world_cup_data(groups: pd.DataFrame, matches: pd.DataFrame) -> None:
    expected_groups = {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "Korea Republic"],
    }
    actual = {
        group: list(groups.loc[groups["Group"] == group, "Team"])
        for group in expected_groups
    }
    assert actual == expected_groups, f"2022 group data mismatch: {actual}"
    assert len(matches) == 64, "The 2022 match schedule should contain 64 matches."
    group_stage = matches[matches["Stage"].str.contains("Group", case=False, na=False)]
    assert set(group_stage["Home Team"]) | set(group_stage["Away Team"]) == set(groups["Team"])


def expected_home(elo_home: float, elo_away: float) -> float:
    return 1.0 / (
        1.0 + 10.0 ** ((elo_away - (elo_home + ELO_HOME_ADVANTAGE)) / 400.0)
    )


def build_pre_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features from state available immediately before each match."""
    df = df.sort_values(["Date", "ID"]).reset_index(drop=True).copy()
    elo = defaultdict(lambda: 1500.0)
    recent = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    goals_for = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    goals_against = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    rows = []

    for _, row in df.iterrows():
        home, away = row["Home Team"], row["Away Team"]
        h_elo, a_elo = elo[home], elo[away]
        h_recent, a_recent = list(recent[home]), list(recent[away])
        h_gf, a_gf = list(goals_for[home]), list(goals_for[away])
        h_ga, a_ga = list(goals_against[home]), list(goals_against[away])

        def avg(values, default=0.0):
            return float(np.mean(values)) if values else default

        def form(values):
            return float(np.mean(values)) if values else 0.5

        h_win_rate = avg([x == 1.0 for x in h_recent], 0.5)
        a_win_rate = avg([x == 1.0 for x in a_recent], 0.5)
        h_gd = avg([gf - ga for gf, ga in zip(h_gf, h_ga)])
        a_gd = avg([gf - ga for gf, ga in zip(a_gf, a_ga)])
        form_diff = form(h_recent) - form(a_recent)
        elo_diff = h_elo - a_elo
        gd_diff = h_gd - a_gd

        rows.append(
            {
                "home_elo": h_elo,
                "away_elo": a_elo,
                "elo_diff": elo_diff,
                "home_form5": form(h_recent),
                "away_form5": form(a_recent),
                "form_diff": form_diff,
                "home_win_rate5": h_win_rate,
                "away_win_rate5": a_win_rate,
                "win_rate_diff": h_win_rate - a_win_rate,
                "home_gf5": avg(h_gf),
                "away_gf5": avg(a_gf),
                "gf_diff": avg(h_gf) - avg(a_gf),
                "home_ga5": avg(h_ga),
                "away_ga5": avg(a_ga),
                "ga_diff": avg(h_ga) - avg(a_ga),
                "home_gd5": h_gd,
                "away_gd5": a_gd,
                "gd_diff": gd_diff,
                "home_stadium": float(bool(row.get("Home Stadium", False))),
                "tournament_weight_feature": row["Tournament Weight"],
                "abs_elo_diff": abs(elo_diff),
                "elo_diff_sq": (elo_diff / 400.0) ** 2,
                "form_diff_sq": form_diff**2,
                "gd_diff_sq": gd_diff**2,
            }
        )

        # IMPORTANT: update team state only after the current match features are made.
        if row["Result"] == "Home":
            s_home, s_away = 1.0, 0.0
        elif row["Result"] == "Away":
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        e_home = expected_home(h_elo, a_elo)
        elo[home] = h_elo + ELO_K * (s_home - e_home)
        elo[away] = a_elo + ELO_K * (s_away - (1.0 - e_home))

        recent[home].append(s_home)
        recent[away].append(s_away)
        goals_for[home].append(float(row["Home Goals"]))
        goals_for[away].append(float(row["Away Goals"]))
        goals_against[home].append(float(row["Away Goals"]))
        goals_against[away].append(float(row["Home Goals"]))

    feature_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)


def recency_weight(dates: pd.Series, cutoff: str | pd.Timestamp) -> np.ndarray:
    age_years = (
        pd.Timestamp(cutoff) - pd.to_datetime(dates)
    ).dt.days.to_numpy() / 365.25
    age_years = np.maximum(age_years, 0.0)
    return np.exp(-np.log(2) * age_years / HALF_LIFE_YEARS)


def make_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.05,
                        max_iter=3000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.90,
            min_child_weight=3,
            reg_lambda=2.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def fit_model(model, X, y, weights):
    model = clone(model)
    if isinstance(model, Pipeline):
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def align_proba(model, proba) -> np.ndarray:
    aligned = np.zeros((len(proba), 3), dtype=float)
    for j, cls in enumerate(model.classes_):
        aligned[:, int(cls)] = proba[:, j]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def evaluate_predictions(y_true, proba) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=[0, 1, 2]),
        "brier": float(
            np.mean(
                [
                    brier_score_loss((y_true == cls).astype(int), proba[:, cls])
                    for cls in [0, 1, 2]
                ]
            )
        ),
    }


def walk_forward_evaluation(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evaluate only on future periods not used for training."""
    splits = [
        ("2010-2013", "2010-01-01", "2014-01-01"),
        ("2014-2017", "2014-01-01", "2018-01-01"),
        ("2018-2020", "2018-01-01", "2021-01-01"),
        ("2021-2022", "2021-01-01", "2022-10-01"),
    ]
    rows = []
    oof = {name: [] for name in make_models()}

    for split_name, val_start, val_end in splits:
        train = data[data["Date"] < pd.Timestamp(val_start)]
        val = data[
            (data["Date"] >= pd.Timestamp(val_start))
            & (data["Date"] < pd.Timestamp(val_end))
        ]
        X_train, X_val = train[FEATURES], val[FEATURES]
        y_train = train["Result"].map(LABEL_MAP).to_numpy()
        y_val = val["Result"].map(LABEL_MAP).to_numpy()
        weights = train["Tournament Weight"].to_numpy() * recency_weight(
            train["Date"], val_start
        )

        for name, base_model in make_models().items():
            model = fit_model(base_model, X_train, y_train, weights)
            proba = align_proba(model, model.predict_proba(X_val))
            m = evaluate_predictions(y_val, proba)
            m.update(
                {
                    "model": name,
                    "split": split_name,
                    "n_train": len(train),
                    "n_val": len(val),
                }
            )
            rows.append(m)
            oof[name].append((y_val, proba))

    result_df = pd.DataFrame(rows)
    return result_df, oof


def select_ensemble_weights(oof: dict) -> tuple[float, float]:
    """Choose a simple Logistic/XGBoost blend from out-of-time validation."""
    y_all = np.concatenate([x[0] for x in oof["Logistic Regression"]])
    p_log = np.vstack([x[1] for x in oof["Logistic Regression"]])
    p_xgb = np.vstack([x[1] for x in oof["XGBoost"]])

    candidates = np.linspace(0.0, 1.0, 21)
    best = None
    for log_weight in candidates:
        p = log_weight * p_log + (1.0 - log_weight) * p_xgb
        score = log_loss(y_all, p, labels=[0, 1, 2])
        if best is None or score < best[0]:
            best = (score, log_weight)
    return float(best[1]), float(1.0 - best[1])


def team_state_as_of(data: pd.DataFrame) -> dict:
    """Return current team ratings/form immediately after the last historical match."""
    elo = defaultdict(lambda: 1500.0)
    recent = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    goals_for = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    goals_against = defaultdict(lambda: deque(maxlen=FORM_WINDOW))

    for _, row in data.sort_values(["Date", "ID"]).iterrows():
        h, a = row["Home Team"], row["Away Team"]
        h_elo, a_elo = elo[h], elo[a]
        if row["Result"] == "Home":
            s_h, s_a = 1.0, 0.0
        elif row["Result"] == "Away":
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5
        e_h = expected_home(h_elo, a_elo)
        elo[h] = h_elo + ELO_K * (s_h - e_h)
        elo[a] = a_elo + ELO_K * (s_a - (1 - e_h))
        recent[h].append(s_h)
        recent[a].append(s_a)
        goals_for[h].append(float(row["Home Goals"]))
        goals_for[a].append(float(row["Away Goals"]))
        goals_against[h].append(float(row["Away Goals"]))
        goals_against[a].append(float(row["Home Goals"]))

    def state(team):
        r = list(recent[team])
        gf = list(goals_for[team])
        ga = list(goals_against[team])
        return {
            "elo": elo[team],
            "form5": float(np.mean(r)) if r else 0.5,
            "win_rate5": float(np.mean([x == 1 for x in r])) if r else 0.5,
            "gf5": float(np.mean(gf)) if gf else 0.0,
            "ga5": float(np.mean(ga)) if ga else 0.0,
            "gd5": float(np.mean([x - y for x, y in zip(gf, ga)])) if gf else 0.0,
        }

    return {team: state(team) for team in set(data["Home Team"]) | set(data["Away Team"])}


def make_prediction_row(home: str, away: str, states: dict, host: bool, tournament_weight_value: float = 1.0) -> pd.DataFrame:
    h, a = states[home], states[away]
    elo_diff = h["elo"] - a["elo"]
    form_diff = h["form5"] - a["form5"]
    gd_diff = h["gd5"] - a["gd5"]
    return pd.DataFrame(
        [
            {
                "home_elo": h["elo"],
                "away_elo": a["elo"],
                "elo_diff": elo_diff,
                "home_form5": h["form5"],
                "away_form5": a["form5"],
                "form_diff": form_diff,
                "home_win_rate5": h["win_rate5"],
                "away_win_rate5": a["win_rate5"],
                "win_rate_diff": h["win_rate5"] - a["win_rate5"],
                "home_gf5": h["gf5"],
                "away_gf5": a["gf5"],
                "gf_diff": h["gf5"] - a["gf5"],
                "home_ga5": h["ga5"],
                "away_ga5": a["ga5"],
                "ga_diff": h["ga5"] - a["ga5"],
                "home_gd5": h["gd5"],
                "away_gd5": a["gd5"],
                "gd_diff": gd_diff,
                "home_stadium": float(host),
                "tournament_weight_feature": tournament_weight_value,
                "abs_elo_diff": abs(elo_diff),
                "elo_diff_sq": (elo_diff / 400.0) ** 2,
                "form_diff_sq": form_diff**2,
                "gd_diff_sq": gd_diff**2,
            }
        ]
    )


def train_final_models(data: pd.DataFrame, log_weight: float, xgb_weight: float) -> tuple[object, object, float, float]:
    cutoff = data["Date"].max() + pd.Timedelta(days=1)
    X = data[FEATURES]
    y = data["Result"].map(LABEL_MAP).to_numpy()
    weights = data["Tournament Weight"].to_numpy() * recency_weight(data["Date"], cutoff)
    log_model = fit_model(make_models()["Logistic Regression"], X, y, weights)
    xgb_model = fit_model(make_models()["XGBoost"], X, y, weights)
    return log_model, xgb_model, log_weight, xgb_weight


def match_probabilities(home, away, states, log_model, xgb_model, log_weight, xgb_weight, host=False):
    X = make_prediction_row(home, away, states, host)
    p_log = align_proba(log_model, log_model.predict_proba(X))
    p_xgb = align_proba(xgb_model, xgb_model.predict_proba(X))
    return (log_weight * p_log + xgb_weight * p_xgb)[0]


def train_goal_models(data: pd.DataFrame):
    """Poisson models are used only to generate plausible scorelines for group tie-breakers."""
    cutoff = data["Date"].max() + pd.Timedelta(days=1)
    X = data[FEATURES]
    weights = data["Tournament Weight"].to_numpy() * recency_weight(data["Date"], cutoff)
    home_model = Pipeline(
        [("scale", StandardScaler()), ("model", PoissonRegressor(alpha=0.5, max_iter=1000))]
    )
    away_model = Pipeline(
        [("scale", StandardScaler()), ("model", PoissonRegressor(alpha=0.5, max_iter=1000))]
    )
    home_model.fit(X, data["Home Goals"], model__sample_weight=weights)
    away_model.fit(X, data["Away Goals"], model__sample_weight=weights)
    return home_model, away_model


def score_expectation(home, away, states, home_goal_model, away_goal_model, host=False):
    X = make_prediction_row(home, away, states, host)
    lh = float(home_goal_model.predict(X)[0])
    la = float(away_goal_model.predict(X)[0])
    return max(lh, 0.05), max(la, 0.05)


def build_match_cache(teams, states, log_model, xgb_model, log_weight, xgb_weight, home_goal_model, away_goal_model):
    """Precompute every ordered team-pair probability once; Monte Carlo then becomes fast."""
    cache = {}
    for home in teams:
        for away in teams:
            if home == away:
                continue
            for host in (False, True):
                p = match_probabilities(home, away, states, log_model, xgb_model, log_weight, xgb_weight, host=host)
                lh, la = score_expectation(home, away, states, home_goal_model, away_goal_model, host=host)
                cache[(home, away, host)] = (p, lh, la)
    return cache


def simulate_match(
    home,
    away,
    states,
    log_model,
    xgb_model,
    log_weight,
    xgb_weight,
    home_goal_model,
    away_goal_model,
    host=False,
    knockout=False,
    rng=None,
    cache=None,
):
    rng = rng or np.random.default_rng(SEED)
    if cache is None:
        p = match_probabilities(home, away, states, log_model, xgb_model, log_weight, xgb_weight, host)
        hg, ag = score_expectation(home, away, states, home_goal_model, away_goal_model, host)
    else:
        p, hg, ag = cache[(home, away, host)]
    outcome = rng.choice(["Away", "Draw", "Home"], p=p)

    # Draws are legitimate in group stage. In knockouts, a draw after 90/120 minutes
    # is resolved by penalties; we use 50/50 because penalties are high-variance.
    if knockout and outcome == "Draw":
        outcome = rng.choice(["Away", "Home"])

    home_goals = int(rng.poisson(hg))
    away_goals = int(rng.poisson(ag))

    # Keep scoreline consistent with sampled match result for group standings.
    if outcome == "Home" and home_goals <= away_goals:
        home_goals = away_goals + 1
    elif outcome == "Away" and away_goals <= home_goals:
        away_goals = home_goals + 1
    elif outcome == "Draw":
        tie_score = min(home_goals, away_goals)
        home_goals = away_goals = tie_score

    winner = home if outcome == "Home" else away if outcome == "Away" else None
    return winner, home_goals, away_goals, p


def build_group_matches(groups: pd.DataFrame, wc_matches: pd.DataFrame | None = None):
    """Use the actual 2022 group-stage home/away fixture order when available."""
    if wc_matches is not None:
        group_stage = wc_matches[wc_matches["Stage"].str.contains("Group", case=False, na=False)].copy()
        return {
            group: list(
                group_stage.loc[
                    group_stage.apply(
                        lambda r: groups.loc[groups["Team"] == r["Home Team"], "Group"].iloc[0] == group,
                        axis=1,
                    ),
                    ["Home Team", "Away Team"],
                ].itertuples(index=False, name=None)
            )
            for group in sorted(groups["Group"].unique())
        }

    # Fallback if the official schedule file is not supplied.
    matches = {}
    for group in sorted(groups["Group"].unique()):
        teams = groups.loc[groups["Group"] == group, "Team"].tolist()
        matches[group] = [(teams[i], teams[j]) for i in range(4) for j in range(i + 1, 4)]
    return matches


def rank_group(table: dict, rng=None) -> list[str]:
    """Rank by points/GD/GF; exact residual ties are randomized rather than order-biased."""
    rng = rng or np.random.default_rng(SEED)
    buckets = defaultdict(list)
    for team, stats in table.items():
        key = (stats["points"], stats["gd"], stats["gf"])
        buckets[key].append(team)
    ordered = []
    for key in sorted(buckets, reverse=True):
        tied = buckets[key]
        rng.shuffle(tied)
        ordered.extend(tied)
    return ordered


def simulate_tournament(groups, group_matches, states, log_model, xgb_model, log_weight, xgb_weight, home_goal_model, away_goal_model, rng):
    standings = {}
    for group, matches in group_matches.items():
        table = {
            team: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
            for team in groups.loc[groups["Group"] == group, "Team"]
        }
        for home, away in matches:
            host = home == "Qatar" or away == "Qatar"
            winner, hg, ag, _ = simulate_match(
                home, away, states, log_model, xgb_model, log_weight, xgb_weight,
                home_goal_model, away_goal_model, host=host, knockout=False, rng=rng
            )
            table[home]["gf"] += hg; table[home]["ga"] += ag
            table[away]["gf"] += ag; table[away]["ga"] += hg
            table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
            table[away]["gd"] = table[away]["gf"] - table[away]["ga"]
            if winner is None:
                table[home]["points"] += 1; table[away]["points"] += 1
            elif winner == home:
                table[home]["points"] += 3
            else:
                table[away]["points"] += 3
        ordered = rank_group(table)
        standings[group] = ordered

    # 2022 Round-of-16 bracket.
    r16 = [
        (standings["A"][0], standings["B"][1]),
        (standings["C"][0], standings["D"][1]),
        (standings["B"][0], standings["A"][1]),
        (standings["D"][0], standings["C"][1]),
        (standings["E"][0], standings["F"][1]),
        (standings["G"][0], standings["H"][1]),
        (standings["F"][0], standings["E"][1]),
        (standings["H"][0], standings["G"][1]),
    ]

    def knockout_round(pairs):
        winners = []
        for home, away in pairs:
            winner, _, _, _ = simulate_match(
                home, away, states, log_model, xgb_model, log_weight, xgb_weight,
                home_goal_model, away_goal_model, host=False, knockout=True, rng=rng
            )
            winners.append(winner)
        return winners

    qf = knockout_round(r16)
    sf = knockout_round([(qf[0], qf[1]), (qf[2], qf[3]), (qf[4], qf[5]), (qf[6], qf[7])])
    final = knockout_round([(sf[0], sf[1]), (sf[2], sf[3])])
    champion = final[0]

    return standings, champion


def run_simulations(groups, group_matches, states, models, n=N_SIMULATIONS):
    log_model, xgb_model, log_weight, xgb_weight, home_goal_model, away_goal_model = models
    teams = groups["Team"].tolist()
    stage_counts = {
        team: {stage: 0 for stage in ["R16", "QF", "SF", "Final", "Champion"]}
        for team in teams
    }
    rng = np.random.default_rng(SEED)
    cache = build_match_cache(teams, states, log_model, xgb_model, log_weight, xgb_weight, home_goal_model, away_goal_model)

    for _ in range(n):
        standings = {}
        for group, matches in group_matches.items():
            table = {
                team: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
                for team in groups.loc[groups["Group"] == group, "Team"]
            }
            for home, away in matches:
                host = home == "Qatar" or away == "Qatar"
                winner, hg, ag, _ = simulate_match(
                    home,
                    away,
                    states,
                    log_model,
                    xgb_model,
                    log_weight,
                    xgb_weight,
                    home_goal_model,
                    away_goal_model,
                    host=host,
                    knockout=False,
                    rng=rng,
                    cache=cache,
                )
                table[home]["gf"] += hg
                table[home]["ga"] += ag
                table[away]["gf"] += ag
                table[away]["ga"] += hg
                table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
                table[away]["gd"] = table[away]["gf"] - table[away]["ga"]
                if winner is None:
                    table[home]["points"] += 1
                    table[away]["points"] += 1
                elif winner == home:
                    table[home]["points"] += 3
                else:
                    table[away]["points"] += 3
            standings[group] = rank_group(table, rng)

        # Group-stage qualification = reaching the Round of 16.
        for group, ordered in standings.items():
            for team in ordered[:2]:
                stage_counts[team]["R16"] += 1

        # Official 2022 Round-of-16 bracket.
        r16 = [
            (standings["A"][0], standings["B"][1]),
            (standings["C"][0], standings["D"][1]),
            (standings["B"][0], standings["A"][1]),
            (standings["D"][0], standings["C"][1]),
            (standings["E"][0], standings["F"][1]),
            (standings["G"][0], standings["H"][1]),
            (standings["F"][0], standings["E"][1]),
            (standings["H"][0], standings["G"][1]),
        ]

        r16_winners = []
        for home, away in r16:
            winner, *_ = simulate_match(
                home,
                away,
                states,
                log_model,
                xgb_model,
                log_weight,
                xgb_weight,
                home_goal_model,
                away_goal_model,
                knockout=True,
                rng=rng,
                cache=cache,
            )
            r16_winners.append(winner)
        for team in r16_winners:
            stage_counts[team]["QF"] += 1

        qf_pairs = [
            (r16_winners[0], r16_winners[1]),
            (r16_winners[2], r16_winners[3]),
            (r16_winners[4], r16_winners[5]),
            (r16_winners[6], r16_winners[7]),
        ]
        qf_winners = []
        for home, away in qf_pairs:
            winner, *_ = simulate_match(
                home,
                away,
                states,
                log_model,
                xgb_model,
                log_weight,
                xgb_weight,
                home_goal_model,
                away_goal_model,
                knockout=True,
                rng=rng,
                cache=cache,
            )
            qf_winners.append(winner)
        for team in qf_winners:
            stage_counts[team]["SF"] += 1

        sf_pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
        sf_winners = []
        for home, away in sf_pairs:
            winner, *_ = simulate_match(
                home,
                away,
                states,
                log_model,
                xgb_model,
                log_weight,
                xgb_weight,
                home_goal_model,
                away_goal_model,
                knockout=True,
                rng=rng,
                cache=cache,
            )
            sf_winners.append(winner)
        for team in sf_winners:
            stage_counts[team]["Final"] += 1

        champion, *_ = simulate_match(
            sf_winners[0],
            sf_winners[1],
            states,
            log_model,
            xgb_model,
            log_weight,
            xgb_weight,
            home_goal_model,
            away_goal_model,
            knockout=True,
            rng=rng,
            cache=cache,
        )
        stage_counts[champion]["Champion"] += 1

    table = pd.DataFrame.from_dict(stage_counts, orient="index")
    table.index.name = "Team"
    table = table.reset_index()
    for stage in ["R16", "QF", "SF", "Final", "Champion"]:
        table[f"{stage} Probability"] = table[stage] / n
    return table.sort_values("Champion Probability", ascending=False)


def save_results(validation: pd.DataFrame, probabilities: pd.DataFrame, groups: pd.DataFrame, states: dict):
    validation.to_csv(RESULTS_DIR / "walk_forward_model_comparison.csv", index=False)
    probabilities.to_csv(RESULTS_DIR / "world_cup_probabilities.csv", index=False)
    with open(RESULTS_DIR / "final_team_states.json", "w") as f:
        json.dump(states, f, indent=2)

    summary = (
        validation.groupby("model")[["accuracy", "log_loss", "brier"]]
        .mean()
        .sort_values("log_loss")
    )
    summary.to_csv(RESULTS_DIR / "model_summary.csv")

    plt.figure(figsize=(10, 6))
    x = np.arange(len(summary))
    plt.bar(x - 0.2, summary["accuracy"], width=0.2, label="Accuracy")
    plt.bar(x, summary["log_loss"], width=0.2, label="Log Loss")
    plt.bar(x + 0.2, summary["brier"], width=0.2, label="Brier")
    plt.xticks(x, summary.index, rotation=20, ha="right")
    plt.title("Out-of-Time Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=180)
    plt.close()

    top = probabilities.head(12).sort_values("Champion Probability")
    plt.figure(figsize=(10, 6))
    plt.barh(top["Team"], top["Champion Probability"] * 100)
    plt.xlabel("World Cup win probability (%)")
    plt.title("2022 World Cup Champion Probabilities — Monte Carlo")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "champion_probabilities.png", dpi=180)
    plt.close()


def main():
    print("Loading data...")
    hist, groups, wc_matches = prepare_data()
    print(f"Historical matches used: {len(hist):,}")
    print(f"Date range: {hist['Date'].min().date()} to {hist['Date'].max().date()}")
    print(f"Target distribution:\n{hist['Result'].value_counts(normalize=True).round(3)}")

    print("\nBuilding leakage-safe pre-match features...")
    featured = build_pre_match_features(hist)

    print("\nWalk-forward validation...")
    validation, oof = walk_forward_evaluation(featured)
    summary = validation.groupby("model")[["accuracy", "log_loss", "brier"]].mean().sort_values("log_loss")
    print("\nMean out-of-time performance:")
    print(summary.round(4).to_string())

    log_weight, xgb_weight = select_ensemble_weights(oof)
    print(f"\nSelected ensemble: Logistic={log_weight:.2f}, XGBoost={xgb_weight:.2f}")

    print("\nTraining final pre-tournament models...")
    log_model, xgb_model, _, _ = train_final_models(featured, log_weight, xgb_weight)
    home_goal_model, away_goal_model = train_goal_models(featured)
    states = team_state_as_of(featured)

    missing = sorted(set(groups["Team"]) - set(states))
    if missing:
        raise ValueError(f"Missing team states for World Cup teams: {missing}")

    print("\nRunning Monte Carlo tournament simulation...")
    group_matches = build_group_matches(groups, wc_matches)
    probabilities = run_simulations(
        groups,
        group_matches,
        states,
        (log_model, xgb_model, log_weight, xgb_weight, home_goal_model, away_goal_model),
        n=N_SIMULATIONS,
    )

    print("\nTop predicted teams:")
    cols = ["Team", "R16 Probability", "QF Probability", "SF Probability", "Final Probability", "Champion Probability"]
    print((probabilities[cols].head(15).set_index("Team") * 100).round(2).to_string())

    save_results(validation, probabilities, groups, states)
    print(f"\nSaved results to: {RESULTS_DIR}")
    print("Files:")
    for p in sorted(RESULTS_DIR.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
