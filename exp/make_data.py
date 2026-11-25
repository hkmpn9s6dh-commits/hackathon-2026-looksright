"""Generate the eight task datasets.  Deterministic: same seeds, same bytes.

Run once with `py -3 exp/make_data.py`.  The CSVs it writes are committed to the
repository, so that a reviewer never needs to regenerate anything; this script
exists so that the provenance of every number is checkable.

Each task is a small, hand-written data-generating process.  The trap is part of
the process, not painted on afterwards: the duplicate rows really do land in both
folds, the sentinel really is the mode of its column, the collider really is
produced after the outcome, and so on.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from looksright.spec import SEED, TASKS  # noqa: E402

DATA = os.path.join(ROOT, "data")
LABELS = os.path.join(DATA, "_labels")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _clip(x, lo, hi):
    x = np.maximum(x, lo) if lo is not None else x
    x = np.minimum(x, hi) if hi is not None else x
    return x


# --------------------------------------------------------------------------
# 1. wine_dup
# --------------------------------------------------------------------------


def make_wine_dup(spec, rng):
    def draw(n):
        q_star = rng.normal(5.9, 0.75, n)
        alcohol = _clip(8.6 + 1.55 * (q_star - 5.9) + rng.normal(0, 0.45, n), 8.0, 15.5)
        volatile = _clip(0.52 - 0.13 * (q_star - 5.9) + rng.normal(0, 0.07, n), 0.08, 1.2)
        sulphates = _clip(0.53 + 0.10 * (q_star - 5.9) + rng.normal(0, 0.09, n), 0.2, 1.3)
        citric = _clip(0.30 + 0.05 * (q_star - 5.9) + rng.normal(0, 0.13, n), 0.0, 1.1)
        sugar = _clip(rng.gamma(2.2, 1.0, n) + 0.4, 0.6, 14.0)
        chlorides = _clip(rng.lognormal(-1.85, 0.28, n), 0.02, 0.35)
        fixed = _clip(rng.normal(8.1, 1.1, n), 4.0, 13.5)
        free_so2 = _clip(rng.normal(16.0, 6.0, n), 2.0, 60.0)
        total_so2 = _clip(free_so2 * 2.6 + rng.normal(18, 12, n), 12.0, 190.0)
        ph = _clip(3.20 - 0.35 * (alcohol - 10.0) * 0.1 + rng.normal(0, 0.11, n), 2.85, 3.75)
        density = _clip(0.9955 - 0.00075 * (alcohol - 10.0) + rng.normal(0, 0.0011, n), 0.989, 1.004)
        quality = _clip(np.round(q_star + rng.normal(0, 0.42, n)), 3, 9)
        df = pd.DataFrame(
            {
                "fixed_acidity": fixed,
                "volatile_acidity": volatile,
                "citric_acid": citric,
                "residual_sugar": sugar,
                "chlorides": chlorides,
                "free_so2": free_so2,
                "total_so2": total_so2,
                "density": density,
                "ph": ph,
                "sulphates": sulphates,
                "alcohol": alcohol,
                "quality": quality,
            }
        )
        return df

    # 820 distinct bottles.  The winery re-tasted 230 of them, and the re-tasted
    # bottles are not a random subset: they are the ones the panel argued about,
    # which in practice means the high-scoring end of the flight.
    base = draw(820)
    order = base["quality"].sort_values(ascending=False).index
    reserved = list(order[:260])
    extra_a = rng.choice(reserved, 200, replace=False)
    extra_b = rng.choice(reserved, 60, replace=False)
    rows = []
    for i in range(820):
        rows.append(base.iloc[i])
    for i in list(extra_a) + list(extra_b) + list(extra_b):
        rows.append(base.loc[i])
    train = pd.DataFrame(rows).reset_index(drop=True)
    # Shuffle the export order so that the duplicates are not adjacent.
    train = train.iloc[rng.permutation(len(train))].reset_index(drop=True)
    train.insert(0, "row_id", np.arange(1, len(train) + 1))

    ev = draw(spec.n_eval)
    ev.insert(0, "row_id", np.arange(100001, 100001 + len(ev)))
    labels = ev[["row_id", "quality"]].copy()
    feats = ev.drop(columns=["quality"])
    return train, feats, labels


# --------------------------------------------------------------------------
# 2. sentinel
# --------------------------------------------------------------------------


def make_sentinel(spec, rng):
    n = spec.n_train + spec.n_eval

    age = _clip(rng.normal(57, 11.0, n), 30, 84).round()
    sex = rng.choice(["F", "M"], n, p=[0.52, 0.48])
    bmi = _clip(rng.normal(27.2, 4.4, n), 16.0, 46.0)
    sbp = _clip(rng.normal(129, 15.0, n), 92, 195).round()
    chol = _clip(rng.normal(206, 33.0, n), 118, 330)
    glucose = _clip(rng.normal(101, 17.0, n), 62, 210)
    hba1c = _clip(rng.normal(5.6, 0.72, n), 4.1, 10.2)
    prior = rng.poisson(0.65, n)
    smoker = rng.choice(["no", "yes"], n, p=[0.78, 0.22])

    latent = (
        -2.78
        + 0.066 * (age - 57)
        + 0.115 * (bmi - 27.2)
        + 0.024 * (sbp - 129)
        + 0.0145 * (chol - 206)
        + 0.0198 * (glucose - 101)
        + 1.15 * (hba1c - 5.6)
        + 0.75 * prior
        + 0.70 * (smoker == "yes")
        + rng.normal(0, 0.45, n)
    )
    readmitted = rng.binomial(1, sigmoid(latent))

    df = pd.DataFrame(
        {
            "age": age.astype(int),
            "sex": sex,
            "bmi": bmi.round(2),
            "sbp": sbp.astype(int),
            "chol": chol.round(1),
            "glucose": glucose.round(1),
            "hba1c": hba1c.round(2),
            "prior_admissions": prior.astype(int),
            "smoker": smoker,
            "readmitted": readmitted.astype(int),
        }
    )

    # The lab system's missing-value codes.  MCAR: the missingness itself means
    # nothing, so the only thing these codes can do is mislead a model that reads
    # them as numbers.
    def poke(col, mask, code):
        m = rng.random(n) < mask
        df.loc[m, col] = code
        return int(m.sum())

    poke("chol", 0.25, 999.0)  # "not ordered" -> looks like catastrophic cholesterol
    poke("glucose", 0.08, -999.0)
    poke("hba1c", 0.30, -99.0)
    poke("bmi", 0.03, -999.0)
    poke("sbp", 0.02, 0)

    train = df.iloc[: spec.n_train].copy()
    ev = df.iloc[spec.n_train :].copy()
    train.insert(0, "row_id", np.arange(1, len(train) + 1))
    ev.insert(0, "row_id", np.arange(100001, 100001 + len(ev)))
    labels = ev[["row_id", "readmitted"]].copy()
    feats = ev.drop(columns=["readmitted"]).reset_index(drop=True)
    return train.reset_index(drop=True), feats, labels


# --------------------------------------------------------------------------
# 3. target_leak
# --------------------------------------------------------------------------


def make_target_leak(spec, rng):
    n = spec.n_train + spec.n_eval
    region = rng.choice(["east", "north", "south", "west", "central", "northeast"], n)
    region_eff = {"east": -0.25, "north": 0.05, "south": -0.10, "west": 0.18,
                  "central": 0.10, "northeast": 0.22}
    income = _clip(rng.lognormal(9.6, 0.45, n), 2500, 80000).round(-2)
    loan = _clip(rng.lognormal(9.9, 0.55, n), 2000, 120000).round(-2)
    dti = _clip(rng.beta(2.4, 3.2, n) * 0.9 + 0.05, 0.05, 0.95)
    credit = _clip(rng.normal(672, 62, n), 430, 830).round()
    tenure = _clip(rng.gamma(2.4, 1.6, n), 0.1, 25.0)

    latent = (
        -2.25
        - 0.0115 * (credit - 672)
        + 2.35 * (dti - 0.42)
        + 0.75 * (np.log(loan) - 9.9)
        - 0.085 * tenure
        + np.array([region_eff[r] for r in region])
        - 0.35 * (np.log(income) - 9.6)
        + rng.normal(0, 0.35, n)
    )
    default = rng.binomial(1, sigmoid(latent))

    # Post-outcome fields.  In the warehouse these are filled in months after the
    # loan is booked; on the application form they do not exist.
    collections = np.where(default == 1, rng.poisson(2.6, n), rng.poisson(0.10, n)).astype(float)
    collections = np.where(rng.random(n) < 0.12, collections + rng.poisson(1.0, n), collections)
    recovered = np.where(
        default == 1,
        _clip(rng.lognormal(8.6, 0.7, n) * rng.binomial(1, 0.55, n), 0, None),
        0.0,
    )
    days_to_pay = np.where(
        default == 1, _clip(rng.normal(52, 18, n), 5, 130), _clip(rng.normal(23, 6, n), 3, 60)
    )

    df = pd.DataFrame(
        {
            "income": income,
            "loan_amount": loan,
            "dti": dti.round(3),
            "credit_score": credit.astype(int),
            "employment_years": tenure.round(2),
            "region": region,
            "collections_30d": collections,
            "recovered_amount": recovered.round(2),
            "days_to_first_payment": days_to_pay.round(1),
            "default_90d": default.astype(int),
        }
    )
    train = df.iloc[: spec.n_train].copy()
    ev = df.iloc[spec.n_train :].copy()
    train.insert(0, "row_id", np.arange(1, len(train) + 1))
    ev.insert(0, "row_id", np.arange(100001, 100001 + len(ev)))
    labels = ev[["row_id", "default_90d"]].copy()
    # The application-time file does not carry the three post-outcome columns.
    feats = ev.drop(columns=["default_90d", "collections_30d", "recovered_amount",
                             "days_to_first_payment"]).reset_index(drop=True)
    return train.reset_index(drop=True), feats, labels


# --------------------------------------------------------------------------
# 4. temporal
# --------------------------------------------------------------------------


def make_temporal(spec, rng):
    stores = ["S1", "S2", "S3"]
    days_total = 730
    horizon = 90
    rows = []
    for si, s in enumerate(stores):
        base = [235.0, 168.0, 300.0][si]
        trend = [0.315, 0.235, 0.400][si]
        phase = [0.0, 1.9, 3.4][si]
        for d in range(days_total):
            dow = (d + 3) % 7
            promo = int(rng.random() < 0.14)
            price = round(float(_clip(1.0 + rng.normal(0, 0.035), 0.88, 1.12)), 3)
            temp = 14.0 + 9.0 * np.sin(2 * np.pi * (d - 100) / 365) + rng.normal(0, 3.0)
            y = (
                base
                + trend * d
                + 34.0 * np.sin(2 * np.pi * d / 365 + phase)
                + 11.0 * (dow >= 5)
                - 9.0 * (dow == 0)
                + promo * (21.0 + 8.0 * (dow >= 5))
                - 120.0 * (price - 1.0)
                + 0.55 * (temp - 14.0)
                + rng.normal(0, 12.5)
            )
            rows.append(
                {
                    "date": "2025-01-01",
                    "store_id": s,
                    "dow": dow,
                    "month": 1,
                    "promo": promo,
                    "price_index": price,
                    "temp_c": round(float(temp), 2),
                    "units": round(float(y), 1),
                    "_d": d,
                }
            )
    df = pd.DataFrame(rows)
    # A real calendar, so that a reader sees dates rather than an index.
    dates = pd.date_range("2024-01-01", periods=days_total, freq="D")
    df["date"] = [dates[d] for d in df["_d"]]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["month"] = [int(d[5:7]) for d in df["date"]]
    df = df.drop(columns=["_d"])

    cutoff = days_total - horizon
    train_mask = df["date"] < dates[cutoff].strftime("%Y-%m-%d")
    train = df[train_mask].copy()
    ev = df[~train_mask].copy()
    train.insert(0, "row_id", np.arange(1, len(train) + 1))
    ev.insert(0, "row_id", np.arange(100001, 100001 + len(ev)))
    labels = ev[["row_id", "units"]].copy()
    feats = ev.drop(columns=["units"]).reset_index(drop=True)
    return train.reset_index(drop=True), feats, labels


# --------------------------------------------------------------------------
# 5. metric_mismatch
# --------------------------------------------------------------------------


def make_metric_mismatch(spec, rng):
    n = spec.n_train + spec.n_eval
    batches = np.array(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"])
    batch = rng.choice(batches, n, p=[0.15, 0.14, 0.10, 0.16, 0.10, 0.13, 0.11, 0.11])
    speed = _clip(rng.normal(48, 7.5, n), 26, 72)
    tenure = _clip(rng.gamma(2.0, 9.0, n), 0.3, 120)
    humidity = _clip(rng.normal(48, 9.0, n), 22, 78)
    wear = _clip(rng.gamma(3.0, 1.8, n), 0.2, 10.0)
    vib = _clip(rng.gamma(2.6, 0.9, n), 0.4, 7.5)
    temp = _clip(rng.normal(31.5, 3.2, n), 22, 42)

    # Two routes to a defect.
    # (a) a batch-specific cluster: batches B3 and B5 with a worn tool.  Rare, but
    #     almost deterministic once it fires -> a tree can isolate it, and it is
    #     where most of the head precision lives.
    # (b) a diffuse linear route spread over the whole run -> it lifts the bulk of
    #     the ranking, which is what ROC-AUC rewards.
    route_a = (
        ((batch == "B3") | (batch == "B5"))
        & (wear > 5.4)
        & (vib > 2.4)
    )
    z = (
        -4.55
        + 0.42 * (wear - 5.4)
        + 0.55 * (vib - 2.6)
        + 0.035 * (humidity - 48)
        + 0.030 * (temp - 31.5)
        - 0.020 * (tenure - 18)
        + 0.012 * (speed - 48)
        + rng.normal(0, 0.75, n)
    )
    p = sigmoid(z)
    p = np.where(route_a, _clip(p + 0.16, 0, 0.97), p)
    defect = rng.binomial(1, p)

    df = pd.DataFrame(
        {
            "line_speed": speed.round(2),
            "material_batch": batch,
            "operator_tenure": tenure.round(2),
            "ambient_humidity": humidity.round(1),
            "tool_wear": wear.round(3),
            "vibration": vib.round(3),
            "temperature": temp.round(2),
            "defect": defect.astype(int),
        }
    )
    train = df.iloc[: spec.n_train].copy()
    ev = df.iloc[spec.n_train :].copy()
    train.insert(0, "row_id", np.arange(1, len(train) + 1))
    ev.insert(0, "row_id", np.arange(100001, 100001 + len(ev)))
    labels = ev[["row_id", "defect"]].copy()
    feats = ev.drop(columns=["defect"]).reset_index(drop=True)
    return train.reset_index(drop=True), feats, labels


# --------------------------------------------------------------------------
# 6. confounding
# --------------------------------------------------------------------------


def make_confounding(spec, rng):
    n = spec.n_train + spec.n_eval
    severity = _clip(rng.beta(2.2, 2.0, n) * 10.0, 0.0, 10.0)
    comorbidity = rng.poisson(0.25 + 0.22 * severity)
    age = _clip(rng.normal(46 + 1.6 * severity, 12.0, n), 18, 88).round()
    sex = rng.choice(["F", "M"], n, p=[0.51, 0.49])
    bmi = _clip(rng.normal(25.0 + 0.35 * severity, 4.2, n), 16.5, 44.0)
    marker = _clip(rng.normal(50 + 3.1 * severity, 9.0, n), 5, 160)

    p_treat = sigmoid(-0.85 + 0.44 * (severity - 5.0) + 0.18 * (comorbidity - 1.6))
    treated = rng.binomial(1, p_treat)

    # Baseline recovery: mild patients recover on their own, severe ones do not.
    y0 = sigmoid(
        2.55
        - 0.44 * severity
        - 0.16 * comorbidity
        + 0.010 * (age - 50)
        - 0.02 * (bmi - 26)
        + rng.normal(0, 0.35, n)
    )
    # The drug helps exactly the patients who would otherwise do badly, and does
    # almost nothing for the mild ones.  This is why "predict who recovers" and
    # "predict who benefits" are different questions.
    uplift = _clip(0.60 * sigmoid(1.15 * (severity - 7.2)) + rng.normal(0, 0.02, n), 0.0, 0.90)
    y1 = _clip(y0 + uplift, 0.0, 1.0)

    recovered = np.where(treated == 1, rng.binomial(1, y1), rng.binomial(1, y0))

    df = pd.DataFrame(
        {
            "age": age.astype(int),
            "severity": severity.round(2),
            "comorbidity": comorbidity.astype(int),
            "sex": sex,
            "bmi": bmi.round(2),
            "baseline_marker": marker.round(1),
            "treated": treated.astype(int),
            "recovered": recovered.astype(int),
        }
    )
    train = df.iloc[: spec.n_train].copy()
    ev = df.iloc[spec.n_train :].copy()
    train.insert(0, "row_id", np.arange(1, len(train) + 1))
    ev.insert(0, "row_id", np.arange(100001, 100001 + len(ev)))
    labels = ev[["row_id", "recovered"]].copy()
    # Grader-side truth for the policy comparison.  Never shipped with the task.
    truth = pd.DataFrame(
        {
            "row_id": labels["row_id"].values,
            "y0": y0[spec.n_train :],
            "y1": y1[spec.n_train :],
            "uplift": uplift[spec.n_train :],
        }
    )
    feats = ev.drop(columns=["recovered", "treated"]).reset_index(drop=True)
    return train.reset_index(drop=True), feats, labels, truth


# --------------------------------------------------------------------------
# 7. winner_curse
# --------------------------------------------------------------------------


def make_winner_curse(spec, rng):
    n = spec.n_train + spec.n_eval
    tenure = _clip(rng.gamma(2.2, 7.0, n), 0.2, 72)
    charge = _clip(rng.normal(68, 22, n) + rng.normal(0, 3, n), 18, 145)
    total = _clip(tenure * charge * _clip(rng.normal(1.0, 0.06, n), 0.75, 1.25), 20, 9000)
    contract = rng.choice(["monthly", "one_year", "two_year"], n, p=[0.55, 0.26, 0.19])
    calls = rng.poisson(0.85, n)
    late = rng.poisson(0.42, n)
    usage_drop = _clip(rng.normal(-0.02, 0.16, n), -0.9, 0.7)

    # Ten weak effects.  No single one is strong; that is the point of the task.
    latent = (
        -1.62
        - 0.030 * (tenure - 15)
        + 0.008 * (charge - 68)
        + 0.30 * calls
        + 0.34 * late
        + 1.10 * usage_drop
        + np.where(contract == "monthly", 0.42, np.where(contract == "one_year", 0.10, -0.32))
        + 0.010 * (total - 1000) / 100
        + rng.normal(0, 0.85, n)
    )
    churn = rng.binomial(1, sigmoid(latent))

    df = pd.DataFrame(
        {
            "tenure_months": tenure.round(2),
            "monthly_charge": charge.round(2),
            "total_charges": total.round(2),
            "contract": contract,
            "support_calls": calls.astype(int),
            "late_payments": late.astype(int),
            "usage_drop": usage_drop.round(3),
            "churn": churn.astype(int),
        }
    )
    train = df.iloc[: spec.n_train].copy()
    ev = df.iloc[spec.n_train :].copy()
    train.insert(0, "row_id", np.arange(1, len(train) + 1))
    ev.insert(0, "row_id", np.arange(100001, 100001 + len(ev)))
    labels = ev[["row_id", "churn"]].copy()
    feats = ev.drop(columns=["churn"]).reset_index(drop=True)
    return train.reset_index(drop=True), feats, labels


# --------------------------------------------------------------------------
# 8. pseudo_id
# --------------------------------------------------------------------------


def make_pseudo_id(spec, rng):
    n = spec.n_train + spec.n_eval
    visits = rng.poisson(4.2, n)
    pages = _clip(rng.gamma(3.0, 1.7, n), 0.5, 24)
    minutes = _clip(rng.gamma(2.4, 4.2, n), 0.4, 90)
    opens = rng.poisson(2.1, n)
    discount = rng.binomial(1, 0.22, n)
    device = rng.choice(["desktop", "mobile", "tablet"], n, p=[0.44, 0.44, 0.12])
    region = rng.choice(["east", "north", "south", "west", "central"], n)

    latent = (
        -1.85
        + 0.22 * (visits - 4.2)
        + 0.16 * (pages - 5.1)
        + 0.030 * (minutes - 10.1)
        + 0.15 * (opens - 2.1)
        + 0.55 * discount
        + np.where(device == "desktop", 0.18, np.where(device == "mobile", -0.12, 0.0))
        + np.array([{"east": 0.10, "north": -0.05, "south": 0.02,
                     "west": -0.12, "central": 0.06}[r] for r in region])
        + rng.normal(0, 0.55, n)
    )
    converted = rng.binomial(1, sigmoid(latent))

    df = pd.DataFrame(
        {
            "visits_30d": visits.astype(int),
            "pages_per_visit": pages.round(2),
            "time_on_site_min": minutes.round(2),
            "email_opens": opens.astype(int),
            "discount_used": discount.astype(int),
            "device": device,
            "region": region,
            "converted": converted.astype(int),
        }
    )

    # The CRM export is sorted by a weighted outcome before it is numbered, so the
    # running number is a noisy copy of the label.  Two exports: the training one
    # (ids 1..n_train) and a later one (ids continue).
    train = df.iloc[: spec.n_train].copy()
    key = train["converted"].values + rng.normal(0, 0.72, spec.n_train)
    order = np.argsort(key)
    train = train.iloc[order].reset_index(drop=True)
    train.insert(0, "record_id", np.arange(1, len(train) + 1))

    ev = df.iloc[spec.n_train :].copy()
    ev.insert(0, "record_id", np.arange(spec.n_train + 1, spec.n_train + 1 + len(ev)))
    labels = ev[["record_id", "converted"]].copy()
    feats = ev.drop(columns=["converted"]).reset_index(drop=True)
    labels = labels.rename(columns={"record_id": "row_id"})
    labels["row_id"] = np.arange(100001, 100001 + len(labels))
    feats.insert(0, "row_id", np.arange(100001, 100001 + len(feats)))
    return train, feats, labels


# --------------------------------------------------------------------------
# dictionary / meta
# --------------------------------------------------------------------------


DICT = {
    "wine_dup": {
        "fixed_acidity": (4.0, 13.5, "g/L"), "volatile_acidity": (0.08, 1.2, "g/L"),
        "citric_acid": (0.0, 1.1, "g/L"), "residual_sugar": (0.6, 14.0, "g/L"),
        "chlorides": (0.02, 0.35, "g/L"), "free_so2": (2.0, 60.0, "mg/L"),
        "total_so2": (12.0, 190.0, "mg/L"), "density": (0.989, 1.004, "g/mL"),
        "ph": (2.85, 3.75, ""), "sulphates": (0.2, 1.3, "g/L"),
        "alcohol": (8.0, 15.5, "% vol"), "quality": (3.0, 9.0, "0-10"),
    },
    "sentinel": {
        "age": (18.0, 100.0, "years"), "bmi": (14.0, 50.0, "kg/m2"),
        "sbp": (60.0, 250.0, "mmHg"), "chol": (100.0, 350.0, "mg/dL"),
        "glucose": (50.0, 300.0, "mg/dL"), "hba1c": (3.5, 15.0, "%"),
        "prior_admissions": (0.0, 12.0, "count"),
    },
    "target_leak": {
        "income": (1000.0, 200000.0, "CNY/month"), "loan_amount": (1000.0, 200000.0, "CNY"),
        "dti": (0.0, 1.2, "ratio"), "credit_score": (300.0, 850.0, "points"),
        "employment_years": (0.0, 45.0, "years"),
        "collections_30d": (0.0, 30.0, "count"), "recovered_amount": (0.0, 500000.0, "CNY"),
        "days_to_first_payment": (0.0, 120.0, "days"),
    },
    "temporal": {
        "dow": (0.0, 6.0, "0=Mon"), "month": (1.0, 12.0, "1-12"), "promo": (0.0, 1.0, "0/1"),
        "price_index": (0.8, 1.2, "ratio"), "temp_c": (-20.0, 45.0, "C"),
        "units": (0.0, 900.0, "units/day"),
    },
    "metric_mismatch": {
        "line_speed": (10.0, 80.0, "units/min"), "operator_tenure": (0.0, 240.0, "months"),
        "ambient_humidity": (0.0, 100.0, "%"), "tool_wear": (0.0, 12.0, "index"),
        "vibration": (0.0, 10.0, "mm/s"), "temperature": (15.0, 50.0, "C"),
        "defect": (0.0, 1.0, "0/1"),
    },
    "confounding": {
        "age": (18.0, 95.0, "years"), "severity": (0.0, 10.0, "score"),
        "comorbidity": (0.0, 10.0, "count"), "bmi": (14.0, 50.0, "kg/m2"),
        "baseline_marker": (0.0, 200.0, "index"), "treated": (0.0, 1.0, "0/1"),
        "recovered": (0.0, 1.0, "0/1"),
    },
    "winner_curse": {
        "tenure_months": (0.0, 100.0, "months"), "monthly_charge": (0.0, 200.0, "CNY"),
        "total_charges": (0.0, 12000.0, "CNY"), "support_calls": (0.0, 20.0, "count"),
        "late_payments": (0.0, 20.0, "count"), "usage_drop": (-1.0, 1.0, "ratio"),
        "churn": (0.0, 1.0, "0/1"),
    },
    "pseudo_id": {
        "record_id": (1.0, 2000.0, "export order, not a customer key"),
        "visits_30d": (0.0, 40.0, "count"), "pages_per_visit": (0.0, 40.0, "pages"),
        "time_on_site_min": (0.0, 150.0, "min"), "email_opens": (0.0, 30.0, "count"),
        "discount_used": (0.0, 1.0, "0/1"), "converted": (0.0, 1.0, "0/1"),
    },
}

TEXT = {
    "wine_dup": ("葡萄酒评分的复检批次：同一酒样被重复录入。", "Synthetic, generated from a documented wine-quality-shaped process. CC-BY-4.0."),
    "sentinel": ("体检回访表：检验系统把缺测写成 999 / -999 / -99。", "Synthetic, modelled on a diabetes readmission sheet. CC-BY-4.0."),
    "target_leak": ("小额贷申请：导出表里留着放款后才产生的催收字段。", "Synthetic, modelled on a lending application flow. CC-BY-4.0."),
    "temporal": ("三家门店两年日销：下一季度的 90 天预测。", "Synthetic, four-season daily demand with a linear trend. CC-BY-4.0."),
    "metric_mismatch": ("产线抽检：每天只能看 5%，指标是预算内的召回。", "Synthetic, modelled on an in-line inspection station. CC-BY-4.0."),
    "confounding": ("观察性队列：医生按病情开药，问该给谁用药。", "Synthetic with known potential outcomes; the counterfactuals are held by the grader. CC-BY-4.0."),
    "winner_curse": ("600 行的流失样本，12 个候选里挑一个，报一个期望 AUC。", "Synthetic, deliberately under-powered. CC-BY-4.0."),
    "pseudo_id": ("CRM 导出表：按结果排序后顺序编号。", "Synthetic, modelled on a marketing export. CC-BY-4.0."),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.path.isdir(LABELS):
        os.makedirs(LABELS)
    meta = {
        "benchmark": "LooksRight",
        "version": "1.0",
        "seed": SEED,
        "labels_dir": "data/_labels",
        "note_zh": (
            "训练文件含目标列；评分文件只含预测时点拿得到的列；目标列与（混杂任务的）潜在结局"
            "放在 data/_labels/ 与 data/_private/，给评分脚本用。对外发布资源包时应把这两处去掉，"
            "由维护方另行保管。"
        ),
        "license": "CC-BY-4.0 (data) / MIT (code)",
        "tasks": {},
    }
    generators = {
        "wine_dup": make_wine_dup,
        "sentinel": make_sentinel,
        "target_leak": make_target_leak,
        "temporal": make_temporal,
        "metric_mismatch": make_metric_mismatch,
        "confounding": make_confounding,
        "winner_curse": make_winner_curse,
        "pseudo_id": make_pseudo_id,
    }

    for i, spec in enumerate(TASKS):
        rng = np.random.default_rng(SEED + 1000 * (i + 1))
        out = generators[spec.task_id](spec, rng)
        truth = None
        if len(out) == 4:
            train, feats, labels, truth = out
        else:
            train, feats, labels = out

        assert len(train) == spec.n_train, (spec.task_id, len(train), spec.n_train)
        assert len(feats) == spec.n_eval, (spec.task_id, len(feats), spec.n_eval)

        d = os.path.join(DATA, spec.task_id)
        if not os.path.isdir(d):
            os.makedirs(d)
        train_path = os.path.join(d, "train.csv")
        feat_path = os.path.join(d, "eval_features.csv")
        lab_path = os.path.join(LABELS, spec.task_id + ".csv")
        train.to_csv(train_path, index=False, float_format="%.4f", encoding="utf-8", lineterminator="\n")
        feats.to_csv(feat_path, index=False, float_format="%.4f", encoding="utf-8", lineterminator="\n")
        labels.to_csv(lab_path, index=False, float_format="%.4f", encoding="utf-8", lineterminator="\n")
        if truth is not None:
            pdir = os.path.join(DATA, "_private")
            if not os.path.isdir(pdir):
                os.makedirs(pdir)
            truth.to_csv(os.path.join(pdir, spec.task_id + "_potential.csv"), index=False,
                         float_format="%.6f", encoding="utf-8", lineterminator="\n")

        cols = []
        for c in feats.columns:
            if c == "row_id":
                cols.append({"name": c, "dtype": "int", "note": "每一行的编号；不是客户/单位编号"})
                continue
            lo, hi, unit = DICT.get(spec.task_id, {}).get(c, (None, None, ""))
            dtype = "str" if feats[c].dtype == object else ("int" if str(feats[c].dtype).startswith("int") else "float")
            entry = {"name": c, "dtype": dtype, "unit": unit}
            if lo is not None:
                entry["plausible_range"] = [lo, hi]
                entry["range_source"] = "data dictionary (设备/业务量程)"
            cols.append(entry)

        meta["tasks"][spec.task_id] = {
            "en": spec.en,
            "zh": spec.zh,
            "target": spec.target,
            "kind": spec.kind,
            "metric": spec.metric,
            "budget": spec.budget,
            "trap": spec.trap,
            "description_zh": TEXT[spec.task_id][0],
            "provenance": TEXT[spec.task_id][1],
            "n_train": int(len(train)),
            "n_eval": int(len(feats)),
            "columns": cols,
            "files": {
                "train_sha256": sha256(train_path),
                "eval_features_sha256": sha256(feat_path),
            },
        }
        print("%-16s train=%-6d eval=%-5d target=%s  %s" % (
            spec.task_id, len(train), len(feats), spec.target, spec.trap))

    meta_path = os.path.join(DATA, "meta.json")
    with io.open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
