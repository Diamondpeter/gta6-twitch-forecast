"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (Python 3.13, stdlib only).

Model: a launch's release day has a characteristic shape once you divide by its
first hour. The case carries dozens of past launches ("comparables"), each with
hourly viewers from 14 days before its release day to 3 days after. For every
comparable whose release-day curve is complete, take the log ratio
ln(viewers[h] / viewers[1]) for h = 2..24. The forecast for this case is hour 1's
print times a robust aggregate of those ratios.

Not all past launches say much about this one. What separates them is how much
of the launch was already live before the UTC day began: a game that unlocks at
midnight local time, region by region, already has New Zealand, Asia and part
of Europe playing in the hours before 00:00 UTC, while a game with a single
late global unlock sits near zero until its hour comes. The feature

    a1 = ln(median viewers in the 6 hours before day_start / hour-1 viewers)

measures exactly that, for the comparables (from their own hourly series) and
for this case (from its history_hourly). Each comparable is weighted by a
Gaussian kernel on a1 (bandwidth 0.35) plus a small floor, doubled when the
comparable is a console-only launch like this one.

Per hour, the five quantiles blend two views of the weighted log-ratio
distribution, in log space: a normal fit (weighted median, weighted IQR scaled
to a standard deviation, shrunk by 0.9) at 70% and the weighted empirical
quantiles at 30%. The normal view is the better centre for launches like this
one; the empirical share keeps honest tails for the hours where past launches
disagreed wildly. The normal spread is widened by sqrt(12 / n_eff), capped at
1.5, where n_eff is the weights' effective sample size: when the kernel
concentrates on a handful of launches, the fit knows less than the IQR of
those few suggests.

Validation while building (leave-one-out over the comparables, and the four
practice cases scored against their actual release days): mean pinball loss on
the log scale about 0.74 / 0.23, against the persistence baseline's 0.93 / 0.39.

The grader runs `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim: standard library only, no network. Outside the container,
MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file could be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Model constants, tuned on leave-one-out over the comparables and checked on
# the four practice cases (see the docstring).
KERNEL_BANDWIDTH = 0.35
KERNEL_FLOOR = 0.02
CONSOLE_BONUS = 2.0
SD_SHRINK = 0.9
EMPIRICAL_SHARE = 0.3


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up; a whole number comes back as an int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def day_curve(comp):
    """{hour: viewers} for hours 1..24 of a comparable's release day."""
    start = parse_ts(comp["day_start"])
    out = {}
    for row in comp.get("hourly") or []:
        try:
            delta = (parse_ts(row["from"]) - start).total_seconds()
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= delta < 24 * 3600 and viewers:
            out[int(delta // 3600) + 1] = viewers
    return out


def pre_day_viewers(comp):
    """A comparable's hourly viewers in the 14 days before its release day."""
    start = parse_ts(comp["day_start"])
    rows = []
    for row in comp.get("hourly") or []:
        try:
            delta = (parse_ts(row["from"]) - start).total_seconds()
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if -14 * 24 * 3600 <= delta < 0 and viewers:
            rows.append((delta, viewers))
    rows.sort()
    return [v for _, v in rows]


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def launch_feature(pre_viewers, first_hour):
    """a1: how live the launch already was in the 6 hours before the day."""
    tail = [v for v in pre_viewers[-6:] if v and v > 0]
    if not tail or not first_hour or first_hour <= 0:
        return None
    return math.log(median(tail) / first_hour)


def weighted_quantile(pairs, tau):
    """Weighted quantile of [(value, weight)], linear within the hit step."""
    if not pairs:
        return 0.0
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    acc = 0.0
    for i, (value, w) in enumerate(pairs):
        acc += w
        if acc / total >= tau:
            if i + 1 < len(pairs) and w > 0:
                frac = (tau * total - (acc - w)) / w
                frac = min(1.0, max(0.0, frac))
                return value + frac * (pairs[i + 1][0] - value)
            return value
    return pairs[-1][0]


def comparable_weight(comp_feature, target_feature, console_only):
    if comp_feature is not None and target_feature is not None:
        d = comp_feature - target_feature
        weight = KERNEL_FLOOR + math.exp(-(d * d) / (2 * KERNEL_BANDWIDTH**2))
    elif target_feature is None:
        weight = KERNEL_FLOOR + 1.0
    else:
        weight = KERNEL_FLOOR
    if console_only:
        weight *= CONSOLE_BONUS
    return weight


def baseline_quantiles(p1, h):
    """The house persistence forecast, used when no comparable is usable."""
    return [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    # This case's own launch feature, from the whole hours before day_start.
    day_start = case["day_start"]
    own_pre = [
        row["viewers"]
        for row in (case.get("history_hourly") or [])
        if row.get("from", "") < day_start and row.get("viewers")
    ]
    target_feature = launch_feature(own_pre, p1)

    # Weighted log-ratio samples per hour from the comparables.
    samples = {h: [] for h in range(2, 25)}
    weights = []
    for comp in case.get("comparables") or []:
        curve = day_curve(comp)
        if len(curve) < 24 or not curve.get(1):
            continue
        feature = launch_feature(pre_day_viewers(comp), curve[1])
        console_only = "PC" not in (comp.get("platforms") or "")
        weight = comparable_weight(feature, target_feature, console_only)
        weights.append(weight)
        base = curve[1]
        for h in range(2, 25):
            if curve.get(h):
                samples[h].append((math.log(curve[h] / base), weight))

    # Effective sample size of the weights: a concentrated kernel means the
    # spread must widen, because fewer launches are really speaking.
    if weights:
        total_w = sum(weights)
        n_eff = total_w * total_w / sum(w * w for w in weights)
        sd_mult = min(1.5, math.sqrt(max(1.0, 12.0 / n_eff)))
    else:
        sd_mult = 1.0

    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        pairs = samples.get(h) or []
        if not pairs:
            qs = baseline_quantiles(p1, h)
        else:
            med = weighted_quantile(pairs, 0.5)
            q25 = weighted_quantile(pairs, 0.25)
            q75 = weighted_quantile(pairs, 0.75)
            sd = max(0.08, (q75 - q25) / 1.349) * SD_SHRINK * sd_mult
            qs = []
            for tau, z in zip(LEVELS, Z):
                normal_log = med + z * sd
                empirical_log = weighted_quantile(pairs, tau)
                blended = (1 - EMPIRICAL_SHARE) * normal_log + EMPIRICAL_SHARE * empirical_log
                qs.append(p1 * math.exp(blended))
        # Enforce the output contract: non-decreasing, finite, in (0, 1e9].
        qs = [min(1e9, max(1e-6, q)) for q in qs]
        qs.sort()
        rows.append({"hour": h, "q": [round2(q) for q in qs]})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
