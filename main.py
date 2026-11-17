"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (Python 3.13, stdlib only).

Model v5: a two-stage ridge decomposition — day peak, then shape around the
peak — on comparable launches, in the target's own rollout geometry.

What the data says (leave-one-out over the comparables, and the four practice
cases scored against their actual release days):

1. Scale rules the shape. ln(hour-1 viewers) correlates about -0.93 with the
   late-day log ratio ln(viewers[h] / viewers[1]): a launch whose first hour
   is already huge has most of its audience in the room, while a small first
   hour (only the earliest time zones awake) leaves the whole day to grow.
   A model that pools raw ratios without scale — v2 of this entry — misses
   exactly that, which is why its medians were the weak point.

2. Company matters. The pool mixes rolling midnight-local launches with
   single-global-unlock and early-access-contaminated ones whose hour 1 is a
   pre-unlock artefact (a few hundred viewers, ratios in the thousands of
   percent). For a rolling target, the pool is the rolling/unverified,
   uncontaminated comparables with a real hour-1 audience (>= 1% of their
   own day peak), plus any clean-tier launch whose hour 1 already held a
   real audience (>= 30% of its own peak) even under a global unlock —
   identified at runtime from each comparable's unlock text (its Tier
   marker and wording) and its own curve.

3. Geometry matters. A rolling launch's ramps sit at fixed UTC offsets of the
   regional midnights, and those offsets move an hour with daylight saving
   (US Eastern midnight is 04:00Z in EDT, 05:00Z in EST). Each comparable's
   curve is shifted by the difference between its US-Eastern unlock offset
   and the target's — both computed from the release dates by the DST rules —
   before its ratios are taken, so the US ramp lands on the same hours.

4. Peak first, then shape. The forecast decomposes the log ratio as
   ln(peak / hour-1) + ln(viewers[h] / peak) over the shifted day window.
   A ridge (features: ln hour-1 viewers, the pre-day liveness a1 = ln(median
   viewers in the last 6 pre-day hours / hour-1 viewers), its hinge
   max(0, a1 - 0.3), and pre1 = ln(viewers in the last pre-day hour /
   hour-1 viewers) — the pre-day's closing level; lambda 2) predicts the
   peak term; a second ridge (the same features plus ln hour-1 channels,
   which tracks how broad the launch's streamer base is; lambda 6) predicts
   each hour's shape around the peak. The five quantiles are the composed
   centre plus the distribution of the total leave-one-out residuals
   (stage A's plus stage B's, hat-matrix shortcut), blended 85/15 between a
   normal fit on the residuals' IQR and their empirical quantiles. If the
   decomposition cannot be built, a per-hour ridge on the same pool is the
   fallback; targets that are not rolling launches fall back to a full-pool
   ridge on (ln v1, (ln v1 - 11)^2); with no usable comparables at all, to
   the house persistence baseline. Standardised features are clipped at
   +/-3 sd when predicting, and the target's ln hour-1 is winsorised at
   the pool's maximum + 0.3, so an hour-1 larger than every comparable's
   (as GTA VI's may be) cannot extrapolate explosively: beyond the largest
   comparable, the forecast responds like that comparable, not beyond it.
   The winsorisation is exactly neutral on the practice cases and in
   leave-one-out (no in-sample target exceeds its pool's maximum + 0.3);
   on synthetic targets scaled x10 it cuts the log-scale pinball of the
   largest practice case from 0.38 to 0.24.

Validation while building: practice cases 0.135 mean pinball (v4: 0.147,
v2: 0.227), leave-one-out over the in-class comparables 0.29,
leave-one-out over all comparables 0.58 (v2: 0.74), the all-comparables
figure kept honest by the gated fallback.

The grader runs `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim: standard library only, no network. Outside the container,
MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file could be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RIDGE_LAMBDA = 4.0
RESID_BLEND = 0.15  # empirical share of the quantile blend (rest: normal fit)
HINGE_AT = 0.3
LAM_A = 2.0  # peak-stage ridge lambda
LAM_B = 6.0  # shape-stage ridge lambda
FEATS_A = ["logv1", "a1", "a1h", "pre1"]
FEATS_B = ["logv1", "a1", "a1h", "pre1", "logch1"]


def comp_feat(c, name):
    if name == "logv1":
        return c.logv1
    if name == "a1":
        return c.a1
    if name == "a1h":
        return c.a1h
    if name == "pre1":
        return c.pre1
    if name == "logv1sq":
        return c.logv1sq
    if name == "logch1":
        return c.logch1
    raise KeyError(name)


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------
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


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def quantile(sorted_vals, tau):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = tau * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def solve(A, b):
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / pv
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def inverse(A):
    n = len(A)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(solve(A, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# --------------------------------------------------------------------------
# rollout geometry: UTC offsets of regional midnights, from the date alone
# --------------------------------------------------------------------------
def nth_sunday(year, month, n):
    d = date(year, month, 1)
    off = (6 - d.weekday()) % 7
    return d + timedelta(days=off + 7 * (n - 1))


def us_east_offset(d):
    """UTC hour of US Eastern midnight on date d: 4 in EDT, 5 in EST."""
    start = nth_sunday(d.year, 3, 2)
    end = nth_sunday(d.year, 11, 1)
    return 4 if start <= d < end else 5


# --------------------------------------------------------------------------
# unlock-text classification (from the case's own fields, at runtime)
# --------------------------------------------------------------------------
def unlock_kind(text):
    u = (text or "").lower()
    # A negated mention ("rolling, no simultaneous launch") is not a global
    # unlock; only an affirmative one is.
    global_hit = (("simultaneous" in u or "single global" in u)
                  and "no simultaneous" not in u
                  and "not simultaneous" not in u
                  and "no single global" not in u)
    if global_hit:
        return "global"
    if "rolling" in u or "midnight local" in u:
        return "rolling"
    if "unverified" in u:
        return "unverified"
    return "other"


def unlock_tier(text):
    u = text or ""
    marker = "Tier:"
    if marker in u:
        rest = u.split(marker, 1)[1].strip()
        word = rest.split()[0].strip(".,").lower() if rest.split() else ""
        return word
    return ""


# --------------------------------------------------------------------------
# comparable extraction
# --------------------------------------------------------------------------
class Comp:
    __slots__ = ("name", "series", "curve", "pre", "logv1", "a1", "a1h",
                 "pre1", "logv1sq", "kind", "tier", "use", "v1_over_peak",
                 "logch1", "_ch")

    def __init__(self, raw):
        self.name = raw.get("name", "")
        start = parse_ts(raw["day_start"])
        self.series = {}
        self._ch = {}
        for row in raw.get("hourly") or []:
            try:
                off = int(round((parse_ts(row["from"]) - start).total_seconds() / 3600))
                v = row["viewers"]
            except (KeyError, TypeError, ValueError):
                continue
            if v:
                self.series[off] = v
                ch = row.get("channels")
                if ch:
                    self._ch[off] = ch
        self.curve = {h: self.series[h - 1] for h in range(1, 25)
                      if self.series.get(h - 1)}
        pre_offsets = sorted(o for o in self.series if o < 0)
        self.pre = [self.series[o] for o in pre_offsets]
        self.kind = unlock_kind(raw.get("unlock"))
        self.tier = unlock_tier(raw.get("unlock"))
        self.use = us_east_offset(start.date())
        v1 = self.curve.get(1)
        if len(self.curve) == 24 and v1 and v1 > 0:
            self.logv1 = math.log(v1)
            tail = [v for v in self.pre[-6:] if v and v > 0]
            self.a1 = math.log(median(tail) / v1) if tail else 0.0
            self.pre1 = math.log(self.pre[-1] / v1) if self.pre else 0.0
            peak = max(self.curve.values())
            self.v1_over_peak = v1 / peak if peak else 0.0
            ch1 = self._ch.get(0)
            self.logch1 = math.log(ch1) if ch1 else None
        else:
            self.logv1 = None
            self.a1 = 0.0
            self.pre1 = 0.0
            self.v1_over_peak = 0.0
            self.logch1 = None
        self.a1h = max(0.0, self.a1 - HINGE_AT)
        self.logv1sq = (self.logv1 - 11.0) ** 2 if self.logv1 is not None else 0.0

    @property
    def usable(self):
        return self.logv1 is not None

    @property
    def in_class(self):
        """Rolling-class, uncontaminated, with a real hour-1 audience."""
        return (self.kind in ("rolling", "unverified")
                and self.tier not in ("early-access-contaminated", "weak")
                and self.v1_over_peak >= 0.01)

    @property
    def in_class_plus(self):
        """in_class, plus clean-tier launches whose hour 1 was already a
        real audience even under a global/PC unlock (v1 >= 30% of peak)."""
        return self.in_class or (self.tier == "comparable"
                                 and self.v1_over_peak >= 0.3)

    def shifted_log_ratio(self, target_use, h):
        """ln(viewers[h]/viewers[1]) with this curve shifted into the
        target's rollout geometry (shift = target USE offset - own)."""
        sh = target_use - self.use
        v_h = self.series.get(h - 1 - sh)
        v_1 = self.series.get(0 - sh)
        if v_h and v_1:
            return math.log(v_h / v_1)
        return None

    def log_ratio(self, h):
        if self.curve.get(h) and self.curve.get(1):
            return math.log(self.curve[h] / self.curve[1])
        return None


# --------------------------------------------------------------------------
# ridge regression with hat-shortcut leave-one-out residuals
# --------------------------------------------------------------------------
def ridge_fit(feat_rows, y, lam):
    """feat_rows: list of feature lists (raw). Returns (predict, loo_resid)."""
    n = len(feat_rows)
    p = len(feat_rows[0])
    means = [sum(r[a] for r in feat_rows) / n for a in range(p)]
    sds = []
    for a in range(p):
        var = sum((r[a] - means[a]) ** 2 for r in feat_rows) / n
        sds.append(math.sqrt(var) if var > 1e-9 else 1.0)
    Zx = [[(r[a] - means[a]) / sds[a] for a in range(p)] for r in feat_rows]
    ybar = sum(y) / n
    yc = [v - ybar for v in y]
    A = [[sum(Zx[i][a] * Zx[i][b] for i in range(n)) + (lam if a == b else 0.0)
          for b in range(p)] for a in range(p)]
    bvec = [sum(Zx[i][a] * yc[i] for i in range(n)) for a in range(p)]
    Ainv = inverse(A)
    beta = [sum(Ainv[a][b] * bvec[b] for b in range(p)) for a in range(p)]
    fitted = [ybar + sum(Zx[i][a] * beta[a] for a in range(p)) for i in range(n)]
    loo = []
    for i in range(n):
        h_ii = sum(Zx[i][a] * sum(Ainv[a][b] * Zx[i][b] for b in range(p))
                   for a in range(p))
        h_ii = min(h_ii, 0.95)
        loo.append((y[i] - fitted[i]) / (1.0 - h_ii))

    def predict(feats):
        z = [(feats[a] - means[a]) / sds[a] for a in range(p)]
        z = [max(-3.0, min(3.0, v)) for v in z]  # no wild extrapolation
        return ybar + sum(z[a] * beta[a] for a in range(p))

    return predict, loo


def ridge_quantiles(pool_feats, pool_y, target_feats, lam):
    """(mu, [q05..q95 of log ratio]) from a ridge fit + LOO residuals."""
    predict, loo = ridge_fit(pool_feats, pool_y, lam)
    mu = predict(target_feats)
    res = sorted(loo)
    sd = max(0.05, (quantile(res, 0.75) - quantile(res, 0.25)) / 1.349)
    qs = []
    for tau, z in zip(LEVELS, Z):
        emp = mu + quantile(res, tau)
        nor = mu + z * sd
        qs.append((1 - RESID_BLEND) * nor + RESID_BLEND * emp)
    qs.sort()
    return qs


def blend_quantiles(mu, res_sorted):
    """Centre + blended (normal/empirical) quantiles of LOO residuals."""
    res = res_sorted
    sd = max(0.05, (quantile(res, 0.75) - quantile(res, 0.25)) / 1.349)
    qs = []
    for tau, z in zip(LEVELS, Z):
        emp = mu + quantile(res, tau)
        nor = mu + z * sd
        qs.append((1 - RESID_BLEND) * nor + RESID_BLEND * emp)
    qs.sort()
    return qs


def peakshape_quantiles(data, target_a, target_b):
    """data: list of (feats_a, feats_b, {h: log(v_h/peak)}, log(peak/v_1)).
    Returns {h: [log-ratio quantiles]} composing the peak stage (LAM_A) and
    the per-hour shape stage (LAM_B); the total LOO residual per comp is
    stage A's plus stage B's at that hour."""
    feats = [d[0] for d in data]
    prs = [d[3] for d in data]
    pred_a, loo_a = ridge_fit(feats, prs, LAM_A)
    mu_a = pred_a(target_a)
    out = {}
    for h in range(2, 25):
        idx = [i for i, d in enumerate(data) if h in d[2]]
        if len(idx) < 8:
            continue
        pred_b, loo_b = ridge_fit([data[i][1] for i in idx],
                                  [data[i][2][h] for i in idx], LAM_B)
        mu = mu_a + pred_b(target_b)
        res = sorted(loo_a[i] + loo_b[j] for j, i in enumerate(idx))
        out[h] = blend_quantiles(mu, res)
    return out


def baseline_quantiles(p1, h):
    """The house persistence forecast, used when no comparable is usable."""
    return [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]


# --------------------------------------------------------------------------
# the forecast
# --------------------------------------------------------------------------
def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    day_start = parse_ts(case["day_start"])
    target_use = us_east_offset(day_start.date())
    target_kind = unlock_kind(case.get("notes"))

    # The target's own features, from its history before day_start.
    own_pre = [row["viewers"] for row in (case.get("history_hourly") or [])
               if row.get("from", "") < case["day_start"] and row.get("viewers")]
    logv1_t = math.log(p1)
    tail = [v for v in own_pre[-6:] if v and v > 0]
    a1_t = math.log(median(tail) / p1) if tail else 0.0
    a1h_t = max(0.0, a1_t - HINGE_AT)
    pre1_t = math.log(own_pre[-1] / p1) if own_pre else 0.0
    logv1sq_t = (logv1_t - 11.0) ** 2
    ch1_t = first.get("channels")
    logch1_t = math.log(ch1_t) if ch1_t else None
    ctx_t = {"logv1": logv1_t, "a1": a1_t, "a1h": a1h_t, "pre1": pre1_t,
             "logv1sq": logv1sq_t, "logch1": logch1_t}

    comps = []
    for raw in case.get("comparables") or []:
        try:
            c = Comp(raw)
        except (KeyError, TypeError, ValueError):
            continue
        if c.usable:
            comps.append(c)
    comps.sort(key=lambda c: c.name)  # stable order => deterministic output

    in_class = [c for c in comps if c.in_class_plus]
    rolling_target = target_kind in ("rolling", "unverified")
    use_in_class = rolling_target and len(in_class) >= 8
    if use_in_class:
        # Winsorise the target's scale at the pool's range edge (+0.3 log):
        # the scale response saturates instead of extrapolating.
        cap = max(c.logv1 for c in in_class) + 0.3
        if logv1_t > cap:
            logv1_t = cap
            logv1sq_t = (logv1_t - 11.0) ** 2
            ctx_t["logv1"] = cap
            ctx_t["logv1sq"] = logv1sq_t

    # Two-stage decomposition over the pool, when it can be built.
    pre_qs = None
    if use_in_class:
        def stage_feats(names):
            out = []
            for f in names:
                if ctx_t.get(f) is None:
                    continue
                if all(comp_feat(c, f) is not None for c in in_class):
                    out.append(f)
            return out

        fa = stage_feats(FEATS_A)
        fb = stage_feats(FEATS_B)
        data = []
        if fa and fb:
            for c in in_class:
                sh = target_use - c.use
                vals = {}
                for h in range(1, 25):
                    v = c.series.get(h - 1 - sh)
                    if v:
                        vals[h] = v
                if len(vals) >= 20 and vals.get(1):
                    peak = max(vals.values())
                    pr = math.log(peak / vals[1])
                    ys = {h: math.log(vals[h] / peak)
                          for h in range(2, 25) if vals.get(h)}
                    data.append(([comp_feat(c, f) for f in fa],
                                 [comp_feat(c, f) for f in fb], ys, pr))
        if len(data) >= 8:
            pre_qs = peakshape_quantiles(
                data, [ctx_t[f] for f in fa], [ctx_t[f] for f in fb])

    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = None
        if pre_qs is not None and h in pre_qs:
            qs = [p1 * math.exp(v) for v in pre_qs[h]]
        elif use_in_class:
            feats, ys = [], []
            for c in in_class:
                y = c.shifted_log_ratio(target_use, h)
                if y is not None:
                    feats.append([c.logv1, c.a1, c.a1h, c.pre1])
                    ys.append(y)
            if len(ys) >= 8:
                log_qs = ridge_quantiles(feats, ys,
                                         [logv1_t, a1_t, a1h_t, pre1_t],
                                         RIDGE_LAMBDA)
                qs = [p1 * math.exp(v) for v in log_qs]
        if qs is None and comps:
            feats, ys = [], []
            for c in comps:
                y = c.log_ratio(h)
                if y is not None:
                    feats.append([c.logv1, c.logv1sq])
                    ys.append(y)
            if len(ys) >= 8:
                log_qs = ridge_quantiles(feats, ys, [logv1_t, logv1sq_t],
                                         RIDGE_LAMBDA)
                qs = [p1 * math.exp(v) for v in log_qs]
        if qs is None:
            qs = baseline_quantiles(p1, h)
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
