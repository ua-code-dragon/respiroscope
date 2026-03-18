from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any, Iterable, Sequence, Tuple, Dict, List


@dataclass(frozen=True)
class OneWayAnovaResult:
    k: int
    N: int
    df_between: int
    df_within: int
    grand_mean: float
    ss_between: float
    ss_within: float
    ss_total: float
    ms_between: float
    ms_within: float
    F: float
    p_value: float
    eta2: float      # SS_between / SS_total
    omega2: float    # (SS_between - df_between*MS_within) / (SS_total + MS_within)


def _ffunction(F: float, dfn: int, dfd: int) -> float:
    """
    Survival function for F distribution: P(F_{dfn,dfd} >= F).
    Tries SciPy first; falls back to mpmath if SciPy isn't available.
    """
    if F < 0 or dfn <= 0 or dfd <= 0:
        return float("nan")

    try:
        from scipy.stats import f  # type: ignore
        return float(f.sf(F, dfn, dfd))
    except Exception:
        # F CDF can be expressed via regularized incomplete beta:
        # CDF = I_{ (dfn*F)/(dfn*F+dfd) }(dfn/2, dfd/2)
        try:
            import mpmath as mp  # type: ignore
            x = (dfn * F) / (dfn * F + dfd)
            a = dfn / 2.0
            b = dfd / 2.0
            cdf = mp.betainc(a, b, 0, x, regularized=True)
            return float(1 - cdf)
        except Exception:
            raise ImportError("Need scipy or mpmath installed to compute the F-test p-value.")


def one_way_anova(
    factors: Sequence[Any],
    responses: Sequence[float],
    *,
    drop_nan: bool = True,
) -> OneWayAnovaResult:
    """
    One-way ANOVA for association between a categorical factor and a numeric response.

    Inputs
    - factors: list/array of group labels (categorical)
    - responses: list/array of numeric values (same length as factors)

    Returns: SSB, SSW, F, p-value, and effect sizes (eta^2, omega^2).
    Works with unequal group sizes (unbalanced).

    Notes
    - Classic ANOVA assumes independence, approx normal residuals, and equal variances.
    """
    if len(factors) != len(responses):
        raise ValueError("factors and responses must have the same length.")
    if len(factors) < 2:
        raise ValueError("Need at least 2 observations.")

    # Group data
    groups: Dict[Any, List[float]] = {}
    for g, y in zip(factors, responses):
        if drop_nan and (y is None or (isinstance(y, float) and isnan(y))):
            continue
        groups.setdefault(g, []).append(float(y))

    if len(groups) < 2:
        raise ValueError("Need at least 2 non-empty groups after filtering.")
    if any(len(v) == 0 for v in groups.values()):
        raise ValueError("Found an empty group (unexpected).")

    k = len(groups)
    ns = {g: len(v) for g, v in groups.items()}
    N = sum(ns.values())

    if N <= k:
        raise ValueError(f"Insufficient data: need N > k, got N={N}, k={k}.")

    # Means
    means = {g: (sum(v) / len(v)) for g, v in groups.items()}
    grand_mean = sum(ns[g] * means[g] for g in groups) / N

    # Sums of squares
    ss_between = sum(ns[g] * (means[g] - grand_mean) ** 2 for g in groups)
    ss_within = 0.0
    for g, vals in groups.items():
        m = means[g]
        ss_within += sum((y - m) ** 2 for y in vals)
    ss_total = ss_between + ss_within  # numerically stable partition

    # Degrees of freedom
    df_between = k - 1
    df_within = N - k

    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # F and p-value
    F = ms_between / ms_within if ms_within > 0 else float("inf")
    p_value = _ffunction(F, df_between, df_within) if ms_within > 0 else 0.0

    # Effect sizes
    eta2 = (ss_between / ss_total) if ss_total > 0 else float("nan")
    omega2_num = ss_between - df_between * ms_within
    omega2_den = ss_total + ms_within
    omega2 = (omega2_num / omega2_den) if omega2_den > 0 else float("nan")

    return OneWayAnovaResult(
        k=k,
        N=N,
        df_between=df_between,
        df_within=df_within,
        grand_mean=grand_mean,
        ss_between=ss_between,
        ss_within=ss_within,
        ss_total=ss_total,
        ms_between=ms_between,
        ms_within=ms_within,
        F=F,
        p_value=p_value,
        eta2=eta2,
        omega2=omega2,
    )


# --- Example usage ---
if __name__ == "__main__":
    factors = ["A", "A", "A", "B", "B", "C", "C", "C", "C"]
    responses = [10, 9, 11, 6, 7, 13, 12, 14, 13]

    res = one_way_anova(factors, responses)
    print(res)
    print("SSB:", res.ss_between)
    print("SSW:", res.ss_within)
    print("F:", res.F)
    print("p:", res.p_value)
    print("eta^2:", res.eta2)
    print("omega^2:", res.omega2)

