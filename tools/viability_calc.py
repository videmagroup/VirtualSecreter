#!/usr/bin/env python3
"""Viability threshold calculator for the virtual clinic secretary.

Reproduces the numbers in docs/viability-thresholds.md so they can be
re-run when the dollar rate, wages or measured model performance change.
Money is in million toman unless a name says otherwise.

Example:
    python3 tools/viability_calc.py --fx 300000 --gross 30
"""

import argparse

# Cloud voice pipelines, USD per conversation minute (October 2026 prices).
CLOUD_PIPELINES = {
    "cascaded ASR+LLM+TTS": 0.02,
    "realtime mini": 0.035,
    "realtime flagship": 0.085,
}

# Hardware cost ranges in USD-equivalent Iranian street prices.
MVP_HARDWARE_USD = (1_350, 2_385)
FULL_HARDWARE_USD = (2_730, 4_920)
SHARED_SERVER_USD = 6_730
CLINICS_PER_SERVER = 25

# Per-clinic monthly costs that do not depend on the dollar rate.
COLOCATION, SMS, SUPPORT = 1.0, 0.8, 1.5


def employer_cost(gross, min_wage):
    """Full monthly cost of a formally employed secretary."""
    insurance = 0.23 * gross
    eidi = min(2 * gross, 3 * min_wage) / 12
    sanavat = gross / 12
    return gross + insurance + eidi + sanavat


def to_usd(million_toman, fx):
    return million_toman * 1e6 / fx


def custom_build_monthly(dev_pm, hardware_usd, fx, dev_pm_cost, months=36):
    """Monthly-equivalent (low, high) cost of a single-clinic custom build."""
    result = []
    for pm, hw_usd, maint_fte in zip(dev_pm, hardware_usd, (0.25, 0.5)):
        dev = pm * dev_pm_cost
        hardware = hw_usd * fx / 1e6
        result.append((dev + hardware) / months + maint_fte * dev_pm_cost)
    return tuple(result)


def saas_variable_cost(fx):
    server_share = SHARED_SERVER_USD * fx / 1e6 / CLINICS_PER_SERVER / 36
    return server_share + COLOCATION + SMS + SUPPORT


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fx", type=float, default=260_000, help="toman per USD")
    p.add_argument("--gross", type=float, default=25, help="secretary gross pay, M toman/month")
    p.add_argument("--min-wage", type=float, default=16.6255, help="monthly minimum wage, M toman")
    p.add_argument("--hours", type=float, default=8, help="paid hours per day")
    p.add_argument("--days", type=float, default=25, help="working days per month")
    p.add_argument("--dev-pm-cost", type=float, default=90, help="loaded developer cost, M toman/month")
    p.add_argument("--team", type=float, default=3, help="product team size after launch")
    p.add_argument("--slot-accuracy", type=float, default=0.90, help="per-attempt slot accuracy")
    p.add_argument("--catch-rate", type=float, default=0.95, help="share of errors caught by read-back")
    p.add_argument("--prefill", type=float, default=150, help="LLM prompt processing, tokens/s")
    p.add_argument("--decode", type=float, default=27, help="LLM generation, tokens/s")
    p.add_argument("--new-tokens", type=float, default=200, help="uncached prompt tokens per turn")
    a = p.parse_args()

    fx = a.fx
    cost = employer_cost(a.gross, a.min_wage)
    paid_minutes = a.hours * 60 * a.days
    per_minute = cost * 1e6 / paid_minutes

    print(f"== Human baseline (FX {fx:,.0f} toman/USD) ==")
    print(f"secretary employer cost: {cost:.1f} M toman/month = ${to_usd(cost, fx):,.0f}")
    print(f"minimum wage: ${to_usd(a.min_wage, fx):,.0f}/month")
    print(f"cost per paid minute: {per_minute:,.0f} toman = ${per_minute / fx:.4f}")

    print("\n== Cloud per-minute parity (sanctions aside, fixed costs ignored) ==")
    for name, usd_per_min in CLOUD_PIPELINES.items():
        breakeven_fx = per_minute / usd_per_min
        parity_usd = usd_per_min * paid_minutes
        print(f"{name:22s} break-even FX <= {breakeven_fx:9,.0f}  "
              f"or secretary cost >= ${parity_usd:,.0f}/month")

    print("\n== Single-clinic custom build, monthly equivalent over 3 years ==")
    for label, dev_pm, hw in (("phone/messaging MVP", (8, 12), MVP_HARDWARE_USD),
                              ("full system", (24, 36), FULL_HARDWARE_USD)):
        lo, hi = custom_build_monthly(dev_pm, hw, fx, a.dev_pm_cost)
        lo0, hi0 = custom_build_monthly(dev_pm, (0, 0), fx, a.dev_pm_cost)
        print(f"{label:20s} {lo:5.1f}-{hi:5.1f} M/month "
              f"(with free hardware: {lo0:5.1f}-{hi0:5.1f}) vs secretary {cost:.1f}")

    print("\n== Product (SaaS) for many clinics ==")
    dev = 20 * a.dev_pm_cost
    team = a.team * a.dev_pm_cost
    var = saas_variable_cost(fx)
    print(f"development {dev:,.0f} M, team {team:,.0f} M/month, variable {var:.1f} M/clinic/month")
    for price in (8, 10, 12, 15, 20):
        margin = price - var
        clinics = team / margin if margin > 0 else float("inf")
        minutes = price * 1e6 / per_minute
        paybacks = []
        for n in (50, 100):
            contribution = n * margin - team
            paybacks.append(f"N={n}: {dev / contribution:5.1f} mo" if contribution > 0 else f"N={n}: never")
        print(f"price {price:4.1f} M: break-even {clinics:5.0f} clinics; "
              f"parity with human at >= {minutes:,.0f} handled min/month; " + ", ".join(paybacks))

    print("\n== Outcome accuracy from read-back confirmation ==")
    for slot in sorted({0.80, 0.85, 0.90, 0.95, a.slot_accuracy}):
        residual = (1 - slot) * (1 - a.catch_rate)
        print(f"slot accuracy {slot:.0%}, catch {a.catch_rate:.0%} -> outcome accuracy {1 - residual:.2%}")

    print("\n== Turn latency on given LLM speeds ==")
    asr_final, first_audio, first_sentence = 0.3, 0.2, 15
    ttft = a.new_tokens / a.prefill + first_sentence / a.decode
    print(f"LLM time to first sentence: {ttft:.2f} s; end-to-end turn: {ttft + asr_final + first_audio:.2f} s "
          f"(target <= 1.5 s)")

    print("\n== Secretary cost in USD by dollar rate and next-year wage growth ==")
    print("FX \\ wage growth   +0%     +45%    +60%")
    for rate in (150_000, 200_000, 260_000, 350_000, 450_000):
        row = [to_usd(employer_cost(a.gross * g, a.min_wage * g), rate) for g in (1.0, 1.45, 1.6)]
        print(f"{rate:9,.0f}       " + "  ".join(f"${v:5,.0f}" for v in row))


if __name__ == "__main__":
    main()
