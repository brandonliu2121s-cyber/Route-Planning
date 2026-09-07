"""
PIL BUNKERING DECISION-SUPPORT ENGINE
=======================================
MaritimeONE Case Summit 2026 -- PIL Challenge: Optimising Bunkering Strategy

A single, self-contained pipeline covering:
  1. Route & synthetic price data generation
  2. ML price forecasting (Gradient Boosted Trees, per port)
  3. Bunkering optimisation (Dynamic Programming over a discretised
     fuel-state space -- a shortest-path formulation)
  4. Baseline strategies for comparison
  5. Monte Carlo simulation & scenario stress tests
  6. Charts + results.json for the presentation deck / dashboard

Run directly:  python3 pil_bunkering_optimizer.py
"""

import json
import os
import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

RNG_SEED = 42
np.random.seed(RNG_SEED)
plt.rcParams["figure.dpi"] = 140
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


# ===========================================================================
# SECTION 1 -- VESSEL, ROUTE & SYNTHETIC PRICE DATA
# ===========================================================================
# Route chosen: PIL's South West Africa Service (SWS), Far East -> Africa,
# as referenced in the case briefing for the "O"-Class vessels.
# All figures (consumption rate, tank capacity, price levels) are
# illustrative assumptions, clearly flagged as such.

VESSEL = {
    "name": "Kota Oasis (O-Class)",
    "service_speed_knots": 18.5,
    "daily_consumption_mt": 180.0,
    "tank_capacity_mt": 4000.0,
    "safety_buffer_mt": 500.0,
    "starting_fuel_mt": 1800.0,
    "fuel_grade": "VLSFO",
}

ROUTE = [
    {"port": "Shanghai",   "country": "China",        "leg_distance_nm": 0,    "base_price": 620, "volatility": 0.018, "supply_risk": 0.10},
    {"port": "Singapore",  "country": "Singapore",    "leg_distance_nm": 2360, "base_price": 598, "volatility": 0.015, "supply_risk": 0.05},
    {"port": "Colombo",    "country": "Sri Lanka",    "leg_distance_nm": 1590, "base_price": 614, "volatility": 0.020, "supply_risk": 0.12},
    {"port": "Durban",     "country": "South Africa", "leg_distance_nm": 3700, "base_price": 651, "volatility": 0.028, "supply_risk": 0.22},
    {"port": "Cape Town",  "country": "South Africa", "leg_distance_nm": 800,  "base_price": 644, "volatility": 0.026, "supply_risk": 0.20},
    {"port": "Tema",       "country": "Ghana",        "leg_distance_nm": 3300, "base_price": 683, "volatility": 0.035, "supply_risk": 0.30},
]

route_df = pd.DataFrame(ROUTE)
route_df["cum_distance_nm"] = route_df["leg_distance_nm"].cumsum()
route_df["leg_days"] = route_df["leg_distance_nm"] / (VESSEL["service_speed_knots"] * 24)
route_df["cum_days"] = route_df["leg_days"].cumsum()
route_df["leg_consumption_mt"] = route_df["leg_days"] * VESSEL["daily_consumption_mt"]


def generate_price_history(n_hist_days=180, n_future_days=30):
    """Synthetic daily VLSFO price series per port: trend + weekly
    seasonality + AR(1)-ish volatility clustering + random supply-risk
    shocks. Returns (history_df, future_actual_df); future_actual is the
    'ground truth' used to score forecast accuracy honestly."""
    all_hist, all_future = [], []
    total_days = n_hist_days + n_future_days
    t = np.arange(total_days)

    for _, row in route_df.iterrows():
        port = row["port"]
        base, vol, risk = row["base_price"], row["volatility"], row["supply_risk"]

        trend = base + 0.04 * t + 6 * np.sin(2 * np.pi * t / 120)
        weekly = 3 * np.sin(2 * np.pi * t / 7)

        noise = np.zeros(total_days)
        eps = np.random.normal(0, base * vol, total_days)
        for i in range(1, total_days):
            noise[i] = 0.6 * noise[i - 1] + eps[i]

        shocks = np.zeros(total_days)
        shock_days = np.random.rand(total_days) < (risk * 0.01)
        shock_magnitude = np.random.normal(base * 0.12, base * 0.04, total_days)
        shocks[shock_days] = shock_magnitude[shock_days]
        for i in range(1, total_days):
            shocks[i] += 0.5 * shocks[i - 1] if not shock_days[i] else 0

        price = np.clip(trend + weekly + noise + shocks, base * 0.6, base * 1.8)
        dates = pd.date_range("2026-01-01", periods=total_days, freq="D")
        df = pd.DataFrame({"date": dates, "port": port, "price_usd_per_mt": price, "is_shock_day": shock_days})
        all_hist.append(df.iloc[:n_hist_days])
        all_future.append(df.iloc[n_hist_days:])

    return pd.concat(all_hist, ignore_index=True), pd.concat(all_future, ignore_index=True)


# ===========================================================================
# SECTION 2 -- ML PRICE FORECASTING (Gradient Boosted Trees, quantile)
# ===========================================================================

FEATURES = ["lag_1", "lag_3", "lag_7", "lag_14", "roll_mean_7", "roll_std_7",
            "roll_mean_30", "day_of_week", "day_of_year", "t"]


def _make_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    for lag in [1, 3, 7, 14]:
        df[f"lag_{lag}"] = df["price_usd_per_mt"].shift(lag)
    df["roll_mean_7"] = df["price_usd_per_mt"].shift(1).rolling(7).mean()
    df["roll_std_7"] = df["price_usd_per_mt"].shift(1).rolling(7).std()
    df["roll_mean_30"] = df["price_usd_per_mt"].shift(1).rolling(30).mean()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["t"] = np.arange(len(df))
    return df


def train_and_forecast(history_df, route_df, vessel, quantiles=(0.1, 0.5, 0.9)):
    """Per port: train quantile GBR models, recursively forecast forward to
    that port's scheduled arrival day, return point + 80% interval forecast."""
    results = []
    for _, r in route_df.iterrows():
        port = r["port"]
        arrival_day_offset = int(round(r["cum_days"]))

        port_hist = history_df[history_df["port"] == port].copy()
        feat_df = _make_features(port_hist).dropna().reset_index(drop=True)
        X_train, y_train = feat_df[FEATURES], feat_df["price_usd_per_mt"]

        q_models = {}
        for q in quantiles:
            model = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=150,
                                               max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(X_train, y_train)
            q_models[q] = model

        working = port_hist[["date", "price_usd_per_mt"]].copy()
        last_date = working["date"].max()
        median_model = q_models[0.5]
        for step in range(1, arrival_day_offset + 1):
            tmp = working.copy()
            tmp["port"] = port
            feat = _make_features(tmp).iloc[[-1]]
            next_price = median_model.predict(feat[FEATURES])[0]
            next_date = last_date + pd.Timedelta(days=step)
            working = pd.concat([working, pd.DataFrame({"date": [next_date], "price_usd_per_mt": [next_price]})], ignore_index=True)

        tmp = working.copy()
        tmp["port"] = port
        feat_final = _make_features(tmp).iloc[[-1]][FEATURES]
        p10 = q_models[0.1].predict(feat_final)[0]
        p50 = q_models[0.5].predict(feat_final)[0]
        p90 = q_models[0.9].predict(feat_final)[0]
        p10, p90 = min(p10, p90), max(p10, p90)

        arrival_date = last_date + pd.Timedelta(days=arrival_day_offset)
        results.append({"port": port, "arrival_day_offset": arrival_day_offset, "arrival_date": arrival_date,
                         "forecast_p10": round(p10, 2), "forecast_p50": round(p50, 2), "forecast_p90": round(p90, 2)})

    return pd.DataFrame(results)


def backtest_accuracy(forecast_df, future_actual_df):
    """Honest MAPE: forecast_p50 vs synthetic ground truth (possible only
    because we generated the data ourselves)."""
    rows = []
    for _, r in forecast_df.iterrows():
        actual_row = future_actual_df[(future_actual_df["port"] == r["port"]) & (future_actual_df["date"] == r["arrival_date"])]
        if not actual_row.empty:
            actual = actual_row["price_usd_per_mt"].values[0]
            err = abs(actual - r["forecast_p50"]) / actual * 100
            rows.append({"port": r["port"], "actual": round(actual, 2), "forecast": r["forecast_p50"], "abs_pct_error": round(err, 2)})
    return pd.DataFrame(rows)


# ===========================================================================
# SECTION 3 -- OPTIMISATION ALGORITHM (Dynamic Programming)
# ===========================================================================
# Classic "fuel stop" / shortest-path problem, generalised with a tank
# capacity ceiling and a safety-buffer floor. State = fuel level on board
# immediately after any bunkering decision at a given port.

def _enforce_no_phantom_purchase(result):
    """
    A price >= 1e8 is the sentinel this model uses to mean "unavailable at
    any price" -- either a stress-test port closure, or (new) a port that
    can't supply the vessel's fuel grade at all. It's a solver trick to
    discourage bunkering there, NOT a real transaction price. If a strategy
    still ends up "buying" fuel there because it had no alternative (which
    happened here: an LNG dual-fuel run with an unavoidable unavailable
    port produced a $459-BILLION result before this check existed), that
    strategy is genuinely infeasible -- you cannot actually buy fuel at an
    unavailable port -- so report it as infeasible rather than a real cost.
    Applied uniformly to every strategy function so none of them can
    silently produce this failure mode again.
    """
    if not result["feasible"]:
        return result
    if (result["plan"]["bunker_mt"] > 0.5).any() and (result["plan"]["price_usd_per_mt"] >= 1e8).any():
        stuck = result["plan"][(result["plan"]["bunker_mt"] > 0.5) & (result["plan"]["price_usd_per_mt"] >= 1e8)]
        if len(stuck):
            return {"feasible": False, "plan": None, "total_cost": None}
    return result


def optimize_bunkering(route_df, prices, vessel, grid_step=10):
    cap, buf = vessel["tank_capacity_mt"], vessel["safety_buffer_mt"]
    states = np.arange(buf, cap + 1e-6, grid_step)
    n_states, n_ports = len(states), len(route_df)
    INF = float("inf")

    dp = np.full((n_ports, n_states), INF)
    parent = np.full((n_ports, n_states), -1, dtype=int)
    bunkered_amt = np.zeros((n_ports, n_states))

    start_fuel = vessel["starting_fuel_mt"]
    p0 = route_df.iloc[0]["port"]
    price0 = prices[p0]
    for s in range(n_states):
        target = states[s]
        if target < start_fuel - 1e-6:
            continue
        bunker = target - start_fuel
        dp[0, s] = bunker * price0
        bunkered_amt[0, s] = bunker

    for i in range(1, n_ports):
        leg_consumption = route_df.iloc[i]["leg_consumption_mt"]
        price = prices[route_df.iloc[i]["port"]]

        for s_prev in range(n_states):
            if dp[i - 1, s_prev] == INF:
                continue
            fuel_arrive = states[s_prev] - leg_consumption
            if fuel_arrive < buf - 1e-6:
                continue
            fuel_arrive = max(fuel_arrive, buf)

            # A "buy nothing" option must always be free -- represented at
            # the grid point AT OR BELOW the true continuous fuel level so
            # the algorithm is never forced into a phantom purchase purely
            # due to discretisation (e.g. at a port priced at infinity).
            s_free = int(np.clip(np.floor((fuel_arrive - buf) / grid_step), 0, n_states - 1))

            cand_cost = dp[i - 1, s_prev] + np.where(
                np.arange(n_states) == s_free, 0.0, np.maximum(states - fuel_arrive, 0.0) * price)
            valid = np.arange(n_states) >= s_free
            improve = valid & (cand_cost < dp[i])
            better_idx = np.where(improve)[0]
            if better_idx.size:
                dp[i, better_idx] = cand_cost[better_idx]
                parent[i, better_idx] = s_prev
                bunkered_amt[i, better_idx] = np.where(better_idx == s_free, 0.0, states[better_idx] - fuel_arrive)

    last = n_ports - 1
    if np.all(dp[last] == INF):
        return {"feasible": False, "plan": None, "total_cost": None}

    tie_break = (states - buf) * 1e-6
    best_s = int(np.argmin(dp[last] + tie_break))
    total_cost = dp[last, best_s]

    plan_rows = []
    s = best_s
    for i in range(last, -1, -1):
        port = route_df.iloc[i]["port"]
        bunker = bunkered_amt[i, s]
        plan_rows.append({"port": port, "bunker_mt": round(bunker, 1), "fuel_after_mt": round(states[s], 1),
                          "price_usd_per_mt": round(prices[port], 2), "cost_usd": round(bunker * prices[port], 2)})
        s = parent[i, s] if i > 0 else -1

    return _enforce_no_phantom_purchase({"feasible": True, "plan": pd.DataFrame(plan_rows).iloc[::-1].reset_index(drop=True), "total_cost": round(total_cost, 2)})


# ---------------------------------------------------------------------------
# Baseline strategies (for the "expected business impact" comparison)
# ---------------------------------------------------------------------------

def baseline_just_in_time(route_df, prices, vessel):
    """Realistic naive baseline: bunker only the minimum needed for the
    NEXT leg (+ buffer), regardless of price -- proxy for planning without
    a price-aware decision-support tool. This is the headline comparison."""
    buf, n = vessel["safety_buffer_mt"], len(route_df)
    fuel = vessel["starting_fuel_mt"]
    rows, total_cost = [], 0
    for i, r in route_df.iterrows():
        port = r["port"]
        if i > 0:
            fuel -= r["leg_consumption_mt"]
            if fuel < buf - 1e-6:
                return {"feasible": False, "plan": None, "total_cost": None}
        next_leg = route_df.iloc[i + 1]["leg_consumption_mt"] if i < n - 1 else 0
        bunker = max(0, buf + next_leg - fuel)
        fuel += bunker
        cost = bunker * prices[port]
        total_cost += cost
        rows.append({"port": port, "bunker_mt": round(bunker, 1), "fuel_after_mt": round(fuel, 1),
                     "price_usd_per_mt": round(prices[port], 2), "cost_usd": round(cost, 2)})
    return _enforce_no_phantom_purchase({"feasible": True, "plan": pd.DataFrame(rows), "total_cost": round(total_cost, 2)})


def baseline_bunker_every_port(route_df, prices, vessel):
    """Naive strategy: top up to full at every port except the last
    (no reason to buy fuel with nowhere to use it within this horizon)."""
    cap, buf, n = vessel["tank_capacity_mt"], vessel["safety_buffer_mt"], len(route_df)
    fuel = vessel["starting_fuel_mt"]
    rows, total_cost = [], 0
    for i, r in route_df.iterrows():
        port = r["port"]
        if i > 0:
            fuel -= r["leg_consumption_mt"]
            if fuel < buf - 1e-6:
                return {"feasible": False, "plan": None, "total_cost": None}
        bunker = (cap - fuel) if i < n - 1 else 0
        fuel = cap if i < n - 1 else fuel
        cost = bunker * prices[port]
        total_cost += cost
        rows.append({"port": port, "bunker_mt": round(bunker, 1), "fuel_after_mt": round(fuel, 1),
                     "price_usd_per_mt": round(prices[port], 2), "cost_usd": round(cost, 2)})
    return _enforce_no_phantom_purchase({"feasible": True, "plan": pd.DataFrame(rows), "total_cost": round(total_cost, 2)})


def baseline_cheapest_port_only(route_df, prices, vessel):
    """Naive strategy: fill up once at whichever port looks cheapest.
    Included because it's the intuitive-but-often-infeasible strategy a
    junior planner might default to."""
    cap, buf = vessel["tank_capacity_mt"], vessel["safety_buffer_mt"]
    cheapest_port = min(prices, key=prices.get)
    fuel = vessel["starting_fuel_mt"]
    rows, total_cost = [], 0
    for i, r in route_df.iterrows():
        port = r["port"]
        if i > 0:
            fuel -= r["leg_consumption_mt"]
        if fuel < buf - 1e-6:
            return {"feasible": False, "plan": None, "total_cost": None}
        bunker = 0
        if port == cheapest_port:
            bunker = cap - fuel
            fuel = cap
        cost = bunker * prices[port]
        total_cost += cost
        rows.append({"port": port, "bunker_mt": round(bunker, 1), "fuel_after_mt": round(fuel, 1),
                     "price_usd_per_mt": round(prices[port], 2), "cost_usd": round(cost, 2)})
    return _enforce_no_phantom_purchase({"feasible": True, "plan": pd.DataFrame(rows), "total_cost": round(total_cost, 2)})

# ===========================================================================
# SECTION 4 -- SCENARIO ANALYSIS (Monte Carlo + stress tests)
# ===========================================================================

def monte_carlo_prices(route_df, n_scenarios=500, seed=7):
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n_scenarios):
        draw = {}
        for _, r in route_df.iterrows():
            base, vol, risk = r["base_price"], r["volatility"], r["supply_risk"]
            price = base * (1 + rng.normal(0, vol * 3))
            if rng.random() < risk * 0.5:
                price *= 1 + abs(rng.normal(0.15, 0.08))
            draw[r["port"]] = max(price, base * 0.5)
        scenarios.append(draw)
    return scenarios


def run_monte_carlo(route_df, vessel, n_scenarios=500):
    scenarios = monte_carlo_prices(route_df, n_scenarios)
    opt_costs, jit_costs = [], []
    for prices in scenarios:
        r_opt = optimize_bunkering(route_df, prices, vessel)
        r_jit = baseline_just_in_time(route_df, prices, vessel)
        if r_opt["feasible"]:
            opt_costs.append(r_opt["total_cost"])
        if r_jit["feasible"]:
            jit_costs.append(r_jit["total_cost"])
    summary = pd.DataFrame({
        "strategy": ["Optimised (DP)", "Just-in-time baseline"],
        "mean_cost": [np.mean(opt_costs), np.mean(jit_costs)],
        "p10_cost": [np.percentile(opt_costs, 10), np.percentile(jit_costs, 10)],
        "p90_cost": [np.percentile(opt_costs, 90), np.percentile(jit_costs, 90)],
        "std_cost": [np.std(opt_costs), np.std(jit_costs)],
    })
    return summary, opt_costs, jit_costs


def stress_test_port_spike(route_df, base_prices, vessel, port, spike_pct=40):
    shocked = dict(base_prices)
    shocked[port] = shocked[port] * (1 + spike_pct / 100)
    return optimize_bunkering(route_df, shocked, vessel), shocked


def stress_test_port_closure(route_df, base_prices, vessel, closed_port):
    shocked = dict(base_prices)
    shocked[closed_port] = 1e9
    return optimize_bunkering(route_df, shocked, vessel)


# ===========================================================================
# SECTION 5 -- MAIN: run everything, produce charts + results.json
# ===========================================================================

def main():
    print("=" * 70)
    print("PIL BUNKERING DECISION-SUPPORT ENGINE")
    print("=" * 70)

    # ---- Data ----
    hist, fut = generate_price_history()
    print(f"\n[1] Generated {len(hist)} days of synthetic price history across {route_df.shape[0]} ports")

    # ---- ML forecast ----
    forecast_df = train_and_forecast(hist, route_df, VESSEL)
    backtest_df = backtest_accuracy(forecast_df, fut)
    mape = backtest_df["abs_pct_error"].mean()
    print(f"\n[2] ML price forecast trained (Gradient Boosted Trees, per port)")
    print(forecast_df[["port", "arrival_date", "forecast_p10", "forecast_p50", "forecast_p90"]])
    print(f"    Backtest MAPE vs synthetic ground truth: {mape:.2f}%")
    forecast_prices = dict(zip(forecast_df["port"], forecast_df["forecast_p50"]))

    # ---- Optimisation ----
    opt_result = optimize_bunkering(route_df, forecast_prices, VESSEL)
    jit_result = baseline_just_in_time(route_df, forecast_prices, VESSEL)
    full_result = baseline_bunker_every_port(route_df, forecast_prices, VESSEL)
    cheap_result = baseline_cheapest_port_only(route_df, forecast_prices, VESSEL)

    savings_vs_jit = jit_result["total_cost"] - opt_result["total_cost"]
    # guard divide-by-zero: if the baseline itself costs $0 (e.g. every
    # price zeroed out), "% saved" is undefined, not 0 -- not reachable via
    # the demo's ML-forecasted prices, but run_custom_route() lets a user
    # zero out PORTS_DB prices directly, so this is a real path, not
    # theoretical.
    savings_pct = (100 * savings_vs_jit / jit_result["total_cost"]) if jit_result["total_cost"] > 0 else None
    savings_pct_str = f"{savings_pct:.1f}%" if savings_pct is not None else "N/A (baseline cost is $0)"

    print(f"\n[3] Optimisation complete (Dynamic Programming, shortest-path over fuel states)")
    print(opt_result["plan"])
    print(f"\n    Optimised total cost:        ${opt_result['total_cost']:,.0f}")
    print(f"    JIT (price-blind) baseline:  ${jit_result['total_cost']:,.0f}")
    print(f"    Savings:                     ${savings_vs_jit:,.0f}  ({savings_pct_str})")
    print(f"    'Fill every port' baseline:  ${full_result['total_cost']:,.0f}")
    print(f"    'Cheapest port only':        {'INFEASIBLE (exceeds tank capacity)' if not cheap_result['feasible'] else cheap_result['total_cost']}")

    # ---- Scenario / Monte Carlo ----
    mc_summary, opt_costs, jit_costs = run_monte_carlo(route_df, VESSEL, n_scenarios=500)
    print(f"\n[4] Monte Carlo (500 price-path scenarios)")
    print(mc_summary.to_string(index=False))

    spike_result, spike_prices = stress_test_port_spike(route_df, forecast_prices, VESSEL, "Durban", 40)
    closure_result = stress_test_port_closure(route_df, forecast_prices, VESSEL, "Cape Town")

    # ---- Charts ----
    _make_charts(hist, forecast_df, opt_result, jit_result, full_result, savings_pct, opt_costs, jit_costs)
    print(f"\n[5] Charts saved to {OUT}")

    # ---- results.json ----
    fleet_impact = fleet_wide_impact(savings_vs_jit)
    print(f"\n[7] Fleet-wide extrapolation (illustrative -- see fleet_wide_impact() docstring for assumptions)")
    print(f"    Per-voyage saving:        ${fleet_impact['per_voyage_savings_usd']:,.0f}")
    print(f"    Vessels applying:         {fleet_impact['vessels_applying']}")
    print(f"    Voyages/vessel/year:      {fleet_impact['voyages_per_vessel_per_year']}")
    print(f"    ==> Estimated annual saving: ${fleet_impact['annual_savings_usd']:,.0f}")

    results = {
        "vessel": VESSEL,
        "route": route_df.to_dict("records"),
        "forecast": forecast_df.assign(arrival_date=forecast_df["arrival_date"].astype(str)).to_dict("records"),
        "backtest_mape": round(mape, 2),
        "optimized_plan": opt_result["plan"].to_dict("records"),
        "optimized_total_cost": opt_result["total_cost"],
        "jit_baseline_plan": jit_result["plan"].to_dict("records"),
        "jit_baseline_total_cost": jit_result["total_cost"],
        "full_baseline_total_cost": full_result["total_cost"],
        "cheapest_only_feasible": cheap_result["feasible"],
        "savings_usd": round(savings_vs_jit, 2),
        "savings_pct": round(savings_pct, 2) if savings_pct is not None else None,
        "monte_carlo": mc_summary.to_dict("records"),
        "stress_durban_spike_plan": spike_result["plan"].to_dict("records"),
        "stress_capetown_closure_plan": closure_result["plan"].to_dict("records") if closure_result["feasible"] else None,
        "fleet_wide_impact": fleet_impact,
    }
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[6] results.json written to {OUT}")
    print("\nDONE.")

    return {"forecast_prices": forecast_prices, "route_ports": list(route_df["port"]), "vessel": VESSEL}


def _make_charts(hist, forecast_df, opt_result, jit_result, full_result, savings_pct, opt_costs, jit_costs):
    # Chart 1: price history + forecast
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(route_df)))
    for idx, (_, r) in enumerate(route_df.iterrows()):
        port = r["port"]
        h = hist[hist["port"] == port]
        ax.plot(h["date"], h["price_usd_per_mt"], color=colors[idx], alpha=0.55, linewidth=1, label=f"{port} (history)")
        fc_row = forecast_df[forecast_df["port"] == port].iloc[0]
        ax.scatter([fc_row["arrival_date"]], [fc_row["forecast_p50"]], color=colors[idx], marker="D", s=70, zorder=5, edgecolor="black")
        ax.plot([fc_row["arrival_date"]] * 2, [fc_row["forecast_p10"], fc_row["forecast_p90"]], color=colors[idx], linewidth=2.5, alpha=0.9)
    ax.set_title("VLSFO Price History & ML Forecast at Vessel Arrival Date (diamond = forecast, bar = 80% interval)", fontsize=11)
    ax.set_ylabel("USD / MT")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/chart1_price_forecast.png")
    plt.close(fig)

    # Chart 2: bunkering plan
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ports = opt_result["plan"]["port"]
    ax1.bar(ports, opt_result["plan"]["bunker_mt"], color="#c0392b")
    ax1.set_ylabel("Bunkered (MT)")
    ax1.set_title("Optimised Bunkering Plan: How Much to Buy at Each Port", fontsize=11)
    for i, v in enumerate(opt_result["plan"]["bunker_mt"]):
        if v > 5:
            ax1.text(i, v + 40, f"{v:,.0f} MT", ha="center", fontsize=8)
    ax2.plot(ports, opt_result["plan"]["fuel_after_mt"], marker="o", color="#2c3e50", linewidth=2, label="Fuel after port call")
    ax2.axhline(VESSEL["safety_buffer_mt"], color="red", linestyle="--", linewidth=1, label="Safety buffer")
    ax2.axhline(VESSEL["tank_capacity_mt"], color="gray", linestyle=":", linewidth=1, label="Tank capacity")
    ax2.set_ylabel("Fuel on board (MT)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/chart2_bunkering_plan.png")
    plt.close(fig)

    # Chart 3: cost comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = ["Optimised\n(DP algorithm)", "Just-in-time\n(price-blind)", "Fill every port\n(naive)"]
    costs = [opt_result["total_cost"], jit_result["total_cost"], full_result["total_cost"]]
    bars = ax.bar(strategies, costs, color=["#27ae60", "#e67e22", "#c0392b"])
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, c + 20000, f"${c:,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Total voyage fuel cost (USD)")
    savings_pct_label = f"{savings_pct:.1f}%" if savings_pct is not None else "an undefined %"
    ax.set_title(f"Strategy Comparison -- Optimiser saves {savings_pct_label} vs price-blind baseline", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/chart3_cost_comparison.png")
    plt.close(fig)

    # Chart 4: Monte Carlo distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(jit_costs, bins=30, alpha=0.55, label="Just-in-time baseline", color="#e67e22")
    ax.hist(opt_costs, bins=30, alpha=0.65, label="Optimised (DP)", color="#27ae60")
    ax.axvline(np.mean(jit_costs), color="#e67e22", linestyle="--")
    ax.axvline(np.mean(opt_costs), color="#27ae60", linestyle="--")
    ax.set_xlabel("Total voyage fuel cost (USD)")
    ax.set_ylabel("Frequency (out of 500 simulated price scenarios)")
    ax.set_title("Monte Carlo Stress Test: Cost Distribution Under Price Uncertainty", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/chart4_monte_carlo.png")
    plt.close(fig)


# ===========================================================================
# SECTION 6 -- CUSTOM ROUTE BUILDER (pick ANY start port, end port, and
# waypoints -- not just the fixed SWS demo rotation)
# ===========================================================================
# This is what lets PIL configure "our voyage goes from X to Y calling at
# Z" and get a full bunkering plan back, instead of only the one
# hardcoded rotation used for the demo above.

PORTS_DB = {
    # Far East
    "Shanghai":     {"lat": 31.23, "lon": 121.47, "base_price": 620, "volatility": 0.018, "supply_risk": 0.10, "lng_available": False, "congestion_days": 0},
    "Ningbo":       {"lat": 29.87, "lon": 121.54, "base_price": 615, "volatility": 0.017, "supply_risk": 0.10, "lng_available": False, "congestion_days": 0},
    "Busan":        {"lat": 35.10, "lon": 129.04, "base_price": 610, "volatility": 0.016, "supply_risk": 0.08, "lng_available": True, "congestion_days": 0},
    "Singapore":    {"lat": 1.29,  "lon": 103.85, "base_price": 598, "volatility": 0.015, "supply_risk": 0.05, "lng_available": True, "congestion_days": 0},
    # South / Middle East
    "Colombo":      {"lat": 6.93,  "lon": 79.85,  "base_price": 614, "volatility": 0.020, "supply_risk": 0.12, "lng_available": False, "congestion_days": 0},
    "Jebel Ali":    {"lat": 25.01, "lon": 55.06,  "base_price": 605, "volatility": 0.017, "supply_risk": 0.10, "lng_available": True, "congestion_days": 0},
    # Africa
    "Durban":       {"lat": -29.87, "lon": 31.02, "base_price": 651, "volatility": 0.028, "supply_risk": 0.22, "lng_available": False, "congestion_days": 3},
    "Cape Town":    {"lat": -33.92, "lon": 18.42, "base_price": 644, "volatility": 0.026, "supply_risk": 0.20, "lng_available": False, "congestion_days": 2},
    "Tema":         {"lat": 5.67,  "lon": -0.02,  "base_price": 683, "volatility": 0.035, "supply_risk": 0.30, "lng_available": False, "congestion_days": 1},
    "Lagos":        {"lat": 6.45,  "lon": 3.39,   "base_price": 690, "volatility": 0.038, "supply_risk": 0.32, "lng_available": False, "congestion_days": 4},
    # Latin America
    "Santos":       {"lat": -23.96, "lon": -46.33, "base_price": 660, "volatility": 0.030, "supply_risk": 0.24, "lng_available": False, "congestion_days": 0},
    "Buenos Aires": {"lat": -34.60, "lon": -58.38, "base_price": 665, "volatility": 0.032, "supply_risk": 0.25, "lng_available": False, "congestion_days": 0},
    "Callao":       {"lat": -12.05, "lon": -77.14, "base_price": 670, "volatility": 0.030, "supply_risk": 0.24, "lng_available": False, "congestion_days": 0},
    "Manzanillo":   {"lat": 19.05, "lon": -104.32, "base_price": 655, "volatility": 0.028, "supply_risk": 0.22, "lng_available": False, "congestion_days": 0},
    # Europe (reference hub)
    "Rotterdam":    {"lat": 51.92, "lon": 4.48,   "base_price": 600, "volatility": 0.014, "supply_risk": 0.06, "lng_available": True, "congestion_days": 0},
}
# NOTE: prices above are illustrative regional reference prices, not the
# trained ML forecast used for the SWS demo -- in production, the
# forecasting module in Section 2 would be retrained on each port's own
# price history before being trusted for a custom route.
#
# lng_available flags the ports we're treating as established LNG bunkering
# hubs (Singapore, Rotterdam, Jebel Ali, Busan) based on general knowledge of
# where LNG bunkering infrastructure currently exists -- this is an
# illustrative simplification, not a verified live feed of PIL's actual
# approved LNG suppliers. The point it demonstrates is real regardless: LNG
# availability is a genuine port-by-port constraint, sharply narrower than
# VLSFO, and a dual-fuel vessel's bunkering decision has to respect that.
#
# congestion_days are ILLUSTRATIVE starting points reflecting general,
# publicly-reported berth-congestion patterns at these African ports
# (Durban and Lagos in particular have had well-documented multi-day
# waiting periods) -- NOT a live feed of current port authority data.
# Matches the HTML dashboard's defaults exactly; override per-call via
# build_custom_route()'s congestion_overrides parameter.

# Illustrative default rates for the congestion/speed physics below --
# matches the HTML dashboard's defaults exactly so numbers agree between
# the two whenever both use their own defaults.
IDLE_CONSUMPTION_MT_PER_DAY = 5.0   # auxiliary/hotel generator load while waiting at anchor for a berth
PORT_DUES_USD = 8000                # per port call, every port on the route
DELIVERY_FEE_USD = 5000             # per port, only where fuel is actually taken
CONGESTION_COST_PER_DAY_USD = 15000  # demurrage-style $/day of congestion, optional overlay

# Real named vessels per class -- ties the demo to PIL's actual fleet rather
# than a generic class label. LNG Dual-Fuel entries are illustrative
# (representative of PIL's newbuild dual-fuel programme referenced in the
# briefing) since no specific hull names for that programme were provided.
FLEET_VESSELS = {
    "O-Class": ["Kota Oasis", "Kota Ocean", "Kota Odyssey", "Kota Orkid"],
    "E/EL-Class": ["Kota Eagle", "Kota Embun", "Kota Emerald", "Kota Elok", "Kota Ebony", "Kota Elan"],
    "LNG Dual-Fuel (Newbuild)": ["LNG Newbuild 1", "LNG Newbuild 2", "LNG Newbuild 3"],
}

VESSEL_CLASSES = {
    "O-Class": {
        "service_speed_knots": 18.5, "daily_consumption_mt": 180.0,
        "tank_capacity_mt": 4000.0, "safety_buffer_mt": 500.0,
        "starting_fuel_mt": 1800.0, "dwt_mt": 152000, "fuel_grade": "VLSFO",
    },
    "E/EL-Class": {
        "service_speed_knots": 19.0, "daily_consumption_mt": 165.0,
        "tank_capacity_mt": 3600.0, "safety_buffer_mt": 450.0,
        "starting_fuel_mt": 1600.0, "dwt_mt": 118000, "fuel_grade": "VLSFO",
    },
    "LNG Dual-Fuel (Newbuild)": {
        "service_speed_knots": 18.5, "daily_consumption_mt": 165.0,
        "tank_capacity_mt": 3200.0, "safety_buffer_mt": 450.0,
        "starting_fuel_mt": 1500.0, "dwt_mt": 140000, "fuel_grade": "LNG",
    },
}
# LNG Dual-Fuel specs are illustrative -- representative of PIL's newbuild
# dual-fuel programme referenced in the briefing, not confirmed technical
# specifications. Lower tank capacity per MT reflects LNG's lower energy
# density versus VLSFO (LNG tanks are typically larger by volume but we're
# working in mass/MT terms here for consistency with the rest of the model).
# DWT (deadweight tonnage) figures are illustrative assumptions, used only
# to compute the ton-mile transport-work metric -- substitute PIL's actual
# per-class DWT for a production version.


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles. This is a straight-line
    approximation -- real voyages follow shipping lanes and round
    landmasses, so actual sailed distance will usually be longer. Fine for
    an illustrative route plot; swap in AIS/routing-API distances for a
    production tool."""
    R_NM = 3440.065
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return R_NM * 2 * np.arcsin(np.sqrt(a))


def build_custom_route(port_names, vessel, actual_speed_knots=None, idle_consumption_mt_per_day=None):
    """
    Build a route_df for ANY sequence of ports PIL specifies -- the first
    entry is the start port, the last is the end port, everything between
    is a waypoint call, in order.

    port_names : list[str], e.g. ["Shanghai", "Singapore", "Santos"]
    vessel     : one of VESSEL_CLASSES[...]
    actual_speed_knots : the OPERATING speed for this voyage, if different
        from the vessel's design/service speed (slow steaming). Defaults to
        the design speed (no override). Fuel consumption scales with the
        CUBE of speed (standard bunker-planning approximation) while
        transit time scales inversely with speed, so total fuel for a
        FIXED distance scales with speed SQUARED -- this is the actual
        mechanism behind "slow steaming" cutting bunker costs, not just a
        longer voyage. Matches the HTML dashboard's speed slider exactly.
    idle_consumption_mt_per_day : auxiliary/hotel-generator fuel burned per
        day of port congestion (waiting for a berth) -- separate from
        main-engine sailing consumption, and unaffected by vessel speed
        (waiting doesn't care how fast you sailed to get there). Defaults
        to IDLE_CONSUMPTION_MT_PER_DAY.
    """
    missing = [p for p in port_names if p not in PORTS_DB]
    if missing:
        raise ValueError(f"Unknown port(s): {missing}. Available: {sorted(PORTS_DB)}")
    if len(port_names) < 2:
        raise ValueError("A route needs at least a start port and an end port.")

    design_speed = vessel["service_speed_knots"]
    speed = actual_speed_knots if actual_speed_knots else design_speed
    speed_ratio = speed / design_speed
    idle_rate = idle_consumption_mt_per_day if idle_consumption_mt_per_day is not None else IDLE_CONSUMPTION_MT_PER_DAY

    rows = []
    for i, name in enumerate(port_names):
        info = PORTS_DB[name]
        if i == 0:
            leg_nm = 0.0
        else:
            prev = PORTS_DB[port_names[i - 1]]
            leg_nm = haversine_nm(prev["lat"], prev["lon"], info["lat"], info["lon"])
        leg_days = leg_nm / (speed * 24)  # transit time at the ACTUAL (possibly slow-steamed) speed
        effective_daily_rate = vessel["daily_consumption_mt"] * (speed_ratio ** 3)  # cubic law
        sail_consumption = leg_days * effective_daily_rate
        congestion_days = info.get("congestion_days", 0)
        idle_consumption = congestion_days * idle_rate
        rows.append({
            "port": name, "leg_distance_nm": leg_nm, "leg_days": leg_days,
            "congestion_days": congestion_days, "idle_consumption_mt": idle_consumption,
            "sail_consumption_mt": sail_consumption, "leg_consumption_mt": sail_consumption + idle_consumption,
            "base_price": info["base_price"], "volatility": info["volatility"], "supply_risk": info["supply_risk"],
        })

    df = pd.DataFrame(rows)
    df["cum_distance_nm"] = df["leg_distance_nm"].cumsum()
    # cum_days includes BOTH transit time and congestion waiting time at
    # each port (waiting happens after arrival, before departing onward) --
    # matches the HTML dashboard's roadmap date calculation exactly.
    df["cum_days"] = (df["leg_days"] + df["congestion_days"]).cumsum()
    return df


def compute_operational_costs(route_df, opt_result, port_dues_usd=PORT_DUES_USD,
                               delivery_fee_usd=DELIVERY_FEE_USD,
                               congestion_cost_per_day=CONGESTION_COST_PER_DAY_USD):
    """
    Port dues/agency fees per call, bunker delivery fee per port where fuel
    is actually taken, and congestion/demurrage cost -- all layered ON TOP
    of the optimiser's fuel-cost result, NOT fed back into it (unlike idle
    fuel consumption above, which IS baked into leg_consumption_mt and
    genuinely affects the optimiser's decision). A fully joint optimisation
    would need these fees to influence which ports get chosen too; treat
    this as a rough overlay, not a re-optimised plan. Mirrors the HTML
    dashboard's Operational Cost Add-ons panel exactly, including that
    distinction.
    """
    if not opt_result["feasible"]:
        return {"enabled": False, "num_calls": 0, "num_bunker_calls": 0, "dues_total": 0, "delivery_total": 0,
                "congestion_days_total": 0, "congestion_total": 0, "addon_total": 0}
    num_calls = len(route_df)
    num_bunker_calls = int((opt_result["plan"]["bunker_mt"] > 0.5).sum())
    dues_total = num_calls * port_dues_usd
    delivery_total = num_bunker_calls * delivery_fee_usd
    congestion_days_total = float(route_df["congestion_days"].sum())
    congestion_total = congestion_days_total * congestion_cost_per_day
    return {
        "enabled": True, "num_calls": num_calls, "num_bunker_calls": num_bunker_calls,
        "dues_total": dues_total, "delivery_total": delivery_total,
        "congestion_days_total": congestion_days_total, "congestion_total": congestion_total,
        "addon_total": dues_total + delivery_total + congestion_total,
    }


def apply_fuel_grade_constraint(route_port_names, base_prices, fuel_grade):
    """
    LNG availability is a real, sharply-narrower constraint than VLSFO --
    most of PIL's network can bunker VLSFO, only a handful of established
    hubs currently support LNG. This is the model's answer to that: for a
    dual-fuel vessel running on LNG, any port that can't supply it is priced
    at a sentinel so high the optimiser will never choose to bunker there
    unless doing so is the only way to remain feasible -- reusing the exact
    same mechanism the port-closure stress test already uses (see
    optimize_bunkering's zero-bunker fix). VLSFO vessels are unaffected.
    """
    if fuel_grade != "LNG":
        return dict(base_prices)
    return {
        port: (price if PORTS_DB[port]["lng_available"] else 1e9)
        for port, price in base_prices.items()
    }


def run_custom_route(port_names, vessel_class="O-Class", vessel_name=None,
                      actual_speed_knots=None, idle_consumption_mt_per_day=None,
                      include_operational_costs=False, port_dues_usd=PORT_DUES_USD,
                      delivery_fee_usd=DELIVERY_FEE_USD, congestion_cost_per_day=CONGESTION_COST_PER_DAY_USD,
                      voyage_start_date=None, verbose=True):
    """
    End-to-end: build the route (with optional slow-steaming speed and
    congestion/idle-fuel physics), price it with illustrative reference
    prices (fuel-grade-constrained if the vessel is LNG dual-fuel),
    optimise bunkering, and report distance / ton-mile / bunker totals,
    estimated arrival dates, and (optionally) operational cost add-ons --
    the same numbers and the same methodology as the HTML dashboard.

    actual_speed_knots : operate at other than the vessel's design speed
        (slow steaming or above-design running). None = design speed.
    idle_consumption_mt_per_day : auxiliary fuel burn per day of port
        congestion. None = IDLE_CONSUMPTION_MT_PER_DAY default.
    include_operational_costs : if True, add port dues / delivery fees /
        congestion demurrage on top of the fuel-cost total (matches the
        HTML dashboard's "Include in totals" toggle).
    voyage_start_date : "YYYY-MM-DD" string for estimated arrival dates.
        None = today.
    """
    vessel = VESSEL_CLASSES[vessel_class]
    fuel_grade = vessel.get("fuel_grade", "VLSFO")
    vessel_name = vessel_name or FLEET_VESSELS[vessel_class][0]
    design_speed = vessel["service_speed_knots"]
    speed = actual_speed_knots if actual_speed_knots else design_speed

    route = build_custom_route(port_names, vessel, actual_speed_knots=speed,
                                idle_consumption_mt_per_day=idle_consumption_mt_per_day)
    base_prices = dict(zip(route["port"], route["base_price"]))
    prices = apply_fuel_grade_constraint(port_names, base_prices, fuel_grade)

    opt = optimize_bunkering(route, prices, vessel)
    jit = baseline_just_in_time(route, prices, vessel)
    opcosts = compute_operational_costs(route, opt, port_dues_usd, delivery_fee_usd, congestion_cost_per_day) \
        if include_operational_costs else compute_operational_costs(route, {"feasible": False})

    total_nm = route["cum_distance_nm"].iloc[-1]
    total_consumed_mt = route["leg_consumption_mt"].sum()
    total_idle_mt = route["idle_consumption_mt"].sum()
    total_congestion_days = route["congestion_days"].sum()
    ton_miles_million = total_nm * vessel["dwt_mt"] / 1e6  # million DWT-nautical-miles

    start_date = pd.Timestamp(voyage_start_date) if voyage_start_date else pd.Timestamp.now().normalize()
    arrival_dates = [None] + [
        (start_date + pd.Timedelta(days=float(d))).strftime("%Y-%m-%d")
        for d in route["cum_days"].iloc[1:]
    ]

    summary = {
        "route": port_names,
        "vessel_class": vessel_class,
        "vessel_name": vessel_name,
        "fuel_grade": fuel_grade,
        "design_speed_knots": design_speed,
        "actual_speed_knots": speed,
        "total_distance_nm": round(total_nm, 1),
        "total_bunker_consumed_mt": round(total_consumed_mt, 1),
        "total_idle_consumption_mt": round(total_idle_mt, 1),
        "total_congestion_days": round(float(total_congestion_days), 1),
        "total_ton_miles_million": round(ton_miles_million, 2),
        "estimated_arrival_dates": dict(zip(port_names, arrival_dates)),
        "optimized_plan": opt["plan"].to_dict("records") if opt["feasible"] else None,
        "optimized_total_cost": opt["total_cost"],
        "jit_baseline_total_cost": jit["total_cost"] if jit["feasible"] else None,
        "operational_costs": opcosts,
        "all_in_total_cost": (opt["total_cost"] + opcosts["addon_total"]) if opt["feasible"] and opcosts["enabled"] else None,
    }

    if verbose:
        print("\n" + "=" * 70)
        print(f"CUSTOM ROUTE: {' -> '.join(port_names)}  ({vessel_name}, {vessel_class}, {fuel_grade})")
        print("=" * 70)
        speed_note = f"  [slow steaming, design {design_speed} kn]" if speed < design_speed \
            else f"  [above design speed {design_speed} kn]" if speed > design_speed else ""
        print(f"Operating speed: {speed:.1f} kn{speed_note}")
        print(route[["port", "leg_distance_nm", "congestion_days", "sail_consumption_mt", "idle_consumption_mt", "leg_consumption_mt"]].to_string(index=False))
        print(f"\nTotal distance:          {total_nm:,.0f} nm")
        print(f"Total bunker consumed:   {total_consumed_mt:,.0f} MT  (of which {total_idle_mt:,.1f} MT idle/hotel load while congested)")
        print(f"Total ton-miles:         {ton_miles_million:,.2f} million DWT-nm")
        if total_congestion_days > 0:
            congested = [p for p in port_names if PORTS_DB[p].get("congestion_days", 0) > 0]
            congestion_parts = [f"{p} (+{PORTS_DB[p]['congestion_days']}d)" for p in congested]
            print(f"Congestion:              {total_congestion_days:.0f} day(s) total \u2014 {', '.join(congestion_parts)}")
        if fuel_grade == "LNG":
            unavailable = [p for p in port_names if not PORTS_DB[p]["lng_available"]]
            if unavailable:
                print(f"LNG NOT available at:   {', '.join(unavailable)} -- optimiser routed around these")
        if opt["feasible"]:
            print(f"\nOptimised bunkering plan (with estimated arrival dates):")
            plan_display = opt["plan"].copy()
            plan_display.insert(1, "est_arrival", [arrival_dates[i] or "-" for i in range(len(plan_display))])
            print(plan_display.to_string(index=False))
            print(f"\nOptimised total cost:    ${opt['total_cost']:,.0f}")
            if jit["feasible"]:
                sav = jit["total_cost"] - opt["total_cost"]
                sav_pct_str = f"{100*sav/jit['total_cost']:.1f}%" if jit["total_cost"] > 0 else "N/A (baseline cost is $0)"
                print(f"JIT baseline cost:       ${jit['total_cost']:,.0f}")
                print(f"Savings:                 ${sav:,.0f}  ({sav_pct_str})")
            if opcosts["enabled"]:
                print(f"\nOperational cost add-ons (layered on top, not fed back into the optimiser):")
                print(f"  Port dues ({opcosts['num_calls']} calls):        ${opcosts['dues_total']:,.0f}")
                print(f"  Delivery fees ({opcosts['num_bunker_calls']} bunkering ports): ${opcosts['delivery_total']:,.0f}")
                print(f"  Congestion ({opcosts['congestion_days_total']:.0f} days):        ${opcosts['congestion_total']:,.0f}")
                print(f"  Total add-on:                 ${opcosts['addon_total']:,.0f}")
                print(f"  ALL-IN voyage cost:            ${opt['total_cost'] + opcosts['addon_total']:,.0f}")
        else:
            print("\nNo feasible plan for this route with the given tank capacity / safety buffer"
                  + (" / LNG availability" if fuel_grade == "LNG" else "") + ".")

    return summary


def fleet_wide_impact(per_voyage_savings_usd, vessels_applying=100, voyages_per_vessel_per_year=6):
    """
    The case brief primes judges with PIL's fleet scale (~100 vessels) and
    bunker cost share (40-60% of voyage opex) -- this is the multiplication
    that turns a single-voyage saving into the number an executive actually
    weighs a decision on. Both assumptions are adjustable and should be
    stated explicitly whenever this number is quoted:
      - vessels_applying: how many of PIL's ~100 vessels run a comparable
        long-haul, multi-port rotation where this approach would apply.
        Defaulted to the full fleet as an upper-bound estimate; a more
        conservative planning figure would use PIL's actual route mix.
      - voyages_per_vessel_per_year: assumed round trips per vessel per
        year on a rotation of this length (~10,000-12,000 nm). 6 is a
        rough illustrative figure, not sourced from PIL's actual schedule.
    """
    annual_savings_usd = per_voyage_savings_usd * vessels_applying * voyages_per_vessel_per_year
    return {
        "per_voyage_savings_usd": round(per_voyage_savings_usd, 2),
        "vessels_applying": vessels_applying,
        "voyages_per_vessel_per_year": voyages_per_vessel_per_year,
        "annual_savings_usd": round(annual_savings_usd, 2),
    }




def launch_dashboard(forecast_prices, route_port_names, vessel, open_browser=True):
    """
    Bakes the ML-forecasted SWS-route prices (and the optimised route) into
    a fresh copy of the HTML dashboard, then pops it open in the default
    browser -- so running this script goes straight from "ran the model"
    to "here's the interactive tool showing what it found."

    Only the SWS route's ports get genuinely ML-forecasted prices here --
    that's what the forecasting module in Section 2 actually trained on.
    Every other port in the dashboard's broader 15-port network still shows
    the same static illustrative reference prices as when you open the
    dashboard directly; this function doesn't fabricate forecasts for ports
    the model was never trained on. The dashboard's "LIVE DATA" badge vs.
    "STATIC REFERENCE PRICES" badge tells you which you're looking at.

    Requires pil_route_builder_dashboard.html to sit next to this script
    (or already be in the outputs folder) -- download both files into the
    same folder for this to work.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_candidates = [
        os.path.join(script_dir, "pil_route_builder_dashboard.html"),
        os.path.join(OUT, "pil_route_builder_dashboard.html"),
    ]
    template_path = next((p for p in template_candidates if os.path.exists(p)), None)
    if template_path is None:
        print("\n[dashboard] Couldn't find pil_route_builder_dashboard.html next to this "
              "script or in the outputs folder -- skipping auto-launch. Put both files in "
              "the same folder to enable this.")
        return None

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    live_data = {
        "generatedAt": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "ports": {name: round(float(price), 2) for name, price in forecast_prices.items()},
        "vessels": {"O-Class": {"dailyConsumptionMt": vessel["daily_consumption_mt"], "tankCapacityMt": vessel["tank_capacity_mt"]}},
        "vesselClass": "O-Class",
        "route": route_port_names,
        "startingFuel": vessel["starting_fuel_mt"],
        "buffer": vessel["safety_buffer_mt"],
        # exact per-port leg distance/consumption from THIS run's route_df --
        # lets the dashboard match the console output exactly for this route,
        # rather than silently recomputing its own (different) great-circle
        # distances. Falls back to normal recomputation the moment the user
        # edits the route to anything Python didn't compute.
        "legDistanceNm": dict(zip(route_df["port"], route_df["leg_distance_nm"].round(6))),
        "legConsumptionMt": dict(zip(route_df["port"], route_df["leg_consumption_mt"].round(6))),
    }

    marker = "const PYTHON_LIVE_DATA = null;"
    if marker not in html:
        print("\n[dashboard] Template has changed and no longer contains the expected "
              "injection point -- skipping auto-launch.")
        return None

    html = html.replace(marker, f"const PYTHON_LIVE_DATA = {json.dumps(live_data)};", 1)

    live_path = os.path.join(OUT, "pil_dashboard_live.html")
    with open(live_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[dashboard] Live dashboard written to {live_path}")

    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(live_path)}")
        print("[dashboard] Opening in your default browser...")

    return live_path


if __name__ == "__main__":
    run_summary = main()

    # -----------------------------------------------------------------------
    # EXAMPLE: build & optimise a custom route -- edit these three lines to
    # set PIL's own start port, end port, and any waypoints in between.
    # -----------------------------------------------------------------------
    CUSTOM_ROUTE_PORTS = ["Shanghai", "Singapore", "Cape Town", "Santos"]
    CUSTOM_VESSEL_CLASS = "O-Class"  # or "E/EL-Class" / "LNG Dual-Fuel (Newbuild)"

    custom_summary = run_custom_route(CUSTOM_ROUTE_PORTS, CUSTOM_VESSEL_CLASS)
    with open(f"{OUT}/custom_route_results.json", "w") as f:
        json.dump(custom_summary, f, indent=2, default=str)
    print(f"\ncustom_route_results.json written to {OUT}")

    # -----------------------------------------------------------------------
    # EXAMPLE: an LNG dual-fuel newbuild on a route mixing LNG-capable and
    # LNG-incapable ports. The optimiser routes around Colombo (no LNG
    # supply there) by bunkering more heavily at Singapore beforehand --
    # this is the fuel-grade constraint actually changing the plan, not
    # just the price.
    # -----------------------------------------------------------------------
    LNG_FEASIBLE_ROUTE = ["Busan", "Singapore", "Colombo", "Jebel Ali"]
    lng_summary = run_custom_route(LNG_FEASIBLE_ROUTE, "LNG Dual-Fuel (Newbuild)")
    with open(f"{OUT}/lng_dual_fuel_results.json", "w") as f:
        json.dump(lng_summary, f, indent=2, default=str)
    print(f"\nlng_dual_fuel_results.json written to {OUT}")

    # -----------------------------------------------------------------------
    # EXAMPLE: the SAME LNG vessel on a route where LNG availability makes
    # the voyage genuinely impossible as configured (too far between the
    # LNG-capable ports for the tank to bridge) -- the model correctly
    # reports this as infeasible rather than silently bunkering fuel that
    # doesn't exist. Catching this before the voyage is committed to is
    # itself part of the tool's value, not just finding savings.
    # -----------------------------------------------------------------------
    LNG_INFEASIBLE_ROUTE = ["Shanghai", "Singapore", "Cape Town", "Santos"]
    lng_infeasible_summary = run_custom_route(LNG_INFEASIBLE_ROUTE, "LNG Dual-Fuel (Newbuild)")

    # -----------------------------------------------------------------------
    # EXAMPLE: the SWS route as a custom route (touches Durban/Cape Town/
    # Tema, all with illustrative congestion defaults), at design speed vs.
    # slow steaming, with operational cost add-ons included -- same
    # methodology, same numbers, as the HTML dashboard's equivalent
    # scenario. Demonstrates that slow steaming is a real, very large lever
    # (dropping to 14 kn cuts this voyage's fuel cost by well over half),
    # separate from the congestion/idle-fuel physics above.
    # -----------------------------------------------------------------------
    SPEED_DEMO_ROUTE = ["Shanghai", "Singapore", "Colombo", "Durban", "Cape Town", "Tema"]
    print("\n" + "#" * 70)
    print("# SPEED CONFIGURATION DEMO -- design speed vs. slow steaming")
    print("#" * 70)
    design_speed_summary = run_custom_route(SPEED_DEMO_ROUTE, "O-Class", include_operational_costs=True)
    slow_steam_summary = run_custom_route(SPEED_DEMO_ROUTE, "O-Class", actual_speed_knots=14, include_operational_costs=True)
    print(f"\nSlow steaming to 14 kn: ${design_speed_summary['optimized_total_cost']:,.0f} -> "
          f"${slow_steam_summary['optimized_total_cost']:,.0f} "
          f"({100*(1 - slow_steam_summary['optimized_total_cost']/design_speed_summary['optimized_total_cost']):.0f}% less fuel cost, "
          f"at the price of a longer voyage)")
    with open(f"{OUT}/speed_configuration_results.json", "w") as f:
        json.dump({"design_speed": design_speed_summary, "slow_steaming_14kn": slow_steam_summary}, f, indent=2, default=str)
    print(f"speed_configuration_results.json written to {OUT}")

    # Pop the interactive dashboard open, pre-loaded with the SWS route's
    # ML-forecasted prices. Set open_browser=False if you just want the
    # live HTML file written without a browser window popping up.
    launch_dashboard(run_summary["forecast_prices"], run_summary["route_ports"], run_summary["vessel"])


