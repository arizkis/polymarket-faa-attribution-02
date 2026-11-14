# SPDX-License-Identifier: MIT
import json
import math
import os
from bisect import bisect_left
from pathlib import Path


BAND_STEPS = (4, 14, 31, 54, 84, 120, 163, 213)
MEMBER_A_SCALE_MULTIPLIERS = (1.0, 1.1, 1.1, 1.0, 1.1, 1.1, 1.1, 1.1)
MEMBER_B_SHRINK_MULTIPLIERS = (0.5, 2.25, 2.25, 1.0, 2.25, 2.25, 2.25, 2.25)
MEMBER_B_SCALE_MULTIPLIERS = (1.15, 1.15, 1.15, 1.15, 1.075, 1.0, 1.15, 1.15)
MEMBER_C_SCALE_MULTIPLIERS = (1.0, 1.05, 1.05, 1.0, 1.05, 1.05, 1.05, 1.05)
MEMBER_A_SHRINK = 500.0
MEMBER_B_SHRINK = 15.0
MEMBER_A_GLOBAL_SCALE = 0.52
MEMBER_B_GLOBAL_SCALE = 0.46
MEMBER_C_GLOBAL_SCALE = 0.46
MEMBER_D_SCALE = 0.85
MEMBER_D_SIGMA_CAP = 1.5
MEMBER_D_TRAILING_MINUTES = 360
MEMBER_D_Z_BY_QUANTILE = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
BLEND_WEIGHT_A = 0.35
BLEND_WEIGHT_B = 0.2
BLEND_WEIGHT_C = 0.15
BLEND_WEIGHT_D = 0.3
HL_NORMAL = 480.0
HL_CRASH = 240.0
CRASH_Z = 1.5
CRASH_WINDOW_MINUTES = 120
REGIME_WIDTH = 0.5
QUANTILE_FLOOR = 0.001
QUANTILE_CEILING = 0.999
LONGSHOT_MULT = 1.35
LONGSHOT_FINE_GRID_THRESHOLD = 0.02
SUMREV_MIN_MARKETS = 8
SUMREV_TARGET_SUM = 1.0
SUMREV_SUM_TOLERANCE = 0.25
SUMREV_STRENGTH = 0.5
SUMREV_HORIZON_MINUTES = 240
DECAY_LOOKBACK_MINUTES = 120
DECAY_DRIFT_THRESHOLD = -0.35
DECAY_VOLATILITY_MULT = 1.15
DECAY_RATE = 0.025
DECAY_MEDIAN_FLOOR_FRACTION = 0.35
CRASH_DRIFT_RATE = 0.035
CRASH_FLOOR_FRAC = 0.25


def logit(price):
    clipped = min(QUANTILE_CEILING, max(QUANTILE_FLOOR, price))
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def recency_weight(end_index, length, half_life):
    return 0.5 ** ((length - 1 - end_index) / half_life)


def rolling_volatilities(series):
    squared_change_prefix = [0.0]
    for index in range(1, len(series)):
        change = series[index] - series[index - 1]
        squared_change_prefix.append(squared_change_prefix[-1] + change * change)

    volatilities = []
    for position in range(len(series)):
        count = min(60, position)
        if count == 0:
            volatilities.append(1e-6)
            continue
        squared_sum = (
            squared_change_prefix[position]
            - squared_change_prefix[position - count]
        )
        volatilities.append(max(1e-6, math.sqrt(squared_sum / count)))
    return volatilities


def weighted_rms(weighted_changes):
    total_weight = 0.0
    weighted_square_sum = 0.0
    for change, weight in weighted_changes:
        total_weight += weight
        weighted_square_sum += weight * change * change
    if total_weight == 0.0:
        return 0.0
    return math.sqrt(weighted_square_sum / total_weight)


def unweighted_rms(changes):
    if not changes:
        return 0.0
    return math.sqrt(sum(change * change for change in changes) / len(changes))


def weighted_quantiles(samples, quantiles):
    if not samples:
        return [0.0] * len(quantiles)

    ordered = sorted(samples, key=lambda sample: sample[0])
    total_weight = sum(weight for _, weight in ordered)
    if total_weight == 0.0:
        return [0.0] * len(quantiles)

    targets = sorted(enumerate(quantiles), key=lambda item: item[1])
    results = [None] * len(quantiles)
    cumulative_weight = 0.0
    target_index = 0
    for value, weight in ordered:
        midpoint = (cumulative_weight + weight / 2.0) / total_weight
        cumulative_weight += weight
        while target_index < len(targets) and midpoint >= targets[target_index][1]:
            results[targets[target_index][0]] = value
            target_index += 1

    last_value = ordered[-1][0]
    for index, result in enumerate(results):
        if result is None:
            results[index] = last_value
    return results


def effective_sample_size(samples):
    total_weight = sum(weight for _, weight in samples)
    squared_weight_sum = sum(weight * weight for _, weight in samples)
    if squared_weight_sum == 0.0:
        return 0.0
    return total_weight * total_weight / squared_weight_sum


def sanitize_prices(values):
    if not isinstance(values, list):
        return [], 0
    prices = []
    last_valid = None
    valid_count = 0
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if last_valid is not None:
                prices.append(last_valid)
            continue
        try:
            price = float(value)
        except OverflowError:
            if last_valid is not None:
                prices.append(last_valid)
            continue
        if not math.isfinite(price):
            if last_valid is not None:
                prices.append(last_valid)
            continue
        prices.append(price)
        last_valid = price
        valid_count += 1
    return prices, valid_count


def prepare_market(market):
    prices = market["prices"]
    series = [logit(price) for price in prices]
    volatilities = rolling_volatilities(series)
    changes = [series[index] - series[index - 1] for index in range(1, len(series))]
    step_rms = unweighted_rms(changes)
    crash_window = min(len(changes), CRASH_WINDOW_MINUTES)
    crash_displacement = (
        sum(changes[-crash_window:]) if crash_window else 0.0
    )
    crash_threshold = (
        CRASH_Z * step_rms * math.sqrt(crash_window)
    )
    half_life = (
        HL_CRASH
        if crash_window > 0 and crash_displacement < -crash_threshold
        else HL_NORMAL
    )
    return {
        "id": market.get("id"),
        "prices": prices,
        "series": series,
        "flat": len(prices) <= 1,
        "checkpoint": prices[-1] if prices else 0.5,
        "volatilities": volatilities,
        "vol_now": volatilities[-1] if volatilities else 1e-6,
        "step_rms": step_rms,
        "half_life": half_life,
        "one_minute_changes": [
            (
                series[index] - series[index - 1],
                recency_weight(index, len(series), half_life),
            )
            for index in range(1, len(series))
        ],
    }


def samples_for_step(market, space_key, step, regime_matching=False):
    series = market[space_key]
    length = len(series)
    samples = []
    for end_index in range(step, length):
        start_index = end_index - step
        weight = recency_weight(end_index, length, market["half_life"])
        if regime_matching:
            regime_distance = abs(
                math.log(market["volatilities"][start_index] / market["vol_now"])
            )
            weight *= math.exp(-regime_distance / REGIME_WIDTH)
        samples.append(
            (series[end_index] - series[start_index], weight)
        )
    return samples


def build_member_statistics(
    markets, quantiles, space_key, regime_matching=False, unweighted=False
):
    own_statistics = [{} for _ in markets]
    pooled_statistics = {}
    for band_index, step in enumerate(BAND_STEPS):
        pooled_samples = []
        for market_index, market in enumerate(markets):
            samples = samples_for_step(market, space_key, step, regime_matching)
            own_statistics[market_index][band_index] = (
                weighted_quantiles(samples, quantiles),
                effective_sample_size(samples),
            )
            pooled_samples.extend(samples)
        pooled_statistics[band_index] = weighted_quantiles(pooled_samples, quantiles)

    changes_by_market = [
        [
            (
                market[space_key][index] - market[space_key][index - 1],
                recency_weight(index, len(market[space_key]), market["half_life"]),
            )
            for index in range(1, len(market[space_key]))
        ]
        for market in markets
    ]
    if unweighted:
        market_rms = [
            unweighted_rms([change for change, _ in changes])
            for changes in changes_by_market
        ]
        pooled_rms = unweighted_rms(
            [change for changes in changes_by_market for change, _ in changes]
        )
    else:
        market_rms = [weighted_rms(changes) for changes in changes_by_market]
        pooled_rms = weighted_rms(
            [change for changes in changes_by_market for change in changes]
        )
    return own_statistics, pooled_statistics, market_rms, pooled_rms


def forecast_member(
    market,
    own_statistics,
    pooled_statistics,
    quantiles,
    horizon,
    space_key,
    own_rms,
    pooled_rms,
    shrink_by_band,
    scale_by_band,
    logit_space,
    longshot,
):
    checkpoint = market["checkpoint"]
    checkpoint_value = logit(checkpoint) if logit_space else checkpoint
    rows = []
    for forecast_horizon in range(1, horizon + 1):
        band_index = next(
            (
                index
                for index, step in enumerate(BAND_STEPS)
                if forecast_horizon <= step
            ),
            len(BAND_STEPS) - 1,
        )
        step = BAND_STEPS[band_index]
        own_quantiles, ess = own_statistics[band_index]
        pooled_quantiles = pooled_statistics[band_index]
        if own_rms == 0.0 or pooled_rms == 0.0:
            pooled_scale = 1.0
        else:
            pooled_scale = own_rms / pooled_rms
        shrink = shrink_by_band[band_index]
        scale = scale_by_band[band_index]
        horizon_scale = math.sqrt(forecast_horizon / step)

        row = []
        for index, tau in enumerate(quantiles):
            if tau == 0.5:
                row.append(checkpoint)
                continue
            change = (
                ess * own_quantiles[index]
                + shrink * pooled_quantiles[index] * pooled_scale
            ) / (ess + shrink)
            if longshot and checkpoint < 0.01:
                change *= LONGSHOT_MULT
            if logit_space:
                value = sigmoid(checkpoint_value + change * horizon_scale * scale)
            else:
                value = min(
                    1.0,
                    max(0.0, checkpoint_value + change * horizon_scale * scale),
                )
            row.append(value)
        rows.append(row)
    return rows


def member_d_sigmas(markets):
    own_sigmas = []
    for market in markets:
        changes = [
            market["series"][index] - market["series"][index - 1]
            for index in range(1, len(market["series"]))
        ]
        trailing_changes = changes[-MEMBER_D_TRAILING_MINUTES:]
        own_sigmas.append(unweighted_rms(trailing_changes))
    sigma_pool = percentile(own_sigmas, 0.5) if own_sigmas else None
    if sigma_pool is None or sigma_pool == 0.0:
        return own_sigmas
    sigma_cap = MEMBER_D_SIGMA_CAP * sigma_pool
    return [min(sigma, sigma_cap) for sigma in own_sigmas]


def forecast_member_d(market, quantiles, horizon, sigma):
    checkpoint = market["checkpoint"]
    checkpoint_logit = logit(checkpoint)
    widening = LONGSHOT_MULT if checkpoint < 0.01 else 1.0
    rows = []
    for forecast_horizon in range(1, horizon + 1):
        spread = sigma * math.sqrt(forecast_horizon) * MEMBER_D_SCALE * widening
        row = []
        for index, _ in enumerate(quantiles):
            z_value = MEMBER_D_Z_BY_QUANTILE[index]
            row.append(sigmoid(checkpoint_logit + z_value * spread))
        rows.append(row)
    return rows


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def history_tick(prices):
    distinct = sorted(set(prices))
    differences = [
        right - left
        for left, right in zip(distinct, distinct[1:])
        if right > left
    ]
    return min(differences) if differences else None


def decay_metadata(market):
    logit_prices = market["series"]
    changes = [
        logit_prices[index] - logit_prices[index - 1]
        for index in range(1, len(logit_prices))
    ]
    lookback_start = max(0, len(logit_prices) - 1 - DECAY_LOOKBACK_MINUTES)
    drift = logit_prices[-1] - logit_prices[lookback_start]
    recent_changes = changes[-DECAY_LOOKBACK_MINUTES:]
    vol_recent = unweighted_rms(recent_changes)
    quiet_decay = (
        drift < DECAY_DRIFT_THRESHOLD
        and vol_recent <= DECAY_VOLATILITY_MULT * market["step_rms"]
    )
    return quiet_decay


def crash_metadata(market):
    logit_prices = market["series"]
    changes = [
        logit_prices[index] - logit_prices[index - 1]
        for index in range(1, len(logit_prices))
    ]
    window = min(len(changes), CRASH_WINDOW_MINUTES)
    if window == 0:
        return False
    displacement = sum(changes[-window:])
    threshold = CRASH_Z * market["step_rms"] * math.sqrt(window)
    return displacement < -threshold


def apply_decay_drift(market, rows, quantiles):
    if not decay_metadata(market):
        return

    checkpoint = market["checkpoint"]
    median_index = next(
        (index for index, tau in enumerate(quantiles) if tau == 0.5),
        len(quantiles) // 2,
    )
    median_floor = DECAY_MEDIAN_FLOOR_FRACTION * checkpoint
    for horizon, row in enumerate(rows, start=1):
        median = min(1.0, max(0.0, row[median_index]))
        drift_amount = DECAY_RATE * math.sqrt(horizon)
        proposed_logit = logit(median) - drift_amount
        proposed_median = sigmoid(proposed_logit)
        resulting_median = max(median_floor, proposed_median)
        resulting_median = min(1.0, max(0.0, resulting_median))
        actual_logit_shift = logit(resulting_median) - logit(median)
        for index, value in enumerate(row):
            row[index] = sigmoid(logit(value) + actual_logit_shift)


def apply_crash_drift(market, rows, quantiles):
    if not crash_metadata(market):
        return

    checkpoint = market["checkpoint"]
    median_index = next(
        (index for index, tau in enumerate(quantiles) if tau == 0.5),
        len(quantiles) // 2,
    )
    median_floor = CRASH_FLOOR_FRAC * checkpoint
    for horizon, row in enumerate(rows, start=1):
        median = min(1.0, max(0.0, row[median_index]))
        drift_amount = CRASH_DRIFT_RATE * math.sqrt(horizon)
        proposed_logit = logit(median) - drift_amount
        proposed_median = sigmoid(proposed_logit)
        resulting_median = max(median_floor, proposed_median)
        resulting_median = min(1.0, max(0.0, resulting_median))
        actual_logit_shift = logit(resulting_median) - logit(median)
        for index, value in enumerate(row):
            row[index] = sigmoid(logit(value) + actual_logit_shift)


def apply_sum_reversion(entries, forecast_rows, quantiles):
    market_count = len(entries)
    initial_sum = sum(entry[2] for entry in entries)
    if (
        market_count < SUMREV_MIN_MARKETS
        or abs(initial_sum - SUMREV_TARGET_SUM) > SUMREV_SUM_TOLERANCE
    ):
        return

    median_index = next(
        (index for index, tau in enumerate(quantiles) if tau == 0.5),
        len(quantiles) // 2,
    )
    horizon_count = len(forecast_rows[0]) if forecast_rows else 0
    for horizon_index in range(horizon_count):
        median_sum = sum(
            rows[horizon_index][median_index] for rows in forecast_rows
        )
        correction = (
            (initial_sum - median_sum)
            * ((horizon_index + 1) / SUMREV_HORIZON_MINUTES)
            * SUMREV_STRENGTH
            / market_count
        )
        for rows in forecast_rows:
            row = rows[horizon_index]
            for index in range(len(row)):
                row[index] = min(1.0, max(0.0, row[index] + correction))
            for index in range(1, len(row)):
                row[index] = max(row[index], row[index - 1])


def finish_forecast(market, blended_rows):
    prices = market["prices"]
    checkpoint = market["checkpoint"]
    reversion_move = None
    if checkpoint > 0.10 and len(prices) >= 91:
        move = checkpoint - prices[-91]
        trailing_moves = [
            abs(prices[index] - prices[index - 90])
            for index in range(90, len(prices))
        ]
        if abs(move) >= percentile(trailing_moves, 0.8):
            reversion_move = move

    tick = history_tick(prices)
    unique_prices = sorted(set(prices))
    midpoints = [
        (left + right) / 2.0
        for left, right in zip(unique_prices, unique_prices[1:])
    ] if checkpoint < LONGSHOT_FINE_GRID_THRESHOLD else []
    final_rows = []
    for horizon, row in enumerate(blended_rows, start=1):
        if reversion_move is not None:
            factor = 0.5 if reversion_move > 0.0 else 0.25
            shift = -factor * reversion_move * math.sqrt(horizon / 240.0)
            row = [value + shift for value in row]
        if tick is not None:
            snapped = []
            for value in row:
                grid_value = round(value / tick) * tick
                if midpoints:
                    position = bisect_left(midpoints, value)
                    nearest_midpoint = min(
                        midpoints[max(0, position - 1):position + 1],
                        key=lambda point: abs(point - value),
                    )
                    if abs(nearest_midpoint - value) < abs(grid_value - value):
                        grid_value = nearest_midpoint
                snapped.append(grid_value)
            row = snapped
        row = [min(1.0, max(0.0, value)) for value in row]
        for index in range(1, len(row)):
            row[index] = max(row[index], row[index - 1])
        final_rows.append([round(value, 6) for value in row])
    return {"id": market["id"], "q": final_rows}


def forecast_market(market, own_statistics, pooled_statistics, quantiles, horizon):
    checkpoint = market["checkpoint"]
    checkpoint_logit = logit(checkpoint)
    forecast_rows = []

    for forecast_horizon in range(1, horizon + 1):
        band_index = next(
            index
            for index, step in enumerate(BAND_STEPS)
            if forecast_horizon <= step
        ) if forecast_horizon <= BAND_STEPS[-1] else len(BAND_STEPS) - 1
        step = BAND_STEPS[band_index]
        own_quantiles, ess = own_statistics[band_index]
        pooled_quantiles = pooled_statistics[band_index]
        if market["rms"] == 0.0 or market["pooled_rms"] == 0.0:
            pooled_scale = 1.0
        else:
            pooled_scale = market["rms"] / market["pooled_rms"]

        h_step_scale = math.sqrt(forecast_horizon / step)
        price_quantiles = []
        for index, tau in enumerate(quantiles):
            if tau == 0.5:
                price_quantiles.append(checkpoint)
                continue
            blended_change = (
                ess * own_quantiles[index]
                + MEMBER_A_SHRINK * pooled_quantiles[index] * pooled_scale
            ) / (ess + MEMBER_A_SHRINK)
            if checkpoint < 0.01:
                blended_change *= LONGSHOT_MULT
            price = sigmoid(
                checkpoint_logit
                + blended_change
                * h_step_scale
                * MEMBER_A_GLOBAL_SCALE
                * MEMBER_A_SCALE_MULTIPLIERS[band_index]
            )
            price_quantiles.append(min(1.0, max(0.0, price)))

        for index in range(1, len(price_quantiles)):
            price_quantiles[index] = max(
                price_quantiles[index], price_quantiles[index - 1]
            )
        forecast_rows.append([round(price, 6) for price in price_quantiles])

    return {"id": market["id"], "q": forecast_rows}


def flat_forecast(market_id, checkpoint, horizon):
    value = round(min(1.0, max(0.0, checkpoint)), 6)
    return {"id": market_id, "q": [[value] * 5 for _ in range(horizon)]}


def forecast_case(case_data):
    quantiles = case_data["quantiles"]
    horizon = case_data["horizon_minutes"]
    entries = []
    markets = []
    for source_market in case_data["markets"]:
        if not isinstance(source_market, dict):
            source_market = {}
        prices, valid_count = sanitize_prices(source_market.get("prices"))
        market_id = source_market.get("id")
        checkpoint = prices[-1] if prices else 0.5
        try:
            market = prepare_market({"id": market_id, "prices": prices})
            market["flat"] = valid_count <= 1
            entries.append((market, market_id, checkpoint))
            if not market["flat"]:
                markets.append(market)
        except Exception:
            entries.append((None, market_id, checkpoint))

    member_a_own, member_a_pooled, member_a_rms, member_a_pooled_rms = (
        build_member_statistics(markets, quantiles, "series", regime_matching=True)
    )
    member_b_own, member_b_pooled, member_b_rms, member_b_pooled_rms = (
        build_member_statistics(markets, quantiles, "series")
    )
    member_c_own, member_c_pooled, member_c_rms, member_c_pooled_rms = (
        build_member_statistics(markets, quantiles, "prices", unweighted=True)
    )
    prepared_markets = [entry[0] for entry in entries if entry[0] is not None]
    member_d_sigma_by_id = {
        id(market): sigma
        for market, sigma in zip(prepared_markets, member_d_sigmas(prepared_markets))
    }

    for market, rms in zip(markets, member_a_rms):
        market["rms"] = rms
        market["pooled_rms"] = member_a_pooled_rms

    markets_by_id = {id(market): index for index, market in enumerate(markets)}
    forecast_rows_by_market = []
    for market, market_id, checkpoint in entries:
        if market is None or market["flat"]:
            forecast_rows_by_market.append(
                (market, market_id, checkpoint, flat_forecast(market_id, checkpoint, horizon)["q"])
            )
            continue
        try:
            market_index = markets_by_id[id(market)]
            member_a_rows = forecast_market(
                market,
                member_a_own[market_index],
                member_a_pooled,
                quantiles,
                horizon,
            )["q"]
            member_b_rows = forecast_member(
                market,
                member_b_own[market_index],
                member_b_pooled,
                quantiles,
                horizon,
                "series",
                member_b_rms[market_index],
                member_b_pooled_rms,
                tuple(MEMBER_B_SHRINK * multiplier for multiplier in MEMBER_B_SHRINK_MULTIPLIERS),
                tuple(MEMBER_B_GLOBAL_SCALE * multiplier for multiplier in MEMBER_B_SCALE_MULTIPLIERS),
                True,
                True,
            )
            member_c_rows = forecast_member(
                market,
                member_c_own[market_index],
                member_c_pooled,
                quantiles,
                horizon,
                "prices",
                member_c_rms[market_index],
                member_c_pooled_rms,
                (7.5,) * len(BAND_STEPS),
                tuple(MEMBER_C_GLOBAL_SCALE * multiplier for multiplier in MEMBER_C_SCALE_MULTIPLIERS),
                False,
                False,
            )
            member_d_rows = forecast_member_d(
                market,
                quantiles,
                horizon,
                member_d_sigma_by_id[id(market)],
            )
            blended_rows = [
                [
                    BLEND_WEIGHT_A * value_a
                    + BLEND_WEIGHT_B * value_b
                    + BLEND_WEIGHT_C * value_c
                    + BLEND_WEIGHT_D * value_d
                    for value_a, value_b, value_c, value_d in zip(
                        row_a, row_b, row_c, row_d
                    )
                ]
                for row_a, row_b, row_c, row_d in zip(
                    member_a_rows, member_b_rows, member_c_rows, member_d_rows
                )
            ]
            if any(not math.isfinite(value) for row in blended_rows for value in row):
                raise ValueError("non-finite forecast")
        except Exception:
            market = None
            blended_rows = flat_forecast(market_id, checkpoint, horizon)["q"]
        forecast_rows_by_market.append((market, market_id, checkpoint, blended_rows))

    apply_sum_reversion(entries, [item[3] for item in forecast_rows_by_market], quantiles)
    forecasts = []
    for market, market_id, checkpoint, rows in forecast_rows_by_market:
        if market is None or market["flat"]:
            value_rows = [
                [round(min(1.0, max(0.0, value)), 6) for value in row]
                for row in rows
            ]
            forecasts.append({"id": market_id, "q": value_rows})
            continue
        try:
            if crash_metadata(market):
                apply_crash_drift(market, rows, quantiles)
            else:
                apply_decay_drift(market, rows, quantiles)
            forecast = finish_forecast(market, rows)
            if any(
                not math.isfinite(value)
                for row in forecast["q"]
                for value in row
            ):
                raise ValueError("non-finite forecast")
            forecasts.append(forecast)
        except Exception:
            forecasts.append(flat_forecast(market_id, checkpoint, horizon))
    return {"forecasts": forecasts}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in sorted(input_dir.glob("*.in"), key=lambda path: path.name):
        case_data = json.loads(input_path.read_text(encoding="utf-8"))
        result = forecast_case(case_data)
        output_path = output_dir / f"{case_data['case']}.out"
        output_path.write_text(
            json.dumps(result, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
