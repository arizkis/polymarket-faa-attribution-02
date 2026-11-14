# SPDX-License-Identifier: MIT
import json
import math
import os
from pathlib import Path


BAND_STEPS = (4, 14, 31, 54, 84, 120, 163, 213)
MEMBER_B_SHRINK_MULTIPLIERS = (0.5, 2.25, 2.25, 1.0, 2.25, 2.25, 2.25, 2.25)
MEMBER_B_SCALE_MULTIPLIERS = (1.15, 1.15, 1.15, 1.15, 1.075, 1.0, 1.15, 1.15)
SHRINK = 50.0
MEMBER_SCALE = 0.575
RECENCY_HALF_LIFE = 720.0
REGIME_WIDTH = 0.5
QUANTILE_FLOOR = 0.001
QUANTILE_CEILING = 0.999


def logit(price):
    clipped = min(QUANTILE_CEILING, max(QUANTILE_FLOOR, price))
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def recency_weight(end_index, length):
    return 0.5 ** ((length - 1 - end_index) / RECENCY_HALF_LIFE)


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
    return {
        "id": market.get("id"),
        "prices": prices,
        "series": series,
        "flat": len(prices) <= 1,
        "checkpoint": prices[-1] if prices else 0.5,
        "volatilities": volatilities,
        "vol_now": volatilities[-1] if volatilities else 1e-6,
        "one_minute_changes": [
            (
                series[index] - series[index - 1],
                recency_weight(index, len(series)),
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
        weight = recency_weight(end_index, length)
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
            (market[space_key][index] - market[space_key][index - 1],
             recency_weight(index, len(market[space_key])))
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
                change *= 1.2
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
    final_rows = []
    for horizon, row in enumerate(blended_rows, start=1):
        if reversion_move is not None:
            factor = 0.5 if reversion_move > 0.0 else 0.25
            shift = -factor * reversion_move * math.sqrt(horizon / 240.0)
            row = [value + shift for value in row]
        if tick is not None:
            row = [round(value / tick) * tick for value in row]
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
                + SHRINK * pooled_quantiles[index] * pooled_scale
            ) / (ess + SHRINK)
            if checkpoint < 0.01:
                blended_change *= 1.2
            price = sigmoid(
                checkpoint_logit
                + blended_change * h_step_scale * MEMBER_SCALE
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

    for market, rms in zip(markets, member_a_rms):
        market["rms"] = rms
        market["pooled_rms"] = member_a_pooled_rms

    markets_by_id = {id(market): index for index, market in enumerate(markets)}
    forecasts = []
    for market, market_id, checkpoint in entries:
        if market is None or market["flat"]:
            forecasts.append(flat_forecast(market_id, checkpoint, horizon))
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
                tuple(15.0 * multiplier for multiplier in MEMBER_B_SHRINK_MULTIPLIERS),
                tuple(0.5 * multiplier for multiplier in MEMBER_B_SCALE_MULTIPLIERS),
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
                (0.5,) * len(BAND_STEPS),
                False,
                False,
            )
            blended_rows = [
                [
                    0.6 * value_a + 0.2 * value_b + 0.2 * value_c
                    for value_a, value_b, value_c in zip(row_a, row_b, row_c)
                ]
                for row_a, row_b, row_c in zip(
                    member_a_rows, member_b_rows, member_c_rows
                )
            ]
            forecast = finish_forecast(market, blended_rows)
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
