"""Student A: Zheke Yi (25655150), Group 26.
POST /features provides Return20 using completed UTC daily candles.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal
import math
import logging

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


app = FastAPI(
    title="Student A — Return20 API",
    description="20-day return from public exchange daily data.",
    version="1.0.0",
)

logger = logging.getLogger(__name__)

DAY_MS = 86_400_000

# USD pairs are used when falling back to Bitfinex.
BITFINEX_PAIRS = {
    "BTC": "tBTCUSD",
    "ETH": "tETHUSD",
    "XRP": "tXRPUSD",
    "SOL": "tSOLUSD",
    "DOGE": "tDOGE:USD",
    "XMR": "tXMRUSD",
    "XLM": "tXLMUSD",
    "TRX": "tTRXUSD",
}


class FeatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: Literal[
        "BTC", "ETH", "XRP", "SOL",
        "DOGE", "XMR", "XLM", "TRX",
    ]

    forecast_horizon: Literal[7] = 7


def read_json(url, params):
    response = requests.get(
        url,
        params=params,
        timeout=(3, 7),
    )
    response.raise_for_status()

    rows = response.json()

    if not isinstance(rows, list) or not rows:
        raise ValueError("Exchange returned no candle data.")

    return rows


def validate_closes(rows, close_index, start_ms, today_ms):
    """Require exactly 21 consecutive completed UTC daily closes."""
    closes = {}

    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Unexpected candle format.")

        timestamp = int(row[0])

        # Ignore candles outside the requested completed period.
        if not start_ms <= timestamp < today_ms:
            continue

        if timestamp % DAY_MS != 0:
            raise ValueError("Candle is not aligned to UTC midnight.")

        if timestamp in closes:
            raise ValueError("Duplicate daily candle.")

        close = float(row[close_index])

        if not math.isfinite(close) or close <= 0:
            raise ValueError("Invalid closing price.")

        closes[timestamp] = close

    expected = [
        start_ms + i * DAY_MS
        for i in range(21)
    ]

    if sorted(closes) != expected:
        raise ValueError(
            "Missing daily candles or yesterday's candle unavailable."
        )

    return [closes[timestamp] for timestamp in expected]


def load_closes(token, start_ms, today_ms):
    """Try Binance, then Bitfinex; never mix providers in one window."""
    providers = []

    if token != "XMR":
        for host in (
            "https://data-api.binance.vision",
            "https://api.binance.com",
        ):
            providers.append({
                "provider": "Binance",
                "pair": f"{token}USDT",
                "quote_currency": "USDT",
                "url": f"{host}/api/v3/klines",
                "params": {
                    "symbol": f"{token}USDT",
                    "interval": "1d",
                    "startTime": start_ms,
                    "endTime": today_ms - 1,
                    "limit": 30,
                },
                "close_index": 4,
            })

    pair = BITFINEX_PAIRS[token]

    providers.append({
        "provider": "Bitfinex",
        "pair": pair,
        "quote_currency": "USD",
        "url": (
            "https://api-pub.bitfinex.com/v2/candles/"
            f"trade:1D:{pair}/hist"
        ),
        "params": {
            "start": start_ms,
            "end": today_ms - 1,
            "sort": 1,
            "limit": 30,
        },
        "close_index": 2,
    })

    for provider in providers:
        try:
            rows = read_json(
                provider["url"],
                provider["params"],
            )

            closes = validate_closes(
                rows,
                provider["close_index"],
                start_ms,
                today_ms,
            )

            source = {
                "provider": provider["provider"],
                "pair": provider["pair"],
                "quote_currency": provider["quote_currency"],
            }

            return closes, source

        except (
            requests.RequestException,
            ValueError,
            TypeError,
            IndexError,
            OverflowError,
        ) as exc:
            logger.warning(
                "%s data unavailable for %s: %s",
                provider["provider"],
                token,
                type(exc).__name__,
            )

    raise HTTPException(
        status_code=503,
        detail=(
            f"Unable to obtain 21 consecutive completed daily "
            f"candles for {token}. Please retry later."
        ),
    )


@app.get("/")
def root():
    return {
        "student": "Zheke Yi",
        "student_id": "25655150",
        "group": "26",
        "service": "Return20 feature API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    # Service availability only; exchanges are checked per request.
    return {
        "status": "ok",
        "exchange_status": "checked when /features is called",
    }


@app.post("/features")
def features(request: FeatureRequest):
    today = datetime.now(timezone.utc).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    observation_date = today - timedelta(days=1)

    # Yesterday plus the preceding 20 days = 21 daily closes.
    start = observation_date - timedelta(days=20)

    today_ms = int(today.timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)

    closes, source = load_closes(
        request.token,
        start_ms,
        today_ms,
    )

    return20 = closes[-1] / closes[0] - 1

    if not math.isfinite(return20):
        raise HTTPException(
            status_code=502,
            detail="Unable to calculate a finite Return20.",
        )

    warnings = []

    if source["provider"] == "Bitfinex":
        warnings.append(
            "Uses Bitfinex USD candles; values may differ "
            "from Binance USDT candles."
        )

    return {
        "token": request.token,
        "date": observation_date.date().isoformat(),
        "observation_date": observation_date.date().isoformat(),
        "prediction_date": (
            observation_date + timedelta(days=7)
        ).date().isoformat(),
        "forecast_horizon": request.forecast_horizon,
        "features": {
            "Return20": return20,
        },
        "data_source": source,
        "warnings": warnings,
    }
