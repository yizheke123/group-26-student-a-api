"""Student A: Zheke Yi (25655150), Group 26.

Provide Return20 and predict the Asset Performance Index
seven calendar days ahead using the exported XGBoost model.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Literal
import logging
import math

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from xgboost import XGBRegressor


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "api_xgboost.json"

DAY_MS = 86_400_000

FEATURE_NAMES = [
    "Return20",
    "TrendStrength",
    "RelativeVolume",
    "Volatility20",
]

# Your current notebook trained and evaluated BTCUSDT only.
TRAINED_TOKENS = ["BTC"]

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


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_error = None

    try:
        model = XGBRegressor()
        model.load_model(MODEL_PATH)

        if model.get_booster().feature_names != FEATURE_NAMES:
            raise ValueError("Model feature names do not match.")

        app.state.model = model

    except Exception:
        logger.exception("Unable to load the XGBoost model.")
        app.state.model_error = (
            "Model unavailable. Check models/api_xgboost.json "
            "and the deployment logs."
        )

    yield


app = FastAPI(
    title="Student A — Asset Performance API",
    description="Return20 features and seven-day API predictions.",
    version="2.0.0",
    lifespan=lifespan,
)


class FeatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: Literal[
        "BTC", "ETH", "XRP", "SOL",
        "DOGE", "XMR", "XLM", "TRX",
    ]

    forecast_horizon: Literal[7] = 7


# --------------------------------------------------
# Public exchange data
# --------------------------------------------------

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


def validate_candles(
    rows,
    close_index,
    volume_index,
    start_ms,
    today_ms,
):
    """Require 21 consecutive completed UTC daily candles."""
    candles = {}

    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Unexpected candle format.")

        timestamp = int(row[0])

        if not start_ms <= timestamp < today_ms:
            continue

        if timestamp % DAY_MS != 0:
            raise ValueError("Candle is not aligned to UTC midnight.")

        if timestamp in candles:
            raise ValueError("Duplicate daily candle.")

        close = float(row[close_index])
        volume = float(row[volume_index])

        if not math.isfinite(close) or close <= 0:
            raise ValueError("Invalid closing price.")

        if not math.isfinite(volume) or volume < 0:
            raise ValueError("Invalid trading volume.")

        candles[timestamp] = {
            "close": close,
            "volume": volume,
        }

    expected = [
        start_ms + i * DAY_MS
        for i in range(21)
    ]

    if sorted(candles) != expected:
        raise ValueError(
            "Missing daily candles or yesterday's candle unavailable."
        )

    return [candles[timestamp] for timestamp in expected]


def load_candles(token, start_ms, today_ms):
    """Use one provider for the entire daily window."""
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
                "volume_index": 5,
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
        "volume_index": 5,
    })

    for provider in providers:
        try:
            rows = read_json(
                provider["url"],
                provider["params"],
            )

            candles = validate_candles(
                rows,
                provider["close_index"],
                provider["volume_index"],
                start_ms,
                today_ms,
            )

            source = {
                "provider": provider["provider"],
                "pair": provider["pair"],
                "quote_currency": provider["quote_currency"],
            }

            return candles, source

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


def market_window(token):
    today = datetime.now(timezone.utc).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    observed = today - timedelta(days=1)
    start = observed - timedelta(days=20)

    candles, source = load_candles(
        token,
        int(start.timestamp() * 1000),
        int(today.timestamp() * 1000),
    )

    return candles, source, observed


def source_warnings(source):
    if source["provider"] == "Bitfinex":
        return [
            "Uses Bitfinex USD candles. The model was trained "
            "on Binance BTCUSDT; prices and volume may differ."
        ]

    return []


# --------------------------------------------------
# Feature calculations — match the notebook
# --------------------------------------------------

def calculate_model_features(candles):
    closes = [row["close"] for row in candles]
    volumes = [row["volume"] for row in candles]

    sma20 = mean(closes[-20:])
    average_volume20 = mean(volumes[-20:])

    if average_volume20 <= 0:
        raise HTTPException(
            status_code=422,
            detail="20-day average volume must be positive.",
        )

    daily_returns = [
        closes[i] / closes[i - 1] - 1
        for i in range(1, len(closes))
    ]

    values = {
        "Return20": closes[-1] / closes[0] - 1,
        "TrendStrength": closes[-1] / sma20 - 1,
        "RelativeVolume": volumes[-1] / average_volume20,
        "Volatility20": stdev(daily_returns),
    }

    if not all(math.isfinite(value) for value in values.values()):
        raise HTTPException(
            status_code=502,
            detail="Unable to calculate finite model features.",
        )

    return values


def current_api(values):
    rs = min(max((values["Return20"] + 0.20) / 0.40, 0), 1)
    ts = min(max((values["TrendStrength"] + 0.10) / 0.20, 0), 1)
    rv = min(values["RelativeVolume"], 2) / 2
    ps = 1 - values["Volatility20"]

    return 100 * (
        0.40 * rs
        + 0.30 * ts
        + 0.20 * rv
        + 0.10 * ps
    )


def require_model():
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail=app.state.model_error,
        )

    return app.state.model


# --------------------------------------------------
# Required endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "student": "Zheke Yi",
        "student_id": "25655150",
        "group": "26",
        "service": "Asset Performance Analytics",
        "target": "API_7d",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    loaded = app.state.model is not None

    return {
        "status": "ready" if loaded else "degraded",
        "model_loaded": loaded,
        "model_error": app.state.model_error,
        "exchange_status": "checked per request",
    }


@app.post("/features")
def features(request: FeatureRequest):
    """Return20 remains available for all eight tokens."""
    candles, source, observed = market_window(request.token)

    return20 = (
        candles[-1]["close"] / candles[0]["close"] - 1
    )

    if not math.isfinite(return20):
        raise HTTPException(
            status_code=502,
            detail="Unable to calculate a finite Return20.",
        )

    return {
        "token": request.token,
        "date": observed.date().isoformat(),
        "observation_date": observed.date().isoformat(),
        "prediction_date": (
            observed + timedelta(days=7)
        ).date().isoformat(),
        "forecast_horizon": 7,
        "features": {
            "Return20": return20,
        },
        "data_source": source,
        "warnings": (
            [
                "Uses Bitfinex USD candles; values may differ "
                "from Binance USDT candles."
            ]
            if source["provider"] == "Bitfinex"
            else []
        ),
    }


@app.post("/predict")
def predict(request: FeatureRequest):
    """Calculate the four features and run the exported model."""
    model = require_model()

    if request.token not in TRAINED_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=(
                "The current model was trained on BTCUSDT only. "
                "Prediction for this token is not supported yet."
            ),
        )

    candles, source, observed = market_window(request.token)
    values = calculate_model_features(candles)

    inputs = pd.DataFrame(
        [values],
        columns=FEATURE_NAMES,
    )

    prediction = float(model.predict(inputs)[0])

    if not math.isfinite(prediction):
        raise HTTPException(
            status_code=500,
            detail="Model returned a non-finite prediction.",
        )

    return {
        "token": request.token,
        "target": "API_7d",
        "prediction": prediction,
        "current_api": current_api(values),
        "unit": "API score points",
        "date": observed.date().isoformat(),
        "observation_date": observed.date().isoformat(),
        "prediction_date": (
            observed + timedelta(days=7)
        ).date().isoformat(),
        "forecast_horizon": 7,
        "features_used": values,
        "feature_sources": {
            name: source["provider"]
            for name in FEATURE_NAMES
        },
        "data_source": source,
        "warnings": source_warnings(source),
    }


@app.get("/model-metadata")
def model_metadata():
    model = require_model()

    return {
        "model_type": "XGBRegressor",
        "target": "API_7d",
        "forecast_horizon_days": 7,
        "trained_tokens": TRAINED_TOKENS,
        "training_pair": "BTCUSDT",
        "features": FEATURE_NAMES,
        "feature_definitions": {
            "Return20": "close[t] / close[t-20] - 1",
            "TrendStrength": "close[t] / SMA20[t] - 1",
            "RelativeVolume": "volume[t] / AverageVolume20[t]",
            "Volatility20": (
                "Sample standard deviation of 20 daily simple "
                "returns; ddof=1; not annualised"
            ),
        },
        "feature_importance": {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES,
                model.feature_importances_,
            )
        },
        "importance_type": "gain",
        "number_of_trees": model.get_booster().num_boosted_rounds(),
    }
