# Student A — Return20 API

Group: 26  
Student: Zheke Yi  
Student ID: 25655150

## Purpose

This FastAPI service provides Return20 for the group project.
It retrieves daily prices from public exchange APIs without API keys.

Return20 is calculated using the closing price of the last completed
UTC day and the closing price 20 calendar days earlier.
Values are decimals: 0.05 means 5%.

## Supported Tokens

BTC, ETH, XRP, SOL, DOGE, XMR, XLM and TRX.

## Endpoints

- GET / — service information
- GET /health — service status
- POST /features — calculate Return20
- GET /docs — interactive API documentation

## Example Request

POST /features

```json
{
  "token": "BTC",
  "forecast_horizon": 7
}
```

The response includes features.Return20, observation_date,
prediction_date and the exchange data source.

## Run Locally

Use Python 3.12.13.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs.

## Render Deployment

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Limitations

Exchange availability and trading-pair support can change.
The service returns an error if complete daily data cannot be obtained.
Binance uses USDT pairs; the Bitfinex fallback uses USD pairs,
so results may differ between providers.

This version provides the shared Return20 feature.
Model prediction and model metadata endpoints are not implemented yet.
