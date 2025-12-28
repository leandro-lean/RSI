import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import pandas_market_calendars as mcal


DOCS_DIR = Path("docs")
TICKERS_FILE = Path("tickers.txt")
LATEST_JSON = DOCS_DIR / "latest.json"
LATEST_HTML = DOCS_DIR / "latest.html"

INTERVAL_MINUTES = 30
RSI_PERIOD = 14

# Tolerancia para evitar tomar la vela "en formación" si el job arrancó justo al cierre.
SAFETY_LAG_MINUTES = 2

# Descarga por bloques para reducir fallas / throttling.
CHUNK_SIZE = 50
PAUSE_BETWEEN_CHUNKS_SEC = 0.6

US_EASTERN = pytz.timezone("US/Eastern")


@dataclass
class ScanResult:
    as_of_bar_end_utc: str
    as_of_bar_end_et: str
    updated_utc: str
    oversold: List[dict]
    overbought: List[dict]
    errors: List[dict]


def load_tickers() -> List[str]:
    raw = [line.strip().upper() for line in TICKERS_FILE.read_text(encoding="utf-8").splitlines()]
    raw = [t for t in raw if t]
    # Dedup preservando orden
    seen = set()
    tickers = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            tickers.append(t)
    return tickers


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    # RSI de Wilder: promedios suavizados equivalentes a EMA(alpha=1/period) sin adjust.
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if getattr(idx, "tz", None) is None:
        # si viene naive, asumimos UTC
        df = df.copy()
        df.index = df.index.tz_localize(timezone.utc)
    else:
        df = df.copy()
        df.index = df.index.tz_convert(timezone.utc)
    return df


def build_nyse_schedule_map(start_date: str, end_date: str) -> Dict[datetime.date, Tuple[pd.Timestamp, pd.Timestamp]]:
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start_date, end_date=end_date)
    # schedule suele venir tz-aware (habitualmente UTC); forzamos UTC por seguridad.
    opens = sched["market_open"].dt.tz_convert(timezone.utc)
    closes = sched["market_close"].dt.tz_convert(timezone.utc)

    out = {}
    for session_ts in sched.index:
        d = session_ts.date()
        out[d] = (opens.loc[session_ts], closes.loc[session_ts])
    return out


def get_last_closed_bar_end_utc(schedule_map: Dict, now_utc: datetime) -> Optional[pd.Timestamp]:
    # Detecta el último cierre de vela 30m ya consolidado, basado en horario NYSE.
    effective_now = now_utc - timedelta(minutes=SAFETY_LAG_MINUTES)

    # Buscamos el "session date" que corresponde a effective_now (UTC).
    # Usamos la fecha UTC como punto de partida, pero validamos contra schedule_map.
    for candidate_date in [effective_now.date(), (effective_now - timedelta(days=1)).date()]:
        if candidate_date not in schedule_map:
            continue
        open_utc, close_utc = schedule_map[candidate_date]
        if effective_now < open_utc.to_pydatetime():
            continue  # antes de abrir, no hay vela cerrada en esta sesión
        session_now = min(effective_now, close_utc.to_pydatetime())
        elapsed = session_now - open_utc.to_pydatetime()
        bars_closed = int(elapsed.total_seconds() // (INTERVAL_MINUTES * 60))
        if bars_closed <= 0:
            return None
        bar_end = open_utc.to_pydatetime() + timedelta(minutes=INTERVAL_MINUTES * bars_closed)
        # No exceder el cierre
        if bar_end > close_utc.to_pydatetime():
            bar_end = close_utc.to_pydatetime()
        return pd.Timestamp(bar_end, tz=timezone.utc)

    return None


def filter_rth_30m_bars(df: pd.DataFrame, schedule_map: Dict) -> pd.DataFrame:
    # df index UTC, representa inicio de vela.
    # Mantener velas cuyo inicio esté dentro de [open, close) de cada sesión.
    if df.empty:
        return df

    idx = df.index
    keep = []
    for ts in idx:
        d = ts.date()
        if d not in schedule_map:
            keep.append(False)
            continue
        open_utc, close_utc = schedule_map[d]
        keep.append((ts >= open_utc) and (ts < close_utc))
    return df.loc[keep]


def extract_close_frame(download_df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if download_df is None or download_df.empty:
        return None

    if isinstance(download_df.columns, pd.MultiIndex):
        # group_by="ticker" usual: level 0 ticker, level 1 OHLCV
        if ticker in download_df.columns.get_level_values(0):
            if ("Close" in download_df[ticker].columns):
                return download_df[ticker]["Close"].dropna()
        # alternativa: level 0 OHLCV, level 1 ticker
        if ticker in download_df.columns.get_level_values(1):
            if "Close" in download_df.columns.get_level_values(0):
                return download_df["Close"][ticker].dropna()

    # single ticker (no multiindex)
    if "Close" in download_df.columns:
        return download_df["Close"].dropna()

    return None


def render_html(result: ScanResult) -> str:
    def table_html(rows: List[dict], title: str) -> str:
        if not rows:
            return f"<h2>{title}</h2><p>Sin señales.</p>"
        hdr = "<tr><th>Ticker</th><th>RSI(14)</th><th>Bar End (ET)</th><th>Close</th></tr>"
        body = "\n".join(
            f"<tr><td>{r['ticker']}</td><td>{r['rsi']:.2f}</td><td>{r['bar_end_et']}</td><td>{r.get('close','')}</td></tr>"
            for r in rows
        )
        return f"<h2>{title} ({len(rows)})</h2><table>{hdr}{body}</table>"

    err_html = ""
    if result.errors:
        err_rows = "\n".join(
            f"<tr><td>{e['ticker']}</td><td>{e['reason']}</td></tr>" for e in result.errors
        )
        err_html = f"""
        <h2>Errores / Sin datos ({len(result.errors)})</h2>
        <table><tr><th>Ticker</th><th>Motivo</th></tr>{err_rows}</table>
        """

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RSI 30m Scanner</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .meta {{ color: #444; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 26px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f3f3f3; }}
    h1 {{ margin-top: 0; }}
  </style>
</head>
<body>
  <h1>RSI(14) Scanner — 30m — NYSE RTH</h1>
  <div class="meta">
    <div><b>As-of bar end (ET):</b> {result.as_of_bar_end_et}</div>
    <div><b>As-of bar end (UTC):</b> {result.as_of_bar_end_utc}</div>
    <div><b>Updated (UTC):</b> {result.updated_utc}</div>
  </div>
  {table_html(result.oversold, "Oversold (RSI < 30)")}
  {table_html(result.overbought, "Overbought (RSI > 70)")}
  {err_html}
</body>
</html>
"""


def read_previous_asof() -> Optional[str]:
    if not LATEST_JSON.exists():
        return None
    try:
        data = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
        return data.get("as_of_bar_end_utc")
    except Exception:
        return None


def scan() -> ScanResult:
    tickers = load_tickers()

    now_utc = datetime.now(timezone.utc)

    # schedule map para cubrir últimos días (incluye early closes y feriados)
    start = (now_utc - timedelta(days=10)).date().isoformat()
    end = (now_utc + timedelta(days=1)).date().isoformat()
    schedule_map = build_nyse_schedule_map(start, end)

    target_bar_end_utc = get_last_closed_bar_end_utc(schedule_map, now_utc)
    if target_bar_end_utc is None:
        # Mercado cerrado o todavía no cerró la primera vela de 30m
        as_of_utc = ""
        as_of_et = ""
        return ScanResult(
            as_of_bar_end_utc=as_of_utc,
            as_of_bar_end_et=as_of_et,
            updated_utc=now_utc.isoformat(),
            oversold=[],
            overbought=[],
            errors=[{"ticker": "ALL", "reason": "Market closed or first 30m bar not closed yet"}],
        )

    # Evitar commits si seguimos en la misma vela objetivo
    prev_asof = read_previous_asof()
    if prev_asof == target_bar_end_utc.isoformat():
        # Sin novedad
        return ScanResult(
            as_of_bar_end_utc=target_bar_end_utc.isoformat(),
            as_of_bar_end_et=target_bar_end_utc.tz_convert(US_EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z"),
            updated_utc=now_utc.isoformat(),
            oversold=[],
            overbought=[],
            errors=[{"ticker": "ALL", "reason": "No new 30m bar since last update"}],
        )

    target_bar_start_utc = (target_bar_end_utc - pd.Timedelta(minutes=INTERVAL_MINUTES)).tz_convert(timezone.utc)

    oversold = []
    overbought = []
    errors = []

    # Descarga por chunks
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        try:
            df = yf.download(
                tickers=chunk,
                interval="30m",
                period="5d",
                group_by="ticker",
                auto_adjust=False,
                prepost=False,
                threads=True,
                progress=False,
            )
            if df is None or df.empty:
                for t in chunk:
                    errors.append({"ticker": t, "reason": "Empty download"})
                continue

            df = ensure_utc_index(df)

            for t in chunk:
                close = extract_close_frame(df, t)
                if close is None or close.empty:
                    errors.append({"ticker": t, "reason": "No Close series"})
                    continue

                # Frame con Close + filtro RTH
                tmp = pd.DataFrame({"Close": close}).dropna()
                tmp = ensure_utc_index(tmp)
                tmp = filter_rth_30m_bars(tmp, schedule_map)

                if tmp.empty or "Close" not in tmp.columns:
                    errors.append({"ticker": t, "reason": "No RTH bars"})
                    continue

                rsi = wilder_rsi(tmp["Close"], RSI_PERIOD)
                if target_bar_start_utc not in rsi.index:
                    errors.append({"ticker": t, "reason": "Missing target bar"})
                    continue

                rsi_val = float(rsi.loc[target_bar_start_utc])
                if np.isnan(rsi_val):
                    errors.append({"ticker": t, "reason": "RSI NaN"})
                    continue

                close_val = float(tmp["Close"].loc[target_bar_start_utc])
                bar_end_et = target_bar_end_utc.tz_convert(US_EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z")

                row = {
                    "ticker": t,
                    "rsi": rsi_val,
                    "bar_end_et": bar_end_et,
                    "close": round(close_val, 4),
                }

                if rsi_val < 30:
                    oversold.append(row)
                elif rsi_val > 70:
                    overbought.append(row)

        except Exception as e:
            for t in chunk:
                errors.append({"ticker": t, "reason": f"Exception: {type(e).__name__}"})

        time.sleep(PAUSE_BETWEEN_CHUNKS_SEC)

    oversold.sort(key=lambda x: x["rsi"])
    overbought.sort(key=lambda x: x["rsi"], reverse=True)

    return ScanResult(
        as_of_bar_end_utc=target_bar_end_utc.isoformat(),
        as_of_bar_end_et=target_bar_end_utc.tz_convert(US_EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z"),
        updated_utc=now_utc.isoformat(),
        oversold=oversold,
        overbought=overbought,
        errors=errors,
    )


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    result = scan()

    payload = {
        "as_of_bar_end_utc": result.as_of_bar_end_utc,
        "as_of_bar_end_et": result.as_of_bar_end_et,
        "updated_utc": result.updated_utc,
        "oversold": result.oversold,
        "overbought": result.overbought,
        "errors": result.errors,
    }

    LATEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LATEST_HTML.write_text(render_html(result), encoding="utf-8")


if __name__ == "__main__":
    main()


