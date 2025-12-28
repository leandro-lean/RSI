import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import pandas_market_calendars as mcal


# =========================
# Config
# =========================
DOCS_DIR = Path("docs")
TICKERS_FILE = Path("tickers.txt")
LATEST_JSON = DOCS_DIR / "latest.json"
LATEST_HTML = DOCS_DIR / "latest.html"

INTERVAL_MINUTES = 30
RSI_PERIOD = 14

# Para evitar tomar la vela "en formación" si el job arranca justo al cierre.
SAFETY_LAG_MINUTES = 2

# Universo grande: bajar en bloques + pausas
CHUNK_SIZE = 50
PAUSE_BETWEEN_CHUNKS_SEC = 0.6

# Si el as-of no cambió, permitimos "refrescar" el mismo cierre por X minutos
# (por si Yahoo/yfinance aún no entregó la última vela o viene incompleta).
REFRESH_SAME_ASOF_WINDOW_MINUTES = 20

US_EASTERN = pytz.timezone("US/Eastern")


# =========================
# Data structures
# =========================
@dataclass
class ScanResult:
    as_of_bar_end_utc: str
    as_of_bar_end_et: str
    updated_utc: str
    oversold: List[dict]
    overbought: List[dict]
    errors: List[dict]


# =========================
# Helpers
# =========================
def load_tickers() -> List[str]:
    raw = [line.strip().upper() for line in TICKERS_FILE.read_text(encoding="utf-8").splitlines()]
    raw = [t for t in raw if t]

    # Dedup preservando orden
    seen = set()
    out = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = df.index
    df = df.copy()
    if getattr(idx, "tz", None) is None:
        df.index = df.index.tz_localize(timezone.utc)
    else:
        df.index = df.index.tz_convert(timezone.utc)
    return df


def build_nyse_schedule_map(start_date: str, end_date: str) -> Dict[datetime.date, Tuple[pd.Timestamp, pd.Timestamp]]:
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start_date, end_date=end_date)

    opens = sched["market_open"].dt.tz_convert(timezone.utc)
    closes = sched["market_close"].dt.tz_convert(timezone.utc)

    out: Dict[datetime.date, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for session_ts in sched.index:
        d = session_ts.date()
        out[d] = (opens.loc[session_ts], closes.loc[session_ts])
    return out


def get_last_closed_bar_end_utc(schedule_map: Dict, now_utc: datetime) -> Optional[pd.Timestamp]:
    """
    Devuelve el fin (bar_end) de la última vela 30m ya cerrada en NYSE RTH.
    Si el mercado está cerrado (fin de semana/feriado), retrocede al último día hábil
    y devuelve el último bar_end de esa sesión (típicamente 16:00 ET).
    """
    effective_now = now_utc - timedelta(minutes=SAFETY_LAG_MINUTES)

    # schedule_map solo incluye sesiones (días hábiles). Ordenamos de más reciente a más viejo.
    session_dates = sorted(schedule_map.keys(), reverse=True)

    for d in session_dates:
        open_utc, close_utc = schedule_map[d]
        open_dt = open_utc.to_pydatetime()    # tz-aware (UTC)
        close_dt = close_utc.to_pydatetime()  # tz-aware (UTC)

        # Si aún no llegamos a la apertura de esa sesión, seguir retrocediendo
        if effective_now < open_dt:
            continue

        # Si estamos después del cierre, fijamos session_now en el cierre (última vela del día)
        session_now = min(effective_now, close_dt)

        elapsed = session_now - open_dt
        bars_closed = int(elapsed.total_seconds() // (INTERVAL_MINUTES * 60))
        if bars_closed <= 0:
            continue

        bar_end = open_dt + timedelta(minutes=INTERVAL_MINUTES * bars_closed)

        # No exceder el cierre (incluye early close)
        if bar_end > close_dt:
            bar_end = close_dt

        # Convertir a pandas Timestamp en UTC sin pasar tz= si ya es tz-aware
        ts = pd.Timestamp(bar_end)
        if ts.tz is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        return ts

    return None


def filter_rth_30m_bars(df: pd.DataFrame, schedule_map: Dict) -> pd.DataFrame:
    """
    Filtra barras 30m por horario RTH del NYSE. Index debe ser UTC y representa inicio de vela.
    Mantiene barras cuyo inicio está en [open, close).
    """
    if df is None or df.empty:
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


def extract_close_series(download_df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if download_df is None or download_df.empty:
        return None

    # Caso multiindex típico de yfinance cuando hay múltiples tickers.
    if isinstance(download_df.columns, pd.MultiIndex):
        # Nivel 0 = ticker, nivel 1 = OHLCV
        if ticker in download_df.columns.get_level_values(0):
            sub = download_df[ticker]
            if "Close" in sub.columns:
                return sub["Close"].dropna()

        # Alternativa: nivel 0 = OHLCV, nivel 1 = ticker
        if ticker in download_df.columns.get_level_values(1):
            if "Close" in download_df.columns.get_level_values(0):
                return download_df["Close"][ticker].dropna()

        return None

    # Caso single ticker
    if "Close" in download_df.columns:
        return download_df["Close"].dropna()

    return None


def read_previous_payload() -> Optional[dict]:
    if not LATEST_JSON.exists():
        return None
    try:
        return json.loads(LATEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_payload_for_compare(payload: dict) -> dict:
    """
    Para evitar commits por 'updated_utc' solamente, comparamos el contenido relevante.
    """
    if payload is None:
        return {}
    out = dict(payload)
    out.pop("updated_utc", None)
    return out


def render_html(result: ScanResult) -> str:
    def table_html(rows: List[dict], title: str, empty_msg: str) -> str:
        if not rows:
            return f"<h2>{title}</h2><p>{empty_msg}</p>"
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
  {table_html(result.oversold, "Oversold (RSI < 30)", "Sin señales.")}
  {table_html(result.overbought, "Overbought (RSI > 70)", "Sin señales.")}
  {err_html}
</body>
</html>
"""


# =========================
# Core scan
# =========================
def scan() -> Optional[ScanResult]:
    tickers = load_tickers()
    now_utc = datetime.now(timezone.utc)

    # Cubrir fines de semana largos / feriados: 21 días hacia atrás
    start = (now_utc - timedelta(days=21)).date().isoformat()
    end = (now_utc + timedelta(days=1)).date().isoformat()
    schedule_map = build_nyse_schedule_map(start, end)

    target_bar_end_utc = get_last_closed_bar_end_utc(schedule_map, now_utc)
    if target_bar_end_utc is None:
        # No hay sesión válida en el rango (muy improbable). Publicamos estado.
        result = ScanResult(
            as_of_bar_end_utc="",
            as_of_bar_end_et="",
            updated_utc=now_utc.isoformat(),
            oversold=[],
            overbought=[],
            errors=[{"ticker": "ALL", "reason": "No valid NYSE session found in schedule range"}],
        )
        return result

    # Si ya publicamos este as-of hace mucho, no vale la pena recalcular (salvo ventana corta de refresh)
    prev = read_previous_payload()
    prev_asof = (prev or {}).get("as_of_bar_end_utc", "")
    prev_comp = normalize_payload_for_compare(prev) if prev else None

    age_minutes = (now_utc - target_bar_end_utc.to_pydatetime()).total_seconds() / 60.0
    allow_refresh_same_asof = age_minutes <= REFRESH_SAME_ASOF_WINDOW_MINUTES

    if prev_asof == target_bar_end_utc.isoformat() and not allow_refresh_same_asof:
        # Mismo cierre y ya pasó la ventana de consolidación de datos: no hacemos nada.
        return None

    target_bar_start_utc = (target_bar_end_utc - pd.Timedelta(minutes=INTERVAL_MINUTES)).tz_convert(timezone.utc)

    oversold: List[dict] = []
    overbought: List[dict] = []
    errors: List[dict] = []

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
                prepost=False,  # RTH; igual filtramos por calendario NYSE
                threads=True,
                progress=False,
            )
            if df is None or df.empty:
                for t in chunk:
                    errors.append({"ticker": t, "reason": "Empty download"})
                time.sleep(PAUSE_BETWEEN_CHUNKS_SEC)
                continue

            df = ensure_utc_index(df)

            for t in chunk:
                close = extract_close_series(df, t)
                if close is None or close.empty:
                    errors.append({"ticker": t, "reason": "No Close series"})
                    continue

                tmp = pd.DataFrame({"Close": close}).dropna()
                tmp = ensure_utc_index(tmp)
                tmp = filter_rth_30m_bars(tmp, schedule_map)

                if tmp.empty:
                    errors.append({"ticker": t, "reason": "No RTH bars"})
                    continue

                rsi = wilder_rsi(tmp["Close"], RSI_PERIOD)

                # yfinance index = inicio de vela; usamos target_bar_start_utc
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

    result = ScanResult(
        as_of_bar_end_utc=target_bar_end_utc.isoformat(),
        as_of_bar_end_et=target_bar_end_utc.tz_convert(US_EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z"),
        updated_utc=now_utc.isoformat(),
        oversold=oversold,
        overbought=overbought,
        errors=errors,
    )

    # Comparación: si no cambia el contenido relevante, no escribimos archivos (evita commits inútiles)
    new_payload = {
        "as_of_bar_end_utc": result.as_of_bar_end_utc,
        "as_of_bar_end_et": result.as_of_bar_end_et,
        "updated_utc": result.updated_utc,
        "oversold": result.oversold,
        "overbought": result.overbought,
        "errors": result.errors,
    }
    new_comp = normalize_payload_for_compare(new_payload)

    if prev_comp is not None and new_comp == prev_comp:
        return None

    return result


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    result = scan()
    if result is None:
        return  # sin cambios => nada que commitear

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


