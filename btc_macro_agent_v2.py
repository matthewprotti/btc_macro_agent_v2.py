#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Macro-Tracker Agent — v2.2 (pandas-safe)

Changes vs v2.1:
- Replaced DatetimeIndex.get_loc(method="pad") with get_indexer(..., method="pad")
  + safe fallback when indexer returns -1.
- Retains robust FRED loader, guarded WALCL/DFII10, benign Binance failures.
"""

import os, io, re, sys, json, hashlib, logging, math
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False

# --------------------------- Config -----------------------------------------
DATA_DIR = os.environ.get("BTC_MACRO_DATA_DIR", "/mnt/data/data")
TRACK_CSV = os.environ.get("BTC_MACRO_TRACK", "/mnt/data/track.csv")
LOG_FILE  = os.environ.get("BTC_MACRO_LOG", "/mnt/data/log.txt")
os.makedirs(DATA_DIR, exist_ok=True)

RUN_TZ = timezone.utc
REFIT_WINDOW_DAYS = 180
GLI_LAG_DAYS = 70
PREDICT_NEXT_DAY = True  # use X_{t-1} to predict ΔlogBTC_t

# Optional notifiers
SLACK_WEBHOOK  = os.environ.get("SLACK_WEBHOOK", "").strip()
TELEGRAM_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TELEGRAM_CHAT  = os.environ.get("TG_CHAT_ID", "").strip()
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASS = os.environ.get("SMTP_PASS", "").strip()
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)

# ------------------------ HTTP + fetch --------------------------------------
def _http_get(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        logging.warning(f"HTTP GET failed for {url}: {e}")
        return None

def fetch_to_file(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in (1, 2):
        data = _http_get(url)
        if data:
            with open(path, "wb") as f:
                f.write(data)
            logging.info(f"Fetched {url} → {path} ({len(data)} bytes)")
            return
        logging.warning(f"Retry {attempt} failed: {url}")
    logging.error(f"Writing empty file after failures: {path}")
    open(path, "wb").close()

SOURCES = [
    ("BTCUSD.csv", "https://stooq.com/q/d/l/?s=btcusd&i=d"),
    ("DXY.csv",    "https://stooq.com/q/d/l/?s=dx.c&i=d"),
    ("WALCL.csv",  "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WALCL"),
    ("M2SL.csv",   "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"),
    ("REALYLD.csv","https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"),
    ("FUNDING.json","https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000"),
    ("OPENINT.json","https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT"),
    # Optional locals: data/BIS_GLI.csv ; data/ETF_FLOWS.csv or data/etf_flows.html
]

def fetch_all(skip: bool = False):
    if skip:
        logging.info("SKIP_FETCH=1 → skipping remote downloads.")
        return
    for fname, url in SOURCES:
        fetch_to_file(url, os.path.join(DATA_DIR, fname))

# ---------------------------- Loaders ---------------------------------------
def load_stooq_csv(path: str, col_close: str = "Close") -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["date", "close"])
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python")
    for c in df.columns:
        if str(c).lower().startswith("date"):
            df = df.rename(columns={c: "date"})
    if "date" not in df.columns or col_close not in df.columns:
        return pd.DataFrame(columns=["date","close"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    out = df[["date", col_close]].rename(columns={col_close: "close"})
    return out.sort_values("date").dropna()

def load_fred_two_col(path: str, value_name: str) -> pd.DataFrame:
    """Robust FRED loader → 2 cols ['date', value_name] (may be empty)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["date", value_name])
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python")
    if df.empty:
        return pd.DataFrame(columns=["date", value_name])

    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    df = df.dropna(how="all")

    # Choose date/value columns flexibly
    date_candidates = [c for c in cols if c.lower() in ("date","observation_date") or c.upper().startswith("DATE")]
    date_col = date_candidates[0] if date_candidates else cols[0]
    value_candidates = [c for c in cols if c != date_col]
    val_col = None
    for c in value_candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(1, len(df)//5):
            val_col = c; break
    if val_col is None:
        val_col = value_candidates[-1] if value_candidates else cols[-1]

    out = df[[date_col, val_col]].copy()
    out.columns = ["date", value_name]
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")

def load_bis_gli(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["date", "global_m2"])
    df = pd.read_csv(path)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "global_m2"})
    if "global_m2" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "global_m2"})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["global_m2"] = pd.to_numeric(df["global_m2"], errors="coerce")
    return df.dropna().sort_values("date")

def parse_etf_flows(path_csv_or_html: str) -> pd.DataFrame:
    """Parse Farside ETF flows (CSV or saved HTML) → df[date, net_flow]."""
    if not os.path.exists(path_csv_or_html) or os.path.getsize(path_csv_or_html) == 0:
        return pd.DataFrame(columns=["date","net_flow"])
    ext = os.path.splitext(path_csv_or_html)[1].lower()
    raw = open(path_csv_or_html, "rb").read()

    if ext in (".html",".htm"):
        try:
            tables = pd.read_html(io.BytesIO(raw))
            target = None
            for t in tables:
                if isinstance(t.columns, pd.MultiIndex):
                    last = t.columns[-1]
                    if any("total" in str(level).lower() for level in last):
                        target = t.copy(); break
                else:
                    if "total" in " ".join(map(str,t.columns)).lower():
                        target = t.copy(); break
            if target is None:
                target = tables[0].copy()
            df = target
            # Find date-like column
            def is_dateish(x: str) -> bool:
                s = str(x).strip()
                return bool(re.search(r"\d{1,2}\s+\w+\s+\d{4}", s)) or bool(re.match(r"\d{4}-\d{2}-\d{2}", s))
            date_col = None
            for c in df.columns:
                if any(is_dateish(v) for v in df[c].astype(str).head(10).tolist()):
                    date_col = c; break
            if date_col is None:
                return pd.DataFrame(columns=["date","net_flow"])
            flow_col = df.columns[-1]
            df = df[[date_col, flow_col]].copy()
            df.columns = ["date","total"]
        except Exception as e:
            logging.warning(f"read_html failed: {e}")
            return pd.DataFrame(columns=["date","net_flow"])
    else:
        df = pd.read_csv(io.BytesIO(raw))
        cols = [c.lower() for c in df.columns]
        date_idx = cols.index("date") if "date" in cols else 0
        flow_idx = cols.index("total") if "total" in cols else (len(cols)-1)
        df = df[[df.columns[date_idx], df.columns[flow_idx]]]
        df.columns = ["date","total"]

    def to_float(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().replace(",", "")
        s = s.replace("–","-").replace("—","-").replace("\u2212","-")
        if s in ("","-"): return 0.0
        if s.startswith("(") and s.endswith(")"): s = "-" + s[1:-1]
        try: return float(s)
        except Exception: return np.nan

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, dayfirst=True)
    df["net_flow"] = df["total"].apply(to_float)
    return df.dropna(subset=["date"])[["date","net_flow"]].sort_values("date")

# ---------------------------- Features --------------------------------------
def build_features(today_utc: datetime) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    diags: Dict[str, Any] = {}

    btc = load_stooq_csv(os.path.join(DATA_DIR,"BTCUSD.csv")).rename(columns={"close":"btc_close"})
    btc.set_index("date", inplace=True)
    diags["btc_last"] = str(btc.index.max())

    btc["btc_50d"] = btc["btc_close"].rolling(50).mean()
    btc["btc_200d"] = btc["btc_close"].rolling(200).mean()

    dxy = load_stooq_csv(os.path.join(DATA_DIR,"DXY.csv")).rename(columns={"close":"dxy"})
    dxy.set_index("date", inplace=True)
    btc = btc.join(dxy["dxy"], how="left")
    btc["dxy_delta"] = btc["dxy"].diff()

    # WALCL (guarded)
    wal = load_fred_two_col(os.path.join(DATA_DIR,"WALCL.csv"), "walcl")
    if wal.empty or "walcl" not in wal.columns:
        btc["fed_delta"] = np.nan
    else:
        wal.set_index("date", inplace=True)
        wal["fed_delta"] = wal["walcl"].diff()
        btc["fed_delta"] = wal["fed_delta"].reindex(btc.index).ffill()

    # M2 YoY (guarded)
    m2 = load_fred_two_col(os.path.join(DATA_DIR,"M2SL.csv"), "m2")
    if not m2.empty and "m2" in m2.columns:
        m2["dm_m2_yoy"] = m2["m2"].pct_change(12)
        last_m2_date = m2["date"].max()
        diags["m2_last"] = str(last_m2_date)
        btc["dm_m2_yoy"] = m2.set_index("date")["dm_m2_yoy"].reindex(btc.index).ffill()
        diags["m2_stale_days"] = int((btc.index.max() - last_m2_date).days) if isinstance(last_m2_date, pd.Timestamp) else None
    else:
        btc["dm_m2_yoy"] = np.nan
        diags["m2_stale_days"] = None

    # Real yield (guarded)
    real = load_fred_two_col(os.path.join(DATA_DIR,"REALYLD.csv"), "real_yld")
    if real.empty or "real_yld" not in real.columns:
        btc["real_yld"] = np.nan
        btc["real_yld_delta"] = np.nan
    else:
        real.set_index("date", inplace=True)
        real = real.reindex(btc.index).ffill()
        btc["real_yld"] = real["real_yld"]
        btc["real_yld_delta"] = btc["real_yld"].diff()

    # ETF flows
    flows_csv  = os.path.join(DATA_DIR,"ETF_FLOWS.csv")
    flows_html = os.path.join(DATA_DIR,"etf_flows.html")
    if os.path.exists(flows_csv):
        etf = parse_etf_flows(flows_csv)
    elif os.path.exists(flows_html):
        etf = parse_etf_flows(flows_html)
    else:
        etf = pd.DataFrame(columns=["date","net_flow"])
    etf = etf.set_index("date").sort_index()
    etf["etf_3d"] = etf["net_flow"].rolling(3).sum()
    etf_aligned = etf["etf_3d"].reindex(btc.index).ffill()
    btc["etf_3d"] = etf_aligned
    diags["etf_3d_stale"] = (
        not pd.notna(etf_aligned.iloc[-1]) or
        (etf.index.max() is pd.NaT) or
        (btc.index.max() - etf.index.max() > pd.Timedelta(days=5))
    )

    # Funding (annualized from last 3 prints)
    annual = np.nan
    fund_path = os.path.join(DATA_DIR,"FUNDING.json")
    try:
        if os.path.exists(fund_path) and os.path.getsize(fund_path) > 0:
            arr = json.load(open(fund_path))
            if isinstance(arr, dict): arr = [arr]
            arr = [a for a in arr if "fundingRate" in a]
            arr.sort(key=lambda x: int(x.get("fundingTime",0)))
            last3 = arr[-3:]
            rates = [float(x["fundingRate"]) for x in last3]
            if rates:
                annual = float(np.mean(rates) * 3 * 365)
    except Exception as e:
        logging.warning(f"FUNDING parse failed: {e}")
    btc["funding"] = float(annual) if pd.notna(annual) else np.nan

    # Global M2 lag (optional)
    gli = load_bis_gli(os.path.join(DATA_DIR,"BIS_GLI.csv"))
    if not gli.empty:
        gli.set_index("date", inplace=True)
        btc["global_m2_lag"] = gli["global_m2"].reindex(btc.index).ffill().shift(GLI_LAG_DAYS)
    else:
        btc["global_m2_lag"] = np.nan

    end_date = pd.Timestamp(datetime.now(tz=RUN_TZ).date(), tz=RUN_TZ)
    btc = btc[btc.index <= end_date]
    return btc, diags

# ----------------------------- Gates ----------------------------------------
def evaluate_gates(df: pd.DataFrame, date: pd.Timestamp) -> Dict[str,int]:
    r = df.loc[date]
    fed_ok  = (r["fed_delta"] >= 0) if pd.notna(r["fed_delta"]) else False
    m2_ok   = (r["dm_m2_yoy"] > 0.03) if pd.notna(r["dm_m2_yoy"]) else False
    real_ok = (r["real_yld"] < 1.8) if pd.notna(r["real_yld"]) else False
    dxy_ok  = (r["dxy"] < 100) if pd.notna(r["dxy"]) else False
    macro_gate = 1 if (fed_ok + m2_ok + real_ok + dxy_ok) >= 2 else 0

    price_gate = 0
    if pd.notna(r.get("btc_50d")) and pd.notna(r.get("btc_200d")):
        price_gate = int((r["btc_close"] > r["btc_50d"]) and (r["btc_close"] > r["btc_200d"]))

    sent_gate = 0
    if pd.notna(r.get("etf_3d")) and pd.notna(r.get("funding")):
        sent_gate = 1 if (r["etf_3d"] >= 250) and (r["funding"] <= 0.15) else 0

    return {"macro_gate": macro_gate, "price_gate": price_gate, "sent_gate": sent_gate}

# ---------------------------- Regression ------------------------------------
def refit_needed(today_utc: datetime) -> bool:
    return today_utc.weekday() == 0  # Monday

def last_complete_day_for_refit(df: pd.DataFrame, today_utc: datetime) -> Optional[pd.Timestamp]:
    end = pd.Timestamp(today_utc.date(), tz=RUN_TZ)
    Xcols = ["dm_m2_yoy","dxy_delta","real_yld_delta","etf_3d","global_m2_lag"]
    y = np.log(df["btc_close"]).diff()
    X = df[Xcols].shift(1)
    good = (~X.isna().any(axis=1)) & (~y.isna())
    candidates = df.index[(good) & (df.index <= end)]
    return candidates[-1] if len(candidates) else None

def fit_regression(df: pd.DataFrame, refit_day: pd.Timestamp):
    end = refit_day
    start = end - pd.Timedelta(days=REFIT_WINDOW_DAYS)
    window = df.loc[(df.index >= start) & (df.index <= end)].copy()
    y = np.log(window["btc_close"]).diff()
    X = window[["dm_m2_yoy","dxy_delta","real_yld_delta","etf_3d","global_m2_lag"]].shift(1)
    D = pd.concat([y, X], axis=1).dropna()
    if D.empty or D.shape[0] < 30:
        return {}, {"refit_date": None, "refit_n": int(D.shape[0])}
    Y = D.iloc[:,0].values
    Xmat = D.iloc[:,1:].values
    names = D.columns[1:].tolist()
    meta = {"refit_date": str(end.date()), "refit_n": int(D.shape[0])}
    if HAVE_SM:
        Xsm = sm.add_constant(Xmat)
        model = sm.OLS(Y, Xsm).fit(cov_type="HAC", cov_kwds={"maxlags":5})
        betas = dict(zip(["intercept"] + names, model.params.tolist()))
        meta["refit_r2"] = float(model.rsquared)
    else:
        Xols = np.column_stack([np.ones(len(Y)), Xmat])
        beta_vec = np.linalg.lstsq(Xols, Y, rcond=None)[0]
        betas = dict(zip(["intercept"] + names, beta_vec.tolist()))
        yhat = Xols @ beta_vec
        ssr = float(np.sum((Y - yhat)**2)); sst = float(np.sum((Y - np.mean(Y))**2))
        meta["refit_r2"] = 1.0 - ssr/sst if sst > 0 else np.nan
    return betas, meta

def apply_betas(df: pd.DataFrame, betas: Dict[str,float]) -> pd.Series:
    if not betas:
        return pd.Series(dtype=float)
    names = ["dm_m2_yoy","dxy_delta","real_yld_delta","etf_3d","global_m2_lag"]
    Xrow = df[names].shift(1).iloc[-1]
    if Xrow.isna().any():
        return pd.Series(dtype=float)
    intercept = betas.get("intercept", 0.0)
    delta = intercept + float(np.dot(Xrow.values, np.array([betas.get(n,0.0) for n in names])))
    last_close = df["btc_close"].iloc[-1]
    fitted = float(np.exp(np.log(last_close) + delta))
    return pd.Series([fitted], index=[df.index[-1]])

# ----------------------- Notifier & utilities -------------------------------
def _pad_index(index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp:
    """Return last index value <= ts; fallback to first/last gracefully."""
    idx = index.get_indexer([ts], method="pad")[0]
    if idx == -1:
        # target earlier than first index value
        return index[0]
    return index[idx]

def gate_change_alert(prev_gates: Dict[str,int], new_gates: Dict[str,int], date: str) -> Optional[str]:
    changes = []
    for k in ("macro_gate","price_gate","sent_gate"):
        if prev_gates.get(k) != new_gates.get(k):
            changes.append(f"{k}: {prev_gates.get(k)} → {new_gates.get(k)}")
    if not changes:
        return None
    return f"Gate change on {date}: " + "; ".join(changes)

def send_alert(msg: str):
    sent = False
    if SLACK_WEBHOOK:
        try:
            import urllib.request, json as _json
            data = _json.dumps({"text": msg}).encode("utf-8")
            req = urllib.request.Request(SLACK_WEBHOOK, data=data, headers={"Content-Type":"application/json"})
            urllib.request.urlopen(req, timeout=10).read()
            sent = True
        except Exception as e:
            logging.warning(f"Slack alert failed: {e}")
    if (not sent) and TELEGRAM_TOKEN and TELEGRAM_CHAT:
        try:
            import urllib.parse, urllib.request
            base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT, "text": msg})
            req = urllib.request.Request(base, data=data.encode("utf-8"), headers={"Content-Type":"application/x-www-form-urlencoded"})
            urllib.request.urlopen(req, timeout=10).read()
            sent = True
        except Exception as e:
            logging.warning(f"Telegram alert failed: {e}")
    if (not sent) and SMTP_HOST and ALERT_EMAIL_TO:
        try:
            import smtplib
            from email.message import EmailMessage
            em = EmailMessage()
            em["Subject"] = "BTC Macro-Tracker gate change"
            em["From"] = SMTP_USER or "bot@example"
            em["To"] = ALERT_EMAIL_TO
            em.set_content(msg)
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
                s.starttls()
                if SMTP_USER and SMTP_PASS: s.login(SMTP_USER, SMTP_PASS)
                s.send_message(em); sent = True
        except Exception as e:
            logging.warning(f"Email alert failed: {e}")
    if not sent:
        logging.info("No alert channel configured; wrote message to log only.")
        logging.info(msg)

def row_hash(d: Dict[str,Any]) -> str:
    keys = ["btc_close","fed_delta","dm_m2_yoy","dxy","real_yld","etf_3d","funding",
            "btc_50d","btc_200d","macro_gate","price_gate","sent_gate"]
    payload = "|".join(str(d.get(k,"")) for k in keys)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

# ------------------------------ Main ----------------------------------------
def run(manual: bool = False):
    skip = os.environ.get("SKIP_FETCH","").strip().lower() in ("1","true","yes","y")
    fetch_all(skip=skip)

    today_utc = datetime.now(tz=RUN_TZ)
    df, diags = build_features(today_utc)
    if df.empty:
        raise SystemExit("No data available to compute features.")

    tgt_date = pd.Timestamp(datetime.now(tz=RUN_TZ).date(), tz=RUN_TZ)
    if PREDICT_NEXT_DAY:
      # Use asof to get the closest date less than or equal to tgt_date
      signal_date = df.index.asof(tgt_date)
    else:
      signal_date = tgt_date

    gates = evaluate_gates(df, signal_date)

    betas = {}; meta  = {"refit_date": None, "refit_n": None, "refit_r2": None}
    if refit_needed(today_utc) or manual:
        refit_day = last_complete_day_for_refit(df, today_utc)
        if refit_day is not None:
            betas, meta = fit_regression(df, refit_day)

    fitted = apply_betas(df, betas)
    fitted_price = float(fitted.iloc[0]) if not fitted.empty else math.nan

    last = df.iloc[-1]
    row = {
        "date": str(last.name.date()),
        "btc_close": round(float(last["btc_close"]), 8) if pd.notna(last["btc_close"]) else "",
        "macro_gate": int(gates["macro_gate"]),
        "price_gate": int(gates["price_gate"]),
        "sent_gate":  int(gates["sent_gate"]),
        "fed_delta":  round(float(last["fed_delta"]), 8) if pd.notna(last["fed_delta"]) else "",
        "dm_m2_yoy":  round(float(last["dm_m2_yoy"]), 8) if pd.notna(last["dm_m2_yoy"]) else "",
        "dxy":        round(float(last["dxy"]), 8) if pd.notna(last["dxy"]) else "",
        "real_yld":   round(float(last["real_yld"]), 8) if pd.notna(last["real_yld"]) else "",
        "etf_3d":     round(float(last["etf_3d"]), 8) if pd.notna(last["etf_3d"]) else "",
        "funding":    round(float(last["funding"]), 8) if pd.notna(last["funding"]) else "",
        "btc_50d":    round(float(last["btc_50d"]), 8) if pd.notna(last["btc_50d"]) else "",
        "btc_200d":   round(float(last["btc_200d"]), 8) if pd.notna(last["btc_200d"]) else "",
        "fitted_price": round(float(fitted_price), 8) if not math.isnan(fitted_price) else "",
        "beta_dm2":   betas.get("dm_m2_yoy", ""),
        "beta_dxy":   betas.get("dxy_delta", ""),
        "beta_real":  betas.get("real_yld_delta", ""),
        "beta_etf":   betas.get("etf_3d", ""),
        "beta_gm2":   betas.get("global_m2_lag", ""),
        "intercept":  betas.get("intercept", ""),
        "refit_date": meta.get("refit_date",""),
        "refit_r2":   meta.get("refit_r2",""),
        "refit_n":    meta.get("refit_n",""),
        "m2_stale_days": diags.get("m2_stale_days",""),
        "etf_3d_stale":  int(diags.get("etf_3d_stale", False)),
        "row_hash": ""
    }
    row["row_hash"] = row_hash(row)

    existed = os.path.exists(TRACK_CSV)
    prev_gates: Dict[str,int] = {}
    if existed:
        try:
            track = pd.read_csv(TRACK_CSV)
            if "date" in track.columns:
                if (track["date"] == row["date"]).any():
                    track = track[track["date"] != row["date"]]
                if len(track) >= 1:
                    last_row = track.iloc[-1].to_dict()
                    for k in ("macro_gate","price_gate","sent_gate"):
                        if k in last_row: prev_gates[k] = int(last_row.get(k))
            else:
                track = pd.DataFrame(columns=list(row.keys()))
        except Exception:
            track = pd.DataFrame(columns=list(row.keys()))
    else:
        track = pd.DataFrame(columns=list(row.keys()))

    track = pd.concat([track, pd.DataFrame([row])], ignore_index=True).reindex(columns=list(row.keys()))
    track.to_csv(TRACK_CSV, index=False)
    logging.info(f"Appended row for {row['date']} → {TRACK_CSV}")

    if prev_gates:
        msg = gate_change_alert(prev_gates, gates, row["date"])
        if msg:
            send_alert(msg)
            with open(os.path.join(os.path.dirname(TRACK_CSV), "gate_change.json"), "w") as f:
                json.dump({"date": row["date"], "prev": prev_gates, "new": gates, "message": msg}, f)

if __name__ == "__main__":
    manual = any(arg.lower() in ("run_now","--run-now","-r") for arg in sys.argv[1:])
    try:
        run(manual=manual)
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        sys.exit(2)
