from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _num(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        return None


def _pick_account(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None
    name_cols = [c for c in ["account_nm", "account_detail", "account_id"] if c in df.columns]
    for cand in candidates:
        for col in name_cols:
            sub = df[df[col].astype(str).str.contains(cand, case=False, na=False)]
            if not sub.empty:
                for value_col in ["thstrm_amount", "thstrm_add_amount", "frmtrm_amount", "frmtrm_q_amount"]:
                    if value_col in sub.columns:
                        return _num(sub.iloc[0][value_col])
    return None


def _return_months(history, months):
    if history is None or history.empty or "close" not in history.columns:
        return None
    h = history.dropna(subset=["date", "close"]).copy()
    last = h.iloc[-1]
    target = last["date"] - pd.DateOffset(months=months)
    old = h.iloc[(h["date"] - target).abs().argmin()]
    return (float(last["close"]) / float(old["close"]) - 1) * 100


def _series_last(macro, key):
    s = (macro or {}).get("series", {}).get(key, {})
    rows = s.get("data", [])
    if not rows:
        return None, None
    row = rows[-1]
    return _num(row.get("DATA_VALUE")), row.get("TIME")


def _series_df(macro, key):
    s = (macro or {}).get("series", {}).get(key, {})
    rows = s.get("data", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _yoy(df):
    if df.empty or "DATA_VALUE" not in df.columns or "TIME" not in df.columns:
        return None
    d = df[["TIME", "DATA_VALUE"]].copy()
    d["DATA_VALUE"] = pd.to_numeric(d["DATA_VALUE"], errors="coerce")
    d = d.dropna().sort_values("TIME")
    if len(d) < 5:
        return None
    try:
        latest = float(d.iloc[-1]["DATA_VALUE"])
        prev = float(d.iloc[-5]["DATA_VALUE"])
        return (latest / prev - 1) * 100 if prev else None
    except Exception:
        return None


def build_feature_table(company_info, financials, market, macro, notices, asof):
    rows = []
    f = financials.copy() if financials is not None else pd.DataFrame()

    revenue = _pick_account(f, ["매출액", "수익(매출액)", "Revenue"])
    op_income = _pick_account(f, ["영업이익", "영업이익(손실)", "OperatingIncome"])
    net_income = _pick_account(f, ["당기순이익", "당기순이익(손실)", "NetIncome"])
    equity = _pick_account(f, ["자본총계", "TotalEquity"])
    assets = _pick_account(f, ["자산총계", "TotalAssets"])
    current_assets = _pick_account(f, ["유동자산", "CurrentAssets"])
    current_liab = _pick_account(f, ["유동부채", "CurrentLiabilities"])
    cash = _pick_account(f, ["현금및현금성자산", "현금및현금성 자산", "CashAndCashEquivalents"])
    debt = _pick_account(f, ["차입금", "단기차입금", "장기차입금", "Borrowings"])
    cfo = _pick_account(f, ["영업활동현금흐름", "영업활동으로 인한 현금흐름", "NetCashProvidedByUsedInOperatingActivities"])
    capex = _pick_account(f, ["유형자산의 취득", "유형자산취득", "PurchaseOfPropertyPlantAndEquipment"])
    interest = _pick_account(f, ["이자비용", "금융원가", "InterestExpense"])
    tax = _pick_account(f, ["법인세비용", "법인세비용(수익)", "IncomeTaxExpense"])

    current_price = (market or {}).get("current_price")
    hist = (market or {}).get("history")

    vol = dd = None
    if hist is not None and not hist.empty and "close" in hist.columns:
        h = hist.dropna(subset=["close"]).copy()
        daily = h["close"].pct_change().dropna()
        if len(daily) >= 20:
            vol = daily.std() * np.sqrt(252) * 100
        peak = h["close"].cummax()
        dd = ((h["close"] / peak) - 1).min() * 100

    tax_rate = max(0.0, min(0.35, tax / op_income)) if op_income not in (None, 0) and tax is not None else None
    nopat = op_income * (1 - tax_rate) if op_income is not None and tax_rate is not None else None
    invested_capital = equity + (debt or 0) - (cash or 0) if equity is not None else None
    roic = nopat / invested_capital * 100 if nopat is not None and invested_capital not in (None, 0) else None
    roe = net_income / equity * 100 if net_income is not None and equity not in (None, 0) else None
    roa = net_income / assets * 100 if net_income is not None and assets not in (None, 0) else None
    op_margin = op_income / revenue * 100 if op_income is not None and revenue not in (None, 0) else None
    current_ratio = current_assets / current_liab if current_assets is not None and current_liab not in (None, 0) else None
    debt_equity = debt / equity * 100 if debt is not None and equity not in (None, 0) else None
    interest_cov = op_income / interest if op_income is not None and interest not in (None, 0) else None
    fcf = cfo + capex if cfo is not None and capex is not None else None
    cfo_ni = cfo / net_income if cfo is not None and net_income not in (None, 0) else None

    def add(cat, feature, value, unit="", flag="OK", source="OpenDART"):
        rows.append({"category": cat, "feature": feature, "value": value, "unit": unit, "quality_flag": flag, "source": source})

    add("Value", "Current Price", current_price, "KRW", source="Naver Finance")
    add("Value", "PER", None, "x", "PENDING", "Market cap/EPS mapping")
    add("Value", "PBR", None, "x", "PENDING", "Market cap/book mapping")
    add("Value", "EV/EBITDA", None, "x", "PENDING", "EV/EBITDA mapping")
    add("Value", "FCF Yield", None, "%", "PENDING", "Needs market cap")

    for name, value, unit in [
        ("ROIC", roic, "%"), ("ROE", roe, "%"), ("ROA", roa, "%"),
        ("Operating Margin", op_margin, "%"), ("Current Ratio", current_ratio, "x"),
        ("Debt / Equity", debt_equity, "%"), ("Interest Coverage", interest_cov, "x"),
        ("CFO / Net Income", cfo_ni, "x"), ("FCF", fcf, "KRW million"),
    ]:
        add("Quality", name, value, unit)

    for months, label in [(1, "1M Return"), (3, "3M Return"), (6, "6M Return"), (12, "12M Return")]:
        add("Momentum", label, _return_months(hist, months), "%", source="Naver Finance")
    add("Momentum", "Annualized Volatility", vol, "%", source="Naver Finance")
    add("Momentum", "Max Drawdown", dd, "%", source="Naver Finance")

    latest_volume = avg_volume_20d = None
    if hist is not None and not hist.empty and "volume" in hist.columns:
        v = pd.to_numeric(hist["volume"], errors="coerce").dropna()
        if not v.empty:
            latest_volume = float(v.iloc[-1])
            avg_volume_20d = float(v.tail(20).mean()) if len(v) >= 20 else None
    add("Momentum", "Latest Volume", latest_volume, "shares", source="Naver Finance")
    add("Momentum", "20D Average Volume", avg_volume_20d, "shares", source="Naver Finance")

    add("Risk / Accounting", "Receivable Growth vs Sales Growth", None, "pp", "PENDING", "Needs multi-year statements")
    add("Risk / Accounting", "Inventory Growth vs Sales Growth", None, "pp", "PENDING", "Needs multi-year statements")
    add("Risk / Accounting", "CFO - Net Income", cfo - net_income if cfo is not None and net_income is not None else None, "KRW million")
    add("Risk / Accounting", "Debt", debt, "KRW million")
    add("Risk / Accounting", "Recent High-Risk Filings", _count_high_risk_notices(notices), "count", source="OpenDART")

    # Macro features
    base, base_t = _series_last(macro, "base_rate")
    ktb3, ktb3_t = _series_last(macro, "ktb_3y")
    ktb10, ktb10_t = _series_last(macro, "ktb_10y")
    usd, usd_t = _series_last(macro, "usdkrw")
    cpi, cpi_t = _series_last(macro, "cpi")
    gdp, gdp_t = _series_last(macro, "gdp_real")

    spread = ktb10 - ktb3 if ktb10 is not None and ktb3 is not None else None
    cpi_yoy = _yoy(_series_df(macro, "cpi"))
    gdp_yoy = _yoy(_series_df(macro, "gdp_real"))

    macro_items = [
        ("Policy Rate", base, "%", base_t),
        ("KTB 3Y", ktb3, "%", ktb3_t),
        ("KTB 10Y", ktb10, "%", ktb10_t),
        ("10Y - 3Y Spread", spread, "%p", ktb10_t),
        ("USD/KRW", usd, "KRW/USD", usd_t),
        ("CPI Index", cpi, "index", cpi_t),
        ("CPI YoY", cpi_yoy, "%", cpi_t),
        ("Real GDP", gdp, "level", gdp_t),
        ("Real GDP YoY", gdp_yoy, "%", gdp_t),
    ]
    for feature, value, unit, obs in macro_items:
        add("Macro", feature, value, unit, "OK" if value is not None else "PENDING", "ECOS")
    add("Macro", "Base Rate Observation", base_t, "", source="ECOS")

    add("Industry", "Industry / Sector Code", company_info.get("induty_code") or company_info.get("industry_code"), "", source="OpenDART/company info")
    add("Company", "Fiscal Year", company_info.get("acc_mt"), "", source="OpenDART")
    add("Company", "Employees", company_info.get("emp_stdn_nb"), "people", source="OpenDART")

    return pd.DataFrame(rows)


def _count_high_risk_notices(notices):
    if notices is None or notices.empty or "report_nm" not in notices.columns:
        return 0
    keywords = ["유상증자", "전환사채", "횡령", "배임", "소송", "감사의견", "관리종목", "상장폐지"]
    return int(notices["report_nm"].astype(str).str.contains("|".join(keywords), case=False, na=False).sum())
