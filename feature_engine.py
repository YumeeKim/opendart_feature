from __future__ import annotations

import math
from typing import Any, Dict

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
            mask = df[col].astype(str).str.contains(cand, case=False, na=False)
            sub = df[mask]
            if not sub.empty:
                for value_col in ["thstrm_amount", "thstrm_add_amount", "frmtrm_amount", "frmtrm_q_amount"]:
                    if value_col in sub.columns:
                        return _num(sub.iloc[0][value_col])
    return None


def _annual_return(history: pd.DataFrame, months: int):
    if history is None or history.empty or "close" not in history.columns:
        return None
    h = history.dropna(subset=["date", "close"]).copy()
    if h.empty:
        return None
    last = h.iloc[-1]
    target = last["date"] - pd.DateOffset(months=months)
    old = h.iloc[(h["date"] - target).abs().argmin()]
    return (float(last["close"]) / float(old["close"]) - 1) * 100


def build_feature_table(company_info: dict, financials: pd.DataFrame, market: dict | None, macro: dict | None, notices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
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
    receivable = _pick_account(f, ["매출채권", "AccountsReceivable"])
    inventory = _pick_account(f, ["재고자산", "Inventories"])
    cfo = _pick_account(f, ["영업활동현금흐름", "영업활동으로 인한 현금흐름", "NetCashProvidedByUsedInOperatingActivities"])
    capex = _pick_account(f, ["유형자산의 취득", "유형자산취득", "PurchaseOfPropertyPlantAndEquipment"])
    interest = _pick_account(f, ["이자비용", "금융원가", "InterestExpense"])
    tax = _pick_account(f, ["법인세비용", "법인세비용(수익)", "IncomeTaxExpense"])

    current_price = (market or {}).get("current_price")
    hist = (market or {}).get("history")

    # Market features
    ret_1m = _annual_return(hist, 1)
    ret_3m = _annual_return(hist, 3)
    ret_6m = _annual_return(hist, 6)
    ret_12m = _annual_return(hist, 12)

    vol = None
    dd = None
    if hist is not None and not hist.empty and "close" in hist.columns:
        h = hist.dropna(subset=["close"]).copy()
        daily = h["close"].pct_change().dropna()
        if len(daily) >= 20:
            vol = daily.std() * np.sqrt(252) * 100
        peak = h["close"].cummax()
        dd = ((h["close"] / peak) - 1).min() * 100 if not h.empty else None

    # Basic ratios. These are deliberately transparent first-version formulas.
    tax_rate = None
    if op_income and tax is not None and op_income != 0:
        tax_rate = max(0.0, min(0.35, tax / op_income))
    nopat = op_income * (1 - tax_rate) if op_income is not None and tax_rate is not None else None
    invested_capital = None
    if equity is not None:
        invested_capital = equity + (debt or 0) - (cash or 0)
    roic = (nopat / invested_capital * 100) if nopat is not None and invested_capital not in (None, 0) else None
    roe = (net_income / equity * 100) if net_income is not None and equity not in (None, 0) else None
    roa = (net_income / assets * 100) if net_income is not None and assets not in (None, 0) else None
    op_margin = (op_income / revenue * 100) if op_income is not None and revenue not in (None, 0) else None
    current_ratio = (current_assets / current_liab) if current_assets is not None and current_liab not in (None, 0) else None
    debt_equity = (debt / equity * 100) if debt is not None and equity not in (None, 0) else None
    interest_cov = (op_income / interest) if op_income is not None and interest not in (None, 0) else None
    fcf = (cfo + capex) if cfo is not None and capex is not None else None
    fcf_yield = None  # Requires market cap; intentionally left uncalculated until market-cap mapping is added.
    cfo_ni = (cfo / net_income) if cfo is not None and net_income not in (None, 0) else None

    def add(cat, feature, value, unit="", flag="OK", source="OpenDART"):
        rows.append({"category": cat, "feature": feature, "value": value, "unit": unit, "quality_flag": flag, "source": source})

    add("Value", "Current Price", current_price, "KRW", source="Naver Finance")
    add("Value", "PER", None, "x", "PENDING", "Not enough market-cap/EPS data yet")
    add("Value", "PBR", None if current_price is None else None, "x", "PENDING", "Market cap/book value mapping will be added next")
    add("Value", "EV/EBITDA", None, "x", "PENDING", "EBITDA mapping will be added next")
    add("Value", "FCF Yield", fcf_yield, "%", source="OpenDART + Naver Finance")

    add("Quality", "ROIC", roic, "%")
    add("Quality", "ROE", roe, "%")
    add("Quality", "ROA", roa, "%")
    add("Quality", "Operating Margin", op_margin, "%")
    add("Quality", "Current Ratio", current_ratio, "x")
    add("Quality", "Debt / Equity", debt_equity, "%")
    add("Quality", "Interest Coverage", interest_cov, "x")
    add("Quality", "CFO / Net Income", cfo_ni, "x")
    add("Quality", "FCF", fcf, "KRW million")

    add("Momentum", "1M Return", ret_1m, "%", source="Naver Finance")
    add("Momentum", "3M Return", ret_3m, "%", source="Naver Finance")
    add("Momentum", "6M Return", ret_6m, "%", source="Naver Finance")
    add("Momentum", "12M Return", ret_12m, "%", source="Naver Finance")
    add("Momentum", "Annualized Volatility", vol, "%", source="Naver Finance")
    add("Momentum", "Max Drawdown", dd, "%", source="Naver Finance")

    add("Risk / Accounting", "Receivable Growth vs Sales Growth", None, "pp", "PENDING", "Requires multi-year statement panel mapping")
    add("Risk / Accounting", "Inventory Growth vs Sales Growth", None, "pp", "PENDING", "Requires multi-year statement panel mapping")
    add("Risk / Accounting", "CFO - Net Income", (cfo - net_income) if cfo is not None and net_income is not None else None, "KRW million")
    add("Risk / Accounting", "Debt", debt, "KRW million")
    add("Risk / Accounting", "Recent High-Risk Filings", int(_count_high_risk_notices(notices)), "count", source="OpenDART")

    macro_rate = None
    if macro and macro.get("base_rate_like_series"):
        try:
            last = macro["base_rate_like_series"][-1]
            macro_rate = last.get("DATA_VALUE")
        except Exception:
            pass
    add("Macro", "Policy/Rate Series", macro_rate, "", source="ECOS")
    add("Industry", "Industry / Sector", company_info.get("induty_code") or company_info.get("industry_code"), "", source="OpenDART/company info")
    add("Company", "Fiscal Year", company_info.get("acc_mt"), "", source="OpenDART")
    add("Company", "Employees", company_info.get("emp_stdn_nb"), "people", source="OpenDART")

    return pd.DataFrame(rows)


def _count_high_risk_notices(notices: pd.DataFrame) -> int:
    if notices is None or notices.empty or "report_nm" not in notices.columns:
        return 0
    keywords = ["유상증자", "전환사채", "횡령", "배임", "소송", "감사의견", "관리종목", "상장폐지"]
    s = notices["report_nm"].astype(str)
    return int(s.str.contains("|".join(keywords), case=False, na=False).sum())
