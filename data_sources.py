from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

DART_BASE = "https://opendart.fss.or.kr/api"
ECOS_BASE = "https://ecos.bok.or.kr/api"


def _request_json(url: str, params: dict, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("status") not in (None, "000"):
        raise RuntimeError(f"API error {data.get('status')}: {data.get('message')}")
    return data


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def _load_corp_codes(api_key: str) -> pd.DataFrame:
    raw = requests.get(f"{DART_BASE}/corpCode.xml", params={"crtfc_key": api_key}, timeout=30)
    raw.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(raw.content))
    xml_name = next(name for name in zf.namelist() if name.lower().endswith(".xml"))
    df = pd.read_xml(io.BytesIO(zf.read(xml_name)), xpath="/result/list")
    if df is None or df.empty:
        raise RuntimeError("OpenDART corpCode.xml에서 기업 목록을 읽지 못했습니다.")
    for col, width in (("corp_code", 8), ("stock_code", 6)):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(width)
    return df


class OpenDARTClient:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def resolve_company(self, query: str) -> Optional[dict]:
        df = _load_corp_codes(self.api_key)
        q = query.strip().lower()
        names = df["corp_name"].astype(str)
        exact = df[names.str.lower().eq(q)]
        if exact.empty:
            exact = df[names.str.lower().str.contains(q, na=False)]
        if exact.empty:
            return None
        row = exact.iloc[0].to_dict()
        row["corp_code"] = str(row.get("corp_code", "")).zfill(8)
        row["stock_code"] = str(row.get("stock_code", "")).zfill(6)
        return {k: (None if pd.isna(v) else v) for k, v in row.items()}

    def get_company_info(self, corp_code: str) -> dict:
        corp_code = str(corp_code).zfill(8)
        data = _request_json(f"{DART_BASE}/company.json", {"crtfc_key": self.api_key, "corp_code": corp_code})
        return {k: v for k, v in data.items() if k not in {"status", "message"}}

    def get_financials(self, corp_code: str, year: int) -> pd.DataFrame:
        corp_code = str(corp_code).zfill(8)
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": "11011",
            "fs_div": "CFS",
        }
        r = requests.get(f"{DART_BASE}/fnlttSinglAcntAll.json", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "000":
            params["fs_div"] = "OFS"
            r = requests.get(f"{DART_BASE}/fnlttSinglAcntAll.json", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        return pd.DataFrame(data.get("list", []))

    def search_filings(self, corp_code: str, bgn_de: date, end_de: date) -> pd.DataFrame:
        corp_code = str(corp_code).zfill(8)
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bgn_de": bgn_de.strftime("%Y%m%d"),
            "end_de": end_de.strftime("%Y%m%d"),
            "page_no": 1,
            "page_count": 100,
        }
        data = _request_json(f"{DART_BASE}/list.json", params)
        rows = data.get("list", [])
        if not rows:
            return pd.DataFrame(columns=["rcept_no", "report_nm", "rcept_dt", "flr_nm", "corp_name"])
        return pd.DataFrame(rows)


class NaverFinanceClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://finance.naver.com/",
        })

    def _get_fchart_history(self, code: str, count: int = 1250) -> pd.DataFrame:
        r = self.session.get(
            "https://fchart.stock.naver.com/sise.nhn",
            params={"symbol": code, "timeframe": "day", "count": count, "requestType": "0"},
            timeout=30,
        )
        r.raise_for_status()
        r.encoding = "euc-kr"
        root = ET.fromstring(r.text)
        rows = []
        for item in root.findall(".//item"):
            parts = item.attrib.get("data", "").split("|")
            if len(parts) == 6:
                rows.append(parts)
        if not rows:
            raise RuntimeError("Naver chart endpoint returned no price rows.")
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    def _get_current_from_html(self, code: str) -> Optional[float]:
        r = self.session.get(f"https://finance.naver.com/item/main.naver?code={code}", timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "euc-kr"
        soup = BeautifulSoup(r.text, "html.parser")
        for css in ["p.no_today span.blind", "div.today p.no_today span.blind"]:
            node = soup.select_one(css)
            if node:
                try:
                    return float(node.get_text(strip=True).replace(",", ""))
                except ValueError:
                    pass
        return None

    def get_snapshot_and_history(self, stock_code: Optional[str]) -> dict:
        if not stock_code:
            raise RuntimeError("DART에서 종목코드를 찾지 못했습니다.")
        code = str(stock_code).zfill(6)
        history = self._get_fchart_history(code)
        current_price = float(history.iloc[-1]["close"])
        try:
            current = self._get_current_from_html(code)
            if current is not None:
                current_price = current
        except Exception:
            pass
        return {"stock_code": code, "current_price": current_price, "history": history, "market_status": "OK"}


class ECOSClient:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("ECOS API key is empty.")

    def _statistic(
        self,
        stat_code: str,
        cycle: str,
        start: str,
        end: str,
        item_code1: Optional[str] = None,
        item_code2: Optional[str] = None,
        timeout: int = 30,
    ) -> pd.DataFrame:
        # ECOS StatisticSearch path: .../{stat}/{cycle}/{start}/{end}/{item1}/{item2}/...
        parts = [ECOS_BASE, "StatisticSearch", self.api_key, "json", "kr", "1", "10000", stat_code, cycle, start, end]
        if item_code1:
            parts.append(item_code1)
        if item_code2:
            parts.append(item_code2)
        url = "/".join(parts)
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if "RESULT" in data:
            result = data["RESULT"]
            raise RuntimeError(f"ECOS API error {result.get('CODE')}: {result.get('MESSAGE')}")
        rows = data.get("StatisticSearch", {}).get("row", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "TIME" in df.columns:
            df["TIME"] = df["TIME"].astype(str)
        if "DATA_VALUE" in df.columns:
            df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
        return df

    @staticmethod
    def _monthly_period(offset_months: int = 0) -> str:
        now = pd.Timestamp.today().to_period("M") - offset_months
        return now.strftime("%Y%m")

    def get_macro_snapshot(self) -> dict:
        end_m = self._monthly_period(0)
        start_m = self._monthly_period(72)
        end_d = pd.Timestamp.today().strftime("%Y%m%d")
        start_d = (pd.Timestamp.today() - pd.Timedelta(days=365 * 6)).strftime("%Y%m%d")
        result: Dict[str, Any] = {"status": "OK", "series": {}, "errors": []}

        specs = {
            "base_rate": ("722Y001", "M", start_m, end_m, "0101000", None, "기준금리", "%"),
            "ktb_3y": ("817Y002", "D", start_d, end_d, "010200000", None, "국고채 3년", "%"),
            "ktb_10y": ("817Y002", "D", start_d, end_d, "010210000", None, "국고채 10년", "%"),
            "usdkrw": ("731Y004", "M", start_m, end_m, "0000001", "0000100", "원/달러", "KRW/USD"),
            "cpi": ("901Y009", "M", start_m, end_m, "0", None, "소비자물가지수", "index"),
            "gdp_real": ("200Y102", "Q", "2020Q1", "2026Q4", "10111", None, "실질 GDP", "level"),
        }

        for key, (stat, cycle, start, end, item1, item2, label, unit) in specs.items():
            try:
                df = self._statistic(stat, cycle, start, end, item1, item2)
                if not df.empty:
                    result["series"][key] = {
                        "label": label,
                        "unit": unit,
                        "stat_code": stat,
                        "item_code1": item1,
                        "item_code2": item2,
                        "data": df.to_dict(orient="records"),
                    }
                else:
                    result["series"][key] = {"label": label, "unit": unit, "data": []}
                    result["errors"].append(f"{key}: no data")
            except Exception as exc:
                result["series"][key] = {"label": label, "unit": unit, "data": []}
                result["errors"].append(f"{key}: {exc}")

        ok_count = sum(bool(v.get("data")) for v in result["series"].values())
        if ok_count == len(specs):
            result["status"] = "OK"
        elif ok_count > 0:
            result["status"] = "PARTIAL"
        else:
            result["status"] = "ERROR"
        return result
