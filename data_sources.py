from __future__ import annotations

import io
import zipfile
from datetime import date
from typing import Any, Dict, Optional

import pandas as pd
import requests
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


class OpenDARTClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def resolve_company(self, query: str) -> Optional[dict]:
        raw = requests.get(f"{DART_BASE}/corpCode.xml", params={"crtfc_key": self.api_key}, timeout=30)
        raw.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(raw.content))
        xml = zf.read(zf.namelist()[0])
        df = pd.read_xml(io.BytesIO(xml), xpath="/result/list")
        q = query.strip().lower()
        exact = df[df["corp_name"].astype(str).str.lower().eq(q)]
        if exact.empty:
            exact = df[df["corp_name"].astype(str).str.lower().str.contains(q, na=False)]
        if exact.empty:
            return None
        row = exact.iloc[0].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in row.items()}

    def get_company_info(self, corp_code: str) -> dict:
        data = _request_json(f"{DART_BASE}/company.json", {"crtfc_key": self.api_key, "corp_code": corp_code})
        return {k: v for k, v in data.items() if k not in {"status", "message"}}

    def get_financials(self, corp_code: str, year: int) -> pd.DataFrame:
        # Consolidated annual statements first; fallback to separate statements.
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
        rows = data.get("list", [])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def search_filings(self, corp_code: str, bgn_de: date, end_de: date) -> pd.DataFrame:
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
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"})

    def get_snapshot_and_history(self, stock_code: Optional[str]) -> dict:
        if not stock_code:
            raise RuntimeError("DART에서 종목코드를 찾지 못했습니다.")
        code = str(stock_code).zfill(6)
        snapshot_url = f"https://finance.naver.com/item/main.naver?code={code}"
        html = self.session.get(snapshot_url, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")
        def text(css):
            node = soup.select_one(css)
            return node.get_text(" ", strip=True) if node else None
        current = text("#chart_area td.no_up") or text("#chart_area td.no_down") or text("#chart_area .blind")
        history = pd.read_html(f"https://finance.naver.com/item/sise_day.naver?code={code}")[0]
        history = history.dropna(how="all").copy()
        history.columns = [str(c) for c in history.columns]
        rename = {"날짜": "date", "종가": "close", "전일비": "change", "시가": "open", "고가": "high", "저가": "low", "거래량": "volume"}
        history = history.rename(columns=rename)
        for c in ["date", "close", "open", "high", "low", "volume"]:
            if c in history.columns:
                history[c] = pd.to_numeric(history[c].astype(str).str.replace(",", "", regex=False), errors="coerce") if c != "date" else pd.to_datetime(history[c], errors="coerce")
        history = history.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        current_price = float(history.iloc[-1]["close"]) if not history.empty else None
        return {"stock_code": code, "current_price": current_price, "history": history, "snapshot_raw": {"current": current}}


class ECOSClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _statistic(self, stat_code: str, item_code1: str, cycle: str = "MM", start: str = "202001", end: str = "999999") -> pd.DataFrame:
        url = f"{ECOS_BASE}/StatisticSearch/{self.api_key}/json/kr/1/200000/{stat_code}/{cycle}/{start}/{end}/{item_code1}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = data.get("StatisticSearch", {}).get("row", [])
        return pd.DataFrame(rows)

    def get_macro_snapshot(self) -> dict:
        # These are intentionally conservative starter queries. ECOS has many series and codes; exact series can be expanded later.
        result: Dict[str, Any] = {}
        try:
            df = self._statistic("722Y001", "0101000", start="202001", end="999999")
            if not df.empty:
                result["base_rate_like_series"] = df.tail(36).to_dict(orient="records")
        except Exception:
            result["base_rate_like_series"] = []
        return result
