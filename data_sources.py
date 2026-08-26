from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Optional, Dict

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

try:
    import streamlit as st
except Exception:
    st = None

BASE_URL = "https://opendart.fss.or.kr/api"
DART_BASE = BASE_URL
DART_VIEWER = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
REPORT_CODES = {
    "annual": "11011",
    "semiannual": "11012",
    "q1": "11013",
    "q3": "11014",
}

class OpenDartError(RuntimeError):
    pass


@dataclass
class Company:
    corp_code: str
    corp_name: str
    stock_code: Optional[str] = None
    corp_cls: Optional[str] = None
    ceo_nm: Optional[str] = None
    jurir_no: Optional[str] = None
    homepage: Optional[str] = None
    adres: Optional[str] = None


class OpenDartClient:
    def __init__(self, api_key: str, timeout: int = 30):
        if not api_key or len(api_key) < 20:
            raise ValueError("OPENDART_API_KEY가 필요합니다.")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OpenDART-Valuation-Agent/1.0"})
        self._corp_codes: Optional[pd.DataFrame] = None

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {"crtfc_key": self.api_key, **params}
        r = self.session.get(f"{BASE_URL}/{endpoint}.json", params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if str(data.get("status")) != "000":
            raise OpenDartError(f"OpenDART 오류 {data.get('status')}: {data.get('message')}")
        return data

    def download_binary(self, endpoint: str, params: dict[str, Any]) -> bytes:
        params = {"crtfc_key": self.api_key, **params}
        r = self.session.get(f"{BASE_URL}/{endpoint}.xml", params=params, timeout=self.timeout)
        r.raise_for_status()
        # OpenDART binary endpoints return ZIP bytes with success/error encoded in headers/body.
        return r.content

    def corp_codes(self, refresh: bool = False) -> pd.DataFrame:
        if self._corp_codes is not None and not refresh:
            return self._corp_codes
        params = {"crtfc_key": self.api_key}
        r = self.session.get(f"{BASE_URL}/corpCode.xml", params=params, timeout=self.timeout)
        r.raise_for_status()
        raw = r.content
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                xml_name = zf.namelist()[0]
                xml = zf.read(xml_name)
        except zipfile.BadZipFile as e:
            raise OpenDartError("corpCode.xml 응답이 ZIP이 아닙니다. API 키 또는 OpenDART 상태를 확인하세요.") from e
        soup = BeautifulSoup(xml, "xml")
        rows = []
        for item in soup.find_all("list"):
            rows.append({
                "corp_code": item.find("corp_code").text.strip() if item.find("corp_code") else "",
                "corp_name": item.find("corp_name").text.strip() if item.find("corp_name") else "",
                "stock_code": item.find("stock_code").text.strip() if item.find("stock_code") else "",
                "modify_date": item.find("modify_date").text.strip() if item.find("modify_date") else "",
            })
        self._corp_codes = pd.DataFrame(rows)
        return self._corp_codes

    def find_company(self, company_name: str) -> Company:
        df = self.corp_codes()
        exact = df[df["corp_name"].eq(company_name)]
        if exact.empty:
            exact = df[df["corp_name"].str.contains(re.escape(company_name), na=False)]
        if exact.empty:
            raise OpenDartError(f"기업을 찾지 못했습니다: {company_name}")
        if len(exact) > 1:
            # Prefer a listed entity with a stock code.
            listed = exact[exact["stock_code"].ne("")]
            if not listed.empty:
                exact = listed
        row = exact.iloc[0]
        info = self.company(row["corp_code"])
        return Company(
            corp_code=row["corp_code"],
            corp_name=info.get("corp_name") or row["corp_name"],
            stock_code=info.get("stock_code") or row["stock_code"] or None,
            corp_cls=info.get("corp_cls"),
            ceo_nm=info.get("ceo_nm"),
            jurir_no=info.get("jurir_no"),
            homepage=info.get("hm_url"),
            adres=info.get("adres"),
        )

    def company(self, corp_code: str) -> dict[str, Any]:
        return self._get_json("company", {"corp_code": corp_code})

    def disclosures(self, corp_code: str, start: str, end: str, page_count: int = 100) -> list[dict[str, Any]]:
        data = self._get_json("list", {
            "corp_code": corp_code,
            "bgn_de": start,
            "end_de": end,
            "page_no": 1,
            "page_count": min(page_count, 100),
        })
        return data.get("list", [])

    def financials(self, corp_code: str, year: int, reprt_code: str = "11011") -> list[dict[str, Any]]:
        data = self._get_json("fnlttSinglAcntAll", {
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": reprt_code,
            "fs_div": "CFS",
        })
        return data.get("list", [])

    def financial_indicators(self, corp_code: str, year: int, reprt_code: str = "11011") -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for code in ("M210000", "M220000", "M230000", "M240000"):
            try:
                data = self._get_json("fnlttSinglIndx", {
                    "corp_code": corp_code,
                    "bsns_year": str(year),
                    "reprt_code": reprt_code,
                    "idx_cl_code": code,
                })
                out.extend(data.get("list", []))
            except OpenDartError:
                continue
        return out

    def stock_count(self, corp_code: str, year: int, reprt_code: str = "11011") -> list[dict[str, Any]]:
        data = self._get_json("stockTotqySttus", {
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": reprt_code,
        })
        return data.get("list", [])

    # Compatibility wrappers used by the current Feature Engine app.
    def resolve_company(self, query: str) -> Optional[dict[str, Any]]:
        company = self.find_company(query)
        return asdict(company)

    def get_company_info(self, corp_code: str) -> dict[str, Any]:
        return self.company(corp_code)

    def get_financials(self, corp_code: str, year: int) -> pd.DataFrame:
        rows = self.financials(corp_code, year)
        if rows:
            return pd.DataFrame(rows)
        # Match the behavior of the original working app: fall back from CFS to OFS.
        params = {
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": REPORT_CODES["annual"],
            "fs_div": "OFS",
        }
        data = self._get_json("fnlttSinglAcntAll", params)
        return pd.DataFrame(data.get("list", []))

    def search_filings(self, corp_code: str, bgn_de: date, end_de: date) -> pd.DataFrame:
        rows = self.disclosures(corp_code, bgn_de.strftime("%Y%m%d"), end_de.strftime("%Y%m%d"))
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["rcept_no", "report_nm", "rcept_dt", "flr_nm", "corp_name"])

    def get_document_zip(self, rcept_no: str) -> tuple[Optional[str], bytes]:
        r = self.session.get(f"{BASE_URL}/document.xml", params={"crtfc_key": self.api_key, "rcept_no": rcept_no}, timeout=self.timeout)
        r.raise_for_status()
        content = r.content
        try:
            zf = zipfile.ZipFile(io.BytesIO(content))
            return r.headers.get("Content-Type"), content
        except zipfile.BadZipFile as e:
            raise OpenDartError(f"공시원문을 ZIP으로 받지 못했습니다: {rcept_no}") from e



ECOS_BASE = "https://ecos.bok.or.kr/api"

OpenDARTClient = OpenDartClient

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
