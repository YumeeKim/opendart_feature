from __future__ import annotations

import json
import os
import traceback
from datetime import date

import pandas as pd
import streamlit as st

from ai_agent import InvestmentAI
from data_sources import OpenDARTClient, NaverFinanceClient, ECOSClient
from feature_engine import build_feature_table

SECRET_ALIASES = {
    "OPENDART_API_KEY": ["OPENDART_API_KEY", "DART_API_KEY", "OPEN_DART_API_KEY"],
    "ECOS_API_KEY": ["ECOS_API_KEY", "BOK_ECOS_API_KEY"],
    "OPENAI_API_KEY": ["OPENAI_API_KEY", "OPENAI_KEY", "GPT_API_KEY"],
    "OPENAI_MODEL": ["OPENAI_MODEL"],
}
SECTION_NAMES = ("default", "api_keys", "secrets", "keys")


def _text(value):
    return "" if value is None else str(value).strip()


def load_settings():
    values = {}
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}
    for canonical, aliases in SECRET_ALIASES.items():
        value = None
        for name in aliases:
            try:
                if name in secrets and _text(secrets[name]):
                    value = _text(secrets[name])
                    break
            except Exception:
                pass
        if not value:
            for section_name in SECTION_NAMES:
                try:
                    if section_name not in secrets:
                        continue
                    section = secrets[section_name]
                    for name in aliases:
                        if name in section and _text(section[name]):
                            value = _text(section[name])
                            break
                    if value:
                        break
                except Exception:
                    pass
        if not value:
            for name in aliases:
                value = _text(os.getenv(name))
                if value:
                    break
        if value:
            values[canonical] = value
    return values


def secret_status(settings):
    return {k: bool(settings.get(k)) for k in SECRET_ALIASES}


def feature_dict(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    out = {}
    for row in df.to_dict(orient="records"):
        key = row.get("feature")
        if not key:
            continue
        out[key] = {
            "value": row.get("value"),
            "unit": row.get("unit"),
            "quality_flag": row.get("quality_flag"),
            "source": row.get("source"),
            "category": row.get("category"),
        }
    return out


def compact_df(df: pd.DataFrame, columns: list[str], n: int = 25):
    if df is None or df.empty:
        return []
    use = [c for c in columns if c in df.columns]
    return df[use].head(n).to_dict(orient="records")


st.set_page_config(page_title="기업 Feature + AI Analyst", page_icon="📈", layout="wide")
st.title("📈 기업 Feature Engine + AI Analyst")
st.caption("OpenDART + Naver Finance + ECOS → 정량 Feature → GPT 해석")

settings = load_settings()
with st.sidebar:
    st.header("API / 분석 설정")
    dart_key = st.text_input("OpenDART API Key", value=settings.get("OPENDART_API_KEY", ""), type="password")
    ecos_key = st.text_input("ECOS API Key", value=settings.get("ECOS_API_KEY", ""), type="password")
    openai_key = settings.get("OPENAI_API_KEY", "")
    openai_model = settings.get("OPENAI_MODEL", "gpt-5.6-luna")
    st.write(f"OpenDART: {'감지됨' if dart_key else '없음'}")
    st.write(f"ECOS: {'감지됨' if ecos_key else '없음'}")
    st.write(f"OpenAI/GPT: {'감지됨' if openai_key else '없음'}")
    st.caption(f"AI 모델: {openai_model}")
    st.caption("현재 단계의 AI는 계산 결과와 공시 목록을 해석합니다. 숫자 계산은 Python이 담당합니다.")

company = st.text_input("분석할 기업명", placeholder="예: 삼성전자")
year = st.number_input("재무 기준연도", min_value=2015, max_value=date.today().year, value=max(2015, date.today().year - 1), step=1)
run = st.button("🚀 데이터 수집 + AI 분석", type="primary", use_container_width=True)

if run:
    if not company.strip():
        st.error("기업명을 입력하세요.")
        st.stop()
    if not dart_key:
        st.error("OpenDART API Key가 필요합니다.")
        st.stop()
    if not openai_key:
        st.error("OPENAI_API_KEY를 Streamlit Cloud Secrets에 등록하세요.")
        st.stop()

    try:
        with st.status("데이터 수집 중...", expanded=True) as status:
            dart = OpenDARTClient(dart_key)
            corp = dart.resolve_company(company.strip())
            if not corp:
                st.error("OpenDART에서 기업을 찾지 못했습니다.")
                st.stop()
            status.write("✓ 기업코드 확인")
            financials = dart.get_financials(corp["corp_code"], year=year)
            notices = dart.search_filings(corp["corp_code"], bgn_de=date(year, 1, 1), end_de=date.today())
            company_info = dart.get_company_info(corp["corp_code"])
            status.write("✓ DART 재무/공시 수집")

            market = None
            market_msg = ""
            try:
                market = NaverFinanceClient().get_snapshot_and_history(corp.get("stock_code"))
                status.write("✓ 시장가격/거래량 수집")
            except Exception as e:
                market_msg = f"네이버 증권 데이터 수집 실패: {e}"
                status.write("⚠ 시장데이터 일부 실패")

            macro = None
            macro_msg = ""
            if ecos_key:
                try:
                    macro = ECOSClient(ecos_key).get_macro_snapshot()
                    status.write("✓ ECOS 거시데이터 수집")
                except Exception as e:
                    macro_msg = f"ECOS 데이터 수집 실패: {e}"
                    status.write("⚠ ECOS 일부 실패")
            else:
                macro_msg = "ECOS API Key가 없어 거시 Feature가 비어 있습니다."

            features = build_feature_table(company_info, financials, market, macro, notices, pd.Timestamp.today().normalize())
            status.write("✓ Feature Engine 계산")

            dossier = {
                "company": {
                    "corp_name": company_info.get("corp_name", company.strip()),
                    "stock_code": corp.get("stock_code"),
                    "corp_code": corp.get("corp_code"),
                    "corp_cls": company_info.get("corp_cls"),
                    "sector_code": company_info.get("induty_code") or company_info.get("industry_code"),
                },
                "features": feature_dict(features),
                "recent_filings": compact_df(notices, ["rcept_dt", "report_nm", "flr_nm", "corp_name"], n=30),
                "market_snapshot": {
                    "current_price": (market or {}).get("current_price"),
                    "market_status": (market or {}).get("market_status"),
                },
                "macro_summary": {
                    "status": (macro or {}).get("status"),
                    "policy_rate": feature_dict(features).get("Policy Rate"),
                    "ktb_3y": feature_dict(features).get("KTB 3Y"),
                    "ktb_10y": feature_dict(features).get("KTB 10Y"),
                    "usd_krw": feature_dict(features).get("USD/KRW"),
                    "cpi": feature_dict(features).get("CPI Index"),
                    "real_gdp_yoy": feature_dict(features).get("Real GDP YoY"),
                },
            }

            status.write("🤖 GPT 투자 리서치 해석 중...")
            ai = InvestmentAI(openai_key, model=openai_model).analyze(dossier)
            status.update(label="분석 완료", state="complete")

        if market_msg:
            st.warning(market_msg)
        if macro_msg:
            st.warning(macro_msg)

        st.success(f"{company_info.get('corp_name', company)} 분석 완료")
        top = st.columns(6)
        top[0].metric("AI 판단", ai.get("decision", "-"))
        top[1].metric("현재가", f"{market.get('current_price'):,.0f}원" if market and market.get('current_price') else "-")
        fd = feature_dict(features)
        top[2].metric("ROIC", f"{fd.get('ROIC', {}).get('value')}%" if fd.get('ROIC', {}).get('value') is not None else "-")
        top[3].metric("ROE", f"{fd.get('ROE', {}).get('value')}%" if fd.get('ROE', {}).get('value') is not None else "-")
        top[4].metric("12M 수익률", f"{fd.get('12M Return', {}).get('value')}%" if fd.get('12M Return', {}).get('value') is not None else "-")
        top[5].metric("거시상태", (macro or {}).get("status", "-"))

        t1, t2, t3, t4 = st.tabs(["🤖 AI 판단", "📊 Feature", "📄 공시/시장", "🧩 원본/JSON"])
        with t1:
            st.subheader("한 줄 결론")
            st.info(ai.get("executive_summary", ""))
            st.markdown("### 판단 근거")
            for x in ai.get("decision_reasons", []): st.write(f"• {x}")
            for title, key in [
                ("Quality", "quality_assessment"),
                ("Valuation", "valuation_assessment"),
                ("Momentum", "momentum_assessment"),
                ("Macro", "macro_assessment"),
            ]:
                st.markdown(f"### {title}")
                for x in ai.get(key, []): st.write(f"• {x}")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### 🟢 매수 조건")
                for x in ai.get("buy_conditions", []): st.write(f"• {x}")
            with c2:
                st.markdown("### 🟡 보류 조건")
                for x in ai.get("hold_conditions", []): st.write(f"• {x}")
            with c3:
                st.markdown("### 🔴 매도 조건")
                for x in ai.get("sell_conditions", []): st.write(f"• {x}")

            st.markdown("### 산업별 숨은 신호")
            for s in ai.get("industry_hidden_signals", []):
                icon = "🟢" if s.get("impact") == "positive" else "🔴" if s.get("impact") == "negative" else "🟡"
                st.write(f"{icon} **{s.get('signal')}** — {s.get('evidence')} · 신뢰도 {s.get('confidence')}")

            st.markdown("### 회계 이상징후 해석")
            for s in ai.get("accounting_risk_observations", []):
                icon = "🔴" if s.get("severity") == "high" else "🟠" if s.get("severity") == "medium" else "🟡"
                st.write(f"{icon} **{s.get('flag')}** — {s.get('evidence')} · {s.get('why_it_matters')}")

            missing = ai.get("missing_data", [])
            if missing:
                st.markdown("### 현재 부족한 데이터")
                for x in missing: st.write(f"• {x}")
            st.caption(f"AI 판단 신뢰도: {ai.get('confidence', '-')}")

        with t2:
            st.subheader("Feature Engine")
            st.dataframe(features[["category", "feature", "value", "unit", "quality_flag", "source"]], use_container_width=True, hide_index=True)

            with st.expander("거시 시계열"):
                if macro:
                    for key in ["base_rate", "ktb_3y", "ktb_10y", "usdkrw", "cpi", "gdp_real"]:
                        series = macro.get("series", {}).get(key, {})
                        rows = series.get("data", [])
                        if not rows:
                            continue
                        df = pd.DataFrame(rows)
                        if "TIME" not in df.columns or "DATA_VALUE" not in df.columns:
                            continue
                        df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
                        df = df.dropna(subset=["DATA_VALUE", "TIME"]).copy()
                        time_text = df["TIME"].astype(str).str.strip()
                        if time_text.str.fullmatch(r"\d{8}").all():
                            df["DATE"] = pd.to_datetime(time_text, format="%Y%m%d", errors="coerce")
                        elif time_text.str.fullmatch(r"\d{6}").all():
                            df["DATE"] = pd.to_datetime(time_text, format="%Y%m", errors="coerce")
                        elif time_text.str.fullmatch(r"\d{4}Q[1-4]").all():
                            q = time_text.str.extract(r"(\d{4})Q([1-4])")
                            df["DATE"] = pd.to_datetime(q[0] + "-" + (((q[1].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2)) + "-01", errors="coerce")
                        else:
                            df["DATE"] = pd.to_datetime(time_text, errors="coerce")
                        df = df.dropna(subset=["DATE"]).sort_values("DATE")
                        st.write(series.get("label", key))
                        st.line_chart(df.set_index("DATE")["DATA_VALUE"], height=220)

        with t3:
            st.subheader("최근 공시")
            if notices is not None and not notices.empty:
                cols = [c for c in ["rcept_dt", "report_nm", "flr_nm", "corp_name"] if c in notices.columns]
                st.dataframe(notices[cols].head(50), use_container_width=True, hide_index=True)
            else:
                st.info("최근 공시가 없습니다.")
            st.subheader("시장 데이터")
            if market and "history" in market:
                st.dataframe(market["history"].tail(120), use_container_width=True, hide_index=True)
            else:
                st.info("시장 데이터가 없습니다.")

        with t4:
            st.json({"ai": ai, "features": feature_dict(features), "company": company_info, "market": market, "macro": macro})
            result = {"ai": ai, "features": feature_dict(features), "company": company_info, "market": market, "macro": macro}
            st.download_button("⬇️ 결과 JSON 다운로드", data=json.dumps(result, ensure_ascii=False, indent=2, default=str), file_name="ai_feature_result.json", mime="application/json")

    except Exception as e:
        st.error("분석 중 오류가 발생했습니다.")
        st.code(str(e))
        with st.expander("개발용 오류 로그"):
            st.code(traceback.format_exc())
