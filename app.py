import os
import traceback
from datetime import date

import pandas as pd
import streamlit as st

from data_sources import OpenDARTClient, NaverFinanceClient, ECOSClient
from feature_engine import build_feature_table

SECRET_ALIASES = {
    "OPENDART_API_KEY": ["OPENDART_API_KEY", "DART_API_KEY", "OPEN_DART_API_KEY"],
    "ECOS_API_KEY": ["ECOS_API_KEY", "BOK_ECOS_API_KEY"],
    "OPENAI_API_KEY": ["OPENAI_API_KEY", "OPENAI_KEY", "GPT_API_KEY"],
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


st.set_page_config(page_title="OpenDART Feature Engine", page_icon="📊", layout="wide")
st.title("📊 기업 Feature Engine")
st.caption("OpenDART + Naver Finance + 한국은행 ECOS | GPT는 아직 사용하지 않습니다.")
settings = load_settings()

with st.sidebar:
    st.header("데이터 설정")
    dart_key = st.text_input("OpenDART API Key", value=settings.get("OPENDART_API_KEY", ""), type="password")
    ecos_key = st.text_input("ECOS API Key", value=settings.get("ECOS_API_KEY", ""), type="password")
    status = secret_status(settings)
    st.divider()
    st.write("**Secrets 상태**")
    st.write(f"OpenDART: {'감지됨' if status['OPENDART_API_KEY'] else '없음'}")
    st.write(f"ECOS: {'감지됨' if status['ECOS_API_KEY'] else '없음'}")
    st.write(f"OpenAI/GPT: {'감지됨' if status['OPENAI_API_KEY'] else '없음'}")
    st.caption("API Key는 화면에 표시하지 않습니다.")

company = st.text_input("분석할 기업명", placeholder="예: 삼성전자")
year = st.number_input("재무 기준연도", min_value=2015, max_value=date.today().year, value=max(2015, date.today().year - 1), step=1)
run = st.button("🔍 데이터 수집 및 Feature 계산", type="primary", use_container_width=True)

if run:
    if not company.strip():
        st.error("기업명을 입력하세요.")
        st.stop()
    if not dart_key:
        st.error("OpenDART API Key가 필요합니다.")
        st.stop()
    try:
        with st.spinner("DART 재무/공시 데이터를 수집하는 중..."):
            dart = OpenDARTClient(dart_key)
            corp = dart.resolve_company(company.strip())
            if not corp:
                st.error("OpenDART에서 기업을 찾지 못했습니다.")
                st.stop()
            financials = dart.get_financials(corp["corp_code"], year=year)
            notices = dart.search_filings(corp["corp_code"], bgn_de=date(year, 1, 1), end_de=date.today())
            company_info = dart.get_company_info(corp["corp_code"])

        market, macro = None, None
        market_msg, macro_msg = "", ""
        try:
            market = NaverFinanceClient().get_snapshot_and_history(corp.get("stock_code"))
        except Exception as e:
            market_msg = f"네이버 증권 데이터 수집 실패: {e}"

        if ecos_key:
            try:
                with st.spinner("ECOS 거시데이터를 수집하는 중..."):
                    macro = ECOSClient(ecos_key).get_macro_snapshot()
            except Exception as e:
                macro_msg = f"ECOS 데이터 수집 실패: {e}"
        else:
            macro_msg = "ECOS API Key가 없어 거시 Feature가 비어 있습니다."

        features = build_feature_table(company_info, financials, market, macro, notices, pd.Timestamp.today().normalize())
        st.success(f"{company_info.get('corp_name', company)} 데이터 준비 완료")
        if market_msg: st.warning(market_msg)
        if macro_msg: st.warning(macro_msg)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("기업", company_info.get("corp_name", company))
        c2.metric("종목코드", corp.get("stock_code", "-"))
        c3.metric("현재가", f"{market.get('current_price'):,.0f}" if market and market.get("current_price") else "-")
        macro_status = macro.get("status") if macro else "-"
        c4.metric("ECOS", macro_status)

        st.subheader("Feature Engine 결과")
        st.dataframe(features[["category", "feature", "value", "unit", "quality_flag", "source"]], use_container_width=True, hide_index=True)

        st.subheader("핵심 거시지표")
        macro_features = features[features["category"] == "Macro"].copy()
        st.dataframe(macro_features[["feature", "value", "unit", "quality_flag"]], use_container_width=True, hide_index=True)

        with st.expander("ECOS 시계열 차트"):
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
                    df = df.dropna(subset=["DATA_VALUE"]).copy()
                    df.index = range(len(df))
                    st.write(series.get("label", key))
                    st.line_chart(df.set_index(df.columns[0])["DATA_VALUE"])
            else:
                st.info("ECOS 데이터가 없습니다.")

        with st.expander("OpenDART 재무 데이터"):
            st.dataframe(financials, use_container_width=True, hide_index=True)
        with st.expander("최근 공시 목록"):
            st.dataframe(notices, use_container_width=True, hide_index=True)
        with st.expander("시장 데이터"):
            if market and "history" in market:
                st.dataframe(market["history"].tail(120), use_container_width=True, hide_index=True)
            else:
                st.info("시장 데이터가 없습니다.")
        with st.expander("ECOS 원본/상태 정보"):
            st.json(macro if macro else {})

        csv = features.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Feature CSV 다운로드", csv, file_name=f"{company_info.get('stock_code','company')}_features.csv", mime="text/csv")
    except Exception as e:
        st.error("분석 중 오류가 발생했습니다.")
        st.code(str(e))
        with st.expander("개발용 오류 로그"):
            st.code(traceback.format_exc())
