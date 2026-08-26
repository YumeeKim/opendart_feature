from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


class InvestmentAI:
    """AI interpretation layer. Deterministic calculations remain outside the LLM."""

    def __init__(self, api_key: str, model: str = "gpt-5.6-luna"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 없습니다.")
        self.client = OpenAI(api_key=api_key)
        self.model = model or "gpt-5.6-luna"

    def _clean_json(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise RuntimeError("OpenAI 응답이 비어 있습니다.")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match:
                return json.loads(match.group(0))
            raise RuntimeError("OpenAI 응답을 JSON으로 해석하지 못했습니다.")

    def analyze(self, dossier: dict[str, Any]) -> dict[str, Any]:
        system = """
너는 한국 상장기업을 분석하는 buy-side 투자 리서치 애널리스트다.

중요 규칙:
1) 입력된 숫자와 사실을 바꾸거나 새 숫자를 만들어내지 마라.
2) 계산되지 않은 PER/PBR/EV-EBITDA/DCF/적정주가는 절대로 추정해서 만들지 마라. 데이터가 없으면 null 또는 '계산 대기'라고 하라.
3) ROIC, ROE, FCF, 성장률, 모멘텀, 변동성, 회계위험, 거시변수는 제공된 값만 사용하라.
4) 공시 목록의 제목만 제공될 경우, 공시 본문을 읽었다고 가장하지 마라. 제목에서 관찰 가능한 신호만 말하라.
5) 산업/거시 해석은 입력된 산업코드·시장·거시데이터에 근거하고, 확인되지 않은 산업 사실은 '확인 필요'라고 표시하라.
6) 매수/보류/매도는 투자자문이 아니라 '조건부 정량 판단'으로 표현하라.
7) 왜 그런 판단을 내렸는지 입력 데이터의 근거를 함께 제시하라.
8) 반드시 JSON만 출력하라.
"""

        schema = {
            "executive_summary": "string",
            "decision": "BUY_CANDIDATE|HOLD|SELL_CANDIDATE|INSUFFICIENT_DATA",
            "decision_reasons": ["string"],
            "quality_assessment": ["string"],
            "valuation_assessment": ["string"],
            "momentum_assessment": ["string"],
            "macro_assessment": ["string"],
            "industry_hidden_signals": [
                {
                    "signal": "string",
                    "impact": "positive|negative|neutral",
                    "evidence": "string",
                    "confidence": "high|medium|low",
                }
            ],
            "accounting_risk_observations": [
                {
                    "flag": "string",
                    "severity": "high|medium|low",
                    "evidence": "string",
                    "why_it_matters": "string",
                }
            ],
            "buy_conditions": ["string"],
            "hold_conditions": ["string"],
            "sell_conditions": ["string"],
            "missing_data": ["string"],
            "confidence": "high|medium|low",
        }

        user = {
            "task": "현재 확보된 Feature와 공시/거시 정보를 바탕으로 기업의 투자 의사결정 조건을 해석하라.",
            "required_output_schema": schema,
            "dossier": dossier,
        }

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False, default=str)},
            ],
        )
        return self._clean_json(getattr(response, "output_text", ""))
