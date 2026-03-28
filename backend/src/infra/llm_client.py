import asyncio
import json
import logging
from datetime import datetime
from functools import partial

from src.domain.anomaly import AnomalyResult
from src.domain.intent import Intent
from src.domain.sensor import Sensor
from src.errors.intent import IntentGenerationError, IntentParseError
from src.infra.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert spatial sound analyst.
You analyze anomaly detection results from environmental sound sensors.
Given sensor information, anomaly details, and context, \
provide a structured assessment.

Always respond in valid JSON with these exact fields:
- "judgment": string - What is happening and why (2-3 sentences)
- "recommendation": string - What action should be taken (1-2 sentences)
- "urgency": string - One of: "low", "medium", "high", "critical"
- "supplement": string - Additional context or notes (1 sentence)

Consider the time of day, season, sensor location type, \
and distance score when making your assessment. \
Normal distance scores are close to 0, \
anomalous scores are significantly above the threshold."""


def _build_user_prompt(
    anomaly: AnomalyResult,
    sensor: Sensor,
    context: dict,
) -> str:
    now = anomaly.timestamp
    month = now.month
    if month in (3, 4, 5):
        season = "spring"
    elif month in (6, 7, 8):
        season = "summer"
    elif month in (9, 10, 11):
        season = "autumn"
    else:
        season = "winter"

    hour = now.hour
    if 5 <= hour < 10:
        time_period = "early morning"
    elif 10 <= hour < 17:
        time_period = "daytime"
    elif 17 <= hour < 21:
        time_period = "evening"
    else:
        time_period = "nighttime"

    ratio = anomaly.distance / max(anomaly.threshold, 1e-6)

    parts = [
        f"Sensor: {sensor.sensor_id} ({sensor.name})",
        f"Location: {sensor.location}",
        f"Time: {now.isoformat()} ({season}, {time_period})",
        f"Anomaly distance score: {anomaly.distance:.4f}",
        f"Threshold: {anomaly.threshold:.4f}",
        f"Score/Threshold ratio: {ratio:.2f}x",
    ]

    # ラベル情報があれば追加
    matched_labels = getattr(anomaly, "matched_labels", [])
    baseline_categories = getattr(anomaly, "baseline_categories", [])
    if baseline_categories:
        parts.append(f"Baseline sounds: [{', '.join(baseline_categories)}]")
    if matched_labels:
        parts.append(f"Matched sounds: [{', '.join(matched_labels)}]")
    if baseline_categories and matched_labels:
        missing = sorted(set(baseline_categories) - set(matched_labels))
        if missing:
            parts.append(f"Potentially missing: [{', '.join(missing)}]")

    for key, value in context.items():
        parts.append(f"{key}: {value}")

    return "\n".join(parts)


def _parse_response(
    text: str,
    sensor_id: str,
    timestamp: datetime,
) -> Intent:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        # JSON部分を抽出 (前後の余分なテキストを除去)
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            cleaned = cleaned[start:end]

        data = json.loads(cleaned)
        return Intent(
            sensor_id=sensor_id,
            timestamp=timestamp,
            judgment=data["judgment"],
            recommendation=data["recommendation"],
            urgency=data["urgency"],
            supplement=data.get("supplement", ""),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Intent parse error: %s (response: %.200s)", e, text)
        raise IntentParseError(str(e)) from e


def _fallback_intent(
    sensor_id: str,
    timestamp: datetime,
    anomaly: AnomalyResult,
) -> Intent:
    """JSON parse失敗時のフォールバックIntent."""
    return Intent(
        sensor_id=sensor_id,
        timestamp=timestamp,
        judgment=(
            f"Anomaly detected with distance score {anomaly.distance:.4f} "
            f"(threshold: {anomaly.threshold:.4f}). "
            "Automated interpretation unavailable."
        ),
        recommendation="Manual inspection recommended.",
        urgency="medium",
        supplement="LLM interpretation failed; using fallback.",
    )


class LLMClient:
    """Local LLM client using Qwen2.5-Instruct via transformers."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def initialize(self) -> None:
        """Initialize the local LLM model."""
        if not settings.LLM_ENABLED:
            logger.warning(
                "LLM_ENABLED=false, LLM client disabled (simulation mode)",
            )
            return
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            model_name = settings.LLM_MODEL_NAME
            logger.info("Loading LLM: %s", model_name)

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info("LLM loaded: %s", model_name)
        except Exception as e:
            raise IntentGenerationError(
                f"LLM init failed: {e}",
            ) from e

    @property
    def is_available(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _generate_sync(self, user_prompt: str) -> str:
        """Synchronous generation (run in executor)."""
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        # 入力部分を除外して生成部分のみデコード
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    async def interpret(
        self,
        anomaly: AnomalyResult,
        sensor: Sensor,
        context: dict | None = None,
    ) -> Intent:
        """Generate intent/interpretation for an anomaly result."""
        if not self.is_available:
            raise IntentGenerationError("LLM client not initialized")
        try:
            user_prompt = _build_user_prompt(
                anomaly,
                sensor,
                context or {},
            )
            logger.info(
                "Requesting LLM interpretation for sensor=%s",
                sensor.sensor_id,
            )

            # asyncio.run_in_executor でブロッキング推論を非同期化
            loop = asyncio.get_running_loop()
            try:
                response_text = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(self._generate_sync, user_prompt),
                    ),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "LLM inference timed out (30s) for sensor=%s",
                    sensor.sensor_id,
                )
                return _fallback_intent(
                    anomaly.sensor_id,
                    anomaly.timestamp,
                    anomaly,
                )

            # parse (リトライ1回)
            try:
                intent = _parse_response(
                    response_text,
                    sensor_id=anomaly.sensor_id,
                    timestamp=anomaly.timestamp,
                )
            except IntentParseError:
                logger.warning("First parse failed, retrying generation")
                try:
                    response_text = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(self._generate_sync, user_prompt),
                        ),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "LLM retry timed out (30s) for sensor=%s",
                        sensor.sensor_id,
                    )
                    return _fallback_intent(
                        anomaly.sensor_id,
                        anomaly.timestamp,
                        anomaly,
                    )
                try:
                    intent = _parse_response(
                        response_text,
                        sensor_id=anomaly.sensor_id,
                        timestamp=anomaly.timestamp,
                    )
                except IntentParseError:
                    logger.warning("Retry parse also failed, using fallback intent")
                    return _fallback_intent(
                        anomaly.sensor_id,
                        anomaly.timestamp,
                        anomaly,
                    )

            logger.info(
                "Intent generated: sensor=%s urgency=%s",
                sensor.sensor_id,
                intent.urgency,
            )
            return intent
        except (IntentGenerationError, IntentParseError):
            raise
        except Exception as e:
            raise IntentGenerationError(str(e)) from e
