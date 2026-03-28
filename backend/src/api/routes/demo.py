import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.api.deps import ALERTS, SCORE_HISTORY, SENSORS, state_lock
from src.domain.alert import Alert
from src.domain.anomaly import AnomalyResult
from src.domain.intent import Intent
from src.domain.score import ScoreEntry
from src.infra.audio_loader import SUPPORTED_FORMATS
from src.infra.config import settings
from src.use_case.run_detection_loop import (
    MAX_ALERTS,
    MAX_SCORE_HISTORY,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo")

MAX_INJECT_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


class DetectRequest(BaseModel):
    sensor_id: str | None = Field(
        None,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]*$",
    )


class InjectRequest(BaseModel):
    sensor_id: str = Field(
        ...,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    audio_source: str = Field(..., max_length=500)


class SimulateRequest(BaseModel):
    scenario: str = Field(..., max_length=50)
    duration_points: int = Field(10, ge=1, le=100)

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, v: str) -> str:
        valid = {
            "normal",
            "hvac_failure",
            "unusual_activity",
        }
        if v not in valid:
            raise ValueError(f"Unknown scenario: {v}. Valid: {valid}")
        return v


class Scenario(BaseModel):
    name: str
    description: str
    sensor_id: str
    audio_source: str


DEMO_SCENARIOS: list[Scenario] = [
    Scenario(
        name="normal",
        description="通常の環境音（異常なし）",
        sensor_id="urban",
        audio_source="",
    ),
    Scenario(
        name="hvac_failure",
        description="空調システム停止 - 機械音が消失",
        sensor_id="indoor",
        audio_source="",
    ),
    Scenario(
        name="unusual_activity",
        description="公園エリアで夜間の不審な活動を検知",
        sensor_id="park",
        audio_source="",
    ),
]

_SCENARIO_SCORES: dict[str, dict[str, tuple[float, float]]] = {
    "normal": {
        "urban": (0.1, 0.3),
        "indoor": (0.1, 0.3),
        "park": (0.1, 0.3),
    },
    "hvac_failure": {
        "urban": (0.1, 0.3),
        "indoor": (0.6, 0.9),
        "park": (0.1, 0.3),
    },
    "unusual_activity": {
        "urban": (0.1, 0.3),
        "indoor": (0.1, 0.3),
        "park": (0.6, 0.9),
    },
}

_SCENARIO_INTENTS: dict[str, dict] = {
    "hvac_failure": {
        "sensor_id": "indoor",
        "judgment": (
            "空調システムが停止した可能性があります。"
            "通常聞こえるはずの機械音（室外機のハム音）が検出されず、"
            "不自然な静寂に置き換わっています。"
        ),
        "recommendation": (
            "3Fの空調設備を点検してください。"
            "電源供給とコンプレッサーの状態を確認する必要があります。"
        ),
        "urgency": "high",
        "supplement": (
            "空調停止が長引くと、室内の空気環境と温度管理に影響が出る可能性があります。"
        ),
    },
    "unusual_activity": {
        "sensor_id": "park",
        "judgment": (
            "公園エリアで夜間の不審な音響活動を検知しました。"
            "通常の営業時間外に人の活動を示す"
            "音響パターンが確認されています。"
        ),
        "recommendation": ("警備員による公園南入口付近の巡回を推奨します。"),
        "urgency": "medium",
        "supplement": ("検出された音響パターンは、野生動物や天候に起因するものとは異なります。"),
    },
}


async def _generate_scenario_points(
    scenario: str,
    duration_points: int,
) -> list[ScoreEntry]:
    """Generate synthetic score data for a scenario."""
    from datetime import timedelta

    from src.api.deps import SENSORS, faiss_index, faiss_indices, llm_client

    score_ranges = _SCENARIO_SCORES.get(scenario, _SCENARIO_SCORES["normal"])
    threshold = settings.ANOMALY_THRESHOLD
    generated: list[ScoreEntry] = []

    base_time = datetime.now(tz=timezone.utc)
    for i in range(duration_points):
        # 各ポイントを30秒ずつずらして時系列を作る
        now = base_time + timedelta(seconds=i * 30)
        for sensor_id, (lo, hi) in score_ranges.items():
            distance = random.uniform(lo, hi)

            # Enrich AnomalyResult with baseline/label info when index is available
            idx = faiss_indices.get(sensor_id) or faiss_index
            baseline_cats = idx.get_all_labels() if idx and idx.is_built else []

            result = AnomalyResult(
                sensor_id=sensor_id,
                timestamp=now,
                distance=distance,
                is_anomaly=distance >= threshold,
                threshold=threshold,
                matched_labels=[],
                baseline_categories=baseline_cats,
            )

            entry = ScoreEntry(
                sensor_id=sensor_id,
                timestamp=result.timestamp.isoformat(),
                distance=result.distance,
            )
            generated.append(entry)

            alert = None
            if result.is_anomaly:
                intent = None

                # Try real LLM interpretation for hvac_failure scenario
                if scenario == "hvac_failure" and llm_client.is_available:
                    try:
                        from src.use_case.generate_intent import generate_intent

                        sensor_obj = next((s for s in SENSORS if s.sensor_id == sensor_id), None)
                        if sensor_obj:
                            intent = await generate_intent(result, sensor_obj, llm_client)
                    except Exception as e:
                        logger.warning(
                            "Real LLM intent failed for %s, using fallback: %s",
                            sensor_id,
                            e,
                        )

                # Fallback to hardcoded intent
                if intent is None:
                    intent_data = _SCENARIO_INTENTS.get(scenario)
                    if intent_data and (intent_data["sensor_id"] == sensor_id):
                        intent = Intent(
                            sensor_id=sensor_id,
                            timestamp=now,
                            judgment=intent_data["judgment"],
                            recommendation=intent_data["recommendation"],
                            urgency=intent_data["urgency"],
                            supplement=intent_data["supplement"],
                        )

                alert = Alert(
                    alert_id=str(uuid4()),
                    sensor_id=sensor_id,
                    timestamp=now,
                    anomaly=result,
                    intent=intent,
                )

            async with state_lock:
                SCORE_HISTORY.append(entry)
                if alert is not None:
                    ALERTS.append(alert)

    async with state_lock:
        if len(SCORE_HISTORY) > MAX_SCORE_HISTORY:
            del SCORE_HISTORY[: len(SCORE_HISTORY) - MAX_SCORE_HISTORY]
        if len(ALERTS) > MAX_ALERTS:
            del ALERTS[: len(ALERTS) - MAX_ALERTS]

    return generated


@router.post("/detect")
async def demo_detect(req: DetectRequest | None = None) -> dict:
    """Trigger immediate detection (bypass the 30s loop)."""
    from src.use_case.run_detection_loop import (
        _detect_one_sensor,
    )

    sensor_filter = req.sensor_id if req else None
    target_sensors = (
        [s for s in SENSORS if s.sensor_id == sensor_filter] if sensor_filter else SENSORS
    )

    results = []
    for sensor in target_sensors:
        try:
            await _detect_one_sensor(sensor.sensor_id)
            results.append({"sensor_id": sensor.sensor_id, "status": "ok"})
        except Exception as e:
            logger.warning("Detection failed for %s: %s", sensor.sensor_id, e)
            results.append(
                {
                    "sensor_id": sensor.sensor_id,
                    "status": "error",
                }
            )

    return {"triggered": len(results), "results": results}


@router.post("/inject")
async def demo_inject(req: InjectRequest) -> JSONResponse:
    """Inject an anomaly audio file into a sensor's stream."""
    source = Path(req.audio_source).resolve()
    allowed_base = settings.DATA_DIR.resolve()
    if not source.is_relative_to(allowed_base):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": "Source must be within data directory",
            },
        )
    if source.suffix.lower() not in SUPPORTED_FORMATS:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": "Only audio files allowed",
            },
        )
    if not source.exists():
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "detail": "Source not found",
            },
        )
    if source.is_symlink():
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": "Symlinks not allowed",
            },
        )
    try:
        file_size = source.stat().st_size
    except OSError:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": "Cannot read file size",
            },
        )
    if file_size > MAX_INJECT_FILE_SIZE:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": "File exceeds 50MB limit",
            },
        )

    target_dir = settings.streams_dir / req.sensor_id
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(target_dir.glob("*"))
    next_num = len(existing)
    target = target_dir / f"{next_num:04d}{source.suffix}"
    shutil.copy2(source, target)

    logger.info("Injected %s -> %s", source, target)
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "injected": str(target)},
    )


@router.post("/simulate")
async def demo_simulate(req: SimulateRequest) -> JSONResponse:
    """Simulate a scenario by injecting synthetic score data."""

    generated = await _generate_scenario_points(req.scenario, req.duration_points)

    logger.info(
        "Simulated scenario '%s' with %d points",
        req.scenario,
        len(generated),
    )

    async with state_lock:
        alerts_total = len(ALERTS)

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "scenario": req.scenario,
            "points_generated": len(generated),
            "alerts_total": alerts_total,
        },
    )


@router.post("/reset")
async def demo_reset() -> dict:
    """Clear all alerts and score history."""
    async with state_lock:
        ALERTS.clear()
        SCORE_HISTORY.clear()
    logger.info("Demo reset: cleared all alerts and scores")
    return {"status": "ok"}


class GenerateRequest(BaseModel):
    sensor_id: str = Field(
        "urban",
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    sound_type: str = Field(
        "anomaly",
        description="normal / anomaly / silence",
    )
    auto_detect: bool = Field(
        True,
        description="生成後に即時検知を実行するか",
    )

    @field_validator("sound_type")
    @classmethod
    def validate_sound_type(cls, v: str) -> str:
        valid = {"normal", "anomaly", "silence"}
        if v not in valid:
            raise ValueError(f"Unknown sound_type: {v}. Valid: {valid}")
        return v


# センサー環境別の音声パターン
_ENV_AUDIO_PARAMS: dict[str, dict] = {
    "urban": {
        "base_freq": 80,
        "harmonics": [160, 320],
        "noise_level": 0.15,
        "description": "交通音 (低周波エンジン音 + 路面ノイズ)",
    },
    "indoor": {
        "base_freq": 120,
        "harmonics": [240, 4000],
        "noise_level": 0.05,
        "description": "室内音 (空調ハム + 静かな背景)",
    },
    "park": {
        "base_freq": 2000,
        "harmonics": [4000, 6000],
        "noise_level": 0.08,
        "description": "公園音 (鳥のさえずり + 風)",
    },
}


def _generate_audio(
    sound_type: str,
    sensor_id: str = "urban",
    sr: int = 48000,
    duration: float = 3.0,
) -> np.ndarray:
    """センサー環境に応じたランダム音声を生成する."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples)
    params = _ENV_AUDIO_PARAMS.get(sensor_id, _ENV_AUDIO_PARAMS["urban"])

    if sound_type == "normal":
        # その環境の「ふだんの音」に近い音を生成
        audio = params["noise_level"] * np.random.randn(n_samples)
        audio += 0.15 * np.sin(2 * np.pi * params["base_freq"] * t)
        for h in params["harmonics"]:
            audio += 0.05 * np.sin(2 * np.pi * h * t)

    elif sound_type == "silence":
        # 完全無音 (機械停止をシミュレート)
        audio = np.zeros(n_samples, dtype=np.float64)

    else:
        # 異常音: 人の声(スピーチ風)の合成で確実にベースラインと離す
        # CLAPは音声認識ベースなので、音声的な特徴を持つ信号が最も異なる
        audio = np.zeros(n_samples, dtype=np.float64)

        # フォルマント周波数で擬似音声を生成 (母音の共鳴周波数)
        formants = [700, 1200, 2500]  # /a/ の近似
        for f in formants:
            audio += 0.3 * np.sin(2 * np.pi * f * t)

        # ピッチ変動 (人の声らしさ)
        pitch = 150 + 30 * np.sin(2 * np.pi * 5 * t)
        audio += 0.4 * np.sin(2 * np.pi * np.cumsum(pitch / sr))

        # 断続的なオン/オフ (話している/止まる)
        for _ in range(random.randint(5, 10)):
            start = random.randint(0, n_samples - sr)
            length = random.randint(sr // 4, sr)
            audio[start : start + length] *= 3.0

        # ノイズで質感を付ける
        audio += 0.15 * np.random.randn(n_samples)
        audio *= 2.0

    return audio.astype(np.float32)


@router.post("/generate")
async def demo_generate(req: GenerateRequest) -> dict:
    """ランダム音声を生成してセンサーストリームに注入し、即時検知を実行."""
    try:
        import soundfile as sf
    except ImportError:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": "soundfile not installed",
            },
        )

    # 音声生成
    audio = _generate_audio(req.sound_type, req.sensor_id)

    # 一時ファイルに保存 (ストリームを汚さない)
    tmp_dir = settings.DATA_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / f"demo_{req.sensor_id}_{req.sound_type}.wav"
    sf.write(str(output_path), audio, 48000)

    logger.info(
        "Generated %s audio -> %s",
        req.sound_type,
        output_path,
    )

    result = {
        "status": "ok",
        "sound_type": req.sound_type,
        "sensor_id": req.sensor_id,
        "file": str(output_path),
    }

    # 生成した音声で即時検知
    if req.auto_detect:
        try:
            import asyncio

            from src.api.deps import clap_model, faiss_index, faiss_indices
            from src.infra.audio_loader import load_audio
            from src.use_case.detect_anomaly import detect_anomaly_async
            from src.use_case.run_detection_loop import (
                _create_alert_if_anomaly,
                _record_result,
            )

            # 生成したファイルを直接検知に使う
            idx = faiss_indices.get(req.sensor_id)
            if idx is None or not idx.is_built:
                idx = faiss_index

            if clap_model.is_loaded and idx.is_built:
                loop = asyncio.get_running_loop()
                loaded_audio = await loop.run_in_executor(None, load_audio, output_path)
                detection = await detect_anomaly_async(
                    audio=loaded_audio,
                    sensor_id=req.sensor_id,
                    clap=clap_model,
                    index=idx,
                    threshold=settings.ANOMALY_THRESHOLD,
                )
                await _record_result(detection, req.sensor_id)
                await _create_alert_if_anomaly(
                    detection,
                    req.sensor_id,
                )
                result["detected"] = True
                result["distance"] = detection.distance
                result["is_anomaly"] = detection.is_anomaly
            else:
                result["detected"] = False
                result["reason"] = "models not loaded"
        except Exception as e:
            logger.warning("Auto-detect failed: %s", e)
            result["detected"] = False

    # 一時ファイルをクリーンアップ
    try:
        output_path.unlink(missing_ok=True)
    except OSError:
        pass

    return result


@router.get("/scenarios")
async def get_scenarios() -> list[Scenario]:
    """Return available demo scenarios."""
    return DEMO_SCENARIOS
