# 空間音 異常検知システム 設計書

## 概要

環境音の「ふだんの音」をAIが学習し、異常を検知して意味づけするシステム。

音声センサーデータから「いま、この空間で何が起きているのか」をリアルタイムで判断する。

## アーキテクチャ

```
[AudioSet] → [CLAP埋め込み] → [Faissインデックス] = ベースライン (ふだんの音)
                                      ↓
[センサー音声] → [CLAP埋め込み] → [Faiss検索] → 距離スコア
                                      ↓
                              距離 >= 閾値 ?
                              ↓ YES          ↓ NO
                     [Qwen2.5-7B推論]    正常スコア記録
                     判断文生成 (1-3秒)
                              ↓
                        アラート発火
                     「空調停止の可能性」
```

## GPU活用ポイント

| 処理 | CPU | GPU | 備考 |
|------|-----|-----|---------------|
| CLAP埋め込み | 5秒/件 | 0.3秒/件 | **3秒間隔のリアルタイム監視を実現** |
| Qwen2.5-7B推論 | 60秒 (実用外) | 1-3秒 | **検知のたびにAIが判断文を生成** |
| 大量データ学習 | 12時間 | 1時間 | **5万件のベースラインを短時間で構築** |

CPUでは3秒間隔の検知が不可能 (1回の処理に5秒かかるため)。

GPUなら0.3秒で処理できるので、3秒間隔で3センサーを常時監視できる。

## データパイプライン

```bash
make pipeline SOURCE=unbalanced LIMIT=50000
```

| Step | 処理 | 入力 | 出力 |
|------|------|------|------|
| 1 | AudioSetダウンロード | HuggingFace | data/raw/*.wav + metadata.jsonl |
| 2 | CLAP埋め込み (GPU) | 音声ファイル | data/embeddings/embeddings.npy |
| 3 | Faissインデックス構築 | 埋め込みベクトル + categories.py | data/index/{sensor}_baseline.faiss |
| 4 | ベースライン/テスト分離 | 全音声 | 70%ベースライン / 30%ストリーム |

## 技術スタック

| モジュール | 役割 |
|-----------|------|
| CLAP (laion/larger_clap_music_and_speech) | 音声 → 512次元ベクトル変換 |
| Faiss (IVFFlat, Inner Product / コサイン類似度, nlist=256, nprobe=64) | ベクトル類似検索 |
| Qwen2.5-7B-Instruct | 異常の意味づけ (ローカルGPU推論) |
| FastAPI + asyncio | 非同期APIサーバー + 検知ループ |
| React + Recharts | リアルタイムダッシュボード |

全てオープンソース (Apache 2.0 / MIT / CC BY 4.0)。クローズドAPIは使用しない。

## ドメインモデル

| モデル | ファイル | 役割 |
|--------|---------|------|
| AnomalyResult | domain/anomaly.py | 検知結果 (距離, 異常判定, matched_labels, baseline_categories) |
| Intent | domain/intent.py | LLMの判断文 (judgment, recommendation, urgency, supplement) |
| Alert | domain/alert.py | 異常検知 + 判断文をまとめたアラート (intent は Optional) |
| ScoreEntry | domain/score.py | 時系列スコア (sensor_id, timestamp: str, distance: float) |
| Sensor | domain/sensor.py | センサー定義 (sensor_id, name, location) |
| SensorStatus | domain/status.py | センサー状態 (is_active, last_checked, current_distance, is_anomaly) |
| Baseline | domain/baseline.py | インデックス情報 (sensor_id, created_at, sample_count, index_path) |
| CATEGORY_SENSOR_MAP | domain/categories.py | AudioSetラベル → センサーID マッピング (urban/indoor/park) |
| EmbeddingPort | domain/ports.py | CLAP埋め込みの Protocol (embed, embed_batch, is_loaded) |
| IndexPort | domain/ports.py | Faiss検索の Protocol (search, build, save, load, get_all_labels) |
| LLMPort | domain/ports.py | LLM推論の Protocol (interpret, is_available) |

## レイヤー構成

```
api/        → ルーティング + ミドルウェア (レート制限, リクエストサイズ制限)
  routes/   → status (/api/sensors/*), alerts (/api/alerts),
              baseline (/api/baseline/*), demo (/api/demo/*)
  deps.py   → SENSORS, ALERTS, SCORE_HISTORY, state_lock (asyncio.Lock),
              clap_model, faiss_index, faiss_indices, llm_client,
              save_state/load_state (JSONL永続化)
  main.py   → create_app(), lifespan (モデルロード + 検知ループ起動)
use_case/   → ビジネスロジック (Protocol経由でinfraに依存)
  detect_anomaly.py     → CLAP埋め込み + Faiss検索 → AnomalyResult
  generate_intent.py    → LLMPort.interpret 呼び出し → Intent
  run_detection_loop.py → 定期検知ループ + シミュレーションモード
  get_status.py         → スコア履歴からセンサー状態を算出
domain/     → 型定義 + Protocol定義 (外部依存ゼロ)
infra/      → 外部接続 (CLAP, Faiss, Qwen, config, audio_loader)
errors/     → カスタム例外 (AppError基底, audio/index/intent/model)
```

use_case層は `domain/ports.py` の Protocol に依存し、infra の具体実装には直接依存しない。

## 共有状態とスレッドセーフティ

| 状態 | 型 | 保護 |
|------|-----|------|
| ALERTS | list[Alert] | asyncio.Lock (state_lock) |
| SCORE_HISTORY | list[ScoreEntry] | asyncio.Lock (state_lock) |
| SENSORS | list[Sensor] | 起動時に確定 (変更なし) |
| faiss_indices | dict[str, FaissIndex] | lifespan で初期化 (読み取りのみ) |

`state_lock` は `deps.py` で定義され、検知ループ (`run_detection_loop`) とAPIルート (`demo.py`, `status.py`, `alerts.py`) の両方から使用される。

## リアルタイム検知フロー

1. `lifespan` 起動時に `init_detection_context` でモジュール変数にコンテキストを設定
2. `run_detection_loop` が `DETECTION_INTERVAL_SEC` ごとに全センサーを巡回
3. 各センサーの `data/streams/{sensor_id}/` からカーソルで次の音声ファイルを取得 (巡回)
4. CLAP で 512次元ベクトルに変換
5. センサー別 Faiss インデックスで最近傍5件を検索、最大類似度 (best match) から距離 (1 - max_similarity) を算出
6. 距離 >= ANOMALY_THRESHOLD なら異常判定
6.5. CLAPテキストエンコーダで無音判定: `embed_text("silence")` とのコサイン類似度 > 0.45 なら距離を 1.0 に強制
7. 異常時: Qwen2.5-7B が判断文を JSON で生成 (LLMPort.interpret, run_in_executor で非同期)
8. `asyncio.Lock` (state_lock) で ALERTS / SCORE_HISTORY を保護しながら追加
9. 上限管理: SCORE_HISTORY は最大1000件、ALERTS は最大100件 (古いものから削除)
10. モデル未ロード or 音声ファイルなし時はシミュレーションモードで疑似スコア生成
11. フロントエンドが VITE_POLL_INTERVAL ポーリングでグラフ + アラートを更新

## APIエンドポイント一覧

| メソッド | パス | 説明 |
|---------|------|------|
| GET | /api/health | ヘルスチェック (CLAP/Faiss/LLM の状態) |
| GET | /api/sensors/status | 全センサーの現在状態 |
| GET | /api/sensors/scores | 時系列スコア (limit, sensor_id フィルタ) |
| GET | /api/alerts | アラート一覧 (新しい順) |
| GET | /api/baseline/{sensor_id} | ベースライン情報 |
| POST | /api/demo/detect | 即時検知 (sensor_id 指定可) |
| POST | /api/demo/simulate | シナリオ再生 (normal / hvac_failure / unusual_activity) |
| POST | /api/demo/generate | ランダム音声生成 + 即時検知 (一時ファイル方式) |
| POST | /api/demo/inject | 既存音声ファイルをストリームに注入 |
| POST | /api/demo/reset | ALERTS + SCORE_HISTORY クリア |
| GET | /api/demo/scenarios | 利用可能なシナリオ一覧 |

## デモ機能

| 操作 | API | 動作 |
|------|-----|------|
| シナリオ再生 | POST /api/demo/simulate | normal / hvac_failure / unusual_activity のスコアを生成。duration_points で時系列ポイント数を指定 |
| 音声生成+検知 | POST /api/demo/generate | センサー環境別のランダム音声を生成 (normal/anomaly/silence) → data/tmp/ に一時保存 → CLAP + Faiss で即時検知 → 一時ファイル自動削除 |
| 即時検知 | POST /api/demo/detect | ストリーム内の次の音声で指定/全センサーを即時検知 |
| 音声注入 | POST /api/demo/inject | data/ 内の既存音声ファイルをセンサーストリームにコピー (50MB制限, シンボリックリンク不可) |
| データクリア | POST /api/demo/reset | ALERTS + SCORE_HISTORY をクリア |
| シナリオ一覧 | GET /api/demo/scenarios | 利用可能なデモシナリオの一覧を返却 |

音声生成 (`/api/demo/generate`) は `data/tmp/` に一時ファイルを保存し、検知後に自動削除する。ストリームを汚染しない。

## フロントエンド構成

| ファイル | 役割 |
|---------|------|
| Dashboard.tsx | メインダッシュボード (スコアグラフ + アラート + センサー状態) |
| SoundPulse.tsx | リアルタイムスコアグラフ |
| AlertList.tsx | アラート一覧表示 |
| DemoPanel.tsx | デモ操作パネル (シナリオ再生, 音声生成, 検知, リセット) |
| ErrorBoundary.tsx | React エラーバウンダリ (クラッシュ時のフォールバック表示) |
| SensorMap.tsx | センサー配置マップ |
| StatCards.tsx | 統計カード表示 |
| useStatus.ts | センサー状態のポーリングフック |
| useAlerts.ts | アラートのポーリングフック |
| useScores.ts | スコア履歴のポーリングフック |
| api.ts | API呼び出しユーティリティ |
| types.ts | TypeScript型定義 |
| config.ts | フロントエンド設定 (API URL等) |
| formatDistance.ts | 距離スコアのフォーマット表示 |
| formatRelativeTime.ts | 相対時刻のフォーマット表示 |

## 環境別設定

| | ローカル (CPU) | GPUサーバー |
|---|---|---|
| DEVICE | cpu | cuda |
| FAISS_USE_GPU | false | true |
| LLM_ENABLED | false | true |
| DETECTION_INTERVAL_SEC | 300 (シミュレーション) | 3 (リアルタイム) |
| VITE_POLL_INTERVAL | 5000ms | 2000ms |
| ANOMALY_THRESHOLD | 0.55 | 0.55 |

コード上のデフォルト値 (`config.py`): ANOMALY_THRESHOLD=0.55, DETECTION_INTERVAL_SEC=300。
`.env.example` / `.env.production.example` で上書きして使用する。

## ミドルウェア

| 機能 | 設定値 | 説明 |
|------|--------|------|
| リクエストサイズ制限 | 10MB | Content-Length ヘッダーで制限 |
| レート制限 | 300リクエスト/60秒 (IP単位) | /api/health は除外 |
| リクエストログ | JSON形式 | メソッド, パス, ステータスコード, 所要時間 |
| エラーハンドリング | AppError → JSON応答 | カスタム例外を統一的にJSON変換 |

## ライセンス

| 名称 | ライセンス | 商用利用 |
|------|-----------|---------|
| Google AudioSet | CC BY 4.0 | OK |
| CLAP | Apache 2.0 | OK |
| Qwen2.5-7B-Instruct | Apache 2.0 | OK |
| Faiss | MIT | OK |
