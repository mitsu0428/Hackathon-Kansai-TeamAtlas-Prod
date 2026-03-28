# セットアップ・実行方法

## 前提環境

| 環境 | ローカル開発 (CPU) | GPUサーバー (本番) |
|------|-------------------|-------------------|
| OS | macOS / Linux | Linux |
| Python | 3.11+ | 3.11+ |
| Node.js | 18+ | 18+ |
| GPU | 不要 | NVIDIA H100 / A5000 (24GB VRAM推奨) |
| その他 | - | - |

### GPU VRAM使用量

| モデル | VRAM |
|--------|------|
| CLAP | 約1GB |
| Qwen2.5-7B-Instruct (float16) | 約14GB |
| Faiss + その他 | 約1GB |
| **合計** | **約16GB** |

VRAM不足時は `Qwen2.5-3B-Instruct` (約6GB) に変更可能。

## ローカル開発 (CPU, シミュレーションモード)

```bash
git clone git@github.com:mitsu0428/Hackathon-Kansai-TeamAtlas.git
cd Hackathon-Kansai-TeamAtlas
make setup             # Python + Node.js 依存インストール
cp .env.example .env   # 環境変数
make dev               # バックエンド(:8000) + フロントエンド(:5173) 起動
```

http://localhost:5173 を開いてデモパネルから操作。

## GPUサーバー (本番)

```bash
# SSH (ポートフォワード付き)
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 user@gpu-server

# セットアップ (初回のみ)
git clone git@github.com:mitsu0428/Hackathon-Kansai-TeamAtlas.git
cd Hackathon-Kansai-TeamAtlas
make setup-gpu
cp .env.production.example .env

# AudioSetダウンロード (HuggingFaceから直接取得)
backend/.venv/bin/python scripts/download_dataset.py --source balanced --limit 0

# ストリーム分割 (AudioSetラベルでurban/indoor/parkに自動分類)
backend/.venv/bin/python scripts/prepare_streams.py --from-raw

# CLAP埋め込み (Slurm環境ならGPUジョブ投入)
make embed
# squeue -u $USER でジョブ完了を確認後:

# センサー別インデックス構築
make update-all

# 起動
make dev
```

## 学習方法 (データパイプライン)

`make pipeline` が以下を自動実行:

```
Step 1/4: AudioSetダウンロード     → data/raw/*.wav + metadata.jsonl
Step 2/4: CLAP埋め込み (GPU)       → data/embeddings/embeddings.npy
Step 3/4: Faissインデックス構築     → data/index/{sensor}_baseline.faiss
Step 4/4: ベースライン/テスト分離   → 70%学習 / 30%ストリーム
```

| コマンド | データ量 | 所要時間 |
|---------|---------|---------|
| `make pipeline LIMIT=500` | 500件 | 数分 |
| `make pipeline SOURCE=unbalanced LIMIT=50000` | 50,000件 | 1-2時間 |
| `make pipeline SOURCE=unbalanced LIMIT=0` | 約200万件 | 数日 |

## 推論方法 (リアルタイム検知)

`make dev` でサーバー起動後、自動的に検知ループが開始:

1. `DETECTION_INTERVAL_SEC` 秒ごとに全センサーの音声を取得
2. CLAPで512次元ベクトルに変換
3. Faissでベースラインと比較 → 距離スコア算出
4. 距離 >= `ANOMALY_THRESHOLD` なら異常判定
5. 異常時: Qwen2.5-7Bが判断文をJSON生成

## エントリーポイント

| 用途 | コマンド / ファイル |
|------|-------------------|
| サーバー起動 | `make dev` → `uvicorn src.api.main:app` |
| データパイプライン | `make pipeline` |
| AudioSetダウンロード | `python scripts/download_dataset.py` |
| CLAP埋め込み | `python batch/jobs/run_embed.py` |
| インデックス構築 | `python batch/jobs/run_update.py --all-sensors` |
| ストリーム分離 | `python scripts/prepare_streams.py --from-raw` |

## 環境変数 (.env)

| 変数名 | CPU (ローカル) | GPU (本番) | 説明 |
|--------|--------------|-----------|------|
| `DEVICE` | `cpu` | `cuda` | PyTorchデバイス |
| `FAISS_USE_GPU` | `false` | `true` | Faiss GPU使用フラグ |
| `LLM_ENABLED` | `false` | `true` | AI判断文の有効化 |
| `LLM_MODEL_NAME` | - | `Qwen/Qwen2.5-7B-Instruct` | 使用LLM |
| `LLM_MAX_NEW_TOKENS` | `512` | `512` | LLM最大トークン数 |
| `DETECTION_INTERVAL_SEC` | `300` | `3` | 検知間隔 (秒) |
| `ANOMALY_THRESHOLD` | `0.55` | `0.55` | 異常判定の閾値 |
| `VITE_POLL_INTERVAL` | `5000` | `2000` | フロント更新間隔 (ms) |
| `VITE_API_URL` | `http://localhost:8000` | - | APIベースURL |
| `CORS_ORIGINS` | `localhost:5173,...` | `localhost:5173,...` | 許可オリジン (カンマ区切り) |
| `DEMO_ENABLED` | `true` | `false` | デモエンドポイントの有効化 |
| `SLACK_WEBHOOK_URL` | - | (設定時のみ) | Slack通知用Webhook URL |
| `HOST` | `0.0.0.0` | `0.0.0.0` | サーバーホスト |
| `PORT` | `8000` | `8000` | サーバーポート |
| `DEBUG` | `false` | `false` | デバッグモード |

テンプレート: `.env.example` (CPU用) / `.env.production.example` (GPU用)

## なぜGPUが必要か

| 処理 | CPU | GPU | 速度差 |
|------|-----|-----|--------|
| CLAP埋め込み (1件) | 5秒 | **0.3秒** | 15倍 |
| Qwen2.5-7B推論 (1回) | 60秒 | **1-3秒** | 20倍 |
| ベースライン構築 (5万件) | 12時間 | **1時間** | 12倍 |
| **リアルタイム監視** | **不可能** | **3秒間隔** | - |

CPUでは1回の検知に5秒かかるため、3秒間隔の常時監視はGPUでしか実現できない。
