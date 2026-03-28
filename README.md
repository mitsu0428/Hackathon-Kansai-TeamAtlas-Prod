# 空間音 異常検知システム

環境音の「ふだんの音」をAIが学習し、変化をリアルタイムで検知して自然言語で意味づけするシステム。
音声センサーデータから「いま、この空間で何が起きているのか」を判断する。

## 主な機能

- **消えた音の検知**: 空調停止など、あるべき音が消えた状態を検知
- **現れた音の検知**: 突発ノイズなど、ふだんにない音の出現を検知
- **AIによる判断文生成**: 検知結果をLLMが解釈し、日本語で「何が起きているか」「何をすべきか」を出力
- **リアルタイムダッシュボード**: 3秒間隔 (GPU) で検知結果をグラフ表示
- **デモ操作パネル**: シナリオ再生、ランダム音声生成+即時検知をブラウザから操作

### ユースケース

- 工場・ビル設備の音響監視 (空調故障、モーター異常)
- 公園・公共施設の夜間セキュリティ (不審音の検知)
- データセンターの環境監視 (冷却ファン停止の早期検出)

## 技術スタック

| カテゴリ | 技術 | 用途 |
|---------|------|------|
| 音声埋め込み | CLAP (laion/larger_clap_music_and_speech) | 音声→512次元ベクトル変換 |
| ベクトル検索 | Faiss (IVFFlat, nlist=256) | ベースラインとの類似度検索 |
| 判断文生成 | Qwen2.5-7B-Instruct | 異常の意味づけ (ローカルGPU推論) |
| バックエンド | FastAPI + asyncio | 非同期APIサーバー + 検知ループ |
| フロントエンド | React + TypeScript + Recharts | リアルタイムダッシュボード |
| パッケージ管理 | uv (Python) + npm (Node.js) | 依存管理 |

## モデル情報

| モデル | バージョン | ライセンス | 取得先 |
|--------|----------|-----------|--------|
| CLAP | laion/larger_clap_music_and_speech | Apache 2.0 | https://huggingface.co/laion/larger_clap_music_and_speech |
| Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | Apache 2.0 | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |

- いずれもファインチューニングなし。事前学習済みモデルをそのまま使用
- CLAP: 音声を512次元ベクトルに変換。HuggingFace Transformers経由でロード
- Qwen2.5-7B: JSON形式で判断文を生成。`run_in_executor` で非同期推論

## 使用データ

| データセット | 件数 | ライセンス | 用途 |
|-------------|------|-----------|------|
| Google AudioSet (balanced_train) | 約22,000件 | CC BY 4.0 | ベースライン構築 (デフォルト) |
| Google AudioSet (unbalanced_train) | 約2,000,000件 | CC BY 4.0 | 大量データでの本番運用 |

- 取得方法: `make pipeline` で HuggingFace (agkphysics/AudioSet) から自動ダウンロード
- AudioSet公式: https://research.google.com/audioset/
- HuggingFace: https://huggingface.co/datasets/agkphysics/AudioSet
- ダウンロードスクリプト: `scripts/download_dataset.py` (`--source balanced|unbalanced`)

## クイックスタート

```bash
make setup-gpu                    # 依存インストール
cp .env.production.example .env   # 環境変数
make pipeline SOURCE=balanced LIMIT=5000  # データ学習 (5000件, 約30分)
make dev                          # 起動 → http://localhost:5173
```

詳細は [docs/setup.md](docs/setup.md) を参照。

## ディレクトリ構成

```
Hackathon-Kansai-TeamAtlas/
├── backend/
│   ├── pyproject.toml          # Python依存定義
│   ├── uv.lock                 # ロックファイル
│   └── src/
│       ├── api/                # FastAPI ルーティング + ミドルウェア
│       │   ├── main.py         # アプリケーションファクトリ + lifespan
│       │   ├── deps.py         # シングルトン + 共有状態 (asyncio.Lock)
│       │   ├── middleware.py   # リクエストログ, レート制限, サイズ制限
│       │   └── routes/         # status, alerts, baseline, demo
│       ├── domain/             # 型定義 + Protocol (外部依存ゼロ)
│       │   ├── anomaly.py      # AnomalyResult
│       │   ├── intent.py       # Intent (LLM判断文)
│       │   ├── alert.py        # Alert (異常検知 + 判断文)
│       │   ├── score.py        # ScoreEntry (時系列スコア)
│       │   ├── sensor.py       # Sensor (センサー定義)
│       │   ├── status.py       # SensorStatus (センサー状態)
│       │   ├── baseline.py     # Baseline (インデックス情報)
│       │   ├── categories.py   # センサー別カテゴリマッピング
│       │   └── ports.py        # EmbeddingPort, IndexPort, LLMPort
│       ├── use_case/           # ビジネスロジック (Protocol依存)
│       │   ├── detect_anomaly.py
│       │   ├── generate_intent.py
│       │   ├── run_detection_loop.py
│       │   └── get_status.py
│       ├── infra/              # 外部接続 (CLAP, Faiss, Qwen)
│       │   ├── clap_model.py
│       │   ├── faiss_index.py
│       │   ├── llm_client.py
│       │   ├── audio_loader.py
│       │   └── config.py       # pydantic-settings (.env読み込み)
│       └── errors/             # カスタム例外 (AppError基底)
│           ├── base.py         # AppError
│           ├── audio.py
│           ├── index.py
│           ├── intent.py
│           └── model.py
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── components/         # Dashboard, SoundPulse, AlertList,
│       │                       # DemoPanel, ErrorBoundary, SensorMap, StatCards
│       ├── hooks/              # useStatus, useAlerts, useScores
│       └── lib/                # api, types, config, constants, formatDistance, formatRelativeTime
├── scripts/
│   ├── download_dataset.py     # AudioSetダウンロード (HuggingFace経由)
│   ├── download_testdata.py    # ESC-50テストデータダウンロード
│   ├── prepare_streams.py      # ベースライン/テスト分離
│   └── notify.sh               # Slack通知スクリプト
├── batch/
│   ├── jobs/
│   │   ├── run_embed.py        # CLAP埋め込みバッチ
│   │   └── run_update.py       # Faissインデックス構築
│   └── scripts/
│       ├── embed_batch.sh      # Slurm用埋め込みスクリプト
│       └── baseline_update.sh  # ベースライン更新スクリプト
├── data/                       # gitignore対象
│   ├── raw/                    # ダウンロード音声 + metadata.jsonl
│   ├── embeddings/             # CLAP埋め込みベクトル
│   ├── index/                  # Faissインデックス
│   ├── streams/                # センサーシミュレーション用音声
│   ├── state/                  # アラート/スコア永続化 (JSONL)
│   └── tmp/                    # デモ音声生成の一時ファイル
├── docs/
│   ├── pitch.md                # 発表スライド
│   ├── setup.md                # セットアップ・実行手順
│   ├── architecture.md         # アーキテクチャ設計
│   ├── sequence.md             # シーケンス図
│   └── uniqueness.md           # システムの強み
├── docker-compose.yml          # Docker構成
├── Makefile                    # ビルド・実行コマンド
├── .github/workflows/ci.yml   # CI/CDパイプライン
├── .env.example                # CPU用環境変数テンプレート
└── .env.production.example     # GPU用環境変数テンプレート
```

## 制約・注意事項

- AudioSetの音声は HuggingFace (agkphysics/AudioSet) から取得。yt-dlp/ffmpeg は不要
- Qwen2.5-7B-Instruct のJSON出力は100%安定ではない (パース失敗時は自動リトライ→フォールバック)
- ベースラインの品質はデータ量に依存。500件では誤検知が多く、5万件以上を推奨
- ローカルCPU環境ではLLM推論を無効化 (`LLM_ENABLED=false`) し、シミュレーションモードで動作
- フロントエンドはSSHポートフォワード経由でのアクセスを想定

## ライセンス

本プロジェクトのソースコードは MIT License で公開。

使用しているモデル・データのライセンス:

| 名称 | 種別 | ライセンス | 商用利用 | URL |
|------|------|-----------|---------|-----|
| Google AudioSet | データ | CC BY 4.0 | OK | https://research.google.com/audioset/ |
| CLAP | モデル | Apache 2.0 | OK | https://huggingface.co/laion/larger_clap_music_and_speech |
| Qwen2.5-7B-Instruct | モデル | Apache 2.0 | OK | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| Faiss | ライブラリ | MIT | OK | https://github.com/facebookresearch/faiss |
