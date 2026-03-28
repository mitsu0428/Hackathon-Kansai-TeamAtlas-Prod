.PHONY: setup setup-gpu dev dev-back dev-front pipeline download embed update-all prepare-streams download-test prepare-streams-test build lint clean

# Load .env if it exists (export all variables to sub-processes)
-include .env
export

VENV := backend/.venv/bin
PYTHON := $(VENV)/python
UV := $(shell command -v uv 2>/dev/null)
NOTIFY := ./scripts/notify.sh
export TOKENIZERS_PARALLELISM := false

# ==============================================================
#  セットアップ (初回のみ)
# ==============================================================

setup:  ## ローカル開発 (CPU)
	$(UV) venv --python 3.11 backend/.venv
	$(UV) pip install --python $(PYTHON) -e "backend/.[cpu,dev]"
	cd frontend && npm install
	@$(NOTIFY) "setup 完了 (CPU)" success

setup-gpu:  ## GPUサーバー
	$(UV) venv --python 3.11 backend/.venv
	$(UV) pip install --python $(PYTHON) -e "backend/.[gpu,dev]"
	cd frontend && npm install
	@$(NOTIFY) "setup-gpu 完了" success

# ==============================================================
#  GPUサーバー: データパイプライン (これ1つで全部やる)
# ==============================================================

LIMIT ?= 0
SOURCE ?= balanced

pipeline:  ## データパイプライン一括実行
	@$(NOTIFY) "pipeline 開始 (source=$(SOURCE), limit=$(LIMIT))" success
	@echo "=== Pipeline Start ==="
	$(MAKE) download SOURCE=$(SOURCE) LIMIT=$(LIMIT) && \
	$(MAKE) embed && \
	$(MAKE) update-all && \
	$(MAKE) prepare-streams && \
	$(NOTIFY) "pipeline 完了" success || \
	$(NOTIFY) "pipeline 失敗" failure

# ==============================================================
#  データパイプライン (個別実行)
# ==============================================================

download:  ## AudioSet (HuggingFace) ダウンロード
	$(PYTHON) scripts/download_dataset.py --source $(SOURCE) --limit $(LIMIT) && $(NOTIFY) "download 完了" success || $(NOTIFY) "download 失敗" failure

embed:  ## CLAP埋め込み生成
	@mkdir -p logs
	@(command -v sbatch >/dev/null 2>&1 && sbatch batch/scripts/embed_batch.sh || $(PYTHON) batch/jobs/run_embed.py) && $(NOTIFY) "embed 完了" success || $(NOTIFY) "embed 失敗" failure

update-all:  ## センサー別インデックス構築
	$(PYTHON) batch/jobs/run_update.py --all-sensors && $(NOTIFY) "update-all 完了" success || $(NOTIFY) "update-all 失敗" failure

prepare-streams:  ## ストリーム分離
	$(PYTHON) scripts/prepare_streams.py --from-raw --sensors urban,indoor,park --per-sensor 200 --split-ratio 0.7 && $(NOTIFY) "prepare-streams 完了" success || $(NOTIFY) "prepare-streams 失敗" failure

# ==============================================================
#  ローカルdev用
# ==============================================================

download-test:  ## ESC-50ダウンロード
	$(PYTHON) scripts/download_testdata.py

prepare-streams-test:  ## ESC-50でストリーム作成
	$(PYTHON) scripts/prepare_streams.py --from-esc50

# ==============================================================
#  サーバー起動
# ==============================================================

dev:  ## バックエンド + フロントエンド同時起動
	@$(NOTIFY) "dev サーバー起動" success || true
	@trap 'kill 0' INT TERM EXIT; \
	cd backend && TOKENIZERS_PARALLELISM=false OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES ./.venv/bin/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 --loop asyncio & \
	cd frontend && npm run dev -- --host 0.0.0.0 & \
	wait

dev-back:  ## バックエンドのみ
	cd backend && TOKENIZERS_PARALLELISM=false OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES $(CURDIR)/$(VENV)/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 --loop asyncio

dev-front:  ## フロントエンドのみ
	cd frontend && npm run dev -- --host 0.0.0.0

# ==============================================================
#  品質管理
# ==============================================================

build:  ## フロントエンド本番ビルド
	cd frontend && npm run build

lint:  ## ruff + TypeScriptチェック
	$(VENV)/ruff check backend/src/
	cd frontend && npx tsc --noEmit

clean:  ## 生成データ削除
	rm -rf data/embeddings/* data/index/* data/tmp/*
