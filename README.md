REOAC 快速指南
================

::: info
三個核心網路：
- **Backbone (MDLM HF: `kuleshov-group/mdlm-owt`)**：~6e8 量級參數，下載檔約 679 MB。`finetune_mode=lora` 時只訓練 LoRA，`full` 時全參數更新。
- **Actor (MLP policy)**：輸入=11 個聚合特徵 + (可選) backbone hidden，輸出 5 個 operator logits，參數量數十萬級。
- **Critic (MLP Q head)**：同 Actor 輸入，輸出 5 個 operator Q 值，參數量數十萬級。
:::

環境與安裝
----------
```bash
# 建議有 GPU + CUDA，flash-attn 需搭配對應 torch 版本
pip install -r requirements.txt
# 如需 flash-attn（GPU）：uv pip install torch --index-url https://download.pytorch.org/whl/cu121
# 再裝 flash-attn：uv pip install flash-attn --no-build-isolation
```

資料下載
--------
```bash
# GSM8K
bash scripts/download_data.sh gsm8k configs/reoac_default.yaml all
# MATH
bash scripts/download_data.sh math configs/mdlm_small_math.yaml all
```
下載後會生成：
- `data/processed/gsm8k_{train,test}.jsonl` 或 `math_{train,test}.jsonl`
- `data/raw/...` 為原始備份

訓練
----
```bash
# LoRA（預設較安全）
bash scripts/train_reoac.sh configs/reoac_default.yaml lora
# 全參數 fine-tune（顯存/算力更高）
bash scripts/train_reoac.sh configs/reoac_default.yaml full
```
產出位於 `runs/<exp_name>/<timestamp>/`：
- `checkpoints/actor.pt`, `critic.pt`, `backbone.pt` 或 `backbone_lora/`
- `metrics.jsonl`（每 iteration 一筆 loss）
- `samples/*.json`

推論 / 評估
-----------
```bash
# 使用最新 checkpoints（第三個參數可指定 checkpoints 資料夾，省略則用當前 config）
bash scripts/eval_gsm8k.sh configs/reoac_default.yaml runs/<exp>/<ts>/checkpoints
bash scripts/eval_math.sh  configs/mdlm_small_math.yaml runs/<exp>/<ts>/checkpoints
```
結果輸出至 `runs/eval_gsm8k.json` / `runs/eval_math.json`。

主要可調參數（configs/*）
------------------------
- **資料**：`dataset.train_path / eval_path`
- **Rollout**：`max_steps`、`gen_len`、`branch_steps`、`branch_k`、`cost_lambda`
- **LoRA / Full**：`finetune_mode`（`lora` or `full`）；`lora.r/alpha/dropout/target_modules`
- **學習率**：`optim.actor_lr / critic_lr / backbone_lr`
- **更新步數**：`update.critic_steps / actor_steps / backbone_steps`
- **訓練長度**：`num_iterations`、`episodes_per_iter`、`buffer_size`

注意事項
--------
- MDLM HF 模型需要 **GPU + CUDA** 才能搭配 flash-attn（否則可移除 flash-attn，仍可在 CPU/GPU 跑但較慢）。
- `tokenizer_source: mdlm` 會使用 third_party/mdlm 的 tokenizer 邏輯；若遇到不支援，可改回 `auto`。
- `full` 模式顯存需求高，建議先用 `lora` 確認流程與 loss 下降，再切換 `full`。
- `metrics.jsonl` 可檢查 loss 是否隨 iteration 下降；若全為 0 或不變，表示沒有有效學習訊號。
