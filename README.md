REOAC 快速指南
================

::: info
三個核心網路：
- **Backbone (E2D2 HF: `kuleshov-group/e2d2-owt`)**：~6e8 量級參數，下載檔約 679 MB。`finetune_mode=lora` 時只訓練 LoRA，`full` 時全參數更新。
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
# 若要 DeepSeekMath 風格的 MATH 評估：pip install latex2sympy2
# 使用 sympy.parsing.latex 時可能需要：pip install antlr4-python3-runtime
```

系統架構與流程（重要）
--------------------
REOAC 由三條主線組成：**Backbone** 產生 logits / hidden、**Critic** 產生特徵與 Q 值、**Actor** 選 operator。
核心流程在 `src/rl/rollout.py` 的 `RolloutEngine`：
1. **Tokenize + 初始化**：把 prompt tokenize，接在後面補 `gen_len` 個 `[MASK]`。  
2. **Batch rollout**：每個 step 將整批 tokens 送進 backbone 一次 forward（`batch_size` 控制）。  
3. **Critic 特徵**：  
   - `fast_critic=false`（預設嚴格）：`RelationalEnergyCritic` 會建立 prompt/gen 的關聯矩陣，再用 `aggregators.py` 計算 11 個特徵。  
   - `fast_critic=true`（高速）：直接使用 hidden（若有）+ 0 向量替代特徵，避免 CPU 大量字串/regex 解析。  
4. **Actor 選 operator**：Actor 在 GPU 上一次推論整批特徵，再依 `selection_mode` 選操作（`argmax` 或 `sample`）。  
5. **Apply operator**：`src/operators/apply_operator.py` 根據 token 字串（或 fast 模式簡化）決定 update mask。  
6. **Unmask / 更新 token**：用 logits 依 operator mask 更新 token（`argmax` 或 `sample`），並進到下一 step。  

關鍵檔案對應：
- **Backbone**：`src/backbone/mdlm_wrapper.py`（E2D2）/ `src/backbone/sedd_wrapper.py`  
- **Rollout**：`src/rl/rollout.py`  
- **Actor / Critic**：`src/actor/operator_policy.py`、`src/critic/rel_energy.py`  
- **特徵計算**：`src/critic/aggregators.py`  
- **Operator**：`src/operators/*`  
- **訓練 loop**：`src/rl/trainer.py`

資料下載
--------
```bash
# 會自動建立 dataset/、runs/ 並下載資料
# GSM8K
bash scripts/prep_data.sh gsm8k configs/reoac_default.yaml all
# MATH
bash scripts/prep_data.sh math configs/mdlm_small_math.yaml all
```
下載後會生成：
- `dataset/processed/gsm8k_{train,test}.jsonl` 或 `math_{train,test}.jsonl`
- `dataset/raw/...` 為原始備份
- `runs/` 供訓練輸出

訓練
----
```bash
# LoRA（預設較安全）
bash scripts/train_reoac.sh configs/reoac_default.yaml lora
# 全參數 fine-tune（顯存/算力更高）
bash scripts/train_reoac.sh configs/reoac_default.yaml full
# 多 GPU（DDP）
REOAC_NPROC=4 bash scripts/train_reoac.sh configs/reoac_default.yaml lora
```
產出位於 `runs/<exp_name>/<timestamp>/`：
- `checkpoints/actor.pt`, `critic.pt`, `backbone.pt` 或 `backbone_lora/`
- `metrics.jsonl`（每 iteration 一筆 loss）
- `samples/*.json`

進度列會顯示：
- `R`：平均 reward
- `C`：平均 cost
- `Lq/Lp/Lb`：critic / actor / backbone loss

訓練細節
--------
- **Rollout buffer**：`buffer_size` 控制保留多少 episodes，`num_iterations * episodes_per_iter` 決定總 rollout。  
- **更新頻率**：`update.critic_steps / actor_steps / backbone_steps` 控制每次 iteration 的反傳次數。  
- **多 GPU**：用 `torchrun` 啟動；`scripts/train_reoac.sh` 會自動偵測 GPU 數，或手動 `REOAC_NPROC=2`。  
- **fast_critic**：跳過關聯矩陣與文字特徵，速度快很多但行為會變（訓練/評估不再與論文型特徵一致）。  
- **fast_operator**：在 fast_critic 下進一步跳過 token 字串解析（更快但改動更大）。
- **ensure_update / fallback_operator**：若選到的 operator 沒有任何 update_positions，會自動改成 fallback（避免生成一直是 `<mask>`）。

推論 / 評估
-----------
```bash
# 使用最新 checkpoints（第三個參數可指定 checkpoints 資料夾，省略則用當前 config）
bash scripts/eval_gsm8k.sh configs/reoac_default.yaml runs/<exp>/<ts>/checkpoints
bash scripts/eval_math.sh  configs/mdlm_small_math.yaml runs/<exp>/<ts>/checkpoints
```
結果輸出至 `runs/eval_gsm8k.json` / `runs/eval_math.json`。

評估標準（與訓練一致）
----------------------
訓練的 reward 與評估的 accuracy 使用**同一套 verifier**：
- **GSM8K**：`src/eval/verifier_gsm8k.py`  
  - 取 `####` 之後、或文字中的最後一個數字做比對。  
  - 這與大多數 GSM8K baseline 一致（取最後一個數字）。  
- **MATH**：`src/eval/verifier_math.py`  
  - 優先取 `\\boxed{...}`；否則取最後盒內內容。  
  - 先嘗試 `sympy` 等價（若有安裝），否則用數值/字串相等。  

與 LLaDOU / DeepSeekMath 等常見做法的差異：
- GSM8K 方式基本一致。  
- MATH 部分可切換 parser：`eval.math_parser` 支援 `latex2sympy2`、`sympy_latex`、`sympy`、`auto`。  
  DeepSeekMath/不少論文會用更強的 latex→sympy 正規化（例如 `latex2sympy2` / `sympy.parsing.latex`）。  
  若要**完全對齊**，建議使用 `latex2sympy2`（並確保套件可用）。  

主要可調參數（configs/*）
------------------------
- **資料**：`dataset.train_path / eval_path`
- **Rollout**：`max_steps`、`gen_len`、`branch_steps`、`branch_k`、`cost_lambda`
- **Batch / 速度**：`rollout.batch_size`、`rollout.fast_critic`、`rollout.fast_operator`
- **穩定性**：`rollout.ensure_update`、`rollout.fallback_operator`
- **LoRA / Full**：`finetune_mode`（`lora` or `full`）；`lora.r/alpha/dropout/target_modules`
- **學習率**：`optim.actor_lr / critic_lr / backbone_lr`
- **更新步數**：`update.critic_steps / actor_steps / backbone_steps`
- **訓練長度**：`num_iterations`、`episodes_per_iter`、`buffer_size`
- **效能**：`backbone.torch_dtype`（`bfloat16`/`float16`）
- **Checkpoint**：`logging.checkpoint_interval`、`logging.save_backbone`、`logging.save_actor_critic`
- **評估對齊**：`eval.math_parser`（`latex2sympy2`/`sympy_latex`/`sympy`/`auto`）
- **Reward 形狀化**：`reward.mode`（`strict`/`shaped`；訓練可用 shaped、評估固定 strict）

注意事項
--------
- E2D2 HF 模型需要 **GPU + CUDA** 才能搭配 flash-attn（否則可移除 flash-attn，仍可在 CPU/GPU 跑但較慢）。
- `tokenizer_source: e2d2` 會直接使用 HF AutoTokenizer（e2d2 沒有 dataloader）；若需要舊版 MDLM 的 tokenizer 邏輯可改成 `mdlm`，或直接用 `auto`。
- `full` 模式顯存需求高，建議先用 `lora` 確認流程與 loss 下降，再切換 `full`。
- `metrics.jsonl` 可檢查 loss 是否隨 iteration 下降；若全為 0 或不變，表示沒有有效學習訊號。
- 若要消除 E2D2 的載入 warning，可在 config 設定 `backbone.suppress_load_warnings: true`。
- 多 GPU 下若 checkpoint 寫入很慢，可能導致 NCCL timeout；可提高 `logging.checkpoint_interval` 或關閉 `logging.save_backbone`。
