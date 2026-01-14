# REOAC Architecture and Theory

這份文件整理目前專案的**數學原理**、**整體 pipeline**、**模型組成**以及實作細節，
方便你後續做實驗對照與報告整理。

---

## 1) 核心概念總覽

REOAC 主要把「擴散式語言模型 (diffusion LM)」和「操作子策略 (operator policy)」結合：

1. **Backbone (E2D2 / MDLM)** 做去噪，輸入是 prompt + mask tokens。  
2. **Relation Energy** 建立 prompt 與生成 token 的對齊矩陣，萃取特徵。  
3. **Critic** 估計某個操作子在當前狀態的價值 (Q value)。  
4. **Actor** 根據 critic 給的 Q 值選擇 operator。  
5. **Operator** 決定「要更新哪些 token」。  
6. **Reward** 來自 GSM8K / MATH 的答案比對。  

這形成一個「**在去噪過程中學習如何修補 token**」的 RL 系統，並把「對齊/關聯」當作
actor/critic 的核心訊號來源。

---

## 2) Backbone 的數學模型（Diffusion LM）

### 2.1 吸收式/遮罩式擴散
擴散模型在文本上用「mask」作為噪聲，從乾淨序列 `x0` 生成 `xt`：

```
q(xt | x0, t) = 依照噪聲比例 alpha_t 將 token 改成 [MASK]
```

E2D2 / MDLM 使用 **吸收式 (absorbing) diffusion**，表示「mask 是吸收狀態」：
一旦 token 被 mask，就必須依靠模型去預測原本 token。

### 2.2 去噪目標
模型輸出 `p_theta(x0 | xt)` 的 logits 或 log-probabilities。
在本專案中：

- E2D2 的 forward 輸出 `denoiser_output`（logits 或 log-probs）。
- 我們用它做 token 更新與 backbone loss。

### 2.3 Noise schedule
用 `sigma(t)` 控制噪聲程度。`configs/reoac_default.yaml` 裡可選：

```
cosine:  sigma(t) = sigma_min + 0.5*(sigma_max - sigma_min)*(1 + cos(pi*t))
linear:  sigma(t) = sigma_max + (sigma_min - sigma_max)*t
```

每個 rollout step 對應一個 `t`，總步數由 `backbone.num_steps` 控制。

### 2.4 生成區與更新規則
- **prompt 區**：原始題目 tokens，透過 `context_mask` 保護，不會被 mask。
- **生成區**：在 prompt 後追加 `gen_len` 個 `[MASK]`，模型只對這段做去噪。
- **operator**：決定要更新哪些 token（例如只更新數字或算符附近）。
- **ensure_update**：若 operator 沒挑到位置會 fallback，避免「完全不更新」。

---

## 3) RL / Operator Policy 的數學

### 3.1 狀態、行動與回饋
在每個 step t：

```
state s_t = (current tokens, prompt_len, timestep, metadata)
action a_t = 選擇一個 operator (O_NUM / O_OP / O_SCOPE / O_BRANCH / O_FAST)
cost c(a_t) = operator 的固定代價 (base_cost)
```

Episode 結束後有 reward `r`（例如 GSM8K 正確為 1，錯誤為 0，或使用 shaped reward）。

總回報：
```
G = r - lambda * sum_t c(a_t)
```

### 3.2 Critic loss
Critic 預測 Q 值 `Q(s_t, a_t)`，用 MSE 拟合回報：

```
L_q = (Q(s_t, a_t) - G)^2
```

### 3.3 Actor loss
Actor 不直接做 policy gradient，而是用 Q 值轉成 soft target：

```
target_probs = softmax(Q / T)
L_pi = - sum_a target_probs[a] * log pi(a | s_t)
```

`T` 是 temperature（在 config 裡是 `losses.actor_temperature`）。

### 3.4 Backbone loss（選擇性）
Backbone 用「被更新位置」的 cross-entropy 做帶權重的 loss：

```
L_b = sum_{pos in update_positions} CE(logits[pos], target[pos]) * advantage
```

其中 `advantage` 是經過 reward/cost 計算的 episode return。

### 3.5 Advantage 與裁切
- `advantage_positive=true` 時會把負值裁成 0，避免 backbone 被負回饋拖走。
- `advantage_clip` 會把極端值裁到 [-clip, clip]，穩定訓練。

---

## 4) 主要模型組成

### 4.1 Backbone (E2D2)
- HF 模型：`kuleshov-group/e2d2-owt`
- Tokenizer：GPT-2 系列（`tokenizer_name_or_path: gpt2`）
- 使用 `context_mask` 保護 prompt，不會被 mask 掉
- 透過 `apply_unmask` 在 operator 指定的 token 位置更新

### 4.2 Actor
- `src/actor/operator_policy.py`
- MLP：Linear -> ReLU -> Dropout -> Linear  
- 輸入維度 = 11 個聚合特徵 + (可選) backbone hidden  
- 輸出 = 5 個 operator logits（對應 O_NUM/O_OP/O_SCOPE/O_BRANCH/O_FAST）

### 4.3 Critic
- `src/critic/rel_energy.py`
- MLP：Linear -> ReLU -> Dropout -> Linear  
- 輸入維度與 Actor 相同  
- 輸出 = 5 個 operator Q 值  
- 可選 `gpu_critic`，用 GPU 計算 relation/features（避免 CPU bottleneck）

### 4.4 Relation Energy 與特徵向量
建立 prompt 與生成 token 的關聯矩陣 `R`：
```
R_ij = similarity(prompt_i, gen_j)
```
`similarity` 使用 token equality、數字匹配、運算符匹配的簡化規則。  
對每個生成 token 以 `softmax` 正規化 prompt 軸，得到類似注意力分佈。  

11 維特徵（`src/critic/aggregators.py`）：
1. key_coverage  
2. key_mass_ratio  
3. key_conditioned_confidence  
4. prompt_entropy_mean  
5. prompt_entropy_p90  
6. temporal_delta  
7. hallucinated_number_rate  
8. number_copy_rate  
9. answer_form_valid  
10. suspect_mean  
11. suspect_p90  

`use_hidden=true` 時，會再拼接 backbone hidden 的 mean pooling 向量。

---

## 5) Pipeline（完整流程）

1. **Tokenize**
   - prompt -> token ids
   - append `gen_len` 個 mask tokens
   - 建立 `attention_mask` 與 `context_mask`

2. **Rollout loop (max_steps 次)**
   - backbone forward -> logits + hidden
   - critic 產生特徵 + Q 值
   - actor 根據 Q 值選 operator
   - apply_operator 決定更新位置
   - apply_unmask 更新 token

3. **Episode 結束**
   - decode 生成區
   - verifier 算 reward（GSM8K/MATH）
   - 存入 buffer

4. **Training 更新**
   - critic: MSE loss
   - actor: soft Q loss
   - backbone: advantage weighted CE

5. **Buffer 與更新節奏**
   - `buffer_size` 控制保留的 episode 數量
   - `num_iterations` × `episodes_per_iter` 決定總 rollout 數
   - `update.critic_steps/actor_steps/backbone_steps` 控制每次更新的步數

---

## 6) Operator 角色

定義在 `src/operators/definitions.py`：

| Operator | 意義 | 行為 |
|----------|------|------|
| O_NUM | 聚焦數字 token | 更新數字相關位置 |
| O_OP | 聚焦運算符 | 更新算符附近位置 |
| O_SCOPE | 更新尾部區塊 | scope_len 控制範圍 |
| O_BRANCH | 分支探索 | 支援 counterfactual |
| O_FAST | 大範圍更新 | scope_len 更長 |

若選到的 operator 沒有 update_positions，
會觸發 fallback（`fallback_operator`）。

---

## 7) Reward / Evaluation

- **GSM8K**：用最後答案抽取/數字比對做 reward（strict 或 shaped）。  
- **MATH**：`eval.math_parser` 可切換 `latex2sympy2` / `sympy` 解析。  
- **reward.mode**：
  - `strict`：正確/錯誤二值  
  - `shaped`：可回傳介於 [0,1] 的分數  

建議在 sample log 內同時記錄 `pred_extracted` 與 `gt_extracted`，避免「抽不到答案卻誤判正確」。

---

## 8) Logging / Viz

每次訓練輸出在：
```
runs/<exp>/<timestamp>/
```

重要檔案：
- `metrics.jsonl`：loss 記錄  
- `samples/*.json`：每 episode 的 prompt / final_text / reward / relation  
- `viz/`：energy map / embedding / loss 圖（可由 scripts/viz_run.sh 產生）  

---

## 9) 多 GPU / DDP 注意事項

DDP 是 data-parallel，**每張 GPU 各跑一批**。  
如果想「時間減半」，必須把總 episodes 平均分配到 GPU。

NCCL timeout 常見原因：
- 某些 rank 沒有產生 backbone loss（導致 allreduce 不一致）  
- 某 rank 太慢（CPU / I/O / reward parsing bottleneck）  
- 某些 rank 早退或拋例外（collective 順序不同）  

建議：
- `rollout.gpu_critic=true` 降低 CPU 字串計算
- `save_relation=false` 或只抽樣寫檔，避免大量 I/O
- 確保所有 rank 都會進入相同的 backward（避免 early-return）
- 先用較小的 `max_steps/gen_len` 排除超長步造成的 timeout
- 可在 slurm 內加上 `NCCL_ASYNC_ERROR_HANDLING=1` 及適當調整 `NCCL_TIMEOUT`

---

如果你需要，我可以再補：
1. 更完整的數學推導或公式版架構圖  
2. 對照 MDLM / E2D2 原論文的變動點  
3. 整套流程的 LaTeX 版整理
