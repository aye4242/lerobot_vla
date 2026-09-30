# Pi0.5 架构深度解析

> 个人学习笔记，基于 `src/lerobot/policies/pi05/` 源码整理。
> Pi0.5 是 Pi0 的工程精修版本，骨架不变，核心改动集中在信息注入方式与深度。

---

## 1. 项目目标

Pi0.5 是针对 **finetuning 与实际部署** 优化的 VLA 模型，在 Pi0 基础上解决三个具体问题：机器人状态只有 Expert 能"看到"、时间步 t 只在第一层生效、以及跨机器人 finetuning 时归一化不稳定。

| 项目 | 说明 |
|---|---|
| 输入 | 摄像头图像（224×224）+ 自然语言指令 + **离散化后的机器人状态（注入 Prefix 文字 token）** |
| 输出 | 50帧动作序列（50×32维），一次规划50步后重新观测并规划 |
| 核心机制 | Flow Matching（从噪声去噪到动作）+ Shared Attention + **AdaRMS（时间步穿透每层）** |
| 相比 Pi0 | 骨架完全不变；信息流入方式改变：state 进 Prefix、时间步进每层 LayerNorm |

---

## 2. 基础组件

### 2.1 PaliGemma（视觉语言主干）

PaliGemma = **SigLIP**（看图）+ **模态投影层**（连接）+ **Gemma-2B**（理解语言和状态）

#### SigLIP（视觉编码器）

- 输入：224×224 RGB 图像
- 处理：切成 16×16 = **256 个 patch** → 线性投影 → Transformer 自注意力
- 输出：**256 个向量，每个 1024 维**
- Pi0.5 中：可冻结（`freeze_vision_encoder`），预训练特征对机器人场景已足够

#### 模态投影层

- SigLIP 1024维 → Gemma 所需的 **2048维**
- Pi0.5 训练时可选冻结

#### Gemma-2B（语言理解，Pi0.5 中新增：**直接理解离散化状态**）

| 参数 | 值 |
|---|---|
| 层数 | 18层 Transformer |
| hidden_size | 2048 |
| MLP dim | 16384 |
| num_heads | 8，head_dim=256 |
| 输入序列长度 | 最多 456（256图像patch + 200文字+状态token） |

Pi0.5 将 `tokenizer_max_length` 从 48 提升至 **200**，以容纳机器人状态的数字序列。VLM 现在能直接处理状态 token，而不是靠 Expert 间接感知。

### 2.2 动作 Expert（Gemma-300M + AdaRMS）

| 参数 | 值 |
|---|---|
| 层数 | 18层（与 PaliGemma 一一对应，用于 Shared Attention） |
| hidden_size | 1024 |
| MLP dim | 4096 |
| num_heads | 8，head_dim=256 |
| embed_tokens | None（不处理文字token，只处理动作） |
| **AdaRMS** | **每层 pre-norm 和 post-norm 均由时间步 t 的嵌入调制** |

与 Pi0 的 Expert 相比，唯一结构改动是把所有 LayerNorm 替换为 `PiGemmaRMSNorm`（支持条件调制），通过 `use_adarms=[False, True]` 配置：VLM 使用标准 RMSNorm，Expert 全部使用 AdaRMS。

---

## 3. 系统整体架构

### 3.1 序列设计：Prefix / Suffix（Pi0 vs Pi0.5 关键差异）

```
Pi0 序列设计：
  Prefix（VLM 处理，304个向量，2048维）：
    [图像 patch × 256] + [语言 token × 48]
  Suffix（Expert + VLM 联合处理，51个向量，1024维）：
    [state_proj(state) × 1] + [动作 token × 50]
                               ↑ 状态在这里，VLM 看不到

Pi0.5 序列设计：
  Prefix（VLM 处理，最多 456个向量，2048维）：
    [图像 patch × 256] + [语言+状态 token × 最多200]
                                       ↑ 状态在这里，VLM 直接处理
  Suffix（Expert + VLM 联合处理，50个向量，1024维）：
    [动作 token × 50]（无 state token，state_proj 被移除）
```

### 3.2 状态离散化流程（Pi0.5 新增，Processor 完成）

```
机器人关节状态 [B, 32]（浮点）
    ↓ Quantile 归一化 → [-1, 1]
    ↓ np.digitize(bins=np.linspace(-1,1,257)[:-1]) - 1
    ↓ 得到 0~255 的整数序列 [128, 64, 230, ...]
    ↓ 拼入文字 prompt：
      "Task: pick up the cup, State: 128 64 230 ...; Action: "
    ↓ PaliGemmaTokenizer → token_ids [B, max_length=200]
    ↓ embed_language_tokens() → [B, N, 2048]
    ↓ 与图像向量 concat → Prefix [B, 256+N, 2048]
```

每个状态维度占用约 1~3 个 token（取决于数字位数），32维状态约需 50~100 个 token，加上任务描述共约 70~130 个 token，因此 max_length=200 通常足够。

### 3.3 共享注意力机制（Shared Attention）—— 与 Pi0 相同

Pi0.5 保持与 Pi0 完全相同的 Shared Attention 结构：PaliGemma 的18层和 Expert 的18层**一一对应 zip**，每层合并计算注意力。

```
第 i 层（compute_layer_complete）：

PaliGemma 第i层：
  Q_p = prefix_emb × W_Q_p    K_p = prefix_emb × W_K_p    V_p = prefix_emb × W_V_p

Expert 第i层：
  Q_e = suffix_emb × W_Q_e    K_e = suffix_emb × W_K_e    V_e = suffix_emb × W_V_e

拼接（沿序列维度）：
  Q_all = [Q_p | Q_e]    K_all = [K_p | K_e]    V_all = [V_p | V_e]

统一做注意力计算（附加掩码）：
  Attn_all = softmax(Q_all × K_all^T / √d) × V_all

拆分输出：
  prefix_out ← Attn_all[:prefix_len]
  suffix_out ← Attn_all[prefix_len:]

Pi0.5 新增（对 Expert 层）：
  hidden_states, gate = AdaRMS(x, cond=adarms_cond)  ← t 调制 norm
  residual = x + sublayer_out * gate                  ← t 控制残差权重
```

### 3.4 注意力掩码规则（Pi0.5：state 在 Prefix，掩码规则随之改变）

```
               图像patch  状态+语言token  动作token[i]
图像 patch        ✓            ✓               ✗
状态+语言token     ✓            ✓               ✗
动作 token[i]     ✓            ✓            ✓(j≤i)
```

相比 Pi0 的差异：Pi0 的状态 token 在 Suffix，图像和语言看不到状态；Pi0.5 的状态 token 在 Prefix，**图像、语言、状态三者双向互看**，动作 token 可以 attend 到状态（单向），状态不能 attend 到动作（保证 KV Cache 有效）。

---

## 4. 核心创新点

### 4.1 AdaRMS —— 时间步 t 穿透 Expert 全部18层

**Pi0 的问题**：时间步 t 只在 Suffix 输入层被融合一次，之后每一层的 LayerNorm 是标准的、与 t 完全无关的 `RMSNorm(x)`。Flow Matching 要求模型对不同 t（噪声水平）输出不同速度方向，但 t 的信息随层数加深被稀释，深层 Expert 对 t 不敏感。

**Pi0.5 的改法**：

```
时间步嵌入路径（与 action embedding 完全解耦）：

sinusoidal(t) [B, hidden]
  → time_mlp_in:  Linear(hidden → hidden)
  → SiLU
  → time_mlp_out: Linear(hidden → hidden)
  → adarms_cond [B, hidden]  ← 这个向量广播到 Expert 全部 18 层

Expert 每一层内部（pre-norm 和 post-norm 各一次）：
  scale, shift, gate = Linear(adarms_cond, hidden × 3)  ← 每层独立投影
  split → scale [B,1,d], shift [B,1,d], gate [B,1,d]

  h = RMSNorm(x) * (1 + scale) + shift   ← t 调制归一化行为
  h = sublayer(h)
  x = x + h * gate                        ← t 控制残差权重
```

**初始化设计**：`Linear.weight` 全部初始化为零，训练开始时 scale=0, shift=0, gate=0，等价于标准 RMSNorm + 标准残差。模型从稳定的 Pi0 起点出发，逐渐学习 t 的调制策略。

**与 Pi0 action_time_mlp 的本质区别**：

| | Pi0（输入调制） | Pi0.5（全局调制 AdaRMS） |
|---|---|---|
| t 的作用位置 | 修改输入 token 的**值** | 修改每层 LayerNorm 的**行为** |
| 深层感知 t | 间接，靠 token 值传播 | 直接，每层独立线性投影 |
| action embedding | t 与 action 加和融合 | action embedding 纯净，t 独立流动 |
| 参数开销 | 一个 2-layer MLP | 每层各2个 Linear(hidden→hidden×3)，共36个 |

### 4.2 状态离散化注入 Prefix —— VLM 直接理解机器人状态

**Pi0 的问题**：状态通过 `state_proj: Linear(32, 1024)` 变成 Expert Suffix 中的一个向量 token，VLM 在 Prefix 阶段完全看不到状态。VLM 做语言-图像推理时对"当前机器人姿态"一无所知，只靠 Shared Attention 间接感知。

**Pi0.5 的改法**：

```
步骤1：量化（Processor 中完成）
  state [B, 32] → Quantile 归一化 → [-1, 1]
  → np.digitize(bins=256) → 整数序列 0~255

步骤2：序列化为文字
  "Task: pick up the cup, State: 128 64 230 45 ...; Action: "

步骤3：进入 VLM
  → PaliGemmaTokenizer → token_ids
  → embed_language_tokens() → [B, N, 2048]
  → 拼入 Prefix，与图像 patch 一起做双向注意力

步骤4：Suffix 无状态
  embed_suffix(noisy_actions, timestep)  ← 无 state 参数
  state_proj 被移除
```

**效果**：VLM 的预训练能力（数字推理、数值比较、文字理解）直接作用于机器人状态，状态与语言指令、图像观测在 Prefix 内融合，动作 token 读到的是 VLM 处理过的状态表示而非原始线性投影。

### 4.3 Quantile 归一化 —— 跨机器人 finetuning 稳定性

**Pi0 的问题**：ACTION 和 STATE 使用 IDENTITY 归一化（不处理），假设数据在合理范围内。Pi0 面向大规模预训练，数据分布已知。Pi0.5 面向 finetuning，不同机器人的动作/状态值域差异巨大（弧度 vs 度数 vs 归一化开合量），IDENTITY 导致各维度损失量级悬殊。

**Quantile 归一化机制**：

```
训练集统计：对每个维度收集所有样本，记录分位点
  0%分位  →  -1.0
  50%分位 →   0.0（中位数）
 100%分位 →  +1.0

推理时：给定新值 x，查其在训练集中的分位排名，线性映射到 [-1, 1]

效果示意：
  原始分布（95%集中在小范围，5%极端值）：

  密│████████████████░░░░│
  度│                    │
   └───────────────────→ 值
   -2.0  -0.5   0.5   2.0

  归一化后（均匀铺满 [-1, 1]）：

  密│████████████████████│
  度│                    │
   └───────────────────→ 值
   -1.0        0       1.0
```

**关键性质**：极端值只被"推"到 ±1.0 边界，不会像 Z-score 那样撑大 σ 进而压缩正常值的精度。对重尾分布（末端执行器偶尔大位移）更鲁棒。

| | IDENTITY（Pi0） | Z-score | Quantile（Pi0.5） |
|---|---|---|---|
| 不同机器人值域 | 完全不可比 | 依赖 σ，受极端值影响 | 均映射到 [-1,1] |
| 极端值处理 | 原样保留 | 可能超出 ±3σ | 压缩到边界 |
| 正常值分辨率 | 取决于原始值域 | σ 被极端值撑大时下降 | 始终保持 |
| finetuning 稳定性 | 差（跨机器人） | 中 | 好 |

### 4.4 工程改进：RTC、相对动作、Gradient Checkpointing

**Real-Time Chunking（RTC）**：
- Pi0 每次执行完整 50 帧 chunk 后才重新推理；Pi0.5 支持在 chunk 执行过程中提前触发下一次推理，减少实际控制延迟
- 通过 `rtc_config: RTCConfig` 配置，`supports_rtc() → True`

**相对动作（use_relative_actions）**：
- Pi0 只输出绝对关节角度；Pi0.5 可以将动作转换为相对当前状态的增量
- `relative_exclude_joints=["gripper"]`：夹爪保持绝对值，其余关节转相对值
- 减少对绝对坐标的依赖，finetuning 时泛化更好

**Gradient Checkpointing**：
- `gradient_checkpointing: bool = False`，开启后用重计算换显存
- 对 PaliGemma language layers、vision tower、Expert 均支持

---

## 5. 训练 vs 推理结构对比

| 方面 | 训练 | 推理 |
|---|---|---|
| clean_action | **有**（来自数据集） | **无**（目标） |
| 去噪次数 | 随机取**1个t**，做1次前向 | 从t=1.0→0.0，做**10次**前向 |
| adarms_cond | 由当前随机 t 生成 | 由当前 Euler 步的 t 生成，每步更新 |
| KV Cache | **不使用**（use_cache=False） | **使用**（Prefix 含状态，只算1次） |
| state 处理 | Processor 离散化并入 prompt，一次完成 | 同训练，Prefix KV Cache 复用 |
| Suffix 内容 | `action_in_proj(x_t)`，纯动作 token | 同训练，每步重建 |
| 输出用途 | 计算 MSE loss，反向传播 | Euler 步进更新 x_t，最终执行 |

---

## 6. 完整端到端流程（含 Flow Matching）

### 训练

```
① 编码图像
   image [B,H,W,3] → SigLIP → 模态投影
   → img_emb [B, 256, 2048]

② 编码状态 + 语言（state 进 Prefix，Processor 完成）
   state [B,32]   → Quantile 归一化 → digitize(256 bins)
                  → "Task:{task}, State:{ints};\nAction: "
   language (str) → 与上方字符串拼接 → PaliGemma Tokenizer
                  → lang+state_emb [B, N, 2048]
   Prefix = cat(img_emb, lang+state_emb)  [B, 256+N, 2048]

③ Flow Matching 构造带噪动作（训练专有）
   noise ~ N(0,1)                  [B, 50, 32]
   t     ~ Beta(1.5, 1.0)
   x_t   = t×noise + (1-t)×action  ← 带噪中间态
   u_t   = noise - action           ← 训练目标（速度场）

④ 编码带噪动作 + 生成时间调制信号
   adarms_cond = time_mlp(sinusoidal(t))  ← 注入每层 AdaRMS
   x_t → action_in_proj: Linear(32→1024)
   → Suffix [B, 50, 1024]

⑤ 模型前向（Shared Attention，18 层）
   [Prefix | Suffix]
   → PaliGemma ↔ Expert 逐层 Shared Attention
   → Expert 每层 AdaRMS(x, adarms_cond) 由 t 调制
   → suffix_out[-50:] → action_out_proj: Linear(1024→32)
   → v_t [B, 50, 32]

⑥ 计算损失
   Loss = MSE(v_t, u_t)
```

### 推理

```
image + language + state（state 已离散化进 Prefix 文字 token）
    ↓ embed_prefix()
Prefix KV Cache（一次计算，含状态信息，缓存18层 KV）

x_t = N(0,1)  [t=1.0]

循环10次（Euler ODE）：
  t: 1.0 → 0.9 → ... → 0.1
  ┌──────────────────────────────────────────────────────────┐
  │ adarms_cond = time_mlp(sinusoidal(t))                    │
  │ embed_suffix(x_t, t)                                     │
  │   → action_in_proj(x_t) → Suffix [B,50,1024]            │
  │ Shared Attention(Suffix Q, Prefix KV)                    │
  │   Expert 每层 AdaRMS 由 adarms_cond 调制                 │
  │   → suffix_out → action_out_proj → v_t [B,50,32]        │
  └──────────────────────────────────────────────────────────┘
  x_t = x_t + (-0.1) × v_t

x_0 [B,50,32] = 干净动作 → 反 Quantile 归一化 → 执行
```

---

## 7. 数据流

### 6.1 训练数据流

```
数据集样本：(image, instruction, robot_state, clean_action)

image [B, H, W, 3]
  → _preprocess_images()：resize+pad 到 224×224，归一化到 [-1, 1]
  → PaliGemmaWithExpertModel.embed_image()：SigLIP → [B, 256, 1024]
  → 模态投影层 → [B, 256, 2048]

robot_state [B, 32]
  → Quantile 归一化 → [-1, 1]
  → np.digitize(bins=256) → 整数序列
  → 拼入 prompt 字符串："Task: {task}, State: {state_str};\nAction: "
  → Tokenizer → token_ids [B, ≤200]
  → embed_language_tokens() → [B, N, 2048]

Prefix = concat(image_emb, lang+state_emb) → [B, 256+N, 2048]

noise ~ N(0,1) [B, 50, 32]
time ~ Beta(1.5, 1.0) [B]

x_t = time[:,None,None] × noise + (1-time[:,None,None]) × clean_action
u_t = noise - clean_action   ← 训练目标（解析速度）

adarms_cond = time_mlp_out(SiLU(time_mlp_in(sinusoidal(time))))  [B, 1024]

x_t → action_in_proj: Linear(32 → 1024) → [B, 50, 1024]

Suffix = action_emb → [B, 50, 1024]   ← 无 state，无时间步融合

att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
position_ids = cumsum(pad_masks) - 1

[Prefix | Suffix] + adarms_cond → 18层 Shared Attention（Expert 每层 AdaRMS）
  → suffix_out [B, 50, 1024]

suffix_out → action_out_proj: Linear(1024 → 32) → v_t [B, 50, 32]

Loss = MSE(v_t, u_t)[:, :, :actual_action_dim]   标量
```

### 6.2 推理数据流

```
输入：(image, instruction, robot_state)

Step 0：Prefix 单次计算（含状态）
  image + (state 离散化后的) lang prompt → embed_prefix()
    → prefix_embs [B, 256+N, 2048]
  → PaliGemmaWithExpertModel.forward([prefix_embs, None], use_cache=True)
  → past_key_values（18层 KV 缓存）

adarms_cond_t = time_mlp(sinusoidal(t))   ← 每步重算，t 变化

初始化：x_t = N(0,1) [B, 50, 32]（t=1.0）

循环 10 次（Euler ODE）：
  time = 1.0 → 0.9 → ... → 0.1

  denoise_step(prefix_pad_masks, past_key_values, x_t, time)：
    → adarms_cond = time_mlp(sinusoidal(time))
    → embed_suffix(x_t, time) → suffix_embs [B, 50, 1024]
    → 构建 full_att_2d_masks（Suffix Q attend Prefix KV）
    → PaliGemmaWithExpertModel.forward(
          inputs_embeds=[None, suffix_embs],
          past_key_values=past_key_values,
          adarms_cond=[None, adarms_cond])
      # Expert 处理 Suffix，每层用 adarms_cond 调制 LayerNorm
      # Prefix KV 从缓存读取
    → suffix_out[-50:] → action_out_proj → v_t [B, 50, 32]

  x_t = x_t + (-0.1) × v_t   # Euler 步进

输出：x_t [B, 50, 32]（t=0，去噪完成）
  → 反 Quantile 归一化（unnormalize）
  → 前 n_action_steps 帧放入 action_queue
  → select_action() 每次弹出1帧发给机器人
  → 队空后重新调用 predict_action_chunk()（Receding Horizon Control）
```

---

## 7. 冻结策略

| 组件 | 默认 | 原因 |
|---|---|---|
| SigLIP | 可冻结（`freeze_vision_encoder`） | 预训练视觉特征已足够，机器人数据量不足以改善 |
| 模态投影层 | 随 PaliGemma | 与 SigLIP 绑定 |
| Gemma-2B 语言层 | 可冻结（`train_expert_only`） | 语言理解能力来自大规模预训练，小数据微调易遗忘 |
| 动作 Expert（全部层+AdaRMS） | **始终训练** | 从随机初始化，必须从头学；AdaRMS 初始化为零不影响起点 |
| action_in_proj / action_out_proj | **始终训练** | Pi0.5 新增模块，无预训练权重 |
| time_mlp_in / time_mlp_out | **始终训练** | 生成 adarms_cond，Pi0.5 核心参数 |

**Pi0.5 典型 finetuning 策略（`train_expert_only=True`）**：

```
冻结：PaliGemma 全部（VLM 语言层 + 视觉层）
训练：Expert 18层（含 AdaRMS 参数）+ action_in_proj + action_out_proj + time_mlp
效果：保留 VLM 预训练的语言理解和视觉理解（含对状态 token 的理解）
      Expert 从头学习基于 VLM 特征生成动作的能力
```

---

## 8. 关键配置参数（`PI05Config`）

```python
# 模型规模
paligemma_variant     = "gemma_2b"    # PaliGemma 骨干（可选 "gemma_300m" 轻量版）
action_expert_variant = "gemma_300m"  # 动作 Expert 规模（可选 "gemma_2b"）

# 动作维度
max_state_dim  = 32
max_action_dim = 32

# 推理控制
chunk_size           = 50    # 一次预测 50 帧
n_action_steps       = 50    # 执行 50 帧后重新规划
num_inference_steps  = 10    # 去噪步数

# 时间步采样（与 Pi0 完全相同）
time_sampling_beta_alpha = 1.5
time_sampling_beta_beta  = 1.0
time_sampling_offset     = 0.001
time_sampling_scale      = 0.999
min_period = 4e-3            # sinusoidal 编码最小周期
max_period = 4.0             # sinusoidal 编码最大周期

# Pi0.5 新增：归一化（Pi0 用 IDENTITY，Pi0.5 用 QUANTILES）
normalization_mapping = {
    "VISUAL": NormalizationMode.IDENTITY,
    "STATE":  NormalizationMode.QUANTILES,   ← 关键差异
    "ACTION": NormalizationMode.QUANTILES,   ← 关键差异
}

# Pi0.5 新增：相对动作
use_relative_actions    = False              # True = 输出相对当前状态的增量
relative_exclude_joints = ["gripper"]        # 夹爪保持绝对值

# Pi0.5 新增：Tokenizer 长度（Pi0=48，Pi0.5=200，容纳状态序列）
tokenizer_max_length = 200

# Pi0.5 新增：工程优化
gradient_checkpointing = False    # 重计算换显存
compile_model          = False    # torch.compile 加速
rtc_config             = None     # Real-Time Chunking 配置

# 冻结控制
freeze_vision_encoder = False
train_expert_only     = False

# 优化器
optimizer_lr             = 2.5e-5
optimizer_grad_clip_norm = 1.0
scheduler_warmup_steps   = 1_000
scheduler_decay_steps    = 30_000
scheduler_decay_lr       = 2.5e-6
```

---

## 9. 类结构速查

```
PI05Policy  （src/lerobot/policies/pi05/modeling_pi05.py）
  ├── select_action(batch)        → 单步动作（维护 action_queue）
  ├── predict_action_chunk(batch) → 完整推理一次（调用 sample_actions）
  └── forward(batch)              → 训练前向，返回 loss（支持 reduction="none"）

  └── PI05Pytorch
        ├── forward(...)          → Flow Matching 训练（MSE loss）
        ├── sample_actions(...)   → 10 步去噪推理
        ├── denoise_step(...)     → 单步去噪（复用 KV Cache + adarms_cond）
        ├── embed_prefix(...)     → 图像 + 语言（含状态） → prefix_embs
        └── embed_suffix(...)     → noisy_actions + timestep → suffix_embs + adarms_cond
              注：Pi0.5 无 state 参数，state 已在 Prefix 中

        └── PaliGemmaWithExpertModel（use_adarms=[False, True]）
              ├── embed_image(...)           → SigLIP 前向
              ├── embed_language_tokens(...) → Gemma Embedding
              └── forward(inputs_embeds=[prefix, suffix], adarms_cond=[None, cond])
                    ├── prefix only   → 返回 KV Cache（推理 Step 0）
                    ├── suffix only   → 复用 KV Cache（推理 Step 1~10）
                    │     Expert 每层：AdaRMS(x, cond=adarms_cond) → scale/shift/gate
                    └── prefix+suffix → Shared Attention（训练）
                          compute_layer_complete(inputs_embeds, ..., adarms_cond=[None, cond])

Processor（src/lerobot/policies/pi05/processor_pi05.py）
  └── Pi05PrepareStateTokenizerProcessorStep
        → 状态 Quantile 归一化 → 离散化 → 拼入 prompt 字符串
  └── TokenizerProcessorStep
        → "google/paligemma-3b-pt-224" tokenizer，max_length=200，padding_side="right"
```

---

## 10. 本地复现与实验结果

已完成 Pi0.5 在本地 LIBERO MuJoCo 环境中的端到端复现和批量评估。

### 10.1 评估配置

| 项目 | 配置 |
|---|---|
| 模型 | `pi05_libero_finetuned`（本地 checkpoint） |
| checkpoint 文件 | 7,473,096,344 字节（约 7.47 GB） |
| checkpoint tensor | 812 个 tensor，共存储 3,616,757,520 个参数元素 |
| 环境/任务 | `libero_object` / task ID `0`（pick up the alphabet soup and place it in the basket） |
| 仿真平台 | LIBERO（基于 robosuite / MuJoCo） |
| 机械臂模型 | Franka Panda：单臂 7 自由度 + Panda 双指夹爪；控制器为 LIBERO `OSC_POSE` / relative OSC |
| seed | 1000，与 Pi0 使用相同初始状态序列 |
| 回合数 | 20 |
| 最大步数 | 280 steps/episode |
| `n_action_steps` | 10（官方 LIBERO 评估设置；每次执行 10 个动作后重新规划） |
| Flow Matching 推理步数 | 10 |
| 数据类型 | `float32` |
| 相机 | LIBERO 默认相机，空相机配置 `empty_cameras=1` |

### 10.2 实测结果（N=20）

评估输出：`outputs/eval/pi05_libero_object_task0_n20_final/eval_info.json`

| 指标 | 实测值 |
|---|---:|
| 成功序列 | `[T,F,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T]` |
| 成功次数 | 19/20 |
| 成功率 | **95.0%** |
| Wilson 95% 置信区间 | **76.4%-99.1%** |
| 平均总 reward | **0.95** |
| 平均最大 reward | **0.95** |
| 总评估时间 | **6969.50 s（约 116.16 min）** |
| 平均每回合时间 | **348.48 s（约 5 分 48 秒）** |
| 评估视频 | `outputs/eval/pi05_libero_object_task0_n20_final/videos/libero_object_0/` |

此前 N=5 初测为 80.0%（4/5），总耗时 1539.09 秒、平均每回合 307.82 秒，保留用于复现链路的历史记录；扩展到相同 seed 序列的 N=20 后为 95.0%（19/20）。N=20 比 N=5 更能抑制偶然波动，但置信区间仍不算窄，而且这里只覆盖单个 LIBERO task，不能当作完整 LIBERO benchmark 成绩。

Pi0.5 与 Pi0 使用相同 task、seed 和 20 个初始状态进行配对评估。Pi0 为 13/20，因此 Pi0.5 高 **30 个百分点**；Pi0.5 单独成功 7 回合、Pi0 单独成功 1 回合，Exact McNemar 双侧检验 `p≈0.0703`。当前结果支持明显优势趋势，但尚不足以宣称传统 `p<0.05` 意义下的统计显著性。

### 10.3 Viewer 复现现象与覆盖边界

Pi0.5 已在同一 LIBERO Viewer 中完成实时推理和 MuJoCo 可视化。`libero_object/task 0` 的 instruction 为 `pick up the alphabet soup and place it in the basket`；运行日志出现 LIBERO success detection，说明模型可以完成该单物品抓取放置任务。Viewer 使用 `n_action_steps=10`，用户日志中的完整 chunk 推理通常约为 8.5–9.2 秒；首次 chunk 会因模型和 CUDA 初始化明显更慢。这是本机交互日志的近似值，不是独立基准延迟。

同一个 `pi05_libero_finetuned` checkpoint 可以切换 `libero_spatial`、`libero_object`、`libero_goal` 和 `libero_10` 四个 suite；但目前正式批量数字仅覆盖 `libero_object/task 0` N=20。其余 suite 已完成启动器和任务切换支持，不应把“能启动/能显示场景”写成四套 suite 的成功率。

---

## 11. 简历版本

> `libero_object` task 0 已完成 N=20 评测；完整套件和多 seed 结果仍待补充。

### One-line

在 LeRobot 框架内部署 Pi0.5（PaliGemma-2B + Gemma-300M Expert）推理流程，核心改进包括：状态离散化注入 Prefix（VLM 直接理解机器人状态）、AdaRMS 时间步穿透 Expert 全18层、以及 Quantile 归一化支持跨机器人 finetuning；在 LIBERO `libero_object` task 0 上进行 N=20 评估，取得 **95.0%（19/20）成功率**。

### 3-bullet

- **部署 Pi0.5 VLA 推理**：在 LeRobot 内跑通 Pi0.5 端到端推理——状态离散化为 256 级整数序列注入 Prefix（tokenizer_max_length=200），PaliGemma-2B 直接理解机器人状态；10步 Flow Matching Euler ODE 生成50帧动作，Expert 每层 AdaRMS 由时间步 t 调制归一化的 scale/shift/gate；LIBERO `libero_object` task 0（N=20）成功率 **95.0%**
- **理解 AdaRMS 时间步全局调制**：Pi0 时间步仅在输入层融合（action_time_mlp），深层 Expert 对 t 不敏感；Pi0.5 通过 time_mlp_in→SiLU→time_mlp_out 生成 adarms_cond，广播至 Expert 全18层各自的 Linear(hidden→hidden×3) 投影，实现 RMSNorm(x)×(1+scale)+shift 和残差门控 gate，与 DiT 的 Adaptive LayerNorm 机制等价
- **掌握 Quantile 归一化跨机器人适配**：Pi0 使用 IDENTITY 归一化，依赖预训练数据分布已知；Pi0.5 对 STATE 和 ACTION 均使用分位数归一化，将各维度值域统一映射到 [-1,1]，对重尾分布（极端位移）鲁棒，无需针对新机器人手动调整 lr 或值域

### STAR（面试版）

**Situation**：在 Pi0 基础上，针对 finetuning 场景存在三个问题：状态只有 Expert 能感知、时间步信息深层稀释、不同机器人值域悬殊导致训练不稳定。

**Task**：理解 Pi0.5 的三项改进机制，部署并验证端到端推理流程。

**Action**：
1. 梳理 `PI05Policy → PI05Pytorch → PaliGemmaWithExpertModel` 类结构，对比 Pi0 定位三处关键改动：Processor 中状态离散化（`Pi05PrepareStateTokenizerProcessorStep`）、`compute_layer_complete` 中 AdaRMS 调用（`layernorm_forward(layer.input_layernorm, x, adarms_cond)`）、以及 config 中 QUANTILES 归一化配置
2. 验证 AdaRMS 路径：`sinusoidal(t)→time_mlp_in→SiLU→time_mlp_out→adarms_cond`，追踪 `adarms_cond` 如何通过 `forward(..., adarms_cond=[None, cond])` 传入 Expert 每一层的 `PiGemmaRMSNorm.forward(x, cond)`，确认 `scale, shift, gate = Linear(cond, d×3).chunk(3)` 逻辑
3. 验证状态路径：状态不再经过 `state_proj`，而由 Processor 离散化为整数序列嵌入 Prefix，`embed_suffix` 不再接收 state 参数，Suffix 只含 50 个 action token

**Result**：Pi0.5 在本地 LIBERO `libero_object` task 0 的 N=20 评估中完成 19 个回合，成功率 **95.0%**、Wilson 95% CI **76.4%-99.1%**，平均总 reward **0.95**。在相同 20 个初始状态上较 Pi0 高 30 个百分点，但 McNemar `p≈0.0703`，因此当前应描述为明显优势趋势，而非已达到统计显著。

### 已测指标与量化缺口

| 指标 | 状态 | 如何获取 |
|---|---|---|
| 任务成功率 | **95.0%（19/20；Wilson 95% CI 76.4%-99.1%）** | `outputs/eval/pi05_libero_object_task0_n20_final/eval_info.json` |
| checkpoint 参数元素 | **3,616,757,520（812 tensors）** | 本地 `model.safetensors` 逐 tensor shape 统计 |
| checkpoint 文件大小 | **7,473,096,344 字节** | 本地文件 `stat` |
| Viewer chunk 推理延迟 | 约 8.5–9.2 s（稳定运行日志；首次更慢） | `run_pi05_libero_viewer.sh` 终端日志，非标准微基准 |
| GPU 显存占用（GB） | needs measurement | `nvidia-smi` 推理时快照 |
| 与 Pi0 成功率对比 | **+30 pp；discordant 7 vs 1；McNemar p≈0.0703** | 相同 task、seed、20 个初始状态 |
