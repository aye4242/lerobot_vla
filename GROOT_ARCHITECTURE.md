# Groot N1.7 架构深度解析

> 个人学习笔记，基于 `src/lerobot/policies/groot/` 源码整理。
> 官方模型：`nvidia/GR00T-N1.7-3B`，论文：GR00T N1 Technical Report (arXiv 2503.14734)

## 本机复现状态（更新至 2026-08-10）

当前已经完成 GR00T N1.7 在本机的基础复现链路，并通过一次真实 LIBERO 观测到动作的端到端 smoke test。

| 项目 | 本机配置 / 结果 |
|---|---|
| 隔离环境 | `/extdata/hdd2/lerobot-smolvla/.venv-groot` |
| LeRobot extras | `.[groot,libero]` |
| 模型 | GR00T N1.7 的 LIBERO Spatial / Object / Goal / LIBERO-10 专用 checkpoint |
| 本地 checkpoint | `hf-cache-groot/gr00t17-lerobot-libero_<suite>-640`，四套均已下载 |
| 仿真 | LIBERO `libero_object` task 0，MuJoCo/robosuite |
| 仿真机械臂 | Franka Emika Panda，7 维末端相对动作接口 |
| embodiment | `libero_sim`，ID 2 |
| 任务 | `pick up the alphabet soup and place it in the basket` |
| 相机输入 | `image` + `wrist_image`；环境的 `image2` 映射为 `wrist_image` |
| 预测 / 执行 horizon | 每次预测 16 步，执行 8 步后重新规划 |
| Flow Matching | 4 个 inference timesteps |
| 精度 | FP32；Tesla M40 不支持 BF16，关闭 Flash Attention |
| 参数量 | 3,144,016,000 |
| GPU 常驻显存 | 约 11.99 GiB（纯模型）；带仿真约 12.5 GiB |
| 首块真实推理 | 23.617 秒，成功输出并执行 7 维动作 |

smoke test 的首个动作是：

```text
[0.0482, -0.2386, 0.0238, -0.0054, 0.0289, -0.0034, -1.0]
```

这证明 checkpoint 加载、双相机输入、语言编码、状态打包、GR00T 动作生成、LIBERO 动作解码和 MuJoCo 执行链已经连通。当前尚未完成完整回合成功率评测，因此不能把 smoke test 当作任务成功率。

### 本机兼容处理

- checkpoint 保存的 `base_model_path` 指向训练机 `/cache/huggingface/...`，加载时改为当前 checkpoint 自身；该 LeRobot checkpoint 已包含完整模型权重。
- 官方 `nvidia/Cosmos-Reason2-2B` 处理器仓库需要授权。本机只下载公开 `Qwen/Qwen3-VL-2B-Instruct` 的 11.5 MB tokenizer/image/video processor 配置，不重复下载 2B 模型权重。
- `.venv-groot` 通过符号链接复用 Pi0 环境已有的 409 MB LIBERO assets，避免重复保存。
- 15 GiB RAM 小于 12.6 GB FP32 checkpoint 加载峰值需求，首次启动会大量使用 swap；实测模型构造约 417 秒，完整进程约 9 分钟。

### 运行命令

实时 Viewer：

```bash
cd /home/aitech/Workspace/VLA/lerobot
bash ./examples/libero/run_groot_libero_viewer.sh
```

窗口默认暂停。`Space` 运行/暂停，`C` 切换策略固定视角与自由观察视角；自由视角下左键拖动旋转、右键拖动平移、滚轮缩放。

单回合评测入口：

```bash
N_EPISODES=1 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/groot_libero_object_task0_n1 \
bash ./examples/libero/run_groot_libero_eval.sh
```

正式指标应在完整单回合通过后再扩大到 5/20 回合，避免在当前约 24 秒一次重规划的速度下直接启动长时间批量任务。

---

## 1. 项目目标

Groot N1.7 是 NVIDIA 发布的**人形机器人通用基础模型**，核心目标是用一个模型支持多种机器人形态（Multi-Embodiment），同时保持对复杂语言指令和视觉场景的深度理解。

| 项目 | 说明 |
|---|---|
| 输入 | 摄像头图像（多张）+ 自然语言指令 + 机器人状态（最大132维）+ embodiment_id |
| 输出 | 40帧动作序列（最大132维），关节角度、速度、末端执行器姿态等 |
| 核心机制 | System 2（VLM 慢思考）+ System 1（DiT 快思考）+ Flow Matching 4步去噪 |
| 多具身支持 | 最多32种机器人，具身专属编解码器，DiT 骨干共享 |

---

## 2. 基础组件

### 2.1 Qwen3-VL Backbone（System 2 慢思考）

Qwen3-VL（Cosmos-Reason2-2B）= Visual Encoder（看图）+ Language Model（理解语言）

#### Visual Encoder

- 输入：多张图像，默认 230×230 裁剪后缩放至 256×256
- 输出：图像 token 序列 [B, N_img, 2048]

#### Language Model

| 参数 | 值 |
|---|---|
| 基础模型 | nvidia/Cosmos-Reason2-2B |
| 实际使用层数 | 前 16 层（`select_layer=16`，非最终层） |
| 隐状态维度 | 2048 |

取第16层而非最终层：深层越"语言化"，越不适合运动控制；中间层保留更多视觉-运动相关特征。

#### VLM 特征精炼（VLLN + VL Self-Attention）

```
backbone 第16层输出 [B, N_img+N_text, 2048]
    ↓ VLLN：LayerNorm（可训练，适配动作任务）
    ↓ VL Self-Attention：独立 4 层 Transformer
      图像 token 和文字 token 再做一轮 Self-Attention
    = vl_embeds [B, N_img+N_text, 2048]（传给 Action Head）
```

### 2.2 Action Head（System 1 快思考）

#### 具身专属编解码器（CategorySpecificMLP）

多具身核心实现：把 N 种机器人的权重打包为三维参数张量，按 embodiment_id 索引：

```
W = Parameter(32, input_dim, hidden_dim)   ← 32种机器人各一套权重
selected_W = W[embodiment_id]              ← 按机器人类型取出对应权重
output = x @ selected_W                    ← 用专属权重计算
```

三处使用具身专属权重：

| 模块 | 类型 | 维度变化 |
|---|---|---|
| state_encoder | CategorySpecificMLP（2层） | [B,132] → [B,1,1536] |
| action_encoder | MultiEmbodimentActionEncoder（3层Linear） | [B,40,132] → [B,40,1536] |
| action_decoder | CategorySpecificMLP（2层） | [B,40,1536] → [B,40,132] |

#### AlternateVLDiT（32层交替注意力）

| 参数 | 值 |
|---|---|
| 总层数 | 32 |
| 注意力头数/head_dim | 32头 × 48 = 1536 |
| Cross-Attention 来源 | vl_embeds（2048维，投影后对齐） |
| 归一化 | AdaLayerNorm（由时间步 t 调制） |
| 层交替规则 | 偶数层=Cross-Attn，奇数层=Self-Attn |

---

## 3. 系统整体架构

### 3.1 两模块架构

> 官方架构图：
> ![Groot N1.7 架构图](./groot-n1-7-architecture.png)

```
输入：图像(N张) + 语言指令 + 状态(132维) + embodiment_id
        │
        ├──────────────────────────────────┐
        ↓                                  ↓
Module 1: Qwen3-VL Backbone          Module 2: Action Head
（System 2 慢思考）                   （System 1 快思考）

图像 → Visual Encoder                state → state_encoder(emb_id) → state_tok
语言 → LM Embedding                  noisy_action → action_encoder(t, emb_id) → action_tok
↓ 16层 Self-Attention                sa_embs = cat(state_tok, action_tok)
↓ VLLN + VL Self-Attn(4层)           ↓ AlternateVLDiT(32层)
→ vl_embeds（冻结后只读）  ──────────→ Cross-Attn 单向读取 vl_embeds
                                      ↓ action_decoder(emb_id)
                                      → v_t → Euler 4步 → 40×132 动作
```

**三条规则：**
- Module 1 输出 vl_embeds 是**只读的**：Action Head 通过 Cross-Attn 单向读取，不修改
- 状态**绕过** Module 1，直接进入 Action Head
- 图文 token **看不到**动作 token（单向，保证 vl_embeds 不受动作影响）

### 3.2 Token 类型与来源

| Token 类型 | 来源 | 维度 | 数量 |
|---|---|---|---|
| 图像 token | Visual Encoder（Qwen3-VL 视觉部分） | 2048 | N_img |
| 文字 token | LM Embedding（Qwen3-VL 语言部分） | 2048 | N_text |
| 状态 token | state_encoder（具身专属 MLP） | 1536 | 1 |
| 动作 token | action_encoder（具身专属 Linear） | 1536 | 40 |

### 3.3 AlternateVLDiT 内部（32层）

```
sa_embs = cat(state_tok, action_tok)  [B, 41, 1536]

Layer 0  [Cross-Attn]：
  Q = sa_embs（41个 token 提问）
  K,V = vl_embeds（VLM 特征回答）
  → action/state tokens 读取场景语义理解
  AdaLayerNorm(sa_embs, t) 调制归一化

Layer 1  [Self-Attn]：
  Q = K = V = sa_embs（41个 token 互看）
  → state_tok 与 action_tok 内部协商
  AdaLayerNorm(sa_embs, t) 调制归一化

Layer 2  [Cross-Attn] → Layer 3  [Self-Attn] → ...（重复16轮）

model_output [B, 41, 1536]
取后40个 → action_decoder(emb_id) → v_t [B, 40, 132]
```

### 3.4 注意力规则总结

| | Cross-Attn 层（偶数） | Self-Attn 层（奇数） |
|---|---|---|
| Q 来源 | sa_embs（自身） | sa_embs（自身） |
| K/V 来源 | vl_embeds（VLM） | sa_embs（自身） |
| 解决的问题 | 动作与场景的语义对齐 | 40步动作的时序一致性+状态协商 |
| VLM 能看到动作吗 | 否（单向） | — |

---

## 4. 核心创新点

### 4.1 Multi-Embodiment 多具身设计

**问题**：不同机器人关节数、动作维度、状态含义完全不同。共享 MLP 权重会导致不同机器人的梯度互相干扰。

**解法**：CategorySpecificMLP——每种机器人有独立的 MLP 权重，但 DiT 32层共享。

```
普通 MLP：              具身专属 MLP：
W = (in, out)           W = (32, in, out)  ← 32套独立权重
output = x @ W          selected = W[emb_id]
                        output = x @ selected
```

**为什么 DiT 可以共享**：DiT 学习的是"如何根据场景+状态+噪声预测速度场"，这是跨机器人的通用物理直觉。只有输入/输出接口（具身专属 MLP）需要针对各机器人的特定关节空间定制。

### 4.2 AlternateVLDiT 交替注意力

**问题**：动作生成需要同时理解场景（图文）和动作序列内部的时序逻辑。单一的 Cross-Attn 无法处理 action tokens 之间的协调。

**解法**：两种注意力交替进行
- Cross-Attn（每2层1次）：从 VLM 特征中获取场景语义
- Self-Attn（每2层1次）：action tokens 之间协调，保证40步动作连贯

### 4.3 System 1 / System 2 设计哲学

借用认知科学中"双系统"思想：
- **System 2（Qwen3-VL）**：慢、深度、语义推理，负责"看清楚场景、理解指令"
- **System 1（AlternateVLDiT）**：快、自动、直觉反应，负责"快速生成流畅动作"

两者通过 Cross-Attention 单向连接：System 2 先完整思考，System 1 按图索骥。

### 4.4 Flow Matching（4步，t=0→1）

Groot 使用与 Pi0/Pi0.5 相同的 Flow Matching 框架，但**方向约定相反**：

| | Pi0/Pi0.5 | Groot N1.7 |
|---|---|---|
| 噪声端 | t=1 | t=0 |
| 干净端 | t=0 | t=1 |
| 插值公式 | t×noise+(1-t)×action | (1-t)×noise+t×action |
| 速度目标 | noise-action | action-noise |
| 推理步数 | 10步 | **4步** |
| t 离散化 | 连续浮点 | 离散为 0~999 整数桶 |

4步足够的原因：AlternateVLDiT 32层更重（每步计算量更大），4步在精度和速度间取平衡。

---

## 5. 完整端到端流程（含 Flow Matching）

### 训练

```
① 编码图像 + 语言（Qwen3-VL Backbone，通常冻结）
   image × N → Visual Encoder → img_tokens
   language  → LM Embedding  → text_tokens
   [img_tokens | text_tokens] → 16层 Self-Attention（图文双向融合）
   → 取第16层隐状态 → VLLN → VL Self-Attn(4层)
   → vl_embeds [B, seq_len, 2048]

② 编码状态（具身专属）
   state [B, 132]
   → state_encoder(embodiment_id): Linear→ReLU→Linear
   → state_features [B, 1, 1536]

③ Flow Matching 构造带噪动作（训练专有）
   noise ~ N(0,1)                      [B, 40, 132]
   t     ~ Beta(1.5, 1.0)
   x_t   = (1-t)×noise + t×action      ← 带噪中间态（t=0:纯噪声，t=1:干净动作）
   u_t   = action - noise               ← 训练目标（速度场方向）

④ 编码带噪动作（具身专属，含时间步）
   t_disc = int(t × 1000)               ← 离散化为 0~999
   x_t + sinusoidal(t_disc) → action_encoder(embodiment_id)
   → action_features [B, 40, 1536]
   → (+) 可学习位置编码（区分0~39步位置）
   sa_embs = cat(state_features, action_features)  [B, 41, 1536]

⑤ 模型前向（AlternateVLDiT，32层交替）
   Layer 0  [Cross]: sa_embs Q → vl_embeds K/V  ← 读场景特征
   Layer 1  [Self] : sa_embs 内部 Q/K/V          ← 内部协商
   Layer 2  [Cross]: sa_embs Q → vl_embeds K/V
   Layer 3  [Self] : sa_embs 内部 Q/K/V
   ...（共32层，各16次）...
   每层前：AdaLayerNorm(x, t_disc) 调制归一化行为
   → model_output [B, 41, 1536]

⑥ 解码速度（具身专属）
   model_output[-40:]
   → action_decoder(embodiment_id): Linear→ReLU→Linear
   → v_pred [B, 40, 132]

⑦ 计算损失
   Loss = MSE(v_pred, u_t) × action_mask
```

### 推理

```
一次性预计算（结果复用整个推理过程）：
   image + language → Backbone → VLLN → VL Self-Attn → vl_embeds  ← 缓存
   state → state_encoder(emb_id) → state_features                  ← 缓存

x = N(0,1)  [B, 40, 132]   （初始纯噪声，t=0）

循环 4 次（Euler ODE，t 从 0 走向 1）：
   step 0: t_cont=0.00  t_disc=0
   step 1: t_cont=0.25  t_disc=250
   step 2: t_cont=0.50  t_disc=500
   step 3: t_cont=0.75  t_disc=750

   每步执行：
   ┌──────────────────────────────────────────────────────────┐
   │ ④ x → action_encoder(t_disc, emb_id) → action_features  │
   │      sa_embs = cat(state_features, action_features)       │
   │ ⑤ AlternateVLDiT(sa_embs, vl_embeds, t_disc)            │
   │      32层 Cross/Self 交替 + AdaLayerNorm(t_disc)         │
   │      → model_output[-40:]                                 │
   │ ⑥ action_decoder(emb_id) → v_t [B, 40, 132]             │
   └──────────────────────────────────────────────────────────┘
   x = x + (1/4) × v_t   ← Euler 步进

x [B, 40, 132] = 干净动作（t=1）→ 反归一化 → 发给机器人执行
```

---

## 6. 训练参数与推理对应关系

训练时反向传播只更新部分参数，推理时所有参数（含冻结）都参与计算。

| 模块 | 训练状态 | 原因 | 推理中的位置和作用 |
|---|---|---|---|
| Qwen3-VL Visual Encoder | ❌ 冻结 | 预训练视觉特征已足够 | 步骤①：提取图像 token |
| Qwen3-VL LM 16层 | ❌ 冻结 | 语言理解来自大规模预训练 | 步骤①：图文 Self-Attention |
| **VLLN** | ✅ 训练 | 精炼特征适配动作任务 | 步骤①：vl_embeds 归一化 |
| **VL Self-Attn（4层）** | ✅ 训练 | 蒸馏场景特征 | 步骤①：精炼 vl_embeds |
| **state_encoder** | ✅ 训练 | 学习如何表示机器人状态 | 步骤②：state→state_features |
| **action_encoder** | ✅ 训练 | 学习如何表示带噪动作+时间步 | 每推理步④：x_t→action_features |
| **position_embedding** | ✅ 训练 | 学习40步动作的位置感知 | 每推理步④：叠加在 action_features 上 |
| **AlternateVLDiT 32层** | ✅ 训练 | 核心速度场预测能力 | 每推理步⑤：32层 Cross/Self Attn |
| **action_decoder** | ✅ 训练 | 学习DiT输出→速度的映射 | 每推理步⑥：model_output→v_t |

**关键规律**：
- 冻结模块（Qwen3-VL 主体）在推理中只运行**一次**（缓存 vl_embeds）
- 训练模块在推理的**每个4步循环**中都重新计算（action_encoder/DiT/action_decoder）
- 具身专属模块（state_encoder/action_encoder/action_decoder）每次都用 `W[embodiment_id]` 索引当前机器人的权重

---

## 7. 训练 vs 推理对比

| 方面 | 训练 | 推理 |
|---|---|---|
| Backbone 运行次数 | 每样本1次 | **1次（vl_embeds 缓存）** |
| state_encoder 运行次数 | 每样本1次 | **1次（state_features 缓存）** |
| x_t 来源 | Flow Matching 插值：(1-t)×noise+t×action | 从纯噪声 N(0,1) 开始 |
| t 的取法 | 随机采样 Beta(1.5,1.0) → 单个 t | 固定序列：0,250,500,750（4步） |
| DiT 运行次数 | **1次**（随机取一个 t） | **4次**（Euler ODE 循环） |
| 输出 | Loss = MSE(v_pred, u_t)，反向传播 | 干净动作 x [B,40,132]，执行 |
| action_mask | 参与 Loss 计算（过滤无效维度） | 不涉及 |

---

## 8. 冻结策略

| 组件 | 默认状态 | 配置参数 |
|---|---|---|
| Qwen3-VL Visual Encoder | ❌ 冻结 | `tune_visual=False` |
| Qwen3-VL LM 16层 | ❌ 冻结 | `tune_llm=False` |
| VLLN + VL Self-Attn | ✅ 训练 | `tune_vlln=True` |
| state/action encoder/decoder | ✅ 训练 | `tune_projector=True` |
| AlternateVLDiT | ✅ 训练 | `tune_diffusion_model=True` |

**可选：解冻顶层 LLM**（`tune_top_llm_layers=N`）：解冻最后 N 层 LM，使语言理解能微调适配目标任务，需配合更小学习率。

**典型 finetuning 配置**：
```
冻结：Qwen3-VL Visual + LM（保留预训练的视觉-语言理解）
训练：VLLN + VL Self-Attn + 具身编解码器 + AlternateVLDiT
效果：利用大模型的理解能力，只训练"动作生成"相关模块
```

---

## 9. 关键配置参数（`GR00TN17Config`）

```python
# 骨干模型
model_name           = "nvidia/Cosmos-Reason2-2B"   # Qwen3-VL 骨干
select_layer         = 16                            # 取第16层隐状态

# 动作维度
max_state_dim        = 132                           # 最大状态维度（人形机器人）
max_action_dim       = 132                           # 最大动作维度
action_horizon       = 40                            # 一次预测40步
max_num_embodiments  = 32                            # 最多32种机器人

# DiT 结构
hidden_size          = 1024                          # action token 内部维度
input_embedding_dim  = 1536                          # 编码器输出维度
diffusion_model_cfg:
  num_layers         = 32                            # DiT 层数
  num_attention_heads= 32                            # 注意力头数
  attention_head_dim = 48                            # head_dim
  norm_type          = "ada_norm"                    # AdaLayerNorm

# VL Self-Attention
vl_self_attention_cfg:
  num_layers         = 4                             # 精炼 VLM 特征的层数
  num_attention_heads= 32
  attention_head_dim = 64

# 推理 Flow Matching
num_inference_timesteps = 4                          # Euler ODE 步数
noise_beta_alpha     = 1.5                           # Beta 分布 α
noise_beta_beta      = 1.0                           # Beta 分布 β
num_timestep_buckets = 1000                          # t 离散化桶数

# 归一化
use_percentiles      = True                          # 类似 Quantile 归一化

# 训练开关
tune_llm             = False
tune_visual          = False
tune_vlln            = True
tune_projector       = True
tune_diffusion_model = True
tune_top_llm_layers  = 0                             # 可选解冻顶层 LLM

# 正则化
state_dropout_prob   = 0.2                           # 训练时随机丢弃状态（防止过依赖状态）
attend_text_every_n_blocks = 2                       # 每2层做一次 Cross-Attn
use_alternate_vl_dit = True                          # 使用交替注意力
```

---

## 10. 类结构速查

```
GR00TPolicy（src/lerobot/policies/groot/modeling_groot.py）
  ├── select_action(batch)          → 单步动作（维护 action_queue）
  ├── predict_action_chunk(batch)   → 完整推理一次
  └── forward(batch)                → 训练前向，返回 loss

  ├── Qwen3Backbone
  │     ├── visual                  → Visual Encoder（图像编码）
  │     ├── language_model          → LM 前16层（图文 Self-Attention）
  │     └── forward()               → vl_embeds（经VLLN+VL Self-Attn精炼）
  │
  └── GR00TN17ActionHead
        ├── vlln                    → LayerNorm（精炼 vl_embeds）
        ├── vl_self_attention       → 4层 Self-Attn Transformer
        ├── state_encoder           → CategorySpecificMLP（具身专属状态编码）
        ├── action_encoder          → MultiEmbodimentActionEncoder（具身专属动作编码）
        ├── position_embedding      → 可学习位置编码（40步）
        ├── model                   → AlternateVLDiT（32层核心）
        ├── action_decoder          → CategorySpecificMLP（具身专属速度解码）
        ├── forward()               → 训练：loss
        ├── get_action()            → 推理：4步Euler ODE
        └── _encode_features()      → 缓存 vl_embeds + state_features

AlternateVLDiT（src/lerobot/policies/groot/action_head/cross_attention_dit.py）
  每层 BasicTransformerBlock：
    ├── norm1：AdaLayerNorm（由 t 调制）
    ├── attn1：Cross-Attn（偶数层）或 Self-Attn（奇数层）
    ├── norm3：LayerNorm
    └── ff：FeedForward（FFN）
```

---

## 11. 本地复现与实验结果

### 11.1 已完成的复现范围

本机已经完成 GR00T N1.7 的模型加载、LIBERO 环境接入、实时 Viewer、动作执行和批量评测入口，不再是仅完成源码阅读的状态。

| 项目 | 本地结果 |
|---|---|
| 独立环境 | `/extdata/hdd2/lerobot-smolvla/.venv-groot` |
| 仿真环境 | LIBERO + robosuite + MuJoCo |
| 仿真机械臂 | Franka Emika Panda，7 维末端相对动作 |
| 已准备 suite | `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` |
| checkpoint 数量 | 4 套 suite 专用 checkpoint |
| 单个权重大小 | 12,576,215,296 字节（约 12.6 GB） |
| Viewer | 支持实时画面、暂停/运行、重置和自由相机 |
| 成功判断 | 使用 LIBERO 原生 BDDL `info["is_success"]`，覆盖四个 suite |
| 动作策略 | 预测 16 步，执行 8 步后重新规划 |
| 精度 | FP32，关闭 BF16 和 Flash Attention |

四个本地 checkpoint 均已下载完成。已核对的 SHA-256 包括：

```text
libero_spatial  8a6e55ca705ab60c6d9c1eaf585dbda53f1b92bffdf6050c18ce68eb3afc6e69
libero_goal     71c8220d03c1d429fe90861869e82189e944aeef9a58aea13bc4f3140dfdb0ef
libero_10       8a7f3c0fb13cc84f89bbc7af1a675431ddd7b16ce47bebb12b4298f6b2827a94
```

### 11.2 端到端 smoke test

已在 `libero_object/task 0` 上完成从真实环境观测到 MuJoCo 动作执行的端到端 smoke test：

```text
Instruction: pick up the alphabet soup and place it in the basket
First action: [0.0482, -0.2386, 0.0238, -0.0054, 0.0289, -0.0034, -1.0]
First action-chunk inference: 23.617 s
```

该结果证明以下链路已经真实运行，而不是只完成模型实例化：

1. LIBERO 双相机图像与机器人状态采集。
2. Qwen3-VL 图像/语言编码与 GR00T 输入打包。
3. 4 步 Flow Matching 动作生成。
4. 动作反归一化和 7 维 LIBERO 控制接口转换。
5. Panda 机械臂在 MuJoCo 中执行预测动作。

### 11.3 Viewer 复现范围与当前现象

统一 Viewer 已验证以下场景可以按照 suite/task-id 正确创建、加载对应权重并显示官方 LIBERO instruction：

| suite / task | 已验证内容 | 结果 |
|---|---|---|
| `libero_object / 0` | 场景创建、双相机输入、首个动作块和 MuJoCo 执行 | 完成 smoke test |
| `libero_10 / 1` | `put both the cream cheese box and the butter in the basket` 场景和指令 | Viewer 正确加载 |
| `libero_10 / 3` | `put the black bowl in the bottom drawer of the cabinet and close it` 场景和指令 | Viewer 正确加载 |

`libero_spatial`、`libero_object`、`libero_goal` 和 `libero_10` 均使用各自的 GR00T checkpoint；Viewer 不会把 Object 权重误用于其他 suite。当前观察到的 task 1/task 3 画面处于 `PAUSED step=0`，只证明环境和指令映射正确，不等同于任务成功。

### 11.4 本机性能现象

| 指标 | 实测值 / 现象 |
|---|---|
| 参数量 | 3,144,016,000 |
| 纯模型显存 | 约 11.99 GiB |
| 模型加仿真显存 | 约 12.5 GiB |
| 首块动作推理 | 23.617 秒 |
| 模型构造 | 约 417 秒 |
| 首次完整启动 | 约 7-10 分钟 |
| 主机内存限制 | 15 GiB RAM，加载阶段明显使用 swap |

模型能够生成数值有效的连续动作并驱动仿真机械臂，说明本机兼容修改有效。当前性能瓶颈主要是 Tesla M40 只能使用 FP32，以及主机内存不足导致模型加载使用 swap，而不是 LIBERO 环境或动作接口未接通。

### 11.5 当前实验边界与评估状态

目前已经完成 GR00T N1.7 的本地复现、四套 checkpoint 准备、统一 Viewer、实时相机交互、场景/指令映射、首个真实动作执行和单回合标准评测。task 1/task 3 的 Viewer 截图属于场景加载验证，不是成功回合记录。

2026-08-10 在 `libero_object/task 0`、seed 1000、`n_action_steps=8` 下完成 N=1：

| 指标 | N=1 实测值 |
|---|---:|
| 成功 | **1/1（100%）** |
| 成功检测步数 | 139 |
| 总 reward / 最大 reward | 1.0 / 1.0 |
| 评测时间 | 124.54 秒 |
| 输出 | `outputs/eval/groot_libero_object_task0_n1/eval_info.json` |

首次 N=20 在第20回合约120步时发生 Tesla M40 内核 `NVRM Xid 79: GPU has fallen off the bus`，没有最终 JSON，故该次中间进度不能作为结果。主机恢复后按完全相同协议完成正式重跑（seed 1000--1019）：

| 指标 | N=20 正式实测值 |
|---|---:|
| 成功率 | **18/20（90.0%）** |
| Wilson 95% CI | 69.9%--97.2% |
| 平均总 reward / 最大 reward | 0.90 / 0.90 |
| 总评测时间 / 平均每回合 | 2491.90 秒 / 124.59 秒 |
| 失败 seed | 1003、1011 |
| 输出 | `outputs/eval/groot_libero_object_task0_n20_final_retry/eval_info.json` |

前10回合视频位于 `outputs/eval/groot_libero_object_task0_n20_final_retry/videos/libero_object_0/`。评测器现会逐回合原子写入 progress；本次的 `progress/libero_object_0.json` 最终状态为 `completed`。

该正式评测的复现实验命令为：

```bash
LIBERO_SUITE=libero_object \
LIBERO_TASK_IDS='[0]' \
N_EPISODES=20 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/groot_libero_object_task0_n20_final_retry \
bash ./examples/libero/run_groot_libero_eval.sh
```

同一 `libero_object/task 0`、同一20个 seed 下：Pi0 为 13/20（65%，37.05秒/回合），GR00T 为 18/20（90%，124.59秒/回合），Pi0.5 为 19/20（95%，348.48秒/回合）。GR00T 对 Pi0 高25个百分点，但配对 McNemar exact two-sided `p=0.1797`；GR00T 对 Pi0.5 低5个百分点，`p=1.0`，N=20不足以支持显著性结论。三者执行 horizon 分别为 10、8、10，因此这是同任务行为效果比较，不是严格吞吐比较。

---

## 12. 简历版本

### One-line

在 LeRobot 中完成 GR00T N1.7 与 LIBERO 的本地部署和 MuJoCo 可视化推理，接入 Spatial/Object/Goal/LIBERO-10 四套 checkpoint，跑通 Qwen3-VL、32 层 AlternateVLDiT、4 步 Flow Matching 到 Panda 机械臂动作执行的端到端链路；3.144B 参数模型在 Tesla M40 上占用约 11.99 GiB 显存，首块推理 23.617 秒。

### 3-bullet

- **部署 GR00T N1.7 多具身推理**：在 LeRobot 内接入 LIBERO 四个标准 suite，跑通双相机观测、语言编码、4 步 Flow Matching、动作反归一化和 Panda/MuJoCo 执行；3.144B 参数模型占用约 11.99 GiB 显存，首块动作推理 23.617 秒
- **理解 Multi-Embodiment 设计**：状态/动作编解码器使用 `Parameter(32, in, out)` 三维权重张量，按 embodiment_id 索引当前机器人专属权重；DiT 32层共享（学习通用速度场），只有 I/O 接口是机器人专属的，实现了"一个模型，多种机器人"而无梯度干扰
- **掌握 System 1/System 2 架构**：Qwen3-VL（System 2）先独立完成图文理解，输出 vl_embeds 后冻结；AlternateVLDiT（System 1）通过 Cross-Attn 单向读取，每2层读一次场景特征、每2层做一次 action 内部协商，AdaLayerNorm 保证每层感知当前时间步 t

### STAR（面试版）

**Situation**：需要理解并部署 Groot N1.7——NVIDIA 的人形机器人通用基础模型，核心挑战是多具身支持和高效 VLM-动作交互。

**Task**：在 LeRobot 内跑通 Groot N1.7 推理，理解 Multi-Embodiment、AlternateVLDiT、System 1/System 2 三个核心机制。

**Action**：
1. 梳理 `GR00TPolicy → GR00TN17ActionHead → AlternateVLDiT` 三层结构，定位 `CategorySpecificMLP.forward`（具身专属权重索引）和 `BasicTransformerBlock.forward`（单层 Cross/Self-Attn）两个核心实现
2. 验证推理路径：`_encode_features` 一次性计算并缓存 vl_embeds 和 state_features；`get_action_with_features` 中4步 Euler 循环每步重新编码带噪动作（action_encoder），通过32层 DiT 预测速度，`x += (1/4) × v_t` 步进
3. 理解训练-推理参数对应：VLLN/VL Self-Attn/state_encoder/action_encoder/DiT/action_decoder 均在训练中更新，推理时原样使用；Qwen3-VL 主体冻结，推理只计算一次缓存

**Result**：完成 Spatial、Object、Goal、LIBERO-10 四套 checkpoint 的本地准备和统一启动器，在 `libero_object/task 0`、固定20个 seed 的正式评测中取得 **18/20（90.0%，Wilson 95% CI 69.9%--97.2%）**；平均124.59秒/回合。模型加仿真显存约12.5 GiB，首次启动约7--10分钟。此前一次评测因Tesla M40 Xid 79中断，但恢复后重跑已完整产出结果。

### 已有指标与量化缺口

| 指标 | 状态 | 如何获取 |
|---|---|---|
| 单回合任务结果 | **1/1成功；第139步完成** | `outputs/eval/groot_libero_object_task0_n1/eval_info.json` |
| N=20任务成功率 | **18/20（90.0%，Wilson 95% CI 69.9%--97.2%）** | `outputs/eval/groot_libero_object_task0_n20_final_retry/eval_info.json` |
| N=20 reward / 速度 | 0.90 / 124.59 秒每回合 | 同上；总时长2491.90秒，失败seed为1003、1011 |
| 首块动作推理 | 23.617 秒 | Viewer 实测日志 |
| 纯模型 GPU 显存 | 约 11.99 GiB | `nvidia-smi` 实测 |
| 模型加仿真显存 | 约 12.5 GiB | Viewer 运行时实测 |
| 参数量 | 3,144,016,000 | 本地模型统计 |
| 首次完整启动 | 约 7-10 分钟 | 本机加载日志 |
| 与 Pi0/Pi0.5 成功率对比 | Pi0 65%，GR00T 90%，Pi0.5 95% | 同 task/seed/N=20；差异未达显著性（McNemar p=0.1797、1.0） |
