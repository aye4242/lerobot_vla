# SmolVLA 架构文档

> 参考文件：`src/lerobot/policies/smolvla/`
> 论文：https://huggingface.co/papers/2506.01844

---

## 1. 项目目标

SmolVLA（Small Vision-Language-Action）是 Hugging Face 设计的轻量级机器人控制策略，目标是在消费级 GPU 和边缘设备上部署可用的 VLA。

相比同类模型（Pi0 ~2.3B），SmolVLA 将参数量压缩至 ~575M（约 4× 缩减），通过 Cross-Attention 架构解耦 VLM 与 Action Expert，使两者可以独立扩缩，同时保留 Flow Matching 的动作质量。

**成功标准：** 在 SO-100/SO-101 机械臂、Aloha 双臂、仿真环境（LIBERO、Gym Pusht）上，以 Receding Horizon Control（每次生成 50 步动作块）完成多步操作任务。

---

## 2. 上游依赖

### SmolVLM2-500M-Video-Instruct

原本用于视频理解与图文问答。SmolVLA 保留其完整的视觉语言理解能力，将图像 patch 特征、语言指令、机器人状态统一编码，输出每一层的 K/V 表征。Action Expert 通过 Cross-Attention 借用这些表征生成动作，而不需要自己理解视觉和语言。

### Flow Matching（Lipman et al. 2022）

不直接预测目标动作，而是学习一个速度场 v_θ：给定当前时刻 t 下带噪动作 x_t，预测"应该朝哪个方向走"。训练时随机采一个时刻 t，推理时从纯噪声出发沿着速度场积分 10 步还原干净动作。

优势：天然处理多模态动作分布（同一场景多种合理轨迹），且推理结构（固定前缀 + 循环去噪）天然支持 KV Cache 优化。

### LeRobot 训练框架

提供数据集加载（LeRobotDataset）、训练循环（HuggingFace Accelerator）、检查点管理。SmolVLAPolicy 通过 PreTrainedPolicy 接口接入，无需修改框架核心。

---

## 3. 系统架构

### 三个核心组件

```
┌─────────────────────────────────────────────────────────┐
│                    SmolVLM2-500M                         │
│                                                          │
│  ┌──────────┐    ┌────────────┐    ┌─────────────────┐  │
│  │  SigLIP  │───►│ Connector  │    │  Text Model     │  │
│  │ (视觉编码)│    │(模态投影器)│    │  (16层 Transformer)│ │
│  └──────────┘    └─────┬──────┘    └────────┬────────┘  │
│                        │                    │            │
│          图像patches ───┘    语言tokens+状态 ─┘           │
│                        ↓    拼接 → Prefix                │
│             VLM 16层处理 → 每层生成 K/V Cache             │
└──────────────────────────┬──────────────────────────────┘
                           │ K/V（16层）
                           ↓
┌──────────────────────────────────────────────────────────┐
│                  Action Expert                            │
│              (hidden = VLM hidden × 0.75)                │
│                                                          │
│   偶数层 ─── Expert Self-Attention（动作token相互通信）   │
│   奇数层 ─── Cross-Attention（Expert Q → VLM K/V）       │
│                                                          │
│   输入：噪声动作 + 时间步 t（fused by action_time_mlp）   │
│   输出：速度场 v_t → Euler ODE → 干净动作                │
└──────────────────────────────────────────────────────────┘
```

**SmolVLM2-500M** 负责理解世界（图像、语言、当前状态），产出 K/V 供 Expert 查询。

**Action Expert** 负责生成动作，参数量约为 VLM 的 75% 宽度，通过 Cross-Attention 读取 VLM 理解，通过 Self-Attention 保证动作块内部连贯。

**Flow Matching（VLAFlowMatching）** 串联两者：训练时计算速度场误差，推理时驱动 10 步 ODE。

---

### Prefix 与 Suffix

SmolVLA 把输入分成两个部分分别处理：

```
Prefix（VLM 处理）：
  [image patches × N] [language tokens × 48] [state × 1]
   ←────── 双向注意力 ──────→                ↑
                              state 能看图文，图文不看 state

Suffix（Expert 处理）：
  [action token × 50]  ← 噪声动作 + 时间步 t 融合后的嵌入
   ←─── 因果注意力（action token 相互因果） ───→
        同时 cross-attend 到 Prefix 的 K/V
```

**为什么 state 在 Prefix 而不是 Suffix：**
state 需要让 Expert 通过 K/V Cache 感知，因此必须进入 VLM 的编码流程。但 state 是机器人传感器读数，不应污染 VLM 预训练的图文理解——所以用非对称 att_mask：state 能读取图文，图文看不到 state。

---

### 模块清单

| 模块 | 文件:行 | 职责 |
|---|---|---|
| `SmolVLAPolicy` | `modeling_smolvla.py:226` | LeRobot 接口层，推理队列，图像/状态预处理 |
| `VLAFlowMatching` | `modeling_smolvla.py:541` | 核心模型：Flow Matching 训练与推理 |
| `SmolVLMWithExpertModel` | `smolvlm_with_expert.py:72` | VLM + Expert 联合前向，KV Cache，cross-attn 层替换 |
| `state_proj` | `modeling_smolvla.py:583` | 状态向量→VLM 空间（Linear 32→vlm_hidden） |
| `action_in_proj` | `modeling_smolvla.py:586` | 噪声动作→Expert 空间（Linear 32→expert_hidden） |
| `action_out_proj` | `modeling_smolvla.py:587` | Expert 输出→动作空间（Linear expert_hidden→32） |
| `action_time_mlp` | `modeling_smolvla.py:589-594` | 融合动作嵌入与时间步嵌入（2层 MLP + SiLU） |

---

## 4. 创新点

### 4.1 Cross-Attention with Interleaved Self-Attention

**问题：** 纯 Cross-Attention 让 Expert 只能问 VLM，动作 token 彼此看不见，50步动作块缺乏内部一致性。纯 Self-Attention 则无法引入 VLM 的语义。

**解法：** 每两层交替一次注意力类型。

```
Expert Layer 0 （偶数） ── Self-Attention
  action[0] ◄──► action[1] ◄──► ... ◄──► action[49]
  动作 token 相互通信，保证连续性

Expert Layer 1 （奇数） ── Cross-Attention
  action Q  ────────────────► VLM K/V
  动作借阅场景语义，理解"在做什么任务"

Expert Layer 2 （偶数） ── Self-Attention
  ...
```

Expert 的 k_proj / v_proj 被替换为 `Linear(vlm_kv_dim → expert_kv_dim)`，允许 Expert（小 hidden）跨维度查询 VLM（大 hidden）的 K/V，不要求两者 head_dim 相同（这是与 Pi0 Shared Attention 的本质区别）。

---

### 4.2 State 嵌入 Prefix 的非对称掩码

**问题：** 机器人状态（关节角度）必须对 Expert 可见，但让状态双向参与 VLM 自注意力会破坏预训练语义理解。

**解法：** 状态 token 加入 Prefix 末尾，但使用不对称 att_mask：

```
[image patches]  [language tokens]  [state token]
 att_mask = 0      att_mask = 0      att_mask = 1
 ← 双向互相看 →                      ↑只能向左看
```

规则：att_mask 的累积和决定谁能看谁（cumsum[i] ≤ cumsum[j] → token i 能看 token j）。

结果是 state 能读取完整图文上下文，图文的表征不被状态信息污染。KV Cache 中包含 state 的表征，Expert cross-attn 可以感知当前机器人状态。

---

### 4.3 KV Cache：前缀只算一次

**问题：** Flow Matching 推理需要 10 次前向，若每次重新计算 Prefix（图像 + 语言 + 状态），开销增加 10×。

**解法：** 推理分两阶段。

```
阶段 1（Prefix Pass）：
  VLM 完整处理 Prefix → 生成并保存 16层 K/V Cache
  只执行一次

阶段 2（去噪循环 × 10）：
  Expert 只处理 50 个 action token
  Cross-Attn 层直接读缓存，不重新计算 VLM
  每步 Euler 更新：x_t -= 0.1 × v_t
```

前缀通常包含数百个 image patch token，是计算开销的主要来源；缓存后 10 步去噪每步仅处理 50 个 action token。

---

### 4.4 action_time_mlp：动作与时间的融合

**问题：** 动作向量（32维连续值）和时间步 t（标量，表示去噪进度）性质不同、维度不同，不能直接相加。

**解法：** 两者各自投影到 expert_hidden 维后拼接，过两层 MLP（Linear + SiLU + Linear）融合：

```
noisy_action [B,50,32] → action_in_proj → [B,50,H]
timestep t [B]  → sinusoidal_encoding → [B,H] → expand → [B,50,H]
                                   ↓ concat
                            [B,50,2H] → MLP → [B,50,H]
                                   ↓
                         送入 Expert 第 0 层
```

时间编码使用正弦余弦位置编码（min_period=4e-3, max_period=4.0），在 [0,1] 区间提供精细的频率分辨率。

---

### 4.5 Beta(1.5, 1.0) 时间步偏置采样

**问题：** 均匀采样 t∈[0,1] 时，模型在接近 t=0（干净动作附近）浪费训练资源——此时噪声小、预测容易。

**解法：** 训练时对 t 使用 Beta(1.5, 1.0) 分布采样（均值=0.6，右偏），让大多数训练样本集中在 t>0.5 的高噪声区域，模型更多练习"从混乱中找方向"。

---

## 5. 完整端到端流程（含 Flow Matching）

### 训练

```
① 编码图像
   image [B,H,W,3] → SmolVLM2 vision encoder
   → img_emb [B, N_img, 2048]

② 编码语言
   language (str) → Tokenizer → SmolVLM2 lang embed
   → lang_emb [B, N_text, 2048]

③ 编码状态（进入 Prefix，非对称掩码）
   state [B,32] → state_proj: Linear(32→2048)
   → state_tok [B, 1, 2048]
   （state 能 attend 图文，图文不能 attend state）
   Prefix = cat(img_emb, state_tok, lang_emb)  [B, N_prefix, 2048]
   → SmolVLM2 处理 Prefix，生成 KV Cache（16层）

④ Flow Matching 构造带噪动作（训练专有）
   noise ~ N(0,1)                  [B, 50, 32]
   t     ~ Beta(1.5, 1.0)
   x_t   = t×noise + (1-t)×action  ← 带噪中间态
   u_t   = noise - action           ← 训练目标（速度场）

⑤ 编码带噪动作 + 时间步融合
   x_t + sinusoidal(t) → action_time_mlp
   → action_emb [B, 50, 2048]
   → Suffix（Expert 处理）

⑥ 模型前向（Expert 16 层，Cross-Attention 架构）
   偶数层：Expert Self-Attn（action tokens 互看）
   奇数层：Cross-Attn（Expert Q → VLM KV Cache）
   → expert_out[-50:] → action_out_proj: Linear(2048→32)
   → v_t [B, 50, 32]

⑦ 计算损失
   Loss = MSE(v_t, u_t)
```

### 推理

```
image + language + state
    ↓ embed_prefix()
Prefix KV Cache（SmolVLM2 一次前向，含 state，缓存16层 KV）

x_t = N(0,1)  [t=1.0]

循环10次（Euler ODE）：
  t: 1.0 → 0.9 → ... → 0.1
  ┌──────────────────────────────────────────────────────────────┐
  │ embed_suffix(x_t, t)                                         │
  │   x_t + sinusoidal(t) → action_time_mlp → action_emb        │
  │   → Suffix [B,50,2048]                                       │
  │ Expert 处理 Suffix：                                          │
  │   偶数层：Expert Self-Attn（action tokens 互看）              │
  │   奇数层：Cross-Attn（Expert Q → VLM KV Cache）              │
  │   → expert_out → action_out_proj → v_t [B,50,32]            │
  └──────────────────────────────────────────────────────────────┘
  x_t = x_t + (-0.1) × v_t

x_0 [B,50,32] = 干净动作 → 反归一化 → 执行
```

---

## 6. 训练 vs 推理

| 方面 | 训练 | 推理 |
|---|---|---|
| **时间步 t** | 随机采样 Beta(1.5,1.0) | 固定序列 1.0→0.9→…→0.1（10步） |
| **输入** | 真实动作 + 随机噪声 → 混合 x_t | 纯高斯噪声作为 x_1 |
| **目标** | 预测速度场 u_t = noise - actions | 积分速度场还原动作 |
| **损失** | MSE(v_θ, u_t)，对有效动作步取均值 | 无损失，输出 50 步动作块 |
| **Prefix 处理** | 与 Suffix 拼接，一次前向通过全部16层 | 独立前向一次，生成 KV Cache |
| **VLM 状态** | eval() + frozen | eval() + frozen |
| **Expert 状态** | train()，requires_grad=True | eval() |
| **动作消费** | backward() 更新 Expert 权重 | 放入队列，逐步 popleft() 执行 |

**训练与推理的结构非对称性：**
训练时 Prefix 和 Suffix 拼接后一起前向，所有16层都同时看到图文状态和噪声动作（shared attention）。推理时 Prefix 独立前向填充 KV Cache，后续去噪只处理 Suffix（cross-attn 读 Cache）。两种模式产生的表征不完全相同，但实践中效果等价。

---

## 6. 冻结策略

默认配置（`train_expert_only=True`）：

| 组件 | 状态 | 原因 |
|---|---|---|
| SigLIP 视觉编码器 | 冻结 | 预训练视觉特征稳定，机器人数据无法改善 |
| Connector（模态投影器） | 冻结 | 属于 VLM，随 VLM 冻结 |
| VLM 文本模型（16层） | 冻结 | 语义理解能力是资产，数据量不足以微调 |
| Action Expert（16层） | 训练 | 从随机初始化学习动作生成 |
| state_proj | 训练 | 需要学习状态表示与 VLM 空间的对齐 |
| action_in/out_proj | 训练 | 动作空间映射，任务相关 |
| action_time_mlp | 训练 | 动作-时间融合，Expert 输入核心 |

**Fine-tuning 模式**（`train_expert_only=False`）：开放 VLM 文本层训练，适合从 `smolvla_base` 预训练权重继续适配目标任务，需搭配更小学习率。

**PEFT 支持：** 默认目标为 Expert 的 q/v_proj + 所有投影层（`modeling_smolvla.py:500`），可将可训练参数量进一步压缩至 Expert 的一小部分。

---

## 7. 关键配置参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `vlm_model_name` | SmolVLM2-500M-Video-Instruct | VLM 骨干 |
| `chunk_size` | 50 | 单次生成动作步数 |
| `num_steps` | 10 | Flow Matching ODE 去噪步数 |
| `max_state_dim / max_action_dim` | 32 / 32 | 状态/动作维度上限（不足补0） |
| `tokenizer_max_length` | 48 | 语言指令最大 token 数 |
| `num_vlm_layers` | 16 | VLM 实际使用的层数 |
| `expert_width_multiplier` | 0.75 | Expert hidden = VLM hidden × 0.75 |
| `self_attn_every_n_layers` | 2 | 每2层插入一次 Expert self-attention |
| `attention_mode` | `"cross_attn"` | 默认 Cross-Attention 模式 |
| `freeze_vision_encoder` | True | SigLIP 冻结 |
| `train_expert_only` | True | VLM 全部冻结，只训 Expert |
| `optimizer_lr` | 1e-4 | AdamW 学习率 |
| `resize_imgs_with_padding` | (512, 512) | 图像 resize 目标尺寸 |

---

## 8. 类结构

```
SmolVLAPolicy                          ← LeRobot 接口
  └── VLAFlowMatching                  ← 核心模型（Flow Matching）
        ├── SmolVLMWithExpertModel     ← VLM + Expert 联合 Transformer
        │     ├── vlm (SmolVLM2, 冻结)
        │     └── lm_expert (小模型，训练)
        ├── state_proj                 ← 状态嵌入
        ├── action_in_proj             ← 动作嵌入
        ├── action_out_proj            ← 速度场输出
        └── action_time_mlp (×2)      ← 动作+时间融合
```

**SmolVLAPolicy** 管理外部接口：图像预处理（resize + 归一化）、状态/动作 padding、推理队列（Receding Horizon Control）。

**VLAFlowMatching** 实现 Flow Matching 逻辑：训练时采样 t 和噪声、计算 MSE loss；推理时分离 Prefix KV Cache 填充和10步去噪循环。

**SmolVLMWithExpertModel** 实现联合 Transformer：逐层判断使用 self-attn 还是 cross-attn，管理 KV Cache 的写入（fill_kv_cache=True）和读取（fill_kv_cache=False）。cross-attn 层的 k_proj/v_proj 在初始化时被替换为适配 Expert 维度的新 Linear 层。

---

## 9. 与 Pi0 对比

| 维度 | SmolVLA | Pi0 |
|---|---|---|
| VLM 骨干 | SmolVLM2-500M | PaliGemma（SigLIP + Gemma-2B） |
| 总参数量 | ~575M | ~2.3B（约4×） |
| Expert 大小 | VLM hidden × 0.75，同源结构 | Gemma-300M，独立模型，hidden=1024 |
| VLM-Expert 交互 | Cross-Attention（单向）+ 每2层 self-attn | Shared Attention（双向，Q/K/V 沿序列拼接） |
| State 位置 | Prefix（VLM 编码，单向掩码） | Suffix（Expert 处理） |
| 维度约束 | Expert head_dim 可与 VLM 不同 | Expert head_dim 必须等于 VLM（拼接条件） |
| 图像分辨率 | 512×512（padding 保持比例） | 224×224（SigLIP 标准） |
| Flow Matching | 完全相同（10步，chunk=50，Beta(1.5,1.0)） | 完全相同 |
| KV Cache | 相同原理（前缀一次，10步复用） | 相同原理 |

**核心架构差异一句话：**
Pi0 是 VLM 与 Expert "合开一个会"（双向 Shared Attention，从第1层起互相看），SmolVLA 是 VLM 先开完会形成报告（K/V Cache），Expert 逐层查阅报告（Cross-Attention），偶尔内部讨论（Self-Attention）。

---

## 10. VLABench 复现实验

### 10.1 实验目标与当前状态

目标是在 Tesla M40 上离线加载官方 `lerobot/smolvla_vlabench` checkpoint，通过 VLABench 的 `select_toy` 任务验证完整的视觉-语言-动作闭环：理解指令、定位目标、抓取、搬运并放入指定容器。

当前复现状态为**功能链路和正式评估链路均已打通**：模型能够抓起指定玩具、搬运并在部分回合完成最终放置。固定随机实例重复控制 20 次得到 `1/20`（5.0%），而官方 `track_1_in_distribution` 的 `select_toy` 前 20 个固定配置得到 `13/20`（65.0%，Wilson 95% CI 43.3%-81.9%）。前者是稳定性诊断，后者才是当前覆盖多目标、多布局的主结果；二者都不是官方完整 benchmark 成绩。

| 项目 | 状态 |
|---|---|
| checkpoint 离线加载 | 完成 |
| Tesla M40 推理兼容 | 完成 |
| `dm_control.viewer` 实时可视化 | 完成 |
| 自然语言指令读取 | 完成 |
| 三相机输入语义对齐 | 完成 |
| EEF 动作到 Franka 关节控制 | 完成 |
| 指定物体抓取与抬起 | 已观察到 |
| 搬运到目标容器附近 | 已观察到 |
| 稳定放入并完成任务 | 官方 track-1 前20配置中完成 13/20 |
| 多回合成功率统计 | 固定实例 1/20；官方 track-1 前20配置 13/20 |

### 10.2 实验环境

| 项目 | 配置 |
|---|---|
| GPU | Tesla M40 24 GB（Maxwell，Compute Capability 5.2） |
| CPU | Intel i7-8700（6C/12T） |
| 内存 | 15 GiB，GUI 场景下存在较高 swap 压力 |
| Python | 3.12，`/extdata/hdd2/lerobot-smolvla/.venv` |
| MuJoCo | 3.2.2 |
| 仿真平台 | VLABench（基于 `dm_control` / MuJoCo） |
| 机械臂模型 | VLABench `franka`：Franka 单臂，7 个关节 + 两个平行夹爪关节（控制量共 9 维） |
| Viewer | `dm_control.viewer` + GLFW |
| VLABench | `/extdata/hdd2/lerobot-smolvla/sim/VLABench` |
| checkpoint | `lerobot/smolvla_vlabench`，本地 HF snapshot |
| 训练数据 | `lerobot/vlabench_unified`：10,977 episodes、3,114,872 frames、295 tasks、10 FPS |

本地 checkpoint 的 `model.safetensors` 参数统计如下。它将 VLM 裁剪为 16 层，因此实际存储参数量低于通用架构描述中的约 575M：

| 组件 | 参数量 |
|---|---:|
| VLM | 350,165,248（350.165M） |
| `lm_expert` | 98,245,824（98.246M） |
| 状态/动作适配层 | 1,635,104（1.635M） |
| Expert + 适配层 | 99,880,928（99.881M） |
| checkpoint 总计 | 450,046,176（450.046M） |

checkpoint 共包含 500 个张量，其中 BF16 参数 446,772,624，FP32 参数 3,273,552。

Tesla M40 不支持原生 BF16 GEMM。SmolVLA 的 eager attention 在聚合 value states 时会触发 BF16 batched matmul，因此对 Ampere 之前的 CUDA 设备增加了 FP32 matmul 回退，再转换回原 dtype。该修改位于 `src/lerobot/policies/smolvla/smolvlm_with_expert.py`。

桌面 OpenGL 需要预加载系统 `libstdc++`：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

原生 `mujoco.viewer` 在当前 MuJoCo 3.2.2/VLABench 模型组合下会触发 `mjv_makeSceneState: mjvSceneState buffer is not fully used`，因此稳定可视化路径是 `--viewer dm-control`，不是原生 Viewer。

### 10.3 基准复现命令

```bash
cd /home/aitech/Workspace/VLA/lerobot

VLABENCH_ROOT=/extdata/hdd2/lerobot-smolvla/sim/VLABench/VLABench \
HF_HOME=/extdata/hdd2/lerobot-smolvla/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MUJOCO_GL=glfw \
DISPLAY=:0 \
XAUTHORITY=/home/aitech/.Xauthority \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LD_LIBRARY_PATH=/extdata/hdd2/lerobot-smolvla/.venv/lib/python3.12/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-} \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
/extdata/hdd2/lerobot-smolvla/.venv/bin/python \
examples/vlabench/run_smolvla_viewer.py \
  --viewer dm-control \
  --task select_toy \
  --policy-path /extdata/hdd2/lerobot-smolvla/hf-cache/hub/models--lerobot--smolvla_vlabench/snapshots/4fd586e12dc14b04d9d606ddbb77448df4f0ff29 \
  --device cuda \
  --seed 1000 \
  --n-action-steps 50 \
  --render-resolution 256 \
  --window-width 960 \
  --window-height 720 \
  --log-actions-every 1
```

该命令保留 VLABench 的默认物理参数，不再增加无效的显式覆盖。实测默认值为：

```text
physics timestep = 0.001 s
control timestep = 0.1 s
physics substeps = 100
solver = Newton
integrator = implicitfast
iterations = 100
tolerance = 1e-8
gravity_z = -9.81
```

### 10.4 已定位并修复的问题

#### 1. VLABench 指令读取错误

VLABench 动态生成的自然语言指令由 `task.get_instruction()` 提供。只读取旧属性会退化成任务名 `select_toy`，使 VLA 缺少具体目标。环境 wrapper 现在优先使用 `get_instruction()`；实验中实际指令为：

```text
Put the spiderman into the giftbox_seen
```

#### 2. 三相机语义映射错误（主要问题）

VLABench Franka 的原始相机顺序为：

```text
0 = right
1 = left
2 = forward
3 = wrist
```

错误映射曾将三路输入组织为：

```text
image        <- right
second_image <- left
wrist_image  <- forward
```

这会让模型把前视图当成腕部视图，导致物体高度、抓取距离和近距离接触判断失真。结合 `vlabench_unified` 训练集图像统计与真实环境四路画面统计，正确映射确定为：

```text
image        <- forward
second_image <- right
wrist_image  <- wrist
```

修复后 wrapper 按 MuJoCo 相机名称解析语义，并以 VLABench 标准顺序作为回退。真实环境验证中，映射后的三路图像均值与原始 `forward/right/wrist` 图像完全一致。该修复是从“无法接触物体”进展到“成功抓起并搬运”的决定性变化，不是物理参数调优带来的偶然结果。

#### 3. EEF 动作不能直接写入 MuJoCo ctrl

checkpoint 输出 7 维绝对 EEF 动作：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

Franka 的 MuJoCo actuator 需要 9 维控制：7 个关节目标加 2 个夹爪指目标。环境 wrapper 完成以下转换：

1. robot-base position 加回机器人世界坐标偏移。
2. Euler XYZ 转换为 MuJoCo 使用的 WXYZ quaternion。
3. 使用 `qpos_from_site_pose` 求解 7 个 Franka 关节目标。
4. 将数据集约定的 `gripper=1/0` 转成手指位置 `0.04/0.0`。

独立控制实验给末端发送低 10 cm 的目标后，实际 robot-frame `z` 在四个控制周期内从 `0.432` 降到 `0.331`，最终误差约 1 mm。因此机械臂下探限制不来自坐标系、IK 或 MuJoCo 动力学。

#### 4. Viewer 与动作诊断不足

新增 `examples/vlabench/run_smolvla_viewer.py`，在 `dm_control.viewer` 中运行相同的 policy processor 和 EEF-to-joint 控制路径。动作日志同时输出：

```text
current_eef  当前实际末端状态
target_eef   模型输出的绝对目标
delta_pos    目标位置减当前实际位置
ctrl         IK 后的关节和夹爪控制
```

该日志可以区分两类问题：`target_eef` 要求下降但 `current_eef` 不跟随属于控制问题；两者紧密跟随但目标本身不够低或不够高则属于策略输出问题。

### 10.5 `n_action_steps` 对照实验

checkpoint 的 `chunk_size=50`，单次最多生成 50 步动作，因此 `n_action_steps` 不能超过 50。

| `n_action_steps` | 实验现象 | 结论 |
|---:|---|---|
| 20 | 无法到达目标 | 重新规划过于频繁，破坏长动作轨迹连续性 |
| 30 | 无法到达目标 | 与 20 类似，动作块在完成接近前被替换 |
| 40 | 可以到达目标，但抓不住 | 接近阶段基本完成，抓取时序仍被重新规划打断 |
| 50 | 表现最好；抓起目标并搬运到盒子附近 | 与训练动作块长度一致，应作为当前基准 |
| 60 | 配置报错 | `n_action_steps > chunk_size`，不是 GPU 算力错误 |

该 checkpoint 更依赖完整 50 步动作块内部的连贯性。缩短 horizon 并没有带来更好的闭环纠偏，反而让不同 Flow Matching 采样块在接近和闭爪阶段产生不一致。因此当前实验应固定 `n_action_steps=50`。

### 10.6 已观察到的任务行为

在 `select_toy`、seed 1000、`n_action_steps=50` 的实验中观察到以下阶段：

1. 根据指令选择 Spiderman，而不是场景中的其他玩具。
2. 末端移动到目标上方并向下接近。
3. 夹爪闭合并成功抓起目标。
4. 抬起目标并向 `giftbox_seen` 搬运。
5. 到达盒子附近，但未充分抬高并对准盒口，最终没有完全放入。

抓取和搬运成功说明相机语义修复后，模型已经能够执行多阶段任务。放置阶段日志中 `current_eef` 与 `target_eef` 通常只有毫米到厘米级差异，表明机械臂基本执行了模型要求；失败原因是模型没有生成足够的抬高、水平居中和最终下降动作，而不是控制器无法达到目标。

推理时间呈明显的动作队列特征：

```text
首次推理/预热：约 10.3 s
后续每个新 50 步动作块：约 2.0-2.3 s
队列内单步取动作：约 0.001-0.003 s
```

Viewer 中推理期间仿真暂停，因此较长推理延迟主要影响交互帧率，不会让物理世界在等待期间继续演化。

### 10.7 同一随机实例的稳定性诊断

新增可重复执行的评估入口：

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate

N_EPISODES=20 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/smolvla_vlabench_select_toy_n20_final \
bash ./examples/vlabench/run_smolvla_vlabench_eval.sh
```

这条命令没有启用官方 deterministic track。VLABench 在首次加载环境时抽取一次目标和布局，后续 reset 只重置状态，因此 N=20 实际上是同一个随机实例被重复控制 20 次，适合诊断控制稳定性，不适合代表多目标 benchmark。固定配置为 `select_toy`、seed 1000、episode length 500、渲染分辨率 256、单环境串行评估和 `n_action_steps=50`。本机结果：

| 指标 | 实测值 |
|---|---:|
| 回合数 | 20 |
| 成功序列 | `[F,F,F,T,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F]` |
| 成功率 | **5.0%（1/20）** |
| Wilson 95% 置信区间 | **0.9%-23.6%** |
| 总评估时间 | 4527.39 s（约 75.46 min） |
| 平均每回合时间 | 226.37 s |
| `avg_sum_reward` | 0.0 |
| `avg_max_reward` | 0.0 |
| 观察到的 GPU framebuffer | 约 1169-1176 MiB |
| 观察到的进程 RSS | 约 5.20 GiB |
| 观察到的 CPU 占用 | 约 74.5% |

成功发生在 episode 3。原始结果与视频位于：

```text
outputs/eval/smolvla_vlabench_select_toy_n20_final/eval_info.json
outputs/eval/smolvla_vlabench_select_toy_n20_final/videos/select_toy_0/
```

此前 N=5 初测为 20%（1/5），对应相同实例的前 5 次控制；因此该数字不能与多布局、多目标 benchmark 混用。表中的资源数据是运行快照，不是峰值资源统计。

VLABench 在成功回合中仍报告 reward 为 0，因此这里的成功率采用环境 success predicate，而不从 reward 推导。这说明当前 wrapper 的成功终止判定可用，但 reward 汇总与成功判定不一致，后续若用 reward 比较算法需要先修正或明确该差异。

该 `5.0%` 只能说明当前 checkpoint 在这一固定实例上的控制稳定性较低。它不能直接与 Pi0/Pi0.5 的 LIBERO 成功率比较，因为环境、任务、checkpoint、训练数据和控制接口均不同。

### 10.8 官方 deterministic track（主结果）

wrapper 新增 `--env.deterministic_track`。seed `1000+i` 映射到 track 中第 `i` 个 episode，并为每个 episode 重新构建 VLABench 场景，因此目标物体、干扰物和布局按官方配置变化。

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate
VLABENCH_DETERMINISTIC_TRACK=/extdata/hdd2/lerobot-smolvla/sim/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json \
VLABENCH_TRACK_SEED_OFFSET=1000 N_EPISODES=20 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/smolvla_vlabench_select_toy_track1_n20 \
bash ./examples/vlabench/run_smolvla_vlabench_eval.sh
```

`select_toy` track-1 前 20 个官方固定配置本机结果：

| 指标 | 结果 |
|---|---:|
| 回合数 | 20 |
| 成功回合 | 13 |
| 成功率 | **65.0%（13/20）** |
| Wilson 95% CI | **43.3%-81.9%** |
| 总评估时间 | 3246.41 s（约 54.1 min） |
| 平均每回合 | 162.32 s |

阶段漏斗：

| 阶段 | 通过 | 阶段成功率 |
|---|---:|---:|
| 目标物体被抓持 | 20/20 | **100.0%** |
| 目标物体抬升至少 5 cm | 17/20 | **85.0%** |
| 环境 success predicate 判定放置完成 | 13/20 | **65.0%** |
| 曾抓到错误物体 | 2/20 | **10.0%** |

`stage_target_reached` 是末端到目标实体中心小于 10 cm 的严格接近代理（本批为 4/20）；玩具实体有体积，夹爪接触点不等于实体中心，所以该字段不能单独解释为视觉识别成功率。可靠阶段结论是：模型几乎总能抓到目标，少数回合未抬升，剩余主要失败发生在容器内最终放置。每回合 JSON 保存 instruction、target_entity、target_container、track config index、最小距离和最大抬升高度。

7 个失败回合可进一步分解为：

| 失败位置 | 回合数 | 目标 |
|---|---:|---|
| 抓持后未抬升 5 cm | 3 | `donald`、`alien`、`sanji` |
| 已抬升但未完成放置 | 4 | `mickey`、`ironman`、`buzz_lightyear`、`ironman` |
| 期间曾误抓其他物体 | 2 | `mickey`、`ironman`（包含在上述放置失败中） |

因此当前 20 回合不支持“主要失败在识别”的判断。更准确的描述是：目标选择和抓取链路已经可用，约 15% 的回合损失在稳定抬升，另约 20% 损失在抬升后的搬运/放置。由于没有模型内部视觉分类标签，`抓持目标` 是识别与定位能力的任务级代理，而不是纯视觉识别准确率。

原始结果：`outputs/eval/smolvla_vlabench_select_toy_track1_n20/eval_info.json`。

### 10.9 当前结论

1. **主要集成错误已经解决。** 相机槽位错配是此前无法接触物体的核心原因；修复后行为发生了确定性的质量跃迁。
2. **物理和控制链路基本正确。** 默认 MuJoCo 参数可以正常下探，IK 对绝对 EEF 目标的跟踪误差很小。
3. **`n_action_steps=50` 是当前 checkpoint 的最佳设置。** 更短 horizon 会破坏已学习的动作块连续性，60 则超出模型结构上限。
4. **官方多目标结果明显高于固定实例诊断。** deterministic track 为 13/20（65.0%）；阶段指标显示损失集中在抬升（3 回合）和最终放置（4 回合），而不是目标抓取。
5. **当前成果已经超过单纯功能性复现。** 实时 Viewer、官方固定配置评估、成功判定、阶段 JSON 和视频证据均已打通；下一阶段应扩大 track 和任务覆盖。

### 10.10 后续改进思路

按优先级推进：

1. **扩大官方覆盖。** 当前完成 `track_1_in_distribution` 的 `select_toy` 前 20 个配置；下一步可跑完整 50 配置及其他 deterministic tracks/任务。
2. **细化阶段化指标。** 当前已记录 instruction、目标物体/容器、抓取、抬升、误抓和放置；后续可增加容器中心距离、松爪时刻和失败阶段标签。
3. **分析放置阶段动作。** 对进入盒子附近后的 `target_eef`、夹爪状态和物体位置作轨迹图，确认失败来自抬升不足、水平偏差、下降时机还是提前松爪。
4. **优先补充放置数据。** 若多 seed 都能抓取但稳定卡在放置阶段，应增加“抓取后抬高、移动到容器中心、下降、松爪”的示范，针对当前 checkpoint 继续微调，而不是继续调整 MuJoCo solver。
5. **再考虑控制层增强。** 只有在确认模型目标合理但执行误差较大时，才考虑动作平滑、阶段感知重规划或安全高度约束；当前日志不支持把控制器作为首要瓶颈。
6. **硬件升级属于性能优化。** Ampere 及更新 GPU 可以消除 M40 的 BF16 回退并显著降低推理时间，但不会直接修复最终放置策略。

### 10.11 相关实现与测试

| 文件 | 作用 |
|---|---|
| `src/lerobot/envs/vlabench.py` | 指令提取、相机语义映射、状态转换、EEF-to-joint IK |
| `examples/vlabench/run_smolvla_viewer.py` | 交互 Viewer、策略推理、物理参数显示、动作诊断 |
| `examples/vlabench/run_smolvla_vlabench_eval.sh` | 离线 checkpoint 的正式多回合 `lerobot-eval` 入口 |
| `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | Tesla M40 BF16 attention 回退 |
| `tests/envs/test_vlabench.py` | 指令和相机映射测试 |
| `tests/envs/test_vlabench_viewer.py` | Viewer callback、动作队列重置和参数解析测试 |
| `tests/policies/smolvla/test_smolvlm_with_expert.py` | pre-Ampere attention 回退测试 |

当前定向环境与 Viewer 测试结果为 `10 passed`。真实环境还额外验证了三路图像映射和 10 cm EEF 下探控制。完整 GPU policy rollout 由桌面 M40 会话执行；工具沙箱不可见 CUDA，不能替代该硬件验证。

---

## 11. 简历版本

### One-line

基于 SmolVLM2-500M，设计 Cross-Attention + Interleaved Self-Attention 的轻量级机器人 VLA（~575M 参数），Flow Matching 生成50步动作块，推理 KV Cache 复用将 Prefix 计算开销降低10×；VLABench `select_toy` 固定实例诊断为5.0%，官方 track-1 前20配置为65.0%。

### 3-bullet

- 实现 SmolVLA 策略：SmolVLM2-500M 编码图像/语言/状态输出 KV Cache，Action Expert（VLM 宽度×0.75）通过 Cross-Attention 查询 Cache，Flow Matching 10步去噪生成连贯50步动作序列
- 设计交替注意力机制（每2层插入 Expert self-attention），解决纯 Cross-Attention 下动作 token 无法相互通信的动作块不连贯问题；设计非对称 att_mask 将机器人状态嵌入 Prefix，避免状态信息污染 VLM 图文表征
- 推理阶段实现 Prefix KV Cache 复用：VLM 前向只执行一次，10步 ODE 去噪仅处理50个 action token；Tesla M40 实测首次预热约10.3秒，后续50步动作块约2.0-2.3秒，并集成 LeRobot 统一训练与评估框架

### STAR（面试版）

**Situation：** 主流 VLA 模型（Pi0 ~2.3B，OpenVLA ~7B）在消费级 GPU 上推理延迟难以满足实时控制（10-30 Hz）要求，限制实际部署。

**Task：** 设计轻量级 VLA，大幅降低参数量的同时保留视觉-语言-动作端到端学习能力，支持 LeRobot 标准任务评估。

**Action：**
以 SmolVLM2-500M 替代 PaliGemma-2B，将总参数压缩至 ~575M。设计 Cross-Attention 机制：将 Expert 的 k_proj/v_proj 替换为 `Linear(vlm_kv_dim → expert_kv_dim)`，使小 Expert 能跨维度查询大 VLM 的 K/V，解除 Pi0 Shared Attention 要求两者 head_dim 相同的约束。引入交替 Self/Cross-Attention（每2层），兼顾动作内部连贯与语义查询。将 state 嵌入 Prefix 并设计非对称掩码，让 Expert 通过 KV Cache 感知当前机器人状态。推理时分离 Prefix 前向（填充 KV Cache）和去噪循环，10步 ODE 只处理 50 个 action token。

**Result：**
通用架构参数量约575M，相比 Pi0 约2.3B 减少约4×；本地16层 VLABench checkpoint 实测为450.046M，其中 Expert 与适配层为99.881M。Tesla M40 上后续50步动作块推理约2.0-2.3秒；固定随机实例诊断为5.0%（1/20），官方 track-1 前20配置为65.0%（13/20）。

### 量化缺口

| 指标 | 状态 |
|---|---|
| 推理延迟（单帧，A100/4090） | [needs measurement] |
| SO-101 任务成功率 vs Pi0 基线 | [needs measurement] |
| LIBERO benchmark 成功率 | [needs measurement] |
| VLABench `select_toy` 固定实例稳定性诊断 | 5.0%（1/20；Wilson 95% CI 0.9%-23.6%） |
| VLABench `select_toy` 官方 track-1 前20配置 | **65.0%（13/20；Wilson 95% CI 43.3%-81.9%）** |
| 本地 checkpoint 总参数量 | 450.046M |
| Expert 实际参数量 | `lm_expert` 98.246M；含适配层 99.881M |
| 训练收敛所需步数（smolvla_base 微调） | [needs measurement] |
