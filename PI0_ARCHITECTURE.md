# Pi0 架构深度解析

> 个人学习笔记，基于 `src/lerobot/policies/pi0/` 源码整理。
> 官方用户指南见 `docs/source/policy_pi0_README.md`。

---

## 1. 项目目标

Pi0 是一个**视觉-语言-动作模型（VLA）**，让机器人根据摄像头图像和自然语言指令生成连续的关节控制序列。

| 项目 | 说明 |
|---|---|
| 输入 | 摄像头图像（224×224）+ 自然语言指令 + 机器人关节状态（32维） |
| 输出 | 50帧动作序列（50×32维），一次规划50步后重新观测并规划 |
| 核心机制 | Flow Matching（从噪声去噪到动作）+ Shared Attention（PaliGemma与Expert共享注意力） |

---

## 2. 基础组件

### 2.1 PaliGemma（视觉语言主干）

PaliGemma = Google VLM = **SigLIP**（看图）+ **模态投影层**（连接）+ **Gemma-2B**（理解语言）

#### SigLIP（视觉编码器）

- 类型：Vision Transformer，Sigmoid Loss for Image-Language Pretraining
- 输入：224×224 RGB 图像
- 处理：切成 16×16 = **256 个 patch**（每个 14×14 像素）→ 线性投影 → Transformer 自注意力
- 输出：**256 个向量，每个 1024 维**
- 预训练：数十亿图文对，视觉特征天然对齐语言语义空间
- Pi0 中：**默认冻结**（`freeze_vision_encoder=False` 但通常设 True），梯度不传入

#### 模态投影层

- 类型：线性投影（MLP）
- 作用：SigLIP 1024维 → Gemma 所需的 **2048维**
- 在 PaliGemma 预训练时已训好，Pi0 训练时可选冻结

#### Gemma-2B（语言+跨模态理解）

| 参数 | 值 |
|---|---|
| 层数 | 18层 Transformer |
| hidden_size | 2048 |
| MLP dim | 16384 |
| num_heads | 8，head_dim=256 |
| 输入序列长度 | 304（256图像patch + 48文字token） |

- 每层：多头自注意力（图文**双向互看**）+ FFN（MLP）
- 输出：304个融合图文语义的上下文向量
- Pi0中角色：提供对"观测+任务"的深度理解，供 Expert 参考

### 2.2 动作 Expert（Gemma-300M）

| 参数 | 值 |
|---|---|
| 层数 | 18层（与 PaliGemma 一一对应，用于 Shared Attention） |
| hidden_size | 1024 |
| MLP dim | 4096 |
| num_heads | 8，head_dim=128 |
| embed_tokens | None（不处理文字token，只处理动作） |

- 从随机初始化开始，Pi0训练时**全部参与梯度更新**
- 任务：基于 PaliGemma 的理解，对带噪动作预测去噪速度

---

## 3. 系统整体架构

### 3.1 序列设计：Prefix / Suffix

```
完整序列 = [Prefix] + [Suffix]

Prefix（由 PaliGemma 处理，304个向量，2048维）：
  [图像 patch × 256] + [语言 token × 48]

Suffix（由 Expert 处理，51个向量，1024维）：
  [状态 token × 1] + [动作 token × 50]
```

Prefix 和 Suffix 维度不同（2048 vs 1024），但在 Shared Attention 中通过各自的 Q/K/V 投影层对齐到同一 head_dim 进行计算。

### 3.2 共享注意力机制（Shared Attention）

Pi0 的核心结构：PaliGemma 的18层和 Expert 的18层**一一对应 zip**，每层合并计算注意力。

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
```

**关键效果**：每一层 Expert 的动作 token 都能"看到"PaliGemma 当前层的图像和语言理解，实现深度融合。

### 3.3 注意力掩码规则

```
               图像patch  语言token  状态token  动作token[i]
图像 patch       ✓          ✓          ✗           ✗
语言 token       ✓          ✓          ✗           ✗
状态 token       ✓          ✓          ✓           ✗
动作 token[i]    ✓          ✓          ✓        ✓(j≤i)
```

- **Prefix 双向**：图文互看，学习跨模态对齐
- **Prefix 不看 Suffix**：保证 Prefix KV Cache 有效（Prefix表示不依赖动作）
- **Suffix 看全部 Prefix**：动作生成必须感知图像+指令
- **Suffix 内部 Causal**：动作token[i]不能超前知道动作token[i+1]

---

## 4. 核心创新点

### 4.1 共享注意力（Shared Attention）

- **解决的问题**：若用 Cross-Attention（Expert Q → PaliGemma KV），Expert 在每层只能看到 PaliGemma 的输出，而无法在注意力层内与 PaliGemma 的中间表示**双向交互**
- **机制**：zip PaliGemma层和Expert层，在每层内拼接 Q/K/V 做统一注意力，输出再拆分
- **效果**：Expert 在每一层都能"读取"和"影响" PaliGemma 当前层的表示，深度融合 vs SmolVLA 的 Cross-Attention（单向）

### 4.2 Flow Matching（动作生成机制）

- **解决的问题**：动作空间高维（50×32=1600维），直接回归容易陷入均值模糊；扩散模型步数多（50~1000步），推理慢
- **机制**：

```
定义两个端点：
  t=0 → clean action（真实动作）
  t=1 → pure noise（高斯噪声）

中间状态线性插值：
  x_t = t × noise + (1-t) × action

训练目标（velocity）：
  u_t = noise - action   ← 解析解，不需要估计

损失函数：
  L = MSE(v_θ(x_t, t, context), u_t)

推理（Euler ODE，10步）：
  x_{t-0.1} = x_t + (-0.1) × v_θ(x_t, t, context)
  从 t=1.0 → t=0.0，每步修正一次方向
```

- **效果对比扩散模型**：

| | Flow Matching | 扩散模型（DDPM） |
|---|---|---|
| 路径 | 直线插值 | 随机游走 |
| 推理步数 | **10步** | 50~1000步 |
| 训练目标 | velocity（解析） | 噪声（估计） |
| 路径确定性 | ODE（确定） | SDE（随机） |

### 4.3 KV Cache 推理优化

- **解决的问题**：10步去噪中，图像+语言内容固定不变，重复计算 PaliGemma 的 K/V 是纯浪费
- **机制**：推理开始时，对 Prefix（图像+语言）做一次前向，缓存 18层×2（K+V）共36个矩阵；后续10步去噪只计算 Suffix（状态+动作+时间）的 Q/K/V，Prefix 的 KV 直接从缓存读取
- **效果**：推理速度约提升 10×（相当于只多了 Expert 处理 Suffix 的10步开销）

### 4.4 时间步非均匀采样 Beta(1.5, 1.0)

- **解决的问题**：均匀采样导致模型在"容易的小t"和"困难的大t"上花相同力气，但大t（噪声多）才是瓶颈
- **机制**：t ~ Beta(1.5, 1.0)，概率密度在 t≈0.7~1.0 区间最高
- **效果**：模型在噪声最多时仍能找到正确方向，整体动作质量更好

---

## 5. 训练 vs 推理结构对比

| 方面 | 训练 | 推理 |
|---|---|---|
| clean_action | **有**（来自数据集） | **无**（要生成的目标） |
| 去噪次数 | 随机取**1个t**，做1次前向 | 从t=1.0→0.0，做**10次**前向 |
| 噪声 | 随机采样 + Beta(1.5,1.0)时间步 | 随机初始化 x_1，逐步去噪 |
| KV Cache | **不使用**（use_cache=False） | **使用**（Prefix只算1次） |
| SigLIP梯度 | **截断**（冻结时） | 不涉及 |
| 输出用途 | 计算 MSE loss，反向传播 | Euler步进更新 x_t，最终执行 |
| 样本加权 | 可选 RA-BC（reduction="none"） | 不涉及 |

---

## 6. 完整端到端流程（含 Flow Matching）

### 训练

```
① 编码图像
   image [B,H,W,3] → SigLIP → 模态投影
   → img_emb [B, 256, 2048]

② 编码语言
   language (str) → Tokenizer → embed_language
   → lang_emb [B, 48, 2048]
   Prefix = cat(img_emb, lang_emb)  [B, 304, 2048]

③ 编码状态（进入 Suffix）
   state [B,32] → state_proj: Linear(32→1024)
   → state_tok [B, 1, 1024]

④ Flow Matching 构造带噪动作（训练专有）
   noise ~ N(0,1)                  [B, 50, 32]
   t     ~ Beta(1.5, 1.0)
   x_t   = t×noise + (1-t)×action  ← 带噪中间态
   u_t   = noise - action           ← 训练目标（速度场）

⑤ 编码带噪动作 + 时间步融合
   x_t → action_in_proj: Linear(32→1024) → action_emb
   sinusoidal(t) → action_time_mlp → time_emb
   action_tok = action_emb + time_emb
   Suffix = cat(state_tok, action_tok)  [B, 51, 1024]

⑥ 模型前向（Shared Attention，18 层）
   [Prefix | Suffix]
   → PaliGemma ↔ Expert 逐层 Shared Attention（Q/K/V 拼接）
   → suffix_out[-50:] → action_out_proj: Linear(1024→32)
   → v_t [B, 50, 32]

⑦ 计算损失
   Loss = MSE(v_t, u_t)
```

### 推理

```
image + language + state
    ↓ embed_prefix()
Prefix KV Cache（一次计算，缓存18层 KV）

x_t = N(0,1)  [t=1.0]

循环10次（Euler ODE）：
  t: 1.0 → 0.9 → ... → 0.1
  ┌──────────────────────────────────────────────────────────┐
  │ embed_suffix(state, x_t, t)                              │
  │   action_in_proj(x_t) + action_time_mlp(sinusoidal(t))  │
  │   → Suffix [B,51,1024]                                   │
  │ Shared Attention(Suffix Q, Prefix KV Cache)              │
  │   → suffix_out → action_out_proj → v_t [B,50,32]        │
  └──────────────────────────────────────────────────────────┘
  x_t = x_t + (-0.1) × v_t

x_0 [B,50,32] = 干净动作 → 反归一化 → 执行
```

---

## 7. 数据流

### 6.1 训练数据流

```
数据集样本：(image, instruction, robot_state, clean_action)

image [B, H, W, 3]
  → _preprocess_images()：resize+pad到224×224，归一化到[-1,1]
  → PaliGemmaWithExpertModel.embed_image()：SigLIP → [B, 256, 1024]
  → 模态投影层 → [B, 256, 2048]

instruction（字符串）
  → Tokenizer → token_ids [B, 48]
  → embed_language_tokens() → [B, 48, 2048]

Prefix = concat → [B, 304, 2048]

robot_state [B, 32] → pad到max_state_dim=32
  → state_proj → [B, 1, 1024]

noise ~ N(0,1) [B, 50, 32]
time ~ Beta(1.5,1.0) [B, 1, 1]

x_t = time×noise + (1-time)×clean_action   [B, 50, 32]
u_t = noise - clean_action                  [B, 50, 32] ← 训练目标

x_t → action_in_proj → [B, 50, 1024]
time → 正余弦编码 → [B, 1, 1024]
concat(action_vec, time_vec) → action_time_mlp → [B, 50, 1024]

Suffix = concat(state_vec, action_vecs) → [B, 51, 1024]

[Prefix | Suffix] → 18层 Shared Attention → suffix_out [B, 51, 1024]

suffix_out[-50:] → action_out_proj → v_t [B, 50, 32]

Loss = MSE(v_t, u_t)   标量
```

### 6.2 推理数据流

```
输入：(image, instruction, robot_state)

Step 0：Prefix 单次计算
  image + instruction → embed_prefix() → prefix_embs [B, 304, 2048]
  → PaliGemmaWithExpertModel.forward([prefix_embs, None], use_cache=True)
  → past_key_values（18层KV缓存，不再修改）

初始化：
  x_t = N(0,1) [B, 50, 32]（t=1.0）

循环 10 次（step=0..9）：
  time = 1.0 + step × (-0.1)   # 1.0, 0.9, ..., 0.1

  denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)：
    → embed_suffix(state, x_t, time) → suffix_embs [B, 51, 1024]
    → PaliGemmaWithExpertModel.forward([None, suffix_embs], past_key_values=kv)
      # Expert处理 Suffix，K/V从缓存读取 Prefix部分
    → suffix_out[-50:] → action_out_proj → v_t [B, 50, 32]

  x_t = x_t + (-0.1) × v_t   # Euler步进

输出：x_t [B, 50, 32]（t=0，去噪完成）
  → × std + mean（反归一化）
  → 前 n_action_steps 帧放入 action_queue
  → 每次 select_action() 弹出1帧发给机器人
  → 队空后重新调用 predict_action_chunk()（Receding Horizon Control）
```

---

## 7. 冻结策略

| 组件 | 默认 | 原因 |
|---|---|---|
| SigLIP | 可冻结（`freeze_vision_encoder`） | 预训练视觉特征已够好，机器人数据量不足以改善 |
| 模态投影层 | 随 PaliGemma | 与 SigLIP 绑定 |
| Gemma-2B 语言层 | 可冻结（`train_expert_only`） | 语言理解能力来自大规模预训练，小数据微调易遗忘 |
| 动作 Expert（全部） | **始终训练** | 从随机初始化，必须从头学 |
| 投影层（state/action/time MLP） | **始终训练** | Pi0 新增模块，无预训练权重 |

**两阶段训练策略（典型做法）：**

```
阶段1（大规模预训练）：
  freeze_vision_encoder = True
  train_expert_only = True
  → 在海量多机器人数据上，只训 Expert + 投影层
  → 建立"动作生成"的基础能力

阶段2（任务微调）：
  freeze_vision_encoder = True
  train_expert_only = False（解冻 Gemma 层，极小学习率）
  → 针对特定机器人/特定任务精调
  → 提升任务成功率
```

---

## 8. 关键配置参数（`PI0Config`）

```python
# 模型规模
paligemma_variant     = "gemma_2b"    # PaliGemma 骨干
action_expert_variant = "gemma_300m"  # 动作 Expert 规模

# 动作维度（不足则 zero-pad，支持不同形态机器人）
max_state_dim  = 32
max_action_dim = 32

# 推理控制
chunk_size           = 50    # 一次预测 50 帧
n_action_steps       = 50    # 执行 50 帧后重新规划
num_inference_steps  = 10    # 去噪步数

# 时间步采样
time_sampling_beta_alpha = 1.5    # Beta(1.5, 1.0)，偏向大 t
time_sampling_beta_beta  = 1.0
time_sampling_offset     = 0.001  # 避免 t=0 精确值
time_sampling_scale      = 0.999  # 避免 t=1 精确值

# 优化器（use_policy_training_preset=True 时自动应用）
optimizer_lr             = 2.5e-5
optimizer_grad_clip_norm = 1.0
scheduler_warmup_steps   = 1000
scheduler_decay_steps    = 30000
scheduler_decay_lr       = 2.5e-6   # CosineDecay 最终 lr

# 冻结控制
freeze_vision_encoder = False   # True = 冻结 SigLIP
train_expert_only     = False   # True = 只训 Expert + 投影层
```

---

## 9. 类结构速查

```
PI0Policy  （src/lerobot/policies/pi0/modeling_pi0.py）
  ├── select_action(batch)        → 单步动作（维护 action_queue）
  ├── predict_action_chunk(batch) → 完整推理一次（调用 sample_actions）
  └── forward(batch)              → 训练前向，返回 loss

  └── PI0Pytorch
        ├── forward(...)          → Flow Matching 训练（MSE loss）
        ├── sample_actions(...)   → 10 步去噪推理
        ├── denoise_step(...)     → 单步去噪（复用 KV Cache）
        ├── embed_prefix(...)     → 图像 + 语言 → prefix_embs
        └── embed_suffix(...)     → state + action + time → suffix_embs

        └── PaliGemmaWithExpertModel
              ├── embed_image(...)           → SigLIP 前向
              ├── embed_language_tokens(...) → Gemma Embedding
              └── forward(inputs_embeds=[prefix, suffix])
                    ├── prefix only   → 返回 KV Cache（推理 Step 0）
                    ├── suffix only   → 复用 KV Cache（推理 Step 1~10）
                    └── prefix+suffix → Shared Attention（训练）
```

---

## 10. 本地复现、实现与实验结果

### 10.1 复现目标与最终状态

本地复现目标不是只验证 Pi0 类能够实例化，而是完成以下端到端链路：

```text
LIBERO 场景与固定初始状态
  → agentview + wrist camera + robot state
  → LeRobot 环境预处理
  → Pi0 tokenizer / image processor / normalization
  → 10 步 Flow Matching 推理
  → 50 帧 action chunk
  → LIBERO relative OSC controller
  → 实时 MuJoCo 画面与任务成功判定
```

当前状态：

- `lerobot/pi0_libero_finetuned` 已在本地完整加载。
- Pi0 + LIBERO 官方任务已完成实时交互复现。
- `libero_object task 0` 中，Pi0 能抓取 alphabet soup 并放入 basket。
- 实测动作连续性和平滑程度明显优于此前本机的 SmolVLA + VLABench rollout。
- 实时 Viewer、自由观察相机、推理进度日志、成功后画面保留和正式批量评测入口均已实现。
- 已使用标准 `lerobot-eval` 完成 task 0 的 20-episode 评测，成功率为 **65.0%（13/20）**。

### 10.2 本地硬件与运行约束

| 项目 | 本地配置/处理 |
|---|---|
| GPU | Tesla M40 24 GB，compute capability 5.2 |
| 主机内存 | 约 15 GB RAM，模型加载容易产生 CPU 内存峰值 |
| BF16 | M40 不支持原生 BF16，运行时使用 FP32 |
| 仿真平台 | LIBERO（基于 robosuite / MuJoCo） |
| 机械臂模型 | Franka Panda：单臂 7 自由度 + Panda 双指夹爪；控制器为 LIBERO `OSC_POSE` / relative OSC |
| MuJoCo 渲染 | `MUJOCO_GL=glfw`、`DISPLAY=:0` |
| Viewer | Tk/Pillow 显示 + robosuite offscreen renderer |
| policy checkpoint | `/extdata/hdd2/lerobot-smolvla/hf-cache/pi0_libero_finetuned` |
| LIBERO assets | 安装于当前 `.venv` 的 `libero/libero/assets` |

EGL 在该主机上不可用，因此复现脚本统一通过桌面 GLFW/OpenGL 渲染。系统 `libstdc++` 通过 `LD_PRELOAD` 优先加载，避免 Conda/系统 OpenGL ABI 冲突。

### 10.3 低 CPU 内存 checkpoint 加载

原始加载方式容易同时保留完整 checkpoint、转换后的 state dict 和模型参数副本，在 15 GB RAM 主机上产生较高峰值。本地对 `src/lerobot/policies/pi0/modeling_pi0.py` 增加了低内存流式加载路径：

1. 从 `model.safetensors` 逐 tensor 读取。
2. 直接匹配模型目标参数。
3. 将 checkpoint BF16 tensor 转换到模型实际运行 dtype（M40 上为 FP32）。
4. 立即复制到目标 CUDA 参数，不在 CPU 内存中构建第二份完整 state dict。
5. 每 100 个 tensor 输出加载进度。

实际日志：

```text
Streaming Pi0 weights to cuda from: .../pi0_libero_finetuned/model.safetensors
Loaded 100/777 checkpoint tensors
...
Loaded 777/777 checkpoint tensors
All Pi0 keys loaded successfully with low CPU memory usage!
```

该适配解决的是本地硬件加载问题，不改变 checkpoint 权重含义或 Pi0 推理算法。

### 10.4 LIBERO 环境与官方处理链

真实行为验证使用：

```text
pi0_libero_finetuned + LIBERO
```

没有使用 `pi0_base + VLABench` 作为最终行为结论。后者缺少 VLABench 微调和相机语义标定，只能作为接口 smoke test。

实时 Viewer 保留 LeRobot 官方 Pi0/LIBERO 数据处理路径：

```python
policy_input = preprocess_observation(observation)
policy_input["task"] = [task_description]
policy_input = env_preprocessor(policy_input)
policy_input = preprocessor(policy_input)

action = policy.select_action(policy_input)
action = postprocessor(action)
action_transition = env_postprocessor({ACTION: action})
observation, reward, terminated, truncated, info = env.step(action_numpy)
```

因此 Viewer 不是绕过 LeRobot 预处理直接调用网络，也没有手工修改 Pi0 输出动作含义。

### 10.5 实时 Viewer 实现

新增或扩展的主要文件：

| 文件 | 作用 |
|---|---|
| `examples/libero/run_pi0_libero_viewer.py` | Pi0 实时推理、Tk Viewer、按键、动作队列和成功状态显示 |
| `examples/libero/run_pi0_libero_viewer.sh` | 本机环境变量、checkpoint 和默认参数启动器 |
| `examples/libero/run_pi0_libero_eval.sh` | 标准 `lerobot-eval` 批量评测入口 |
| `src/lerobot/envs/libero.py` | 成功后复位控制、自由观察相机和交互辅助接口 |
| `src/lerobot/envs/configs.py` | LIBERO Viewer/评测行为配置 |
| `src/lerobot/policies/pi0/modeling_pi0.py` | M40/低内存 checkpoint 流式加载 |

Viewer 主要能力：

- `Space`：运行/暂停。
- `I`：实验性地输入下一条语言指令。
- 默认自由观察视角；鼠标可旋转、平移和缩放。
- `C`：自由观察视角与固定策略视角切换。
- 推理期间窗口继续处理事件，不再表现为完全无响应。
- 新 action chunk 开始、持续计算、完成和队列剩余动作都会输出终端日志。
- LIBERO 成功后不会立即重置到初始状态，可观察最终放置结果。

自由观察相机只负责显示。Pi0 的 `agentview_image` 和腕部相机保持官方固定相机，不受鼠标视角影响。

### 10.6 成功后“物体弹回原位”的原因与修复

早期 Viewer 中，物体进入 basket 后立即恢复初始位置。该现象不是物理弹跳，而是环境成功后执行了：

```python
if terminated:
    self.reset()
```

此外 LIBERO 底层 `done` 本身也被 BDDL 成功谓词覆盖。交互模式现已将以下行为拆开：

- 是否报告 BDDL success。
- 是否将 success 转换为 episode termination。
- termination 后是否立即自动 reset。
- success 后是否继续若干动作，让夹爪松开并撤离。

正式批量评测仍保持原有单 episode 自动结束语义；只在交互 Viewer 中保留最终场景。

### 10.7 推理速度与 action chunk 现象

当前配置：

```text
chunk_size = 50
n_action_steps = 50
num_inference_steps = 10
```

实测日志中：

- 第一个 action chunk 通常约 14～18 秒，包含首次 CUDA/模型路径预热。
- 后续 chunk 曾观察到约 1.2～4.7 秒，受显存、系统负载和 Viewer 状态影响。
- action queue 未耗尽时，`select_action()` 只弹出缓存动作，通常约 1～3 ms。
- 每执行 50 个环境动作才重新观测并生成下一 action chunk。

典型日志：

```text
Pi0 chunk inference START | step=1 | n_action_steps=50
Pi0 chunk inference still running | step=1 | elapsed=2.0s
Pi0 chunk inference DONE | step=1 | 14.592s | action=[...]
Pi0 rollout active | step=10 | queued_actions=40
```

Pi0 的“顺滑”主要来自连续 action chunk 和 LIBERO 官方控制接口的匹配，而不是每个仿真 step 都重新运行一次完整 VLM。

### 10.8 已观察到的任务效果

验证任务：

```text
suite       = libero_object
task_id     = 0
instruction = pick up the alphabet soup and place it in the basket
```

实际行为阶段：

1. 从固定相机识别 alphabet soup 和 basket。
2. 机械臂平滑移动到目标物体。
3. 下降并闭合夹爪，成功抓取物体。
4. 抬起并移动到 basket 上方。
5. 将物体放入 basket，LIBERO `In` 谓词报告成功。
6. 继续执行少量后续动作后松爪并撤离。

该任务已经观察到完整成功 rollout。与此前 SmolVLA 的本地实验相比：

| 观察项 | SmolVLA + VLABench | Pi0 + LIBERO |
|---|---|---|
| 运动连续性 | 存在明显抖动和重复修正 | 更平滑、轨迹更连贯 |
| 抓取 | 对参数和相机映射敏感 | 官方 task 0 抓取稳定性更好 |
| 放置 | 曾接近成功或单次成功 | 已完整放入 basket 并触发 success |
| 任务接口匹配 | VLABench 适配链 | 官方 Pi0-LIBERO 微调与环境链 |

这不是严格的模型横向基准：两者使用不同环境、checkpoint、训练数据和任务。当前结论只表示在本机已运行的对应复现实验中，Pi0 的可见效果更好，不能据此直接得出架构层面的绝对优劣。

### 10.9 官方任务能力与连续换指令边界

`pi0_libero_finetuned` 并非只会抓 alphabet soup。`libero_object` 包含十个官方单物品任务，例如 milk 对应 `task_id=7`、butter 对应 `task_id=6`。每个任务应使用自己的 BDDL、物体布局和固定初始状态进行验证。

实验性 Viewer 支持完成 task 0 后按 `I` 改成 milk，但实际观察到模型仍停留或继续靠近 basket。日志确认新 prompt 已进入新的推理，旧 action queue 也已清空，因此根因不是按键或缓存，而是状态分布发生变化：

- alphabet soup 已经在 basket 中。
- 机械臂位于上一任务的终止姿态。
- milk 在 task 0 中只是干扰物，位置不同于官方 milk task 的目标位置。
- 训练演示没有包含成功后不 reset、再切换下一指令的轨迹。

所以“同一场景连续把六个物品放入 basket”不能作为当前 checkpoint 的已实现能力。可靠实现需要新的多阶段 BDDL、连续任务示范数据和 Pi0 微调，或在上层任务规划器中为各子任务选择对应环境/策略。

### 10.10 运行命令

进入环境并启动实时 Viewer：

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate
bash ./examples/libero/run_pi0_libero_viewer.sh
```

验证官方 milk task：

```bash
bash ./examples/libero/run_pi0_libero_viewer.sh \
  --task libero_object \
  --task-id 7
```

批量评测 task 0：

```bash
LIBERO_SUITE=libero_object \
LIBERO_TASK_IDS='[0]' \
N_EPISODES=20 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/pi0_libero_object_task0_n20_final \
bash ./examples/libero/run_pi0_libero_eval.sh
```

完整运行命令、任务编号和参数速查见 `VLA_REPRODUCTION_RUNBOOK.md`。

### 10.11 验证情况与待量化内容

已完成：

- 777/777 checkpoint tensors 完整加载。
- `All Pi0 keys loaded successfully` 校验通过。
- 真实 GPU + GLFW + LIBERO 一步 smoke test 通过。
- `libero_object task 0` 完整成功 rollout 已观察到。
- 标准 `lerobot-eval`（seed=1000、task 0、N=20）结果为 13 次成功、7 次失败，成功率 65.0%。
- 环境配置相关测试 13 项通过。
- Python 语法检查和 `git diff --check` 通过。

本次批量评测结果：

| 指标 | 实测值 |
|---|---:|
| suite / task | `libero_object / 0` |
| episodes | 20 |
| 成功序列 | `[F,T,T,T,F,T,T,T,F,T,T,T,F,F,T,T,T,F,F,T]` |
| 成功次数 | 13/20 |
| `pc_success` | **65.0%** |
| Wilson 95% 置信区间 | **43.3%-81.9%** |
| `avg_sum_reward` | 0.65 |
| `avg_max_reward` | 0.65 |
| rollout 总时间 `eval_s` | 740.93 s |
| 平均每回合 `eval_ep_s` | 37.05 s |

原始结果保存在：

```text
outputs/eval/pi0_libero_object_task0_n20_final/eval_info.json
```

评估视频保存在：

```text
outputs/eval/pi0_libero_object_task0_n20_final/videos/libero_object_0/
```

此前 N=5 初测为 60%（3/5），保留用于复现链路的历史记录；扩展到 N=20 后为 65.0%（13/20）。N=20 降低了偶然性，但 95% 区间仍较宽，且只覆盖 `libero_object` task 0，不能替代多个 seed 和全部十个 task 的完整评测。

Pi0 与 Pi0.5 的 N=20 评估使用相同 task、seed 和初始状态序列。Pi0.5 为 19/20，较 Pi0 高 **30 个百分点**；配对结果中 Pi0.5 单独成功 7 回合、Pi0 单独成功 1 回合，Exact McNemar 双侧检验 `p≈0.0703`。这显示 Pi0.5 的明显优势趋势，但当前样本量下尚不能宣称达到传统 `p<0.05` 的统计显著性。

仍需补充：

- 固定 seed 下每个 task 的 N-episode 成功率。
- 平均/中位 action chunk 推理延迟。
- 峰值 GPU 显存与 CPU 内存。
- 与 SmolVLA 在同一环境、同一任务、同一评测协议下的严格对照实验；当前 SmolVLA/VLABench 数字不可直接比较。

---

## 11. 简历版本

> `libero_object task 0` 已完成 N=20 评测；全部任务、多 seed 的整体成功率仍需进一步测量。

### 上游能力 vs 本地贡献

| 上游（已有） | 本地贡献 |
|---|---|
| PaliGemma 预训练权重（Google） | 在 LeRobot 框架内跑通 Pi0 推理流程 |
| SigLIP 视觉编码能力 | 验证 Shared Attention + KV Cache 推理路径 |
| LeRobot Pi0 实现 | 配置 PI0Config 适配目标机器人 |
| Flow Matching 算法（论文） | 调试端到端推理链条（embed_prefix → 10步去噪 → 动作输出） |

---

### One-line

在 LeRobot 框架内部署 Pi0（PaliGemma-2B + Gemma-300M Expert）推理流程，基于 18 层 Shared Attention 与 10 步 Flow Matching ODE 去噪，实现语言指令驱动的机器人控制，并在 LIBERO object task 0 的 20-episode 评测中取得 **65.0%（13/20）成功率**。

---

### 3-bullet

- **部署 Pi0 VLA 推理**：在 LeRobot 内跑通 Pi0 端到端推理——SigLIP 提取 256 patch 向量（[B,256,1024]）经投影适配，PaliGemma-2B（18层，hidden=2048）与 Gemma-300M Expert（18层，hidden=1024）通过逐层 Q/K/V 序列维度拼接实现 Shared Attention，10 步 Flow Matching Euler ODE 生成 50 帧动作序列；LIBERO object task 0 的 N=20 成功率 **65.0%（13/20）**
- **验证 KV Cache 推理优化**：Prefix（256图像patch + 48语言token，共 304 个 2048 维向量）KV 在推理 Step 0 缓存，后续 10 步去噪仅重算 Suffix（51 个 1024 维向量），减少约 90% 的 PaliGemma 前向计算量
- **掌握 Flow Matching 机制**：训练：随机取 t ~ Beta(1.5,1.0)，构造 `x_t = t·noise + (1-t)·action`，单次前向 MSE 监督速度 `u_t = noise - action`；推理：从 `x_1 = noise` 出发，10 步 Euler ODE `x_{t-0.1} = x_t - 0.1·v_θ` 到达 `x_0 ≈ action`

---

### STAR（面试版）

**Situation**：研究团队需要在机器人平台部署 VLA 控制策略，Pi0 是基于 2B PaliGemma + 300M Expert Shared Attention 的代表性实现。

**Task**：在 LeRobot 内跑通 Pi0 推理，理解 Shared Attention、Flow Matching、KV Cache 三个核心机制，为后续任务训练和适配奠定基础。

**Action**：
1. 梳理 `PI0Policy → PI0Pytorch → PaliGemmaWithExpertModel` 三层类结构，定位 `compute_layer_complete`（Shared Attention 单层）和 `denoise_step`（单步去噪+KV 复用）两个核心函数
2. 验证推理路径：SigLIP [B,256,1024] → 模态投影 → [B,256,2048]，拼接语言向量 [B,48,2048] 得 Prefix [B,304,2048]，一次前向缓存 18 层 KV；10 步 `denoise_step` 各构造 Suffix [B,51,1024]，复用缓存执行 Euler 步进 `x_t += -0.1·v_t`
3. 理解注意力掩码：Prefix 双向 / Prefix 不看 Suffix（保证 KV Cache 有效）/ Suffix 因果（动作 token 不超前看），明确掩码设计与 KV Cache 有效性的因果关系

**Result**：Pi0 推理在 LeRobot + LIBERO 内完整跑通，已观察到 `libero_object task 0` 抓取、搬运、放置成功的完整 rollout；标准 N=20 评测取得 **65.0%（13/20）成功率**、Wilson 95% CI **43.3%-81.9%**、平均 37.05 秒/episode。每个 action chunk 的 Prefix 仅计算一次并复用 KV Cache，实时 Viewer 和复现文档已整理供团队复用。

---

### 已测指标与量化缺口

| 指标 | 状态 | 如何获取 |
|---|---|---|
| task 0 成功率（N=20） | **65.0%（13/20；Wilson 95% CI 43.3%-81.9%）** | `outputs/eval/pi0_libero_object_task0_n20_final/eval_info.json` |
| 与 Pi0.5 配对比较 | **-30 pp；discordant 1 vs 7；McNemar p≈0.0703** | 相同 task、seed、20 个初始状态 |
| 全部任务成功率 | needs measurement | 对 task 0..9 执行更大 N、多 seed 评测 |
| 单次推理延迟（ms） | needs measurement | 计时 `predict_action_chunk()` |
| GPU 显存占用（GB） | needs measurement | `nvidia-smi` 推理时快照 |
| 与论文成功率差距 | needs measurement | 对照论文 eval protocol |
