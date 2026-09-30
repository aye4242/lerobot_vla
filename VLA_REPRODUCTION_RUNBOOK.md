# SmolVLA、Pi0、Pi0.5 与 GR00T 复现运行手册

本文集中记录当前机器上 SmolVLA/VLABench、Pi0/LIBERO 与 Pi0.5/LIBERO 的可运行命令、交互方式、任务编号、实验结论和已知限制。

## 1. 公共环境

仓库与运行环境：

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate
```

主要路径：

| 内容 | 路径 |
|---|---|
| Python 环境 | `/extdata/hdd2/lerobot-smolvla/.venv` |
| Hugging Face 缓存 | `/extdata/hdd2/lerobot-smolvla/hf-cache` |
| VLABench | `/extdata/hdd2/lerobot-smolvla/sim/VLABench/VLABench` |
| SmolVLA-VLABench checkpoint | `hf-cache/hub/models--lerobot--smolvla_vlabench/snapshots/4fd586e12dc14b04d9d606ddbb77448df4f0ff29` |
| Pi0-LIBERO checkpoint | `hf-cache/pi0_libero_finetuned` |
| Pi0.5-LIBERO checkpoint | `hf-cache/pi05_libero_finetuned` |
| PaliGemma tokenizer | `hf-cache/paligemma-tokenizer` |
| GR00T Python 环境 | `/extdata/hdd2/lerobot-smolvla/.venv-groot` |
| GR00T checkpoints | `/extdata/hdd2/lerobot-smolvla/hf-cache-groot/gr00t17-lerobot-libero_<suite>-640` |

当前主机为 Tesla M40，计算能力 5.2，不支持原生 BF16。Pi0 使用 `float32`，图形渲染使用 `MUJOCO_GL=glfw`。

## 2. SmolVLA：VLABench 实时复现

### 2.1 推荐启动命令

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate

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

### 2.2 Viewer 操作

| 操作 | 功能 |
|---|---|
| `Space` | 运行/暂停 |
| `Backspace` | 重置任务 |
| `F1` | Viewer 帮助 |
| `[` / `]` | 切换相机 |
| 鼠标 | 旋转、平移、缩放视角 |

### 2.3 已验证参数现象

| `n_action_steps` | 实验现象 |
|---:|---|
| `20` | 通常无法到达目标 |
| `30` | 通常无法到达目标 |
| `40` | 可以接近目标，但抓取不稳定 |
| `50` | 当前效果最好，曾完成抓取并接近或完成放置 |
| `60` | 配置报错：超过 checkpoint 的 `chunk_size=50` |

因此当前推荐保持：

```text
n_action_steps = 50
```

### 2.4 当前结论

- SmolVLA 已完成代码、checkpoint、VLABench 环境和实时 Viewer 的复现。
- 正确的自然语言指令来自 VLABench 的 `task.get_instruction()`。
- 三路相机映射和相机标定会显著影响动作质量。
- 正式 N=5 初测成功率为 **20%（1/5）**；模型具有明显随机性，需要更多 rollout 才能得到稳定估计。

### 2.5 SmolVLA 固定实例批量评测

默认评测 `select_toy`、seed 1000、`n_action_steps=50` 和 5 个回合：

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate
bash ./examples/vlabench/run_smolvla_vlabench_eval.sh
```

显式指定回合数和输出目录：

```bash
N_EPISODES=5 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/smolvla_vlabench_select_toy_n5 \
bash ./examples/vlabench/run_smolvla_vlabench_eval.sh
```

当前本机结果：

| 配置 | 结果 |
|---|---:|
| `select_toy`，seed=1000，N=5 | **20%（1/5）** |
| 总评估时间 / 平均每回合 | 1484.84 s / 296.97 s |
| `avg_sum_reward` / `avg_max_reward` | 0.0 / 0.0 |

原始指标：`outputs/eval/smolvla_vlabench_select_toy_n5/eval_info.json`。该模式适合控制稳定性诊断；VLABench reward 即使在成功回合也保持为 0，不能用 reward 代替成功率。

### 2.6 SmolVLA 官方 deterministic track 评测

官方 track 会为每个 episode 使用固定场景配置。当前 wrapper 用 `seed=1000+i` 选择 track 第 `i` 个配置，并在每回合重新构建环境；因此不同回合会覆盖不同目标物体、干扰物和布局。

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate

VLABENCH_DETERMINISTIC_TRACK=/extdata/hdd2/lerobot-smolvla/sim/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json \
VLABENCH_TRACK_SEED_OFFSET=1000 \
N_EPISODES=20 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/smolvla_vlabench_select_toy_track1_n20 \
bash ./examples/vlabench/run_smolvla_vlabench_eval.sh
```

本机结果：`13/20 = 65.0%`，Wilson 95% CI 为 `43.3%-81.9%`。阶段字段保存在 `eval_info.json` 的 `episode_infos` 和 `stage_metrics` 中：

| 阶段 | 通过率 |
|---|---:|
| 目标物体被抓持 | 100.0%（20/20） |
| 目标物体抬升至少 5 cm | 85.0%（17/20） |
| 放置完成（环境 success predicate） | 65.0%（13/20） |
| 曾抓到错误物体 | 10.0%（2/20） |

每回合还记录 `instruction`、`target_entity`、`target_container`、`episode_config_index`、`min_eef_target_distance_m` 和 `max_target_lift_m`。其中 `stage_target_reached` 是末端到目标实体中心小于 10 cm 的严格接近代理，不应直接解释为模型内部的视觉识别准确率。

## 3. Pi0：LIBERO 实时复现

### 3.1 最常用启动命令

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate
bash ./examples/libero/run_pi0_libero_viewer.sh
```

该脚本默认配置：

```text
policy          = pi0_libero_finetuned
suite           = libero_object
task_id         = 0
object          = alphabet soup
n_action_steps  = 50
dtype           = float32
max_steps       = 1000（交互 Viewer）
camera          = 自由观察视角
```

正常加载成功时终端应出现：

```text
All Pi0 keys loaded successfully with low CPU memory usage!
```

第一次 action chunk 推理可能需要十几秒，后续 chunk 通常约数秒。终端会持续打印：

```text
Pi0 chunk inference START
Pi0 chunk inference still running
Pi0 chunk inference DONE
Pi0 rollout active
```

### 3.2 主要按键

日常只需要记住：

| 按键 | 功能 |
|---|---|
| `Space` | 运行/暂停 |
| `I` | 输入下一条语言指令 |

可选操作：

| 按键/鼠标 | 功能 |
|---|---|
| `H` | 只把机械臂恢复到 Home，不重置物体 |
| `C` | 自由观察视角/固定策略视角切换 |
| `R` | 重置整个环境和物体 |
| `Q` / `Esc` | 退出 |
| 鼠标左键拖动 | 旋转自由观察视角 |
| 鼠标右键拖动 | 平移自由观察视角 |
| 鼠标滚轮 | 缩放 |

自由观察相机只影响屏幕显示，不会改变 Pi0 使用的固定策略相机输入。

需要启动时显示固定策略相机：

```bash
bash ./examples/libero/run_pi0_libero_viewer.sh --start-fixed-camera
```

### 3.3 `libero_object` 官方任务编号

| task id | 官方目标物品 |
|---:|---|
| `0` | alphabet soup |
| `1` | cream cheese |
| `2` | salad dressing |
| `3` | bbq sauce |
| `4` | ketchup |
| `5` | tomato sauce |
| `6` | butter |
| `7` | milk |
| `8` | chocolate pudding |
| `9` | orange juice |

单独验证 milk 官方任务：

```bash
bash ./examples/libero/run_pi0_libero_viewer.sh \
  --task libero_object \
  --task-id 7
```

单独验证 butter 官方任务：

```bash
bash ./examples/libero/run_pi0_libero_viewer.sh \
  --task libero_object \
  --task-id 6
```

脚本最后的参数会覆盖默认 `task-id 0`。

### 3.4 常用参数

增加交互回合上限：

```bash
bash ./examples/libero/run_pi0_libero_viewer.sh --max-steps 1500
```

成功后继续执行 50 步，用于松爪和撤离：

```bash
bash ./examples/libero/run_pi0_libero_viewer.sh --post-success-steps 50
```

成功后自动恢复机械臂 Home：

```bash
bash ./examples/libero/run_pi0_libero_viewer.sh --auto-home-after-success
```

启动时覆盖语言指令：

```bash
bash ./examples/libero/run_pi0_libero_viewer.sh \
  --instruction "pick up the milk and place it in the basket"
```

注意：覆盖文字不等于切换到了该物品的官方 BDDL 任务和初始布局。

## 4. Pi0.5：LIBERO 实时推理

Pi0.5 使用与 Pi0 相同的 LIBERO 环境和任务编号，因此可以直接进行横向行为对比。当前机器是 Tesla M40（计算能力 5.2），不支持原生 BF16；启动脚本固定使用 `float32`。官方 Pi0.5-LIBERO checkpoint 还需要本地 PaliGemma tokenizer，并使用 `n_action_steps=10`。

### 4.1 完整启动命令

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate

bash ./examples/libero/run_pi05_libero_viewer.sh \
  --n-action-steps 50 \
  --task libero_object \
  --task-id 0 \
  --start-free-camera \
  --max-steps 280 \
  --post-success-steps 50
```

该脚本已经完整设置以下运行参数：

```text
checkpoint       /extdata/hdd2/lerobot-smolvla/hf-cache/pi05_libero_finetuned
tokenizer        /extdata/hdd2/lerobot-smolvla/hf-cache/paligemma-tokenizer
device           cuda
dtype            float32
n_action_steps   10
empty_camera_0   1（由 checkpoint 配置提供）
MUJOCO_GL        glfw
DISPLAY          :0
HF_HUB_OFFLINE   1
TRANSFORMERS_OFFLINE 1
```

若 checkpoint 或缓存目录不同，可在启动前覆盖：

```bash
PI05_LIBERO_CHECKPOINT=/path/to/pi05_libero_finetuned \
PI_TOKENIZER_PATH=/path/to/paligemma-tokenizer \
bash ./examples/libero/run_pi05_libero_viewer.sh --start-running
```

正常加载时终端必须出现：

```text
Loading Pi0.5 policy: .../pi05_libero_finetuned
Streaming Pi0.5 weights to cuda ...
All Pi0.5 keys loaded successfully with low CPU memory usage!
```

第一次 chunk 推理在本机约需要 18 秒，后续推理时间取决于 GPU 和当前动作块。`Space` 可以暂停/继续，`I` 输入下一条指令，`R` 重置环境，`Q` 或 `Esc` 退出；`C` 切换固定策略相机和自由观察相机。自由相机只改变屏幕视角，不会改变模型实际接收的相机图像。

### 4.2 一个 Pi0.5 checkpoint 切换四个 LIBERO suite

`lerobot/pi05_libero_finetuned` 是覆盖四个标准 suite 的单一 checkpoint，不需要为每个 suite 重新下载模型。启动脚本会检查 suite 名称，并使用对应的默认最大步数：

| suite | 任务性质 | task-id | 默认最大步数 |
|---|---|---:|---:|
| `libero_spatial` | 物体间空间关系和指定区域放置 | `0..9` | 280 |
| `libero_object` | 识别指定物体并完成 pick-and-place | `0..9` | 280 |
| `libero_goal` | 相同场景中的不同目标状态与操作目标 | `0..9` | 300 |
| `libero_10` | 多物体、多阶段、长时序任务 | `0..9` | 520 |

四个 suite 的完整启动命令：

```bash
# Spatial
bash ./examples/libero/run_pi05_libero_viewer.sh \
  --task libero_spatial --task-id 0 --n-action-steps 10 --start-running

# Object
bash ./examples/libero/run_pi05_libero_viewer.sh \
  --task libero_object --task-id 0 --n-action-steps 10 --start-running

# Goal
bash ./examples/libero/run_pi05_libero_viewer.sh \
  --task libero_goal --task-id 0 --n-action-steps 10 --start-running

# LIBERO-10 / Long
bash ./examples/libero/run_pi05_libero_viewer.sh \
  --task libero_10 --task-id 0 --n-action-steps 10 --max-steps 600 --start-running
```

`--task-id` 可以改为 `0..9`。Viewer 启动后会在终端打印该 BDDL task 的官方 instruction；不要仅通过 `--instruction` 把 Object 任务改写成长任务，因为那不会更换场景、初始状态和成功条件。

Viewer 现在直接读取 LIBERO 原生 `info["is_success"]` BDDL 判定，因此四个 suite 都能正确检测完成，包括空间关系、抽屉/炉灶目标和 LIBERO-10 多阶段目标。物体名称匹配只用于界面显示，不参与成功判断。

### 4.3 用 Pi0.5 验证其他官方物品任务

例如验证 `milk`（task 7）或 `butter`（task 6）：

```bash
bash ./examples/libero/run_pi05_libero_viewer.sh --start-running --task-id 7
bash ./examples/libero/run_pi05_libero_viewer.sh --start-running --task-id 6
```

启动时使用策略固定相机：

```bash
bash ./examples/libero/run_pi05_libero_viewer.sh --start-running --start-fixed-camera
```

默认启动脚本现在显式使用 `--start-free-camera`。自由观察视角的鼠标操作只在画面区域生效：左键拖动旋转、右键拖动平移、滚轮缩放。画面没有可点击的“运行/确认”按钮；运行和暂停仍使用 `Space`，终端出现 `Viewer key detected: SPACE` 才表示按键已收到。

如果希望先调整视角、再开始 Pi0.5 推理，不要添加 `--start-running`：

```bash
bash ./examples/libero/run_pi05_libero_viewer.sh \
  --start-free-camera \
  --task-id 0
```

窗口启动后保持暂停，此时先拖动鼠标调整视角，然后按 `Space` 开始执行。推理已经开始时，鼠标事件仍会接收，但自由相机画面通常要等当前 action chunk 计算完成后才明显更新。

### 4.4 常用参数修改

脚本末尾传入的参数会覆盖默认值。例如：

```bash
# 修改官方 LIBERO 任务
bash ./examples/libero/run_pi05_libero_viewer.sh --task-id 7

# 修改动作块执行长度；Pi0.5-LIBERO 官方推荐值是 10
bash ./examples/libero/run_pi05_libero_viewer.sh --n-action-steps 20

# 修改最大步数和成功后的继续执行步数
bash ./examples/libero/run_pi05_libero_viewer.sh \
  --max-steps 1500 \
  --post-success-steps 80

# 覆盖语言指令；这不会切换 BDDL 场景和官方目标布局
bash ./examples/libero/run_pi05_libero_viewer.sh \
  --instruction "pick up the milk and place it in the basket"

# 启动后立即运行；省略该参数则从暂停状态开始
bash ./examples/libero/run_pi05_libero_viewer.sh --start-running
```

不建议随意增大 `n_action_steps`。它只改变每次推理后连续执行多少个已预测动作，不会增加模型能力；数值过大会降低闭环纠错频率。Pi0.5 横向复现优先保留官方值 `10`。

### 4.5 Pi0.5 批量评测

单独评测 `libero_object/task 0` 的 5 回合：

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv/bin/activate

N_EPISODES=5 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/pi05_libero_object_task0_n5 \
bash ./examples/libero/run_pi05_libero_eval.sh
```

与 Pi0 使用同一任务、同一 seed 进行横向比较：

```bash
N_EPISODES=5 \
bash ./examples/libero/run_pi0_pi05_libero_compare.sh
```

严格使用相同动作块长度时：

```bash
PI0_ACTION_STEPS=10 PI05_ACTION_STEPS=10 N_EPISODES=5 \
bash ./examples/libero/run_pi0_pi05_libero_compare.sh
```

批量评测是官方单任务生命周期：成功后结束该回合并重置环境，不等价于 Viewer 中完成一个物品后在同一场景继续输入新指令。

## 5. Pi0 正式批量评测

评测单个 alphabet soup 任务：

```bash
LIBERO_SUITE=libero_object \
LIBERO_TASK_IDS='[0]' \
N_EPISODES=1 \
bash ./examples/libero/run_pi0_libero_eval.sh
```

评测 milk：

```bash
LIBERO_SUITE=libero_object \
LIBERO_TASK_IDS='[7]' \
N_EPISODES=5 \
bash ./examples/libero/run_pi0_libero_eval.sh
```

评测全部十个 `libero_object` 任务：

```bash
LIBERO_SUITE=libero_object \
LIBERO_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]' \
N_EPISODES=5 \
bash ./examples/libero/run_pi0_libero_eval.sh
```

批量评测使用官方单任务生命周期，成功后自动结束并重置，不使用交互 Viewer 的连续指令实验逻辑。

当前已完成的初步评测结果：

| 配置 | 结果 |
|---|---:|
| `libero_object task 0`，seed=1000，N=5 | **60%（3/5）** |
| `avg_sum_reward` / `avg_max_reward` | 0.6 / 0.6 |
| 总 rollout 时间 / 平均每回合 | 196.03 s / 39.21 s |

原始指标：`outputs/eval/pi0_libero_object_task0_n5/eval_info.json`。

## 6. Pi0 与同场景连续指令的限制

`pi0_libero_finetuned` 可以完成多个官方 LIBERO 任务，但每个任务的训练演示都从自己的 BDDL 初始状态开始，并在目标完成后结束。

例如：

- `task 0` 中 alphabet soup 位于目标区域，milk 只是另一位置的干扰物。
- `task 7` 中 milk 位于目标区域，场景中的其他干扰物和位置也发生变化。
- 模型没有训练过“alphabet soup 已在篮子里，然后不重置场景，再输入 milk 指令”的状态。

因此在 `task 0` 完成 alphabet soup 后按 `I` 输入 milk，代码会正确传入新 prompt 并清空旧 action queue，但模型仍可能停留在篮子附近。这属于训练分布限制，不是按键或 prompt 没有生效。

要可靠实现“同一场景逐个把所有物品放进篮子”，需要：

1. 创建包含连续多物品目标的 BDDL 任务。
2. 收集相应的多阶段示范数据。
3. 使用这些数据微调 Pi0。
4. 或使用上层任务规划器，为每个子任务选择对应环境/策略。

交互 Viewer 中的 `I` 换指令功能属于泛化能力实验，不应作为官方成功率结果。

## 7. Pi0 + VLABench 说明

接口 smoke test：

```bash
bash ./examples/vlabench/run_pi0_viewer.sh
```

该命令使用 `pi0_base` 和人工相机名称映射，只能验证：

- Pi0 checkpoint 可以加载。
- VLABench 观测可以进入 Pi0 接口。
- 动作可以经过控制链输出。

它不能证明 Pi0 在 VLABench 上具有有效行为，因为 `pi0_base` 没有使用 VLABench 数据和相机语义进行微调。真正的行为验证应使用：

```text
pi0_libero_finetuned + LIBERO
```

或者重新训练：

```text
Pi0 + VLABench 数据 -> pi0_vlabench checkpoint
```

## 8. 推荐复现顺序

1. 使用 SmolVLA + VLABench `select_toy` 验证实时控制链。
2. 使用 Pi0 + LIBERO `task-id 0` 验证 alphabet soup。
3. 分别运行 `task-id 1..9`，验证 Pi0 的官方单物品任务能力。
4. 使用批量评测统计每个任务的成功率，不以单次运行作为结论。
5. 最后再测试按 `I` 的同场景换指令泛化实验。

更深入的架构和实验分析见：

- `SMOLVLA_ARCHITECTURE.md`
- `PI0_ARCHITECTURE.md`

## 9. GR00T N1.7：LIBERO 实时复现

GR00T 使用独立环境，避免改变已经稳定的 Pi0/Pi0.5 依赖：

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv-groot/bin/activate
```

GR00T N1.7 也覆盖四个 suite，但与 Pi0.5 不同，它使用四个 suite 专用 checkpoint。启动脚本根据 `LIBERO_SUITE` 自动选择模型：

| `LIBERO_SUITE` | 自动选择的 checkpoint |
|---|---|
| `libero_spatial` | `gr00t17-lerobot-libero_spatial-640` |
| `libero_object` | `gr00t17-lerobot-libero_object-640` |
| `libero_goal` | `gr00t17-lerobot-libero_goal-640` |
| `libero_10` | `gr00t17-lerobot-libero_10-640` |

本机 checkpoint 状态（2026-08-05）：

| suite | 本地状态 | 已验证 SHA-256 |
|---|---|---|
| `libero_spatial` | 已下载 | `8a6e55ca705ab60c6d9c1eaf585dbda53f1b92bffdf6050c18ce68eb3afc6e69` |
| `libero_object` | 已下载并完成 Viewer smoke test | checkpoint 已可加载 |
| `libero_goal` | 已下载 | `71c8220d03c1d429fe90861869e82189e944aeef9a58aea13bc4f3140dfdb0ef` |
| `libero_10` | 已下载 | `8a7f3c0fb13cc84f89bbc7af1a675431ddd7b16ce47bebb12b4298f6b2827a94` |

### 9.1 下载 GR00T suite checkpoint

下载模板如下，只需修改 `SUITE`。当前可用代理端口为 `34501`、`34601`、`34514`、`34614`；一个端口长时间无增长时切换端口并重新执行相同命令。

```bash
cd /home/aitech/Workspace/VLA/lerobot
source /extdata/hdd2/lerobot-smolvla/.venv-groot/bin/activate

SUITE=libero_10
PROXY_PORT=34501

http_proxy=http://192.168.100.8:${PROXY_PORT} \
https_proxy=http://192.168.100.8:${PROXY_PORT} \
HTTP_PROXY=http://192.168.100.8:${PROXY_PORT} \
HTTPS_PROXY=http://192.168.100.8:${PROXY_PORT} \
ALL_PROXY= \
all_proxy= \
HF_HOME=/extdata/hdd2/lerobot-smolvla/hf-cache-groot \
/extdata/hdd2/lerobot-smolvla/.venv-groot/bin/hf download \
  nvidia/gr00t17-lerobot-${SUITE}-640 \
  --local-dir /extdata/hdd2/lerobot-smolvla/hf-cache-groot/gr00t17-lerobot-${SUITE}-640
```

四个可用 `SUITE` 值是：

```text
libero_spatial
libero_object
libero_goal
libero_10
```

LIBERO-10 下载完成后校验：

```bash
sha256sum \
  /extdata/hdd2/lerobot-smolvla/hf-cache-groot/gr00t17-lerobot-libero_10-640/model.safetensors
```

输出应以以下值开头：

```text
8a7f3c0fb13cc84f89bbc7af1a675431ddd7b16ce47bebb12b4298f6b2827a94
```

只有校验值完全一致后，才清理下载中断遗留的重复 partial：

```bash
find \
  /extdata/hdd2/lerobot-smolvla/hf-cache-groot/gr00t17-lerobot-libero_10-640/.cache/huggingface/download \
  -type f -name '*.incomplete' -delete
```

模型尚未校验完成时不要执行该清理命令。

### 9.2 四个 suite 实时运行

直接启动与 Pi0.5 相同的 `libero_object/task 0`：

```bash
cd /home/aitech/Workspace/VLA/lerobot
bash ./examples/libero/run_groot_libero_viewer.sh
```

四个 suite 的启动命令：

```bash
# Spatial
LIBERO_SUITE=libero_spatial LIBERO_TASK_ID=0 bash ./examples/libero/run_groot_libero_viewer.sh --start-running

# Object
LIBERO_SUITE=libero_object LIBERO_TASK_ID=0 bash ./examples/libero/run_groot_libero_viewer.sh --start-running

# Goal
LIBERO_SUITE=libero_goal LIBERO_TASK_ID=0 bash ./examples/libero/run_groot_libero_viewer.sh --start-running

# LIBERO-10 / Long
LIBERO_SUITE=libero_10 LIBERO_TASK_ID=0 LIBERO_MAX_STEPS=600 bash ./examples/libero/run_groot_libero_viewer.sh --start-running
```

也可以直接传 `--task libero_goal --task-id 3`；脚本会先解析参数，再选择正确 checkpoint。`task-id` 范围为 `0..9`。

`LIBERO-10` 的 `task-id` 与官方任务固定对应如下。修改 `task-id` 会同时更换场景、自然语言指令、初始状态和 BDDL 成功条件：

| task-id | 官方任务 |
|---:|---|
| 0 | 把 alphabet soup 和 tomato sauce 都放进篮子 |
| 1 | 把 cream cheese box 和 butter 都放进篮子 |
| 2 | 打开炉灶，并把 moka pot 放到炉灶上 |
| 3 | 把 black bowl 放进柜子底层抽屉，并关闭抽屉 |
| 4 | 把 white mug 放在左盘子上，并把 yellow and white mug 放在右盘子上 |
| 5 | 拿起书并放进 caddy 后侧隔层 |
| 6 | 把 white mug 放在盘子上，并把 chocolate pudding 放在盘子右侧 |
| 7 | 把 alphabet soup 和 cream cheese box 都放进篮子 |
| 8 | 把两个 moka pot 都放到炉灶上 |
| 9 | 把 yellow and white mug 放进微波炉并关闭微波炉 |

例如，运行 `task-id 3` 的完整单行命令是：

```bash
LIBERO_SUITE=libero_10 LIBERO_TASK_ID=3 LIBERO_MAX_STEPS=1000 bash ./examples/libero/run_groot_libero_viewer.sh --start-free-camera --n-action-steps 8
```

该命令应在终端打印 `LIBERO suite: libero_10 | task id: 3 | max steps: 1000`，窗口应显示 `step=0/1000`，官方指令应为 `put the black bowl in the bottom drawer of the cabinet and close it`。如果窗口仍显示 `/600`，说明看到的是此前 `task-id 0` 示例启动的旧 Viewer，应先退出旧窗口再重新运行。`--start-free-camera` 和 `--n-action-steps 8` 已是 GR00T 启动脚本的默认值，保留在此处只是为了让实验参数显式可见。

### 9.3 参数说明

- `--n-action-steps 8`：GR00T checkpoint 预测 16 步，但 NVIDIA LIBERO rollout 每执行 8 步就重新规划；启动脚本已使用该值。
- `--start-free-camera`：默认启用自由观察视角，不改变传给模型的固定相机图像。
- `--start-fixed-camera`：启动时显示策略固定相机。
- `--start-running`：窗口出现后立即推理；不传时先调整观察视角，再按 `Space`。
- `--max-steps`：单回合最大仿真步数。
- `--post-success-steps`：成功检测后继续执行的步数。
- `--instruction`：只覆盖语言文本，不改变 task ID 对应的 BDDL 初始布局。

交互键与 Pi0/Pi0.5 viewer 相同：`Space` 运行/暂停，`I` 输入新指令，`C` 切换视角，`R` 重置，`Q/Esc` 退出。自由视角使用左键旋转、右键平移、滚轮缩放。

### 9.4 本机固定设置

```text
checkpoints: /extdata/hdd2/lerobot-smolvla/hf-cache-groot/gr00t17-lerobot-libero_<suite>-640
processor assets: /extdata/hdd2/lerobot-smolvla/hf-cache-groot/qwen3-vl-2b-processor-assets
camera rename: observation.images.image2 -> observation.images.wrist_image
precision: FP32
model VRAM: about 11.99 GiB
first real action-chunk inference: 23.617 s
```

由于主机只有约 15 GiB RAM，12.6 GB FP32 checkpoint 加载时会使用 swap，首次启动约需 7-10 分钟。终端长时间停在 `Loading weights from local directory` 不代表死锁，可用 `nvidia-smi` 观察显存逐步增长到约 12 GiB。

### 9.5 单回合评测

```bash
LIBERO_SUITE=libero_object \
LIBERO_TASK_IDS='[0]' \
N_EPISODES=1 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/groot_libero_object_task0_n1 \
bash ./examples/libero/run_groot_libero_eval.sh
```

评测器现在会在每个回合结束后原子更新 `OUTPUT_DIR/progress/libero_object_0.json`，记录已完成回合的 success、reward、seed 和累计指标；全部回合完成后仍按原格式生成 `OUTPUT_DIR/eval_info.json`。中途发生GPU或进程故障时，先保留 progress 文件，不要删除输出目录。

本机已完成 `libero_object/task 0` 的标准N=1评测：1/1成功，成功发生在第139步，评测耗时124.54秒，结果位于 `outputs/eval/groot_libero_object_task0_n1/eval_info.json`。

首次 N=20 在第20回合发生 Tesla M40 `NVRM Xid 79`，未产生最终 JSON，不能作为正式指标。GPU恢复后已用同一协议完整重跑：**18/20 成功（90.0%，Wilson 95% CI 69.9%--97.2%）**，平均 reward 为0.90，平均124.59秒/回合，失败 seed 为1003和1011。正式结果位于 `outputs/eval/groot_libero_object_task0_n20_final_retry/eval_info.json`，前10个视频在同目录 `videos/libero_object_0/`。progress 文件 `progress/libero_object_0.json` 已标为 `completed`。

下面是用于复现该正式 N=20 实验的命令（不是待执行的重跑步骤）：

```bash
LIBERO_SUITE=libero_object \
LIBERO_TASK_IDS='[0]' \
N_EPISODES=20 \
OUTPUT_DIR=/home/aitech/Workspace/VLA/lerobot/outputs/eval/groot_libero_object_task0_n20_final_retry \
bash ./examples/libero/run_groot_libero_eval.sh
```

四个 suite 使用同一个 Viewer 和通用 BDDL 成功检测，但 GR00T 必须加载与 suite 对应的 checkpoint。若权重目录不存在，脚本会在创建环境和加载模型前直接报出缺失路径，不会误用 Object 模型。

与 Pi0.5 做同任务横向对比时，固定相同的 suite、task-id 和初始 seed，保留各自官方执行 horizon：Pi0.5 为 `10`，GR00T 为 `8`。例如：

```bash
bash ./examples/libero/run_pi05_libero_viewer.sh \
  --task libero_10 --task-id 0 --n-action-steps 10 --seed 1000 --start-running

LIBERO_SUITE=libero_10 LIBERO_TASK_ID=0 \
bash ./examples/libero/run_groot_libero_viewer.sh \
  --n-action-steps 8 --seed 1000 --start-running
```

更深入的架构与本机兼容说明见 `GROOT_ARCHITECTURE.md`。
