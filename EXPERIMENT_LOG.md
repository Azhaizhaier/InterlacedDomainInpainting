# 交织域正式实验记录

当前 B2/P1/P2 的训练和显示测试统一遵循 [`DISPLAY_TEST_PROTOCOL.md`](DISPLAY_TEST_PROTOCOL.md)。

## 2026-08-22：单种子正式训练

### 统一配置

```text
dataset_root = E:\dataset_out_v2
train scenes = scene001-scene008 (1600 samples)
validation   = scene009 (200 samples)
test         = scene010, scene012 (未读取)
seed         = 0
crop         = 256x256
batch        = 8
base         = 32
AMP          = enabled
workers      = 0
cache        = 2 decoded samples
crops/sample = 8
planned steps= 20000 (第一轮实际执行 1000)
val interval = 500 steps
save interval= 500 steps
主模型选择   = validation hole-region PSNR
```

### 启动状态

| Model | Output | Status | Checkpoint |
| --- | --- | --- | --- |
| B2 | `outputs/formal_b2_seed0` | 第一轮完成（1000 steps） | best/last 已保存 |
| P1 | `outputs/formal_p1_seed0` | 第一轮完成（1000 steps） | best/last 已保存 |
| P2 | `outputs/formal_p2_seed0` | 第一轮完成（1000 steps） | best/last 已保存 |

### 本机吞吐基准

RTX 4080 SUPER，256 crop、batch 8、AMP、workers 0：

| Model | Stable sec/step | Estimated 20k time |
| --- | ---: | ---: |
| B2 | 0.53-0.54 s | about 3.0 h |
| P2 | about 0.63 s | about 3.5 h |

上述是短基准和 1000-step 续训测得的吞吐，不是最终质量结果。

### 第一轮结果（1000 steps）

| Model | Best step | Best val hole PSNR | Best val full PSNR | Test hole PSNR | Test full PSNR | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B2 | 1000 | 12.92 | 22.22 | pending | pending | `outputs/formal_b2_seed0/unet_baseline.pt` |
| P1 | 1000 | 15.78 | 22.75 | pending | pending | `outputs/formal_p1_seed0/unet_view_id.pt` |
| P2 | 1000 | 14.33 | 22.81 | pending | pending | `outputs/formal_p2_seed0/unet_neighbor.pt` |

第一轮在固定 `scene009` 验证集上显示：P1 的 hole PSNR 高于 B2 和 P2；P2 的 full PSNR 略高但 hole PSNR 低于 P1。由于只有 1000 steps、单个 seed，这只是方向性结果，不能替代 20k steps 的最终比较。

后续若继续正式训练，使用对应 `.last.pt`，将 `--steps` 改为 20000；训练器会从当前 step 续训，且 best checkpoint 按 hole PSNR 更新。

## 2026-08-24：第一阶段 20k 续训状态

按固定协议从 1000-step checkpoint 续训：

| Model | Output | Status | Current/best validation |
| --- | --- | --- | --- |
| B2 | `outputs/formal_b2_seed0_20k` | 已完成 20,000 steps | best hole PSNR = 16.9051 dB |
| P2 | `outputs/formal_p2_seed0_20k` | 后台运行中 | 等待完成 |

B2 的 `.last.pt` 和 best `.pt` 已保存。P2 使用同样的 Train/Val、seed、crop、batch、AMP 和 20k steps 配置，完成后再运行 Test 分桶统计与固定代表样本导出。

测试集只在模型和超参数冻结后运行，不能使用 `scene010/scene012` 选择 checkpoint。

## 2026-08-24：第一轮填补样本导出

为保存 1000-step 阶段的实际填补结果，从两个冻结 Test 场景分别选择相同的 5 个 sample：

```text
sample indices = 0,3,4,7,9
scene010 + scene012 = 10 samples
 tile size = 1024
 overlap = 128
```

索引覆盖：interp step 1/3，以及 extrap step 2/5/7。B2、P1、P2 使用相同样本和 tiled inference 协议，共保存 30 张 4K 预测图。

阶段样本汇总：

| Method | Samples | Full PSNR | Hole PSNR | SSIM | View PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B2 | 10 | 30.7815 | 15.2026 | 0.938190 | 30.3160 |
| P1 | 10 | 31.3331 | 15.7156 | 0.937334 | 30.9029 |
| P2 | 10 | **31.3984** | **15.9435** | 0.937449 | **30.9452** |

这一代表样本子集上 P2 的 PSNR 最好；与 scene009 固定 crop 验证中 P1 的 hole PSNR 最好并不矛盾，因为样本集合和全图 tiled/crop 协议不同。当前仍属于 1000-step 单种子阶段结果，不能作为最终方法排名。

结果位置：

- `outputs/stage1_samples/b2/predictions/`
- `outputs/stage1_samples/p1/predictions/`
- `outputs/stage1_samples/p2/predictions/`
- `outputs/stage1_samples/stage1_test_results.csv`
- `outputs/stage1_samples/stage1_summary.csv`
- `outputs/stage1_samples/sample_manifest.csv`

B2 另有 `scene010/scene012` 全部 400 samples 的导出，位于 `outputs/test_b2_seed0/`。由于 P1/P2 的 4K 特征构建和 tiled inference 成本明显更高，本阶段没有将其全量 Test 导出伪装为已完成；三方法公平可视化以以上统一 10-sample 包为准。

> 注：本节的 1024/128 是 2026-08-24 的历史阶段导出。固定协议已改为 256/32，后续新结果必须使用 `DISPLAY_TEST_PROTOCOL.md`。

## 2026-08-24：固定协议 256/32 重新导出

使用相同的 10 个代表样本和当前 1000-step best checkpoint，严格按 `DISPLAY_TEST_PROTOCOL.md` 的 `tile=256, overlap=32` 重新导出。

| Method | Samples | Full PSNR | Hole PSNR | SSIM | View PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B2 | 10 | 30.7836 | 15.2059 | 0.938209 | 30.3183 |
| P1 | 10 | 31.3368 | 15.7204 | 0.937350 | 30.9067 |
| P2 | 10 | **31.4099** | **15.9605** | 0.937610 | **30.9568** |

结果目录：

- `outputs/stage1_protocol_256/b2/`
- `outputs/stage1_protocol_256/p1/`
- `outputs/stage1_protocol_256/p2/`
- `outputs/stage1_protocol_256/protocol_256_test_results.csv`
- `outputs/stage1_protocol_256/protocol_256_summary.csv`

每个方法均保存 10 张预测图和 10 行逐样本指标。该结果可用于当前裸眼 3D 屏幕对照；旧的 1024/128 结果仅保留作历史记录，不与本表混用。

## 2026-08-25：20k Test 全量推理与空洞比例分桶

按照 `DISPLAY_TEST_PROTOCOL.md` 对冻结的 B2/P2 20k checkpoint 执行 Test 全量推理。Test 包含 `scene010` 和 `scene012`，共 400 samples（每场景 200），使用 `tile=256, overlap=32`，seed=0；本节指标只用于最终测试汇总，不参与 checkpoint 选择。

| Method | Samples | Full PSNR | Hole PSNR | Valid PSNR | SSIM | View PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 | 400 | 33.3310 | 18.8235 | 37.3961 | 0.9535 | 32.9738 |
| P2 | 400 | **33.4419** | **18.9444** | 37.3961 | **0.9549** | **33.0873** |

场景与模式均值：

| Split | B2 Full / Hole / SSIM | P2 Full / Hole / SSIM |
| --- | --- | --- |
| scene010 (n=200) | 33.0703 / 21.3349 / 0.9389 | 33.0451 / 21.2406 / 0.9389 |
| scene012 (n=200) | 33.5918 / 16.3120 / 0.9681 | **33.8387 / 16.6483 / 0.9710** |
| interp (n=200) | 33.9306 / 18.7146 / 0.9564 | **34.1678 / 19.0877 / 0.9582** |
| extrap (n=200) | 32.7315 / 18.9324 / 0.9507 | 32.7160 / 18.8012 / 0.9517 |

按 hole ratio 分桶后，`>=3%` 大空洞共 80 samples：B2 为 `30.9738 / 19.9513 / 0.9324`（Full / Hole / SSIM），P2 为 `30.9181 / 19.8819 / 0.9325`。因此 P2 的整体增益主要来自 scene012 和 interp 样本；在大空洞桶上仍略低于 B2，符合屏幕观察中大缺口和边缘白像素较明显的现象。后续若优化模型，应单独加强大空洞、边缘一致性和 extrap 条件，而不是只看全体平均值。

完整逐样本结果与分桶汇总：

- B2：`outputs/test_b2_seed0_20k/baseline_test_eval.csv`、`outputs/test_b2_seed0_20k/baseline_test_summary.csv`
- P2：`outputs/test_p2_seed0_20k/neighbor_test_eval.csv`、`outputs/test_p2_seed0_20k/neighbor_test_summary.csv`

P1 在本轮仅有 1000-step 参考结果，未纳入 20k 全量三方法排名；待 P1 按相同协议完成训练后再做公平比较。

## 2026-08-25：第二阶段大洞与边缘问题诊断及改进接口

针对裸眼 3D 屏幕观察到的边缘灰白、大洞不自然和小黑洞，先对 P2/B2 的 400 张 Test PNG 做了逐像素诊断。统计阈值为亮像素 `>=250`、暗像素 `<=5`，边缘带为洞内向有效区域膨胀 3 px 的区域。

| Method | Hole bright ratio | Edge-hole bright ratio | Hole dark ratio | Edge-hole dark ratio |
| --- | ---: | ---: | ---: | ---: |
| B2 | 0.0503% | 0.0575% | 0.6064% | 0.6593% |
| P2 | 0.0556% | 0.0645% | 0.5673% | 0.6169% |
| GT | 0.0031% | - | - | - |

结论：预测 PNG 中确实存在少量异常高亮/暗像素，但比例不足以解释“整片灰白”；主要问题仍是洞边界的颜色、梯度和结构连续性。P2 的黑像素比例略优于 B2，但高亮比例略高，因此第二阶段先优化边界和大洞采样，再单独观察颜色范围约束的副作用。诊断明细保存在 `outputs/test_p2_seed0_20k/pixel_diagnosis.csv` 和 `outputs/test_b2_seed0_20k/pixel_diagnosis.csv`。

代码已加入可控的第二阶段接口，默认参数均为 0，不改变旧实验：

- `--boundary-weight`、`--boundary-radius`：洞内边界带 L1；
- `--gradient-weight`：边界带梯度一致性损失；
- `--large-hole-prob`、`--large-hole-candidates`：从含洞 crop 候选中优先选择 hole ratio 较大的 crop；
- `--range-weight`：对网络原始输出超出 `[0,1]` 的值施加惩罚。

建议下一步先做单变量小规模消融：固定 P2 网络和训练协议，分别加入 boundary loss、大洞采样，再组合两者；每次使用相同 seed、scene009 分桶验证，并导出 scene010/scene012 的 10 个代表样本。稀疏 view-plane consistency 和 depth/背景层信息暂不直接加入，待上述 RGB 边界问题量化后再决定。
