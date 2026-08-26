# Interlaced-domain Hole Filling 项目总文档

> 文档状态：当前交织域研究的权威总览。更新日期：2026-08-21。
>
> 范围边界：本文只整理交织域数据、方法、实验和当前结论。Warp 仅作为数据来源和几何前提，不讨论 Warp 引擎实现或加速性能。

## 1. 项目定位

本项目研究面向柱状透镜裸眼 3D 显示的 **Interlaced-domain Hole Filling / Display-aware Reconstruction**。

单视角 RGB 和 Depth 经相机感知 Warp 得到 8 个视图。Warp 产生的遮挡空洞经 Lenticular Interlacing 后，变成显示域中的子像素级缺失。项目目标是在最终显示域直接恢复这些缺失子像素：

```text
8 warped views + view masks
        -> Lenticular Interlacing
        -> interlaced_input + interlaced_mask
        -> display-aware hole filling
        -> predicted_interlaced_image
        -> compare with interlaced_gt
```

View-domain filling 不是主任务，只作为“逐视图填补后再交织”的对比基线。

## 2. 研究问题与核心假设

### 2.1 研究问题

普通图像修复把 interlaced image 当作常规 RGB 图像，忽略每个子像素由哪个 view 提供。项目研究能否利用确定性的显示映射先验，提高交织域空洞恢复质量并减少跨视图内容串扰。

### 2.2 核心假设

1. `pixel -> subpixel -> view` 的 LUT ownership 可作为有效网络先验。
2. 同 view 的观测在显示域中空间离散，但可由 LUT 重新组织和聚合。
3. 按 view 分组的辅助监督可能比普通显示域卷积更好地保持视图一致性。

这些是待实验验证的假设。当前结果尚不能宣称显示先验已经优于普通 U-Net。

### 2.3 训练硬件与吞吐基准

当前本机为 RTX 4080 SUPER 16 GB。正式数据 loader 已采用 scene-level split、受控 LRU 缓存和同一样本多 crop 采样，避免 4K 图像无上限缓存及重复 PNG 解码。

256x256 crop、batch 8、base 32、AMP、workers 0 的实测稳定吞吐：B2 约 `0.53-0.54 s/step`，P2 约 `0.63 s/step`。20k steps 单模型约需 3.0-3.5 小时，适合本机完成单种子主线实验；多随机种子、P3/P4 和大规模消融再迁移到服务器。

## 3. 已验证的技术基础

### 3.1 任务和 GT

```text
Input : interlaced_input, interlaced_mask
Target: interlaced_gt
```

`interlaced_gt` 必须由 Blender 真实渲染的目标视图使用同一 LUT 交织得到。禁止把算法补全后的 views 再交织作为训练 GT。

主任务是 masked inpainting：

```text
output = input * (1 - mask) + prediction * mask
```

有效位置保持输入不变，训练和评价重点放在空洞区域。全图重建只作为可选的显示域精修任务。

### 3.2 Mask 语义

- `interlaced_mask.png` 为三通道子像素 mask。
- `0` 表示有效，`255` 表示 hole。
- RGB 三个通道可分别来自不同 view，因此必须保留通道级 mask。
- `interlaced_mask_view.png` 仅用于可视化，不替代训练用三通道 mask。

### 3.3 Interlacing LUT

```text
subpixelX = x * 3 + c             # c = 0:B, 1:G, 2:R
r = (subpixelX + KOFF - 3*y*THETA) mod SUBPIXEL
view = min(int(r * view_num / SUBPIXEL), view_num - 1)
```

当前正式参数：

```text
THETA=0.166666
KOFF=0
SUBPIXEL=4.666666
view_num=8
```

参数从 `sample.json` 或 `display.json` 读取，不能在模型中硬编码。

### 3.4 已验证的 LUT 性质

- 精确周期向量为 `(1,6)`。
- `(5,2)` 是数学周期，但浮点边界可能相差 1，只适合作为候选采样结构。
- 每个显示像素的三个子像素恰好属于三个不同 view。
- 每个 view 只覆盖约 33%-42% 的像素位置。
- 周期描述的是映射重复，不代表不同位置的图像内容相等。

### 3.5 可逆性边界

从 8 个完整 RGB views 到一张 interlaced RGB 是有损采样，无法直接恢复 8 个完整 views。

```text
complete views --lossy interlace--> interlaced RGB
interlaced RGB --exact gather--> sparse view planes
sparse view planes --exact scatter--> interlaced RGB
```

因此 de-interlace 只用于提取稀疏的 view-specific evidence。它不是完整 view reconstruction，也不能把插值后的近似 view 当作主评价对象。

### 3.6 Warp 与 GT 的验收关系

`interlaced_input` 来自整数像素 forward warp，`interlaced_gt` 来自 Blender 对目标相机的重新渲染。分数位移取整会造成约 +/-1 px 的固有差异，因此有效子像素不要求与 GT 精确相等。

当前数据验收标准：

- 必需文件和尺寸正确；
- mask 为二值且 `mask_view` 一致；
- 有效区 PSNR >= 30 dB；
- 精确相等率只作诊断，不参与 pass/fail。

相机与 Warp 几何以已经实测标定的实现为准：`shift_px = shift_x * width`，并使用源/目标相机 shift 差进行对齐。该约定在 view-domain 标定中达到约 38.6 dB（4K）和 36.3 dB（低分辨率）。

## 4. 正式数据协议

### 4.1 原始数据与生成结果

```text
E:\DisplayAwareDataset\<scene>\
  rgb\
  depth\
  camera.json
  metadata.json

E:\dataset_out_v2\<scene>\samples\sample_XXXX\
  sample.json
  interlaced_input.png
  interlaced_mask.png
  interlaced_mask_view.png
  interlaced_gt.png
  view\                         # 验收后可删除
```

`E:\dataset_out` 是早期 12-sample 实验档；`E:\dataset_out_v2` 是正式数据根目录。

### 4.2 样本配置

```text
sources = 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
steps   = 1,2,3,5,7
modes   = interp, extrap
```

每个场景生成：

```text
20 sources * 5 steps * 2 modes = 200 samples
```

早期“2 sources * 3 steps * 2 modes = 12 samples”仅用于功能验证，不再作为正式实验协议。

### 4.3 场景与划分

首批 11 个完整场景，共 2200 samples：

| Split | Scenes | Samples |
| --- | --- | ---: |
| Train | scene001-scene008 | 1600 |
| Val | scene009 | 200 |
| Test | scene010, scene012 | 400 |

scene011 完成后作为额外跨场景泛化测试集，不参与训练。scene013 与 scene011 重复，不单独生成。

划分必须以 scene 为单位，禁止把同一 scene 的不同 view 或 sample 随机分散到 Train/Val/Test。

### 4.4 训练采样

- 4K 图像使用 256x256 crop 起步，必要时扩展到 512x512。
- 默认 50% hole-aware crop + 50% random crop。
- 禁止翻转和旋转，以免破坏 LUT 空间结构。
- Color jitter 可以作为增强。
- crop 必须记录 hole ratio，便于按难度分析。

## 5. 方法路线

### 5.1 Baselines

| ID | 方法 | 作用 |
| --- | --- | --- |
| B0 | nearest、horizontal/bidirectional interpolation、morphology、region growing | 非学习规则基线 |
| B1 | LaMa | 通用学习式图像修复基线 |
| B2 | 6-channel U-Net：RGB(3) + mask(3) | 学习式主基线 |

View-domain filling 后再 interlace 应作为独立流程基线，但不作为本项目训练目标。

### 5.2 Display-aware 主线

| ID | 方法 | 要验证的因素 |
| --- | --- | --- |
| P1 | B2 + normalized view-ID map | 显式 ownership 是否有效 |
| P2 | P1 + LUT-guided neighbor aggregation | 同 view evidence 聚合是否有效 |
| P3 | P2 + sparse view-plane consistency loss | 是否降低按 view 分组的误差 |
| P4 | 8-plane output + LUT gather | 显式 view 表示是否值得其额外参数 |

P4 是可选对比，不预设为核心贡献。8-plane 中大量位置没有直接观测，属于过度参数化，只有取得稳定收益后才能提升其研究地位。

Periodic Attention、depth/stereo consistency 和 Interlace-aligned Conv 均为扩展实验，不进入第一轮主线。

## 6. 损失与评价

### 6.1 建议损失

```text
L = L1_full + w_hole * L1_hole
  + w_ssim * L_ssim
  + w_vgg * L_vgg
  + w_view * L_view_l1
  + w_tv * L_view_tv
```

B2/P1/P2 首先固定相同的基础损失和训练预算，确保增量结果可归因。P3 才加入 view-plane loss。

### 6.2 主指标

- Hole-region PSNR / SSIM / LPIPS：主结果。
- Full-image PSNR / SSIM / LPIPS：辅助结果。
- 按 `interp` / `extrap`、step、scene 和 hole-ratio 区间分别统计。

### 6.3 View-aware 指标

- 按 LUT 将预测与 GT 分解为稀疏 view planes。
- 只在各 view 实际采样位置计算 PSNR 和 L1。
- 该指标是 interlaced-domain error 的按 view 分组，不称为完整 view-domain reconstruction quality。
- Depth-warped cross-view consistency 暂未实现，不能提前称为已完成的 View Leakage Error。

## 7. 当前实现与结果

### 7.1 已实现

- 数据读取、样本遍历与验收；
- LUT 构建、周期验证、稀疏 plane gather/scatter；
- hole-aware crop；
- B0 规则方法和 B1 LaMa；
- B2 U-Net 训练与 tiled 全图推理；
- P1 view-ID 输入；
- P2 LUT-guided neighbor features；
- interlaced-domain 与 per-view sampled-position 指标。

### 7.2 当前单样本结果

scene001/sample_0000 的现有全图结果：

| Method | Full PSNR | Hole PSNR | SSIM | View PSNR |
| --- | ---: | ---: | ---: | ---: |
| morphology | 28.03 | 12.36 | 0.9511 | 27.83 |
| LaMa | 27.10 | 11.32 | 0.9460 | 26.90 |
| B2 U-Net | **32.07** | **17.28** | **0.9636** | **31.81** |
| P1 view-ID | 31.36 | 16.34 | 0.9609 | 31.21 |
| P2 neighbor | 30.74 | 15.55 | 0.9613 | 30.58 |

这些结果只来自早期 scene001/sample_0000，不代表正式数据集结论。当前事实是 B2 优于 P1/P2，不能据此宣称显示先验有效。

## 8. 下一阶段执行顺序

1. 生成并验收 `dataset_out_v2` 的 scene001 正式 200 samples。
2. 固定 split、随机种子、训练预算、checkpoint 选择和 tiled inference 协议。
3. 在相同配置下重新训练 B2、P1、P2，至少报告多次运行的 mean +/- std。
4. 检查 P1/P2 未提升的原因，包括 crop LUT origin、特征尺度、归一化和参数量公平性。
5. 只有 P2 路线验证合理后再实现 P3；P4 保持可选。
6. 扩展至全部 Train scenes，在 scene009 调参，最终只在 scene010/scene012 报告测试结果。
7. 按 mode、step、scene 和 hole ratio 分组，避免平均值掩盖 extrap 与大 baseline 的困难样本。

## 9. 风险与判定原则

| 风险 | 判定与处理 |
| --- | --- |
| P1/P2 不如 B2 | 先排除实现和训练公平性问题；仍无收益则诚实保留为负结果并收缩方法 |
| 空洞稀疏 | 使用 hole-aware sampling，并始终报告 hole-region 指标 |
| LUT 周期过拟合固定显示参数 | LUT 和所有偏移动态生成；后续增加 display-parameter 泛化实验 |
| P4 过度参数化 | 不进入默认主线，先以辅助 head 或小规模实验验证 |
| LPIPS 不适合交织纹理 | 与 PSNR/SSIM 和主观可视化共同报告，不单独下结论 |
| Tiled inference 产生边界 | 固定 overlap、padding 和 blend，并用全图/分块一致性测试验证 |

## 10. 文档职责

| 文档 | 职责 |
| --- | --- |
| `INTERLACED_DOMAIN_PROJECT.md` | 项目范围、统一口径、当前状态与路线，发生冲突时以此为准 |
| `EXPERIMENT_PLAN.md` | 网络、损失、实验矩阵和评估协议的详细设计 |
| `DATA_GENERATION_PLAN.md` | 正式数据生成命令、场景状态、样本量与存储计划 |
| `README.md` | 当前代码的安装、命令和运行结果入口 |

桌面实验记录包含早期讨论、过程数据和历史方案，保留溯源价值，但其中的 5-view、12-sample、精确 GT 相等验收及旧路径不再覆盖本文件中的正式口径。
