# B2/P1/P2 固定训练与裸眼 3D 显示测试协议

文档状态：从 2026-08-24 起作为 B2、P1、P2 的统一协议。除非明确记录为新实验，不能改变其中的配置并直接比较结果。

## 1. 训练协议

| 项目 | 固定值 |
| --- | --- |
| Dataset | `E:\dataset_out_v2` |
| Train | scene001-scene008，1600 samples |
| Validation | scene009，200 samples |
| Test | scene010、scene012，400 samples |
| Seed | 0；多 seed 实验另行标记 |
| Crop | 256x256 |
| Batch | 8 |
| U-Net base | 32 |
| Optimizer | AdamW，lr=1e-4 |
| AMP | enabled |
| DataLoader workers | 0 |
| Decoded cache | 2 samples |
| Consecutive crops | 8 crops/sample |
| Total steps | 20,000 |
| Validation interval | 500 steps |
| Checkpoint interval | 500 steps |
| Best checkpoint | scene009 hole-region PSNR 最高 |

三种方法只改变输入表示：

```text
B2 = RGB + mask
P1 = B2 + view-ID map
P2 = P1 + LUT-guided neighbor features
```

三者必须使用相同的 Train/Val、seed、crop、batch、steps、学习率和 checkpoint 选择规则。

## 2. 推理协议

所有屏幕测试和 Test 指标使用：

```text
tile_size = 256
overlap   = 32
AMP       = enabled
```

训练 crop 和推理 tile 必须相同，避免模型在训练和显示推理时看到不同尺度的上下文。预测结果必须经过：

```text
output = input * (1 - mask) + network_prediction * mask
```

有效子像素不得被网络改写。

## 3. 测试数据协议

### 3.1 定量 Test

模型训练和 checkpoint 冻结后，才读取 scene010、scene012 的 400 samples。Test 结果不能用于选择模型、调整阈值或调整 tile 参数。

至少报告：

- full PSNR / SSIM；
- hole-region PSNR；
- valid-region PSNR；
- hole ratio；
- per-view sampled-position PSNR；
- 按 scene、mode、step 分组的均值和标准差。

### 3.2 屏幕显示样本

屏幕观感测试使用固定代表样本索引：

```text
sample indices = 0,3,4,7,9
scene010 + scene012
```

每个方法必须保存同一批样本的：

```text
interlaced_input.png
interlaced_mask.png
interlaced_gt.png
prediction.png
```

旧的 `1024 tile + 128 overlap` 结果只作为历史阶段导出，不得与本协议下的结果直接比较。

## 4. 上屏检查顺序

每次主观测试按以下顺序播放：

1. `interlaced_gt`：确认显示设备、LUT、相位、分辨率和通道顺序正确。
2. `interlaced_input`：记录原始 hole 的可见程度。
3. B2、P1、P2 prediction：使用同一屏幕参数和播放程序。

显示条件必须记录：

- 屏幕分辨率和刷新率；
- 播放软件是否缩放；
- 图像相位/视点排列设置；
- 观看距离和大致视角；
- 是否存在额外锐化、降噪、色彩管理或 gamma 处理。

## 5. 主观问题记录

每个样本至少记录以下现象：

- hole 边界是否可见；
- 是否出现亮边/暗边；
- 是否出现水平、垂直或透镜方向条纹；
- 是否出现颜色跳变或跨 view 串色；
- 视点移动时填补区域是否闪烁；
- 与 GT 相比是否有局部深度或纹理不连续。

主观结论不能只写“自然/不自然”，必须绑定 sample、method 和上述现象。

## 6. 当前阶段执行顺序

1. 用当前 1000-step checkpoint 按 256/32 协议重新导出少量样本，确认推理尺度变化本身不会产生接缝。
2. 将 B2、P1、P2 都续训到 20,000 steps。
3. 用 scene009 hole PSNR 选择最终 checkpoint。
4. 用固定代表样本上屏，并完成 GT/input/pred 三组对照。
5. 最后运行 scene010/scene012 全量定量 Test。

## 7. 禁止事项

- 不使用 Test 场景调参或选择 checkpoint。
- 不把不同 tile/overlap 的结果放在同一张主结果表中。
- 不用 full PSNR 单独判断填补质量。
- 不把 `interlaced_gt` 以外的补全 views 作为 GT。
- 不在显示参数改变后继续沿用旧 LUT 生成的预测结果。
