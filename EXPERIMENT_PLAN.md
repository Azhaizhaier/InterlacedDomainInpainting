# Interlaced-domain 空洞填补：详细试验方案（v2.1）

> 本文保留详细方法与实验设计；项目统一口径、当前状态和文档优先级见 `INTERLACED_DOMAIN_PROJECT.md`。

## 1. 背景与目标

面向裸眼 3D 显示（Lenticular Lens）的 Interlaced-domain Hole Filling：输入单视角 RGB + Depth，经过相机感知 warp 生成 8 个视图，再由 Lenticular Interlacing 映射成显示域的 interlaced 图像。由于 warp 会产生遮挡空洞，需要在 interlaced domain 直接恢复缺失子像素，而不是先完整补全 view-domain RGB 再交织。

本方案的核心故事是 **Display mapping prior**（Lenticular display-aware prior）：

```text
Problem: warp 空洞经过交织后变成沿透镜方向散乱的子像素空洞

Observation: interlaced 图像包含确定性的 pixel -> view -> subpixel 映射

Idea: 利用显示映射先验指导恢复，而不是把 interlaced 图当作普通 RGB 图

Method:
  1. View-ID 编码
  2. LUT-guided neighbor aggregation
  3. View-plane consistency loss
  4. Optional: 8-view plane 表示

Evaluation:
  Interlaced 域指标
  View 一致性指标
  显示重建质量
```

核心假设：

1. interlaced 图像中每个子像素来自哪个 view 由固定 LUT 决定，该映射可以作为网络输入与特征聚合的结构先验。
2. Interlacing 会破坏空洞的空间连续性；恢复隐藏的 view 结构可能改善重建质量。
3. 空洞在 view-domain 内是局部连续的区域，交织后按 view 分组处理可能比普通空间卷积更有效。该判断作为待验证假设，不作为结论。

Interlacing 是 lossy sampling：从 8 个完整 view（24 通道）到 interlaced RGB（3 通道），完整 view-domain 恢复是病态且不必要的。本方法不把 view-domain 重建作为中间目标；view-aware de-interlacing 只用于提取可观测的 view-specific evidence。

## 2. 任务定义与数据

### 2.1 输入输出

```text
Input : interlaced_input (HxWx3, RGB, 0-255), interlaced_mask (HxWx3, 0/255)
Target: interlaced_gt (HxWx3, RGB)
```

GT 定义必须严格为：

```text
GT views -> Lenticular Interlacing -> interlaced_gt
```

禁止使用“补全后的 views 再交织”作为训练目标。

**M1 实测修正**：`interlaced_input` 是整数像素 forward warp，`interlaced_gt` 是 Blender 对目标相机的重新渲染。位移为分数像素，取整后存在 ±1 px 固有差异，因此有效子像素不逐位等于 GT 不是 bug。scene001 有效区 PSNR 约 `30.23-37.65 dB`，几何基本正确。

说明：这里的 `30.23-37.65 dB` 是 interlaced 域有效子像素与 Blender 重渲染 GT 的对比口径；与 2.5 节 view-domain warp 标定验证的 `38.6 dB / 36.3 dB` 不属于同一对比，二者不冲突。

验收口径：mask 结构正确 + 有效区 PSNR >= 30 dB；GT 相等率仅作诊断，`--strict` 不参与默认 pass/fail。

监督方式（已确定）：主任务为 masked inpainting：

```text
pred = input * (1-mask) + network_output * mask
```

训练损失以空洞区域为主，可加全图 + 空洞加权；`interlaced_gt` 用于评估与诊断，不要求逐像素重建输入有效位置。全图重建 GT 可作为可选“显示域精修”扩展任务，不作为主任务。

### 2.2 数据集与路径

- 原始数据集：`E:\DisplayAwareDataset\<scene>\`，包含 `rgb/`、`depth/`、`camera.json`、`metadata.json`。
- 正式生成样本：`E:\dataset_out_v2\<scene>\samples\sample_XXXX\`，包含 `sample.json`、`interlaced_input.png`、`interlaced_mask.png`、`interlaced_gt.png`、`view/`。`E:\dataset_out` 仅为早期 12-sample 试验档。
- 标定/验证脚本：`calibration/` 下的 step0、validate_warp_formula、validate_axis_warp、validate_interlaced_sample 等；相机模型已验证，见 2.5 节。
- Python 项目只读取 npy + camera.json + sample.json，按 LUT 公式交织/反交织，不依赖 C++。

### 2.3 样本规模与划分

- 正式档每场景 200 个样本（20 个 source x 5 个 step x 2 种模式）；早期每场景 12 个样本的配置只用于功能验证。
- 首批 11 个完整场景按场景划分：Train 8 / Val 1 / Test 2，保证同场景样本不跨集合；scene011 完成后只作为额外泛化测试集。
- 按 `interp` / `extrap` 分别统计 hole ratio；extrap 空洞通常更大，不能混在一起掩盖差异。
- 统一 `view_num=8`，避免与旧 rule-based 工程中的 5 view 混淆。

### 2.4 裁剪、采样与增强

- 4K 图随机裁剪 256x256 或 512x512；低分辨率集为 512x288，只能裁 256x256 或使用全图。
- **必须使用 hole-aware sampling**：约 50% 优先裁包含洞的 crop，其余随机采样，否则大多数 crop 无洞，网络会退化为“什么都不做”。
- 每个 crop 记录 hole ratio，训练日志与评估时按此分组。
- 允许 color jitter；禁止水平/垂直翻转和旋转，避免破坏 LUT 空间结构。
- 如需要相位对齐增强，只按周期向量 `(1,6)` 或 `(5,2)` 的整数倍平移裁剪窗口。

### 2.5 相机内参与离轴模型标定（已完成并验证）

相机内参与离轴模型的标定已经完成，并作为数据生成与后续几何约束的已验证基础：

```text
fx = lens / sensor_width * width
   = 50 / 36 * width

shift_px = shift_x * width           # 不是 shift_x * fx
cx = width/2 - shift_px

z0 = 相机位置在前进轴上的坐标        # scene001: z0 = 2.3

目标视图坐标：
u' = x - fx * B / Z + align
align = -(shift_t - shift_s) * width
```

验证结果：

- 使用 `100 -> 101` 视图对实测验证；
- 有效区 PSNR 约 `38.6 dB`（4K）；
- 有效区 PSNR 约 `36.3 dB`（低分辨率）。

该模型用于：

1. 解释 `interlaced_input` 的几何来源；
2. 后续 depth/stereo consistency 损失的 view 间对齐；
3. 评估时计算 depth-warped cross-view consistency error；
4. 若需要重建目标视图或对比 warp 结果，作为统一几何基准。

注意：`z0`（scene001=2.3）是场景零视差平面，`sample.json` 中的 `focus_depth`（如 3.0）是显示焦点深度，二者含义不同，已在数据生成中分别记录。

## 3. 显示映射先验（约束基础）

### 3.1 LUT 公式

```text
subpixelX = x*3 + c            # c: 0=B, 1=G, 2=R
r = (subpixelX + KOFF - 3*y*THETA) mod SUBPIXEL
view = min(int(r * view_num / SUBPIXEL), view_num-1)
```

当前参数：`THETA=0.166666, KOFF=0, SUBPIXEL=4.666666, view_num=8`。LUT 与所有派生偏移由 `sample.json` / `display.json` 动态生成，不硬编码。

### 3.2 已验证的映射事实

| 事实 | 数值/结论 |
| --- | --- |
| 精确周期 `(1,6)` | `LUT[y,x,c] == LUT[y+6,x+1,c]`，沿透镜轴线方向相位不变 |
| 数学周期 `(5,2)` | 相位平移 3 个完整透镜周期，view 编号数学上重复，浮点边界可能差 1，只用于采样结构 |
| 垂直周期 `(0,28)` | 数学上成立 |
| 每像素不同 view 数 | 每个像素恰好有 3 个不同 view 的子像素，不是每个 view 都有 |
| view 平面稀疏度 | 每个 view 覆盖约 33%-42% 的像素（约 1/3 到 5/12），随 view 变化 |
| 周期内出现次数 | 1x6 周期条带（18 子像素）中各 view 出现次数为 `5,2,1,2,1,2,2,3`，固定但不均匀 |

### 3.3 Interlacing 不可逆性的精确定义

```text
8 个完整 RGB view --lossy interlace--> interlaced RGB
interlaced RGB --exact gather--> 8 个稀疏 view 平面
8 个稀疏 view 平面 --exact scatter--> interlaced RGB
```

- 完整 view 逆病态：每个像素只采样每个 view 的一个子像素，其余通道信息在交织时丢失，无法从 interlaced RGB 恢复完整 RGB view。
- Sparse-plane 分解精确可逆：把 interlaced RGB 按 LUT 归组到稀疏 view 平面，只是对已观测子像素做重排；`P_v` 按同一 LUT 重新交织可以 100% 还原原图。
- 因此 InverseInterlacing 的定位是 “view-aware de-interlacing / view evidence extraction”，不是恢复原始 view。

关键结论：

- 这是“映射的周期”，不是“内容的周期”。同一 view 在 `(y,x)` 与 `(y+6,x+1)` 是两个不同的 view 图像位置，颜色不应强制相等。
- `P_v[y,x]` 是稀疏 view 平面，只能在 view 实际出现的像素位置计算监督，或先补全缺失位置。
- 显示参数改变时映射随之改变，所有周期向量与邻域偏移必须由 LUT 动态导出。

## 4. 方法设计（按主线收敛）

### 4.1 输入级：Display-aware encoding（P1）

第一版核心输入：

```text
RGB(3) + mask(3) + view-ID map(3) = 9 通道
```

- `view-ID map[y,x,c] = LUT[y,x,c] / (view_num-1)`，直接告诉网络该子像素属于哪个 view。
- 透镜相位 `phase = r / SUBPIXEL` 与 interlaced depth 不作为第一版核心，放入消融/扩展。

### 4.2 结构级：LUT-guided neighbor aggregation（P2）

先做实现与解释成本最低的版本：

- 对每个子像素，按 LUT 预计算同 view 邻域偏移：`(1,6)`、`(5,2)` 及其组合/倍数。
- 再计算相邻 view 邻域偏移：在局部窗口中按 view 差筛选。
- 聚合时按邻居有效性加权，避免其他 view 的洞影响当前 view 的特征。
- 生成固定数量特征通道作为网络输入，或作为第一个卷积层的附加分支。

后续可选扩展（不在第一版主线）：

- Interlace-aligned Conv：固定偏移卷积，等价于把邻域聚合做成网络层。
- Periodic Attention：低分辨率稀疏 attention，支持长程同 view 信息。
- 可变形卷积：初始偏移设为 LUT 周期向量，由网络微调。

### 4.3 损失级：View-plane consistency loss（P3）

保持有效像素：

```text
out = input * (1-mask) + pred * mask
```

重建损失改为：

- 全图 masked L1：在 mask 合成后的全图上计算，保持有效像素不变。
- 空洞区域加权 L1：空洞占比只有约 0.2%-1%，需要单独加权。
- SSIM / VGG perceptual：在 mask 合成后的全图上计算，避免在稀疏空洞区域上不稳定。

View 平面损失：

- 按 LUT 反交织得到稀疏 view 平面 `P_v[y,x]`。
- `P_v` 与 GT 平面只在 view 实际出现的采样位置计算 L1。
- View 平面 TV 只在 view-domain 内相邻的采样位置对上计算，不按显示域 4 邻域直接算。

可选扩展：

- 立体一致性：需要先把每个 view 缺失通道补全为完整 RGB，再用源 depth 与 baseline 做 warped photo-consistency。
- 立体一致性的 view 间位移与对齐使用 2.5 节已标定的 `fx`、`shift_x`、`z0` 公式，不再重新估计相机参数。
- 深度先验：空洞位于遮挡边界，填充内容应主要来自背景层。

### 4.4 表示级：View-aware evidence decomposition（P4）

定位为辅助表示/实验，不是硬约束。de-interlace 与 re-interlace 是同一 LUT 下的互逆置换，仅重排已观测子像素，不执行完整 view 恢复：

```text
主输出: interlaced RGB
辅助 head: 稀疏 view 平面 P_v
```

P4 变体：

```text
network output: O[y,x,v], v=0..7
out[y,x,c] = O[y,x,LUT[y,x,c]]
```

注意：

- 每像素只有 3 个子像素被 LUT 采样，8-plane 中其余 5 个条目只能靠辅助损失约束，属于过度参数化。
- 因此 8-plane 不作为“输出合法交织”的硬保证，也不作为严格重参数化。
- 该变体用于验证“显式 view 表示”是否比 RGB 主输出加辅助 head 更强，作为论文可选实验。

## 5. 网络架构

### 5.1 B2：U-Net baseline

- 输入：6 通道（RGB 3 + mask 3）。
- 输出：3 通道。
- 编码器：ResNet18 或轻量 encoder-decoder，特征层 64/128/256/512。

### 5.2 P1：+ View-ID

- 输入：9 通道（RGB 3 + mask 3 + view-ID 3）。
- 其余结构与 B2 相同。

### 5.3 P2：+ LUT-guided neighbor aggregation

- 在 P1 基础上加入 LUT 邻域聚合分支。
- 可选替换第一层卷积或作为附加输入特征。

### 5.4 P3：+ View-plane consistency loss

- 网络结构同 P2。
- 增加稀疏 view-plane head 与 view 平面损失，不改主输出表示。

### 5.5 P4：View-plane 表示

- 输出 head 改为 8 通道 `O[y,x,v]`。
- 通过固定 LUT gather 得到 RGB，再做 mask 合成。
- 与 P3 对比，验证显式表示是否优于 RGB 输出加辅助 head。

## 6. 损失函数与权重

建议总损失：

```text
L = L1_full
  + w_hole * L1_hole
  + w_ssim * L_ssim
  + w_vgg * L_vgg
  + w_view * L_view_l1
  + w_tv * L_view_tv
  + w_stereo * L_stereo (可选)
```

初始权重建议：

| 损失 | 权重 |
| --- | --- |
| 全图 masked L1 | 1.0 |
| 空洞加权 L1 | 1.0 - 2.0 |
| SSIM（全图合成后） | 0.1 |
| VGG perceptual（全图合成后） | 0.05 |
| view 平面 L1（采样位置） | 0.5 |
| view 平面 TV（view-domain 邻接对） | 0.05 - 0.2 |
| stereo consistency（可选） | 0.1 |

## 7. 训练配置

| 配置项 | 数值 |
| --- | --- |
| 4K 分辨率 | 256x256 起步，512x512 可选 |
| 低分辨率集 | 256x256 或全图，禁止 512x512 |
| batch size | 256: 8-16；512: 4-8 |
| optimizer | AdamW |
| lr | 1e-4，cosine decay |
| epochs | B2/P1/P2 约 100-200；P3/P4 约 200-300 |
| 混合精度 | 可用时开启 |
| 随机种子 | 固定并记录 |
| 采样策略 | 50% hole-aware crop + 50% 随机 crop |

验证策略：

- 每个 epoch 在 val 集固定裁剪或全图上评估 hole PSNR。
- 记录 best checkpoint 与 last checkpoint。
- 日志记录每个 crop 的 hole ratio，训练后按 hole ratio 分组分析。

## 8. 评估协议

### 8.1 Interlaced-domain 指标

- 全图 PSNR / SSIM / LPIPS。
- 空洞区域 PSNR / SSIM / LPIPS（主指标）。
- 按 `interp` / `extrap` 分组报告。

### 8.2 View-aware 指标

- 直接对预测与 `interlaced_gt` 做 de-interlace：`P_v_pred = deinterlace(pred)`，`P_v_gt = deinterlace(interlaced_gt)`。
- 比较稀疏 view 平面在采样位置的 PSNR/L1，不要求恢复完整 RGB view。
- 该指标本质是 interlaced 子像素误差按 view 分组，用于暴露 per-view 难度差异，不称其为 view-domain 重建质量。
- 如需近似 view 图像可视化，可用固定插值补全缺失通道，但明确标注为近似重建，不进主指标。

### 8.3 View Leakage Error（先定义再使用）

建议定义为可计算的组合指标，所有子指标均基于稀疏 view 平面，不涉及完整 RGB view 恢复：

1. Per-view 采样位置误差：各 view 平面 PSNR/L1，反映单 view 重建质量。
2. Depth-warped cross-view consistency error：用源 depth 与 baseline 将 view 间对齐后比较，反映是否填了错误 view 的内容。
3. Optional cross-view contamination：预测 view 平面与其他 view GT 平面的相似度，用于检测跨 view 串扰。

该指标的具体公式需要在实现 M3 前冻结，避免事后调整。

**M3 已冻结**：当前实现为按 LUT 反交织得到稀疏 view 平面，计算每个 view 采样位置的 PSNR/L1，并报告全部 view 的平均值；source view 对应平面无空洞时 PSNR 为 inf。Depth-warped consistency 需要 depth 与相机参数参与，保留为后续扩展指标。

### 8.4 推理协议

- 显存允许时 4K 全图推理。
- 否则使用 overlap tile + 边界 padding + blend，保证拼接处无接缝。
- 低分辨率集直接全图推理或 256 分块。
- 评估时输入/GT 使用同一推理协议。

### 8.5 可视化与统计

- 展示 interlaced 输入、mask、预测、GT 对比。
- 展示反交织后的稀疏 view 平面与 GT view 平面。
- 统计口径：mean +/- std，模型对比使用 paired t-test 或 Wilcoxon。
- 如条件允许，补充裸眼 3D 或光场显示主观评测。

## 9. 实验矩阵

### 主线实验

| ID | 配置 | 验证假设 | 关键指标 |
| --- | --- | --- | --- |
| B0 | 规则法：最近邻、水平/双向插值、形态学、region growing | 非学习基线水平 | hole PSNR |
| B1 | LaMa（学习式强 baseline，单独列出） | 通用图像修复是否能直接处理交织空洞 | hole PSNR |
| B2 | U-Net，6 通道输入 | 学习式基础模型是否超过规则法与 LaMa | hole PSNR |
| P1 | B2 + view-ID 编码（9 通道） | 显示映射信息是否带来增益 | hole PSNR |
| P2 | P1 + LUT-guided neighbor aggregation | 同 view 邻域聚合是否优于普通卷积 | hole PSNR + view PSNR |
| P3 | P2 + view-plane consistency loss | view 平面监督是否降低跨 view 串扰 | view PSNR + leakage |
| P4 | P3 的 view-plane 表示（8-plane + gather） | 显式 view 表示是否优于 RGB 输出 | 全指标 |

### 可选扩展实验

| ID | 配置 | 验证假设 | 关键指标 |
| --- | --- | --- | --- |
| X1 | P3 + Periodic Attention | 长程同 view 信息是否有帮助 | hole PSNR + view PSNR |
| X2 | P3 + depth 输入与立体一致性 | 深度/几何先验是否带来增益 | view PSNR + leakage |
| X3 | P3 + Interlace-aligned Conv | 固定偏移卷积是否优于邻域聚合特征 | hole PSNR + view PSNR |

增量原则：主线每个实验只在上一实验基础上加一项，结论可归因；X1-X3 只在与 P3 对比时报告，不进入核心故事。

## 10. 消融实验

- view-ID 编码开/关。
- LUT-guided neighbor aggregation 开/关，或改为普通卷积。
- view 平面损失权重（0 / 0.1 / 0.5 / 1.0）。
- view 平面 TV 权重。
- phase 编码开/关（作为 view-ID 的消融）。
- depth 输入开/关。
- hole-aware sampling 比例（0% / 50% / 100%）。
- 裁剪尺寸 256 vs 512。
- P4 中 8-plane 输出 vs RGB 输出加辅助 head。

## 11. 风险与备选方案

| 风险 | 应对 |
| --- | --- |
| 样本少且空洞稀疏 | hole-aware sampling、随机裁剪扩增、预训练 encoder、跨场景验证 |
| LUT 浮点边界导致周期不稳定 | 周期向量只用于采样，不用硬相等；偏移由动态 LUT 生成 |
| view 平面稀疏，损失覆盖不足 | 只在采样位置计算；必要时固定插值补全后加辅助损失 |
| View Leakage Error 难以定义 | 先在 M3 前冻结公式，报告多个子指标 |
| 4K 推理边界效应 | 统一 tile + padding + blend 协议 |
| LPIPS/VGG 在 interlaced 子像素纹理上不可靠 | interlaced 域与 view 域各报一组，或注明选择依据 |
| 8-plane 过度参数化 | 先做 P3 辅助 head，P4 只作为对比实验 |
| 显示参数变化 | LUT 与偏移全部动态生成，方案不绑定单一参数 |
| 计算资源有限 | 256x256 起步，先完成 B0-B2/P1/P2，再做 P3/P4 |

## 12. 里程碑

1. M1：数据读取、LUT 工具、样本验收、hole-aware sampling、hole ratio 统计。
2. M2：B0 规则法、B1 LaMa、B2 U-Net baseline。
3. M3：P1 view-ID 编码、P2 LUT-guided neighbor aggregation，冻结 View Leakage 指标定义。
4. M4：P3 view-plane consistency loss、P4 view-plane 表示对比。
5. M5：可选扩展 X1-X3，按需选择。
6. M6：全指标汇总、消融结论与最终报告。

## 13. 待确认问题

- 推理阶段是否始终能拿到源 depth？若可以，depth 应作为正式输入；否则只能作为训练期损失。
- 显示参数是否固定？若不固定，需在数据卡片中记录并动态生成 LUT。
- view 目录中的 GT 与 `interlaced_gt` 是否完全一致，作为 view 平面监督前先验证。
- 主观评测是否有裸眼 3D 屏条件，还是只能使用 view-domain 代理指标。
- View Leakage Error 的最终公式需要用户确认后再冻结。

## 14. 附录：LUT 与稀疏 view 平面伪代码

```python
# LUT generation
for y in range(H):
    for x in range(W):
        for c in range(3):
            subpixel_x = x * 3 + c
            r = (subpixel_x + KOFF - 3 * y * THETA) % SUBPIXEL
            lut[y, x, c] = min(int(r * view_num / SUBPIXEL), view_num - 1)

# De-interlace to sparse per-view planes
planes = [np.full((H, W), np.nan) for _ in range(view_num)]
for y in range(H):
    for x in range(W):
        for c in range(3):
            v = lut[y, x, c]
            planes[v][y, x] = output[y, x, c]

# Re-interlace from planes (P4 variant)
for y in range(H):
    for x in range(W):
        for c in range(3):
            v = lut[y, x, c]
            output[y, x, c] = planes[v][y, x]
```

de-interlace 与 re-interlace 在固定 LUT 下互为逆置换：前者把已观测子像素按 view 归组，后者按同一 LUT 还原 interlaced 图像，两者都不恢复交织时丢失的通道信息。

已核实数据：

- 每个像素恰好有 3 个不同 view。
- 每个 view 覆盖约 33%-42% 的像素。
- 1x6 周期条带中各 view 出现次数为 `5,2,1,2,1,2,2,3`。

本文件只描述方案，不涉及具体实现。
