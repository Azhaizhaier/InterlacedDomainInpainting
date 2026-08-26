# Interlaced-domain Inpainting

面向裸眼 3D 显示的 interlaced-domain 空洞填补。当前已实现数据/LUT 工具、B0/B1/B2 基线、P1/P2 显示先验实验和评估链路。

研究目标、统一口径、当前结论与后续路线见 [`INTERLACED_DOMAIN_PROJECT.md`](INTERLACED_DOMAIN_PROJECT.md)；详细实验设计见 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)；正式数据生成见 [`DATA_GENERATION_PLAN.md`](DATA_GENERATION_PLAN.md)。

B2/P1/P2 的固定训练与裸眼 3D 显示测试协议见 [`DISPLAY_TEST_PROTOCOL.md`](DISPLAY_TEST_PROTOCOL.md)。

以下使用 `E:\dataset_out` 的命令用于复现实验早期 12-sample 数据；正式 200-sample 数据根目录为 `E:\dataset_out_v2`，运行正式实验时替换 `--dataset-root` 即可。

## 用法

```powershell
python -B -m src.cli stats "E:\dataset_out"
python -B -m src.cli validate "E:\dataset_out\scene001"
python -B -m src.cli validate "E:\dataset_out\scene001" --strict
```

`stats` 按 mode/scene 报告 interlaced hole ratio；`validate` 以“文件完整、mask 二值、`mask_view` 一致、有效区 PSNR >= 30 dB”作为验收条件，并额外报告输入与 GT 的相等比例。`--strict` 会追加输出精确相等诊断，但不改变 pass/fail。

## 单元测试

```powershell
python -B -m unittest discover -s tests -v
```

测试会创建工作区内的临时样本目录，当前只读沙箱下需要授予运行权限。

## M2 基线

规则法基线：

```powershell
python -B -m src.run_baselines --dataset-root "E:\dataset_out" --scene scene001 --max-samples 1 --out-dir "outputs\b0"
```

默认运行 nearest、horizontal、bidirectional、morphology，结果写入 `baseline_results.csv`。可选 `vertical`、`region_growing`，以及 `lama`（需要先安装 `simple_lama_inpainting`）。

U-Net baseline 冒烟训练（单场景兼容模式，不启用独立验证集）：

```powershell
python -B -m src.train_unet --dataset-root "E:\dataset_out" --scene scene001 --crop-size 64 --steps 20 --out-dir "outputs\b2"
```

训练使用 masked inpainting 监督：`pred = input * (1-mask) + network_output * mask`。

B2 全图 tiled 推理：

```powershell
conda run -n InterlacedDomainInpainting python -B -m src.eval_unet --checkpoint "outputs\b2_gpu\unet_b2.pt" --dataset-root "E:\dataset_out" --scene scene001 --sample-index 0 --tile-size 256 --overlap 32 --base 32 --device cuda --out-dir "outputs\b2_full"
```

合并 B0/B1/B2 结果表：

```powershell
python -B -m src.summarize_baselines --inputs "outputs\b0\baseline_results.csv" "outputs\b1\baseline_results.csv" "outputs\b2_full\b2_full_eval.csv" --out "outputs\baseline_summary.csv"
```

## 正式训练

不指定 `--scene` 时，训练器默认只使用 scene001-scene008，并以 scene009 作为验证集；scene010 和 scene012 不会被训练或调参读取。解码后的 4K 图像使用受控 LRU 缓存，同一 batch 默认从同一个样本抽取 8 个不同 crop，避免完整数据集训练时耗尽内存或反复解码 PNG。

B2 正式训练与断点续训：

```powershell
conda run -n InterlacedDomainInpainting python -B -m src.train_unet --dataset-root "E:\dataset_out_v2" --crop-size 256 --steps 20000 --batch-size 8 --base 32 --device cuda --input-mode baseline --out-dir "outputs\formal_b2_seed0"

conda run -n InterlacedDomainInpainting python -B -m src.train_unet --dataset-root "E:\dataset_out_v2" --crop-size 256 --steps 20000 --batch-size 8 --base 32 --device cuda --input-mode baseline --resume "outputs\formal_b2_seed0\unet_baseline.last.pt" --out-dir "outputs\formal_b2_seed0"
```

`--steps` 表示目标优化步数，续训时仍填写最终目标值。默认启用 AMP，每 500 steps 在固定验证 crops 上评估 full/hole PSNR，并按主指标 hole PSNR 保存 best weights，同时保存可恢复的 `.last.pt`。`--cache-samples 2` 约占 166 MB 解码缓存；增加 `--workers` 时每个 worker 都有独立缓存，需要相应计算内存。

RTX 4080 SUPER 实测（256 crop、batch 8、base 32、AMP、workers 0）：B2 稳定约 `0.53-0.54 s/step`，P2 约 `0.63 s/step`。因此 20k steps 约需 B2 `3.0 h`、P2 `3.5 h`，另加定期验证时间。未按样本分组裁剪时 B2 约 `3.10 s/step`；当前默认 `--crops-per-sample 8` 将每个 batch 的 8 个 crop 取自同一 4K 样本，避免重复 PNG 解码。

P1/P2：

```powershell
conda run -n InterlacedDomainInpainting python -B -m src.train_unet --dataset-root "E:\dataset_out_v2" --crop-size 256 --steps 20000 --batch-size 8 --base 32 --device cuda --input-mode view_id --out-dir "outputs\formal_p1_seed0"

conda run -n InterlacedDomainInpainting python -B -m src.train_unet --dataset-root "E:\dataset_out_v2" --crop-size 256 --steps 20000 --batch-size 8 --base 32 --device cuda --input-mode neighbor --out-dir "outputs\formal_p2_seed0"
```

全图推理时用 `--input-mode view_id` 或 `--input-mode neighbor` 对应权重。

## M1 验证结果

使用真实显示参数（`THETA=0.166666, SUBPIXEL=4.666666, view_num=8`）生成 4K LUT 后：

- 精确周期为 `(1,6)`。
- 每个 view 的像素覆盖率在 `32.14%` 到 `42.86%` 之间。
- `interlaced_input` 的有效子像素与 `interlaced_gt` 并不完全相等。

原因来自数据生成方式：`interlaced_input` 是整数像素 forward warp，`interlaced_gt` 是 Blender 对目标相机的重新渲染，位移是分数像素，取整后存在 ±1 px 的固有差异。scene001 有效区 PSNR 范围约 `30.23-37.65 dB`，几何基本正确。

scene001 的 12 个样本均通过结构验收；interp 平均 hole ratio 约 `4.44%`，extrap 约 `6.88%`。

M2/M3 同口径结果（scene001/sample_0000，全图评估）：

| 方法 | full PSNR | hole PSNR | SSIM | view PSNR |
| --- | --- | --- | --- | --- |
| nearest | 21.76 | 5.70 | 0.8963 | 21.62 |
| horizontal | 26.81 | 11.01 | 0.9448 | 26.49 |
| bidirectional | 27.25 | 11.49 | 0.9477 | 27.10 |
| morphology | 28.03 | 12.36 | 0.9511 | 27.83 |
| lama | 27.10 | 11.32 | 0.9460 | 26.90 |
| unet_tiled_baseline | 32.07 | 17.28 | 0.9636 | 31.81 |
| unet_tiled_view_id | 31.36 | 16.34 | 0.9609 | 31.21 |
| unet_tiled_neighbor | 30.74 | 15.55 | 0.9613 | 30.58 |

B2 U-Net GPU 正式训练（RTX 4080 SUPER，256x256 crop，1000 步，batch 8，base 32）后，crop 平均 full PSNR 约 `31.75 dB`。

B1 使用 `simple_lama_inpainting`，模型权重已由该包自带，无需额外下载。

B2 全图 tiled 推理使用 256 tile + 32 overlap，结果为 `full PSNR 32.07 dB`、`hole PSNR 17.28 dB`、`SSIM 0.9636`。

M3 的 P1（view-ID）与 P2（LUT-guided neighbor aggregation）已在相同配置下训练并全图评估。当前 scene001/sample_0000 上，B2 baseline 仍然最优；P1/P2 尚未带来提升，后续需要调整邻居聚合实现或检查训练稳定性。

View-aware 指标已冻结为按 LUT 反交织后的 per-view 采样位置 PSNR 均值；source view 对应平面没有空洞，PSNR 为 inf。

## 监督方式

主任务为 masked inpainting：

```text
pred = input * (1-mask) + network_output * mask
```

训练损失以空洞区域为主，可加全图 + 空洞加权；`interlaced_gt` 用于评估与诊断，不要求逐像素重建输入有效位置。全图重建 GT 可作为可选“显示域精修”扩展任务。

## 目录

```text
src/luts.py       LUT 构建、de-interlace / re-interlace、周期与 view 统计
src/sample.py     样本元数据、PNG 加载、单样本与数据集验收
src/dataset.py    hole-aware 随机裁剪与采样器
src/cli.py        stats / validate 命令行入口
tests/            单元测试
```
