# 正式档数据集生成计划

## 1. 场景盘点（2026-08-21）

扫描 `E:\DisplayAwareDataset` 结果如下：

| 场景 | camera views | rgb 文件数 | 状态 |
| --- | --- | --- | --- |
| scene001 | 200 | 200 | 完整 |
| scene002 | 200 | 200 | 完整 |
| scene003 | 200 | 200 | 完整 |
| scene004 | 200 | 200 | 完整 |
| scene005 | 200 | 200 | 完整 |
| scene006 | 200 | 200 | 完整 |
| scene007 | 200 | 200 | 完整 |
| scene008 | 200 | 200 | 完整 |
| scene009 | 200 | 200 | 完整 |
| scene010 | 200 | 200 | 完整 |
| scene011 | 200 | 2 | 渲染中 |
| scene012 | 200 | 200 | 完整 |
| scene013 | 200 | 5 | 与 scene011 重复，将合并，不单独生成样本 |

正式档首批使用 **11 个完整场景**：

```text
scene001, scene002, scene003, scene004, scene005,
scene006, scene007, scene008, scene009, scene010, scene012
```

## 2. 正式档配置

每个场景统一采用：

```text
sources = 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
steps   = 1,2,3,5,7
modes   = interp, extrap
```

每个场景样本数：

```text
20 sources x 5 steps x 2 modes = 200 samples
```

边界校验：

- `interp` 需要 `source - 4*step >= 0`；
- `extrap` 需要 `source + 7*step <= 199`；
- 最大 step 为 7，因此 `source >= 28` 且 `source <= 150`；
- 本配置 source 范围 `30-144`，全部合法。

## 3. 样本量与存储

### 3.1 样本量

| 批次 | 场景 | 每场景样本数 | 样本数 |
| --- | --- | --- | --- |
| 首批 | 11 个完整场景 | 200 | 2200 |
| 追加 | scene011（scene013 并入，完成后） | 200 | 200 |
| 合计 | 12 个场景 | - | 2400 |

### 3.2 存储估算

单个 4K 样本约 90-100 MB（含 `view/` 下的 8 张目标视图）：

- 首批 2200 样本约 220 GB；
- 全部 2400 样本约 240 GB。

训练只依赖：

```text
interlaced_input.png
interlaced_mask.png
interlaced_gt.png
sample.json
```

因此生成并验收后可以删除 `view/` 目录，每个样本约 20-25 MB，全部约 48-60 GB。

## 4. 生成命令

生成工具：

```text
D:\source_code\TransformationSolution\TransformationSolution\TransformationSolution\x64\Release\TransformationSolution.exe
```

### 4.1 首批 11 个完整场景

```powershell
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene001 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene002 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene003 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene004 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene005 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene006 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene007 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene008 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene009 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene010 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene012 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
```

### 4.2 scene011（scene013 并入，渲染完成后）

```powershell
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene011 E:\dataset_out_v2 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144
```

## 5. 生成顺序与验收

1. 先只生成 scene001 的 200 个样本。
2. 对 scene001 执行验收：

```powershell
python -B -m src.cli stats E:\dataset_out_v2\scene001
python -B -m src.cli validate E:\dataset_out_v2\scene001
```

3. 确认 mask 结构、有效区 PSNR、hole ratio 正常后，再批量生成其余场景。
4. 全部生成后执行全量验收，统计失败样本与按 mode/scene 的 hole ratio。
5. scene011（含原 scene013）渲染完成后生成并验收，再并入同一数据集根目录。
6. 验收通过后可删除各样本 `view/` 目录，释放磁盘空间。

## 6. 场景级划分建议

首批 2200 样本按场景划分：

| 集合 | 场景 | 样本数 |
| --- | --- | --- |
| Train | scene001, scene002, scene003, scene004, scene005, scene006, scene007, scene008 | 1600 |
| Val | scene009 | 200 |
| Test | scene010, scene012 | 400 |

scene011 完成后可作为跨场景泛化测试集，不再参与训练。

`split.json` 记录 `scene -> split` 映射，训练代码按场景加载，保证同场景样本不跨集合。

## 7. Focus depth 变体（可选）

正式档先生成默认 focus（`z0`）。若需要 focus 消融，为 scene001/scene005 单独生成：

```powershell
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene001 E:\dataset_out_focus3p0 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144 3.0
TransformationSolution.exe --build-dataset E:\DisplayAwareDataset\scene005 E:\dataset_out_focus15 1,2,3,5,7 30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144 15.0
```

每个 focus 变体 200 样本，只用于扩展消融，不进入主划分。

## 8. 风险与建议

- 磁盘空间：建议先确认 `E:` 剩余空间；如果不足 250 GB，可先生成 6-8 个场景。
- 生成时间：每个场景 200 个 4K 样本生成时间较长，建议按场景分批执行并保留日志。
- scene011 与 scene013 重复，合并为一个场景；全部渲染完成后统一生成样本，避免重复劳动。
- 若需快速迭代，可另行使用低分辨率场景或先取 5-6 个 source 子集，不占用正式档。
