# 天文底片目标识别（YOLOv5）

基于 [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) 改造的天文底片目标检测项目，面向 FITS 格式的数字化底片，开展目标检测与分类，并提供大幅底片切块处理、检测结果合并及与 SExtractor 星表匹配的研究脚本。

本项目保留了 YOLOv5 的模型与训练框架，主要修改数据读取、单通道输入、数值处理和可视化逻辑。目前为研究代码，部分应用脚本仍需按实际环境修改路径，完整流程尚需统一与验证。

## 项目内容

- **FITS 数据读取**：通过 `astropy.io.fits` 读取底片像素数据。
- **单通道适配**：训练的从零构建分支和根目录检测入口已修改为单通道输入，并调整原有的 `/ 255` 处理。
- **整张底片处理**：将大幅 FITS 底片切成带坐标信息的 PNG 小图，供后续检测使用。
- **结果合并**：根据切片偏移还原大图像素坐标，并用 NMS 去除重叠检测。
- **星表匹配**：提供与 SExtractor 结果进行边界框或位置匹配的分析脚本。

> FITS 直接检测与 PNG 切块检测是当前代码中的两条处理路径，尚未完全统一。运行前请阅读下文“当前限制”。

## 目录结构

```text
.
├── train.py                     # 训练入口
├── val.py                       # 验证入口
├── detect.py                    # 已修改的 FITS 单通道检测入口
├── export.py                    # 上游模型导出工具
├── hubconf.py                   # 上游 Torch Hub 接口
├── models/                      # 模型实现与网络配置
├── utils/                       # 数据加载、预处理、损失、绘图等工具
├── data/
│   ├── dipian.yaml              # other / galaxy 数据集配置
│   ├── dipian1.yaml             # other / star 数据集配置
│   ├── hyps/                    # 训练超参数
│   └── dipian/                  # 本地 FITS 样本，不随代码上传
├── usage/
│   ├── infer_plate.py           # 单张大幅底片切块、检测与合并
│   ├── batch_infer_plate.py     # 批量处理
│   ├── batch_infer_fail.py      # 按另一输入清单批量处理
│   ├── match_ana.py             # 与 SExtractor 检测框匹配
│   ├── match_ana_fail.py        # 与 SExtractor 星表按位置匹配
│   └── detect.py                # 应用脚本使用的另一版检测实现
├── classify/                    # 保留的上游分类代码
├── segment/                     # 保留的上游分割代码
├── runs/                        # 本地训练与推理输出，不随代码上传
├── requirements.txt
└── LICENSE
```

`classify/`、`segment/` 和其他上游工具保留供兼容或参考，本项目说明聚焦底片目标检测。`usage/` 中部分工具与根目录文件同名，但版本不同，不应直接互换。

## 环境安装

在项目根目录创建虚拟环境后安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

主要依赖包括 PyTorch、Torchvision、NumPy、OpenCV、Astropy、Pandas 和 SciPy。使用 GPU 时需安装与运行环境相匹配的 PyTorch。现有依赖文件包含版本范围，尚未提供锁定版本的复现环境。

## 数据准备

训练数据按 YOLO 检测格式组织，图像与标签文件名对应：

```text
dataset/
├── images/
│   ├── train/                   # FITS 图像
│   └── val/
└── labels/
    ├── train/                   # 同名 .txt 标签
    └── val/
```

每行标签包含一个目标，坐标相对于对应图像宽高归一化到 `[0, 1]`：

```text
class_id x_center y_center width height
```

现有两套配置的类别定义如下，使用时应与训练权重保持一致：

| 配置 | 类别 0 | 类别 1 |
| --- | --- | --- |
| `data/dipian.yaml` | other | galaxy |
| `data/dipian1.yaml` | other | star |

将选用配置中的 `path` 改为实际数据集根目录，保留相对的训练、验证目录，例如：

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test:
names:
  0: other
  1: galaxy
```

数据集、标注和训练权重需要自行准备。仓库不提供数据下载或预训练权重下载地址。

## 训练与检测

以下命令说明参数用法，尚未在本次整理中执行训练或推理验证。使用前需完成数据路径配置，并检查“当前限制”中的通道与数据读取问题。

### 训练

当前单通道构建逻辑位于从零训练分支，因此示例显式指定空初始权重和模型配置：

```bash
python train.py --weights '' --cfg models/yolov5s.yaml --data data/dipian.yaml --hyp data/hyps/hyp.scratch-low.yaml --imgsz 640 --batch-size 16 --epochs 100 --project runs/train --name dipian
```

训练结果通常保存在 `runs/train/dipian/`；同名目录已存在时，YOLOv5 默认创建递增编号目录。权重位于对应实验目录的 `weights/` 中。

### FITS 检测

使用自行训练且输入通道兼容的权重，显式传入模型、图像和数据配置：

```bash
python detect.py --weights runs/train/dipian/weights/best.pt --source data/dipian --data data/dipian.yaml --imgsz 640 --conf-thres 0.25 --save-txt --save-conf --project runs/detect --name dipian
```

检测标签位于对应输出目录的 `labels/` 中。启用上述两个保存参数后，每行格式为：

```text
class_id x_center y_center width height confidence
```

这里的框坐标是归一化坐标。根目录检测入口还会为显示结果将底片数值转换为可视化图像。

## 大幅底片与星表分析

`usage/infer_plate.py` 的处理流程为：

```text
FITS 大图 → 重叠切片 → 每片转换为灰度 PNG → 检测
         → 还原大图像素坐标 → 全局 NMS → 输出目标表
```

使用前需修改脚本中的 `large_fits_path`、`weights_path`、`temp_png_dir`、`project_save_dir`、切片尺寸和重叠量，并先解决下述 PNG 读取与通道兼容问题。临时切片目录会被脚本清空重建，应使用专门的临时目录。

该脚本当前写出的合并结果为以下六列，坐标单位为大图像素，并非归一化的 YOLO 训练标签；写出时会跳过类别 0：

```text
x_center y_center width height confidence class_id
```

`batch_infer_plate.py` 和 `batch_infer_fail.py` 使用 CSV 输入清单并记录 JSON 处理进度，需要按实际清单格式和输出目录调整。`match_ana.py` 使用包含目标坐标、边界框和标记的 SExtractor 文本结果；`match_ana_fail.py` 读取 SExtractor 星表，要求存在 `X_IMAGE`、`Y_IMAGE`、`MAG_AUTO` 等列。匹配前需统一坐标原点、坐标单位及列定义。

## 当前限制

1. **数据读取尚未统一**：共享的 `utils/dataloaders.py` 已将图像读取改为 FITS，但大图脚本输出 PNG；两者目前无法直接衔接，需要统一输入格式或按后缀分别读取。
2. **通道配置尚未统一**：根目录检测入口使用单通道预热，`usage/detect.py`、独立验证入口及训练的预训练权重构建分支仍有三通道设置。应核对权重、训练与推理的数据通道。
3. **FITS 校验仍需检查**：训练数据校验仍调用 Pillow 的图像校验逻辑，需要确认实际 FITS 样本能够通过扫描与标签缓存过程。
4. **像素处理需与模型一致**：FITS 路径保留底片数值；PNG 切片路径逐片归一化到 8 位灰度。两者不能假定使用同一模型即可获得一致结果。
5. **实验路径仍需配置**：数据 YAML 和多个应用脚本保留了原实验环境的绝对路径，尚未统一为命令行参数。
6. **尚未公布验证指标**：本 README 不声明检测精度、处理速度或已验证的部署能力，后续可补充实际实验结果和效果图。

## GitHub 文件范围

提交核心源码、数据集配置、依赖文件和项目说明。原始 FITS 底片、权重、完整 `runs/`、临时切片、缓存和虚拟环境保留在本地，并由 `.gitignore` 排除。

需要展示实验效果时，可单独挑选少量可公开的检测图片、曲线和指标放入文档目录；现有 `.gitignore` 会忽略图片，添加时需为展示目录设置例外规则。

## 来源与许可

项目基于 [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) 修改，保留原项目版权与许可声明。仓库现有许可证为 **GNU AGPL v3**，具体内容见 [LICENSE](LICENSE)。底片数据、标注和第三方星表的来源与使用条件应在公开相应材料时单独说明。
# Plate_YOLO
