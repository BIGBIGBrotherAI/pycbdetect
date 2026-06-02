# pycbdetect

[libcbdetect](http://www.cvlibs.net/software/libcbdetect/) 的纯 Python 移植 —— 面向相机标定的全自动亚像素级棋盘格 / 三角网格图案检测。

![Python 版本](https://img.shields.io/badge/python-3.9+-blue.svg)
![许可证](https://img.shields.io/github/license/ftdlyc/libcbdetect)
![状态](https://img.shields.io/badge/status-beta-yellow.svg)

## 概述

`pycbdetect` 实现了完整的角点检测和标定板组装流程，原始算法由 C++ 开发，现以纯 Python 重写，运行时仅依赖 **NumPy** 和 **SciPy**。无需编译 C++ 扩展代码，可在任何支持 Python 3.9+ 的平台无缝部署。

支持的标定图案：
- **棋盘格**（Checkerboard）——经典黑白交替方格阵列
- **三角网格**（Deltille）——三角形密铺图案，适用于鲁棒标定

## 特性

- ✅ 无 C++ 编译扩展 —— 在任意 Python 3.9+ 环境中均可运行
- ✅ 最小化运行时依赖：**NumPy** + **SciPy**
- ✅ 与原版 C++ 库完全相同的算法实现：
  - 模板匹配（快速模式 / 慢速模式）
  - 海森矩阵响应法
  - 局部 Radon 变换
  - 零交叉 + 角度模态过滤
  - 迭代式亚像素位置精修
  - 二次 / 三次多项式曲面拟合
  - 相关性评分 + 非极大值抑制（NMS）
  - 基于能量驱动的标定板方向性生长组装

## 安装

```bash
# 从 PyPI 安装（正式发布后）
pip install pycbdetect

# 或克隆仓库本地安装
git clone https://github.com/ftdlyc/libcbdetect.git
cd libcbdetect/pycbdetect
pip install .
```

如需交互式可视化功能，可安装可选的 `viz` 组件：

```bash
pip install ".[viz]"
```

## 快速入门

### 第一步：检测角点

```python
import numpy as np
from pycbdetect import Params, find_corners

# 加载图片（uint8 格式，RGB 或灰度图均可）
img = np.imread("calibration_photo.jpg")

params = Params(show_processing=True)
corners = find_corners(img, params=params)

print(f"检测到 {len(corners.p)} 个角点")
```

每个检测到的角点包含以下属性：
- `.p[i]` — 亚像素坐标 `(x, y)`，类型为 `np.ndarray`
- `.r[i]` — 使用的检测半径
- `.v1[i]`, `.v2[i]` — 估计的边缘方向向量
- `.score[i]` — 质量分数（越高越好）

### 第二步：组装配准板

```python
from pycbdetect import boards_from_corners

boards = boards_from_corners(img, corners, params=params)
print(f"成功组装 {len(boards)} 块标定板")

for b in boards:
    rows, cols = len(b.idx), len(b.idx[0])
    print(f"  标定板: {rows}×{cols} 网格, {b.num} 个有效单元格")
```

### 第三步：可视化结果（可选）

```python
from pycbdetect import plot_corners, plot_boards

plot_corners(img, corners, title="检测到的角点")
plot_boards(img, corners, boards, title="组装后的标定板")
```

*(需安装 `matplotlib`；通过 `pip install "pycbdetect[viz]"` 安装)*

## 完整工作流示例

```python
import numpy as np
from pycbdetect import (
    Params, DetectMethod, CornerType,
    find_corners, boards_from_corners,
    plot_corners, plot_boards,
)

# --- 配置参数 ---
params = Params(
    show_processing=True,                  # 打印流水线进度信息
    detect_method=DetectMethod.HessianResponse,  # 初始化方法
    corner_type=CornerType.SaddlePoint,           # 棋盘格模式
    polynomial_fit=True,                          # 启用亚像素精修
    radius=[5, 7],                               # 多尺度检测半径
    score_thr=0.01,                              # 最低质量阈值
)

# --- 加载图片 ---
img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# --- 检测角点 ---
corners = find_corners(img, params=params)
print(f"角点数: {len(corners.p)}")

# --- 组装配准板 ---
boards = boards_from_corners(img, corners, params=params)
print(f"标定板块数: {len(boards)}")

# --- 可视化 ---
plot_corners(img, corners)
plot_boards(img, corners, boards)
```

## API 参考

### 核心函数

| 函数 | 签名 | 说明 |
|---|---|---|
| `find_corners` | `find_corners(img, corners=None, params=None)` | 执行完整的角点检测流水线 |
| `boards_from_corners` | `boards_from_corners(img, corners, boards=None, params=None)` | 将角点分组为结构化标定板 |
| `plot_corners` | `plot_corners(img, corners, title="角点")` | 显示角点叠加图（需 matplotlib） |
| `plot_boards` | `plot_boards(img, corners, boards, title="标定板")` | 显示标定板叠加图（需 matplotlib） |

### 数据结构

#### `Params` — 配置类

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `show_processing` | `bool` | `True` | 是否向标准错误输出各阶段处理进度 |
| `norm` | `bool` | `False` | 是否应用图像归一化预处理 |
| `norm_half_kernel_size` | `int` | `31` | 归一化滤波器的半核大小 |
| `polynomial_fit` | `bool` | `True` | 是否启��多项式曲面拟合并进行亚像素精度优化 |
| `polynomial_fit_half_kernel_size` | `int` | `4` | 多项式拟合的窗口半尺寸 |
| `init_loc_thr` | `float` | `0.01` | 初始角点位置的接受阈值 |
| `score_thr` | `float` | `0.01` | 保留角点的最低质量分 |
| `strict_grow` | `bool` | `True` | 严格模式的标定板扩张策略 |
| `overlay` | `bool` | `False` | 允许重叠的标定板假设 |
| `occlusion` | `bool` | `True` | 支持部分遮挡的标定板 |
| `detect_method` | `DetectMethod` | `HessianResponse` | 候选角点初始化方法 |
| `corner_type` | `CornerType` | `SaddlePoint` | 目标角点拓扑类型 |
| `radius` | `List[int]` | `[5, 7]` | 多尺度检测所用的半径列表 |

#### `DetectMethod` — 初始化策略枚举

| 值 | 名称 | 说明 |
|---|---|---|
| `0` | `TemplateMatchFast` | 快速模板匹配（4 对角度组合） |
| `1` | `TemplateMatchSlow` | 穷举模板匹配（大量角度组合） |
| `2` | `HessianResponse` | 海森行列式响应法（推荐） |
| `3` | `LocalizedRadonTransform` | 旋转模糊 Radon 变换 |

#### `CornerType` — 角点拓扑类型枚举

| 值 | 名称 | 说明 |
|---|---|---|
| `0` | `SaddlePoint` | 标准棋盘格角点（两条正交边） |
| `1` | `MonkeySaddlePoint` | 三角网格角点（三条对称分布的边） |

#### `Corner` — 角点容器

| 属性 | 类型 | 说明 |
|---|---|---|
| `p` | `List[np.ndarray]` | 每个角点的位置数组 `[x, y]` |
| `r` | `List[int]` | 对应每次检测所使用的半径 |
| `v1` | `List[np.ndarray]` | 第一条边缘的方向向量 |
| `v2` | `List[np.ndarray]` | 第二条边缘的方向向量 |
| `v3` | `List[np.ndarray]` | 第三条边缘方向（仅限三角网格模式） |
| `score` | `List[float]` | 每个角点的质量评分 |

##### 方法
- `clear()` — 将所有属性重置为空列表

#### `Board` — 已组装的标定板对象

| 属性 | 类型 | 说明 |
|---|---|---|
| `idx` | `List[List[int]]` | 二维网格中的角点索引表（`-1` 表示未占用） |
| `energy` | `List[List[List[float]]]` | 每单元的结构能量张量 |
| `num` | `int` | 已占用的单元格数量 |

### 内部模块

高级用户可以按需直接导入底层模块：

| 模块 | 用途 |
|---|---|
| `pycbdetect.imgproc` | 图像归一化和梯度计算 |
| `pycbdetect.get_init_location` | 各类独立初始化策略的实现 |
| `pycbdetect.filter_corners` | 扇区交替预过滤器 |
| `pycbdetect.refine_corners` | 方向估计 + Gauss-Newton 重定位 |
| `pycbdetect.polynomial_fit` | 锥形加权曲面拟合 |
| `pycbdetect.score_corners` | 相关性打分与阈值裁剪 |
| `pycbdetect.nms` | 密集型和稀疏型 NMS 例程 |
| `pycbdetect.meanshift` | 基于 MeanShift 的直方图峰值搜索 |
| `pycbdetect.board_helpers` | 标定板的初始化、生长、能量评估及过滤原语 |
| `pycbdetect.utils` | 低层辅助工具（图像补丁提取、掩码构建、卷积等） |

## 测试

安装 `dev` 额外依赖后运行冒烟测试：

```bash
pip install ".[dev]"
cd pycbdetect
python smoke_test.py
```

或者运行完整测试套件：

```bash
python -m pytest tests/ -v
```

## 性能说明

作为纯 Python 实现，`pycbdetect` 相比原始的 C++ 版本牺牲了一定的速度以换取便利性。在现代硬件上的典型耗时如下：

| 操作 | 大约时间 |
|---|---|
| 角点检测（单张图片，640×480） | 1～5 秒 |
| 标定板组装 | < 1 秒 |

降低延迟的建议：
- 如果近似位置即可满足需求，可以关闭 `polynomial_fit`
- 缩短 `radius` 列表长度（减少待评估的尺度数）
- 改用 `DetectMethod.TemplateMatchFast` 替代默认的 `HessianResponse`

## 变更记录

### v0.1.0 (2025-06)
- 首次发行
- 完整的角点检测流水线（初始化 → 过滤 → 精炼 → 评分 → NMS）
- 基于能量驱动生长的标定板组装
- 双分辨率融合机制（原生分辨率 + 缩放分辨率双重扫描）
- 借助 matplotlib 提供可视化支持

## 与其他方案的对比

| 特性 | pycbdetect | opencv.calibrateCamera | calibra_tools |
|---|---|---|---|
| 纯 Python | ✅ | ❌（需 OpenCV 绑定） | ⚠️ 混合 |
| 三角网格支持 | ✅ | ❌ | ❌ |
| 亚像素精修 | ✅（迭代高斯牛顿法） | ✅ | ✅ |
| 抗遮挡能力 | ✅ | 有限 | 因场景而异 |
| 跨平台部署 | ✅（pip 一键装） | ⚠️ 二进制 wheel | ⚠️ |

## 贡献指南

欢迎提交贡献！请按以下步骤参与项目开发：

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feat/我的新功能`）
3. 提交修改（`git commit -am "添加某项功能"`）
4. 推送到远程（`git push origin feat/我的新功能`）
5. 发起 Pull Request

提交前请务必运行冒烟测试并确保全部检查通过。

## 引用文献

1. Geiger 等人, *"Automatic Camera and Range Sensor Calibration Using a Single Shot"*（单次拍摄下的自动相机与深度传感器标定）, ICRA 2012  
   http://www.cvlibs.net/publications/GeigerEtAl_ICRA2012.pdf
2. Schönbein 等人, *"Calibrating and Centering Quasi-Central Catadioptric Cameras"*, ICRA 2014
3. Placht 等人, *"ROCHEDE: Robust Checkerboard Advanced Detection"*, ECCV 2014
4. Ha 等人, *"Deltille Grids for Geometric Camera Calibration"*, ICCV 2017
5. Duda & Frese, *"Accurate Detection and Localisation of Checkerboard Corners"*, BMVC 2018

## 许可证

GNU GPL v3 或更高版本 —— 与原 [libcbdetect](http://www.cvlibs.net/software/libcbdetect/) 采用相同许可协议。

详见 [LICENSE](../LICENSE)。
