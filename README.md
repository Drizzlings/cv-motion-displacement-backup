# 运动物体位移测量

本项目对应计算机视觉实践题目 8：运动物体位移测量（真实物理尺寸）。程序使用经典图像处理和几何视觉方法，不使用机器学习模型。

## 运行环境

必须使用已有的 `cv` 虚拟环境，不要在 base 环境运行或安装依赖。

如需补齐依赖，只安装到 `cv` 环境：

```powershell
E:\anaconda\envs\cv\python.exe -m pip install -r requirements.txt
```

## 输入数据

默认读取：

```text
dataset/*.mp4
```

当前项目使用纸上 `5cm x 5cm` 内框作为真实尺寸参照物。内框四个角点由用户手动标出，程序不自动寻找方框。

运行测量程序：

```powershell
E:\anaconda\envs\cv\python.exe src\motion_displacement.py
```

如果 `calibration_points.yaml` 中缺少某个视频的四点，测量程序会先弹出窗口让用户标点。按顺序点击内框四个点：左上、右上、右下、左下。

如果 `target_rois.yaml` 中缺少某个视频的目标框，测量程序会继续弹出窗口让用户框选初始目标区域。之后程序使用 Lucas-Kanade 光流跟踪目标特征点。

需要重新标定和重选目标框时，执行：

```powershell
E:\anaconda\envs\cv\python.exe src\motion_displacement.py --remark
```

只重标并处理某一个视频：

```powershell
E:\anaconda\envs\cv\python.exe src\motion_displacement.py --remark --video 1.mp4
```

## 输出结果

运行后生成：

- `output/videos/*_annotated.mp4`：标注目标框、运动轨迹、当前直线位移和累计路径长度的视频。
- `output/plots/*_displacement.png`：位移随时间变化曲线。
- `output/calibration/*_calibration.jpg`：用户标定四点可视化结果。
- `output/results.csv`：每段视频的统计结果。
- `output/report.md`：解释文档。
- `output/report.docx`：Word 版解释文档。

报告由独立程序生成：

```powershell
E:\anaconda\envs\cv\python.exe src\generate_report.py
```

## 方法概述

1. 用户手动标出内框四角。
2. 将内框四角与真实世界坐标 `(0,0), (50,0), (50,50), (0,50)` 毫米对应，计算单应性矩阵。
3. 用户在初始帧框选运动目标区域。
4. 在目标框内提取 Shi-Tomasi 角点，并使用 Lucas-Kanade 光流逐帧跟踪。
5. 使用特征点的中位位移更新目标中心，并将像素坐标转换为毫米坐标，计算相对起点的直线位移和累计路径长度。

## 说明

当前数据没有人工标注的真实位移，因此程序只输出测量值，不计算误差百分比。若后续提供每段视频的真实位移，可在 `results.csv` 基础上追加误差统计。
