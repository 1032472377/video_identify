# 视频角色提取与识别系统

基于 AI 的影视短剧视频人物自动识别系统，精准提取每个人物的出场时间段，输出结构化结果与评估报告。

## 功能特性

- 自动镜头分割，避免跨镜头追踪错误
- 多人物同时检测与追踪
- 512 维人脸特征提取，支持复杂场景（遮挡、光线变化、服装变化、多角度）
- 全局聚类，跨镜头识别同一人物
- 输出结构化 JSON 时间轴 + 评估报告 + 每个人物代表性截图

## 技术方案

四阶段流水线：

```
视频输入
  ↓
[1] 镜头分割      PySceneDetect — 按转场切分独立镜头
  ↓
[2] 检测 + 追踪   InsightFace + IoU Tracker — 逐镜头生成人物轨迹
  ↓
[3] 特征提取      InsightFace buffalo_sc — 每条轨迹提取 512 维嵌入向量
  ↓
[4] 全局聚类      AgglomerativeClustering (余弦距离) — 跨镜头合并同一人物
  ↓
结构化输出 (JSON + 截图 + 报告)
```

## 环境要求

- Python 3.9+
- macOS / Linux（CPU 推理，无需 GPU）

## 安装

```bash
pip install -r requirements.txt
```

> macOS 若安装 insightface 报错，先执行 `brew install cmake`

## 使用

```bash
python analyze.py /path/to/video.mp4
```

指定输出目录：

```bash
python analyze.py /path/to/video.mp4 -o ./my_output
```

### 全部参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `video` | 输入视频路径 | 必填 |
| `-o, --output` | 输出目录 | `output` |
| `--frame-step` | 每隔 N 帧检测一次 | `3` |
| `--min-face` | 最小人脸像素尺寸 | `40` |
| `--cluster-threshold` | 聚类距离阈值（越小越严格） | `0.45` |
| `--scene-threshold` | 镜头切换检测灵敏度 | `27.0` |
| `--merge-gap` | 合并相邻片段的间隔阈值 (ms) | `500` |

### 调参建议

- 同一人被拆分成多个 → 调大 `--cluster-threshold`（如 0.5~0.55）
- 不同人被合并 → 调小 `--cluster-threshold`（如 0.35~0.40）
- 远景小人脸漏检 → 调小 `--min-face`（如 20~30）

## 输出结果

```
output/
├── result.json          # 结构化时间轴数据
├── report.txt           # 评估分析报告
└── persons/
    ├── person_00.jpg    # 主角代表性截图
    ├── person_01.jpg
    └── ...
```

### result.json 格式

```json
{
  "video": "1.mp4",
  "fps": 25.0,
  "total_frames": 2964,
  "total_duration_ms": 118560,
  "person_count": 20,
  "persons": [
    {
      "person_id": "person_00",
      "total_duration_ms": 42700,
      "appearance_count": 19,
      "appearances": [
        {
          "start_ms": 2520,
          "end_ms": 3840,
          "start_frame": 63,
          "end_frame": 96,
          "duration_ms": 1320
        }
      ]
    }
  ]
}
```

## 实测结果

测试视频：1080×1920，约 2 分钟，25fps，macOS CPU 环境

| 视频 | 时长 | 镜头数 | 识别人物数 | 主角占比 | 耗时 |
|------|------|--------|-----------|---------|------|
| 1.mp4 | 118.6s | 58 | 20 | person_00: 36%, person_01: 24% | ~5.4min |
| 2.mp4 | 118.2s | 69 | 21 | person_00: 30%, person_01: 26%, person_02: 23% | ~2.3min |

## 已知局限与优化方向

| 问题 | 优化方向 |
|------|---------|
| 侧脸/遮挡导致同一人被拆分 | 引入 ByteTrack 替换 IoU 追踪器 |
| CPU 推理较慢 | 使用 GPU 或 CoreML 加速 |
| 特征质量受限于小模型 | 换用 buffalo_l 大模型 |
| 聚类阈值需手动调整 | 引入自适应阈值或人工标注微调 |

## 依赖库

| 库 | 用途 |
|----|------|
| insightface | 人脸检测 + 特征提取 |
| onnxruntime | InsightFace 推理后端 |
| scenedetect | 镜头分割 |
| opencv-python | 视频读取与图像处理 |
| scikit-learn | 聚类算法 |
| numpy | 向量计算 |
| tqdm | 进度显示 |
