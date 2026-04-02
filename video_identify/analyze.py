#!/usr/bin/env python3
"""
视频角色提取与识别系统
流水线: 镜头分割 → 人脸检测+追踪 → 特征提取 → 全局聚类 → 输出时间轴
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class FaceDetection:
    frame_no: int
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    confidence: float
    embedding: Optional[np.ndarray] = None

@dataclass
class Tracklet:
    track_id: str          # "{scene_idx}_{local_id}"
    detections: List[FaceDetection] = field(default_factory=list)
    representative_embedding: Optional[np.ndarray] = None
    best_frame_no: int = -1
    best_bbox: Optional[Tuple] = None

@dataclass
class Person:
    person_id: str
    tracklet_ids: List[str]
    appearances: List[dict]   # [{start_ms, end_ms, start_frame, end_frame}]
    total_duration_ms: int
    representative_frame_no: int
    representative_bbox: Optional[Tuple]

# ─────────────────────────────────────────────
# 阶段 1: 镜头分割
# ─────────────────────────────────────────────

def detect_scenes(video_path: str, threshold: float = 27.0) -> List[Tuple[int, int]]:
    """返回 [(start_frame, end_frame), ...] 列表"""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        print("[WARN] scenedetect 未安装，将整个视频作为单一镜头处理")
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return [(0, total - 1)]

    print("[1/4] 镜头分割中...")
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return [(0, total - 1)]

    scenes = [(s.get_frames(), e.get_frames() - 1) for s, e in scene_list]
    print(f"    检测到 {len(scenes)} 个镜头")
    return scenes

# ─────────────────────────────────────────────
# 阶段 2+3: 人脸检测 + 特征提取（InsightFace）
# ─────────────────────────────────────────────

def load_face_analyzer():
    """加载 InsightFace 分析器（检测+识别一体）"""
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        print("[ERROR] insightface 未安装，请运行: pip install insightface onnxruntime")
        sys.exit(1)

    print("[初始化] 加载 InsightFace 模型（首次运行会自动下载）...")
    app = FaceAnalysis(
        name="buffalo_sc",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app

# ─────────────────────────────────────────────
# 简易 IoU 追踪器
# ─────────────────────────────────────────────

def iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)

class SimpleTracker:
    """基于 IoU 的简易多目标追踪器"""
    def __init__(self, iou_threshold=0.3, max_lost=10):
        self.tracks = {}       # track_id -> {bbox, lost, det_count}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def update(self, detections: List[FaceDetection]) -> List[Tuple[int, FaceDetection]]:
        """返回 [(track_id, detection), ...]"""
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return []

        # 匹配现有轨迹
        matched = {}
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        if unmatched_tracks:
            iou_matrix = np.zeros((len(unmatched_tracks), len(detections)))
            for i, tid in enumerate(unmatched_tracks):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = iou(self.tracks[tid]["bbox"], det.bbox)

            while True:
                if iou_matrix.size == 0:
                    break
                max_val = iou_matrix.max()
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                tid = unmatched_tracks[i]
                matched[tid] = j
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        result = []
        matched_det_indices = set(matched.values())

        for tid, det_idx in matched.items():
            det = detections[det_idx]
            self.tracks[tid]["bbox"] = det.bbox
            self.tracks[tid]["lost"] = 0
            self.tracks[tid]["det_count"] += 1
            result.append((tid, det))

        for tid in unmatched_tracks:
            if tid not in matched:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        for det_idx in range(len(detections)):
            if det_idx not in matched_det_indices:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    "bbox": detections[det_idx].bbox,
                    "lost": 0,
                    "det_count": 1
                }
                result.append((new_id, detections[det_idx]))

        return result

# ─────────────────────────────────────────────
# 阶段 2+3: 逐镜头处理
# ─────────────────────────────────────────────

def process_scenes(
    video_path: str,
    scenes: List[Tuple[int, int]],
    face_app,
    fps: float,
    frame_step: int = 3,
    min_face_size: int = 40,
    min_det_confidence: float = 0.5,
    max_embeddings_per_track: int = 8,
) -> List[Tracklet]:
    """对每个镜头做检测+追踪+特征提取，返回所有 Tracklet"""

    print("[2/4] 人脸检测 + 追踪 + 特征提取中...")
    all_tracklets: List[Tracklet] = []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for scene_idx, (start_f, end_f) in enumerate(tqdm(scenes, desc="镜头")):
        tracker = SimpleTracker(iou_threshold=0.3, max_lost=8)
        scene_tracklets: dict[int, Tracklet] = {}

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frame_no = start_f

        while frame_no <= end_f:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_no - start_f) % frame_step == 0:
                # InsightFace 检测（BGR→RGB）
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_app.get(rgb)

                detections = []
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    w, h = x2 - x1, y2 - y1
                    if w < min_face_size or h < min_face_size:
                        continue
                    conf = float(face.det_score) if hasattr(face, "det_score") else 1.0
                    if conf < min_det_confidence:
                        continue
                    det = FaceDetection(
                        frame_no=frame_no,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        embedding=face.normed_embedding if hasattr(face, "normed_embedding") else None
                    )
                    detections.append(det)

                track_results = tracker.update(detections)

                for local_tid, det in track_results:
                    key = f"{scene_idx}_{local_tid}"
                    if key not in scene_tracklets:
                        scene_tracklets[key] = Tracklet(track_id=key)
                    t = scene_tracklets[key]
                    t.detections.append(det)

            frame_no += 1

        # 为每个 tracklet 计算代表性特征向量
        for key, tracklet in scene_tracklets.items():
            if len(tracklet.detections) < 2:
                continue  # 过滤太短的轨迹（噪声）

            # 筛选有 embedding 的帧
            valid = [d for d in tracklet.detections if d.embedding is not None]
            if not valid:
                continue

            # 按置信度排序，取最好的 N 帧
            valid.sort(key=lambda d: d.confidence, reverse=True)
            top = valid[:max_embeddings_per_track]

            embeddings = np.stack([d.embedding for d in top])
            tracklet.representative_embedding = embeddings.mean(axis=0)
            # L2 归一化
            norm = np.linalg.norm(tracklet.representative_embedding)
            if norm > 0:
                tracklet.representative_embedding /= norm

            # 最佳帧（置信度最高）
            best = top[0]
            tracklet.best_frame_no = best.frame_no
            tracklet.best_bbox = best.bbox

            all_tracklets.append(tracklet)

    cap.release()
    print(f"    共生成 {len(all_tracklets)} 条有效轨迹")
    return all_tracklets

# ─────────────────────────────────────────────
# 阶段 4: 全局聚类
# ─────────────────────────────────────────────

def cluster_tracklets(
    tracklets: List[Tracklet],
    similarity_threshold: float = 0.45,
) -> List[List[int]]:
    """
    用余弦距离 + AgglomerativeClustering 聚类
    返回 [[tracklet_idx, ...], ...] 每组是同一人
    """
    from sklearn.cluster import AgglomerativeClustering

    if len(tracklets) == 0:
        return []
    if len(tracklets) == 1:
        return [[0]]

    print("[3/4] 全局聚类中...")
    embeddings = np.stack([t.representative_embedding for t in tracklets])

    # 余弦距离矩阵
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-8, None)
    cosine_sim = normed @ normed.T
    cosine_dist = np.clip(1.0 - cosine_sim, 0, 2)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=similarity_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(cosine_dist)

    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(idx)

    return list(groups.values())

# ─────────────────────────────────────────────
# 时间轴聚合
# ─────────────────────────────────────────────

def build_timeline(
    tracklets: List[Tracklet],
    groups: List[List[int]],
    fps: float,
    merge_gap_ms: float = 500.0,
) -> List[Person]:
    """将聚类结果转换为 Person 列表，合并相邻时间段"""

    print("[4/4] 生成时间轴...")
    persons = []

    for group_idx, indices in enumerate(groups):
        person_id = f"person_{group_idx:02d}"
        tracklet_ids = []
        all_frame_ranges = []

        best_conf = -1.0
        rep_frame_no = -1
        rep_bbox = None

        for idx in indices:
            t = tracklets[idx]
            tracklet_ids.append(t.track_id)

            if not t.detections:
                continue

            frames = sorted(set(d.frame_no for d in t.detections))
            if frames:
                all_frame_ranges.append((frames[0], frames[-1]))

            # 找最佳代表帧
            for d in t.detections:
                if d.confidence > best_conf and d.embedding is not None:
                    best_conf = d.confidence
                    rep_frame_no = d.frame_no
                    rep_bbox = d.bbox

        if not all_frame_ranges:
            continue

        # 合并时间段
        all_frame_ranges.sort()
        merged = [list(all_frame_ranges[0])]
        for start, end in all_frame_ranges[1:]:
            gap_ms = (start - merged[-1][1]) / fps * 1000
            if gap_ms <= merge_gap_ms:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        appearances = []
        total_ms = 0
        for start_f, end_f in merged:
            start_ms = int(start_f / fps * 1000)
            end_ms = int(end_f / fps * 1000)
            dur = end_ms - start_ms
            total_ms += dur
            appearances.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_frame": start_f,
                "end_frame": end_f,
                "duration_ms": dur,
            })

        persons.append(Person(
            person_id=person_id,
            tracklet_ids=tracklet_ids,
            appearances=appearances,
            total_duration_ms=total_ms,
            representative_frame_no=rep_frame_no,
            representative_bbox=rep_bbox,
        ))

    # 按总出场时长降序排列
    persons.sort(key=lambda p: p.total_duration_ms, reverse=True)
    # 重新编号
    for i, p in enumerate(persons):
        p.person_id = f"person_{i:02d}"

    return persons

# ─────────────────────────────────────────────
# 输出代表性截图
# ─────────────────────────────────────────────

def save_representative_images(
    video_path: str,
    persons: List[Person],
    output_dir: str,
):
    """为每个人物保存一张代表性截图（带人脸框）"""
    img_dir = Path(output_dir) / "persons"
    img_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_cache = {}

    for person in persons:
        fn = person.representative_frame_no
        if fn < 0:
            continue
        if fn not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if ret:
                frame_cache[fn] = frame

        frame = frame_cache.get(fn)
        if frame is None:
            continue

        img = frame.copy()
        if person.representative_bbox:
            x1, y1, x2, y2 = person.representative_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, person.person_id, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out_path = img_dir / f"{person.person_id}.jpg"
        cv2.imwrite(str(out_path), img)

    cap.release()
    print(f"    代表性截图已保存至: {img_dir}")

# ─────────────────────────────────────────────
# 评估报告
# ─────────────────────────────────────────────

def generate_report(
    persons: List[Person],
    video_path: str,
    fps: float,
    total_frames: int,
    elapsed_sec: float,
    output_dir: str,
):
    total_ms = int(total_frames / fps * 1000)
    total_sec = total_ms / 1000

    report_lines = [
        "=" * 60,
        "视频角色识别 - 评估分析报告",
        "=" * 60,
        f"视频文件  : {video_path}",
        f"视频时长  : {total_sec:.1f} 秒 ({total_ms} ms)",
        f"总帧数    : {total_frames}",
        f"帧率      : {fps:.2f} fps",
        f"处理耗时  : {elapsed_sec:.1f} 秒",
        f"识别人物数: {len(persons)}",
        "",
        "─" * 60,
        "各人物出场统计",
        "─" * 60,
    ]

    for p in persons:
        coverage = p.total_duration_ms / total_ms * 100
        report_lines.append(
            f"{p.person_id}  出场时长: {p.total_duration_ms/1000:.1f}s  "
            f"占比: {coverage:.1f}%  片段数: {len(p.appearances)}"
        )
        for seg in p.appearances:
            report_lines.append(
                f"    [{seg['start_ms']/1000:.2f}s ~ {seg['end_ms']/1000:.2f}s]  "
                f"帧 {seg['start_frame']} ~ {seg['end_frame']}  "
                f"时长 {seg['duration_ms']}ms"
            )

    report_lines += [
        "",
        "─" * 60,
        "方案说明",
        "─" * 60,
        "检测模型  : InsightFace buffalo_sc (CPU)",
        "追踪方式  : IoU 匹配简易追踪器",
        "特征维度  : 512 维归一化人脸嵌入向量",
        "聚类算法  : AgglomerativeClustering (余弦距离)",
        "",
        "已知局限性:",
        "  1. 侧脸/遮挡严重时特征不稳定，可能导致同一人被拆分",
        "  2. 极相似外貌（双胞胎/相似演员）可能被合并",
        "  3. 聚类阈值 0.45 为经验值，可根据实际结果调整",
        "  4. CPU 推理速度受限，大量人物场景建议使用 GPU",
        "",
        "优化建议:",
        "  1. 使用 buffalo_l 大模型提升特征质量（需更多内存）",
        "  2. 引入 ByteTrack 替换简易 IoU 追踪器，提升遮挡鲁棒性",
        "  3. 对聚类结果进行人工标注后，可用于有监督的角色识别",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = Path(output_dir) / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n报告已保存至: {report_path}")

# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="视频角色提取与识别")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("-o", "--output", default="output", help="输出目录 (默认: output)")
    parser.add_argument("--frame-step", type=int, default=3, help="每隔 N 帧检测一次 (默认: 3)")
    parser.add_argument("--min-face", type=int, default=40, help="最小人脸像素尺寸 (默认: 40)")
    parser.add_argument("--cluster-threshold", type=float, default=0.45,
                        help="聚类距离阈值，越小越严格 (默认: 0.45)")
    parser.add_argument("--scene-threshold", type=float, default=27.0,
                        help="镜头切换检测阈值 (默认: 27.0)")
    parser.add_argument("--merge-gap", type=float, default=500.0,
                        help="合并相邻出场片段的间隔阈值 ms (默认: 500)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] 视频文件不存在: {args.video}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取视频基本信息
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧")

    start_time = time.time()

    # 四阶段流水线
    scenes = detect_scenes(args.video, threshold=args.scene_threshold)
    face_app = load_face_analyzer()
    tracklets = process_scenes(
        args.video, scenes, face_app, fps,
        frame_step=args.frame_step,
        min_face_size=args.min_face,
    )

    if not tracklets:
        print("[WARN] 未检测到任何人脸，请检查视频内容或降低 --min-face 阈值")
        sys.exit(0)

    groups = cluster_tracklets(tracklets, similarity_threshold=args.cluster_threshold)
    persons = build_timeline(tracklets, groups, fps, merge_gap_ms=args.merge_gap)

    elapsed = time.time() - start_time

    # 输出结果
    save_representative_images(args.video, persons, str(output_dir))

    result = {
        "video": args.video,
        "fps": fps,
        "total_frames": total_frames,
        "total_duration_ms": int(total_frames / fps * 1000),
        "person_count": len(persons),
        "persons": [
            {
                "person_id": p.person_id,
                "total_duration_ms": p.total_duration_ms,
                "appearance_count": len(p.appearances),
                "appearances": p.appearances,
            }
            for p in persons
        ],
    }

    json_path = output_dir / "result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结构化结果已保存至: {json_path}")

    generate_report(persons, args.video, fps, total_frames, elapsed, str(output_dir))

if __name__ == "__main__":
    main()
