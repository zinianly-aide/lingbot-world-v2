# G0.7 盲评指南

## 概述

本目录包含 G0.7 Real A/B/C Evaluation 的盲评材料。45 条视频（5 scenes × 3 seeds × 3 variants A/B/C），唯一变量为 text conditioning：

- **A** = 仅原始 user prompt
- **B** = 原始 prompt + 完整 World JSON description
- **C** = 原始 prompt + compact structured world prompt

所有视频使用相同输入图片、相同 camera/action path、相同 frame_num=13、相同 size、相同 seed、相同 checkpoint、相同 MPS pipeline。

## 辅助对比材料（非盲评文件，仅供参考）

> 以下材料标注了 A/B/C，是辅助你看出差异的工具。正式评分仍请基于 `human_eval.csv` 的匿名视频独立判断，避免被辅助材料的标注影响。

- `montage/`: 5 scenes × 3 seeds = 15 条并排对比视频（A 左 / B 中 / C 右，同帧对齐）
- `frames/`: 每个 scene×seed 的 frame 1/7/13 三行对比图（A 上 / B 中 / C 下）
- 命名：`scene_01..05`（single_subject=01, spatial=02, indoor=03, outdoor=04, camera_motion=05）

## 评分维度（0-5 分，5 = 明显更好/更一致）

| 维度 | 看什么 |
|---|---|
| **intent_fidelity** | 视频是否执行了 user prompt 的意图（如"相机缓慢前移"是否真的前移了） |
| **identity_consistency** | 主体身份/颜色/外观是否在帧间保持一致，不发生变异 |
| **attribute_preservation** | 主体关键属性（颜色、形状、材质）是否保留，不被改变 |
| **spatial_layout** | 物体相对位置/左右/前后/远近/遮挡关系是否合理且一致 |
| **environment_consistency** | 环境（房间/道路/天空/建筑/树木）是否稳定，不发生漂移 |
| **camera_continuity** | 相机运动是否平滑连续，符合 action_path，无跳变/抖动 |
| **temporal_stability** | 帧间是否稳定，无闪烁/抖动/物体忽隐忽现 |
| **hallucination**（负向） | 有无凭空多出物体/场景漂移/语义偏移。0=无幻觉，5=严重幻觉 |

### 派生分数

- **positive_score** = 前 7 个正向维度的均值（intent + identity + attribute + spatial + environment + camera + temporal）
- **adjusted_score** = positive_score − hallucination_penalty
- **hallucination 是负向指标**：分数越高越差，会从 positive_score 中扣除

## Win / Tie / Loss 判定

在 summarize 脚本中，对同一 (scene, seed) 下的 A/B/C 三元组进行配对比较：

- **win**: 某 variant 的 adjusted_score 严格高于另一 variant
- **tie**: 两者 adjusted_score 相等（或差异在 rounding 内）
- **loss**: 严格低于

### "看不出差异 → 记 tie" 是合法结论

如果 A/B/C 三条视频在关键维度上没有可感知的差异，**请如实给相近的分数，让 summarize 判定为 tie**。不必强行分出胜负。

大量 tie 说明 MiniCPM-V world conditioning 对输出质量的影响不显著——这本身就是 G1 Entry Gate 的有效输入（若 C 不能稳定超过 A，则 G1 NOT JUSTIFIED）。

## 评分流程

1. 打开 `human_eval.csv`（45 行匿名视频，blind_id = video_001..045）
2. 播放 `blind/videos/video_0NN.mp4`，逐行评分
3. 如需对比辅助材料，参考 `montage/` 和 `frames/`（但请注意这些标注了 A/B/C，可能影响盲评客观性）
4. 全部 45 行填完后，运行：
   ```bash
   cd /Users/anshi/clawd/lingbot-world-v2
   source venv/bin/activate
   python scripts/summarize_g07_eval.py
   ```
5. 脚本输出 A/B/C 统计（按 variant/scene/seed 的 mean/median/std、各维度、B-A/C-A/C-B、paired win/tie/loss）和 G1 Entry Gate 判定

## G1 Entry Gate 判定标准

G1 JUSTIFIED 需同时满足：
1. C 相比 A 在至少 4/5 个 scene 类型中平均提升
2. identity_consistency 不下降
3. intent_fidelity 不下降
4. hallucination 不显著增加
5. 至少 2/3 seeds 趋势一致
6. C 的 truncation 明显优于 B 或 token budget 更稳定

否则 G1 NOT JUSTIFIED，先优化 World JSON schema / VLM prompt / compact composition。

## 注意事项

- 所有视频为 13 帧，时长约 0.5-1 秒，差异可能微妙
- outdoor 场景分辨率为 768×512（input image aspect ratio 决定），其余为 832×464
- temporal_mad 等客观指标见 `metrics_objective.json`（辅助参考，不参与人工结论）
- 盲评映射唯一权威来源：`blind_map.json`（shuffle_seed=42），评分完成后解盲
