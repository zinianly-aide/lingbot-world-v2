# 视觉 Agent 盲评评分提示词（复制粘贴版）

> 用法：把下方对应提示词整段发给视觉 agent（多模态模型）。视频文件按 pair 喂给它（一次一对）。
> 它输出 JSON 后，你把 JSON 贴回给我，我转成 CSV 并跑汇总。
> 评分全程盲评：agent 不知道哪个是 UMT5 / MiniCPM / A / B / C，只按视频本身质量打分。

---

## A. E2 Paired 评分（15 对：每次看 2 个视频）

```
你是视频生成质量盲评员。下面给你同一组输入（同一张图、同一个文字 prompt、同一个随机 seed、同一个相机运动路径）生成的两个视频，称为 video_X 和 video_Y。它们来自两套不同的文本编码方案，你不知道哪套是哪套，也不需要猜。请你只根据视频本身质量逐维度对比打分。

要求：
1. 完整看完两个视频（约 2–3 秒，13 帧）。
2. 在第 1、第 7、第 13 帧分别暂停对比（如果工具支持抽帧）。
3. 每个维度按 0–5 整数打分，并在括号里写一句话依据。
4. 不要因为一个视频"看起来更新鲜"就打高分；要比较稳定性、一致性、是否有伪影。
5. 两个视频在该维度上没有可观察差异时，给 3 分（相当）。

评分维度（每项 0–5）：
- semantic_fidelity：视频内容是否忠实于"同一输入图 + 同一 prompt"所描述的场景和主体（主体是什么、在做什么、场景是什么）。
  5=两者都完全忠实；3=都有偏差但方向一致；1=至少一个明显跑题/变成别的东西。
- spatial_consistency：物体相对位置、前后关系、遮挡、整体布局是否稳定一致。
  5=两者都稳定；3=都有轻微漂移；1=至少一个物体乱跳/穿模/位置错乱。
- camera_motion：镜头运动（前推/平移/环绕）是否平滑、连续、符合预期；有无突然跳变。
  5=两者都平滑；3=都有点抖但方向对；1=至少一个运动不连贯/跳帧/方向错误。
- temporal_stability：帧与帧之间是否稳定，有无闪烁、颜色突变、局部扭曲、物体变形。
  5=两者都干净；3=都有轻微噪点；1=至少一个明显闪烁/扭曲/果冻效应。
- hallucination：是否凭空出现输入图里没有的物体、文字、人、建筑，或场景漂移。
  反向打分：5=至少一个视频有严重幻觉；3=都有轻微额外细节；1=都无明显幻觉。
- overall_preference：综合上面所有维度，你更愿意看哪一个（或打平）。

最终判定：win（X 更好）/ tie / loss（Y 更好），给出一句话理由。

输出格式（严格 JSON，不要多余文字）：
{
  "pair_id": "<填入>，例如 pair_001",
  "semantic_fidelity_X": 0-5,
  "semantic_fidelity_Y": 0-5,
  "spatial_consistency_X": ...,
  "spatial_consistency_Y": ...,
  "camera_motion_X": ...,
  "camera_motion_Y": ...,
  "temporal_stability_X": ...,
  "temporal_stability_Y": ...,
  "hallucination_X": ...,
  "hallucination_Y": ...,
  "overall_preference_X": ...,
  "overall_preference_Y": ...,
  "verdict": "X_win" | "tie" | "Y_win",
  "reason": "一句话"
}
```

---

## B. G0.7 A/B/C 评分（45 条单视频：每次看 1 个视频 + 它对应的输入图和原始 prompt）

```
你是视频生成质量盲评员。给你：一张输入图、一条文字 prompt、一段由该图+prompt 生成的视频（13 帧，约 2–3 秒）。
该视频可能来自三种提示词变体之一（原始 prompt / 完整场景描述 / 紧凑场景描述），你不知道是哪一种，也不需要猜。请按视频本身质量逐维度打分。

要求：
1. 先看输入图和 prompt，再看视频。
2. 在第 1、7、13 帧暂停对比（尤其第 1 帧应与输入图高度相似）。
3. 每个维度 0–5 整数分，括号一句话依据。
4. 看不出问题就给 4 分（正常），不要刻意挑刺。

评分维度（每项 0–5）：
- identity_consistency：主体（人物/动物/车/树等）身份是否稳定，颜色/形状/类别是否与输入图一致。
  5=完全保持；3=轻微变化；1=变成另一个主体。
- attribute_preservation：颜色、材质、大小、位置等外观属性是否保持。
  5=完全保持；3=轻微变化；1=明显改变。
- spatial_layout：物体相对位置/布局是否稳定，无乱跳。
  5=稳定；3=轻微漂移；1=错乱。
- environment_consistency：背景/环境（房间/街道/天空/天气）是否连贯合理。
  5=连贯；3=有微小不一致；1=环境漂移。
- camera_continuity：镜头运动是否平滑连续，符合 prompt 要求的方向。
  5=平滑；3=轻微卡顿；1=跳变/方向错误。
- temporal_stability：帧间稳定，无闪烁/扭曲/果冻效应。
  5=干净；3=轻微噪点；1=明显伪影。
- intent_fidelity：是否执行了 prompt 明确要求的动作（如"相机前推"、"保持树居中"）。
  5=完全执行；3=部分执行；1=没执行。
- hallucination：是否凭空出现输入图没有的物体/文字/人，或场景漂移。
  反向打分：5=严重幻觉；3=轻微额外细节；1=无明显幻觉。

输出格式（严格 JSON，不要多余文字）：
{
  "video_id": "<填入，例如 video_001>",
  "identity_consistency": 0-5,
  "attribute_preservation": ...,
  "spatial_layout": ...,
  "environment_consistency": ...,
  "camera_continuity": ...,
  "temporal_stability": ...,
  "intent_fidelity": ...,
  "hallucination": ...,
  "reason": "一句话"
}
```

---

## C. 给视觉 agent 的总体约束（贴在最前面）

```
- 这是盲评：不要问"哪个是 baseline"，不要根据文件名猜来源。
- 视频都是真实模型输出，有正常噪点，不要把正常视频压缩噪点判为 temporal_stability 差。
- 如果两个视频肉眼几乎一样，整体偏好就打 tie，不要强行选一个。
- hallucination 是反向指标，分数越低越好（1=无幻觉）。
- 输出必须是合法 JSON，一行一个对象，不要 markdown 代码块包裹（除非你的界面要求）。
```

---

## D. 评分完成后怎么用
- 把视觉 agent 输出的 JSON 整段贴回给我。
- 我会：
  1. 转成 `human_eval_paired.csv` / `human_eval.csv`
  2. 跑 `summarize_e2_new_paired.py` / `summarize_g07_eval.py`
  3. 出 overall + per-scene + per-seed + win/tie/loss + G1 判定
