# LingBot-World 2.0 — E 系列收口操作 SOP

> 仓库：`/Users/anshi/clawd/lingbot-world-v2` ｜ 分支：`feat/e2-paired-video-eval`（E2-new）、`feat/e1.2-adapter-v2`（E1.2）
> 环境：`source venv/bin/activate`
> 状态基线：M0–M5 + G0.7(45 视频) + E0–E1.2 + E2-new(15 视频) 均已完成，盲评材料就绪，**评分未填**
> G1 在 G0.7 盲评结论出来前保持 NOT STARTED

---

## 0. 前置检查（30 秒）

```bash
cd /Users/anshi/clawd/lingbot-world-v2 && git status -sb
git log --oneline -3
ls eval/e2_new/blind/videos/ | wc -l     # 期望 30
ls eval/g0.7/blind/videos/ | wc -l       # 期望 45
```

- 若 git 显示未提交改动，先 commit 到对应分支。
- 若视频数不足，先补齐再进入评分。

---

## 一、E2 paired 盲评收口（15 对 UMT5 vs MiniCPM+Adapter）

**目标**：判定 MiniCPM5-2B + Adapter 是否可作为 LingBot text encoder 替代。

### Step 1.1 看视频（约 20 分钟）
- 匿名视频：`eval/e2_new/blind/videos/video_001..030.mp4`
- 每对 = 2 个匿名文件（同 scene + seed，左右洗牌，文件内不泄露身份）
- 建议并排播放或逐条暂停到同帧对比；看不出差异就记 **tie**

### Step 1.2 填评分表
- 文件：`eval/e2_new/blind/human_eval_paired.csv`（15 行）
- 列：pair_id、video_a、video_b + 6 维度 + win/tie/loss + 备注
- 维度：`semantic_fidelity` / `spatial_consistency` / `camera_motion` / `temporal_stability` / `hallucination` / `overall_preference`（0–5）
- `hallucination` 是负向指标（0=无幻觉，5=严重）；其余 5 是正向
- win/loss 判定：比较 video_a vs video_b，谁在多数关键维度更优

### Step 1.3 跑汇总（评分填完后）
```bash
python scripts/summarize_e2_new_paired.py \
  --csv eval/e2_new/blind/human_eval_paired.csv \
  --map eval/e2_new/blind/blind_map.json
```
输出：overall + per-scene + per-seed 的 win/tie/loss 与各维度均分。
- `temporal_mad` 在辅助文件 `eval/e2_new/aux_metrics.json`，只作参考，不参与人工结论。

### Step 1.4 结论模板
- 若 MiniCPM+Adapter 在多数 pair 与 UMT5 **打平或更好** → 可作为低内存替代候选（内存省 60%）
- 若明显更差 → 保留 UMT5 默认，MiniCPM 分支仅作研究用

---

## 二、G0.7 A/B/C 盲评收口（45 条）

**目标**：判定 MiniCPM-V world conditioning 是否有**稳定可感知收益** → 决定 G1。

### Step 2.1 看对比材料（建议先看这个，省时间）
- **并排 montage**：`eval/g0.7/blind/montage/<scene>_ABC.mp4`（同帧三格：A/B/C）
- **同帧九宫格**：`eval/g0.7/blind/frames/<scene>_frame{1,7,13}.png`
- **评分指南**：`eval/g0.7/blind/BLIND_GUIDE.md`（7 维度各自看什么）

### Step 2.2 填评分表
- 文件：`eval/g0.7/human_eval.csv`（45 行，全匿名：blind_id + 8 维度）
- 维度：`identity_consistency` / `attribute_preservation` / `spatial_layout` / `environment_consistency` / `camera_continuity` / `temporal_stability` / `intent_fidelity` / `hallucination`（0–5，hallucination 负向）
- 提示：A=原始 prompt，B=完整 world description，C=compact world state（文件内不显示，解盲由脚本做）

### Step 2.3 跑汇总（评分填完后）
```bash
python scripts/summarize_g07_eval.py \
  --csv eval/g0.7/human_eval.csv \
  --map eval/g0.7/blind_map.json
```
输出：A/B/C 总体统计、B-A / C-A / C-B 差异、scene 分解、seed 分解、win/tie/loss。

### Step 2.4 G1 Entry Gate 判定（脚本输出后人工核对）
| 条件 | 要求 |
|---|---|
| 提升面 | C 相比 A 在 **≥4/5 scene** 平均提升 |
| 不退化 | identity_consistency 不下降、intent_fidelity 不下降 |
| 幻觉 | hallucination 不显著增加 |
| 稳定性 | ≥2/3 seeds 趋势一致 |
| 预算 | C 的 truncation 明显优于 B 或 token 更稳定 |

- **全部满足 → G1 JUSTIFIED**（可启动 G1 任务书）
- **B 提升但 C 没提升** → 优化 compact prompt 设计，不进入 G1
- **B/C 都没稳定超过 A** → **G1 NOT JUSTIFIED**，先优化 World JSON schema / VLM prompt / compact composition，不做 hidden adapter

---

## 三、Git 收口

### Step 3.1 提交评分与汇总结果
```bash
git add eval/e2_new/blind/human_eval_paired.csv eval/e2_new/*.json
git add eval/g0.7/human_eval.csv eval/g0.7/*.json
git commit -m "eval: E2 paired + G0.7 blind scores and summaries"
```

### Step 3.2 归档与标记
- 旧 E2 epoch5/20 产物：已写 `eval/e2/INVALID.md`，确认不被新流程引用（勿删，留审计）
- 各分支远端确认：
```bash
git fetch --all && git status -sb   # 应显示 0/0 或已 push
```

### Step 3.3 （可选）打基线 tag
```bash
git tag e2-eval-final a327b45    # 或当前 HEAD
git push origin e2-eval-final
```

---

## 四、最终结论报告

在项目根目录生成 `E_SERIES_FINAL.md`，包含：
1. M0–M5 移植结论（PASS，M4 13 帧 / M5 29 帧）
2. G0.7 A/B/C 盲评统计 + **G1 判定**
3. E2 paired 盲评统计 + **MiniCPM5 替代结论**
4. MLX VAE 复验结论（可选 backend）
5. 未验证事项清单（G1、variable-length ablation、G0.7 长视频二次验证）

---

## 五、状态标记速查

| 状态 | 含义 | 允许动作 |
|---|---|---|
| PASS | 已验证 | 继续下一步 |
| FAIL | 有异常 | 记录根因，不跳过 |
| BLOCKED | 有阻塞 | 说明原因与所需输入 |
| NOT RUN | 未执行 | 禁止写 PASS |

**硬约束（任何时候）**：不 merge main；不改 DiT/checkpoint/UMT5/text_embedding；G1 在 G0.7 盲评出结论前不启动；旧 INVALID 产物不参与统计。
