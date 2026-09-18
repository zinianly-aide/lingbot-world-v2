# 视觉盲评汇总报告（E2 paired + G0.7 A/B/C）

评分协议：幻觉维度反向计分（1=无幻觉最好，5=最严重）；正向维度默认无可见问题给 4。并列（tie）合法。盲评阶段不使用 encoder/variant 标签，解盲仅在汇总时通过 `blind_map.json`。

评分文件：
- `eval/e2_new/blind/human_eval_paired.json` / `.csv`
- `eval/g0.7/blind/human_eval.json` / `eval/g0.7/human_eval.csv`
- 汇总：`eval/e2_new/blind/summary.json`、`eval/g0.7/summary.json`

---

## 1. E2：UMT5 baseline vs MiniCPM+E1.2（15 对）

### Overall win/tie/loss
| 编码器 | 胜 | 负 | 平 |
|---|---:|---:|---:|
| UMT5 | 4 | 4 | 7 |
| MiniCPM+E1.2 | 4 | 4 | 7 |

结论：**总体平手**，无可稳定宣称的单侧优势。

### Per-scene
| scene | umt5 win | e12 win | tie |
|---|---:|---:|---:|
| camera_motion | 1 | 2 | 0 |
| indoor | 0 | 0 | 3 |
| outdoor | 2 | 1 | 0 |
| single_subject | 0 | 1 | 2 |
| spatial | 1 | 0 | 2 |

- 室内 3/3 并列，质量接近。
- camera_motion 略偏 E1.2（2:1），outdoor 略偏 UMT5（2:1）。
- single_subject / spatial 多数并列。

### Per-seed
| seed | umt5 | e12 | tie |
|---|---:|---:|---:|
| 42 | 2 | 0 | 3 |
| 123 | 1 | 3 | 1 |
| 2026 | 1 | 1 | 3 |

seed 间趋势不一致（42 偏 UMT5，123 偏 E1.2），**不能据此宣称某一编码器跨 seed 稳定更好**。

### 维度均分（按 encoder，X/Y 感知，n=15）
| dim（幻觉反向） | UMT5 | E1.2 |
|---|---:|---:|
| semantic_fidelity | 3.867 | 3.867 |
| spatial_consistency | 3.867 | 3.867 |
| camera_motion | 3.867 | 3.933 |
| temporal_stability | 3.733 | 3.800 |
| hallucination（越低越好） | 1.600 | 1.733 |
| overall_preference | 3.267 | 3.267 |

E1.2 在 camera/temporal 略高，幻觉均分略高（更差），overall 并列。

### 逐对解盲（摘要）
| pair | scene/seed | 结果 | 备注 |
|---|---|---|---|
| 01 | single_subject/42 | tie | 两者均保持孤树主体 |
| 02 | single_subject/123 | e12 | Y 树形更稳 |
| 03 | single_subject/2026 | tie | — |
| 04 | spatial/42 | tie | 石阵布局均稳定 |
| 05 | spatial/123 | umt5 | e12(X) 中段草地幻觉物体 |
| 06 | spatial/2026 | tie | — |
| 07 | camera_motion/42 | umt5 | e12(Y) 环境漂移更明显 |
| 08 | camera_motion/123 | e12 | umt5(X) 中帧伪影更明显 |
| 09 | camera_motion/2026 | e12 | umt5(Y) 场景漂移更明显 |
| 10 | indoor/42 | tie | — |
| 11 | indoor/123 | tie | — |
| 12 | indoor/2026 | tie | — |
| 13 | outdoor/42 | umt5 | e12(X) 末帧结构略稳于对手时仍判 umt5 更稳 |
| 14 | outdoor/123 | e12 | Y 末帧敌楼更清晰 |
| 15 | outdoor/2026 | umt5 | e12(X) 末帧略糊 |

---

## 2. G0.7：A/B/C prompt 变体（45 条，5 场景 × 3 seed × 3 variant）

变体含义（解盲后）：
- **A**：UMT5 基线（对应 E2 左侧编码器来源）
- **B**：full world prompt
- **C**：compact world prompt

### Variant adjusted_score（正向 7 维均值 − 0.5×幻觉）
| Variant | adjusted mean |
|---|---:|
| A | 3.224 |
| B | 3.133 |
| C | 3.176 |

### 成对差分
| 差分 | adjusted |
|---|---:|
| B−A | −0.091 |
| C−A | −0.048 |
| C−B | +0.043 |

### Paired wins / ties / losses（同 scene+seed）
| 对比 | W | T | L | n |
|---|---:|---:|---:|---:|
| B vs A | 2 | 10 | 3 | 15 |
| C vs A | 1 | 11 | 3 | 15 |
| C vs B | 2 | 10 | 3 | 15 |

大量并列：多数室内 / 长城 / 孤树在视觉帧上未拉开显著差距。

### Per-scene adjusted mean
| scene | A | B | C |
|---|---:|---:|---:|
| camera_motion | 2.12 | 2.17 | 2.62 |
| indoor | 3.50 | 3.50 | 3.50 |
| outdoor | 3.50 | 3.79 | 3.50 |
| single_subject | 3.50 | 3.50 | 3.50 |
| spatial | 3.50 | 2.71 | 2.76 |

- **camera_motion**：C > B > A，compact 在龙/城堡动态场景上相对更好。
- **spatial**：A 明显高于 B/C；石阵 pan 在 full/compact 上形变更常见。
- 其余场景三者接近。

### Per-seed adjusted mean（见 summary.json）
趋势未在 ≥2/3 seed 上对 C 形成稳定优于 A 的方向。

### Prompt token / 截断
| Variant | mean tokens | max tokens | truncated |
|---|---:|---:|---:|
| B | — | 265 | 0 |
| C | — | 215 | 0 |

C 的 token 预算优于 B（max 更低，且均无截断）——这是 compact 的唯一明确优势，但**尚未转化为质量稳定提升**。

### G1 Entry Gate
**结论：G1 NOT JUSTIFIED**

| 检查项 | 结果 |
|---|---|
| C 在 ≥4/5 场景优于 A | **FAIL**（1/5，仅 camera_motion） |
| identity 不下降 | PASS（C=A≈3.87） |
| intent_fidelity 不下降 | **FAIL**（C=3.80 < A=3.93） |
| 幻觉未显著上升 | PASS（C=A≈1.27） |
| ≥2/3 seed 趋势一致 | **FAIL**（1/3） |
| C token 预算优于 B | PASS（B max265 → C max215，均无截断） |

Gate 理由（脚本）：Neither B nor C shows stable improvement over A. Optimize World JSON schema / VLM prompt / compact composition before G1.

---

## 3. 综合结论与建议

1. **E2**：UMT5 与 E1.2 在 15 对上 4–4–7 平手；维度均分几乎重合。当前证据**不支持**宣称 E1.2 adapter 在盲评中稳定优于 UMT5 baseline。
2. **G0.7**：A/B/C adjusted 均值接近（A≈C>B），大量并列。**G1 Entry Gate 未通过**。
3. **明确问题场景**：
   - `camera_motion`（飞向城堡）：动态最弱，但 compact(C) 相对 A 略好。
   - `spatial`（石阵 pan）：B/C 形变更明显，拖累 gate。
4. **compact(C) 的正确解读**：token 更短且无截断，质量与 A 接近而非更好；**不能**作为“已过 G1、可扩 regen”的依据。
5. **下一步建议**：
   - 优化 spatial 的布局锁定 / stone identity，而非只改 prompt 长度。
   - camera_motion 增强前景（龙颈/缰绳）约束与背景稳定性。
   - 在 identity/intent 不降的前提下重做 compact 组合，再跑一轮盲评；通过条件仍是 C>A 的 4/5 场景 + seed 一致性。

---

## 4. 产物路径

| 文件 | 说明 |
|---|---|
| `eval/e2_new/blind/human_eval_paired.json` | E2 逐对 X/Y 维度分 + verdict |
| `eval/e2_new/blind/human_eval_paired.csv` | E2 脚本兼容 CSV（win=left/right/tie） |
| `eval/e2_new/blind/summary.json` | E2 解盲汇总 |
| `eval/g0.7/blind/human_eval.json` | G0.7 45 条维度分 + 理由 |
| `eval/g0.7/human_eval.csv` | G0.7 官方 CSV |
| `eval/g0.7/summary.json` | G0.7 全量统计 + G1 gate |
| `eval/visual_blind_eval_report.md` | 本报告 |
