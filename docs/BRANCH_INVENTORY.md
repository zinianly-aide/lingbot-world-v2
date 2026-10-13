# Branch Inventory / 远端分支台账

> 2026-10-14 远端实际盘点：原有 20 个分支 + 本轮新增 1 个集成分支。状态为治理建议，**不代表在 GitHub 设置了保护/只读权限**。具体策略见 [BRANCH_STRATEGY.md](BRANCH_STRATEGY.md)。

| 分支 | 分组 | 基础/上下游 | 用途与后续动作 |
| --- | --- | --- | --- |
| `main` | stable | 现 HEAD `1895d30` | 已发布基线；不自动合并 |
| `integration/quest-streaming` | **active** | 自 C7.2 `1a78ac1` 建立 | 今后 Quest/持续生成/播放优化唯一集成入口 |
| `feat/c7.2-playback-telemetry` | baseline | C7 → C7.2 | 完整播放策略/遥测历史基线；不再作为新特性目标 |
| `feat/c7-world-session` | legacy | C6 → C7 | 持久 session 与缓冲模型复用；Draft PR #4 |
| `feat/c6-continuous-stream` | legacy | Cloud → C6 | C6 连续分块流；Draft PR #3 |
| `feat/cloud-5090-validation` | legacy | POC → Cloud | 云 GPU 验证；Draft PR #2 |
| `feat/quest-streaming-poc` | legacy | E1.2 → Q1/Q2 | 早期 POC；Draft PR #1；本机工作树含未提交内容 |
| `feat/quest-streaming-q26-reviewed` | reconcile | 从 POC 独立分叉 | 修复语法并整理 Q2.6；需与 C7.2 按文件差异比对 |
| `feat/quest-q2-evidence-20261013` | evidence | Q2.6 Reviewed → reports | Q1/Q2 旧实测 JSON；非新回归 |
| `feat/e0-text-adapter` | legacy | G0.7 → E0 | 文本编码器基线 |
| `feat/e1-adapter-train` | legacy | E0 → E1 | 模型训练/蒸馏阶段 |
| `feat/e1.1-reeval` | legacy | E1 → E1.1 | 原有模型质量评估分支 |
| `feat/e2-paired-video-eval` | legacy | E1 → E2 | 成对视频评估 |
| `feat/e1.2-adapter-v2` | **research-base** | E2 → E1.2 | E 系列模型适配阶段参考线，不与 Quest 工作混合 |
| `feat/e12-evidence-20261013` | evidence | E1.2 → metadata | Embedding 统计/SOP 归档 |
| `feat/vae-mlx-reverify-20261013` | evidence | E1.2 → partial VAE | MLX/MPS 局部验证；非完整五场景 PASS |
| `feat/g0.5-real-validation` | legacy | 早期 G0.5 | 推理验证历史分支 |
| `bot/m4-fix-20260923` | legacy | M4 工作阶段分叉 | M4 轻量测试与修复；不要混淆为当前主线 |
| `feat/g0.7-eval` | **research-base** | G0.7 基线 | 五场景/45 视频评估基础，不改既有 benchmark |
| `feat/g07-blind-preliminary-20261013` | experiment | G0.7 → preliminary | 45 条盲评初评分；正式结论 NOT VERIFIED |
| `feat/minicpm-v47-world-perception-ab` | experiment | G0.7 → VLM A/B | MiniCPM-V 4.7/4.6 实验，不代表替换模型已批准 |

## 清理原则

- `legacy` 表示不建议继续叠加**新工作**，不是可直接删除。
- `evidence` 不向 `main` 合并原始生成物、模型权重或大文件；对应代码若有价值，独立审查后再摘入集成。
- `experiment` 以可复现实验报告与对照为主，保留基线不被覆盖。
- 原 PR #1–#4 都是 Draft，且逐级叠加；维护前先审阅分支差异与目标基线。禁止一键合并整个栈。
- 当前没有远端 tag；未来若清理旧分支，先创建必要的不可变里程碑，确认没有独有提交及 CI/PR 依赖，并经用户批准。
