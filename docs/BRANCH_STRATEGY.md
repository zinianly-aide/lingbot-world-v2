# Branch Strategy / 分支治理

> 审计日期：2026-10-14。范围：`zinianly-aide/lingbot-world-v2`。此文档描述分支职责，**不代表已合并、已完成云端/Quest 真机回归，亦不授权删除分支**。
>
> 完整分支清单见 [BRANCH_INVENTORY.md](BRANCH_INVENTORY.md)。

## 1. 工作流与唯一集成入口

| 分支 | 定位 | 策略 |
| --- | --- | --- |
| `main` | 已发布/保守稳定基线（现为 `1895d30`） | 不在整理任务中直接更新；经 CI、真机、审查与明确批准后再合并 |
| `integration/quest-streaming` | **Quest / GPU / 连续生成新的集成入口** | 基于验证过的 C7.2 GitHub 头 `1a78ac1`；新开发从此分支拉短生命周期特性分支 |
| `feat/e1.2-adapter-v2` | E 系列模型适配/评测参考基线 | 原样保留，训练、模型修改和评估分开审查 |
| `feat/g0.7-eval` | G0.7 World Perception 评测基线 | 保持不变；VLM A/B 和盲评仅走独立实验分支 |

### 提交约定

1. Quest 新功能以 `integration/quest-streaming` 为目标，通过 `feat/<scope>-<purpose>` 分支提交 PR；不得直接从旧 POC/C6/C7 继续叠加新的 feature。
2. 与 CUDA/VR/GPU 无关的单测须在 PR 中执行；真实 LingBot 生成、Quest 播放、延迟和设备内存只能以**实际测试**声明 PASS。
3. 单独归档 `eval/*` 的 JSON/Markdown 与脚本，不提交可重新生成的 .npy、latents、权重、缓存、日志；保存溯源和运行条件。
4. 禁止未经审查的 `main` merge、`--force` push、reset/clean、历史重写和批量删分支。
5. 新分支最好以 PR 合并（squash/rebase 依审查决定）；分支已存在并不等于代码已进入集成入口。
6. 曾通过 GitHub API 复建的提交 SHA 与本地历史可能不同：**按实际文件差异核对**，勿因 SHA 不同盲目重放。

## 2. Quest 历史链及 PR

```text
main
  └── ... G0.7 → E0 → E1/E2 → E1.2
                           └── feat/quest-streaming-poc (#1, Draft)
                               └── feat/cloud-5090-validation (#2, Draft)
                                   └── feat/c6-continuous-stream (#3, Draft)
                                       └── feat/c7-world-session (#4, Draft)
                                           └── feat/c7.2-playback-telemetry
                                               └── integration/quest-streaming
```

此图表达继承关系，**并非**当前集成/验收状态；#1–#4 当前仍为 Draft。不要直接把全部历史 PR 一次性合入 `main`。

### 需要消除的独立分叉

- `feat/quest-streaming-q26-reviewed`：从 POC 派生，含 Q2.6 修复；**已完成代码级核对，无需整体移植**。详见 [Q26_C72_RECONCILIATION.md](Q26_C72_RECONCILIATION.md)。它不是 C7.2 的祖先，后续只作为历史参考。
- `feat/quest-q2-evidence-20261013`：进一步归档 Q1/Q2 结果，不等于代码已入 C7.2。
- `feat/minicpm-v47-world-perception-ab` 与 `feat/g07-blind-preliminary-20261013`：从 G0.7 分叉，实验用途不同。
- `feat/e12-evidence-20261013` 与 `feat/vae-mlx-reverify-20261013`：从 E1.2 分叉，分别保留元数据与 VAE 复验。

Q2.6 与 C7.2 的十文件比较已完成：六文件内容一致，另四文件在 C7.2 有后续增强；**没有发现需要直接移植的 Q2.6 文件**。不得整体 cherry-pick 或倒退替换 C7.2 的连续推流实现。

## 3. CI 与发布门槛

`integration/quest-streaming` 的 Quest workflow 支持 push 自动测试、PR 测试和手动 workflow_dispatch。Python compile、streaming contract、C6/C7/C7.2、replay 单测必须成功，且 CI 不能被标记为 NOT RUN。C7.2 上的历史 workflow 不因本分支的修改而自动改变。

本地 **49/49 项 CPU/contract/unittest PASS**；集成分支最初提交 `77d1543` 的 [GitHub Actions CI 已成功](https://github.com/zinianly-aide/lingbot-world-v2/actions/runs/38430479444)。本轮对账报告提交是文档更新；云端 LingBot 生成、真机 Quest、长时间吞吐及内存压力仍是 **NOT RUN**，不可写 PASS。

发布或合入 `main` 前至少要求：相关 CI 全绿、review、明确功能兼容性说明、云 GPU 真正连续分块播放实测、Quest 面板/方向/断线恢复实测、回滚方案，以及用户批准。

## 4. 清理顺序（本轮均不删除）

1. 保留全部历史分支和其 Draft PR，先给用途贴上 `active` / `legacy` / `experiment` / `evidence` 分类。
2. 把新增 Quest 工作切到 `integration/quest-streaming`；不要再向早期 C6/C7 等节点直接追加业务代码。
3. Q2.6 与 C7.2 已完成逐文件差异对账（见 reconciliation 报告），确认无需额外 cherry-pick；后续新功能在集成入口开发。
4. G0.7 盲评数据只能标识 preliminary；未经人工/视觉验收不可发布 G1 PASS。
5. 拟删除旧分支前需确认无活动 PR、无独有提交、已有 tag/归档以及远端备份，**再次取得明确授权**。

## 5. 本地工作树保护

主工作树 `feat/quest-streaming-poc` 有未提交实验文件并领先远端 10 个本地提交；另一工作树 `feat/c7.2-playback-telemetry` 也可能有与 GitHub API 重建提交不同的 SHA。不要在这些 worktree 执行 reset/clean/force。

在新的空目录检出集成分支，或先创建新 worktree，避免覆盖现有工作。
