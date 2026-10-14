# Q2.6 Reviewed → C7.2 / Integration 代码对账

> 审计：2026-10-14 · 范围：Quest Q1/Q2 至 C7.2 的流式功能。
> 这是历史分支的**文件级差异结论**，不是新的云端 GPU 或 Quest 真机验收。
>
> References:
> - Q2.6 Reviewed：`feat/quest-streaming-q26-reviewed`，commit `296bb898`
> - C7.2：`feat/c7.2-playback-telemetry`，commit `1a78ac1`
> - Integration：`integration/quest-streaming`，首次治理 commit `77d1543`

## 结论

**需要迁入集成入口的 Q2.6 源码文件：0。**

两个分支在 POC 基线上分叉。Q2.6 的一批逻辑并未丢失：六个文件的 Git Blob 内容与 C7.2 相同，余下四个文件由 C6/C7/C7.2 继续增强。Q2.6 Reviewed 曾修复的三行 Python 缩进在 C7.2 中已正确；C7.2 的 CI 编译步骤能解析 Q2 脚本。直接 cherry-pick Q2.6 会与较新实现冲突，并可能退回早期行为。

## 完全一致的 6 个文件

Git Blob SHA 在本机 C7.2 与 Q2.6 原始提交中匹配；GitHub Reviewed 版保留这些文件内容。

| 文件 | 用途 |
| --- | --- |
| `scripts/q1_5_validate_progressive_vae.py` | Q1.5 Progressive VAE 验证 |
| `scripts/q1_validate_latent_tap.py` | Q1 latent tap 验证 |
| `scripts/quest_frame_bridge.py` | Bridge 启动与路径处理 |
| `scripts/quest_stream_replay.py` | 回放 loop、全局递增序列 |
| `tests/test_replay_loop.py` | Replay monotonic sequence 回归 |
| `wan/streaming/vae_progressive.py` | Progressive VAE 解码组件 |

## C7.2 已继续演进的 4 个文件

| 文件 | C7.2 相比 Q2.6 的后续变化 | 结论 |
| --- | --- | --- |
| `scripts/q2_live_quest_poc.py` | 连续发布、外部 Bridge、全局序列偏移、可调 output FPS / KV rolling window、更多报告指标；缩进已正确 | 保留 C7.2 |
| `wan/image2video.py` | 延续 C6/C7 模型生成状态与 `generate-latents` 阶段生命周期处理 | 保留 C7.2 |
| `wan/streaming/buffered_publisher.py` | C7.2 playback policy、延迟与丢帧指标、队列异常处理 | 保留 C7.2 |
| `wan/streaming/frame_publisher_async.py` | 在早期 async publisher 上继续扩展以支持当前连续播放能力 | 保留 C7.2 |

**注意：** “C7.2 已演进”不等于云端端到端性能更好；真实 FPS、TTFF、内存和连续播放体验仍需基准实测。

## 验证证据

- 本机 C7.2 / integration 同源代码执行 `tests.test_m36_staged`、streaming/Quest/C6/C7/C7.2/replay 合计 **49/49 PASS**。
- [集成分支 CI 成功运行 #38430479444](https://github.com/zinianly-aide/lingbot-world-v2/actions/runs/38430479444)，含 Python 编译及 streaming tests，结论 **success**。
- CUDA 4090/5090 当前实测：**NOT RUN**。
- Quest 当前真机及长时间连续播放：**NOT RUN**。
- 旧 Q2.6 报告是历史快照，不能用来宣称本次模型/VR 验收通过。

## 后续实施规则

1. 将 `feat/quest-streaming-q26-reviewed` 标记为 **legacy / reconciled**，不再作为新代码开发入口。
2. `feat/quest-q2-evidence-20261013` 继续只存旧运行证据。
3. 新的修复从 `integration/quest-streaming` 派生 feature 分支，并带 CPU 单测和现场验证记录；不整体合并 Q2.6。
4. 在 GPU 真机复测中若找到可重现的退化，提交精确的修复 PR，注明对应指标和测试；不得凭历史分支差异直接宣称回归。
5. 不修改 `main`，不删除历史分支、不关闭 Draft PR、不 reset/clean 本地脏工作树。
