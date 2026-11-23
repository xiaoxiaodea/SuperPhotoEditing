# 视觉定向修图（2026-11-16）

## 为什么改

同日的定向复验（`.task-cache/directed-probe-2026-11-16/README.md`）用同一个 LightCraft 引擎，由看过基线的人手写全局参数和 3–5 个线性 / 径向蒙版。6 张照片中定向版有 5 张最好或并列最好，原有流程的最佳候选没有一张超过定向版。差距主要来自局部蒙版：原配方模型只能表达主体 / 天空 / 背景三种目标，且同一目标会被合并成一个；15 个候选的 `masks` 全为空。几何蒙版只需要坐标和羽化，不需要分割模型，视觉模型可以直接给出。

## 流程（`src/superphoto/directed.py`）

1. **基线**：沿用 `pipeline._prepare_lightroom_baseline` 导出的未编辑 LightCraft 渲染，以及同一结果上的技术分析。
2. **规划**：一次视觉请求，发送基线和带 0.1 坐标网格的副本。模型返回读图（场景、主体框、问题按严重度排序、需保留的特征）以及 A / B / C 三套完整方案。提示词写明蒙版坐标约定（径向 `rx` 按图宽、`ry` 按图高的比例）、常用幅度，以及此前实测的教训：亮场景不整体压灰、暗场景局部提亮并保护灯光、雾是气氛、避免青绿偏色、JPEG 亮部没有可恢复的细节。
3. **清洗与编译**：`sanitize_plan` 只保留支持的控件，数值裁到安全范围，丢弃未知键，色轮色相按 360 取模，最多 6 个蒙版，退化形状（起止点重合、半径为 0）直接丢弃。所有改动写入备注。`compile_document` 生成 LightCraft 显影 JSON。JPEG 的全局色温按 6500K / 0 的恒等基值偏移；RAW 拿不到拍摄色温，因此不执行全局色温偏移。
4. **渲染与损坏检查**：每个版本在原片副本上渲染，三个版本并行。`safety_gate` 复用 `lightroom_qa.assess_lightroom_render`，但 `too_dark`、`too_bright`、`excessive_distance`、`exposure_unresolved`、`exposure_lift_unresolved` 只作为提示。另设线性亮度比的硬边界 0.3–3.5，约 −1.75 至 +1.8 EV，只拦截灾难性结果。
5. **复核与修订**：发送「基线 | 成片」对照板和带网格的成片，以及方案和检查结果。模型逐项判断每个蒙版的目的是达成、部分达成、未达成还是过度，列出损坏，并可返回完整修订方案，方向不限：加强、减弱、移动或缩放蒙版。默认修订 1 轮（`SUPERPHOTO_DIRECTED_MAX_REVISIONS`）。
6. **选择与交付**：满足以下全部条件的轮次才有效：通过损坏检查、复核没有以置信度 ≥0.7 判为变差、没有严重度 ≥0.65 的损坏。在有效轮次中按「判定 + 达成比例 − 损坏」取最高分，同分时取较新的一轮。没有有效轮次的版本单独撤下；三个版本都被撤下时抛出 `DirectedUnavailable`。

`pipeline.run_job` 在写完预览后进入此流程。`auto` 模式在没有视觉模型、全部撤下或出现意外错误时回到原有流程，原因写入 `directed-fallback.json`（意外错误的堆栈另存为 `directed-error.txt`）；`directed` 模式直接报错。

## 产物

- `directed/plan.json`：读图、三套方案、清洗备注和请求审计。
- `directed/<槽位>/round-N/`：`develop.json`、`render.jpg`、`review.json`。
- `variant_X.jpg/.json`：成片和 sidecar（`engine: lightcraft-directed`）。sidecar 含方案、显影 JSON、像素检查、视觉复核和 `style_workflow.adaptation_notes`（结果页「按本图调整」列表）。
- `lightroom-audits/variant_X.json`：带 engine 标记的交付或撤下记录，文字微调从这里读取方案。
- `lightroom-qa.json`：`delivery`（交付数、撤下原因、相近槽位）。

## 文字微调

`refinement.refinement_base` 先检查成片是否由定向流程生成。如果是，就把当前成片（带网格）和原方案交给视觉模型，只修改用户要求的部分，然后在原片副本上重渲，并做损坏检查和可见变化检查。之后的追加微调基于上一版修订后的方案。视觉模型不可用、回答无法解析或方案为空时，保留调整前的版本并说明原因。

## 验证

- `tests/test_directed.py` 共 16 项：清洗、编译、JPEG 与 RAW 的白平衡、检查分级、脚本化视觉模型下的端到端（含修订、撤下、复核拦截、非 JSON 重试）、流水线回退和定向微调。`tests/test_cli.py` 新增 `vision-check` 测试。`tests/test_vision.py` 覆盖 `auto` 顺序（Grok 最后）和临时标记。全套测试结果：629 项通过，1 项跳过。
- 真实 LightCraft 离线验证：`.task-cache/directed-integration-2026-11-16/run.py`。复验的 5 张照片按完整流程运行，视觉模型的回答换成复验时手写的方案。结果如下：
  - 编译出的显影 JSON 与复验文件逐键一致。
  - 渲染与复验成片的平均绝对差为 0.64–1.62/255，与两次基线渲染之间的差同一量级（0.55–1.65）。
  - 15 张渲染全部通过损坏检查。此前被旧检查拒绝的 DSC_1276（`too_bright`）和 DSC_1441（`too_dark`、`excessive_distance`）现在只带提示。DSC_1441 的线性亮度比是 0.459，正因如此，硬边界从 0.45 放宽到 0.3。
  - 每张约 10–15 秒（1600px，三版并行，不含视觉请求）。

## 尚未验证与限制

- **还没有用真实视觉模型跑过**：本机没有密钥。把 `XAI_API_KEY` 写入 `.env` 后，先运行 `uv run superphoto vision-check --list-models`，再运行 `.task-cache/directed-grok-benchmark/run.py`（`directed` 模式，5 张复验照片，输出「基线 | 手写定向 | A | B | C」对照图）。模型能否准确放置蒙版，是这一流程最大的未知数。
- Grok（xAI）只是临时替代。`auto` 的顺序为 Ollama → OpenAI-compatible → xAI，所以本机 Ollama 或配置好的其他视觉模型一旦可用，就会自动取代 Grok，不需要改代码。使用 Grok 时，`directed/plan.json`、`lightroom-qa.json`、`workflow.json` 和 sidecar 的 `directed` 字段都会写入 `temporary_provider: true`，`vision-check` 的报告里也会带上提示。如需固定使用 Grok，设置 `SUPERPHOTO_VISION_PROVIDER=xai`。
- 未使用 LightCraft 的亮度范围蒙版（需要 `lo_feather` 等字段），也未使用画笔；几何蒙版的羽化可以掩盖少量位置误差。
- RAW 的全局色温不执行偏移；所有复验照片都是 JPEG，RAW 尚未测试。
- 差异目标仍沿用平均 Lab 距离 5.0。B 与 A 意图相同，可能被标为“差别较细微”，这只是提示，不会撤下。
- 定向成片的满意 / 不满意评价目前没有方法标识（`method_id` 为空），还不参与相似场景的偏好学习。
