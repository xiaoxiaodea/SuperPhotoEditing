# SuperPhoto

Diagnose a photograph, render distinct Lightroom finishes, learn from your likes and dislikes, and refine the version you choose.

Correction (exposure, white balance, clipping) is separated from look (Portra, cinematic, landscape filmic). Pixel generation is not the editor. Variants are slider recipes you can inspect.

## Run

Double-click **SuperPhoto** on the Desktop (or in `~/Applications`). That starts the local engine and opens a window.

First time, or from the repo:

```bash
cd SuperPhotoEditing
uv sync --extra dev --extra raw --extra semantic
uv run superphoto install-app
open SuperPhoto.app
```

Ollama should be running. The app wakes it and uses `qwen3-vl:32b` for diagnosis. If that model is missing it starts a pull. If Ollama is down it falls back to Grok (when `XAI_API_KEY` is set) then to the local technical critic.

CLI still works:

```bash
uv run superphoto          # same as the app window
uv run superphoto serve    # browser at http://127.0.0.1:8765
```

CLI for a single file:

```bash
uv run superphoto process path/to/photo.jpg
```

Persona 助手不走上面这条命令，也不另起一套引擎。先打开 SuperPhoto，再让助手调用本仓库的技能 `skills/superphoto`。技能只和已经在 `127.0.0.1:8765` 上的服务说话：

```bash
uv run superphoto agent status
uv run superphoto agent edit path/to/photo.jpg
uv run superphoto agent rate <job_id> A like
uv run superphoto agent refine <job_id> A "整体亮一点，保留现在的暖色"
uv run superphoto agent job <job_id>
```

`edit` 和 `refine` 会等到结束，把成片下载到 `~/Library/Application Support/SuperPhoto/agent/<job_id>/`。标准输出是一份 JSON，进度打在标准错误。评价只影响以后相似的照片；微调才会重渲当前这一张。

## Lightroom Classic 自动后期

项目已内置 `vendor/lightroom-mcp`。SuperPhoto 先分析照片的主体、影调、色调和需要保留的现场特征，再规划三套方案。A 自然精修；B 按场景提供银灰质感、冷调电影或森林层次；C 提供暖调胶片，夜景则保留夜色并突出暖光。不同方案具有独立的配色关系，按实际青、绿、橙色占比调整，不仅改变曝光。每套方案使用独立底片副本，经 Lightroom Classic 应用 Develop 参数并导出 JPEG。通过检查的成片进入选择阶段；未通过的方案单独撤下，不会中断其余方案，页面如实显示实际数量。

首次使用：

```bash
uv run superphoto install-lightroom
```

安装或升级插件后，完全退出并重新打开 Lightroom Classic，使新插件生效。之后上传或重试照片时，SuperPhoto 会自动打开 Lightroom、唤醒插件并等待实际连接。已经连接的会话不会被重启。若 Lightroom 停在启动对话框，先在 Lightroom 完成该步骤，再点 **重试这张照片**，无需重新上传。只有自动恢复失败时，才需进入 **文件 → 增效工具管理器 → Lightroom MCP** 检查是否启用并点击 **Start Server**。可检查连接：

```bash
uv run superphoto lightroom-status
```

`POST /api/lightroom/connect` 执行连接恢复；`POST /api/jobs/{job_id}/retry` 用已保存原片新建任务，保留失败记录。连接期间界面单独显示进度，完整错误留在本机记录中。

上传照片后，先从 Lightroom 导出未编辑基线，再用同一套色彩管理结果进行分析。模型为每张照片提出编辑目标、需要保留的现场特征和主体／天空／背景局部调整。没有相似场景反馈时使用上述默认方向；有反馈后按偏好选择三个不同方向，A/B/C 只是这次比较的位置。技术锐化、降噪和几何参数由当前照片决定。

A/B/C 使用 `finishing_knowledge.py` 的场景方法：自然精修、冷调电影、银灰质感、森林层次、暖调胶片、暖色夜景，以及用于偏好探索的柔和淡彩。方法来自用户提供的名人仿色笔记、哈苏／徕卡学习报告，以及已有调色素材的练习图统计和 XMP 配色参考。它们是整理后的经验与可执行起点，没有训练模型，也不把大师成片的整体曝光当作当前照片的目标。

哈苏资料现已完成 **851 张文件的逐图视觉分析**（`ph-main` 464 张、旧收藏 387 张，含不同裁切），另对 57 张进行视觉抽检与校正。每张记录具体优点、作用、可借鉴原则、适用条件、操作与复核要点，以及依赖拍摄的部分。805 张创作图像参与场景检索；介绍肖像、制作场景和拼贴单独保留。查看 [逐图图库](.task-cache/hasselblad-study.html) 和 [接入与验证说明](docs/hasselblad-per-image-study-2026-09-09.md)。

新任务会按题材、场景元素和影调检索最多三张不同作者的参考，条件化指导方案规划、参数保护和成片复核。雾中城市可以借鉴其他题材的雾气层次，但不搬用其配色；黑白参考不会触发自动转黑白。模型提出的操作先经过已校正的规则转换，不能直接执行推测的滑块或不支持的蒙版。`master-study-guidance.json` 记录实际选中的图片及检查点，方便核对资料是否真正参与本次后期。

Ollama 按照片调整这些方法，JSON Schema 开放 HSL、独立曲线、原生局部光线和颜色分级。未提供的控制继承所选方法，显式零值可关闭控制；柔光目标却压暗中间调等矛盾会被修正。规划去重同时考虑曲线、色轮色相、HSL 和局部曝光。`directions-audit.json` 记录知识来源、选择的方法、练习素材提示、模型原文与校验记录。

执行时先以 HSL 和负向饱和度整理颜色，再进行技术修光、原生 AI 蒙版、影调塑形、颜色分级、创意效果和细节输出。影调阶段过强时，先从该阶段之前的准确状态尝试 65%、35% 强度，逐次重新导出检查，全部不合适才撤销。最终配方与审计记录实际保留下来的强度和曲线。

每个阶段都重新导出 Lightroom JPEG，避免缓存预览；像素 QA 检查裁切、饱和度等缺陷，分数不代表审美水平。最终成片由本机视觉模型比较处理前后，原片的实际光色优先于自动生成的保护描述。模型指出损坏后，独立复核交换图片顺序，重新观察原片与成片，不沿用第一轮的指控；只有观察完整且损坏类型一致时才确认。确认后在新副本上减弱引发问题的具体控制，保留该方案的其余设计。视觉模型不可用时明确记录，仅报告技术检查。

实验性成片精修可以在技术通过后继续检查光影、颜色或质感，最多进行两轮。模型先单独观察当前成片，再对已有配方提出小步调整；不能新建没有依据的蒙版。每轮都从原片独立渲染，技术检查通过后，再与保留版本直接比较；只有具体区域的改善得到明确支持才替换。无改善、响应不完整或执行失败均保留此前有效版本。当前实测中，本地审片模型仍漏判关键不足，故此功能默认关闭；`SUPERPHOTO_FINISHING_REVIEW_ENABLED=true` 可用于研究，`SUPERPHOTO_FINISHING_MAX_ROUNDS=0..2` 控制上限，`finishing_review` 保存完整判断记录。

相对白平衡调整必须取得真实色温和色调基值。RAW 的 As Shot 若未提供这两个数值，先保留现场白平衡并记入审计，不用默认色温覆盖原片。插件只在目标确实为当前 Develop 照片时补读真实值，避免借用其他照片的数据。

没有生成可用 AI 蒙版且 Lightroom 确认没有新增蒙版时，保留其他成功的局部调整，记录跳过的目标并移除未执行的配方。单个方案仍失败时仅撤下该方案，继续导出其他结果；全部方案均不合格时仍报告失败。详见 [视觉复核误判与交付修复](docs/visual-review-failure-2026-09-06.md)。

色相筛选出的蓝青、暖橙区域只产生颜色变化提示，不会被当成已确认的天空或肤色损坏，也不参与画质扣分和自动纠色。可用且置信度足够的本地语义分割会限定天空和人物区域；全图过饱和、裁切和极端亮度检查持续生效。详见 [资料接入与验证记录](docs/study-guided-finishing-2026-09-06.md)。

候选差异默认以平均 Lab 距离 5.0 为选择目标。画质通过但相近的候选，会沿所选配色最多尝试三个独立版本，保留已经通过检查的曝光、曲线和蒙版光比。低调白天照片不再因亮度偏低而归入夜景；暗部裁切较多时，B 也不再重复压暗。没有场景记录的旧配方重放兼容原有明暗调整。如果仍相近，交付保留的最佳有效版本并注明差别细微，继续完成其余候选。技术损坏和经确认的视觉损坏仍会拦截。`candidate_selection` 记录实际距离、是否达到目标和被选中的轮次；像素、配方和 Lightroom 底片始终对应同一有效轮次。兼容配置 `SUPERPHOTO_LIGHTROOM_REQUIRE_DISTINCT_CANDIDATES=false` 可关闭去重重试，相似度在任何配置下都不再导致整单失败。

Lightroom 成片与审计分别写入作业的 `lightroom-exports/` 和 `lightroom-audits/`，汇总写入 `lightroom-qa.json`；记录编辑目标、每轮配方、实际参数和蒙版 ID、阶段导出、视觉复核与回退、像素 QA、最终接受结果。插件使用 SDK 原生主体／天空／背景蒙版，以及原生 Color Grading 阴影、中间调和高光色轮（含亮度、混合、平衡），色轮参数必须从 Lightroom 回读确认。升级工程后需重新安装并加载随附插件。精细范围蒙版、手工修复、AI 降噪、Calibration 与 Point Color 尚未自动完成；这些开发范围说明不再作为每张照片的固定提示。视觉复核可用 `SUPERPHOTO_VISUAL_REVIEW_ENABLED=false` 关闭。

## 选图反馈与二次微调

每张成片下方都有 **满意 / 不满意**，可以同时评价多张，再次点击同一选项即可撤销；没有评价的图片不会被当成不满意。反馈自动保存在本机，刷新或重启后仍在。

后续处理相似画面时，系统根据主体场景、亮度、色温和颜色占比，降低不喜欢的调色方法的选择权重，提高喜欢的方法的权重。在兼容该场景的方法中选择三个不同方向，保留探索机会，不把一次不满意变成永久禁用。学习的是实际方法标识，不是 A/B/C 位置或旧 LUT 名称。重复评价覆盖原记录；撤销后旧反馈不会重新生效。`preference-selection.json` 保存场景、匹配反馈、各方法权重及本次选择；权重不是最终入选概率。该反馈影响后续推荐，不会改写已完成的成片。

点击成片预览后，在下方 **想改哪里？** 输入意见，再点 **让 Lightroom 微调这张**。例如“整体亮一点，保留现在的暖色”或“主体亮一点，背景暗一点”。明确指令直接转换为有界参数变化；其他描述由本地视觉模型结合当前成片解析，执行器限制修改范围。无法理解或涉及不支持的局部对象时，会保留成片并说明原因。

微调以实际接受的 Lightroom 配方为基础，在独立原片副本上应用增量、重新导出并检查。完成后显示微调前后对照；通过 **查看版本** 可切回最初成片、下载任一版本，或基于某次微调继续调整。处理进度、输入意见和版本历史均持久保存；刷新页面会恢复进度。失败不会影响原有三张成片。满意度按钮对应上方卡片中的最初成片，微调意见不自动当成好评或差评。

接口：`PUT /api/jobs/{id}/ratings/{slot}`（`verdict: like / dislike / null`）；`POST /api/jobs/{id}/refinements`（`slot`、`prompt`、UUID `request_id`、可选 `parent_id`）；`GET /api/jobs/{id}/refinements/{refinement_id}` 查询进度。重复请求 ID 返回同一任务，同张图已有微调时拒绝重复排队。普通出片与微调共享 Lightroom 工作流锁，避免选图、蒙版和参数互相干扰。完整审计保存于 `refinements/{id}/audit.json`。

## What is local vs API

| Piece | Where |
|---|---|
| Decode, EXIF, histogram, clipping, render | Local |
| Recipe library + preference counts | Local (`data/`) |
| Photographic critique | SpaceXAI `grok-4.6` if `XAI_API_KEY` is set, otherwise local |
| RAW | Optional `rawpy` |
| darktable-cli | Detected if installed; not required for v0.1 |

RAW files stay on disk. The cloud critic only receives a JPEG preview.
