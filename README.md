# SuperPhoto

Diagnose a photograph, render distinct LightCraft finishes, learn from your likes and dislikes, and refine the version you choose.

Correction (exposure, white balance, clipping) is separated from look (Portra, cinematic, landscape filmic). Pixel generation is not the editor. Variants are slider recipes you can inspect.

## Windows / NAS / 云端视觉 API

推荐 Python 3.12（仓库已用 `.python-version` 固定），先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。Windows 可双击 `Start-SuperPhoto.cmd` 安装依赖并打开窗口；需要系统 WebView2 Runtime，也可以直接使用浏览器模式：

```powershell
uv sync --python 3.12 --extra raw --extra dev
uv run superphoto install-lightcraft
uv run superphoto lightcraft-status
uv run superphoto serve
# 浏览器打开 http://127.0.0.1:8765
```

LightCraft 安装器按系统和架构选择官方包，核对 SHA256，只提取 CLI，运行 `--version` 成功后原子替换旧版本。Windows 使用 `lightcraft-cli.exe`，支持 x64/ARM64；Linux 使用 x86_64/aarch64 tar 包；macOS 使用 universal CLI。源码支持不等于已在你的 NAS 上实测，Linux 还需匹配系统运行库。

将 `.env.example` 复制为 `.env` 配置。使用云端视觉时至少设置：

```dotenv
SUPERPHOTO_VISION_PROVIDER=openai
SUPERPHOTO_OPENAI_API_KEY=你的密钥
SUPERPHOTO_OPENAI_BASE_URL=https://你的服务/v1
SUPERPHOTO_OPENAI_MODEL=服务支持的视觉模型名称
SUPERPHOTO_OPENAI_PROTOCOL=chat/completions
SUPERPHOTO_SEMANTIC_ENABLED=0
```

协议可选 `chat/completions` 或 `responses`，按供应商实际支持选择；OpenAI-compatible 不代表两者都实现。**推荐在客户端里配置视觉模型**：点顶栏「视觉模型」，设置保险箱口令后可以添加多个模型（OpenAI、xAI Grok、Gemini、Claude、通义千问、豆包、GLM、OpenRouter、本机 Ollama 或 LM Studio 等任意 OpenAI 兼容服务），并按顺序排优先级，前一个失败时自动换下一个。密钥、服务地址和模型名用口令加密保存在 `data/vision-vault.json`，只在客户端运行时解锁到内存；日志、任务记录、接口和命令行都只看到「视觉模型 1、2…」。详见 `docs/vision-vault-2026-11-16.md`。下面的 `.env` 方式仍然可用，但它是明文。`xai` 使用原有 `XAI_API_KEY` / `XAI_MODEL`，**Grok 只是没有其他视觉模型时的临时替代**。`auto` 的顺序是 Ollama → OpenAI-compatible → xAI → 本地，因此本机 Ollama 或配置好的 OpenAI-compatible 视觉模型一旦可用就会自动取代 Grok；使用 Grok 时，审计会标记 `temporary_provider`，结果页显示「临时替代」。显式云端不会唤醒、下载或探测 Ollama。诊断、方案规划、成片复核、实验性精修、版本比较和复杂文字微调均走同一视觉接口。审计保留实际 provider、model、可用用量及降级情况；健康页的“已配置”不表示已验证网络、额度或模型权限。

NAS 上运行服务时不需要桌面窗口。每个 `SUPERPHOTO_DATA_DIR` 只启动一个服务进程，SQLite、worker 锁和作业目录放在运行主机的本地文件系统；Windows 通过 SMB 访问 NAS 时只把共享作为源片目录，不将 SQLite 放到 SMB。服务运行在 NAS 本机时，可使用其本地挂载卷。局域网开放设置 `SUPERPHOTO_HOST=0.0.0.0` 和长随机 `SUPERPHOTO_ACCESS_TOKEN`；网页登录使用 HttpOnly cookie，程序使用 Bearer token。公网访问仍应经过 HTTPS 反向代理。

服务使用持久化单 worker：未开始的 queued 任务重启后继续；处理中被强制终止的任务明确标记中断，重试创建新目录，保留原片及旧记录。正常关闭会等待当前任务完成。同一目录的第二个服务或独立 `process` 命令会被进程锁拒绝。

目录自动导入需要服务先启动，导入端保持运行：

```powershell
uv run superphoto import-folder "D:/Photos/Incoming" --watch
# NAS 源片示例：uv run superphoto import-folder "//nas/photos/incoming" --watch
# 一次扫描：省略 --watch；--url 可指定远端服务。
```

文件大小和修改时间稳定 5 秒后才提交；输入原片不会修改；按内容哈希记录成功上传，重启或重命名后不会重复导入已记录的内容；作业输出目录不参与导入。传输超时可能发生“服务已收到、客户端未收到确认”，此时重试仍可能重复提交，尚未提供跨网络的严格 exactly-once 保证。失败任务可以在页面重试。

没有 LightCraft 时可用 `SUPERPHOTO_LIGHTROOM_ENABLED=false` 运行基础 NumPy/Pillow 渲染；默认输出长边 1600px JPEG，控制覆盖有限，不支持文字微调。完整 LightCraft 导出默认长边 2400px，可配置 `SUPERPHOTO_LIGHTROOM_EXPORT_EDGE`。未实现完整的原尺寸高位深导出流程。相对白平衡没有真实原片基值时保留 As Shot，不从固定 6500K 推算。

本次 Windows 验证与后续复验方法见 [验证记录](docs/windows-validation.md)。

## macOS Run

Double-click **SuperPhoto** on the Desktop (or in `~/Applications`). That starts the local engine and opens a window.

First time, or from the repo:

```bash
cd SuperPhotoEditing
uv sync --extra dev --extra raw --extra semantic
uv run superphoto install-app
open SuperPhoto.app
```

In auto/Ollama mode the app can wake Ollama and use an installed visual model (default `qwen3-vl:32b`). Model downloads are explicit; startup no longer silently pulls large weights. Auto falls back to configured cloud providers, then the local technical critic. Explicit cloud/local mode does not wake Ollama.

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

`edit` 和 `refine` 会等到结束，把成片下载到平台用户目录下的 `SuperPhoto/agent/<job_id>/`（Windows：`%LOCALAPPDATA%/SuperPhoto/agent/`；macOS：`~/Library/Application Support/SuperPhoto/agent/`；Linux：`$XDG_DATA_HOME/superphoto/agent/` 或 `~/.local/share/superphoto/agent/`）。标准输出是一份 JSON，进度打在标准错误。评价只影响以后相似的照片；微调才会重渲当前这一张。

## LightCraft 自动后期

出片由 [LightCraft](https://github.com/storytold/lightcraft) v0.21.1 的 `lightcraft-cli` 完成。SuperPhoto 先分析主体、现场光线和需要保留的特征，按场景筛选可执行预设，再结合相似照片的反馈及方案差异选出最多三套方向。A/B/C 是本次比较位置，具体风格随照片变化。每套方案在共享的技术校正基础上独立适配，从原片副本写入显影 JSON 并导出 JPEG。过曝 JPEG 的基础曲线与预设曲线按顺序组合，保护曝光时保留预设的影调和配色；风格减弱回到已校正的基础，而不是未编辑原片。真实导出仍过于相似时，先调整当前方向，再从本次场景准入名单中改选其他预设，所有尝试都继续通过画质检查。实际交付的预设与反馈 ID 同步更新。通过检查的成片进入选择阶段；未通过的方案单独撤下，差异不足的有效成片明确提示接近，不把文件数量当成独立风格数量。流程与实测见 [风格流程优化记录](docs/style-workflow-2026-11-14.md)。

旅行素材实测还修正了本地暖色岩石被误判成人像、未裁切的极亮有色画面漏做曝光校正，以及稀疏曲线重试没有实际变化的问题。暖色面积和焦距不再作为人像确认；极暗且亮部余量充分的 JPEG 可按诊断小幅提亮，普通低调照片仍受原限制。照片、实际交付与局限见 [旅行素材流程验证](docs/travel-workflow-2026-11-14.md)。

自然颜色模式会检查实际导出是否给原片近中性亮部新增明显偏色；发现后从原片副本减弱通道调色，保留有效曝光、总影调曲线与局部光比，并重新检查。明确的创意配色不触发这项保护；有裁切或透视变化时跳过未对齐的固定位置颜色比较。候选变自然后可能更接近，仍如实标记。实片结果与边界见 [自然颜色质量优化](docs/natural-color-quality-2026-11-14.md)。

首次使用：

```bash
uv run superphoto install-lightcraft
uv run superphoto lightcraft-status
```

安装器从官方固定版本选择目标平台的命令行，放在 `vendor/lightcraft/lightcraft-cli`（Windows 为 `lightcraft-cli.exe`，不进 git）。也可以自己把 `lightcraft-cli` 放到 `PATH`，或用 `SUPERPHOTO_LIGHTCRAFT_BIN` 指定。升级前先关闭正在运行的服务，避免 Windows 锁定可执行文件。`POST /api/lightroom/connect` 现在只检查这条命令行；`POST /api/jobs/{job_id}/retry` 用已保存原片新建任务，保留失败记录。

滑块单位仍与原来的 Lightroom 配方相同。LightCraft 的色彩科学和主体／天空蒙版（传统启发式，不是 Lightroom 的 AI 蒙版）不会和 Classic 逐像素一致。

上传照片后，先用 LightCraft 导出未编辑基线，再用同一套色彩管理结果进行分析。视觉模型可提出编辑目标、现场特征和主体／天空／背景局部调整；不可用时使用本地技术分析。风格先经过场景准入，再参考适配程度、相似场景反馈和配方差异排序。技术锐化、降噪和几何参数由当前照片决定。

常规候选来自已导入 XMP、内建配方、名人风格研究和本地素材整理出的自主配方。`finishing_knowledge.py` 与逐图研究提供条件化方法参考，模型对选中预设进行稀疏调整：缺省控制继承预设，显式零值关闭对应控制。资料是经验和可执行起点，没有训练模型，也不把示范成片的整体曝光当作当前照片的目标。

`resources` 已整理为 11 条带文件哈希与视频时间点的原则，参与规划、参数保护和成片复核；新增柔灰哑光、雪景清透、夜色光序三个保守起点。雪景需要明确雪景主体，夜景需要场景证据，低亮度或白色像素本身不能作为新风格的准入依据。有明确场景时，保护规则限制雾景过强去雾、人像全局纹理和雪面曲线压灰，基础曝光恢复仍独立处理。本地技术分析无法可靠识别雪景、雾气等对象时，这些语义规则不会自动触发；接入视觉模型后可使用其诊断。原始商业预设保留在本地，没有批量注册或随应用分发。每次任务的 `resource-guidance.json` 保存适用原则、来源与知识版本，`directions-audit.json` 保存实际限制的参数。详见 [素材整理与验证](docs/resource-study-2026-11-13.md)。

哈苏资料现已完成 **851 张文件的逐图视觉分析**（`ph-main` 464 张、旧收藏 387 张，含不同裁切），另对 57 张进行视觉抽检与校正。每张记录具体优点、作用、可借鉴原则、适用条件、操作与复核要点，以及依赖拍摄的部分。805 张创作图像参与场景检索；介绍肖像、制作场景和拼贴单独保留。查看 [逐图图库](.task-cache/hasselblad-study.html) 和 [接入与验证说明](docs/hasselblad-per-image-study-2026-09-09.md)。

新任务会按题材、场景元素和影调检索最多三张不同作者的参考，条件化指导方案规划、参数保护和成片复核。雾中城市可以借鉴其他题材的雾气层次，但不搬用其配色；黑白参考不会触发自动转黑白。模型提出的操作先经过已校正的规则转换，不能直接执行推测的滑块或不支持的蒙版。`master-study-guidance.json` 记录实际选中的图片及检查点，方便核对资料是否真正参与本次后期。

已配置的视觉模型按照片适配所选预设，JSON Schema 开放 HSL、独立曲线、原生局部光线和颜色分级。未提供的控制继承预设，显式零值可关闭控制；柔光目标却压暗中间调等矛盾会被修正。规划去重同时考虑曲线、色轮色相、HSL 和局部曝光。`directions-audit.json` 记录知识来源、所选预设、练习素材提示、模型原文与校验记录。

执行时先以 HSL 和负向饱和度整理颜色，再进行技术修光、启发式主体／天空／背景蒙版、影调塑形、颜色分级、创意效果和细节输出。影调阶段过强时，先从该阶段之前的准确状态尝试 65%、35% 强度，逐次重新导出检查，全部不合适才撤销。最终配方与审计记录实际保留下来的强度和曲线。

每个阶段都重新用 LightCraft 导出 JPEG；像素 QA 检查裁切、饱和度等缺陷，分数不代表审美水平。最终成片由本机视觉模型比较处理前后，原片的实际光色优先于自动生成的保护描述。模型指出损坏后，独立复核交换图片顺序，重新观察原片与成片，不沿用第一轮的指控；只有观察完整且损坏类型一致时才确认。确认后在新副本上减弱引发问题的具体控制，保留该方案的其余设计。视觉模型不可用时明确记录，仅报告技术检查。

实验性成片精修可以在技术通过后继续检查光影、颜色或质感，最多进行两轮。模型先单独观察当前成片，再对已有配方提出小步调整；不能新建没有依据的蒙版。每轮都从原片独立渲染，技术检查通过后，再与保留版本直接比较；只有具体区域的改善得到明确支持才替换。无改善、响应不完整或执行失败均保留此前有效版本。当前实测中，本地审片模型仍漏判关键不足，故此功能默认关闭；`SUPERPHOTO_FINISHING_REVIEW_ENABLED=true` 可用于研究，`SUPERPHOTO_FINISHING_MAX_ROUNDS=0..2` 控制上限，`finishing_review` 保存完整判断记录。

相对白平衡调整必须取得真实色温和色调基值。RAW 的 As Shot 若未提供这两个数值，先保留现场白平衡并记入审计，不用默认色温覆盖原片。只有配方明确要求自定义白平衡时，才会把色温和色调写进 LightCraft。

蒙版写入 LightCraft 的主体、天空或背景选择。若执行器收到「没有生成蒙版」的结果，会保留其他成功的局部调整，记录跳过的目标并移除未执行的配方。单个方案仍失败时仅撤下该方案，继续导出其他结果；全部方案均不合格时仍报告失败。详见 [视觉复核误判与交付修复](docs/visual-review-failure-2026-09-06.md)。

色相筛选出的蓝青、暖橙区域只产生颜色变化提示，不会被当成已确认的天空或肤色损坏，也不参与画质扣分和自动纠色。可用且置信度足够的本地语义分割会限定天空和人物区域；全图过饱和、裁切和极端亮度检查持续生效。详见 [资料接入与验证记录](docs/study-guided-finishing-2026-09-06.md)。

候选差异默认以平均 Lab 距离 5.0 为选择目标。画质通过但相近的候选，会沿所选配色最多尝试三个独立版本，保留已经通过检查的曝光、曲线和蒙版光比。低调白天照片不再因亮度偏低而归入夜景；暗部裁切较多时，B 也不再重复压暗。没有场景记录的旧配方重放兼容原有明暗调整。如果仍相近，交付保留的最佳有效版本并注明差别细微，继续完成其余候选。技术损坏和经确认的视觉损坏仍会拦截。`candidate_selection` 记录实际距离、是否达到目标和被选中的轮次；像素、配方和 Lightroom 底片始终对应同一有效轮次。兼容配置 `SUPERPHOTO_LIGHTROOM_REQUIRE_DISTINCT_CANDIDATES=false` 可关闭去重重试，相似度在任何配置下都不再导致整单失败。

成片与审计仍写入作业的 `lightroom-exports/` 和 `lightroom-audits/`，汇总写入 `lightroom-qa.json`；记录编辑目标、每轮配方、实际参数和蒙版 ID、阶段导出、视觉复核与回退、像素 QA、最终接受结果。显影 JSON 使用 LightCraft 的主体／天空／背景蒙版，以及阴影、中间调和高光色轮（含亮度、混合、平衡），色轮参数必须从本次渲染回读确认。精细范围蒙版、手工修复、AI 降噪、Calibration 与 Point Color 尚未自动完成；这些开发范围说明不再作为每张照片的固定提示。视觉复核可用 `SUPERPHOTO_VISUAL_REVIEW_ENABLED=false` 关闭。

## 视觉定向修图（默认，需要视觉模型）

有可用视觉模型（客户端「视觉模型」里配置的任一服务，或 Ollama / 已配置密钥的 xAI / OpenAI-compatible）且 LightCraft 可用时，新任务改走视觉定向流程：模型看未编辑的 LightCraft 基线和带 0.1 坐标网格的副本，一次写出 A（自然完成）、B（同一意图更有力度）、C（在 A 的光影上调色）三套完整方案。

方案写的是**成片要达到的数值**，而不只是滑块：整帧的黑位、暗部、中间调、亮部、白位和整体鲜艳度，以及每个蒙版目标框里的亮度中位数、色相和鲜艳度。蒙版可以是线性渐变、径向、亮度范围或颜色范围（按归一化坐标放置，每套 2–6 个，例如压天空、提主体、保护灯光、分离远景）。求解器用小尺寸试渲染反复测量，解出曲线、曝光、饱和度和各蒙版的参数，影调收尾后再校一次颜色（曲线重拟合会改变饱和颜色的色相）。解不到的目标会连同原因交还给模型：蒙版没有覆盖目标框、超出安全范围、与整帧影调目标冲突，或色相需要用全局色温 / 色调和分级色轮来转（提示里写明方向和色轮）。

每套方案先限制在固定安全范围内再渲染；像素检查只拦截损坏（裁切、过饱和、A 的中性偏色、极端亮度），偏亮偏暗和改动幅度作为提示交给复核。随后模型对照「基线 | 成片」和测量结果逐项检查每个蒙版的目的是否达成，可返回完整修订方案，默认最多修订 2 轮，取最好的有效轮次交付（不一定是最后一轮）。

`SUPERPHOTO_PIPELINE_MODE=auto`（默认）在没有视觉模型或三套都未通过时回到原有流程，原因写入 `directed-fallback.json`；`directed` 不回退、直接报错；`legacy` 只用原有流程。`SUPERPHOTO_DIRECTED_MAX_REVISIONS=0..3` 控制修订轮数（默认 2），`SUPERPHOTO_DIRECTED_VISION_TIMEOUT` 控制单次请求时限（默认 240 秒）。先用 `uv run superphoto vision-check --list-models` 确认模型可用：它发送一张小图，报告实际模型、延迟和是否看懂图片，不输出密钥。JPEG 的全局色温按 LightCraft 的 6500K 恒等基值偏移；RAW 的全局色温和色调按 LightCraft 读出的拍摄值偏移（`lightcraft-cli run --import <file> develop.wb`）。

RAW 在 LightCraft 里渲染出来是一张偏暗偏灰的底片（比 ACR 暗约 0.7–1.2 EV），方案按「把底片冲成成片」来写：允许的影调和鲜艳度变化比已完成的 JPEG 大，雪景的冷蓝按偏色处理，天空保持浅淡。RAW 的蒙版只管明暗，颜色只做小幅微调（色温 ±24、色调 ±10，饱和度可降到 -70），色相交给全局色温 / 色调、校准和分级；参考库里摄影师 19 张 RAW 修图的 76 个蒙版都没有调颜色。

**参考成片对标**：`resources/` 是本地的修图参考库（预设、带修图记录的 DNG、修图前后对照和教程视频），不进仓库。定向流程以其中 7 张有修图前后对照的 RAW 为基准调优：把 A/B/C 和摄影师的成片裁到同一范围，按 16×20 网格比较平均 Lab 差（越小越接近）。未修图约 25.1，目前最好一轮 A 9.90、B 11.65、C 10.60（不是同一轮）。最新一组改动（画面底部按内容处理、RAW 蒙版颜色限幅、收尾补校颜色、色相方向提示）已通过单元测试和离线复现，尚未在完整一轮中验证。对标工具和记录在本地 `.task-cache/`，不进仓库。

文字微调对这类成片同样由视觉模型修改原方案（只改用户要求的部分），在原片副本上重渲并检查。过程与产物在作业的 `directed/`（方案、每轮显影 JSON、渲染与复核）和 `lightroom-audits/`。设计依据与离线验证见 [视觉定向修图](docs/directed-finishing-2026-11-16.md)。

## 选图反馈与二次微调

每张成片下方都有 **满意 / 不满意**，可以同时评价多张，再次点击同一选项即可撤销；没有评价的图片不会被当成不满意。反馈自动保存在本机，刷新或重启后仍在。

后续处理相似画面时，系统根据主体场景、亮度、色温和颜色占比，降低不喜欢的调色方法的选择权重，提高喜欢的方法的权重。在兼容该场景的方法中选择三个不同方向，保留探索机会，不把一次不满意变成永久禁用。学习的是实际方法标识，不是 A/B/C 位置或旧 LUT 名称。重复评价覆盖原记录；撤销后旧反馈不会重新生效。`preference-selection.json` 保存场景、匹配反馈、各方法权重及本次选择；权重不是最终入选概率。该反馈影响后续推荐，不会改写已完成的成片。

工作区以大图预览为中心，底部 A/B/C 方案条可点击切换，也可按 `1` / `2` / `3`。右上角切换单图与并排对比，按住空格或“按住看原片”可查看调整前版本。右侧 **调整与导出** 面板提供版本切换、下载和文字微调；诊断、读图卡与推荐路线在 **照片分析** 标签中，配方依据与参数默认收起。窄屏下调整面板会移到照片下方。

选中方案后，在 **还想做一点调整？** 输入意见，再点 **开始微调**。例如“整体亮一点，保留现在的暖色”或“主体亮一点，背景暗一点”，也可点击输入框下的快捷建议后继续修改。明确指令直接转换为有界参数变化；其他描述由已配置的视觉模型结合当前成片解析，执行器限制修改范围。无法理解或涉及不支持的局部对象时，会保留成片并说明原因。

微调以实际接受的 LightCraft 配方为基础，在独立原片副本上应用增量、重新导出并检查。完成后显示微调前后对照；通过 **查看版本** 可切回最初成片、下载任一版本，或基于某次微调继续调整。处理进度、输入意见和版本历史均持久保存；刷新页面会恢复进度。失败不会影响原有三张成片。满意度按钮对应上方卡片中的最初成片，微调意见不自动当成好评或差评。

接口：`PUT /api/jobs/{id}/ratings/{slot}`（`verdict: like / dislike / null`）；`POST /api/jobs/{id}/refinements`（`slot`、`prompt`、UUID `request_id`、可选 `parent_id`）；`GET /api/jobs/{id}/refinements/{refinement_id}` 查询进度。重复请求 ID 返回同一任务，同张图已有微调时拒绝重复排队。普通出片与微调共享同一条后期工作流锁，避免选图、蒙版和参数互相干扰。完整审计保存于 `refinements/{id}/audit.json`。

## What is local vs API

| Piece | Where |
|---|---|
| Decode, EXIF, histogram, clipping, render | Local |
| Recipe library + preference counts | Local (`data/`) |
| Visual diagnosis, planning, review, comparison and feedback | Unified Ollama / OpenAI-compatible / xAI (temporary stand-in, tried last) provider; local fallback records unavailable capabilities |
| RAW | Optional `rawpy` |
| darktable-cli | Detected if installed; not required for v0.1 |

RAW files stay on disk. The cloud critic only receives a JPEG preview.
