# Windows 本机验证记录

验证日期：2026-11-13。环境：Windows 10 x64，Python 3.12.10，uv，系统 Edge/WebView2，LightCraft CLI 0.21.1。

## 已实际验证

- `uv sync --python 3.12 --extra dev --extra raw` 安装成功。首次由 uv 自动选择 Python 3.15 时 rawpy 构建失败，已用 `.python-version` 固定 3.12。
- 最终完整回归 `uv run pytest`：517 passed，1 skipped（27.58s）；包含原生检查点持久化回归。跳过的是依赖 Lua 解释器的测试，本机未安装 Lua。
- 官方 Windows portable ZIP 经 SHA256 校验后安装，CLI 实际返回 `lightcraft-cli 0.21.1`。该来源的程序执行已获用户明确批准。
- **完整 LightCraft 应用流程**：仓库风景、人物样片各交付 A/B/C；六张 JPEG 下载及解码成功。风景输出 1600×1067、人物输出 1400×1750（原样片尺寸小于默认 2400 长边，无放大）。两次“整体亮一点，保留现在的暖色”微调均 ready，新版本下载、解码成功，像素/文件与原版本有区别。这里实际使用 local 技术诊断；本机没有可用视觉模型或云端密钥。
- **真实局部控制**：同一风景样片分别以无编辑、天空 -0.5EV 渲染；平均绝对像素变化 8.49/255，上半部 15.99，下半部 1.02。确认天空控制实际影响输出，不能据此保证所有照片的区域选择都准确。
- **配方保存**：接受状态现在真正保存为 LightCraft 显影 JSON，计划参数明确标记为 planned，不冒充引擎元数据回读；保存文件已传给原生 CLI 实际重放，输出像素与对应版本一致（每通道允许 1/255 容差）。
- **RAW 路径**：LightCraft 生成的程序化合成 DNG 成功解码并导出 640×427 JPEG。未验证具体相机机型的 RAW。
- **基础模式**：关闭 LightCraft/Ollama/semantic 后，NumPy/Pillow 也实际完成两张样片各三套 JPEG；风景 1600×1067、人物 1280×1600。网页准确显示基础模式并禁用文字微调。
- 满意度反馈保存成功；基础模式服务停止并重启后，作业 ready 状态、结果路径和 like 记录保留。
- 系统 Edge 实际打开页面、上传照片、等待结果并截图；基础模式与完整 LightCraft 模式均无 JavaScript 异常，截图已查看。
- 独立启用访问令牌的服务完成网页登录、上传、下载和反馈；无令牌访问 `/api/health` 返回 401。
- 原生 pywebview 窗口实际使用 `edgechromium` 打开，读取到网页标题、HTTP 地址及上传按钮后自动关闭。
- 目录导入通过运行中的 HTTP 服务提交两张样片并完成处理；重复扫描没有新任务。
- `uv build --wheel` 成功，检查 wheel 包含网页 HTML/JS/CSS、XMP 预设、vision/tasks 模块。它是 Python 安装包，不是自带 Python 的独立 EXE。

## 本机结果文件（不进版本控制）

- `.task-cache/windows-lightcraft-verification/report.json`：完整引擎出片、微调状态、浏览器验证。
- `.task-cache/windows-lightcraft-verification/windows-browser.png`：完整模式结果页截图。
- `.task-cache/windows-lightcraft-verification/*.jpg`：完整模式交付图。
- `.task-cache/windows-engine-verification/report.json`：局部控制、微调下载、合成 RAW 验证。
- `.task-cache/windows-engine-verification/*-refined.jpg`：两张真实微调结果。
- `.task-cache/windows-verification/report.json`、`windows-browser.png`：基础模式验证。
- `.task-cache/windows-verification/desktop.json`、`restart.json`：原生窗口、实际重启验证。
- `.task-cache/windows-auth-verification/`：开启认证后的浏览器验证。
- `.task-cache/windows-lightcraft-validation/`、`windows-validation/`、`windows-auth-validation/`：隔离的服务作业、SQLite 和审计。

## 离线验证覆盖

统一视觉接口新增 67 项 OpenAI SDK + httpx.MockTransport 测试，覆盖 xAI/OpenAI-compatible × Chat/Responses × 六视觉阶段、独立损坏复核、多图顺序、结构化输出协商、错误/拒绝/截断、超时重试、用量、密钥脱敏，以及 provider/model/usage 的 JSON 持久化。不读取真实 `.env` 或调用真实云端服务。

队列测试覆盖 queued 恢复、running 中断标记、普通与微调串行、正常退出等待、第二个 Windows 进程的真实锁竞争、分块上传及失败清理、Bearer/cookie/跨站请求与令牌轮换。目录导入另覆盖文件变化等待、上传失败重试和输出目录排除。

安装器、白平衡和分割测试覆盖平台资产选择、校验与替换失败保留旧文件、不提取无关归档路径、未知原片白平衡不使用 6500K、分割只放大预测结果、HF_HOME/HF_HUB_CACHE 与完整分片缓存识别。没有安装大型 torch/分割模型做本机内存基准测试。

## 尚未验证的边界

- 云端尚未使用真实 API 密钥验证，本机也没有可用 Ollama 视觉模型；实际出片使用本地技术规则和 LightCraft。强模型的视觉诊断、规划与审美复核须配置模型后再实测。
- NAS/Linux/ARM 仅准备了安装适配和部署约束，没有实机验证。
- 合成 DNG 不能证明具体相机 RAW、As Shot 元数据的兼容性；未取得真实白平衡基值时跳过相对 WB 调整。
- 默认完整导出长边 2400px，完整的原尺寸高位深交付流程尚未完善。
- 偏好记录是测试用样片反馈，保存在隔离验证目录，不影响正常 `data/` 的用户偏好。

## 复验

启动完整 LightCraft 服务，在 PowerShell 中设置：

```powershell
$env:SUPERPHOTO_VISION_PROVIDER="local"
$env:SUPERPHOTO_OLLAMA_ENABLED="0"
$env:SUPERPHOTO_SEMANTIC_ENABLED="0"
$env:SUPERPHOTO_LIGHTROOM_ENABLED="1"
$env:SUPERPHOTO_DATA_DIR="$PWD/.task-cache/windows-lightcraft-validation"
uv run python -m superphoto.cli serve --port 8878
```

另一终端运行（系统已有 Edge，无需下载 Playwright 浏览器）：

```powershell
uv run --with playwright python scripts/verify_windows_flow.py --base http://127.0.0.1:8878 --out .task-cache/windows-lightcraft-verification --browser "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
uv run python scripts/verify_native_engine.py
```

基础模式将 `SUPERPHOTO_LIGHTROOM_ENABLED` 改成 `0`，使用独立数据目录和端口即可。原生窗口可用 `scripts/verify_windows_desktop.py` 验证，脚本会短暂打开真实窗口并自动关闭。

Windows 升级依赖或修改 pyproject 后先关闭服务再 `uv sync`，否则运行中的 `superphoto.exe` 可能锁住待替换启动器。开发时可先同步，再用 `uv run --no-sync python -m superphoto.cli serve`。

启动和驱动过程已经整理成上述可复验脚本；如需将本机环境步骤进一步固化为项目运行技能，可使用 `/run-skill-generator`。
