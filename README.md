# SuperPhoto

本地照片后期工具：看懂照片里有什么，生成多套方向不同的修图方案，再通过选图反馈和文字微调完成后期；你的选择留在本机，用来决定类似照片的方向。支持 JPEG 与 RAW，使用 LightCraft 渲染，保留原片和可检查的显影配方。

## 快速开始

推荐 Python 3.12，先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在仓库目录执行：

```sh
uv sync --python 3.12 --extra raw
uv run superphoto install-lightcraft
uv run superphoto serve
```

浏览器打开 <http://127.0.0.1:8765>。

- **Windows 桌面窗口**：完成上述安装后，双击 `Start-SuperPhoto.cmd`；需要系统 WebView2 Runtime。
- **macOS 桌面窗口**：运行 `uv run superphoto install-app`，再打开 `SuperPhoto.app`。
- **Linux / NAS**：使用浏览器模式；LightCraft 还需匹配系统架构和运行库，尚未完成 NAS 实机验证。

## 配置视觉模型

推荐点击页面顶栏的 **视觉模型**，设置保险箱口令后添加模型。可配置多个服务并按优先级切换，密钥加密保存在本机 `data/vision-vault.json`。

也可以复制 [.env.example](.env.example) 为 `.env`，按供应商填写：

```dotenv
SUPERPHOTO_VISION_PROVIDER=openai
SUPERPHOTO_OPENAI_API_KEY=你的密钥
SUPERPHOTO_OPENAI_BASE_URL=https://你的服务/v1
SUPERPHOTO_OPENAI_MODEL=服务支持的视觉模型名称
SUPERPHOTO_OPENAI_PROTOCOL=chat/completions
```

协议可选 `chat/completions` 或 `responses`，需与服务匹配。支持本机 Ollama；自动模式依次尝试 Ollama、OpenAI 兼容服务、xAI 和本地技术分析。模型下载需手动确认。

`.env` 中的密钥为明文，请勿提交。使用云端视觉服务会发送 JPEG 预览，RAW 原文件保留在本机。更多说明见 [视觉模型保险箱](docs/vision-vault-2026-11-16.md)。

## 使用流程

1. 上传照片，等待生成方案（默认 6 套）。A 是自然完成，其余每套由视觉模型按画面元素选定方向，例如戏剧光影、通透明亮或某种配色，并说明理由。你在类似照片上的选择会被参考，选得多的方向排在前面。
2. 点击方案或按数字键 `1`–`6` 依展示顺序切换；按住空格查看原片，也可并排比较。
3. 对每个方案选择 **pass / 留下 / 最爱**（每张照片一个最爱，再点一次撤销）。选择只保存在本机，用于后续相似照片的方向推荐；pass 只表示这种处理不适合这张，不会彻底排除。
4. 在右侧输入“整体亮一点，保留暖色”等意见，生成微调版本并下载。版本历史可随时切换。
5. 需要原图尺寸时，在下载按钮下方点 **原尺寸导出**：JPEG 适合分享和冲印；16 位 TIFF 适合继续精修，但文件很大（4500 万像素约 210 MB、10 秒左右）。导出会用当前版本保存的显影记录，在原片上重新渲染。影调和颜色与预览一致，锐化、纹理和降噪按原尺寸计算。

LightCraft 和视觉模型均可用时，默认使用视觉定向修图：模型先读出画面元素，再结合本机经验选方向、提出影调、颜色与局部调整目标；求解器试渲染后，用一次合并请求检查所有成片有无光晕、蒙版边缘等损坏。方案数和复核方式可通过 `SUPERPHOTO_DIRECTED_VERSIONS`、`SUPERPHOTO_DIRECTED_MAX_REVISIONS` 调整，详见[多套方案与本地经验](docs/many-versions-2026-11-23.md)。无法完成时自动回到预设流程，原因保存在任务记录中。

没有视觉模型时使用本地技术分析，审美判断和复杂文字微调能力有限。没有 LightCraft 时，可设置 `SUPERPHOTO_LIGHTROOM_ENABLED=false` 启用基础渲染；基础模式不支持文字微调。

## 常用命令

```sh
# 检查渲染引擎
uv run superphoto lightcraft-status

# 验证视觉模型（会发送一张小图）
uv run superphoto vision-check --list-models

# 处理单张照片；与服务使用同一数据目录时，先关闭服务
uv run superphoto process "path/to/photo.jpg"

# 持续导入目录中的照片，需要先启动服务
uv run superphoto import-folder "D:/Photos/Incoming" --watch
```

自动导入支持本地目录和 NAS 共享，文件稳定后才上传；省略 `--watch` 可只扫描一次。助手调用方式见 [SuperPhoto 技能](skills/superphoto/SKILL.md)。

## 配置与部署

完整配置见 [.env.example](.env.example)，常用选项：

| 配置 | 用途 |
| --- | --- |
| `SUPERPHOTO_DATA_DIR` | 数据、任务和反馈目录，默认 `./data` |
| `SUPERPHOTO_PIPELINE_MODE` | `auto` 自动选择；`directed` 仅视觉定向；`legacy` 仅预设流程 |
| `SUPERPHOTO_LIGHTROOM_EXPORT_EDGE` | LightCraft 导出长边，默认 2400px |
| `SUPERPHOTO_LIGHTCRAFT_BIN` | 自定义 LightCraft CLI 路径 |
| `SUPERPHOTO_AI_DENOISE` | 高 ISO 或暗部大幅提亮的 RAW 成片自动使用 AI 降噪，默认开启；降噪数据缓存在 `data/denoise`，上限由 `SUPERPHOTO_AI_DENOISE_CACHE_GB` 设置，默认 4 GB |
| `SUPERPHOTO_SEMANTIC_ENABLED` | 是否启用可选语义分割，低内存环境可设为 `0` |

局域网访问需设置 `SUPERPHOTO_HOST=0.0.0.0` 和长随机 `SUPERPHOTO_ACCESS_TOKEN`；公网部署还需 HTTPS 反向代理。每个数据目录只运行一个服务，数据库与任务目录放在运行主机的本地文件系统，NAS 共享可作为源片目录。

完整流程的方案预览为长边 2400px 的 JPEG，可按需导出原尺寸 JPEG 或 16 位 sRGB TIFF；基础模式只有 1600px 预览，不支持原尺寸导出。高 ISO 或暗部大幅提亮的 Bayer 传感器 RAW（如尼康、索尼、佳能的原生 RAW）会在最终成片上自动使用 LightCraft AI 降噪；JPEG 和 iPhone ProRAW 等线性 DNG 不做。细节修复等功能尚未自动完成，输出仍需人工检查。

## 开发与文档

```sh
uv sync --python 3.12 --extra raw --extra dev
uv run pytest
```

- [视觉定向修图](docs/directed-finishing-2026-11-16.md)
- [多套方案与本地经验](docs/many-versions-2026-11-23.md)
- [风格与候选方案](docs/style-workflow-2026-11-14.md)
- [选图反馈与微调](docs/feedback-and-refinement-2026-09-08.md)
- [Windows 验证记录](docs/windows-validation.md)

本地原片、商业预设、参考资料和输出目录不随仓库分发。
