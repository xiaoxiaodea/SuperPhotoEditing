# 照片工作区重设计

2026-11-13，在 Windows 10 / Edge / WebView2 上实施并验证。

## 布局与操作

- 中性深灰画布、青柠色操作强调、统一无衬线字体，减少装饰性卡片和长段说明。
- 默认单张大图；可切换并排对比。预览使用 contain，横图和竖图均完整显示。
- A/B/C 缩为底部方案条，保留逐张满意／不满意及撤销评价。`1` / `2` / `3` 切换方案。
- 右侧「调整与导出」集中版本选择、下载、文字微调、快捷意见和微调记录。
- 「照片分析」独立标签保留原片、诊断、技术信息、读图卡和推荐路线。方案说明与实际配方默认折叠。
- 按住照片、空格或对比按钮查看调整前版本；微调版的参考图是其实际父版本。
- 新设计覆盖导入、处理进度、结果、登录和错误页面。390px 窄屏将面板移至照片下方。
- 标签支持方向键，上传区和对比按钮支持键盘；系统减少动态效果设置会停用旋转动画。
- Windows 原生窗口的启动底色和字体与新版一致。

## 本机验证

浏览器验证脚本位于 `.task-cache/verify_workspace_ui.py` 与 `.task-cache/verify_workspace_workflows.py`，报告与截图在 `.task-cache/workspace-redesign/`。

实际驱动系统 Edge 完成：

- 恢复已有 LightCraft 风景、人物任务，检查单图／并排对比、鼠标与键盘选片、按住对比、版本选择、真实 JPEG 下载。
- 通过 HTTP 保存评价并查询持久化结果，检查刷新恢复和明确重置后保持导入页。
- 检查桌面和 390px 页面无横向溢出，查看实际截图。
- 临时独立服务开启认证：未登录 API 401、无效令牌报错、有效登录、cookie 导航恢复、手机登录布局。
- 从新版网页上传真实风景样片，在基础模式下完成三套成片并下载；微调输入和按钮正确禁用。
- 上传损坏的测试 JPEG，显示处理失败；点击重试创建新任务，再次失败后可返回导入页。
- 从新版网页提交「整体亮一点，保留现在的暖色」，实际调用 LightCraft 完成微调；处理中刷新恢复，新版本自动出现、可下载，初始版本仍可选择。新记录：`73a4a7a1-52b7-43f9-b3f4-2ef27089b497`。
- 两份浏览器报告均无 JavaScript 异常。
- 原生 pywebview 实际启动，后端 `edgechromium`，标题 `SuperPhoto · 照片工作室`，加载新版 UI 后自动关闭。
- `node --check web/app.js` 通过。
- `uv run --no-sync --with pytest python -m pytest tests/test_api.py tests/test_macos_app.py`：15 passed。首次直接调用环境外的 pytest 入口报 `ModuleNotFoundError: openai`；改为上述项目 Python 环境后通过。
- `uv build --wheel` 成功，`dist/superphoto-0.1.0-py3-none-any.whl` 已更新。

代表性截图：

- `workspace-refined.png`：完成真实微调后的桌面工作区。
- `workspace-compare.png`：并排对比。
- `workspace-analysis.png`：照片分析面板。
- `workspace-portrait.png` / `workspace-mobile.png`：竖图与窄屏。
- `import-desktop.png` / `import-mobile.png`：导入页。
- `processing-desktop.png`、`login-desktop.png`、`error-desktop.png`：进度、登录与失败状态。

验证使用隔离样片数据。真实处理仍使用本地技术规则和 LightCraft；此次界面改造不改变此前记录的云端视觉、NAS 实机和相机 RAW 验证边界。
