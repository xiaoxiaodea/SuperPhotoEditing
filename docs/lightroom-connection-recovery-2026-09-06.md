# Lightroom 连接恢复

## 现场证据

用户失败作业 `01d670e3-e49b-4614-bd4f-c9d32ca7cba7`、`5bacd1ac-71cf-4742-bac8-1f569ac8149f` 均在导出未编辑基线之前退出。Lightroom 进程存在，但插件服务器没有持续运行；不能把进程存在当作可处理照片。

插件偏好确认 `autoStartServer = true`，插件没有被禁用，安装文件与工程一致。原有恢复仅等待并重建 Node 连接进程，无法启动已停止的 Lua 服务。

Lightroom 自身日志 `~/Library/Logs/Adobe/Lightroom/LrClassicLogs/LightroomMCP.log` 显示：17:54:57 插件启动并连接，17:55:01 收到 `Stopping LrSocket servers`。`~/Documents/LrClassicLogs/LightroomMCP.log` 此时没有继续更新，因此不能只依赖这份日志判断插件是否加载。现有证据不能确定此前 Lightroom 停在哪个启动窗口；本机没有授予本次自动化辅助功能读取权限。

## 修复

- 新增 SDK URL 入口，仅接受固定 `lightroom://com.lightroom.mcp/start-server`，不能通过 URL 修改照片、执行脚本或停止服务。
- 处理前先检查真实连接；已连接直接复用。未连接时打开 Lightroom 并唤醒插件，等待 30 秒；仍未连接则重建桥接、再次唤醒并等待 12 秒。调用具有互斥保护，健康会话不会因检查被重启。
- 检查安装的插件是否具有 URL 入口，避免向旧插件发送未知动作。插件升级仍需加载新版本。
- 插件任务结束时清除 `running`，避免异常结束后状态仍显示运行、下一次启动被阻挡。
- 失败页提供中文指引和“重试这张照片”。重试用保存的原片创建新任务，不要求旧任务已经生成三份配方，也不覆盖失败记录。
- 修复前端第一次轮询就结束时仍创建计时器，以及轮询重叠、旧任务响应覆盖新任务的问题。

URL 入口遵循 Adobe 的 [Lightroom Classic SDK Guide](https://ioconsolerykerprodcdn.azureedge.net/static/installers/lr/sdk/2022/cross_platform/v13/doc/Lightroom%20Classic%20SDK%20Guide_1655133965.pdf) 中 `URLHandler` 的注册方式。

## 验证

- Python 全套 148 项测试通过；连接、处理和 API 相关 31 项通过。
- Lua 唤醒 URL 契约检查、原生蒙版契约检查通过；JavaScript 语法检查通过。
- 前端实际脚本在模拟 DOM 中验证：英文连接错误转换、原片重试请求、按钮恢复，以及终态不再轮询、运行中仅保留一个后续轮询。
- 新插件与已安装版本文件一致。更新本机应用、正常重启 Lightroom 后，`POST /api/lightroom/connect` 实际成功，发现 34 个工具。18:05:38 插件日志记录 `Already running`，确认固定 URL 已被 Lightroom 路由，且重复唤醒保持原会话。
- 通过本机应用重试 `5bacd1ac-71cf-4742-bac8-1f569ac8149f`，新作业 `1b65c5ef-b23f-4690-a085-828fb68acb76` 于 18:10:01 完成（约 4 分 22 秒），三份实际 Lightroom 成片已交付；重试前后的原片 SHA-256 一致。
- 城市照片 AB / AC / BC 的平均 Lab 距离分别为 4.20 / 2.88 / 4.63；B 初轮 2.60，经自身方向调整至 4.20，任务正常完成。该指标只描述像素差异，不表示审美水平。
- `.task-cache/recovered-city-comparison.html` 提供原片与三份成片的独立对比页；生成时核对了展示文件和实际接受的 Lightroom 导出字节一致。
- 未重启应用或 Lightroom，继续通过同一连接重试此前失败的山景任务。新作业 `4974d3ef-179a-4212-bb78-bc846686c8f3` 于 18:16:33 完成（约 4 分 35 秒），A/B/C 均已实际导出，原片校验一致。对比页为 `.task-cache/recovered-mountain-comparison.html`。
