-- Run from vendor/lightroom-mcp with a plain Lua interpreter.
package.path = 'plugin/LightroomMCP.lrplugin/?.lua;' .. package.path
local starts, tasks, errors = 0, 0, 0
local fail = false
_G.import = function(name)
    if name == 'LrFunctionContext' then
        return { postAsyncTaskWithContext = function(_, fn) tasks = tasks + 1; fn() end }
    end
    if name == 'LrTasks' then return { pcall = pcall } end
    error('unexpected SDK import: ' .. name)
end
package.loaded.PluginInfoProvider = { startServer = function()
    if fail then error('bind failed') end
    starts = starts + 1
end }
package.loaded.Log = { error = function(msg)
    assert(msg:find('bind failed', 1, true)); errors = errors + 1
end }
local info = dofile('plugin/LightroomMCP.lrplugin/Info.lua')
local handler = dofile('plugin/LightroomMCP.lrplugin/' .. info.URLHandler)
for _, url in ipairs({
    'lightroom://other.plugin/start-server',
    'lightroom://com.lightroom.mcp/stop-server',
    'lightroom://com.lightroom.mcp/start-server?code=anything',
}) do handler.URLHandler(url) end
assert(starts == 0 and tasks == 0)
handler.URLHandler('lightroom://com.lightroom.mcp/start-server')
assert(starts == 1 and tasks == 1)
fail = true
handler.URLHandler('lightroom://com.lightroom.mcp/start-server')
assert(starts == 1 and tasks == 2 and errors == 1)
print('plugin wake URL contract checks passed')
