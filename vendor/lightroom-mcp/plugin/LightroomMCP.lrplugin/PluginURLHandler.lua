-- An explicit SuperPhoto processing/retry request can wake a stopped server.
-- Only this fixed action is accepted; URLs cannot edit photos or run code.
local LrFunctionContext = import 'LrFunctionContext'
local LrTasks = import 'LrTasks'
local Log = require 'Log'

return {
    URLHandler = function(url)
        if url ~= 'lightroom://com.lightroom.mcp/start-server' then return end
        LrFunctionContext.postAsyncTaskWithContext('LightroomMCPWake', function()
            local provider = require 'PluginInfoProvider'
            local ok, err = LrTasks.pcall(provider.startServer)
            if not ok then Log.error('Wake failed: ' .. tostring(err)) end
        end)
    end,
}
