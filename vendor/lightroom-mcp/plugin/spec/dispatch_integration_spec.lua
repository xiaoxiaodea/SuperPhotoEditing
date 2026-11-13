-- Integration smoke: load PluginInfoProvider the way Lightroom does and drive
-- every DISPATCH entry through the real onMessage → dispatchAction path.
--
-- Unit specs call handlers directly, so they cannot catch a handler that is
-- registered under the wrong name, fails to require, or returns something the
-- JSON encoder rejects. This exercises the actual socket message path with a
-- populated fake catalog.

local helper = require 'spec_helper'

local function buildCatalog()
    local p1 = helper.fakePhoto({
        id = "1", path = "/photos/a.NEF", fileName = "a.NEF",
        rating = 5, pickStatus = 1, colorNameForLabel = "red",
        fileFormat = "RAW", cameraModel = "X-T5", isoSpeedRating = 400,
        dateTimeOriginal = "2026-06-01", thumbnailData = "JPEGBYTES",
        developSettings = { Exposure2012 = 0.5 },
    })
    local p2 = helper.fakePhoto({
        id = "2", path = "/photos/b.jpg", fileName = "b.jpg",
        rating = 2, fileFormat = "JPG", dateTimeOriginal = "2026-06-02",
    })

    local collection = helper.fakeCollection("Picks", { p1 })
    local nested = helper.fakeCollection("Finals", {})
    local set = helper.fakeCollectionSet("2026", { nested }, {})

    return helper.fakeCatalog({
        photos = { p1, p2 },
        collections = { collection },
        collectionSets = { set },
        folders = { helper.fakeFolder("2026", "/photos/2026", { p1 }, {}) },
        keywords = { helper.fakeKeyword("Nature", { synonyms = { "outdoors" } }) },
    })
end

-- Minimum viable params per action so each handler reaches its happy path.
--
-- Ordered, not a bare map: several of these mutate shared catalog state (the
-- rename must not run before the tools that address "Picks" by name), and
-- pairs() iteration order in Lua is unspecified, which would make the suite
-- pass or fail depending on hash layout.
local ACTION_PARAMS = {
    { "ping", {} },
    { "search_photos", { rating = 5, rating_operation = ">=" } },
    { "get_catalog_info", {} },
    { "list_folders", { include_photo_counts = true } },
    { "list_keywords", {} },
    { "get_photo_preview", { photo_id = "1", return_base64 = true } },
    { "get_selected_photos", {} },
    { "set_selected_photos", { photo_ids = { "1" } } },
    { "get_photo_metadata", { photo_id = "1" } },
    { "set_photo_metadata", { photo_ids = { "1" }, metadata = { city = "Guangzhou" } } },
    { "set_flag", { photo_ids = { "1" }, flag = "pick" } },
    { "set_color_label", { photo_ids = { "1" }, color = "blue" } },
    { "list_collections", {} },
    { "list_collection_sets", {} },
    { "get_collection_photos", { collection_name = "Picks" } },
    { "create_collection", { name = "Fresh", parent = "2026" } },
    { "create_collection_set", { name = "Archive" } },
    { "create_smart_collection", { name = "Keepers", criteria = { rating = 4, rating_operation = ">=" } } },
    { "add_to_collection", { collection_name = "Picks", photo_ids = { "2" } } },
    { "remove_from_collection", { collection_name = "Picks", photo_ids = { "2" } } },
    -- Mutates the name every earlier entry depends on, so it runs last.
    { "rename_collection", { collection_name = "Picks", new_name = "Selects" } },
    { "delete_collection", { collection_name = "Finals" } },
    { "set_keywords", { photo_ids = { "1" }, add_keywords = { "Sunset" } } },
    { "set_rating", { photo_ids = { "1" }, rating = 4 } },
}

local COVERED_ACTIONS = {}
for _, entry in ipairs(ACTION_PARAMS) do COVERED_ACTIONS[entry[1]] = true end

describe("dispatch integration", function()
    local catalog, mod, binds, sent

    before_each(function()
        catalog = buildCatalog()
        binds, sent = {}, {}

        helper.installImport({
            LrApplication = {
                activeCatalog = function() return catalog end,
                versionTable = function() return { major = 14, minor = 2 } end,
                versionString = function() return "14.2" end,
                developPresetFolders = function() return {} end,
            },
            LrTasks = {
                startAsyncTask = function(fn) fn() end,
                -- startServer runs a monitor loop that only exits when running
                -- goes false, so the loop must be stopped on its first tick or
                -- this spec spins forever (same trick as PluginInfoProvider_spec).
                sleep = function()
                    if _G.LightroomMCP_State then
                        _G.LightroomMCP_State.running = false
                    end
                end,
                pcall = pcall,
            },
            LrLogger = helper.defaultLrLogger(),
            LrDialogs = { message = function() end },
            LrFunctionContext = {
                postAsyncTaskWithContext = function(_, fn)
                    fn({ addCleanupHandler = function() end })
                end,
            },
            LrSocket = {
                bind = function(opts)
                    table.insert(binds, opts)
                    return {
                        close = function() end,
                        reconnect = function() end,
                        send = function(_, data) table.insert(sent, data) end,
                    }
                end,
            },
            LrPrefs = { prefsForPlugin = function() return {} end },
            LrView = { bind = function() end },
            LrUUID = { generateUUID = function() return "uuid" end },
            LrPathUtils = {
                child = function(a, b) return a .. "/" .. b end,
                getStandardFilePath = function() return "/home" end,
                leafName = function(p) return p:match("[^/\\]+$") or p end,
                extension = function(p) return p:match("%.([^.]+)$") or "" end,
                removeExtension = function(p) return (p:gsub("%.[^.]+$", "")) end,
            },
            LrFileUtils = {
                exists = function() return false end,
                createAllDirectories = function() end,
            },
            LrStringUtils = { encodeBase64 = function(d) return "B64:" .. #d end },
            -- Needed only because loading PluginInfoProvider requires every
            -- handler module, including the export/import ones this suite does
            -- not drive.
            LrExportSession = function() return {} end,
        })

        -- Reload every plugin module so the fresh mocks apply.
        for _, name in ipairs({
            'PluginInfoProvider', 'JSON', 'Log', 'Validate', 'PhotoLookup',
            'CollectionLookup', 'HandlerSearch', 'HandlerCollections',
            'HandlerMetadata', 'HandlerMetadataWrite', 'HandlerOrganization',
            'HandlerImport', 'HandlerExport', 'HandlerSelection',
            'HandlerDevelop', 'HandlerCatalog', 'HandlerPreview',
        }) do
            package.loaded[name] = nil
        end

        _G.LightroomMCP_State = nil
        local realOpen = io.open
        io.open = function(path, mode, ...)
            if mode and mode:find("w", 1, true) then
                return { write = function() end, close = function() end }
            end
            return realOpen(path, mode, ...)
        end
        mod = require 'PluginInfoProvider'
        mod.startServer()
        io.open = realOpen
    end)

    -- DISPATCH is module-local, so read the registered action names from source
    -- rather than exporting the table purely for this spec.
    local function dispatchActionNames()
        local file = assert(io.open("plugin/LightroomMCP.lrplugin/PluginInfoProvider.lua", "r"))
        local source = file:read("*a")
        file:close()
        local body = source:match("local DISPATCH = %{(.-)\n%}")
        assert(body, "DISPATCH table not found")
        local names = {}
        for name in body:gmatch("\n%s*([a-z_]+)%s*=") do
            table.insert(names, name)
        end
        return names
    end

    it("routes every advertised action to a live handler", function()
        -- Guards against a DISPATCH entry pointing at a nil field (e.g. a typo in
        -- a handler function name), which unit specs never touch because they
        -- call handlers directly.
        local JSON = require 'JSON'
        local state = _G.LightroomMCP_State
        state.sendConnected = true

        for _, action in ipairs(dispatchActionNames()) do
            sent = {}
            state.responseSocket = { send = function(_, data) table.insert(sent, data) end }
            binds[1].onMessage(nil, JSON:encode({
                id = 1, action = action, hello = state.token, params = {},
            }))

            assert.are.equal(1, #sent, "no response for " .. action)
            local decoded = JSON:decode(sent[1]:gsub("\n$", ""))
            -- Called with empty params, a handler may legitimately reject the
            -- arguments; what must never happen is "Unknown action", which means
            -- the dispatch entry does not resolve to a callable handler.
            if decoded.error then
                assert.is_nil(decoded.error:match("Unknown action"),
                    action .. " is not routed to a handler")
            end
        end
    end)

    it("covers every DISPATCH action in this suite", function()
        -- Forces new tools to be smoke-tested here rather than silently skipped.
        local uncovered = {}
        for _, action in ipairs(dispatchActionNames()) do
            if COVERED_ACTIONS[action] == nil then table.insert(uncovered, action) end
        end
        table.sort(uncovered)
        -- Import/export and develop-preset actions need real files or preset
        -- folders, so they are exercised by their own specs and the e2e runner.
        local EXPECTED_UNCOVERED = {
            "apply_develop_preset", "compare_develop_presets", "copy_develop_settings",
            "create_develop_preset", "export_develop_preset", "export_photos",
            "get_develop_preset", "import_photos", "list_develop_presets",
            "set_develop_settings",
        }
        assert.are.same(EXPECTED_UNCOVERED, uncovered)
    end)

    it("returns a successful JSON response for every smoke-tested action", function()
        local JSON = require 'JSON'
        local state = _G.LightroomMCP_State
        state.sendConnected = true

        for _, entry in ipairs(ACTION_PARAMS) do
            local action, params = entry[1], entry[2]
            sent = {}
            state.responseSocket = { send = function(_, data) table.insert(sent, data) end }

            -- The dispatcher reads arguments from request.params, not from
            -- top-level request fields.
            local request = {
                id = 1,
                action = action,
                hello = state.token,
                params = params,
            }

            binds[1].onMessage(nil, JSON:encode(request))

            assert.are.equal(1, #sent, "no response sent for " .. action)
            local decoded = JSON:decode(sent[1]:gsub("\n$", ""))
            -- A handler error comes back as an error field; assert the happy path
            -- so a broken handler cannot pass by erroring politely.
            assert.is_nil(decoded.error, action .. " errored: " .. tostring(decoded.error))
            assert.is_not_nil(decoded.result, action .. " returned no result")
        end
    end)

    it("reports an unknown action as an error rather than crashing", function()
        local JSON = require 'JSON'
        local state = _G.LightroomMCP_State
        state.sendConnected = true
        state.responseSocket = { send = function(_, data) table.insert(sent, data) end }
        sent = {}

        binds[1].onMessage(nil, JSON:encode({
            id = 9, action = "no_such_action", hello = state.token,
        }))

        local decoded = JSON:decode(sent[1]:gsub("\n$", ""))
        assert.is_not_nil(decoded.error)
    end)
end)
