-- Standalone SDK contract regression checks; run from vendor/lightroom-mcp.
package.path = 'plugin/LightroomMCP.lrplugin/?.lua;' .. package.path
local photo = { localIdentifier = 42 }
local previousPhoto = { localIdentifier = 99 }
local selectedPhoto, selectedMask, currentModule = previousPhoto, 'user-mask', 'library'
local userValues = { local_Exposure = 0.7, local_Contrast = -12,
    local_Whites = 29, local_Blacks = 31, local_Sharpness = 14 }
local masks = { ['user-mask'] = {} }
for key, value in pairs(userValues) do masks['user-mask'][key] = value end
local created, failWrite, noTarget = 0, false, false
local toolAttempts, toolReadyAfter = 0, 1
local ignoredWriteKey, ignoredWriteValue
local accesses = {}
local function assertUserUnchanged()
    for key, value in pairs(userValues) do
        assert(masks['user-mask'][key] == value, 'must preserve user control ' .. key)
    end
    for key, _ in pairs(masks['user-mask']) do
        assert(userValues[key] ~= nil, 'must not add controls to a user mask')
    end
end
local function recordAccess(kind, key, value)
    assert(selectedPhoto == photo, 'local controls must belong to the job photo')
    assert(selectedMask ~= 'user-mask' and masks[selectedMask], 'must only read or write an owned mask')
    table.insert(accesses, { kind = kind, mask = selectedMask, key = key, value = value })
end
local catalog = {
    kAllPhotos = 'all',
    getActiveSources = function() return { 'previous' } end,
    getCurrentViewFilter = function() return { filtersActive = true } end,
    setActiveSources = function() end,
    setViewFilter = function() end,
    withReadAccessDo = function(_, fn) return fn() end,
    getTargetPhoto = function() return selectedPhoto end,
    getTargetPhotos = function() return { selectedPhoto } end,
    setSelectedPhotos = function(_, p) selectedPhoto = p end,
}
local controller = {
    selectTool = function(tool)
        assert(tool == 'masking')
        toolAttempts = toolAttempts + 1
    end,
    getSelectedTool = function() return toolAttempts >= toolReadyAfter and 'masking' or 'crop' end,
    createNewMask = function(kind, target)
        assert(kind == 'aiSelection' and (target == 'subject' or target == 'sky'))
        assert(selectedPhoto == photo and currentModule == 'develop')
        if noTarget then selectedMask = nil; return end
        created = created + 1
        selectedMask = 'owned-' .. created
        masks[selectedMask] = { local_Exposure = 0.6, local_Contrast = 17,
            local_Whites = 50, local_Blacks = -8, local_Clarity = 9, local_Sharpness = 42 }
    end,
    getSelectedMask = function() return selectedMask end,
    getAllMasks = function()
        local result = {}
        for id, _ in pairs(masks) do table.insert(result, { ID = id }) end
        return result
    end,
    selectMask = function(id) selectedMask = masks[id] and id or nil end,
    deleteMask = function(id)
        assert(id ~= 'user-mask', 'must never delete a user mask')
        masks[id] = nil
        selectedMask = nil
    end,
    getRange = function(key)
        if key == 'local_Exposure' then return -4, 4 end
        return -100, 100
    end,
    getValue = function(key)
        recordAccess('read', key)
        return masks[selectedMask][key]
    end,
    setValue = function(key, value)
        recordAccess('write', key, value)
        if failWrite and value == 0.4 then error('simulated SDK write failure') end
        if key == ignoredWriteKey and value == ignoredWriteValue then return end
        masks[selectedMask][key] = value
    end,
}
local modules = {
    LrApplication = { activeCatalog = function() return catalog end },
    LrApplicationView = {
        getCurrentModuleName = function() return currentModule end,
        switchToModule = function(name) currentModule = name end,
    },
    LrDevelopController = controller,
    LrTasks = { pcall = pcall, sleep = function() end },
}
_G.import = function(name) return assert(modules[name], name) end
package.loaded.PhotoLookup = { resolveOne = function() return photo end }
local handler = require 'HandlerMasking'
local requested = { local_Exposure = -1.58, local_Contrast = 28, local_Whites = -61, local_Blacks = -23 }
local result = handler.applyAiMask({ photo_id = 'copy.jpg', target = 'subject', settings = requested })
assert(result.created and result.success)
for key, value in pairs(requested) do
    assert(result.settings[key] == value and masks[result.mask_id][key] == value,
        'must apply and read back the native local value for ' .. key)
    local firstWrite
    for _, access in ipairs(accesses) do
        if access.kind == 'write' and access.mask == result.mask_id and access.key == key then
            firstWrite = access.value
            break
        end
    end
    assert(firstWrite == 0, 'must neutralize remembered sliders before applying ' .. key)
end
assert(masks[result.mask_id].local_Clarity == 0 and masks[result.mask_id].local_Sharpness == 0,
    'must clear remembered brush sliders not present in the request')
assertUserUnchanged()
assert(selectedPhoto == previousPhoto and currentModule == 'library')
selectedMask = 'user-mask'
local updated = handler.applyAiMask({ photo_id = 'copy.jpg', target = 'subject', settings = {
    local_Exposure = 0.2, local_Contrast = 0, local_Whites = 0, local_Blacks = 0 } })
assert(not updated.created and updated.mask_id == result.mask_id and created == 1)
assert(masks[result.mask_id].local_Exposure == 0.2 and masks[result.mask_id].local_Contrast == 0
    and masks[result.mask_id].local_Whites == 0 and masks[result.mask_id].local_Blacks == 0)
assertUserUnchanged()

-- A successful SDK call that silently ignores a slider must fail readback,
-- restore every previous local value, and never use the user's active mask.
ignoredWriteKey, ignoredWriteValue = 'local_Whites', 67
selectedMask = 'user-mask'
local readbackOk, readbackError = pcall(handler.applyAiMask, {
    photo_id = 'copy.jpg', target = 'subject', settings = {
        local_Exposure = -1.58, local_Contrast = 37, local_Whites = 67, local_Blacks = -12 } })
assert(not readbackOk and tostring(readbackError):find('Local adjustment did not read back: local_Whites', 1, true))
assert(masks[result.mask_id].local_Exposure == 0.2 and masks[result.mask_id].local_Contrast == 0
    and masks[result.mask_id].local_Whites == 0 and masks[result.mask_id].local_Blacks == 0,
    'failed readback must restore the entire requested local state')
assertUserUnchanged()
assert(selectedPhoto == previousPhoto and currentModule == 'library')
ignoredWriteKey, ignoredWriteValue = nil, nil

failWrite = true
assert(not pcall(handler.applyAiMask, { photo_id = 'copy.jpg', target = 'subject', settings = { local_Exposure = 0.4 } }))
assert(masks[result.mask_id].local_Exposure == 0.2, 'failed update must restore prior sliders')
assert(not pcall(handler.applyAiMask, { photo_id = 'copy.jpg', target = 'sky', settings = { local_Exposure = 0.4 } }))
assert(masks['owned-2'] == nil, 'failed new mask must be removed')
assert(selectedPhoto == previousPhoto and currentModule == 'library')
assert(not pcall(handler.applyAiMask, { photo_id = 'copy.jpg', target = 'subject', settings = { Exposure2012 = 1 } }))
failWrite = false

-- Develop may be selected while Masking still ignores the first three opens.
toolAttempts, toolReadyAfter, selectedMask = 0, 4, 'user-mask'
local delayed = handler.applyAiMask({ photo_id = 'copy.jpg', target = 'subject', settings = { local_Exposure = 0.1 } })
assert(delayed.success and delayed.mask_id == result.mask_id and toolAttempts == 4,
    'must retry until masking becomes ready after three unsuccessful opens')
assert(created == 2 and masks[result.mask_id].local_Exposure == 0.1)
assertUserUnchanged()

-- Exhausting the opening budget must fail before touching any local control.
toolAttempts, toolReadyAfter, selectedMask = 0, 13, 'user-mask'
local beforeAccesses = #accesses
local toolOk, toolError = pcall(handler.applyAiMask, {
    photo_id = 'copy.jpg', target = 'subject', settings = { local_Exposure = -1.58 } })
assert(not toolOk and tostring(toolError):find('Masking tool did not open; selected tool=crop', 1, true))
assert(toolAttempts == 12, 'must stop after twelve unsuccessful opens')
assert(#accesses == beforeAccesses and created == 2 and masks[result.mask_id].local_Exposure == 0.1,
    'tool readiness failure must not modify or create masks')
assertUserUnchanged()
assert(selectedPhoto == previousPhoto and currentModule == 'library')

-- A failed open must release ownership of the operation for the next request.
toolAttempts, toolReadyAfter = 0, 1
local reset = handler.applyAiMask({ photo_id = 'copy.jpg', target = 'subject', settings = {}, reset = true })
assert(reset.success and masks[result.mask_id] == nil)
assertUserUnchanged()
noTarget = true
local skipped = handler.applyAiMask({ photo_id = 'copy.jpg', target = 'sky', settings = { local_Exposure = 0.1 } })
assert(skipped.skipped and not skipped.success and skipped.reason == 'no_mask_created')
assertUserUnchanged()
assert(created == 2)
assert(selectedPhoto == previousPhoto and currentModule == 'library')
print('native masking contract checks passed')
