-- Native semantic masks: each target is owned by this bridge on one job copy.
-- Never edit an arbitrary user mask or fall back to global adjustments.
local PhotoLookup = require 'PhotoLookup'
local MaskingHandler = {}
local owned = {}
local busy = false
local limits = {
    local_Exposure = { -2, 2 },
    local_Contrast = { -50, 50 }, local_Whites = { -75, 75 }, local_Blacks = { -50, 50 },
    local_Highlights = { -40, 20 }, local_Shadows = { -25, 35 },
    local_Temperature = { -12, 12 }, local_Tint = { -10, 10 },
    local_Saturation = { -25, 15 }, local_Clarity = { -15, 15 },
    local_Texture = { -20, 20 }, local_Dehaze = { -10, 15 },
}

local function maskIds(controller)
    if type(controller.getAllMasks) ~= 'function' then return nil end
    local ids = {}
    for _, mask in ipairs(controller.getAllMasks() or {}) do
        if type(mask.ID) ~= 'string' or mask.ID == '' then return nil end
        table.insert(ids, mask.ID)
    end
    table.sort(ids)
    return table.concat(ids, '|')
end

function MaskingHandler.applyAiMask(args)
    if type(args.photo_id) ~= 'string' or args.photo_id == '' then error('photo_id is required') end
    if args.target ~= 'subject' and args.target ~= 'sky' and args.target ~= 'background' then
        error('target must be subject, sky or background')
    end
    if type(args.settings) ~= 'table' then error('settings is required') end
    for key, value in pairs(args.settings) do
        local bound = limits[key]
        if not bound or type(value) ~= 'number' or value ~= value or value < bound[1] or value > bound[2] then
            error('Unsupported or out-of-range local adjustment: ' .. tostring(key))
        end
    end
    if busy then error('A native masking operation is already running') end
    local LrApplication = import 'LrApplication'
    local LrApplicationView = import 'LrApplicationView'
    local LrDevelopController = import 'LrDevelopController'
    local LrTasks = import 'LrTasks'
    local catalog = LrApplication.activeCatalog()
    local photo
    catalog:withReadAccessDo(function() photo = PhotoLookup.resolveOne(catalog, args.photo_id) end)
    if not photo then error('Photo not found: ' .. args.photo_id) end
    if type(LrDevelopController.createNewMask) ~= 'function' then error('Native AI masking requires Lightroom 11 or later') end
    local owner = tostring(photo.localIdentifier)
    owned[owner] = owned[owner] or {}
    local maskId = owned[owner][args.target]
    local previousPhoto = catalog:getTargetPhoto()
    local previousSelection = catalog:getTargetPhotos()
    local previousModule = LrApplicationView.getCurrentModuleName()
    local previousSources = catalog:getActiveSources()
    local previousFilter = catalog:getCurrentViewFilter()
    busy = true
    local created = false
    local priorValues = {}
    local ok, result = LrTasks.pcall(function()
        -- An imported job copy may be outside the current folder, collection
        -- or rating filter. Make it visible before changing the selection.
        LrApplicationView.switchToModule('library')
        catalog:setViewFilter({ filtersActive = false })
        catalog:setActiveSources({ catalog.kAllPhotos })
        LrTasks.sleep(0.3)
        catalog:setSelectedPhotos(photo, {})
        LrApplicationView.switchToModule('develop')
        local selected = false
        for _ = 1, 40 do
            LrTasks.sleep(0.2)
            local active = catalog:getTargetPhoto()
            if active and tostring(active.localIdentifier) == owner
                and LrApplicationView.getCurrentModuleName() == 'develop' then
                selected = true
                break
            end
        end
        if not selected then
            local active = catalog:getTargetPhoto()
            error('Could not select masking target ' .. owner .. '; active=' .. tostring(active and active.localIdentifier)
                  .. '; module=' .. tostring(LrApplicationView.getCurrentModuleName()))
        end
        -- Develop can report its module before the new image's controls are
        -- ready. Opening Masking during that transition silently does nothing.
        LrTasks.sleep(1.5)
        local toolReady = false
        for _ = 1, 12 do
            LrDevelopController.selectTool('masking')
            LrTasks.sleep(0.5)
            if LrDevelopController.getSelectedTool() == 'masking' then
                toolReady = true
                break
            end
        end
        if not toolReady then
            error('Masking tool did not open; selected tool=' .. tostring(LrDevelopController.getSelectedTool()))
        end
        if args.reset == true then
            if maskId then
                LrDevelopController.deleteMask(maskId)
                owned[owner][args.target] = nil
            end
            return { success = true, target = args.target, reset = true }
        end
        if maskId then
            LrDevelopController.selectMask(maskId)
            if LrDevelopController.getSelectedMask() ~= maskId then error('Owned mask is no longer present') end
            for key, _ in pairs(args.settings) do priorValues[key] = LrDevelopController.getValue(key) end
        else
            local previousId = LrDevelopController.getSelectedMask()
            local beforeIds = maskIds(LrDevelopController)
            LrDevelopController.createNewMask('aiSelection', args.target)
            for _ = 1, 360 do
                maskId = LrDevelopController.getSelectedMask()
                if maskId and maskId ~= '' and maskId ~= previousId then break end
                LrTasks.sleep(0.2)
            end
            if not maskId or maskId == '' or maskId == previousId then
                if beforeIds ~= nil and maskIds(LrDevelopController) == beforeIds then
                    -- No mask was created (e.g. no sky in a boat close-up).
                    -- No sliders have been touched; keep prior masks intact.
                    return { success = false, skipped = true, target = args.target,
                             reason = 'no_mask_created' }
                end
                local masks = require('JSON'):encode(LrDevelopController.getAllMasks() or {})
                error('AI mask creation did not complete; target=' .. args.target .. '; selected=' .. tostring(maskId)
                      .. '; process=' .. tostring(LrDevelopController.getProcessVersion()) .. '; masks=' .. masks:sub(1, 1200))
            end
            owned[owner][args.target] = maskId
            created = true
            -- Lightroom remembers the last mask sliders. Start the new mask at
            -- neutral so a saved brush preset cannot change this photograph.
            for key, _ in pairs(limits) do LrDevelopController.setValue(key, 0) end
            for _, key in ipairs({ 'local_Contrast', 'local_Whites', 'local_Blacks',
                'local_Sharpness', 'local_LuminanceNoise', 'local_Moire', 'local_Defringe',
                'local_Hue', 'local_ToningHue', 'local_ToningSaturation' }) do
                local supported, value = LrTasks.pcall(function() return LrDevelopController.getValue(key) end)
                if supported and type(value) == 'number' then LrDevelopController.setValue(key, 0) end
            end
        end
        local readback = {}
        for key, value in pairs(args.settings) do
            local minimum, maximum = LrDevelopController.getRange(key)
            if type(minimum) ~= 'number' or type(maximum) ~= 'number' then error('SDK does not expose local control ' .. key) end
            if value < minimum or value > maximum then error('SDK range differs for ' .. key) end
            LrDevelopController.setValue(key, value)
        end
        LrTasks.sleep(0.3)
        for key, value in pairs(args.settings) do
            local actual = LrDevelopController.getValue(key)
            if type(actual) ~= 'number' or math.abs(actual - value) > 0.02 then
                error('Local adjustment did not read back: ' .. key)
            end
            readback[key] = actual
        end
        return { success = true, target = args.target, mask_id = maskId, created = created, settings = readback }
    end)
    if not ok and created and maskId then
        LrTasks.pcall(function() LrDevelopController.deleteMask(maskId) end)
        owned[owner][args.target] = nil
    elseif not ok then
        LrTasks.pcall(function()
            for key, value in pairs(priorValues) do LrDevelopController.setValue(key, value) end
        end)
    end
    -- UI restoration is best effort; a failed adjustment still surfaces as an
    -- error and the caller does not claim the local stage was completed.
    LrTasks.pcall(function()
        if previousSources then catalog:setActiveSources(previousSources) end
        if previousFilter then catalog:setViewFilter(previousFilter) end
        if previousPhoto then
            local others = {}
            for _, p in ipairs(previousSelection or {}) do
                if tostring(p.localIdentifier) ~= tostring(previousPhoto.localIdentifier) then table.insert(others, p) end
            end
            catalog:setSelectedPhotos(previousPhoto, others)
        end
        if previousModule then LrApplicationView.switchToModule(previousModule) end
    end)
    busy = false
    if not ok then error(tostring(result)) end
    return result
end

return MaskingHandler
