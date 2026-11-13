-- Standalone SDK-mock regression, no Busted dependency.
package.path = 'plugin/spec/?.lua;plugin/LightroomMCP.lrplugin/?.lua;' .. package.path
local helper = require 'spec_helper'
local checked = 0

local function read(opts)
    opts = opts or {}
    local original = opts.settings or { WhiteBalance = 'As Shot' }
    local photo = helper.fakePhoto({ id = 'target', path = '/study/source.CR3', developSettings = original })
    local catalog = helper.fakeCatalog({ photos = { photo } })
    local reads, targetReads = 0, 0
    local inReadGate = false
    local normalRead = catalog.withReadAccessDo
    catalog.withReadAccessDo = function(self, callback)
        inReadGate = true
        normalRead(self, callback)
        inReadGate = false
    end
    catalog.getTargetPhoto = function()
        assert(not inReadGate, 'Controller read must occur outside catalog read access')
        targetReads = targetReads + 1
        if opts.otherTarget or (opts.targetChanges and targetReads > 1) then
            return { localIdentifier = 'different-photo' }
        end
        return photo
    end
    helper.installImport({
        LrApplication = { activeCatalog = function() return catalog end },
        LrLogger = helper.defaultLrLogger(),
        LrFileUtils = {}, LrPathUtils = {},
        LrTasks = { pcall = pcall },
        LrApplicationView = { getCurrentModuleName = function() return opts.module or 'develop' end },
        LrDevelopController = {
            getValue = function(key)
                assert(not inReadGate)
                reads = reads + 1
                if opts.failure then error('Develop not ready') end
                if key == 'Temperature' then return opts.temperature or 5950 end
                if key == 'Tint' then return opts.tint or 18 end
                error('unexpected control ' .. key)
            end,
        },
    })
    package.loaded.HandlerDevelop = nil
    package.loaded.HandlerMetadata = nil
    local result = require('HandlerMetadata').getPhotoMetadata({ photo_id = 'target' })
    assert(#photo:getWrites() == 0, 'Metadata read must not mutate photo settings')
    checked = checked + 1
    return result.developSettings, reads, original
end

local known, count = read({ settings = { WhiteBalance = 'Custom', Temperature = 6100, Tint = 12 } })
assert(count == 0 and known.temperature == 6100 and known.tint == 12)
assert(known.whiteBalanceReadback.source == 'LrPhoto.getDevelopSettings')

local resolved, _, original = read()
assert(resolved.whiteBalance == 'As Shot')
assert(resolved.temperature == 5950 and resolved.tint == 18)
assert(resolved.sdk.Temperature == 5950 and resolved.sdk.Tint == 18)
assert(resolved.whiteBalanceReadback.source == 'LrDevelopController.getValue')
assert(resolved.whiteBalanceReadback.resolved == true)
assert(original.Temperature == nil and original.Tint == nil)

local partial = read({ settings = { WhiteBalance = 'As Shot', Temperature = 5000 } })
assert(partial.temperature == 5950 and partial.tint == 18, 'Read the complete live pair')

local other, otherCount = read({ otherTarget = true })
assert(otherCount == 0 and other.temperature == nil and other.tint == nil)
assert(other.whiteBalanceReadback.reason == 'photo_not_active_in_develop')

local library, libraryCount = read({ module = 'library' })
assert(libraryCount == 0 and library.temperature == nil)
assert(library.whiteBalanceReadback.resolved == false)

local changed = read({ targetChanges = true })
assert(changed.temperature == nil and changed.tint == nil)
assert(changed.whiteBalanceReadback.reason == 'develop_target_changed')

local failed = read({ failure = true })
assert(failed.temperature == nil and failed.tint == nil)
assert(failed.whiteBalanceReadback.reason == 'develop_readback_failed')

local nonKelvin = read({ temperature = 0 })
assert(nonKelvin.temperature == nil and nonKelvin.tint == nil)
assert(nonKelvin.whiteBalanceReadback.reason == 'numeric_kelvin_white_balance_unavailable')

local invalid = read({ tint = math.huge })
assert(invalid.temperature == nil and invalid.tint == nil)
assert(invalid.whiteBalanceReadback.resolved == false)

print('White balance readback: ' .. tostring(checked) .. ' SDK-mock cases passed')
