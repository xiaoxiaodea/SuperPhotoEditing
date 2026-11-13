local LrApplication = import 'LrApplication'

local PhotoLookup = require 'PhotoLookup'
local Validate = require 'Validate'
local Log = require 'Log'

local SelectionHandler = {}

local function buildResult(photo)
    return {
        id = photo.localIdentifier,
        path = photo:getRawMetadata('path'),
        filename = photo:getFormattedMetadata('fileName'),
        rating = photo:getRawMetadata('rating'),
        dateTimeOriginal = photo:getFormattedMetadata('dateTimeOriginal'),
    }
end

function SelectionHandler.getSelectedPhotos(args)
    args = args or {}
    local catalog = LrApplication.activeCatalog()

    -- floor: the tool schema permits any number, and a fractional offset would
    -- make the page loop index matches[i] with a non-integer key (always nil)
    -- and crash buildResult(nil).
    local limit = math.floor(tonumber(args.limit) or 100)
    if limit < 0 then limit = 0 end
    local offset = math.floor(tonumber(args.offset) or 0)
    if offset < 0 then offset = 0 end

    -- Acquire the target set OUTSIDE withReadAccessDo. getTargetPhotos() reads
    -- the live view selection, which yields to the UI thread; called from
    -- inside the read gate on Windows it deadlocks -- the task never returns
    -- and never releases the gate, so the whole bridge wedges until the 30s
    -- server timeout (issues #134, #124). Only the per-photo metadata reads
    -- need the gate; getTargetPhotos() does not. This mirrors get_photo_metadata,
    -- which works because it only does a non-yielding getAllPhotos() pass.
    Log.info("getSelectedPhotos: requesting target photos")
    local matches = catalog:getTargetPhotos() or {}
    local total = #matches
    Log.info(string.format("getSelectedPhotos: getTargetPhotos returned %d", total))

    local last = math.min(offset + limit, total)
    local results = {}
    catalog:withReadAccessDo(function()
        for i = offset + 1, last do
            table.insert(results, buildResult(matches[i]))
        end
    end)

    Log.info(string.format("getSelectedPhotos: returning %d (offset=%d, limit=%d)",
        #results, offset, limit))

    return {
        count = total,
        photos = results,
        has_more = (offset + #results) < total,
    }
end

-- Write the Lightroom selection so a human can SEE what the agent picked.
--
-- Without this the loop was one-directional: the agent could read the user's
-- selection but never hand a result set back to the UI, so "show me the ten
-- you'd cull" ended as a list of numeric ids the user had to find by hand.
-- setSelectedPhotos needs no write gate (it drives the view, not the database)
-- but the id lookup is a catalog read.
function SelectionHandler.setSelectedPhotos(args)
    args = args or {}
    Validate.requirePhotoIds(args.photo_ids)

    local catalog = LrApplication.activeCatalog()
    local photos = {}
    local missing = {}

    catalog:withReadAccessDo(function()
        local resolved = PhotoLookup.resolveMany(catalog, args.photo_ids)
        for _, entry in ipairs(resolved) do
            if entry.photo then
                table.insert(photos, entry.photo)
            else
                table.insert(missing, entry.id)
            end
        end
    end)

    if #photos == 0 then
        error("No photos found to select")
    end

    -- The SDK signature is (activePhoto, otherSelectedPhotos): the first photo
    -- becomes the "most selected" one and the remainder are passed separately.
    local active = photos[1]
    local others = {}
    for index = 2, #photos do
        table.insert(others, photos[index])
    end

    catalog:setSelectedPhotos(active, others)

    Log.info(string.format("Selected %d photos in Lightroom", #photos))

    return {
        success = true,
        selected = #photos,
        active_photo_id = active.localIdentifier,
        not_found = missing,
        message = string.format("Selected %d photo(s) in Lightroom", #photos),
    }
end

return SelectionHandler
