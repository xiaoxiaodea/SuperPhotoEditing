local LrApplication = import 'LrApplication'

local Validate = require 'Validate'
local Log = require 'Log'

local SearchHandler = {}

local function parseDate(value, name)
    if type(value) ~= "string" then
        error(name .. " must be a string in YYYY-MM-DD format", 0)
    end

    local year, month, day = value:match("^(%d%d%d%d)%-(%d%d)%-(%d%d)$")
    if not year then
        error(name .. " must use YYYY-MM-DD format", 0)
    end

    year = tonumber(year)
    month = tonumber(month)
    day = tonumber(day)

    if month < 1 or month > 12 then
        error(name .. " must use a valid calendar date", 0)
    end

    local timestamp = os.time({ year = year, month = month, day = day, hour = 12 })
    local normalized = os.date("*t", timestamp)
    if normalized.year ~= year or normalized.month ~= month or normalized.day ~= day then
        error(name .. " must use a valid calendar date", 0)
    end

    return value, timestamp
end

local function addDays(timestamp, days)
    return os.date("%Y-%m-%d", timestamp + (days * 86400))
end

-- Criteria enums accepted by catalog:findPhotos, per the Lightroom SDK
-- reference. Values outside these sets are rejected by Lightroom with an opaque
-- internal error, so validate here to give the caller a usable message.
local FILE_FORMATS = { "DNG", "RAW", "JPG", "TIFF", "PNG", "PSD", "VIDEO" }
local ASPECT_RATIOS = { "portrait", "landscape", "square" }
local TREATMENTS = { "grayscale", "color" }
local FLAGS = { "pick", "unflag", "reject" }
local PICK_VALUE_BY_FLAG = { pick = 1, unflag = 0, reject = -1 }
local LABEL_COLOR_VALUES = {
    red = 1, yellow = 2, green = 3, blue = 4, purple = 5,
}
local LABEL_COLORS = { "red", "yellow", "green", "blue", "purple", "none" }
local TEXT_OPERATIONS = { "any", "all", "words", "noneOf", "beginsWith", "endsWith" }
local NUMERIC_OPERATIONS = { "==", "!=", ">", "<", ">=", "<=" }
local RELATIVE_TIME_UNITS = { "hours", "days", "weeks", "months", "years" }
local SORT_KEYS = { "captureTime", "fileName", "extension" }

SearchHandler.FILE_FORMATS = FILE_FORMATS
SearchHandler.ASPECT_RATIOS = ASPECT_RATIOS
SearchHandler.TREATMENTS = TREATMENTS
SearchHandler.FLAGS = FLAGS
SearchHandler.LABEL_COLORS = LABEL_COLORS
SearchHandler.TEXT_OPERATIONS = TEXT_OPERATIONS
SearchHandler.NUMERIC_OPERATIONS = NUMERIC_OPERATIONS
SearchHandler.RELATIVE_TIME_UNITS = RELATIVE_TIME_UNITS

-- Build a findPhotos search descriptor from the tool's filter vocabulary.
--
-- Exported (rather than kept local) so create_smart_collection can persist the
-- exact same descriptor semantics that search_photos evaluates ad hoc. Two
-- separate translations of "rating >= 3" would inevitably drift.
function SearchHandler.buildSearchDesc(args)
    args = args or {}
    local desc = { combine = "intersect" }

    -- Validate modifier params up front, before the paired-filter checks below.
    --
    -- These only take effect alongside their base filter, so validating them
    -- inline would let two mistakes through silently: a typo'd operation
    -- (rating_operation="=~" with no rating) and a dangling modifier the caller
    -- believes is filtering. Both would return an unfiltered result set that
    -- looks like a successful search, which is worse than an error.
    local MODIFIERS = {
        { key = "rating_operation", requires = "rating", allowed = NUMERIC_OPERATIONS },
        { key = "iso_operation", requires = "iso", allowed = NUMERIC_OPERATIONS },
        { key = "text_operation", requires = "text", allowed = TEXT_OPERATIONS },
        { key = "captured_in_last_units", requires = "captured_in_last", allowed = RELATIVE_TIME_UNITS },
        { key = "edited_in_last_units", requires = "edited_in_last", allowed = RELATIVE_TIME_UNITS },
    }
    for _, modifier in ipairs(MODIFIERS) do
        if args[modifier.key] ~= nil then
            Validate.requireOneOf(args[modifier.key], modifier.key, modifier.allowed)
            if args[modifier.requires] == nil then
                error(modifier.key .. " requires " .. modifier.requires, 0)
            end
        end
    end

    if args.filename then
        table.insert(desc, { criteria = "filename", operation = "any", value = args.filename })
    end

    -- Free-text search across everything Lightroom indexes. `text_operation`
    -- exposes contains / starts-with / excludes rather than hardcoding one.
    if args.text then
        local operation = "any"
        if args.text_operation ~= nil then
            operation = Validate.requireOneOf(args.text_operation, "text_operation", TEXT_OPERATIONS)
        end
        table.insert(desc, { criteria = "all", operation = operation, value = args.text })
    end

    if args.rating ~= nil then
        -- rating_operation defaults to "==" to preserve the previous behaviour
        -- of an exact-match rating filter.
        local operation = "=="
        if args.rating_operation ~= nil then
            operation = Validate.requireOneOf(
                args.rating_operation, "rating_operation", NUMERIC_OPERATIONS)
        end
        table.insert(desc, { criteria = "rating", operation = operation, value = args.rating })
    end

    if args.keywords and #args.keywords > 0 then
        for _, kw in ipairs(args.keywords) do
            table.insert(desc, { criteria = "keywords", operation = "all", value = kw })
        end
    end

    if args.flag ~= nil then
        local flag = Validate.requireOneOf(args.flag, "flag", FLAGS)
        table.insert(desc, { criteria = "pick", operation = "==", value = PICK_VALUE_BY_FLAG[flag] })
    end

    if args.color_label ~= nil then
        local color = Validate.requireOneOf(args.color_label, "color_label", LABEL_COLORS)
        -- "none" is its own sentinel in the SDK, not a numeric colour index.
        local value = (color == "none") and "none" or LABEL_COLOR_VALUES[color]
        table.insert(desc, { criteria = "labelColor", operation = "==", value = value })
    end

    if args.file_format ~= nil then
        local format = Validate.requireOneOf(args.file_format, "file_format", FILE_FORMATS)
        table.insert(desc, { criteria = "fileFormat", operation = "==", value = format })
    end

    if args.camera ~= nil then
        table.insert(desc, { criteria = "camera", operation = "any", value = args.camera })
    end

    if args.lens ~= nil then
        table.insert(desc, { criteria = "lens", operation = "any", value = args.lens })
    end

    if args.folder ~= nil then
        table.insert(desc, { criteria = "folder", operation = "any", value = args.folder })
    end

    if args.collection ~= nil then
        table.insert(desc, { criteria = "collection", operation = "any", value = args.collection })
    end

    if args.city ~= nil then
        table.insert(desc, { criteria = "city", operation = "any", value = args.city })
    end

    if args.country ~= nil then
        table.insert(desc, { criteria = "country", operation = "any", value = args.country })
    end

    if args.iso ~= nil then
        local operation = "=="
        if args.iso_operation ~= nil then
            operation = Validate.requireOneOf(
                args.iso_operation, "iso_operation", NUMERIC_OPERATIONS)
        end
        table.insert(desc, { criteria = "isoSpeedRating", operation = operation, value = args.iso })
    end

    -- Boolean criteria use isTrue/isFalse operations, not value comparison.
    if args.has_gps ~= nil then
        table.insert(desc, {
            criteria = "hasGPSData",
            operation = args.has_gps and "isTrue" or "isFalse",
        })
    end

    if args.has_adjustments ~= nil then
        table.insert(desc, {
            criteria = "hasAdjustments",
            operation = args.has_adjustments and "isTrue" or "isFalse",
        })
    end

    if args.is_cropped ~= nil then
        table.insert(desc, {
            criteria = "cropped",
            operation = args.is_cropped and "isTrue" or "isFalse",
        })
    end

    if args.aspect_ratio ~= nil then
        local ratio = Validate.requireOneOf(args.aspect_ratio, "aspect_ratio", ASPECT_RATIOS)
        table.insert(desc, { criteria = "aspectRatio", operation = "==", value = ratio })
    end

    if args.treatment ~= nil then
        local treatment = Validate.requireOneOf(args.treatment, "treatment", TREATMENTS)
        table.insert(desc, { criteria = "treatment", operation = "==", value = treatment })
    end

    -- Relative capture window ("last 30 days"), which is what people actually
    -- ask for and previously required computing absolute dates client-side.
    if args.captured_in_last ~= nil then
        local units = "days"
        if args.captured_in_last_units ~= nil then
            units = Validate.requireOneOf(args.captured_in_last_units, "captured_in_last_units",
                RELATIVE_TIME_UNITS)
        end
        table.insert(desc, {
            criteria = "captureTime",
            operation = "inLast",
            value = args.captured_in_last,
            value2 = 1,
            value_units = units,
        })
    end

    if args.edited_in_last ~= nil then
        local units = "days"
        if args.edited_in_last_units ~= nil then
            units = Validate.requireOneOf(args.edited_in_last_units, "edited_in_last_units",
                RELATIVE_TIME_UNITS)
        end
        table.insert(desc, {
            criteria = "touchTime",
            operation = "inLast",
            value = args.edited_in_last,
            value2 = 1,
            value_units = units,
        })
    end

    if args.start_date and args.end_date then
        local startDate, startTimestamp = parseDate(args.start_date, "start_date")
        local endDate, endTimestamp = parseDate(args.end_date, "end_date")
        if startTimestamp > endTimestamp then
            error("start_date must be on or before end_date", 0)
        end
        table.insert(desc, {
            criteria = "captureTime",
            operation = "in",
            value = startDate,
            value2 = endDate,
        })
    elseif args.start_date then
        local _, startTimestamp = parseDate(args.start_date, "start_date")
        table.insert(desc, { criteria = "captureTime", operation = ">", value = addDays(startTimestamp, -1) })
    elseif args.end_date then
        local _, endTimestamp = parseDate(args.end_date, "end_date")
        table.insert(desc, { criteria = "captureTime", operation = "<", value = addDays(endTimestamp, 1) })
    end

    return desc
end

local function buildResult(photo)
    return {
        id = photo.localIdentifier,
        path = photo:getRawMetadata('path'),
        filename = photo:getFormattedMetadata('fileName'),
        rating = photo:getRawMetadata('rating'),
        dateTimeOriginal = photo:getFormattedMetadata('dateTimeOriginal'),
    }
end

function SearchHandler.searchPhotos(args)
    args = args or {}
    local catalog = LrApplication.activeCatalog()
    local results = {}
    local searchDesc = SearchHandler.buildSearchDesc(args)
    local hasFilters = #searchDesc > 0

    if args.match == "any" then
        searchDesc.combine = "union"
    end

    local limit, offset = Validate.pagination(args)

    local sort = nil
    if args.sort ~= nil then
        sort = Validate.requireOneOf(args.sort, "sort", SORT_KEYS)
    end

    -- Run the catalog query OUTSIDE withReadAccessDo. findPhotos() runs an
    -- async catalog search that yields; called from inside the read gate on
    -- Windows it deadlocks -- the task never returns and never releases the
    -- gate, wedging the whole bridge until the 30s server timeout (issue #124,
    -- same root cause as #134's getTargetPhotos). Only the per-photo metadata
    -- reads need the gate. getAllPhotos() (no-filter path) is a non-yielding
    -- enumeration, but is hoisted out too for a single, consistent structure.
    Log.info(string.format("searchPhotos: querying (hasFilters=%s)", tostring(hasFilters)))
    -- Unrated photos have nil rating; rating>=0 excludes them, so getAllPhotos()
    -- must be used when no filters are specified.
    local matches
    if hasFilters then
        local query = { searchDesc = searchDesc }
        if sort then
            query.sort = sort
            query.ascending = args.ascending == true
        end
        matches = catalog:findPhotos(query)
    else
        matches = catalog:getAllPhotos()
    end
    matches = matches or {}

    local total = #matches
    Log.info(string.format("searchPhotos: query returned %d", total))

    local last = math.min(offset + limit, total)
    catalog:withReadAccessDo(function()
        for i = offset + 1, last do
            table.insert(results, buildResult(matches[i]))
        end
    end)

    Log.info(string.format("Search matched %d photos, returning %d (offset=%d, limit=%d)",
        total, #results, offset, limit))

    local response = {
        count = total,
        photos = results,
        has_more = (offset + #results) < total,
    }

    if not hasFilters then
        response.warning = "No filters applied — scanned full catalog. Provide filename, keywords, rating, or date filters to narrow results and improve performance."
    end

    return response
end

return SearchHandler
