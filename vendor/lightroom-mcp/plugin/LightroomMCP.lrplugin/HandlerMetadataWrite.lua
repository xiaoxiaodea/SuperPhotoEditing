-- Metadata writes: IPTC text, GPS, copyright, color label, flag status.
--
-- The read side lives in HandlerMetadata; this module is the missing write half.
-- Every key here is validated against WRITABLE_KEYS before it reaches
-- setRawMetadata, because the SDK THROWS on an unrecognised key and that
-- exception aborts the enclosing withWriteAccessDo, rolling back the whole
-- batch. An allowlist turns "agent invented a plausible field name" into a
-- clear per-field error instead of a mass rollback.
--
-- Key set and value types come from LrPhoto:setRawMetadata in the Lightroom SDK
-- reference. Note the asymmetry the read side hides: the color label is READ via
-- 'colorNameForLabel' but WRITTEN via either 'label' or 'colorNameForLabel';
-- flag status is 'pickStatus' both ways.

local LrApplication = import 'LrApplication'

local PhotoLookup = require 'PhotoLookup'
local Validate = require 'Validate'
local Log = require 'Log'

local MetadataWriteHandler = {}

-- Free-text IPTC/XMP fields. All take a string; "" clears the field.
local TEXT_KEYS = {
    "title",
    "caption",
    "headline",
    "copyName",
    "creator",
    "creatorJobTitle",
    "creatorAddress",
    "creatorCity",
    "creatorStateProvince",
    "creatorPostalCode",
    "creatorCountry",
    "creatorPhone",
    "creatorEmail",
    "creatorUrl",
    "iptcSubjectCode",
    "descriptionWriter",
    "dateCreated",
    "intellectualGenre",
    "scene",
    "location",
    "city",
    "stateProvince",
    "country",
    "isoCountryCode",
    "jobIdentifier",
    "instructions",
    "provider",
    "source",
    "copyright",
    "rightsUsageTerms",
    "copyrightInfoUrl",
    "personShown",
    "nameOfOrgShown",
    "codeOfOrgShown",
    "event",
    "additionalModelInfo",
    "modelAge",
    "minorModelAge",
    "modelReleaseStatus",
    "modelReleaseID",
    "propertyReleaseID",
    "propertyReleaseStatus",
    "sourceType",
}

-- copyrightState is an enum in the SDK, not free text.
local COPYRIGHT_STATES = { "unknown", "copyrighted", "public domain" }

-- Lightroom's five canonical label colours plus 'none' to clear. Any other
-- string is accepted by the SDK but renders as a white swatch, which is
-- indistinguishable from corruption in the UI — so reject it here.
local LABEL_COLORS = { "red", "yellow", "green", "blue", "purple", "none" }

local WRITABLE_KEYS = {}
for _, key in ipairs(TEXT_KEYS) do
    WRITABLE_KEYS[key] = "string"
end
WRITABLE_KEYS.copyrightState = "copyrightState"
WRITABLE_KEYS.colorNameForLabel = "labelColor"
WRITABLE_KEYS.pickStatus = "pickStatus"
WRITABLE_KEYS.gps = "gps"
WRITABLE_KEYS.gpsAltitude = "number"

MetadataWriteHandler.TEXT_KEYS = TEXT_KEYS
MetadataWriteHandler.COPYRIGHT_STATES = COPYRIGHT_STATES
MetadataWriteHandler.LABEL_COLORS = LABEL_COLORS
MetadataWriteHandler.WRITABLE_KEYS = WRITABLE_KEYS

local function coerceGps(value)
    if type(value) ~= "table" then
        error("gps must be an object with latitude and longitude")
    end
    local latitude = Validate.requireNumber(value.latitude, "gps.latitude")
    local longitude = Validate.requireNumber(value.longitude, "gps.longitude")
    if latitude < -90 or latitude > 90 then
        error("gps.latitude must be between -90 and 90")
    end
    if longitude < -180 or longitude > 180 then
        error("gps.longitude must be between -180 and 180")
    end
    return { latitude = latitude, longitude = longitude }
end

-- Normalise one incoming field to the exact type setRawMetadata expects.
-- Returns the SDK key plus the coerced value; the key can differ from the
-- caller-facing key (see the label note at the top of this file).
local function coerceValue(key, value)
    local kind = WRITABLE_KEYS[key]
    if not kind then
        error("Unsupported metadata key: " .. tostring(key))
    end

    if kind == "string" then
        if type(value) ~= "string" then
            error(key .. " must be a string")
        end
        return key, value
    end

    if kind == "copyrightState" then
        return key, Validate.requireOneOf(value, "copyrightState", COPYRIGHT_STATES)
    end

    if kind == "labelColor" then
        if type(value) ~= "string" then
            error("colorNameForLabel must be a string")
        end
        local normalized = value:lower()
        Validate.requireOneOf(normalized, "colorNameForLabel", LABEL_COLORS)
        -- 'none' clears the label. setRawMetadata('label', '') is the documented
        -- way to clear; passing the literal string "none" would set a label
        -- actually named "none" and show a white swatch.
        if normalized == "none" then
            return "label", ""
        end
        return "label", normalized
    end

    if kind == "pickStatus" then
        local number = Validate.requireNumber(value, "pickStatus")
        if number ~= 1 and number ~= 0 and number ~= -1 then
            error("pickStatus must be 1 (flagged), 0 (unflagged), or -1 (rejected)")
        end
        return key, number
    end

    if kind == "gps" then
        -- json null decodes to nil and vanishes from the table, so an explicit
        -- false is the only way a caller can ask to unset GPS.
        if value == false then
            return key, nil, true
        end
        return key, coerceGps(value)
    end

    if kind == "number" then
        return key, Validate.requireNumber(value, key)
    end

    error("Unsupported metadata key: " .. tostring(key))
end

function MetadataWriteHandler.setPhotoMetadata(args)
    args = args or {}
    Validate.requirePhotoIds(args.photo_ids)

    if type(args.metadata) ~= "table" then
        error("metadata is required")
    end

    -- Validate and coerce EVERY field before opening the write gate. A bad key
    -- discovered halfway through the batch would otherwise roll back the photos
    -- already written in this transaction, so the tool would report an error
    -- after partially-applied work.
    local writes = {}
    local requestedKeys = {}
    for key, value in pairs(args.metadata) do
        if type(key) ~= "string" then
            error("metadata keys must be strings")
        end
        local sdkKey, coerced, unset = coerceValue(key, value)
        table.insert(writes, { key = sdkKey, value = coerced, unset = unset })
        table.insert(requestedKeys, key)
    end

    if #writes == 0 then
        error("metadata is required")
    end
    table.sort(requestedKeys)

    local catalog = LrApplication.activeCatalog()
    local updatedCount = 0
    local missing = {}

    catalog:withWriteAccessDo("Set Photo Metadata", function()
        local resolved = PhotoLookup.resolveMany(catalog, args.photo_ids)
        for _, entry in ipairs(resolved) do
            if entry.photo then
                for _, write in ipairs(writes) do
                    -- nil is a meaningful value here (unset GPS), so the unset
                    -- flag rather than the value decides the call.
                    if write.unset then
                        entry.photo:setRawMetadata(write.key, nil)
                    else
                        entry.photo:setRawMetadata(write.key, write.value)
                    end
                end
                updatedCount = updatedCount + 1
            else
                table.insert(missing, entry.id)
            end
        end
    end)

    Log.info(string.format("Set %d metadata fields on %d photos",
        #writes, updatedCount))

    return {
        success = true,
        updated = updatedCount,
        fields = requestedKeys,
        not_found = missing,
        message = string.format("Updated %d metadata field(s) on %d photo(s)",
            #writes, updatedCount),
    }
end

-- Dedicated flag tool. set_photo_metadata can do this via pickStatus, but
-- picking/rejecting during a cull is frequent enough to deserve a tool whose
-- vocabulary matches the UI ("pick"/"reject"/"unflag") instead of 1/0/-1.
local PICK_STATUS_BY_FLAG = {
    pick = 1,
    unflag = 0,
    reject = -1,
}

function MetadataWriteHandler.setFlag(args)
    args = args or {}
    Validate.requirePhotoIds(args.photo_ids)
    local flag = Validate.requireOneOf(args.flag, "flag", { "pick", "unflag", "reject" })
    local pickStatus = PICK_STATUS_BY_FLAG[flag]

    local catalog = LrApplication.activeCatalog()
    local updatedCount = 0
    local missing = {}

    catalog:withWriteAccessDo("Set Flag", function()
        local resolved = PhotoLookup.resolveMany(catalog, args.photo_ids)
        for _, entry in ipairs(resolved) do
            if entry.photo then
                entry.photo:setRawMetadata('pickStatus', pickStatus)
                updatedCount = updatedCount + 1
            else
                table.insert(missing, entry.id)
            end
        end
    end)

    Log.info(string.format("Set flag %s on %d photos", flag, updatedCount))

    return {
        success = true,
        updated = updatedCount,
        flag = flag,
        pick_status = pickStatus,
        not_found = missing,
        message = string.format("Set flag %s on %d photo(s)", flag, updatedCount),
    }
end

function MetadataWriteHandler.setColorLabel(args)
    args = args or {}
    Validate.requirePhotoIds(args.photo_ids)
    local sdkKey, value = coerceValue("colorNameForLabel", args.color)

    local catalog = LrApplication.activeCatalog()
    local updatedCount = 0
    local missing = {}

    catalog:withWriteAccessDo("Set Color Label", function()
        local resolved = PhotoLookup.resolveMany(catalog, args.photo_ids)
        for _, entry in ipairs(resolved) do
            if entry.photo then
                entry.photo:setRawMetadata(sdkKey, value)
                updatedCount = updatedCount + 1
            else
                table.insert(missing, entry.id)
            end
        end
    end)

    Log.info(string.format("Set color label %s on %d photos",
        tostring(args.color), updatedCount))

    return {
        success = true,
        updated = updatedCount,
        color = args.color:lower(),
        not_found = missing,
        message = string.format("Set color label %s on %d photo(s)",
            args.color:lower(), updatedCount),
    }
end

return MetadataWriteHandler
