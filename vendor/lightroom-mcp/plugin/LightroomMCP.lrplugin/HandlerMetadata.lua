local LrApplication = import 'LrApplication'

local PhotoLookup = require 'PhotoLookup'
local Log = require 'Log'

local MetadataHandler = {}

-- Return the group only when it carries at least one value, so empty IPTC
-- sections are omitted from the response (consistent with the gps group)
-- instead of surfacing a table full of empty strings.
local function nonEmptyGroup(fields)
    for _, v in pairs(fields) do
        if v ~= nil and v ~= "" then return fields end
    end
    return nil
end

local function hslDevelopSettings(developSettings)
    return nonEmptyGroup({
        SaturationAdjustmentRed = developSettings.SaturationAdjustmentRed,
        SaturationAdjustmentOrange = developSettings.SaturationAdjustmentOrange,
        SaturationAdjustmentYellow = developSettings.SaturationAdjustmentYellow,
        SaturationAdjustmentGreen = developSettings.SaturationAdjustmentGreen,
        SaturationAdjustmentAqua = developSettings.SaturationAdjustmentAqua,
        SaturationAdjustmentBlue = developSettings.SaturationAdjustmentBlue,
        SaturationAdjustmentPurple = developSettings.SaturationAdjustmentPurple,
        SaturationAdjustmentMagenta = developSettings.SaturationAdjustmentMagenta,
        HueAdjustmentRed = developSettings.HueAdjustmentRed,
        HueAdjustmentOrange = developSettings.HueAdjustmentOrange,
        HueAdjustmentYellow = developSettings.HueAdjustmentYellow,
        HueAdjustmentGreen = developSettings.HueAdjustmentGreen,
        HueAdjustmentAqua = developSettings.HueAdjustmentAqua,
        HueAdjustmentBlue = developSettings.HueAdjustmentBlue,
        HueAdjustmentPurple = developSettings.HueAdjustmentPurple,
        HueAdjustmentMagenta = developSettings.HueAdjustmentMagenta,
        LuminanceAdjustmentRed = developSettings.LuminanceAdjustmentRed,
        LuminanceAdjustmentOrange = developSettings.LuminanceAdjustmentOrange,
        LuminanceAdjustmentYellow = developSettings.LuminanceAdjustmentYellow,
        LuminanceAdjustmentGreen = developSettings.LuminanceAdjustmentGreen,
        LuminanceAdjustmentAqua = developSettings.LuminanceAdjustmentAqua,
        LuminanceAdjustmentBlue = developSettings.LuminanceAdjustmentBlue,
        LuminanceAdjustmentPurple = developSettings.LuminanceAdjustmentPurple,
        LuminanceAdjustmentMagenta = developSettings.LuminanceAdjustmentMagenta,
    })
end

local function validWhiteBalance(temperature, tint)
    return type(temperature) == 'number' and temperature == temperature
        and temperature >= 2000 and temperature <= 50000
        and type(tint) == 'number' and tint == tint and tint >= -150 and tint <= 150
end

local function completeWhiteBalanceReadback(catalog, photoData)
    local settings = photoData.developSettings
    local readback = { source = 'LrPhoto.getDevelopSettings', resolved = false }
    settings.whiteBalanceReadback = readback
    if validWhiteBalance(settings.temperature, settings.tint) then
        readback.resolved = true
        return
    end

    -- getDevelopSettings may omit computed As Shot Temperature/Tint. Adobe's
    -- supported fallback is getValue in Develop, which addresses ONLY the
    -- active photo. Never switch selection during an ordinary metadata read,
    -- or borrow a different photo's values. See Adobe SDK issue 664106.
    -- Run outside the catalog read gate because controller/UI calls may yield.
    local available, tasks, view, controller = pcall(function()
        return import 'LrTasks', import 'LrApplicationView', import 'LrDevelopController'
    end)
    if not available then
        readback.reason = 'develop_readback_unavailable'
        return
    end
    local ok, values = tasks.pcall(function()
        local active = catalog:getTargetPhoto()
        if view.getCurrentModuleName() ~= 'develop' or not active
            or tostring(active.localIdentifier) ~= tostring(photoData.id) then
            return { reason = 'photo_not_active_in_develop' }
        end
        local temperature = controller.getValue('Temperature')
        local tint = controller.getValue('Tint')
        local stillActive = catalog:getTargetPhoto()
        if view.getCurrentModuleName() ~= 'develop' or not stillActive
            or tostring(stillActive.localIdentifier) ~= tostring(photoData.id) then
            return { reason = 'develop_target_changed' }
        end
        if not validWhiteBalance(temperature, tint) then
            return { reason = 'numeric_kelvin_white_balance_unavailable' }
        end
        return { temperature = temperature, tint = tint }
    end)
    if not ok then
        readback.reason = 'develop_readback_failed'
        return
    end
    if values.reason then
        readback.reason = values.reason
        return
    end
    settings.temperature, settings.tint = values.temperature, values.tint
    settings.sdk.Temperature, settings.sdk.Tint = values.temperature, values.tint
    readback.source = 'LrDevelopController.getValue'
    readback.resolved = true
end

function MetadataHandler.getPhotoMetadata(args)
    if not args.photo_id then
        error("photo_id is required")
    end

    local catalog = LrApplication.activeCatalog()
    local photoData = nil

    catalog:withReadAccessDo(function()
        local photo = PhotoLookup.resolveOne(catalog, args.photo_id)

        if not photo then
            error("Photo not found: " .. args.photo_id)
        end

        -- Get keywords
        local keywords = {}
        local photoKeywords = photo:getRawMetadata('keywords')
        if photoKeywords then
            for _, kw in ipairs(photoKeywords) do
                table.insert(keywords, kw:getName())
            end
        end

        -- Get develop settings
        local developSettings = photo:getDevelopSettings()

        -- GPS is a raw {latitude, longitude} table; omit the group when absent
        -- or when present but carrying no coordinates.
        local gps = photo:getRawMetadata('gps')
        local gpsData = nil
        if gps then
            gpsData = nonEmptyGroup({
                latitude = gps.latitude,
                longitude = gps.longitude,
                altitude = photo:getRawMetadata('gpsAltitude'),
            })
        end

        photoData = {
            id = photo.localIdentifier,
            path = photo:getRawMetadata('path'),
            filename = photo:getFormattedMetadata('fileName'),
            rating = photo:getRawMetadata('rating'),
            colorLabel = photo:getRawMetadata('colorNameForLabel'),
            pickStatus = photo:getRawMetadata('pickStatus'),
            keywords = keywords,
            -- Title / caption / headline (IPTC content description).
            title = photo:getFormattedMetadata('title'),
            caption = photo:getFormattedMetadata('caption'),
            headline = photo:getFormattedMetadata('headline'),
            -- EXIF capture data.
            dateTimeOriginal = photo:getFormattedMetadata('dateTimeOriginal'),
            dateTimeDigitized = photo:getFormattedMetadata('dateTimeDigitized'),
            cameraMake = photo:getFormattedMetadata('cameraMake'),
            cameraModel = photo:getFormattedMetadata('cameraModel'),
            cameraSerialNumber = photo:getFormattedMetadata('cameraSerialNumber'),
            lens = photo:getFormattedMetadata('lens'),
            isoSpeedRating = photo:getFormattedMetadata('isoSpeedRating'),
            focalLength = photo:getFormattedMetadata('focalLength'),
            focalLength35mm = photo:getFormattedMetadata('focalLength35mm'),
            aperture = photo:getFormattedMetadata('aperture'),
            shutterSpeed = photo:getFormattedMetadata('shutterSpeed'),
            exposureBias = photo:getFormattedMetadata('exposureBias'),
            exposureProgram = photo:getFormattedMetadata('exposureProgram'),
            meteringMode = photo:getFormattedMetadata('meteringMode'),
            flash = photo:getFormattedMetadata('flash'),
            dimensions = photo:getFormattedMetadata('dimensions'),
            fileSize = photo:getFormattedMetadata('fileSize'),
            fileFormat = photo:getRawMetadata('fileFormat'),
            artist = photo:getFormattedMetadata('artist'),
            software = photo:getFormattedMetadata('software'),
            gps = gpsData,
            -- IPTC location ("Sublocation" is the SDK `location` field).
            location = nonEmptyGroup({
                sublocation = photo:getFormattedMetadata('location'),
                city = photo:getFormattedMetadata('city'),
                stateProvince = photo:getFormattedMetadata('stateProvince'),
                country = photo:getFormattedMetadata('country'),
                isoCountryCode = photo:getFormattedMetadata('isoCountryCode'),
            }),
            copyright = nonEmptyGroup({
                creator = photo:getFormattedMetadata('creator'),
                notice = photo:getFormattedMetadata('copyright'),
                status = photo:getFormattedMetadata('copyrightState'),
                rightsUsageTerms = photo:getFormattedMetadata('rightsUsageTerms'),
            }),
            developSettings = {
                sdk = require('HandlerDevelop').readAllowedSettings(developSettings),
                whiteBalance = developSettings.WhiteBalance,
                temperature = developSettings.Temperature,
                tint = developSettings.Tint,
                exposure = developSettings.Exposure2012,
                contrast = developSettings.Contrast2012,
                highlights = developSettings.Highlights2012,
                shadows = developSettings.Shadows2012,
                whites = developSettings.Whites2012,
                blacks = developSettings.Blacks2012,
                texture = developSettings.Texture,
                clarity = developSettings.Clarity2012,
                dehaze = developSettings.Dehaze,
                vibrance = developSettings.Vibrance,
                saturation = developSettings.Saturation,
                hsl = hslDevelopSettings(developSettings),
                convertToGrayscale = developSettings.ConvertToGrayscale,
                toneCurveName = developSettings.ToneCurveName2012,
                toneCurve = developSettings.ToneCurvePV2012,
                toneCurveRed = developSettings.ToneCurvePV2012Red,
                toneCurveGreen = developSettings.ToneCurvePV2012Green,
                toneCurveBlue = developSettings.ToneCurvePV2012Blue,
                parametricShadows = developSettings.ParametricShadows,
                parametricDarks = developSettings.ParametricDarks,
                parametricLights = developSettings.ParametricLights,
                parametricHighlights = developSettings.ParametricHighlights,
            }
        }
    end)

    completeWhiteBalanceReadback(catalog, photoData)
    Log.info("Retrieved metadata for photo: " .. args.photo_id)

    return photoData
end

return MetadataHandler
