-- Common test helpers + LR SDK mock factory.
-- Usage from a spec file:
--   local helper = require 'spec.spec_helper'
--   local catalog, photos = helper.mockCatalog({...})
--   helper.installImport({ LrApplication = { activeCatalog = function() return catalog end } })
--   local Handler = require 'HandlerSearch'

local M = {}

-- Make plugin sources requireable.
local lfs_ok = pcall(function() return require 'lfs' end)
local sep = package.config:sub(1, 1)
local pluginRoot = "plugin" .. sep .. "LightroomMCP.lrplugin" .. sep .. "?.lua"
if not package.path:find(pluginRoot, 1, true) then
    package.path = package.path .. ";" .. pluginRoot
end

-- Install a mock `import` global. Subsequent `import 'X'` calls return the mock for X.
function M.installImport(modules)
    _G.import = function(name)
        local m = modules[name]
        if m == nil then
            error("No mock installed for import('" .. tostring(name) .. "')", 2)
        end
        return m
    end
end

-- Default LrLogger stub used by every handler.
function M.defaultLrLogger()
    return setmetatable({}, {
        __call = function()
            return {
                info = function() end,
                warn = function() end,
                error = function() end,
                enable = function() end,
            }
        end,
    })
end

-- Valid Lightroom SDK metadata keys. The real getRawMetadata/getFormattedMetadata
-- THROW on an unsupported key, taking down the enclosing withReadAccessDo. The
-- non-validating mock used to return a value for any key, so a typo'd/invalid key
-- (e.g. `copyrightStatus` for `copyrightState`) shipped green. Validate against
-- this allowlist so specs catch it. Add genuinely-new SDK keys here.
local VALID_METADATA_KEYS = {}
for _, key in ipairs({
    -- File / catalog
    "fileName", "fileSize", "fileFormat", "path", "dimensions",
    "rating", "colorNameForLabel", "pickStatus", "keywords",
    -- Writable color label. The SDK reads the label via 'colorNameForLabel'
    -- but accepts EITHER 'label' or 'colorNameForLabel' on write; handlers
    -- normalize to 'label' so clearing works ('' clears, "none" would set a
    -- literal label named none).
    "label", "copyName",
    -- EXIF
    "dateTimeOriginal", "dateTimeDigitized", "cameraMake", "cameraModel",
    "cameraSerialNumber", "lens", "isoSpeedRating", "focalLength",
    "focalLength35mm", "aperture", "shutterSpeed", "exposureBias",
    "exposureProgram", "meteringMode", "flash", "artist", "software",
    "gps", "gpsAltitude",
    -- IPTC content / location / rights
    "title", "caption", "headline", "location", "city", "stateProvince",
    "country", "isoCountryCode", "creator", "copyright", "copyrightState",
    "rightsUsageTerms",
    -- IPTC creator contact block (writable)
    "creatorJobTitle", "creatorAddress", "creatorCity", "creatorStateProvince",
    "creatorPostalCode", "creatorCountry", "creatorPhone", "creatorEmail",
    "creatorUrl",
    -- IPTC status / workflow (writable)
    "iptcSubjectCode", "descriptionWriter", "dateCreated", "intellectualGenre",
    "scene", "jobIdentifier", "instructions", "provider", "source",
    "copyrightInfoUrl",
    -- IPTC extension (writable)
    "personShown", "nameOfOrgShown", "codeOfOrgShown", "event",
    "additionalModelInfo", "modelAge", "minorModelAge", "modelReleaseStatus",
    "modelReleaseID", "propertyReleaseID", "propertyReleaseStatus", "sourceType",
}) do
    VALID_METADATA_KEYS[key] = true
end
M.VALID_METADATA_KEYS = VALID_METADATA_KEYS

local function readMetadata(meta, key)
    -- `__`-prefixed keys are test-internal sentinels (e.g. __appliedSettings)
    -- that specs read back to assert what a handler wrote; never SDK keys.
    if key:sub(1, 2) ~= "__" and not VALID_METADATA_KEYS[key] then
        error("unsupported metadata key '" .. tostring(key)
            .. "' (add it to spec_helper VALID_METADATA_KEYS if it is a real SDK key)", 2)
    end
    return meta[key]
end

-- Build a fake photo with the given metadata table.
-- meta keys correspond to keys passed to getRawMetadata / getFormattedMetadata / localIdentifier.
function M.fakePhoto(meta)
    return {
        localIdentifier = meta.localIdentifier or meta.id or "photo-id",
        getRawMetadata = function(_, key) return readMetadata(meta, key) end,
        getFormattedMetadata = function(_, key) return readMetadata(meta, key) end,
        getDevelopSettings = function() return meta.developSettings or {} end,
        addKeyword = function(_, kw)
            meta.__addedKeywords = meta.__addedKeywords or {}
            table.insert(meta.__addedKeywords, kw)
        end,
        removeKeyword = function(_, kw)
            meta.__removedKeywords = meta.__removedKeywords or {}
            table.insert(meta.__removedKeywords, kw)
        end,
        setRawMetadata = function(_, key, value)
            -- Validate writes against the same allowlist as reads: the real SDK
            -- throws on an unknown key, which would abort the enclosing
            -- withWriteAccessDo, so a handler writing a typo'd key must fail the
            -- spec rather than silently "succeed".
            if key:sub(1, 2) ~= "__" and not VALID_METADATA_KEYS[key] then
                error("unsupported metadata key '" .. tostring(key)
                    .. "' (add it to spec_helper VALID_METADATA_KEYS if it is a real SDK key)", 2)
            end
            meta[key] = value
            meta.__writes = meta.__writes or {}
            table.insert(meta.__writes, { key = key, value = value })
        end,
        getWrites = function() return meta.__writes or {} end,
        applyDevelopPreset = function(_, preset, plugin)
            meta.__appliedPreset = preset
            meta.__appliedPresetPlugin = plugin
        end,
        applyDevelopSettings = function(_, settings)
            meta.__appliedSettings = settings
        end,
        -- Async JPEG preview. The real SDK invokes the callback later and returns
        -- a request handle; the fake resolves synchronously, which is what the
        -- handler's poll loop tolerates (it checks the flag before sleeping).
        requestJpegThumbnail = function(_, _width, _height, callback)
            if meta.thumbnailError then
                callback(nil, meta.thumbnailError)
            else
                callback(meta.thumbnailData or "JPEGDATA")
            end
            return { __request = true }
        end,
    }
end

-- Build a fake collection.
function M.fakeCollection(name, photos, opts)
    photos = photos or {}
    opts = opts or {}
    local addedPhotos = {}
    local removedPhotos = {}
    local state = { name = name, deleted = false, parent = opts.parent }
    local collection
    collection = {
        getName = function() return state.name end,
        type = function() return "LrCollection" end,
        getPhotos = function() return photos end,
        isSmartCollection = function() return opts.isSmart == true end,
        getParent = function() return state.parent end,
        addPhotos = function(_, ps)
            if opts.isSmart then error("cannot add photos to a smart collection") end
            for _, p in ipairs(ps) do
                table.insert(addedPhotos, p)
                table.insert(photos, p)
            end
        end,
        removePhotos = function(_, ps)
            if opts.isSmart then error("cannot remove photos from a smart collection") end
            for _, p in ipairs(ps) do
                table.insert(removedPhotos, p)
                for index, existing in ipairs(photos) do
                    if existing == p then
                        table.remove(photos, index)
                        break
                    end
                end
            end
        end,
        setName = function(_, newName)
            -- Mirror the SDK: returns false (does not throw) on a sibling clash.
            if opts.nameTaken == newName then return false end
            state.name = newName
            return true
        end,
        delete = function() state.deleted = true end,
        isDeleted = function() return state.deleted end,
        getAddedPhotos = function() return addedPhotos end,
        getRemovedPhotos = function() return removedPhotos end,
    }
    return collection
end

-- Build a fake collection set that nests collections and other sets.
function M.fakeCollectionSet(name, collections, childSets, opts)
    opts = opts or {}
    local state = { name = name, parent = opts.parent }
    return {
        getName = function() return state.name end,
        type = function() return "LrCollectionSet" end,
        getChildCollections = function() return collections or {} end,
        getChildCollectionSets = function() return childSets or {} end,
        getParent = function() return state.parent end,
        setName = function(_, newName) state.name = newName return true end,
    }
end

-- Build a fake catalog folder (Folders panel entry).
function M.fakeFolder(name, path, photos, children)
    return {
        getName = function() return name end,
        getPath = function() return path end,
        type = function() return "LrFolder" end,
        getPhotos = function() return photos or {} end,
        getChildren = function() return children or {} end,
    }
end

-- Build a fake keyword, optionally with children and synonyms.
function M.fakeKeyword(name, opts)
    opts = opts or {}
    return {
        getName = function() return name end,
        type = function() return "LrKeyword" end,
        getChildren = function() return opts.children or {} end,
        getSynonyms = function() return opts.synonyms or {} end,
        getAttributes = function()
            return { includeOnExport = opts.includeOnExport ~= false }
        end,
        getPhotos = function() return opts.photos or {} end,
    }
end

-- Build a fake catalog. opts:
--   photos: array of fake photos
--   collections: array of fake collections
--   collectionSets: array of fake collection sets
function M.fakeCatalog(opts)
    opts = opts or {}
    local photos = opts.photos or {}
    local collections = opts.collections or {}
    local collectionSets = opts.collectionSets or {}
    local createdCollections = {}
    local createdCollectionSets = {}
    local createdSmartCollections = {}
    local createdKeywords = {}
    local writeActionNames = {}
    local lastSearchDesc = nil
    local selection = nil
    local readAccessCount = 0
    local writeAccessCount = 0
    -- Tracks whether a catalog query (getTargetPhotos/findPhotos/getAllPhotos)
    -- was invoked while a withReadAccessDo gate was open. The Windows deadlock
    -- (#134/#124) is exactly that nesting, so handlers must keep their query
    -- OUTSIDE the gate; specs assert getQueriedInsideReadAccess() == false.
    local insideReadAccess = false
    local queriedInsideReadAccess = false
    local function markQuery()
        if insideReadAccess then queriedInsideReadAccess = true end
    end

    -- Numeric comparison shared by rating / isoSpeedRating.
    local function compareNumbers(actual, op, expected)
        if actual == nil then return false end
        if op == "==" then return actual == expected end
        if op == "!=" then return actual ~= expected end
        if op == ">" then return actual > expected end
        if op == "<" then return actual < expected end
        if op == ">=" then return actual >= expected end
        if op == "<=" then return actual <= expected end
        return nil
    end

    local function photoMatches(photo, criterion)
        local crit = criterion.criteria
        local op = criterion.operation

        -- Boolean criteria carry no value, only isTrue / isFalse.
        if op == "isTrue" or op == "isFalse" then
            local want = (op == "isTrue")
            if crit == "hasGPSData" then
                return (photo:getRawMetadata('gps') ~= nil) == want
            elseif crit == "hasAdjustments" then
                return (photo:getRawMetadata('__hasAdjustments') == true) == want
            elseif crit == "cropped" then
                return (photo:getRawMetadata('__cropped') == true) == want
            end
            error("fakeCatalog.findPhotos: unsupported boolean criterion " .. tostring(crit))
        end

        if crit == "filename" and op == "any" then
            local name = photo:getFormattedMetadata('fileName')
            if not name then return false end
            return name:lower():find(criterion.value:lower(), 1, true) ~= nil
        elseif crit == "all" then
            -- Free-text search across the indexed fields the fake models.
            local haystack = table.concat({
                photo:getFormattedMetadata('fileName') or "",
                photo:getFormattedMetadata('title') or "",
                photo:getFormattedMetadata('caption') or "",
            }, " "):lower()
            local needle = tostring(criterion.value):lower()
            if op == "any" or op == "all" or op == "words" then
                return haystack:find(needle, 1, true) ~= nil
            elseif op == "noneOf" then
                return haystack:find(needle, 1, true) == nil
            elseif op == "beginsWith" then
                return haystack:sub(1, #needle) == needle
            elseif op == "endsWith" then
                return haystack:sub(- #needle) == needle
            end
        elseif crit == "rating" then
            local result = compareNumbers(photo:getRawMetadata('rating'), op, criterion.value)
            if result ~= nil then return result end
        elseif crit == "isoSpeedRating" then
            local result = compareNumbers(photo:getRawMetadata('isoSpeedRating'), op, criterion.value)
            if result ~= nil then return result end
        elseif crit == "keywords" and op == "all" then
            local kws = photo:getRawMetadata('keywords') or {}
            local target = criterion.value:lower()
            for _, kw in ipairs(kws) do
                if kw:getName():lower() == target then return true end
            end
            return false
        elseif crit == "pick" and op == "==" then
            return (photo:getRawMetadata('pickStatus') or 0) == criterion.value
        elseif crit == "labelColor" and op == "==" then
            local label = photo:getRawMetadata('colorNameForLabel')
            if criterion.value == "none" then
                return label == nil or label == "" or label == "none"
            end
            -- The SDK matches on a numeric colour index; the fake stores names,
            -- so map through the same order Lightroom uses.
            local order = { red = 1, yellow = 2, green = 3, blue = 4, purple = 5 }
            return label ~= nil and order[tostring(label):lower()] == criterion.value
        elseif crit == "fileFormat" and op == "==" then
            local format = photo:getRawMetadata('fileFormat')
            return format ~= nil and tostring(format):upper() == tostring(criterion.value):upper()
        elseif (crit == "camera" or crit == "lens" or crit == "folder"
                or crit == "collection" or crit == "city" or crit == "country")
                and op == "any" then
            local FIELD_BY_CRITERION = {
                camera = 'cameraModel', lens = 'lens', folder = 'path',
                collection = '__collection', city = 'city', country = 'country',
            }
            local value = photo:getRawMetadata(FIELD_BY_CRITERION[crit])
                or photo:getFormattedMetadata(FIELD_BY_CRITERION[crit])
            if not value then return false end
            return tostring(value):lower():find(tostring(criterion.value):lower(), 1, true) ~= nil
        elseif crit == "aspectRatio" and op == "==" then
            return photo:getRawMetadata('__aspectRatio') == criterion.value
        elseif crit == "treatment" and op == "==" then
            return photo:getRawMetadata('__treatment') == criterion.value
        elseif crit == "captureTime" then
            local t = photo:getRawMetadata('dateTimeOriginal')
            if not t then return false end
            if op == "in" then
                return t >= criterion.value and t <= criterion.value2
            elseif op == ">" then
                return t > criterion.value
            elseif op == "<" then
                return t < criterion.value
            elseif op == "inLast" then
                -- The fake cannot evaluate a real relative window; specs assert
                -- the emitted descriptor instead. Treat as matching so the
                -- surrounding pagination logic still exercises.
                return true
            end
        elseif crit == "touchTime" and op == "inLast" then
            return true
        end
        error("fakeCatalog.findPhotos: unsupported criterion " .. tostring(crit) .. "/" .. tostring(op))
    end

    return {
        getAllPhotos = function() markQuery() return photos end,
        getTargetPhotos = function() markQuery() return opts.targetPhotos or photos end,
        getPath = function() return opts.path or "/catalogs/Test.lrcat" end,
        getFolders = function() return opts.folders or {} end,
        getKeywords = function() return opts.keywords or {} end,
        setSelectedPhotos = function(_, active, others)
            selection = { active = active, others = others or {} }
        end,
        getSelectionState = function() return selection end,
        findPhotos = function(_, opts)
            markQuery()
            local desc = opts and opts.searchDesc or {}
            local out = {}
            for _, photo in ipairs(photos) do
                local matched = 0
                for _, criterion in ipairs(desc) do
                    if photoMatches(photo, criterion) then matched = matched + 1 end
                end
                -- Honour the combine mode so specs can cover match=any (union)
                -- as well as the default intersect.
                local ok
                if desc.combine == "union" then
                    ok = #desc == 0 or matched > 0
                else
                    ok = matched == #desc
                end
                if ok then table.insert(out, photo) end
            end
            if opts and opts.sort == "fileName" then
                table.sort(out, function(a, b)
                    local an = a:getFormattedMetadata('fileName') or ""
                    local bn = b:getFormattedMetadata('fileName') or ""
                    if opts.ascending then return an < bn end
                    return an > bn
                end)
            end
            lastSearchDesc = desc
            return out
        end,
        getLastSearchDesc = function() return lastSearchDesc end,
        getChildCollections = function() return collections end,
        getChildCollectionSets = function() return collectionSets end,
        withReadAccessDo = function(_, fn)
            readAccessCount = readAccessCount + 1
            insideReadAccess = true
            local ok, err = pcall(fn)
            insideReadAccess = false
            if not ok then error(err, 0) end
        end,
        getQueriedInsideReadAccess = function() return queriedInsideReadAccess end,
        withWriteAccessDo = function(_, actionName, fn)
            writeAccessCount = writeAccessCount + 1
            table.insert(writeActionNames, actionName)
            -- The real gate rolls the transaction back when the callback throws;
            -- propagate the error so handlers cannot appear to succeed.
            fn()
        end,
        getWriteActionNames = function() return writeActionNames end,
        findPhotoByLocalIdentifier = function(_, id)
            local target = tostring(id)
            for _, p in ipairs(photos) do
                if tostring(p.localIdentifier) == target then return p end
            end
            return nil
        end,
        createCollection = function(_, name, parent, canReturnPrior)
            local c = M.fakeCollection(name, {}, { parent = parent })
            table.insert(createdCollections,
                { collection = c, name = name, parent = parent, canReturnPrior = canReturnPrior })
            if parent then
                table.insert(parent:getChildCollections(), c)
            else
                table.insert(collections, c)
            end
            return c
        end,
        createCollectionSet = function(_, name, parent, canReturnPrior)
            local s = M.fakeCollectionSet(name, {}, {}, { parent = parent })
            table.insert(createdCollectionSets,
                { set = s, name = name, parent = parent, canReturnPrior = canReturnPrior })
            if parent then
                table.insert(parent:getChildCollectionSets(), s)
            else
                table.insert(collectionSets, s)
            end
            return s
        end,
        createSmartCollection = function(_, name, searchDesc, parent, canReturnPrior)
            local c = M.fakeCollection(name, {}, { parent = parent, isSmart = true })
            table.insert(createdSmartCollections, {
                collection = c, name = name, searchDesc = searchDesc,
                parent = parent, canReturnPrior = canReturnPrior,
            })
            if parent then
                table.insert(parent:getChildCollections(), c)
            else
                table.insert(collections, c)
            end
            return c
        end,
        createKeyword = function(_, name)
            local kw = { getName = function() return name end }
            table.insert(createdKeywords, kw)
            return kw
        end,
        addPhoto = function(_, path)
            local p = M.fakePhoto({ path = path, id = path })
            table.insert(photos, p)
            return p
        end,
        getCreatedCollections = function() return createdCollections end,
        getCreatedCollectionSets = function() return createdCollectionSets end,
        getCreatedSmartCollections = function() return createdSmartCollections end,
        getCreatedKeywords = function() return createdKeywords end,
        getReadAccessCount = function() return readAccessCount end,
        getWriteAccessCount = function() return writeAccessCount end,
    }
end

return M
