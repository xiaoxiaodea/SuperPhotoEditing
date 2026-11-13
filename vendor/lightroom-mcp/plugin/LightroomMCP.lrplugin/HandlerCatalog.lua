-- Catalog structure: folders, keyword hierarchy, catalog info.
--
-- The bridge could previously see collections but not FOLDERS, so an agent had
-- no way to answer "what shoots are in this catalog" or scope work to a folder
-- on disk. Keywords were equally invisible: set_keywords could create them but
-- nothing could list what already existed, so agents invented near-duplicate
-- keywords ("sunset" vs "Sunset") instead of reusing the catalog's vocabulary.
--
-- Read-gate discipline (see CLAUDE.md): folder:getPhotos() and keyword:getPhotos()
-- are potentially-yielding catalog queries and must NOT run inside
-- withReadAccessDo. Nesting a yielding query in the read gate deadlocks on
-- Windows (issues #124/#134). getFolders/getKeywords/getChildren/getName are
-- non-yielding structure reads and are safe inside the gate.

local LrApplication = import 'LrApplication'

local Validate = require 'Validate'
local Log = require 'Log'

local CatalogHandler = {}

-- Same rationale as CollectionLookup.MAX_DEPTH: bound the recursion so a
-- pathological catalog cannot spin inside a read gate.
local MAX_DEPTH = 32

function CatalogHandler.listFolders(args)
    args = args or {}
    local limit, offset = Validate.pagination(args)
    local includePhotoCounts = args.include_photo_counts == true

    local catalog = LrApplication.activeCatalog()
    local flat = {}

    -- Structure walk inside the gate; no photo access here.
    catalog:withReadAccessDo(function()
        local function descend(folder, depth, parentPath)
            if depth > MAX_DEPTH then return end
            local entry = {
                name = folder:getName(),
                path = folder:getPath(),
                depth = depth,
                parent = parentPath,
            }
            table.insert(flat, { entry = entry, folder = folder })
            for _, child in ipairs(folder:getChildren() or {}) do
                descend(child, depth + 1, entry.path)
            end
        end

        for _, folder in ipairs(catalog:getFolders() or {}) do
            descend(folder, 0, nil)
        end
    end)

    -- Photo counts are opt-in: getPhotos() on every folder of a large catalog is
    -- slow, and it must run outside the read gate (see the note at the top).
    if includePhotoCounts then
        for _, row in ipairs(flat) do
            local ok, photos = pcall(function() return row.folder:getPhotos(false) end)
            if ok and type(photos) == "table" then
                row.entry.photoCount = #photos
            end
        end
    end

    local entries = {}
    for _, row in ipairs(flat) do table.insert(entries, row.entry) end

    local slice, total, hasMore = Validate.paginate(entries, limit, offset)

    Log.info(string.format("Listed %d folders, returning %d (offset=%d, limit=%d)",
        total, #slice, offset, limit))

    return {
        count = total,
        folders = slice,
        has_more = hasMore,
    }
end

function CatalogHandler.listKeywords(args)
    args = args or {}
    local limit, offset = Validate.pagination(args)
    local includePhotoCounts = args.include_photo_counts == true

    local catalog = LrApplication.activeCatalog()
    local flat = {}

    catalog:withReadAccessDo(function()
        local function descend(keyword, depth, parentPath)
            if depth > MAX_DEPTH then return end
            local name = keyword:getName()
            local path = parentPath and (parentPath .. " > " .. name) or name
            local entry = {
                name = name,
                path = path,
                depth = depth,
                parent = parentPath,
            }

            -- getSynonyms / getAttributes are absent on some catalogs and on the
            -- spec fakes; probe rather than assume.
            if type(keyword.getSynonyms) == "function" then
                local ok, synonyms = pcall(function() return keyword:getSynonyms() end)
                if ok and type(synonyms) == "table" and #synonyms > 0 then
                    entry.synonyms = synonyms
                end
            end
            if type(keyword.getAttributes) == "function" then
                local ok, attributes = pcall(function() return keyword:getAttributes() end)
                if ok and type(attributes) == "table" then
                    entry.includeOnExport = attributes.includeOnExport
                end
            end

            table.insert(flat, { entry = entry, keyword = keyword })

            local children = {}
            if type(keyword.getChildren) == "function" then
                local ok, result = pcall(function() return keyword:getChildren() end)
                if ok and type(result) == "table" then children = result end
            end
            for _, child in ipairs(children) do
                descend(child, depth + 1, path)
            end
        end

        for _, keyword in ipairs(catalog:getKeywords() or {}) do
            descend(keyword, 0, nil)
        end
    end)

    -- Outside the gate: keyword:getPhotos() yields.
    if includePhotoCounts then
        for _, row in ipairs(flat) do
            local ok, photos = pcall(function() return row.keyword:getPhotos() end)
            if ok and type(photos) == "table" then
                row.entry.photoCount = #photos
            end
        end
    end

    local entries = {}
    for _, row in ipairs(flat) do table.insert(entries, row.entry) end

    local slice, total, hasMore = Validate.paginate(entries, limit, offset)

    Log.info(string.format("Listed %d keywords, returning %d (offset=%d, limit=%d)",
        total, #slice, offset, limit))

    return {
        count = total,
        keywords = slice,
        has_more = hasMore,
    }
end

-- Catalog identity + Lightroom version. Lets an agent confirm which catalog it
-- is talking to before making destructive changes, and adapt to SDK-version
-- differences instead of guessing.
function CatalogHandler.getCatalogInfo(_)
    local catalog = LrApplication.activeCatalog()

    local info = {}

    local ok, path = pcall(function() return catalog:getPath() end)
    if ok then info.path = path end

    -- versionTable() is the structured form; version string is the fallback.
    if type(LrApplication.versionTable) == "function" then
        local versionOk, version = pcall(LrApplication.versionTable)
        if versionOk and type(version) == "table" then
            info.lightroomVersion = {
                major = version.major,
                minor = version.minor,
                revision = version.revision,
                build = version.build,
            }
        end
    end
    if type(LrApplication.versionString) == "function" then
        local versionOk, version = pcall(LrApplication.versionString)
        if versionOk then info.lightroomVersionString = version end
    end

    -- getAllPhotos is a non-yielding enumeration, safe outside a gate.
    local photosOk, photos = pcall(function() return catalog:getAllPhotos() end)
    if photosOk and type(photos) == "table" then
        info.photoCount = #photos
    end

    catalog:withReadAccessDo(function()
        local collectionCount = 0
        local CollectionLookup = require 'CollectionLookup'
        CollectionLookup.walkCollections(catalog, function()
            collectionCount = collectionCount + 1
        end)
        info.collectionCount = collectionCount

        local setCount = 0
        CollectionLookup.walkCollectionSets(catalog, function()
            setCount = setCount + 1
        end)
        info.collectionSetCount = setCount
    end)

    Log.info("Retrieved catalog info")

    info.success = true
    return info
end

return CatalogHandler
