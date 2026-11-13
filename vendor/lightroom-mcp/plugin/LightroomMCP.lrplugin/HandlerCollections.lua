local LrApplication = import 'LrApplication'

local PhotoLookup = require 'PhotoLookup'
local CollectionLookup = require 'CollectionLookup'
local Validate = require 'Validate'
local Log = require 'Log'

local CollectionsHandler = {}

function CollectionsHandler.listCollections(args)
    args = args or {}
    local catalog = LrApplication.activeCatalog()
    local all = {}

    local limit, offset = Validate.pagination(args)

    catalog:withReadAccessDo(function()
        CollectionLookup.walkCollections(catalog, function(collection, path, parentSet)
            local entry = {
                name = path,
                type = collection:type(),
                photoCount = #collection:getPhotos(),
            }
            if parentSet then entry.parent = parentSet:getName() end
            if CollectionLookup.isSmart(collection) then entry.isSmart = true end
            table.insert(all, entry)
        end)
    end)

    local slice, total, hasMore = Validate.paginate(all, limit, offset)

    Log.info(string.format("Found %d collections, returning %d (offset=%d, limit=%d)",
        total, #slice, offset, limit))

    return {
        count = total,
        collections = slice,
        has_more = hasMore,
    }
end

function CollectionsHandler.listCollectionSets(args)
    args = args or {}
    local catalog = LrApplication.activeCatalog()
    local all = {}

    local limit, offset = Validate.pagination(args)

    catalog:withReadAccessDo(function()
        CollectionLookup.walkCollectionSets(catalog, function(collectionSet, path)
            local entry = {
                name = path,
                type = collectionSet:type(),
                collectionCount = #collectionSet:getChildCollections(),
                setCount = #collectionSet:getChildCollectionSets(),
            }
            local parent = collectionSet:getParent()
            if parent then entry.parent = parent:getName() end
            table.insert(all, entry)
        end)
    end)

    local slice, total, hasMore = Validate.paginate(all, limit, offset)

    Log.info(string.format("Found %d collection sets, returning %d (offset=%d, limit=%d)",
        total, #slice, offset, limit))

    return {
        count = total,
        collection_sets = slice,
        has_more = hasMore,
    }
end

function CollectionsHandler.createCollection(args)
    args = args or {}
    Validate.requireString(args.name, "name")
    Validate.optionalString(args.parent, "parent")

    local catalog = LrApplication.activeCatalog()
    local collectionName = args.name
    local created = false
    local parentPath = nil

    catalog:withWriteAccessDo("Create Collection", function()
        -- The `parent` argument was previously accepted by the tool schema and
        -- then dropped on the floor: createCollection was called with the name
        -- only, so every collection landed at the top level while the tool
        -- reported success. Resolve the set and pass it through.
        local parentSet = nil
        if args.parent then
            parentSet, parentPath = CollectionLookup.requireCollectionSet(catalog, args.parent)
        end

        -- canReturnPrior=true makes this idempotent instead of returning nil on
        -- a name clash, which previously surfaced as a bare "success" with no
        -- collection created.
        local collection = catalog:createCollection(collectionName, parentSet, true)
        if not collection then
            error("Could not create collection: " .. collectionName)
        end
        created = true
        Log.info("Created collection: " .. collectionName)
    end)

    return {
        success = created,
        name = collectionName,
        parent = parentPath,
        message = "Collection created: " .. collectionName,
    }
end

function CollectionsHandler.createCollectionSet(args)
    args = args or {}
    Validate.requireString(args.name, "name")
    Validate.optionalString(args.parent, "parent")

    local catalog = LrApplication.activeCatalog()
    local created = false
    local parentPath = nil

    catalog:withWriteAccessDo("Create Collection Set", function()
        local parentSet = nil
        if args.parent then
            parentSet, parentPath = CollectionLookup.requireCollectionSet(catalog, args.parent)
        end
        local collectionSet = catalog:createCollectionSet(args.name, parentSet, true)
        if not collectionSet then
            error("Could not create collection set: " .. args.name)
        end
        created = true
        Log.info("Created collection set: " .. args.name)
    end)

    return {
        success = created,
        name = args.name,
        parent = parentPath,
        message = "Collection set created: " .. args.name,
    }
end

-- Smart collections are defined by a search descriptor rather than membership.
-- The descriptor is built by HandlerSearch from the same filter vocabulary the
-- search_photos tool uses, so a saved smart collection and an ad-hoc search stay
-- consistent instead of drifting apart.
function CollectionsHandler.createSmartCollection(args)
    args = args or {}
    Validate.requireString(args.name, "name")
    Validate.optionalString(args.parent, "parent")

    local HandlerSearch = require 'HandlerSearch'
    local searchDesc = HandlerSearch.buildSearchDesc(args.criteria or {})
    if #searchDesc == 0 then
        error("criteria must contain at least one filter")
    end
    if args.match == "any" then
        searchDesc.combine = "union"
    end

    local catalog = LrApplication.activeCatalog()
    local created = false
    local parentPath = nil

    catalog:withWriteAccessDo("Create Smart Collection", function()
        local parentSet = nil
        if args.parent then
            parentSet, parentPath = CollectionLookup.requireCollectionSet(catalog, args.parent)
        end
        local collection = catalog:createSmartCollection(args.name, searchDesc, parentSet, true)
        if not collection then
            error("Could not create smart collection: " .. args.name)
        end
        created = true
        Log.info("Created smart collection: " .. args.name)
    end)

    return {
        success = created,
        name = args.name,
        parent = parentPath,
        combine = searchDesc.combine or "intersect",
        criteria_count = #searchDesc,
        message = "Smart collection created: " .. args.name,
    }
end

function CollectionsHandler.addToCollection(args)
    args = args or {}
    Validate.requireString(args.collection_name, "collection_name")
    Validate.requirePhotoIds(args.photo_ids)

    local catalog = LrApplication.activeCatalog()
    local addedCount = 0
    local missing = {}
    local resolvedPath = nil

    catalog:withWriteAccessDo("Add Photos to Collection", function()
        local targetCollection, path = CollectionLookup.requireCollection(catalog, args.collection_name)
        resolvedPath = path

        -- addPhotos throws on a smart collection; a named error is clearer than
        -- the SDK's, and it avoids aborting the transaction on a foreseeable case.
        if CollectionLookup.isSmart(targetCollection) then
            error("Cannot add photos to a smart collection: " .. path)
        end

        local photosToAdd = {}
        local resolved = PhotoLookup.resolveMany(catalog, args.photo_ids)
        for _, entry in ipairs(resolved) do
            if entry.photo then
                table.insert(photosToAdd, entry.photo)
            else
                table.insert(missing, entry.id)
            end
        end

        if #photosToAdd > 0 then
            targetCollection:addPhotos(photosToAdd)
            addedCount = #photosToAdd
        end
    end)

    Log.info(string.format("Added %d photos to collection: %s", addedCount, args.collection_name))

    return {
        success = true,
        added = addedCount,
        collection = resolvedPath,
        not_found = missing,
        message = string.format("Added %d photos to collection", addedCount)
    }
end

function CollectionsHandler.removeFromCollection(args)
    args = args or {}
    Validate.requireString(args.collection_name, "collection_name")
    Validate.requirePhotoIds(args.photo_ids)

    local catalog = LrApplication.activeCatalog()
    local removedCount = 0
    local missing = {}
    local resolvedPath = nil

    catalog:withWriteAccessDo("Remove Photos from Collection", function()
        local targetCollection, path = CollectionLookup.requireCollection(catalog, args.collection_name)
        resolvedPath = path

        if CollectionLookup.isSmart(targetCollection) then
            error("Cannot remove photos from a smart collection: " .. path)
        end

        local photosToRemove = {}
        local resolved = PhotoLookup.resolveMany(catalog, args.photo_ids)
        for _, entry in ipairs(resolved) do
            if entry.photo then
                table.insert(photosToRemove, entry.photo)
            else
                table.insert(missing, entry.id)
            end
        end

        if #photosToRemove > 0 then
            targetCollection:removePhotos(photosToRemove)
            removedCount = #photosToRemove
        end
    end)

    Log.info(string.format("Removed %d photos from collection: %s", removedCount, args.collection_name))

    return {
        success = true,
        removed = removedCount,
        collection = resolvedPath,
        not_found = missing,
        message = string.format("Removed %d photos from collection", removedCount),
    }
end

function CollectionsHandler.renameCollection(args)
    args = args or {}
    Validate.requireString(args.collection_name, "collection_name")
    Validate.requireString(args.new_name, "new_name")

    local catalog = LrApplication.activeCatalog()
    local renamed = false
    local resolvedPath = nil

    catalog:withWriteAccessDo("Rename Collection", function()
        local targetCollection, path = CollectionLookup.requireCollection(catalog, args.collection_name)
        resolvedPath = path
        -- setName returns false (rather than throwing) when a sibling already
        -- holds the name; surface that as an error so the caller does not think
        -- the rename landed.
        local ok = targetCollection:setName(args.new_name)
        if ok == false then
            error("A collection or set named '" .. args.new_name .. "' already exists in that parent")
        end
        renamed = true
    end)

    Log.info(string.format("Renamed collection %s to %s", tostring(resolvedPath), args.new_name))

    return {
        success = renamed,
        previous_name = resolvedPath,
        name = args.new_name,
        message = string.format("Renamed collection to %s", args.new_name),
    }
end

-- Destructive: removes the collection itself. Photos are NOT deleted from the
-- catalog — Lightroom only drops the grouping — and the response says so
-- explicitly so the caller does not over-report the blast radius to a user.
function CollectionsHandler.deleteCollection(args)
    args = args or {}
    Validate.requireString(args.collection_name, "collection_name")

    local catalog = LrApplication.activeCatalog()
    local deleted = false
    local resolvedPath = nil
    local photoCount = 0

    catalog:withWriteAccessDo("Delete Collection", function()
        local targetCollection, path = CollectionLookup.requireCollection(catalog, args.collection_name)
        resolvedPath = path
        local ok, photos = pcall(function() return targetCollection:getPhotos() end)
        if ok and type(photos) == "table" then photoCount = #photos end
        targetCollection:delete()
        deleted = true
    end)

    Log.info(string.format("Deleted collection %s", tostring(resolvedPath)))

    return {
        success = deleted,
        collection = resolvedPath,
        released_photos = photoCount,
        photos_deleted = false,
        message = string.format(
            "Deleted collection %s; %d photo(s) remain in the catalog",
            tostring(resolvedPath), photoCount),
    }
end

-- Read the membership of one collection. list_collections gives counts only, so
-- without this an agent had to scan the whole catalog to find out which photos a
-- collection actually contains.
function CollectionsHandler.getCollectionPhotos(args)
    args = args or {}
    Validate.requireString(args.collection_name, "collection_name")
    local limit, offset = Validate.pagination(args)

    local catalog = LrApplication.activeCatalog()
    local photos = {}
    local resolvedPath = nil

    -- getPhotos() yields, so resolve the collection and read membership OUTSIDE
    -- the read gate, then take the gate only for the per-photo metadata reads
    -- (same structure as HandlerSearch / HandlerSelection).
    local targetCollection, path = CollectionLookup.requireCollection(catalog, args.collection_name)
    resolvedPath = path
    local members = targetCollection:getPhotos() or {}

    local slice, total, hasMore = Validate.paginate(members, limit, offset)

    catalog:withReadAccessDo(function()
        for _, photo in ipairs(slice) do
            table.insert(photos, {
                id = photo.localIdentifier,
                path = photo:getRawMetadata('path'),
                filename = photo:getFormattedMetadata('fileName'),
                rating = photo:getRawMetadata('rating'),
                dateTimeOriginal = photo:getFormattedMetadata('dateTimeOriginal'),
            })
        end
    end)

    Log.info(string.format("Collection %s has %d photos, returning %d",
        tostring(resolvedPath), total, #photos))

    return {
        count = total,
        collection = resolvedPath,
        photos = photos,
        has_more = hasMore,
    }
end

return CollectionsHandler
