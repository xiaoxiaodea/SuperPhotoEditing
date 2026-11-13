-- Shared collection / collection-set traversal.
--
-- The catalog exposes collections as a tree: top-level collections plus
-- collection sets that nest arbitrarily. HandlerCollections and HandlerImport
-- each had their own partial walk of it — HandlerImport's searched only the top
-- level, so importing into a collection nested in a set silently logged
-- "Collection not found" and dropped the photos. One traversal here fixes that
-- class of bug for every caller.
--
-- Every function must be called from inside a catalog read or write gate:
-- getChildCollections / getChildCollectionSets / getName are catalog reads.

local CollectionLookup = {}

-- Depth cap: the catalog cannot actually contain a cycle, but a corrupted
-- catalog or an SDK quirk returning a set as its own descendant would otherwise
-- spin forever inside a write gate and wedge the bridge.
local MAX_DEPTH = 32

-- Walk every collection, breadth of tree first, calling visit(collection, path)
-- where `path` is the human-readable "Set / Subset / Name" label used by
-- list_collections and accepted by the name-based selectors.
local function walkCollections(catalog, visit)
    for _, collection in ipairs(catalog:getChildCollections()) do
        visit(collection, collection:getName(), nil)
    end

    local function descend(collectionSet, prefix, depth)
        if depth > MAX_DEPTH then return end
        for _, collection in ipairs(collectionSet:getChildCollections()) do
            visit(collection, prefix .. collection:getName(), collectionSet)
        end
        for _, childSet in ipairs(collectionSet:getChildCollectionSets()) do
            descend(childSet, prefix .. childSet:getName() .. " / ", depth + 1)
        end
    end

    for _, collectionSet in ipairs(catalog:getChildCollectionSets()) do
        descend(collectionSet, collectionSet:getName() .. " / ", 1)
    end
end

CollectionLookup.walkCollections = walkCollections

-- Walk every collection set, calling visit(set, path).
local function walkCollectionSets(catalog, visit)
    local function descend(collectionSet, prefix, depth)
        if depth > MAX_DEPTH then return end
        local path = prefix .. collectionSet:getName()
        visit(collectionSet, path)
        for _, childSet in ipairs(collectionSet:getChildCollectionSets()) do
            descend(childSet, path .. " / ", depth + 1)
        end
    end

    for _, collectionSet in ipairs(catalog:getChildCollectionSets()) do
        descend(collectionSet, "", 1)
    end
end

CollectionLookup.walkCollectionSets = walkCollectionSets

-- Resolve a collection by name. Accepts either the bare leaf name ("Trip") or
-- the full path shown by list_collections ("2024 / Trip"), so an agent can pass
-- back exactly what it was given.
--
-- Ambiguity is an error rather than a silent first-match: picking arbitrarily
-- between two same-named collections in different sets would write photos into
-- the wrong one with no signal to the caller.
function CollectionLookup.findCollection(catalog, name)
    local pathMatches = {}
    local leafMatches = {}

    walkCollections(catalog, function(collection, path)
        if path == name then
            table.insert(pathMatches, { collection = collection, path = path })
        elseif collection:getName() == name then
            table.insert(leafMatches, { collection = collection, path = path })
        end
    end)

    -- A full-path match is unambiguous by construction, so it wins outright.
    if #pathMatches == 1 then return pathMatches[1].collection, pathMatches[1].path end
    if #pathMatches > 1 then
        error("Collection path is ambiguous: " .. name)
    end

    if #leafMatches == 1 then return leafMatches[1].collection, leafMatches[1].path end
    if #leafMatches > 1 then
        local paths = {}
        for _, match in ipairs(leafMatches) do table.insert(paths, match.path) end
        table.sort(paths)
        error("Collection name is ambiguous: " .. name
            .. " (matches: " .. table.concat(paths, "; ") .. "). Pass the full path.")
    end

    return nil, nil
end

function CollectionLookup.requireCollection(catalog, name)
    local collection, path = CollectionLookup.findCollection(catalog, name)
    if not collection then
        error("Collection not found: " .. tostring(name))
    end
    return collection, path
end

-- Resolve a collection set by name or full path. Same ambiguity rules.
function CollectionLookup.findCollectionSet(catalog, name)
    local pathMatches = {}
    local leafMatches = {}

    walkCollectionSets(catalog, function(collectionSet, path)
        if path == name then
            table.insert(pathMatches, { set = collectionSet, path = path })
        elseif collectionSet:getName() == name then
            table.insert(leafMatches, { set = collectionSet, path = path })
        end
    end)

    if #pathMatches == 1 then return pathMatches[1].set, pathMatches[1].path end
    if #pathMatches > 1 then
        error("Collection set path is ambiguous: " .. name)
    end

    if #leafMatches == 1 then return leafMatches[1].set, leafMatches[1].path end
    if #leafMatches > 1 then
        local paths = {}
        for _, match in ipairs(leafMatches) do table.insert(paths, match.path) end
        table.sort(paths)
        error("Collection set name is ambiguous: " .. name
            .. " (matches: " .. table.concat(paths, "; ") .. "). Pass the full path.")
    end

    return nil, nil
end

function CollectionLookup.requireCollectionSet(catalog, name)
    local collectionSet, path = CollectionLookup.findCollectionSet(catalog, name)
    if not collectionSet then
        error("Collection set not found: " .. tostring(name))
    end
    return collectionSet, path
end

-- isSmartCollection is absent on older LrCollection objects and on the spec
-- fakes, so probe defensively rather than calling it blind.
function CollectionLookup.isSmart(collection)
    local method = collection and collection.isSmartCollection
    if type(method) ~= "function" then return false end
    local ok, result = pcall(method, collection)
    return ok and result == true
end

return CollectionLookup
