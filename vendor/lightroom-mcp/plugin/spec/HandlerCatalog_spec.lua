local helper = require 'spec_helper'

local function setup(opts)
    local catalog = helper.fakeCatalog(opts or {})
    helper.installImport({
        LrApplication = {
            activeCatalog = function() return catalog end,
            versionTable = function()
                return { major = 14, minor = 2, revision = 0, build = "202601" }
            end,
            versionString = function() return "14.2" end,
        },
        LrLogger = helper.defaultLrLogger(),
        LrPathUtils = {
            child = function(a, b) return a .. "/" .. b end,
            getStandardFilePath = function() return "/home" end,
        },
        LrFileUtils = { createAllDirectories = function() end },
    })
    package.loaded.HandlerCatalog = nil
    package.loaded.CollectionLookup = nil
    package.loaded.Validate = nil
    package.loaded.Log = nil
    return catalog, require 'HandlerCatalog'
end

describe("HandlerCatalog.listFolders", function()
    local function folderTree()
        local child = helper.fakeFolder("Day 1", "/photos/2026/Day 1", { {}, {} })
        local root = helper.fakeFolder("2026", "/photos/2026", { {} }, { child })
        return { root }
    end

    it("flattens the folder tree with paths, depth and parent", function()
        local _, Handler = setup({ folders = folderTree() })
        local r = Handler.listFolders({})

        assert.are.equal(2, r.count)
        assert.are.equal("2026", r.folders[1].name)
        assert.are.equal("/photos/2026", r.folders[1].path)
        assert.are.equal(0, r.folders[1].depth)
        assert.is_nil(r.folders[1].parent)

        assert.are.equal("Day 1", r.folders[2].name)
        assert.are.equal(1, r.folders[2].depth)
        assert.are.equal("/photos/2026", r.folders[2].parent)
    end)

    it("omits photo counts unless asked", function()
        local _, Handler = setup({ folders = folderTree() })
        assert.is_nil(Handler.listFolders({}).folders[1].photoCount)
    end)

    it("includes photo counts on request", function()
        local _, Handler = setup({ folders = folderTree() })
        local r = Handler.listFolders({ include_photo_counts = true })
        assert.are.equal(1, r.folders[1].photoCount)
        assert.are.equal(2, r.folders[2].photoCount)
    end)

    it("never calls getPhotos inside the read gate", function()
        -- folder:getPhotos() yields; nesting it in withReadAccessDo is the
        -- Windows deadlock from issues #124/#134.
        local insideGate = false
        local sawGetPhotosInsideGate = false
        local probe = {
            getName = function() return "F" end,
            getPath = function() return "/f" end,
            getChildren = function() return {} end,
            getPhotos = function()
                if insideGate then sawGetPhotosInsideGate = true end
                return {}
            end,
        }
        local catalog = helper.fakeCatalog({ folders = { probe } })
        local realGate = catalog.withReadAccessDo
        catalog.withReadAccessDo = function(selfRef, fn)
            insideGate = true
            realGate(selfRef, fn)
            insideGate = false
        end
        helper.installImport({
            LrApplication = { activeCatalog = function() return catalog end },
            LrLogger = helper.defaultLrLogger(),
        })
        package.loaded.HandlerCatalog = nil
        package.loaded.Validate = nil
        package.loaded.Log = nil
        local Handler = require 'HandlerCatalog'

        Handler.listFolders({ include_photo_counts = true })

        assert.is_false(sawGetPhotosInsideGate)
    end)

    it("paginates and reports has_more", function()
        local _, Handler = setup({ folders = folderTree() })
        local first = Handler.listFolders({ limit = 1 })
        assert.are.equal(1, #first.folders)
        assert.is_true(first.has_more)

        local second = Handler.listFolders({ limit = 1, offset = 1 })
        assert.are.equal("Day 1", second.folders[1].name)
        assert.is_false(second.has_more)
    end)

    it("returns an empty list for a catalog with no folders", function()
        local _, Handler = setup({})
        local r = Handler.listFolders({})
        assert.are.equal(0, r.count)
        assert.are.same({}, r.folders)
    end)
end)

describe("HandlerCatalog.listKeywords", function()
    local function keywordTree()
        local child = helper.fakeKeyword("Sunset", { photos = { {}, {} } })
        local root = helper.fakeKeyword("Nature", {
            children = { child },
            synonyms = { "outdoors" },
            photos = { {} },
        })
        return { root }
    end

    it("flattens the hierarchy with ' > ' paths", function()
        local _, Handler = setup({ keywords = keywordTree() })
        local r = Handler.listKeywords({})

        assert.are.equal(2, r.count)
        assert.are.equal("Nature", r.keywords[1].path)
        assert.are.equal("Nature > Sunset", r.keywords[2].path)
        assert.are.equal("Nature", r.keywords[2].parent)
        assert.are.equal(1, r.keywords[2].depth)
    end)

    it("reports synonyms and export flags", function()
        local _, Handler = setup({ keywords = keywordTree() })
        local r = Handler.listKeywords({})
        assert.are.same({ "outdoors" }, r.keywords[1].synonyms)
        assert.is_true(r.keywords[1].includeOnExport)
    end)

    it("omits an empty synonym list rather than emitting an empty array", function()
        local _, Handler = setup({ keywords = keywordTree() })
        assert.is_nil(Handler.listKeywords({}).keywords[2].synonyms)
    end)

    it("includes photo counts on request", function()
        local _, Handler = setup({ keywords = keywordTree() })
        local r = Handler.listKeywords({ include_photo_counts = true })
        assert.are.equal(1, r.keywords[1].photoCount)
        assert.are.equal(2, r.keywords[2].photoCount)
    end)

    it("paginates", function()
        local _, Handler = setup({ keywords = keywordTree() })
        local r = Handler.listKeywords({ limit = 1, offset = 1 })
        assert.are.equal(1, #r.keywords)
        assert.are.equal("Nature > Sunset", r.keywords[1].path)
        assert.is_false(r.has_more)
    end)

    it("returns an empty list for a catalog with no keywords", function()
        local _, Handler = setup({})
        assert.are.equal(0, Handler.listKeywords({}).count)
    end)
end)

describe("HandlerCatalog.getCatalogInfo", function()
    it("reports catalog path, version and photo count", function()
        local _, Handler = setup({
            path = "/catalogs/Studio.lrcat",
            photos = { helper.fakePhoto({ id = "1" }), helper.fakePhoto({ id = "2" }) },
        })
        local r = Handler.getCatalogInfo({})

        assert.is_true(r.success)
        assert.are.equal("/catalogs/Studio.lrcat", r.path)
        assert.are.equal(2, r.photoCount)
        assert.are.equal(14, r.lightroomVersion.major)
        assert.are.equal("14.2", r.lightroomVersionString)
    end)

    it("counts collections and sets across the whole tree", function()
        local nested = helper.fakeCollection("Picks", {})
        local innerSet = helper.fakeCollectionSet("Selects", { nested }, {})
        local outerSet = helper.fakeCollectionSet("2026", {}, { innerSet })
        local _, Handler = setup({
            collections = { helper.fakeCollection("Top", {}) },
            collectionSets = { outerSet },
        })

        local r = Handler.getCatalogInfo({})
        -- One top-level collection plus the one nested two sets deep.
        assert.are.equal(2, r.collectionCount)
        assert.are.equal(2, r.collectionSetCount)
    end)

    it("works on an empty catalog", function()
        local _, Handler = setup({})
        local r = Handler.getCatalogInfo({})
        assert.is_true(r.success)
        assert.are.equal(0, r.photoCount)
        assert.are.equal(0, r.collectionCount)
    end)
end)
