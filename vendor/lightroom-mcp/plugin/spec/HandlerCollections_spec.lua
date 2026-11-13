local helper = require 'spec_helper'

local function setup(opts)
    opts = opts or {}
    local catalog = helper.fakeCatalog(opts)
    helper.installImport({
        LrApplication = { activeCatalog = function() return catalog end },
        LrLogger = helper.defaultLrLogger(),
    })
    package.loaded.HandlerCollections = nil
    package.loaded.CollectionLookup = nil
    package.loaded.HandlerSearch = nil
    package.loaded.Validate = nil
    return catalog, require 'HandlerCollections'
end

describe("HandlerCollections.listCollections", function()
    it("lists top-level collections", function()
        local _, Handler = setup({
            collections = {
                helper.fakeCollection("Trip", { 1, 2, 3 }),
                helper.fakeCollection("Family", { 1 }),
            },
        })
        local r = Handler.listCollections({})
        assert.are.equal(2, r.count)
        assert.are.equal("Trip", r.collections[1].name)
        assert.are.equal(3, r.collections[1].photoCount)
        assert.is_false(r.has_more)
    end)

    it("caps and paginates", function()
        local cols = {}
        for i = 1, 150 do
            table.insert(cols, helper.fakeCollection("c" .. i, {}))
        end
        local _, Handler = setup({ collections = cols })

        local r1 = Handler.listCollections({})
        assert.are.equal(150, r1.count)
        assert.are.equal(100, #r1.collections)
        assert.is_true(r1.has_more)

        local r2 = Handler.listCollections({ limit = 50, offset = 100 })
        assert.are.equal(150, r2.count)
        assert.are.equal(50, #r2.collections)
        assert.are.equal("c101", r2.collections[1].name)
        assert.is_false(r2.has_more)
    end)

    it("descends into collection sets and prefixes names", function()
        local nested = helper.fakeCollection("Inside", {})
        local outerSet = {
            getName = function() return "Outer" end,
            getChildCollections = function() return { nested } end,
            getChildCollectionSets = function() return {} end,
        }
        local _, Handler = setup({ collectionSets = { outerSet } })
        local r = Handler.listCollections({})
        assert.are.equal(1, r.count)
        assert.are.equal("Outer / Inside", r.collections[1].name)
        assert.are.equal("Outer", r.collections[1].parent)
    end)
end)

describe("HandlerCollections.createCollection", function()
    it("creates a collection with the given name", function()
        local catalog, Handler = setup({})
        local r = Handler.createCollection({ name = "New Album" })
        assert.is_true(r.success)
        local created = catalog.getCreatedCollections()
        assert.are.equal(1, #created)
        assert.are.equal("New Album", created[1].name)
        assert.are.equal("New Album", created[1].collection.getName())
    end)

    it("creates the collection inside the requested parent set", function()
        local parentSet = helper.fakeCollectionSet("2026", {}, {})
        local catalog, Handler = setup({ collectionSets = { parentSet } })

        local r = Handler.createCollection({ name = "Picks", parent = "2026" })

        -- Regression: `parent` used to be accepted and silently ignored, so
        -- every collection landed at the top level while reporting success.
        local created = catalog.getCreatedCollections()
        assert.are.equal(parentSet, created[1].parent)
        assert.are.equal("2026", r.parent)
        assert.are.equal(1, #parentSet.getChildCollections())
    end)

    it("resolves a nested parent set by full path", function()
        local innerSet = helper.fakeCollectionSet("Selects", {}, {})
        local outerSet = helper.fakeCollectionSet("2026", {}, { innerSet })
        local catalog, Handler = setup({ collectionSets = { outerSet } })

        Handler.createCollection({ name = "Final", parent = "2026 / Selects" })

        assert.are.equal(innerSet, catalog.getCreatedCollections()[1].parent)
    end)

    it("errors when the named parent set does not exist", function()
        local _, Handler = setup({})
        assert.has_error(function()
            Handler.createCollection({ name = "Picks", parent = "Nope" })
        end)
    end)

    it("errors without name", function()
        local _, Handler = setup({})
        assert.has_error(function() Handler.createCollection({}) end)
    end)
end)

describe("HandlerCollections.listCollectionSets", function()
    it("lists nested sets with full paths and child counts", function()
        local nested = helper.fakeCollection("Inside", {})
        local innerSet = helper.fakeCollectionSet("Selects", { nested }, {})
        local outerSet = helper.fakeCollectionSet("2026", {}, { innerSet })
        local _, Handler = setup({ collectionSets = { outerSet } })

        local r = Handler.listCollectionSets({})

        assert.are.equal(2, r.count)
        assert.are.equal("2026", r.collection_sets[1].name)
        assert.are.equal(1, r.collection_sets[1].setCount)
        assert.are.equal("2026 / Selects", r.collection_sets[2].name)
        assert.are.equal(1, r.collection_sets[2].collectionCount)
    end)

    it("returns an empty list when there are no sets", function()
        local _, Handler = setup({})
        assert.are.equal(0, Handler.listCollectionSets({}).count)
    end)
end)

describe("HandlerCollections.createCollectionSet", function()
    it("creates a top-level set", function()
        local catalog, Handler = setup({})
        local r = Handler.createCollectionSet({ name = "2026" })
        assert.is_true(r.success)
        assert.are.equal("2026", catalog.getCreatedCollectionSets()[1].name)
    end)

    it("nests a set inside another set", function()
        local parentSet = helper.fakeCollectionSet("2026", {}, {})
        local catalog, Handler = setup({ collectionSets = { parentSet } })
        Handler.createCollectionSet({ name = "Selects", parent = "2026" })
        assert.are.equal(parentSet, catalog.getCreatedCollectionSets()[1].parent)
    end)
end)

describe("HandlerCollections.createSmartCollection", function()
    it("builds the search descriptor from the shared criteria vocabulary", function()
        local catalog, Handler = setup({})
        local r = Handler.createSmartCollection({
            name = "Keepers",
            criteria = { rating = 4, rating_operation = ">=" },
        })

        assert.is_true(r.success)
        local created = catalog.getCreatedSmartCollections()[1]
        assert.are.equal("Keepers", created.name)
        assert.are.equal("rating", created.searchDesc[1].criteria)
        assert.are.equal(">=", created.searchDesc[1].operation)
        assert.are.equal(4, created.searchDesc[1].value)
        assert.are.equal("intersect", created.searchDesc.combine)
    end)

    it("uses union when match is any", function()
        local catalog, Handler = setup({})
        Handler.createSmartCollection({
            name = "Either",
            criteria = { flag = "pick", color_label = "red" },
            match = "any",
        })
        assert.are.equal("union", catalog.getCreatedSmartCollections()[1].searchDesc.combine)
    end)

    it("errors when criteria contains no usable filter", function()
        local _, Handler = setup({})
        assert.has_error(function()
            Handler.createSmartCollection({ name = "Empty", criteria = {} })
        end)
    end)

    it("propagates criteria validation errors", function()
        local _, Handler = setup({})
        assert.has_error(function()
            Handler.createSmartCollection({ name = "Bad", criteria = { flag = "maybe" } })
        end)
    end)
end)

describe("HandlerCollections.getCollectionPhotos", function()
    it("lists the photos in a collection with pagination", function()
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg", fileName = "a.jpg", rating = 5 })
        local p2 = helper.fakePhoto({ id = "2", path = "/b.jpg", fileName = "b.jpg" })
        local target = helper.fakeCollection("Target", { p1, p2 })
        local _, Handler = setup({ collections = { target } })

        local r = Handler.getCollectionPhotos({ collection_name = "Target" })
        assert.are.equal(2, r.count)
        assert.are.equal("a.jpg", r.photos[1].filename)
        assert.are.equal(5, r.photos[1].rating)

        local page = Handler.getCollectionPhotos({ collection_name = "Target", limit = 1, offset = 1 })
        assert.are.equal("b.jpg", page.photos[1].filename)
        assert.is_false(page.has_more)
    end)

    it("never reads collection membership inside the read gate", function()
        -- getPhotos() yields, so it must stay outside withReadAccessDo.
        local target = helper.fakeCollection("Target", {})
        local catalog = helper.fakeCatalog({ collections = { target } })
        local insideGate = false
        local sawGetPhotosInsideGate = false
        local realGetPhotos = target.getPhotos
        target.getPhotos = function(...)
            if insideGate then sawGetPhotosInsideGate = true end
            return realGetPhotos(...)
        end
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
        package.loaded.HandlerCollections = nil
        package.loaded.CollectionLookup = nil
        package.loaded.Validate = nil
        local Handler = require 'HandlerCollections'

        Handler.getCollectionPhotos({ collection_name = "Target" })

        assert.is_false(sawGetPhotosInsideGate)
    end)

    it("errors for an unknown collection", function()
        local _, Handler = setup({})
        assert.has_error(function()
            Handler.getCollectionPhotos({ collection_name = "Nope" })
        end)
    end)
end)

describe("HandlerCollections.removeFromCollection", function()
    it("removes the requested photos and leaves the rest", function()
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg" })
        local p2 = helper.fakePhoto({ id = "2", path = "/b.jpg" })
        local target = helper.fakeCollection("Target", { p1, p2 })
        local _, Handler = setup({ photos = { p1, p2 }, collections = { target } })

        local r = Handler.removeFromCollection({
            collection_name = "Target", photo_ids = { "1" },
        })

        assert.is_true(r.success)
        assert.are.equal(1, r.removed)
        assert.are.equal(1, #target.getPhotos())
        assert.are.equal(p2, target.getPhotos()[1])
    end)

    it("reports ids it could not resolve", function()
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg" })
        local target = helper.fakeCollection("Target", { p1 })
        local _, Handler = setup({ photos = { p1 }, collections = { target } })

        local r = Handler.removeFromCollection({
            collection_name = "Target", photo_ids = { "1", "ghost" },
        })

        assert.are.equal(1, r.removed)
        assert.are.same({ "ghost" }, r.not_found)
    end)

    it("refuses to modify a smart collection", function()
        local smart = helper.fakeCollection("Smart", {}, { isSmart = true })
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg" })
        local _, Handler = setup({ photos = { p1 }, collections = { smart } })

        assert.has_error(function()
            Handler.removeFromCollection({ collection_name = "Smart", photo_ids = { "1" } })
        end)
    end)
end)

describe("HandlerCollections.renameCollection", function()
    it("renames the collection and reports both names", function()
        local target = helper.fakeCollection("Old", {})
        local _, Handler = setup({ collections = { target } })

        local r = Handler.renameCollection({ collection_name = "Old", new_name = "New" })

        assert.is_true(r.success)
        assert.are.equal("Old", r.previous_name)
        assert.are.equal("New", r.name)
        assert.are.equal("New", target.getName())
    end)

    it("errors when the new name is already taken in that parent", function()
        -- setName returns false rather than throwing, so the handler must turn
        -- that into an error instead of reporting a rename that did not happen.
        local target = helper.fakeCollection("Old", {}, { nameTaken = "Taken" })
        local _, Handler = setup({ collections = { target } })

        assert.has_error(function()
            Handler.renameCollection({ collection_name = "Old", new_name = "Taken" })
        end)
        assert.are.equal("Old", target.getName())
    end)

    it("errors without both names", function()
        local _, Handler = setup({ collections = { helper.fakeCollection("Old", {}) } })
        assert.has_error(function() Handler.renameCollection({ collection_name = "Old" }) end)
        assert.has_error(function() Handler.renameCollection({ new_name = "New" }) end)
    end)
end)

describe("HandlerCollections.deleteCollection", function()
    it("deletes the collection and states that photos are kept", function()
        local target = helper.fakeCollection("Doomed", { 1, 2, 3 })
        local _, Handler = setup({ collections = { target } })

        local r = Handler.deleteCollection({ collection_name = "Doomed" })

        assert.is_true(r.success)
        assert.is_true(target.isDeleted())
        assert.are.equal(3, r.released_photos)
        -- Guards against over-reporting the blast radius to the user.
        assert.is_false(r.photos_deleted)
    end)

    it("errors for an unknown collection", function()
        local _, Handler = setup({})
        assert.has_error(function()
            Handler.deleteCollection({ collection_name = "Ghost" })
        end)
    end)
end)

describe("collection name resolution", function()
    it("finds a collection nested inside a set by bare name", function()
        local nested = helper.fakeCollection("Inside", {})
        local outerSet = helper.fakeCollectionSet("Outer", { nested }, {})
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg" })
        local _, Handler = setup({ photos = { p1 }, collectionSets = { outerSet } })

        -- Previously only top-level collections were searched, so adding to a
        -- nested collection failed even though list_collections showed it.
        local r = Handler.addToCollection({ collection_name = "Inside", photo_ids = { "1" } })

        assert.are.equal(1, r.added)
        assert.are.equal("Outer / Inside", r.collection)
    end)

    it("accepts the full path shown by list_collections", function()
        local nested = helper.fakeCollection("Inside", {})
        local outerSet = helper.fakeCollectionSet("Outer", { nested }, {})
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg" })
        local _, Handler = setup({ photos = { p1 }, collectionSets = { outerSet } })

        local r = Handler.addToCollection({
            collection_name = "Outer / Inside", photo_ids = { "1" },
        })

        assert.are.equal(1, r.added)
    end)

    it("errors instead of guessing when a bare name is ambiguous", function()
        local a = helper.fakeCollection("Picks", {})
        local b = helper.fakeCollection("Picks", {})
        local setA = helper.fakeCollectionSet("A", { a }, {})
        local setB = helper.fakeCollectionSet("B", { b }, {})
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg" })
        local _, Handler = setup({ photos = { p1 }, collectionSets = { setA, setB } })

        -- Silently choosing one would write photos into the wrong collection.
        local ok, err = pcall(function()
            Handler.addToCollection({ collection_name = "Picks", photo_ids = { "1" } })
        end)

        assert.is_false(ok)
        assert.is_truthy(tostring(err):find("ambiguous", 1, true))
        assert.are.equal(0, #a.getAddedPhotos())
        assert.are.equal(0, #b.getAddedPhotos())
    end)

    it("still resolves an ambiguous name when given its full path", function()
        local a = helper.fakeCollection("Picks", {})
        local b = helper.fakeCollection("Picks", {})
        local setA = helper.fakeCollectionSet("A", { a }, {})
        local setB = helper.fakeCollectionSet("B", { b }, {})
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg" })
        local _, Handler = setup({ photos = { p1 }, collectionSets = { setA, setB } })

        Handler.addToCollection({ collection_name = "B / Picks", photo_ids = { "1" } })

        assert.are.equal(0, #a.getAddedPhotos())
        assert.are.equal(1, #b.getAddedPhotos())
    end)
end)

describe("HandlerCollections.addToCollection", function()
    it("adds matching photos to the named collection", function()
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg" })
        local p2 = helper.fakePhoto({ id = "2", path = "/b.jpg" })
        local target = helper.fakeCollection("Target", {})
        local _, Handler = setup({ photos = { p1, p2 }, collections = { target } })

        local r = Handler.addToCollection({
            collection_name = "Target",
            photo_ids = { "1", "2" },
        })

        assert.is_true(r.success)
        assert.are.equal(2, r.added)
        assert.are.equal(2, #target.getAddedPhotos())
    end)

    it("errors when collection not found", function()
        local _, Handler = setup({})
        assert.has_error(function()
            Handler.addToCollection({ collection_name = "Nope", photo_ids = { "1" } })
        end)
    end)

    it("refuses to add photos to a smart collection", function()
        -- addPhotos throws on a smart collection; a named error is clearer than
        -- the SDK's and avoids aborting the transaction on a foreseeable case.
        local smart = helper.fakeCollection("Smart", {}, { isSmart = true })
        local p1 = helper.fakePhoto({ id = "1", path = "/a.jpg" })
        local _, Handler = setup({ photos = { p1 }, collections = { smart } })

        local ok, err = pcall(function()
            Handler.addToCollection({ collection_name = "Smart", photo_ids = { "1" } })
        end)

        assert.is_false(ok)
        assert.is_truthy(tostring(err):find("smart collection", 1, true))
    end)

    it("marks smart collections in the listing", function()
        local _, Handler = setup({
            collections = { helper.fakeCollection("Smart", {}, { isSmart = true }) },
        })
        assert.is_true(Handler.listCollections({}).collections[1].isSmart)
    end)

    it("errors without required args", function()
        local _, Handler = setup({})
        assert.has_error(function() Handler.addToCollection({ photo_ids = { "1" } }) end)
        assert.has_error(function() Handler.addToCollection({ collection_name = "X" }) end)
    end)
end)
