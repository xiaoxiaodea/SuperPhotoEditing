local helper = require 'spec_helper'

describe("HandlerSearch.searchPhotos", function()
    local catalog, Handler

    before_each(function()
        catalog = helper.fakeCatalog({
            photos = {
                helper.fakePhoto({ id = "1", path = "/a/sunset.jpg", fileName = "sunset.jpg", rating = 5, dateTimeOriginal = "2024-06-01" }),
                helper.fakePhoto({ id = "2", path = "/b/portrait.jpg", fileName = "portrait.jpg", rating = 3, dateTimeOriginal = "2024-06-02" }),
                helper.fakePhoto({ id = "3", path = "/c/landscape.jpg", fileName = "landscape.jpg", rating = 5, dateTimeOriginal = "2024-06-03" }),
            },
        })

        helper.installImport({
            LrApplication = { activeCatalog = function() return catalog end },
            LrLogger = helper.defaultLrLogger(),
        })

        package.loaded.HandlerSearch = nil
        package.loaded.Validate = nil
        Handler = require 'HandlerSearch'
    end)

    it("returns all photos when no filters given", function()
        local result = Handler.searchPhotos({})
        assert.are.equal(3, result.count)
        assert.are.equal(3, #result.photos)
        assert.is_false(result.has_more)
    end)

    it("filters by rating", function()
        local result = Handler.searchPhotos({ rating = 5 })
        assert.are.equal(2, result.count)
        for _, p in ipairs(result.photos) do
            assert.are.equal(5, p.rating)
        end
    end)

    it("filters by filename substring (case-insensitive)", function()
        local result = Handler.searchPhotos({ filename = "PORTRAIT" })
        assert.are.equal(1, result.count)
        assert.are.equal("portrait.jpg", result.photos[1].filename)
    end)

    it("filters by date range", function()
        local result = Handler.searchPhotos({ start_date = "2024-06-02", end_date = "2024-06-02" })
        assert.are.equal(1, result.count)
        assert.are.equal("2", result.photos[1].id)
    end)

    it("filters inclusively by start date", function()
        local result = Handler.searchPhotos({ start_date = "2024-06-02" })
        assert.are.equal(2, result.count)
        assert.are.equal("2", result.photos[1].id)
        assert.are.equal("3", result.photos[2].id)
    end)

    it("filters inclusively by end date", function()
        local result = Handler.searchPhotos({ end_date = "2024-06-02" })
        assert.are.equal(2, result.count)
        assert.are.equal("1", result.photos[1].id)
        assert.are.equal("2", result.photos[2].id)
    end)

    it("rejects malformed date filters before querying Lightroom", function()
        local ok, err = pcall(function()
            Handler.searchPhotos({ start_date = "2024-6-02", end_date = "2024-06-03" })
        end)
        assert.is_false(ok)
        assert.matches("start_date must use YYYY%-MM%-DD format", err)
    end)

    it("rejects invalid calendar dates before querying Lightroom", function()
        local ok, err = pcall(function()
            Handler.searchPhotos({ start_date = "2024-02-30" })
        end)
        assert.is_false(ok)
        assert.matches("start_date must use a valid calendar date", err)
    end)

    it("rejects reversed date ranges before querying Lightroom", function()
        local ok, err = pcall(function()
            Handler.searchPhotos({ start_date = "2024-06-03", end_date = "2024-06-02" })
        end)
        assert.is_false(ok)
        assert.matches("start_date must be on or before end_date", err)
    end)

    it("returns empty when no photo matches", function()
        local result = Handler.searchPhotos({ rating = 1 })
        assert.are.equal(0, result.count)
        assert.are.same({}, result.photos)
        assert.is_false(result.has_more)
    end)

    it("runs findPhotos OUTSIDE the read-access gate (#124 deadlock guard)", function()
        Handler.searchPhotos({ filename = "sunset" })
        assert.is_false(catalog.getQueriedInsideReadAccess())
    end)

    it("runs getAllPhotos OUTSIDE the read-access gate when no filters", function()
        Handler.searchPhotos({})
        assert.is_false(catalog.getQueriedInsideReadAccess())
    end)
end)

describe("HandlerSearch.searchPhotos pagination", function()
    local Handler

    before_each(function()
        local photos = {}
        for i = 1, 250 do
            table.insert(photos, helper.fakePhoto({
                id = tostring(i),
                path = "/p/" .. i .. ".jpg",
                fileName = "p" .. i .. ".jpg",
                rating = 0,
                dateTimeOriginal = "2024-06-01",
            }))
        end
        local catalog = helper.fakeCatalog({ photos = photos })
        helper.installImport({
            LrApplication = { activeCatalog = function() return catalog end },
            LrLogger = helper.defaultLrLogger(),
        })
        package.loaded.HandlerSearch = nil
        Handler = require 'HandlerSearch'
    end)

    it("caps to 100 by default and signals has_more", function()
        local r = Handler.searchPhotos({})
        assert.are.equal(250, r.count)
        assert.are.equal(100, #r.photos)
        assert.is_true(r.has_more)
        assert.are.equal("1", r.photos[1].id)
        assert.are.equal("100", r.photos[100].id)
    end)

    it("respects explicit limit up to total", function()
        local r = Handler.searchPhotos({ limit = 500 })
        assert.are.equal(250, r.count)
        assert.are.equal(250, #r.photos)
        assert.is_false(r.has_more)
    end)

    it("paginates via offset", function()
        local r = Handler.searchPhotos({ limit = 50, offset = 100 })
        assert.are.equal(250, r.count)
        assert.are.equal(50, #r.photos)
        assert.are.equal("101", r.photos[1].id)
        assert.are.equal("150", r.photos[50].id)
        assert.is_true(r.has_more)
    end)

    it("returns empty slice past the end without erroring", function()
        local r = Handler.searchPhotos({ offset = 1000 })
        assert.are.equal(250, r.count)
        assert.are.equal(0, #r.photos)
        assert.is_false(r.has_more)
    end)
end)

-- The descriptor is what Lightroom actually evaluates, and it is also persisted
-- verbatim by create_smart_collection, so these specs assert the emitted
-- criteria rather than only the filtered result set.
describe("HandlerSearch.buildSearchDesc", function()
    local Handler

    before_each(function()
        -- buildSearchDesc is pure, but the module imports LrApplication at load
        -- time, so the mock must be present even though no catalog is used.
        helper.installImport({
            LrApplication = { activeCatalog = function() return helper.fakeCatalog({}) end },
            LrLogger = helper.defaultLrLogger(),
        })
        package.loaded.HandlerSearch = nil
        package.loaded.Validate = nil
        Handler = require 'HandlerSearch'
    end)

    local function firstCriterion(args)
        local desc = Handler.buildSearchDesc(args)
        return desc[1], desc
    end

    it("defaults rating to an exact match and honours an operator", function()
        local exact = firstCriterion({ rating = 3 })
        assert.are.equal("rating", exact.criteria)
        assert.are.equal("==", exact.operation)

        local atLeast = firstCriterion({ rating = 3, rating_operation = ">=" })
        assert.are.equal(">=", atLeast.operation)
    end)

    it("maps flags onto SDK pick values", function()
        assert.are.equal(1, firstCriterion({ flag = "pick" }).value)
        assert.are.equal(0, firstCriterion({ flag = "unflag" }).value)
        assert.are.equal(-1, firstCriterion({ flag = "reject" }).value)
    end)

    it("maps color labels onto numeric indexes and keeps 'none' as a sentinel", function()
        local red = firstCriterion({ color_label = "red" })
        assert.are.equal("labelColor", red.criteria)
        assert.are.equal(1, red.value)
        assert.are.equal("none", firstCriterion({ color_label = "none" }).value)
    end)

    it("uses isTrue/isFalse for boolean criteria and emits no value", function()
        local hasGps = firstCriterion({ has_gps = true })
        assert.are.equal("hasGPSData", hasGps.criteria)
        assert.are.equal("isTrue", hasGps.operation)
        assert.is_nil(hasGps.value)

        assert.are.equal("isFalse", firstCriterion({ has_adjustments = false }).operation)
        assert.are.equal("cropped", firstCriterion({ is_cropped = true }).criteria)
    end)

    it("emits relative capture and edit windows with units", function()
        local captured = firstCriterion({ captured_in_last = 30 })
        assert.are.equal("captureTime", captured.criteria)
        assert.are.equal("inLast", captured.operation)
        assert.are.equal(30, captured.value)
        assert.are.equal("days", captured.value_units)

        local edited = firstCriterion({ edited_in_last = 2, edited_in_last_units = "weeks" })
        assert.are.equal("touchTime", edited.criteria)
        assert.are.equal("weeks", edited.value_units)
    end)

    it("maps EXIF and location filters to their SDK criteria names", function()
        assert.are.equal("camera", firstCriterion({ camera = "X-T5" }).criteria)
        assert.are.equal("lens", firstCriterion({ lens = "35mm" }).criteria)
        assert.are.equal("isoSpeedRating", firstCriterion({ iso = 800 }).criteria)
        assert.are.equal("fileFormat", firstCriterion({ file_format = "RAW" }).criteria)
        assert.are.equal("aspectRatio", firstCriterion({ aspect_ratio = "portrait" }).criteria)
        assert.are.equal("treatment", firstCriterion({ treatment = "grayscale" }).criteria)
        assert.are.equal("city", firstCriterion({ city = "Guangzhou" }).criteria)
        assert.are.equal("folder", firstCriterion({ folder = "2026" }).criteria)
    end)

    it("supports free-text search with a selectable operation", function()
        local anyText = firstCriterion({ text = "sunset" })
        assert.are.equal("all", anyText.criteria)
        assert.are.equal("any", anyText.operation)
        assert.are.equal("beginsWith",
            firstCriterion({ text = "IMG", text_operation = "beginsWith" }).operation)
    end)

    it("defaults to intersect and combines multiple filters", function()
        local _, desc = firstCriterion({ rating = 5, flag = "pick", camera = "X-T5" })
        assert.are.equal(3, #desc)
        assert.are.equal("intersect", desc.combine)
    end)

    it("emits no criteria when nothing is filtered", function()
        assert.are.equal(0, #Handler.buildSearchDesc({}))
    end)

    it("rejects values outside each documented enum", function()
        assert.has_error(function() Handler.buildSearchDesc({ flag = "maybe" }) end)
        assert.has_error(function() Handler.buildSearchDesc({ color_label = "orange" }) end)
        assert.has_error(function() Handler.buildSearchDesc({ file_format = "HEIF" }) end)
        assert.has_error(function() Handler.buildSearchDesc({ treatment = "sepia" }) end)
        assert.has_error(function() Handler.buildSearchDesc({ aspect_ratio = "wide" }) end)
        assert.has_error(function() Handler.buildSearchDesc({ rating_operation = "=~" }) end)
        assert.has_error(function() Handler.buildSearchDesc({ text_operation = "fuzzy" }) end)
        assert.has_error(function()
            Handler.buildSearchDesc({ captured_in_last = 1, captured_in_last_units = "fortnights" })
        end)
    end)

    it("rejects a modifier whose base filter is missing", function()
        -- A modifier only takes effect alongside its base filter, so a dangling
        -- one would silently return an unfiltered set that looks like a real
        -- search. Validating it up front turns that into a clear error.
        assert.has_error(function() Handler.buildSearchDesc({ rating_operation = ">=" }) end)
        assert.has_error(function() Handler.buildSearchDesc({ iso_operation = ">" }) end)
        assert.has_error(function() Handler.buildSearchDesc({ text_operation = "beginsWith" }) end)
        assert.has_error(function()
            Handler.buildSearchDesc({ captured_in_last_units = "weeks" })
        end)
        assert.has_error(function()
            Handler.buildSearchDesc({ edited_in_last_units = "weeks" })
        end)
    end)
end)

describe("HandlerSearch.searchPhotos extended filters", function()
    local catalog, Handler

    before_each(function()
        catalog = helper.fakeCatalog({
            photos = {
                helper.fakePhoto({
                    id = "1", path = "/a/keep.NEF", fileName = "keep.NEF",
                    rating = 5, pickStatus = 1, colorNameForLabel = "red",
                    fileFormat = "RAW", cameraModel = "X-T5", isoSpeedRating = 400,
                    gps = { latitude = 1, longitude = 2 },
                }),
                helper.fakePhoto({
                    id = "2", path = "/b/skip.jpg", fileName = "skip.jpg",
                    rating = 2, pickStatus = -1, fileFormat = "JPG",
                    cameraModel = "A7IV", isoSpeedRating = 6400,
                }),
            },
        })
        helper.installImport({
            LrApplication = { activeCatalog = function() return catalog end },
            LrLogger = helper.defaultLrLogger(),
        })
        package.loaded.HandlerSearch = nil
        package.loaded.Validate = nil
        Handler = require 'HandlerSearch'
    end)

    it("filters by flag", function()
        local r = Handler.searchPhotos({ flag = "pick" })
        assert.are.equal(1, r.count)
        assert.are.equal("1", r.photos[1].id)
    end)

    it("filters by color label", function()
        assert.are.equal(1, Handler.searchPhotos({ color_label = "red" }).count)
        assert.are.equal(1, Handler.searchPhotos({ color_label = "none" }).count)
    end)

    it("filters by rating with a comparison operator", function()
        local r = Handler.searchPhotos({ rating = 3, rating_operation = ">=" })
        assert.are.equal(1, r.count)
        assert.are.equal("1", r.photos[1].id)
    end)

    it("filters by ISO with a comparison operator", function()
        local r = Handler.searchPhotos({ iso = 1600, iso_operation = ">" })
        assert.are.equal(1, r.count)
        assert.are.equal("2", r.photos[1].id)
    end)

    it("filters by GPS presence", function()
        assert.are.equal("1", Handler.searchPhotos({ has_gps = true }).photos[1].id)
        assert.are.equal("2", Handler.searchPhotos({ has_gps = false }).photos[1].id)
    end)

    it("filters by camera and file format", function()
        assert.are.equal(1, Handler.searchPhotos({ camera = "x-t5" }).count)
        assert.are.equal(1, Handler.searchPhotos({ file_format = "RAW" }).count)
    end)

    it("combines filters with AND by default", function()
        assert.are.equal(0, Handler.searchPhotos({ flag = "pick", file_format = "JPG" }).count)
    end)

    it("combines filters with OR when match is any", function()
        local r = Handler.searchPhotos({ flag = "pick", file_format = "JPG", match = "any" })
        assert.are.equal(2, r.count)
    end)

    it("passes sort and direction through to findPhotos", function()
        local r = Handler.searchPhotos({ file_format = "RAW", sort = "fileName", ascending = true })
        assert.are.equal(1, r.count)
    end)

    it("rejects an unsupported sort key", function()
        assert.has_error(function() Handler.searchPhotos({ sort = "rating" }) end)
    end)

    it("keeps extended queries outside the read gate", function()
        Handler.searchPhotos({ flag = "pick", camera = "X-T5" })
        assert.is_false(catalog.getQueriedInsideReadAccess())
    end)
end)
