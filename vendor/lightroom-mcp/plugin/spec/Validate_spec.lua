local helper = require 'spec_helper'

local function setup()
    helper.installImport({ LrLogger = helper.defaultLrLogger() })
    package.loaded.Validate = nil
    return require 'Validate'
end

describe("Validate.requireStringArray", function()
    it("accepts a dense array of non-empty strings", function()
        local Validate = setup()
        assert.are.same({ "a", "b" }, Validate.requireStringArray({ "a", "b" }, "ids"))
    end)

    it("rejects a table with string keys", function()
        local Validate = setup()
        -- A JSON object would otherwise iterate as empty under ipairs and the
        -- handler would report "updated 0" instead of an error.
        assert.has_error(function()
            Validate.requireStringArray({ first = "a" }, "ids")
        end)
    end)

    it("rejects a sparse array with holes", function()
        local Validate = setup()
        assert.has_error(function()
            Validate.requireStringArray({ [1] = "a", [3] = "b" }, "ids")
        end)
    end)

    it("rejects empty and non-string entries", function()
        local Validate = setup()
        assert.has_error(function() Validate.requireStringArray({}, "ids") end)
        assert.has_error(function() Validate.requireStringArray({ "a", "" }, "ids") end)
        assert.has_error(function() Validate.requireStringArray({ "a", 5 }, "ids") end)
        assert.has_error(function() Validate.requireStringArray("a", "ids") end)
    end)

    it("enforces maxItems", function()
        local Validate = setup()
        assert.has_error(function()
            Validate.requireStringArray({ "a", "b", "c" }, "ids", 2)
        end)
    end)

    it("caps photo id arrays at the documented bulk limit", function()
        local Validate = setup()
        local ids = {}
        for i = 1, Validate.MAX_BULK_PHOTO_IDS + 1 do ids[i] = tostring(i) end
        assert.has_error(function() Validate.requirePhotoIds(ids) end)
    end)
end)

describe("Validate.requireOneOf", function()
    it("returns the value when allowed", function()
        local Validate = setup()
        assert.are.equal("pick", Validate.requireOneOf("pick", "flag", { "pick", "reject" }))
    end)

    it("lists the allowed values in the error", function()
        local Validate = setup()
        local ok, err = pcall(function()
            Validate.requireOneOf("maybe", "flag", { "pick", "reject" })
        end)
        assert.is_false(ok)
        assert.is_truthy(tostring(err):find("pick, reject", 1, true))
    end)
end)

describe("Validate.requireIntegerInRange", function()
    it("accepts an in-range integer", function()
        local Validate = setup()
        assert.are.equal(512, Validate.requireIntegerInRange(512, "max_edge", 32, 2048))
    end)

    it("rejects fractions and out-of-range values", function()
        local Validate = setup()
        assert.has_error(function()
            Validate.requireIntegerInRange(1.5, "max_edge", 32, 2048)
        end)
        assert.has_error(function()
            Validate.requireIntegerInRange(1, "max_edge", 32, 2048)
        end)
        assert.has_error(function()
            Validate.requireIntegerInRange("x", "max_edge", 32, 2048)
        end)
    end)
end)

describe("Validate.pagination", function()
    it("defaults to 100 and offset 0", function()
        local Validate = setup()
        local limit, offset = Validate.pagination({})
        assert.are.equal(100, limit)
        assert.are.equal(0, offset)
    end)

    it("floors fractional values", function()
        local Validate = setup()
        -- A fractional offset would index matches[i] with a non-integer key,
        -- which is always nil in Lua and crashes the per-row builder.
        local limit, offset = Validate.pagination({ limit = 10.7, offset = 2.9 })
        assert.are.equal(10, limit)
        assert.are.equal(2, offset)
    end)

    it("clamps negatives to zero", function()
        local Validate = setup()
        local limit, offset = Validate.pagination({ limit = -5, offset = -3 })
        assert.are.equal(0, limit)
        assert.are.equal(0, offset)
    end)
end)

describe("Validate.paginate", function()
    it("slices the requested page and reports has_more", function()
        local Validate = setup()
        local items = { "a", "b", "c", "d" }

        local slice, total, hasMore = Validate.paginate(items, 2, 0)
        assert.are.same({ "a", "b" }, slice)
        assert.are.equal(4, total)
        assert.is_true(hasMore)

        local last, _, moreAfterLast = Validate.paginate(items, 2, 2)
        assert.are.same({ "c", "d" }, last)
        assert.is_false(moreAfterLast)
    end)

    it("returns an empty slice past the end", function()
        local Validate = setup()
        local slice, total, hasMore = Validate.paginate({ "a" }, 10, 5)
        assert.are.same({}, slice)
        assert.are.equal(1, total)
        assert.is_false(hasMore)
    end)
end)
