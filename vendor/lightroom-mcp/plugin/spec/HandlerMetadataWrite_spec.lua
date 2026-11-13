local helper = require 'spec_helper'

-- Load HandlerMetadataWrite with a fake catalog installed.
local function setup(opts)
    local catalog = helper.fakeCatalog(opts or {})
    helper.installImport({
        LrApplication = { activeCatalog = function() return catalog end },
        LrLogger = helper.defaultLrLogger(),
        LrPathUtils = {
            child = function(a, b) return a .. "/" .. b end,
            getStandardFilePath = function() return "/home" end,
        },
        LrFileUtils = { createAllDirectories = function() end },
    })
    package.loaded.HandlerMetadataWrite = nil
    package.loaded.Validate = nil
    package.loaded.PhotoLookup = nil
    package.loaded.Log = nil
    return catalog, require 'HandlerMetadataWrite'
end

local function photos()
    return {
        helper.fakePhoto({ id = "1", path = "/p/a.jpg", fileName = "a.jpg" }),
        helper.fakePhoto({ id = "2", path = "/p/b.jpg", fileName = "b.jpg" }),
    }
end

describe("HandlerMetadataWrite.setPhotoMetadata", function()
    it("writes text fields to every requested photo", function()
        local catalog, Handler = setup({ photos = photos() })
        local r = Handler.setPhotoMetadata({
            photo_ids = { "1", "2" },
            metadata = { copyright = "(c) 2026 Studio", city = "Guangzhou" },
        })

        assert.is_true(r.success)
        assert.are.equal(2, r.updated)
        -- Reported fields are the caller-facing keys, sorted for determinism.
        assert.are.same({ "city", "copyright" }, r.fields)

        local target = catalog.findPhotoByLocalIdentifier(nil, "1")
        assert.are.equal("(c) 2026 Studio", target:getRawMetadata('copyright'))
        assert.are.equal("Guangzhou", target:getRawMetadata('city'))
    end)

    it("opens exactly one write gate for the whole batch", function()
        local catalog, Handler = setup({ photos = photos() })
        Handler.setPhotoMetadata({ photo_ids = { "1", "2" }, metadata = { title = "T" } })
        assert.are.equal(1, catalog.getWriteAccessCount())
    end)

    it("names the undo action so Lightroom's Undo menu is meaningful", function()
        local catalog, Handler = setup({ photos = photos() })
        Handler.setPhotoMetadata({ photo_ids = { "1" }, metadata = { title = "T" } })
        assert.are.same({ "Set Photo Metadata" }, catalog.getWriteActionNames())
    end)

    it("clears a text field when given an empty string", function()
        local catalog, Handler = setup({
            photos = { helper.fakePhoto({ id = "1", caption = "old" }) },
        })
        Handler.setPhotoMetadata({ photo_ids = { "1" }, metadata = { caption = "" } })
        assert.are.equal("", catalog.findPhotoByLocalIdentifier(nil, "1"):getRawMetadata('caption'))
    end)

    it("reports unresolved ids without failing the batch", function()
        local _, Handler = setup({ photos = photos() })
        local r = Handler.setPhotoMetadata({
            photo_ids = { "1", "missing" },
            metadata = { title = "T" },
        })
        assert.are.equal(1, r.updated)
        assert.are.same({ "missing" }, r.not_found)
    end)

    it("writes GPS as a latitude/longitude table", function()
        local catalog, Handler = setup({ photos = photos() })
        Handler.setPhotoMetadata({
            photo_ids = { "1" },
            metadata = { gps = { latitude = 23.13, longitude = 113.26 } },
        })
        local gps = catalog.findPhotoByLocalIdentifier(nil, "1"):getRawMetadata('gps')
        assert.are.equal(23.13, gps.latitude)
        assert.are.equal(113.26, gps.longitude)
    end)

    it("clears GPS when passed false", function()
        local catalog, Handler = setup({
            photos = { helper.fakePhoto({ id = "1", gps = { latitude = 1, longitude = 2 } }) },
        })
        Handler.setPhotoMetadata({ photo_ids = { "1" }, metadata = { gps = false } })
        assert.is_nil(catalog.findPhotoByLocalIdentifier(nil, "1"):getRawMetadata('gps'))
    end)

    it("normalizes a color label onto the writable 'label' key", function()
        local catalog, Handler = setup({ photos = photos() })
        Handler.setPhotoMetadata({ photo_ids = { "1" }, metadata = { colorNameForLabel = "Red" } })

        -- The SDK reads labels via colorNameForLabel but writes via 'label';
        -- assert the actual key used so the mapping cannot silently regress.
        local writes = catalog.findPhotoByLocalIdentifier(nil, "1").getWrites()
        assert.are.equal(1, #writes)
        assert.are.equal("label", writes[1].key)
        assert.are.equal("red", writes[1].value)
    end)

    it("clears the label when color is none", function()
        local catalog, Handler = setup({ photos = photos() })
        Handler.setPhotoMetadata({ photo_ids = { "1" }, metadata = { colorNameForLabel = "none" } })
        local writes = catalog.findPhotoByLocalIdentifier(nil, "1").getWrites()
        assert.are.equal("label", writes[1].key)
        assert.are.equal("", writes[1].value)
    end)

    it("rejects an unsupported metadata key before touching the catalog", function()
        local catalog, Handler = setup({ photos = photos() })
        assert.has_error(function()
            Handler.setPhotoMetadata({ photo_ids = { "1" }, metadata = { notAField = "x" } })
        end)
        -- Validation must happen before the gate opens, otherwise a bad key
        -- rolls back photos already written in the same transaction.
        assert.are.equal(0, catalog.getWriteAccessCount())
    end)

    it("rejects out-of-range GPS coordinates", function()
        local _, Handler = setup({ photos = photos() })
        assert.has_error(function()
            Handler.setPhotoMetadata({
                photo_ids = { "1" },
                metadata = { gps = { latitude = 91, longitude = 0 } },
            })
        end)
        assert.has_error(function()
            Handler.setPhotoMetadata({
                photo_ids = { "1" },
                metadata = { gps = { latitude = 0, longitude = 181 } },
            })
        end)
    end)

    it("rejects a non-string value for a text field", function()
        local _, Handler = setup({ photos = photos() })
        assert.has_error(function()
            Handler.setPhotoMetadata({ photo_ids = { "1" }, metadata = { title = 42 } })
        end)
    end)

    it("rejects an invalid copyrightState", function()
        local _, Handler = setup({ photos = photos() })
        assert.has_error(function()
            Handler.setPhotoMetadata({
                photo_ids = { "1" },
                metadata = { copyrightState = "probably mine" },
            })
        end)
    end)

    it("errors without photo_ids or metadata", function()
        local _, Handler = setup({ photos = photos() })
        assert.has_error(function() Handler.setPhotoMetadata({ metadata = { title = "T" } }) end)
        assert.has_error(function() Handler.setPhotoMetadata({ photo_ids = { "1" } }) end)
        assert.has_error(function()
            Handler.setPhotoMetadata({ photo_ids = { "1" }, metadata = {} })
        end)
    end)

    it("rejects a photo_ids object instead of an array", function()
        local _, Handler = setup({ photos = photos() })
        -- A JSON object would otherwise iterate as empty and report "updated 0".
        assert.has_error(function()
            Handler.setPhotoMetadata({ photo_ids = { first = "1" }, metadata = { title = "T" } })
        end)
    end)
end)

describe("HandlerMetadataWrite.setFlag", function()
    it("maps pick / unflag / reject onto SDK pickStatus values", function()
        for flag, expected in pairs({ pick = 1, unflag = 0, reject = -1 }) do
            local catalog, Handler = setup({ photos = photos() })
            local r = Handler.setFlag({ photo_ids = { "1" }, flag = flag })
            assert.are.equal(expected, r.pick_status)
            assert.are.equal(expected,
                catalog.findPhotoByLocalIdentifier(nil, "1"):getRawMetadata('pickStatus'))
        end
    end)

    it("flags every requested photo and reports the count", function()
        local _, Handler = setup({ photos = photos() })
        local r = Handler.setFlag({ photo_ids = { "1", "2" }, flag = "reject" })
        assert.are.equal(2, r.updated)
        assert.are.equal("reject", r.flag)
    end)

    it("rejects an unknown flag word", function()
        local _, Handler = setup({ photos = photos() })
        assert.has_error(function()
            Handler.setFlag({ photo_ids = { "1" }, flag = "maybe" })
        end)
    end)

    it("errors without photo_ids or flag", function()
        local _, Handler = setup({ photos = photos() })
        assert.has_error(function() Handler.setFlag({ flag = "pick" }) end)
        assert.has_error(function() Handler.setFlag({ photo_ids = { "1" } }) end)
    end)
end)

describe("HandlerMetadataWrite.setColorLabel", function()
    it("sets a label on every requested photo", function()
        local catalog, Handler = setup({ photos = photos() })
        local r = Handler.setColorLabel({ photo_ids = { "1", "2" }, color = "blue" })
        assert.are.equal(2, r.updated)
        assert.are.equal("blue", r.color)
        assert.are.equal("blue",
            catalog.findPhotoByLocalIdentifier(nil, "2"):getRawMetadata('label'))
    end)

    it("accepts mixed case and normalizes it", function()
        local _, Handler = setup({ photos = photos() })
        assert.are.equal("purple", Handler.setColorLabel({ photo_ids = { "1" }, color = "Purple" }).color)
    end)

    it("rejects a color outside Lightroom's five labels", function()
        local _, Handler = setup({ photos = photos() })
        assert.has_error(function()
            Handler.setColorLabel({ photo_ids = { "1" }, color = "orange" })
        end)
    end)
end)
