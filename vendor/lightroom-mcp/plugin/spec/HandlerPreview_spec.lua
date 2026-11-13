local helper = require 'spec_helper'

-- Files the fake io.open "wrote", keyed by path.
local written

local function setup(opts)
    opts = opts or {}
    written = {}
    local catalog = helper.fakeCatalog(opts)
    local existing = opts.existingPaths or {}

    helper.installImport({
        LrApplication = { activeCatalog = function() return catalog end },
        LrLogger = helper.defaultLrLogger(),
        -- The handler polls a flag between sleeps; the fake thumbnail resolves
        -- synchronously so sleep is a no-op here.
        LrTasks = { sleep = function() end, pcall = pcall },
        LrFileUtils = {
            exists = function(path)
                if existing[path] then return existing[path] end
                return false
            end,
            createAllDirectories = function(path) existing[path] = "directory" end,
        },
        LrPathUtils = {
            child = function(a, b) return a .. "/" .. b end,
            leafName = function(p) return p:match("[^/\\]+$") or p end,
            extension = function(p) return p:match("%.([^.]+)$") or "" end,
            removeExtension = function(p) return (p:gsub("%.[^.]+$", "")) end,
        },
        LrStringUtils = {
            -- Deterministic stand-in: the spec asserts the handler passes the
            -- JPEG bytes through encoding, not that it reimplements base64.
            encodeBase64 = function(data) return "B64(" .. data .. ")" end,
        },
    })

    package.loaded.HandlerPreview = nil
    package.loaded.PhotoLookup = nil
    package.loaded.Validate = nil
    package.loaded.Log = nil

    local realOpen = io.open
    io.open = function(path, mode, ...)
        if mode and mode:find("w", 1, true) then
            written[path] = ""
            return {
                write = function(_, data) written[path] = written[path] .. data end,
                close = function() end,
            }
        end
        return realOpen(path, mode, ...)
    end

    return catalog, require 'HandlerPreview', function() io.open = realOpen end
end

local function photo(extra)
    local meta = { id = "1", path = "/p/a.NEF", fileName = "a.NEF" }
    for k, v in pairs(extra or {}) do meta[k] = v end
    return helper.fakePhoto(meta)
end

describe("HandlerPreview.getPhotoPreview", function()
    it("returns the rendered JPEG inline as base64 by default", function()
        local _, Handler, restore = setup({ photos = { photo({ thumbnailData = "JPG" }) } })
        local r = Handler.getPhotoPreview({ photo_id = "1" })
        restore()

        assert.is_true(r.success)
        assert.are.equal("B64(JPG)", r.base64)
        assert.are.equal("image/jpeg", r.mime_type)
        assert.are.equal(3, r.bytes)
        assert.are.equal(1024, r.max_edge)
        assert.are.equal("/p/a.NEF", r.path)
    end)

    it("honours an explicit max_edge", function()
        local _, Handler, restore = setup({ photos = { photo() } })
        local r = Handler.getPhotoPreview({ photo_id = "1", max_edge = 512 })
        restore()
        assert.are.equal(512, r.max_edge)
    end)

    it("omits base64 when return_base64 is false", function()
        local _, Handler, restore = setup({ photos = { photo() } })
        local r = Handler.getPhotoPreview({ photo_id = "1", return_base64 = false })
        restore()
        assert.is_nil(r.base64)
        assert.is_true(r.success)
    end)

    it("saves the preview to a destination directory", function()
        local _, Handler, restore = setup({
            photos = { photo({ thumbnailData = "BYTES" }) },
            existingPaths = { ["/out"] = "directory" },
        })
        local r = Handler.getPhotoPreview({ photo_id = "1", destination_dir = "/out" })
        restore()

        assert.are.equal("/out/a-preview.jpg", r.saved_to)
        assert.are.equal("BYTES", written["/out/a-preview.jpg"])
    end)

    it("creates a missing destination directory", function()
        local _, Handler, restore = setup({ photos = { photo() } })
        local r = Handler.getPhotoPreview({ photo_id = "1", destination_dir = "/new" })
        restore()
        assert.are.equal("/new/a-preview.jpg", r.saved_to)
    end)

    it("accepts an explicit leaf filename and adds a .jpg extension", function()
        local _, Handler, restore = setup({
            photos = { photo() },
            existingPaths = { ["/out"] = "directory" },
        })
        local r = Handler.getPhotoPreview({
            photo_id = "1", destination_dir = "/out", filename = "hero",
        })
        restore()
        assert.are.equal("/out/hero.jpg", r.saved_to)
    end)

    it("refuses a filename containing path separators", function()
        local _, Handler, restore = setup({
            photos = { photo() },
            existingPaths = { ["/out"] = "directory" },
        })
        -- Prevents writing outside the directory the caller named.
        local ok = pcall(function()
            Handler.getPhotoPreview({
                photo_id = "1", destination_dir = "/out", filename = "../escape.jpg",
            })
        end)
        restore()
        assert.is_false(ok)
    end)

    it("rejects a bad filename before spending a render", function()
        local rendered = false
        local probe = helper.fakePhoto({ id = "1", path = "/p/a.NEF", fileName = "a.NEF" })
        probe.requestJpegThumbnail = function(_, _w, _h, callback)
            rendered = true
            callback("JPG")
            return {}
        end
        local _, Handler, restore = setup({
            photos = { probe },
            existingPaths = { ["/out"] = "directory" },
        })

        local ok = pcall(function()
            Handler.getPhotoPreview({
                photo_id = "1", destination_dir = "/out", filename = "sub/dir.jpg",
            })
        end)
        restore()

        assert.is_false(ok)
        -- Rendering can take seconds; a knowable argument error must not cost one.
        assert.is_false(rendered)
    end)

    it("rejects a filename given without a destination directory", function()
        local _, Handler, restore = setup({ photos = { photo() } })
        local ok = pcall(function()
            Handler.getPhotoPreview({ photo_id = "1", filename = "hero.jpg" })
        end)
        restore()
        assert.is_false(ok)
    end)

    it("never overwrites an existing preview file", function()
        local _, Handler, restore = setup({
            photos = { photo() },
            existingPaths = { ["/out"] = "directory", ["/out/a-preview.jpg"] = "file" },
        })
        local ok, err = pcall(function()
            Handler.getPhotoPreview({ photo_id = "1", destination_dir = "/out" })
        end)
        restore()
        assert.is_false(ok)
        assert.is_truthy(tostring(err):find("already exists", 1, true))
    end)

    it("omits oversized inline base64 but still reports the render", function()
        local big = string.rep("x", 1000 * 1024)
        local _, Handler, restore = setup({ photos = { photo({ thumbnailData = big }) } })
        local r = Handler.getPhotoPreview({ photo_id = "1" })
        restore()

        -- Inline delivery is capped independently of render size to protect the
        -- newline-framed socket and the context budget.
        assert.is_nil(r.base64)
        assert.is_true(r.base64_omitted)
        assert.is_truthy(r.base64_omitted_reason:find("inline limit", 1, true))
        assert.are.equal(#big, r.bytes)
    end)

    it("surfaces a render failure as an error", function()
        local _, Handler, restore = setup({
            photos = { photo({ thumbnailError = "preview unavailable" }) },
        })
        local ok, err = pcall(function() Handler.getPhotoPreview({ photo_id = "1" }) end)
        restore()
        assert.is_false(ok)
        assert.is_truthy(tostring(err):find("preview unavailable", 1, true))
    end)

    it("errors for an unknown photo", function()
        local _, Handler, restore = setup({ photos = { photo() } })
        local ok = pcall(function() Handler.getPhotoPreview({ photo_id = "nope" }) end)
        restore()
        assert.is_false(ok)
    end)

    it("errors without photo_id", function()
        local _, Handler, restore = setup({ photos = { photo() } })
        local ok = pcall(function() Handler.getPhotoPreview({}) end)
        restore()
        assert.is_false(ok)
    end)

    it("rejects a max_edge outside the supported range", function()
        local _, Handler, restore = setup({ photos = { photo() } })
        local tooSmall = pcall(function()
            Handler.getPhotoPreview({ photo_id = "1", max_edge = 8 })
        end)
        local tooBig = pcall(function()
            Handler.getPhotoPreview({ photo_id = "1", max_edge = 4096 })
        end)
        restore()
        assert.is_false(tooSmall)
        assert.is_false(tooBig)
    end)
end)
