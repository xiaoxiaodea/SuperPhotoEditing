-- Photo previews: render a JPEG thumbnail to disk and optionally return it
-- inline so the model can actually LOOK at the photo.
--
-- This closes the biggest blind spot in the bridge: every other tool describes
-- photos only as numbers and strings, so an agent asked to "fix the ones that
-- look too warm" had to trust metadata instead of the image. requestJpegThumbnail
-- is the SDK's cheap path for this — it reuses Lightroom's existing preview
-- pyramid and only renders when no preview of sufficient size exists.
--
-- requestJpegThumbnail is ASYNCHRONOUS: it returns a request handle immediately
-- and invokes the callback later, and the docs require a reference to that
-- handle be held until the callback fires. Handlers are already wrapped in
-- LrTasks.startAsyncTask by the dispatcher, so LrTasks.sleep can be used to wait
-- for the callback without blocking Lightroom's UI thread.

local LrApplication = import 'LrApplication'
local LrTasks = import 'LrTasks'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'
local LrStringUtils = import 'LrStringUtils'

local PhotoLookup = require 'PhotoLookup'
local Validate = require 'Validate'
local Log = require 'Log'

local PreviewHandler = {}

-- Keep the ceiling well under the dispatcher's 30s timeout so a preview that
-- genuinely never arrives still surfaces as this handler's clear error rather
-- than an opaque client-side timeout.
local THUMBNAIL_WAIT_SECONDS = 20
local POLL_INTERVAL_SECONDS = 0.1

local MIN_EDGE = 32
local MAX_EDGE = 2048
local DEFAULT_EDGE = 1024

-- Inline base64 travels as one newline-framed JSON line over LrSocket and then
-- through the model's context window. A 2048px JPEG can exceed 1 MB, which is
-- both a framing risk and a context-budget problem, so inline delivery is capped
-- independently of the render size: oversized renders stay on disk only.
local MAX_INLINE_BYTES = 900 * 1024

function PreviewHandler.getPhotoPreview(args)
    args = args or {}
    Validate.requireString(args.photo_id, "photo_id")

    local maxEdge = DEFAULT_EDGE
    if args.max_edge ~= nil then
        maxEdge = Validate.requireIntegerInRange(args.max_edge, "max_edge", MIN_EDGE, MAX_EDGE)
    end

    local returnBase64 = args.return_base64 ~= false

    local destinationDir = args.destination_dir
    if destinationDir ~= nil then
        Validate.requireString(destinationDir, "destination_dir")
    end

    -- Validate the leaf filename BEFORE rendering. Rendering can take seconds,
    -- and failing afterwards would burn that work to report a bad argument that
    -- was knowable up front.
    local requestedLeaf = args.filename
    if requestedLeaf ~= nil then
        Validate.requireString(requestedLeaf, "filename")
        if requestedLeaf == "." or requestedLeaf == ".."
            or LrPathUtils.leafName(requestedLeaf) ~= requestedLeaf then
            error("filename must be a leaf filename without path separators")
        end
        if destinationDir == nil then
            error("filename requires destination_dir")
        end
    end

    local catalog = LrApplication.activeCatalog()

    -- Resolve identity inside the read gate; requestJpegThumbnail itself must NOT
    -- run there (it yields while rendering).
    local photo, photoPath, filename
    catalog:withReadAccessDo(function()
        photo = PhotoLookup.resolveOne(catalog, args.photo_id)
        if not photo then
            error("Photo not found: " .. args.photo_id)
        end
        photoPath = photo:getRawMetadata('path')
        filename = photo:getFormattedMetadata('fileName')
    end)

    if type(photo.requestJpegThumbnail) ~= "function" then
        error("Preview rendering is unavailable in this Lightroom version")
    end

    local jpegData, thumbnailError, settled = nil, nil, false

    -- The request handle must outlive this call until the callback fires, hence
    -- the local: dropping it on the floor lets Lightroom collect the request.
    local request = photo:requestJpegThumbnail(maxEdge, maxEdge, function(data, errorMessage)
        if data then
            jpegData = data
        else
            thumbnailError = errorMessage or "unknown error"
        end
        settled = true
    end)

    local waited = 0
    while not settled and waited < THUMBNAIL_WAIT_SECONDS do
        LrTasks.sleep(POLL_INTERVAL_SECONDS)
        waited = waited + POLL_INTERVAL_SECONDS
    end
    -- Reference the handle after the wait so it cannot be collected early.
    local _ = request

    if not settled then
        error(string.format("Preview render timed out after %ds", THUMBNAIL_WAIT_SECONDS))
    end
    if not jpegData then
        error("Preview render failed: " .. tostring(thumbnailError))
    end

    local result = {
        success = true,
        photo_id = args.photo_id,
        path = photoPath,
        filename = filename,
        max_edge = maxEdge,
        bytes = #jpegData,
        mime_type = "image/jpeg",
    }

    -- Optional durable copy. Written with io.open in binary mode; LrFileUtils has
    -- no "write bytes" primitive.
    if destinationDir then
        if LrFileUtils.exists(destinationDir) == false then
            LrFileUtils.createAllDirectories(destinationDir)
        end
        if LrFileUtils.exists(destinationDir) ~= "directory" then
            error("destination_dir is not a directory")
        end

        local leaf = requestedLeaf
        if leaf == nil then
            local base = filename and LrPathUtils.removeExtension(filename) or "preview"
            leaf = base .. "-preview.jpg"
        end
        if LrPathUtils.extension(leaf) == "" then
            leaf = leaf .. ".jpg"
        end

        local destination = LrPathUtils.child(destinationDir, leaf)
        -- Match export_develop_preset's contract: never clobber existing files.
        if LrFileUtils.exists(destination) then
            error("destination preview already exists; choose a new filename")
        end

        local handle, openError = io.open(destination, "wb")
        if not handle then
            error("Could not write preview: " .. tostring(openError))
        end
        handle:write(jpegData)
        handle:close()
        result.saved_to = destination
    end

    if returnBase64 then
        if #jpegData > MAX_INLINE_BYTES then
            result.base64_omitted = true
            result.base64_omitted_reason = string.format(
                "preview is %d bytes, above the %d byte inline limit; lower max_edge or use destination_dir",
                #jpegData, MAX_INLINE_BYTES)
        else
            result.base64 = LrStringUtils.encodeBase64(jpegData)
        end
    end

    Log.info(string.format("Rendered preview for %s (%d bytes, max_edge=%d)",
        tostring(args.photo_id), #jpegData, maxEdge))

    return result
end

return PreviewHandler
