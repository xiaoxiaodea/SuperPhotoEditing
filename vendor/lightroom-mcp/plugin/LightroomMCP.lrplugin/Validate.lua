-- Shared argument validation for handlers.
--
-- Handlers run behind LrTasks.pcall in PluginInfoProvider.dispatchAction, so a
-- plain `error()` here becomes the MCP tool's error text. Messages therefore
-- name the offending field the way the tool schema does (snake_case), not the
-- Lua local.
--
-- These helpers were duplicated across HandlerDevelop / HandlerOrganization
-- before; centralising them keeps one definition of "what a photo_ids array is"
-- so a new handler cannot silently accept a shape the others reject.

local Validate = {}

Validate.MAX_BULK_PHOTO_IDS = 1000
Validate.MAX_KEYWORDS = 1000

function Validate.requireString(value, name)
    if type(value) ~= "string" or value == "" then
        error(name .. " is required")
    end
    return value
end

function Validate.optionalString(value, name)
    if value == nil then return nil end
    return Validate.requireString(value, name)
end

-- Lua has no array type: `{ [2] = "x" }` and `{ "x" }` are both tables. A
-- caller sending a JSON object where an array belongs would otherwise sail
-- through `ipairs` as an empty list and the handler would report "updated 0"
-- instead of an error. Count pairs() and compare with #value to reject holes
-- and string keys explicitly.
function Validate.requireStringArray(value, name, maxItems)
    if type(value) ~= "table" then
        error(name .. " is required")
    end

    local count = 0
    for key, item in pairs(value) do
        if type(key) ~= "number" or key < 1 or key ~= math.floor(key) then
            error(name .. " must be an array")
        end
        if type(item) ~= "string" or item == "" then
            error(name .. "[" .. tostring(key) .. "] must be a non-empty string")
        end
        count = count + 1
    end

    if count == 0 then
        error(name .. " is required")
    end
    if count ~= #value then
        error(name .. " must be an array")
    end
    if maxItems and count > maxItems then
        error(name .. " must contain at most " .. tostring(maxItems) .. " items")
    end

    return value
end

function Validate.requirePhotoIds(value, name)
    return Validate.requireStringArray(value, name or "photo_ids", Validate.MAX_BULK_PHOTO_IDS)
end

function Validate.requireNumber(value, name)
    local number = tonumber(value)
    if number == nil then
        error(name .. " must be a number")
    end
    return number
end

function Validate.requireIntegerInRange(value, name, minimum, maximum)
    local number = Validate.requireNumber(value, name)
    if number ~= math.floor(number) then
        error(name .. " must be an integer")
    end
    if number < minimum or number > maximum then
        error(string.format("%s must be between %d and %d", name, minimum, maximum))
    end
    return number
end

function Validate.requireOneOf(value, name, allowed)
    if type(value) ~= "string" then
        error(name .. " must be a string")
    end
    for _, candidate in ipairs(allowed) do
        if value == candidate then return value end
    end
    error(name .. " must be one of: " .. table.concat(allowed, ", "))
end

-- Pagination shared by every list/search handler.
--
-- floor() is not cosmetic: the tool schemas accept any number, and a
-- fractional offset makes the page loop index matches[i] with a non-integer
-- key, which is always nil in Lua and crashes the per-row builder.
function Validate.pagination(args, defaultLimit)
    args = args or {}
    local limit = math.floor(tonumber(args.limit) or defaultLimit or 100)
    if limit < 0 then limit = 0 end
    local offset = math.floor(tonumber(args.offset) or 0)
    if offset < 0 then offset = 0 end
    return limit, offset
end

-- Slice `items` for the requested page and report whether more remain.
function Validate.paginate(items, limit, offset)
    local total = #items
    local last = math.min(offset + limit, total)
    local slice = {}
    for index = offset + 1, last do
        table.insert(slice, items[index])
    end
    return slice, total, (offset + #slice) < total
end

return Validate
