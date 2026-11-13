import type { Tool } from "@modelcontextprotocol/sdk/types.js";

type InputSchema = Tool["inputSchema"];

export interface ToolContract {
  name: string;
  description: string;
  luaHandler: string;
  inputSchema: InputSchema;
}

const MAX_BULK_PHOTO_IDS = 1000;
const MAX_KEYWORDS = 1000;

export const POINT_CURVE_SETTING_KEYS = [
  "ToneCurvePV2012",
  "ToneCurvePV2012Red",
  "ToneCurvePV2012Green",
  "ToneCurvePV2012Blue",
] as const;

export const DEVELOP_SETTING_KEYS = [
  "WhiteBalance",
  "Temperature",
  "Tint",
  "Exposure2012",
  "Contrast2012",
  "Highlights2012",
  "Shadows2012",
  "Whites2012",
  "Blacks2012",
  "Texture",
  "Clarity2012",
  "Dehaze",
  "Vibrance",
  "Saturation",
  "SplitToningShadowHue",
  "SplitToningShadowSaturation",
  "ColorGradeShadowLum",
  "ColorGradeMidtoneHue",
  "ColorGradeMidtoneSat",
  "ColorGradeMidtoneLum",
  "SplitToningHighlightHue",
  "SplitToningHighlightSaturation",
  "ColorGradeHighlightLum",
  "ColorGradeBlending",
  "SplitToningBalance",
  "SaturationAdjustmentRed",
  "SaturationAdjustmentOrange",
  "SaturationAdjustmentYellow",
  "SaturationAdjustmentGreen",
  "SaturationAdjustmentAqua",
  "SaturationAdjustmentBlue",
  "SaturationAdjustmentPurple",
  "SaturationAdjustmentMagenta",
  "HueAdjustmentRed",
  "HueAdjustmentOrange",
  "HueAdjustmentYellow",
  "HueAdjustmentGreen",
  "HueAdjustmentAqua",
  "HueAdjustmentBlue",
  "HueAdjustmentPurple",
  "HueAdjustmentMagenta",
  "LuminanceAdjustmentRed",
  "LuminanceAdjustmentOrange",
  "LuminanceAdjustmentYellow",
  "LuminanceAdjustmentGreen",
  "LuminanceAdjustmentAqua",
  "LuminanceAdjustmentBlue",
  "LuminanceAdjustmentPurple",
  "LuminanceAdjustmentMagenta",
  "ParametricShadows",
  "ParametricDarks",
  "ParametricLights",
  "ParametricHighlights",
  "ParametricShadowSplit",
  "ParametricMidtoneSplit",
  "ParametricHighlightSplit",
  ...POINT_CURVE_SETTING_KEYS,
  "ToneCurveName2012",
  "ConvertToGrayscale",
  "Sharpness",
  "SharpenRadius",
  "SharpenDetail",
  "SharpenEdgeMasking",
  "LuminanceSmoothing",
  "LuminanceNoiseReductionDetail",
  "LuminanceNoiseReductionContrast",
  "ColorNoiseReduction",
  "ColorNoiseReductionDetail",
  "ColorNoiseReductionSmoothness",
  "LensProfileEnable",
  "LensManualDistortionAmount",
  "PerspectiveVertical",
  "PerspectiveHorizontal",
  "PerspectiveRotate",
  "PerspectiveScale",
  "PerspectiveAspect",
  "PerspectiveUpright",
  "PostCropVignetteAmount",
  "PostCropVignetteMidpoint",
  "PostCropVignetteRoundness",
  "PostCropVignetteFeather",
  "PostCropVignetteStyle",
  "GrainAmount",
  "GrainSize",
  "GrainFrequency",
  "CropTop",
  "CropLeft",
  "CropBottom",
  "CropRight",
  "CropAngle",
] as const;

const stringArray = (description: string, maxItems?: number) => ({
  type: "array",
  items: { type: "string" },
  minItems: 1,
  ...(maxItems ? { maxItems } : {}),
  description,
});

const photoIdArray = (description: string) =>
  stringArray(description, MAX_BULK_PHOTO_IDS);

const dateStringSchema = (description: string) => ({
  type: "string",
  pattern: "^\\d{4}-\\d{2}-\\d{2}$",
  description,
});

const scalarDevelopSettingValueSchema = {
  oneOf: [{ type: "number" }, { type: "string" }, { type: "boolean" }],
};

const POINT_CURVE_MIN_PAIRS = 2;
const POINT_CURVE_MAX_PAIRS = 32;

const evenLengthSchemas = Array.from(
  { length: POINT_CURVE_MAX_PAIRS - POINT_CURVE_MIN_PAIRS + 1 },
  (_, index) => {
    const length = (POINT_CURVE_MIN_PAIRS + index) * 2;
    return { minItems: length, maxItems: length };
  },
);

const pointCurveDevelopSettingValueSchema = {
  type: "array",
  items: { type: "integer", minimum: 0, maximum: 255 },
  minItems: POINT_CURVE_MIN_PAIRS * 2,
  maxItems: POINT_CURVE_MAX_PAIRS * 2,
  anyOf: evenLengthSchemas,
  description:
    "Flat input/output pairs for a Lightroom point curve, e.g. [0, 0, 64, 48, 192, 210, 255, 255]. Values are integers from 0 to 255 and the array holds 2 to 32 pairs, so its length is always even. Inputs must be strictly increasing and the curve must start at input 0 and end at input 255.",
};

const pointCurveSettingKeySet = new Set<string>(POINT_CURVE_SETTING_KEYS);

const developSettingsProperties = Object.fromEntries(
  DEVELOP_SETTING_KEYS.map((key) => [
    key,
    pointCurveSettingKeySet.has(key)
      ? pointCurveDevelopSettingValueSchema
      : scalarDevelopSettingValueSchema,
  ]),
);

// Writable IPTC/XMP text fields, mirroring HandlerMetadataWrite.TEXT_KEYS.
// Kept as an explicit allowlist (not free-form) because setRawMetadata THROWS on
// an unknown key, which aborts the entire write transaction in Lightroom.
export const WRITABLE_TEXT_METADATA_KEYS = [
  "title",
  "caption",
  "headline",
  "copyName",
  "creator",
  "creatorJobTitle",
  "creatorAddress",
  "creatorCity",
  "creatorStateProvince",
  "creatorPostalCode",
  "creatorCountry",
  "creatorPhone",
  "creatorEmail",
  "creatorUrl",
  "iptcSubjectCode",
  "descriptionWriter",
  "dateCreated",
  "intellectualGenre",
  "scene",
  "location",
  "city",
  "stateProvince",
  "country",
  "isoCountryCode",
  "jobIdentifier",
  "instructions",
  "provider",
  "source",
  "copyright",
  "rightsUsageTerms",
  "copyrightInfoUrl",
  "personShown",
  "nameOfOrgShown",
  "codeOfOrgShown",
  "event",
  "additionalModelInfo",
  "modelAge",
  "minorModelAge",
  "modelReleaseStatus",
  "modelReleaseID",
  "propertyReleaseID",
  "propertyReleaseStatus",
  "sourceType",
] as const;

export const LABEL_COLORS = ["red", "yellow", "green", "blue", "purple", "none"] as const;
export const FLAG_VALUES = ["pick", "unflag", "reject"] as const;
export const FILE_FORMATS = ["DNG", "RAW", "JPG", "TIFF", "PNG", "PSD", "VIDEO"] as const;
export const ASPECT_RATIOS = ["portrait", "landscape", "square"] as const;
export const TREATMENTS = ["grayscale", "color"] as const;
export const NUMERIC_OPERATIONS = ["==", "!=", ">", "<", ">=", "<="] as const;
export const TEXT_OPERATIONS = [
  "any",
  "all",
  "words",
  "noneOf",
  "beginsWith",
  "endsWith",
] as const;
export const RELATIVE_TIME_UNITS = ["hours", "days", "weeks", "months", "years"] as const;
export const PHOTO_SORT_KEYS = ["captureTime", "fileName", "extension"] as const;

const writableMetadataProperties: Record<string, unknown> = Object.fromEntries(
  WRITABLE_TEXT_METADATA_KEYS.map((key) => [
    key,
    { type: "string", description: `IPTC/XMP ${key}; empty string clears the field` },
  ]),
);

writableMetadataProperties.copyrightState = {
  type: "string",
  enum: ["unknown", "copyrighted", "public domain"],
  description: "Copyright status",
};
writableMetadataProperties.colorNameForLabel = {
  type: "string",
  enum: LABEL_COLORS,
  description: "Color label; 'none' clears it",
};
writableMetadataProperties.pickStatus = {
  type: "number",
  enum: [1, 0, -1],
  description: "Flag: 1 = picked, 0 = unflagged, -1 = rejected",
};
writableMetadataProperties.gps = {
  oneOf: [
    {
      type: "object",
      additionalProperties: false,
      properties: {
        latitude: { type: "number", minimum: -90, maximum: 90 },
        longitude: { type: "number", minimum: -180, maximum: 180 },
      },
      required: ["latitude", "longitude"],
    },
    { type: "boolean", enum: [false] },
  ],
  description: "GPS coordinates, or false to clear them",
};
writableMetadataProperties.gpsAltitude = {
  type: "number",
  description: "GPS altitude in meters",
};

// Filters shared by search_photos and create_smart_collection, so an ad-hoc
// search and a saved smart collection accept exactly the same vocabulary.
const searchCriteriaProperties = {
  filename: { type: "string", description: "Search by filename (partial match)" },
  text: {
    type: "string",
    description: "Free-text search across indexed fields (filename, title, caption, keywords)",
  },
  text_operation: {
    type: "string",
    enum: TEXT_OPERATIONS,
    description: "How to match `text` (default any = contains)",
  },
  keywords: stringArray("Search by keywords"),
  rating: { type: "number", description: "Filter by star rating (0-5)", minimum: 0, maximum: 5 },
  rating_operation: {
    type: "string",
    enum: NUMERIC_OPERATIONS,
    description: "Comparison for `rating` (default ==). Use >= for 'three stars and up'.",
  },
  flag: { type: "string", enum: FLAG_VALUES, description: "Filter by flag status" },
  color_label: { type: "string", enum: LABEL_COLORS, description: "Filter by color label" },
  file_format: { type: "string", enum: FILE_FORMATS, description: "Filter by file format" },
  camera: { type: "string", description: "Filter by camera model (partial match)" },
  lens: { type: "string", description: "Filter by lens (partial match)" },
  folder: { type: "string", description: "Filter by folder name (partial match)" },
  collection: { type: "string", description: "Filter by collection name (partial match)" },
  city: { type: "string", description: "Filter by IPTC city (partial match)" },
  country: { type: "string", description: "Filter by IPTC country (partial match)" },
  iso: { type: "number", description: "Filter by ISO speed rating", minimum: 0 },
  iso_operation: {
    type: "string",
    enum: NUMERIC_OPERATIONS,
    description: "Comparison for `iso` (default ==)",
  },
  has_gps: { type: "boolean", description: "Only photos with (or without) GPS coordinates" },
  has_adjustments: { type: "boolean", description: "Only edited (or unedited) photos" },
  is_cropped: { type: "boolean", description: "Only cropped (or uncropped) photos" },
  aspect_ratio: { type: "string", enum: ASPECT_RATIOS, description: "Filter by orientation" },
  treatment: {
    type: "string",
    enum: TREATMENTS,
    description: "Filter by color vs black & white treatment",
  },
  captured_in_last: {
    type: "number",
    minimum: 1,
    description: "Only photos captured within the last N units (see captured_in_last_units)",
  },
  captured_in_last_units: {
    type: "string",
    enum: RELATIVE_TIME_UNITS,
    description: "Unit for `captured_in_last` (default days)",
  },
  edited_in_last: {
    type: "number",
    minimum: 1,
    description: "Only photos edited within the last N units (see edited_in_last_units)",
  },
  edited_in_last_units: {
    type: "string",
    enum: RELATIVE_TIME_UNITS,
    description: "Unit for `edited_in_last` (default days)",
  },
  start_date: dateStringSchema("Start date (YYYY-MM-DD)"),
  end_date: dateStringSchema("End date (YYYY-MM-DD)"),
  match: {
    type: "string",
    enum: ["all", "any"],
    description: "Combine filters with AND (all, default) or OR (any)",
  },
};

const presetSelectorProperties = {
  preset_name: { type: "string", minLength: 1, description: "Develop preset name" },
  preset_uuid: { type: "string", minLength: 1, description: "Develop preset UUID (preferred)" },
  preset_folder: { type: "string", minLength: 1, description: "Preset folder for disambiguation" },
  preset_scope: {
    type: "string",
    enum: ["lightroom", "plugin"],
    description: "Lightroom-visible preset or plugin-managed checkpoint",
  },
};

const presetSelectorSchema: InputSchema = {
  type: "object",
  additionalProperties: false,
  properties: presetSelectorProperties,
  anyOf: [{ required: ["preset_uuid"] }, { required: ["preset_name"] }],
};

export const TOOL_CONTRACTS: ToolContract[] = [
  {
    name: "search_photos",
    luaHandler: "HandlerSearch.searchPhotos",
    description:
      "Search for photos in the Lightroom catalog by metadata, EXIF, location, flag, label, edit state, or capture/edit date (paginated, default limit 100). Providing at least one filter significantly improves performance on large catalogs.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        ...searchCriteriaProperties,
        sort: {
          type: "string",
          enum: PHOTO_SORT_KEYS,
          description: "Sort results by capture time, filename, or extension",
        },
        ascending: {
          type: "boolean",
          description: "Sort ascending (default false); only used with `sort`",
        },
        limit: { type: "number", description: "Max photos to return (default 100)", minimum: 0 },
        offset: { type: "number", description: "Number of photos to skip (default 0)", minimum: 0 },
      },
    },
  },
  {
    name: "get_catalog_info",
    luaHandler: "HandlerCatalog.getCatalogInfo",
    description:
      "Get the active catalog's path, Lightroom version, and photo/collection counts. Use this to confirm which catalog is open before making changes.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {},
    },
  },
  {
    name: "list_folders",
    luaHandler: "HandlerCatalog.listFolders",
    description:
      "List folders in the catalog's Folders panel with their on-disk paths and hierarchy (paginated, default limit 100)",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        include_photo_counts: {
          type: "boolean",
          description: "Include a photo count per folder (slower on large catalogs)",
        },
        limit: { type: "number", description: "Max folders to return (default 100)", minimum: 0 },
        offset: { type: "number", description: "Number of folders to skip (default 0)", minimum: 0 },
      },
    },
  },
  {
    name: "list_keywords",
    luaHandler: "HandlerCatalog.listKeywords",
    description:
      "List the catalog's keyword hierarchy with synonyms and export flags (paginated, default limit 100). Use this before set_keywords to reuse existing keywords instead of creating near-duplicates.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        include_photo_counts: {
          type: "boolean",
          description: "Include a photo count per keyword (slower on large catalogs)",
        },
        limit: { type: "number", description: "Max keywords to return (default 100)", minimum: 0 },
        offset: { type: "number", description: "Number of keywords to skip (default 0)", minimum: 0 },
      },
    },
  },
  {
    name: "get_photo_preview",
    luaHandler: "HandlerPreview.getPhotoPreview",
    description:
      "Render a JPEG preview of one photo so its actual appearance can be inspected. Returns base64 inline by default and can also save the preview to a directory.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_id: { type: "string", minLength: 1, description: "Photo ID or file path" },
        max_edge: {
          type: "number",
          minimum: 32,
          maximum: 2048,
          description: "Longest edge in pixels (default 1024)",
        },
        return_base64: {
          type: "boolean",
          description: "Return the JPEG inline as base64 (default true)",
        },
        destination_dir: {
          type: "string",
          minLength: 1,
          description: "Optional directory to save the preview into; created when missing",
        },
        filename: {
          type: "string",
          minLength: 1,
          description: "Optional leaf filename for the saved preview; requires destination_dir",
        },
      },
      required: ["photo_id"],
      dependentRequired: { filename: ["destination_dir"] },
    },
  },
  {
    name: "get_selected_photos",
    luaHandler: "HandlerSelection.getSelectedPhotos",
    description: "Get currently selected photos in Lightroom (or filmstrip if no selection). Paginated, default limit 100.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        limit: { type: "number", description: "Max photos to return (default 100)", minimum: 0 },
        offset: { type: "number", description: "Number of photos to skip (default 0)", minimum: 0 },
      },
    },
  },
  {
    name: "set_selected_photos",
    luaHandler: "HandlerSelection.setSelectedPhotos",
    description:
      "Select photos in the Lightroom UI so the user can see a result set. The first ID becomes the active photo.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_ids: photoIdArray("Photo IDs or file paths to select; the first becomes active"),
      },
      required: ["photo_ids"],
    },
  },
  {
    name: "get_photo_metadata",
    luaHandler: "HandlerMetadata.getPhotoMetadata",
    description:
      "Get detailed metadata for a specific photo: EXIF, title/caption/headline, GPS (latitude/longitude/altitude), IPTC location (sublocation/city/stateProvince/country/isoCountryCode), copyright, and develop settings",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_id: { type: "string", description: "Photo ID or file path" },
      },
      required: ["photo_id"],
    },
  },
  {
    name: "set_photo_metadata",
    luaHandler: "HandlerMetadataWrite.setPhotoMetadata",
    description:
      "Write IPTC/XMP metadata to photos: title, caption, headline, creator and contact block, copyright and rights, IPTC location (city/state/country), GPS coordinates, color label, and flag status. Pass an empty string to clear a text field.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_ids: photoIdArray("Array of photo IDs or file paths"),
        metadata: {
          type: "object",
          additionalProperties: false,
          minProperties: 1,
          properties: writableMetadataProperties,
          description:
            'Metadata fields to write, e.g. {"copyright": "(c) 2026 Studio", "city": "Guangzhou"}',
        },
      },
      required: ["photo_ids", "metadata"],
    },
  },
  {
    name: "set_flag",
    luaHandler: "HandlerMetadataWrite.setFlag",
    description: "Set the pick/reject flag on photos, using Lightroom's cull vocabulary",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_ids: photoIdArray("Array of photo IDs or file paths"),
        flag: {
          type: "string",
          enum: FLAG_VALUES,
          description: "pick (flagged), unflag (neutral), or reject",
        },
      },
      required: ["photo_ids", "flag"],
    },
  },
  {
    name: "set_color_label",
    luaHandler: "HandlerMetadataWrite.setColorLabel",
    description: "Set the color label on photos; 'none' clears the label",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_ids: photoIdArray("Array of photo IDs or file paths"),
        color: { type: "string", enum: LABEL_COLORS, description: "Label color, or 'none' to clear" },
      },
      required: ["photo_ids", "color"],
    },
  },
  {
    name: "list_collections",
    luaHandler: "HandlerCollections.listCollections",
    description:
      "List all collections in the Lightroom catalog with their full set paths, photo counts, and smart-collection status (paginated, default limit 100)",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        limit: { type: "number", description: "Max collections to return (default 100)", minimum: 0 },
        offset: { type: "number", description: "Number of collections to skip (default 0)", minimum: 0 },
      },
    },
  },
  {
    name: "list_collection_sets",
    luaHandler: "HandlerCollections.listCollectionSets",
    description:
      "List collection sets (the folders of the Collections panel) with their child counts (paginated, default limit 100). Use this to find a valid `parent` for create_collection.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        limit: { type: "number", description: "Max sets to return (default 100)", minimum: 0 },
        offset: { type: "number", description: "Number of sets to skip (default 0)", minimum: 0 },
      },
    },
  },
  {
    name: "get_collection_photos",
    luaHandler: "HandlerCollections.getCollectionPhotos",
    description:
      "List the photos inside one collection (paginated, default limit 100). Accepts a bare collection name or a full 'Set / Name' path.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        collection_name: {
          type: "string",
          minLength: 1,
          description: "Collection name or full 'Set / Name' path",
        },
        limit: { type: "number", description: "Max photos to return (default 100)", minimum: 0 },
        offset: { type: "number", description: "Number of photos to skip (default 0)", minimum: 0 },
      },
      required: ["collection_name"],
    },
  },
  {
    name: "create_collection",
    luaHandler: "HandlerCollections.createCollection",
    description:
      "Create a new collection, optionally inside an existing collection set. Returns the existing collection if the name is already taken.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        name: { type: "string", minLength: 1, description: "Collection name" },
        parent: {
          type: "string",
          minLength: 1,
          description: "Parent collection set name or full path (optional; top level when omitted)",
        },
      },
      required: ["name"],
    },
  },
  {
    name: "create_collection_set",
    luaHandler: "HandlerCollections.createCollectionSet",
    description:
      "Create a collection set to group collections, optionally nested inside another set",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        name: { type: "string", minLength: 1, description: "Collection set name" },
        parent: {
          type: "string",
          minLength: 1,
          description: "Parent collection set name or full path (optional)",
        },
      },
      required: ["name"],
    },
  },
  {
    name: "create_smart_collection",
    luaHandler: "HandlerCollections.createSmartCollection",
    description:
      "Create a smart collection that stays up to date automatically, using the same filter vocabulary as search_photos",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        name: { type: "string", minLength: 1, description: "Smart collection name" },
        criteria: {
          type: "object",
          additionalProperties: false,
          minProperties: 1,
          properties: searchCriteriaProperties,
          description: 'Filters for the smart collection, e.g. {"rating": 4, "rating_operation": ">="}',
        },
        match: {
          type: "string",
          enum: ["all", "any"],
          description: "Match all criteria (default) or any of them",
        },
        parent: {
          type: "string",
          minLength: 1,
          description: "Parent collection set name or full path (optional)",
        },
      },
      required: ["name", "criteria"],
    },
  },
  {
    name: "add_to_collection",
    luaHandler: "HandlerCollections.addToCollection",
    description: "Add photos to a collection",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        collection_name: {
          type: "string",
          minLength: 1,
          description: "Collection name or full 'Set / Name' path",
        },
        photo_ids: photoIdArray("Array of photo IDs or file paths"),
      },
      required: ["collection_name", "photo_ids"],
    },
  },
  {
    name: "remove_from_collection",
    luaHandler: "HandlerCollections.removeFromCollection",
    description:
      "Remove photos from a collection. The photos stay in the catalog; only the collection membership is removed.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        collection_name: {
          type: "string",
          minLength: 1,
          description: "Collection name or full 'Set / Name' path",
        },
        photo_ids: photoIdArray("Array of photo IDs or file paths"),
      },
      required: ["collection_name", "photo_ids"],
    },
  },
  {
    name: "rename_collection",
    luaHandler: "HandlerCollections.renameCollection",
    description: "Rename an existing collection",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        collection_name: {
          type: "string",
          minLength: 1,
          description: "Current collection name or full 'Set / Name' path",
        },
        new_name: { type: "string", minLength: 1, description: "New collection name" },
      },
      required: ["collection_name", "new_name"],
    },
  },
  {
    name: "delete_collection",
    luaHandler: "HandlerCollections.deleteCollection",
    description:
      "Delete a collection. The photos it contained remain in the catalog; only the grouping is removed.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        collection_name: {
          type: "string",
          minLength: 1,
          description: "Collection name or full 'Set / Name' path",
        },
      },
      required: ["collection_name"],
    },
  },
  {
    name: "set_keywords",
    luaHandler: "HandlerOrganization.setKeywords",
    description: "Add or remove keywords from photos",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_ids: photoIdArray("Array of photo IDs or file paths"),
        add_keywords: stringArray("Keywords to add", MAX_KEYWORDS),
        remove_keywords: stringArray("Keywords to remove", MAX_KEYWORDS),
      },
      required: ["photo_ids"],
    },
  },
  {
    name: "set_rating",
    luaHandler: "HandlerOrganization.setRating",
    description: "Set star rating for photos",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_ids: photoIdArray("Array of photo IDs or file paths"),
        rating: {
          type: "number",
          description: "Star rating (0-5)",
          minimum: 0,
          maximum: 5,
        },
      },
      required: ["photo_ids", "rating"],
    },
  },
  {
    name: "import_photos",
    luaHandler: "HandlerImport.importPhotos",
    description: "Import photos into Lightroom catalog",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        source_path: { type: "string", description: "Path to photo or folder to import" },
        collection_name: {
          type: "string",
          description: "Collection to add imported photos to (optional)",
        },
        copy_to: {
          type: "string",
          description: "Destination folder for copying files (optional)",
        },
      },
      required: ["source_path"],
    },
  },
  {
    name: "export_photos",
    luaHandler: "HandlerExport.exportPhotos",
    description: "Export photos from Lightroom",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_ids: photoIdArray("Array of photo IDs or file paths to export"),
        destination: { type: "string", description: "Export destination folder" },
        format: {
          type: "string",
          description: "Export format (jpeg, png, tiff, original)",
          enum: ["jpeg", "png", "tiff", "original"],
        },
        quality: {
          type: "number",
          description: "JPEG quality (0-100)",
          minimum: 0,
          maximum: 100,
        },
        width: { type: "number", description: "Max width in pixels (optional)" },
        height: { type: "number", description: "Max height in pixels (optional)" },
      },
      required: ["photo_ids", "destination"],
    },
  },
  {
    name: "list_develop_presets",
    luaHandler: "HandlerDevelop.listDevelopPresets",
    description: "List Lightroom-visible Develop presets and plugin-managed preset checkpoints",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {},
    },
  },
  {
    name: "get_develop_preset",
    luaHandler: "HandlerDevelop.getDevelopPreset",
    description:
      "Read the settings and backing-file metadata for one exact Develop preset. Use preset_uuid or provide folder/scope when names are duplicated.",
    inputSchema: presetSelectorSchema,
  },
  {
    name: "compare_develop_presets",
    luaHandler: "HandlerDevelop.compareDevelopPresets",
    description:
      "Compare two Develop presets and return a deterministic per-setting diff for iterative style matching",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        base: { ...presetSelectorSchema, description: "Approved historical/base preset" },
        candidate: { ...presetSelectorSchema, description: "Candidate preset checkpoint" },
      },
      required: ["base", "candidate"],
    },
  },
  {
    name: "create_develop_preset",
    luaHandler: "HandlerDevelop.createDevelopPreset",
    description:
      "Create a versioned plugin-managed Develop preset checkpoint from selected settings on one photo. The checkpoint is hidden from the Develop panel; export it for handoff.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_id: { type: "string", minLength: 1, description: "Source photo ID or file path" },
        preset_name: {
          type: "string",
          minLength: 1,
          description: "Unique versioned checkpoint name; existing plugin names are refused",
        },
        settings: {
          type: "array",
          items: { type: "string", enum: DEVELOP_SETTING_KEYS },
          minItems: 1,
          maxItems: DEVELOP_SETTING_KEYS.length,
          uniqueItems: true,
          description: "Explicit Lightroom SDK setting keys to capture from the source photo",
        },
      },
      required: ["photo_id", "preset_name", "settings"],
    },
  },
  {
    name: "export_develop_preset",
    luaHandler: "HandlerDevelop.exportDevelopPreset",
    description:
      "Copy one exact custom or plugin-managed Develop preset backing file to a destination directory. Existing files are never overwritten; built-in presets without backing files cannot be exported.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        ...presetSelectorProperties,
        destination_dir: {
          type: "string",
          minLength: 1,
          description: "Destination directory; created when missing",
        },
        filename: {
          type: "string",
          minLength: 1,
          description: "Optional leaf filename. Extension must match the Lightroom backing file.",
        },
      },
      required: ["destination_dir"],
      anyOf: [{ required: ["preset_uuid"] }, { required: ["preset_name"] }],
    },
  },
  {
    name: "apply_develop_preset",
    luaHandler: "HandlerDevelop.applyDevelopPreset",
    description: "Apply one exact Develop preset to one or more photos",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_ids: photoIdArray("Array of photo IDs or file paths"),
        preset_name: {
          type: "string",
          description: "Preset name",
        },
        preset_uuid: { type: "string", description: "Preset UUID (preferred)" },
        preset_folder: { type: "string", description: "Preset folder for disambiguation" },
        preset_scope: { type: "string", enum: ["lightroom", "plugin"] },
      },
      required: ["photo_ids"],
      anyOf: [{ required: ["preset_uuid"] }, { required: ["preset_name"] }],
    },
  },
  {
    name: "copy_develop_settings",
    luaHandler: "HandlerDevelop.copyDevelopSettings",
    description: "Copy Develop settings from one photo to others",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        source_id: {
          type: "string",
          description: "Source photo ID or file path",
        },
        target_ids: photoIdArray("Target photo IDs or file paths"),
        settings: {
          type: "array",
          items: {
            type: "string",
            enum: DEVELOP_SETTING_KEYS,
          },
          minItems: 1,
          maxItems: DEVELOP_SETTING_KEYS.length,
          description:
            "Optional whitelist of SDK setting keys (e.g., Exposure2012, Contrast2012, HueAdjustmentOrange). Omit to copy all.",
        },
      },
      required: ["source_id", "target_ids"],
    },
  },
  {
    name: "apply_ai_mask",
    luaHandler: "HandlerMasking.applyAiMask",
    description: "Create or update a bridge-owned native Lightroom subject, sky or background mask on one photo. Read back all local slider values. Reset removes only this bridge's mask for that target.",
    inputSchema: {
      type: "object",
      properties: {
        photo_id: { type: "string", minLength: 1 },
        target: { type: "string", enum: ["subject", "sky", "background"] },
        reset: { type: "boolean" },
        settings: {
          type: "object", additionalProperties: false,
          properties: {
            local_Exposure: { type: "number", minimum: -2, maximum: 2 },
            local_Contrast: { type: "number", minimum: -50, maximum: 50 },
            local_Whites: { type: "number", minimum: -75, maximum: 75 },
            local_Blacks: { type: "number", minimum: -50, maximum: 50 },
            local_Highlights: { type: "number", minimum: -40, maximum: 20 },
            local_Shadows: { type: "number", minimum: -25, maximum: 35 },
            local_Temperature: { type: "number", minimum: -12, maximum: 12 },
            local_Tint: { type: "number", minimum: -10, maximum: 10 },
            local_Saturation: { type: "number", minimum: -25, maximum: 15 },
            local_Clarity: { type: "number", minimum: -15, maximum: 15 },
            local_Texture: { type: "number", minimum: -20, maximum: 20 },
            local_Dehaze: { type: "number", minimum: -10, maximum: 15 },
          },
        },
      },
      required: ["photo_id", "target", "settings"],
      additionalProperties: false,
    },
  },
  {
    name: "set_develop_settings",
    luaHandler: "HandlerDevelop.setDevelopSettings",
    description:
      "Set Develop settings directly on a photo. Keys use allowlisted Lightroom SDK names (Exposure2012, WhiteBalance, Contrast2012, Highlights2012, Shadows2012, Whites2012, Blacks2012, Clarity2012, Vibrance, Saturation, HueAdjustmentRed, SaturationAdjustmentOrange, LuminanceAdjustmentYellow, etc.), plus RGB composite and per-channel point curves via ToneCurvePV2012, ToneCurvePV2012Red, ToneCurvePV2012Green, and ToneCurvePV2012Blue.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      properties: {
        photo_id: {
          type: "string",
          description: "Photo ID or file path",
        },
        settings: {
          type: "object",
          properties: developSettingsProperties,
          additionalProperties: false,
          minProperties: 1,
          description:
            "Allowlisted SDK setting key/value pairs (e.g., {\"Exposure2012\": 0.5, \"SaturationAdjustmentOrange\": -10})",
        },
      },
      required: ["photo_id", "settings"],
    },
  },
];
