import { describe, it, expect } from '@jest/globals';
import { Ajv } from 'ajv';
import fs from 'node:fs';
import path from 'node:path';
import { TOOL_DEFINITIONS, listToolsHandler } from '../src/list-tools-handler.js';
import {
  ASPECT_RATIOS,
  DEVELOP_SETTING_KEYS,
  FILE_FORMATS,
  FLAG_VALUES,
  LABEL_COLORS,
  POINT_CURVE_SETTING_KEYS,
  TEXT_OPERATIONS,
  TOOL_CONTRACTS,
  TREATMENTS,
  WRITABLE_TEXT_METADATA_KEYS,
} from '../src/tool-contracts.js';

const EXPECTED_TOOL_NAMES = [
  'search_photos',
  'get_catalog_info',
  'list_folders',
  'list_keywords',
  'get_photo_preview',
  'get_selected_photos',
  'set_selected_photos',
  'get_photo_metadata',
  'set_photo_metadata',
  'set_flag',
  'set_color_label',
  'list_collections',
  'list_collection_sets',
  'get_collection_photos',
  'create_collection',
  'create_collection_set',
  'create_smart_collection',
  'add_to_collection',
  'remove_from_collection',
  'rename_collection',
  'delete_collection',
  'set_keywords',
  'set_rating',
  'import_photos',
  'export_photos',
  'list_develop_presets',
  'get_develop_preset',
  'compare_develop_presets',
  'create_develop_preset',
  'export_develop_preset',
  'apply_develop_preset',
  'copy_develop_settings',
  'set_develop_settings',
  'apply_ai_mask',
] as const;

describe('TOOL_DEFINITIONS', () => {
  // Pinned to the explicit name list above rather than a bare number, so adding
  // a tool without listing it here fails loudly instead of silently shifting a
  // magic constant.
  it('exposes exactly the expected tool catalog', () => {
    expect(TOOL_DEFINITIONS).toHaveLength(EXPECTED_TOOL_NAMES.length);
    expect(TOOL_DEFINITIONS.map((t) => t.name).sort()).toEqual(
      [...EXPECTED_TOOL_NAMES].sort(),
    );
  });

  it('tool names are unique', () => {
    const names = TOOL_DEFINITIONS.map((t) => t.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it.each(EXPECTED_TOOL_NAMES)('"%s" is present', (name) => {
    expect(TOOL_DEFINITIONS.some((t) => t.name === name)).toBe(true);
  });

  it('every tool has name, description, and inputSchema', () => {
    for (const tool of TOOL_DEFINITIONS) {
      expect(typeof tool.name).toBe('string');
      expect(typeof tool.description).toBe('string');
      expect(tool.inputSchema).toBeDefined();
      expect(tool.inputSchema.type).toBe('object');
    }
  });

  it('is generated from tool contracts', () => {
    expect(TOOL_DEFINITIONS).toEqual(
      TOOL_CONTRACTS.map(({ name, description, inputSchema }) => ({
        name,
        description,
        inputSchema,
      })),
    );
  });

  it('rejects unknown top-level arguments for every tool', () => {
    for (const tool of TOOL_DEFINITIONS) {
      expect(tool.inputSchema.additionalProperties).toBe(false);
    }
  });
});

describe('listToolsHandler', () => {
  it('returns { tools: TOOL_DEFINITIONS }', () => {
    const result = listToolsHandler();
    expect(result.tools).toEqual(TOOL_DEFINITIONS);
  });
});

describe('tool required fields', () => {
  function toolRequired(name: string): string[] | undefined {
    return TOOL_DEFINITIONS.find((t) => t.name === name)?.inputSchema.required as string[] | undefined;
  }

  it.each<[string, string[]]>([
    ['get_photo_metadata', ['photo_id']],
    ['set_photo_metadata', ['photo_ids', 'metadata']],
    ['set_flag', ['photo_ids', 'flag']],
    ['set_color_label', ['photo_ids', 'color']],
    ['set_selected_photos', ['photo_ids']],
    ['get_photo_preview', ['photo_id']],
    ['create_collection', ['name']],
    ['create_collection_set', ['name']],
    ['create_smart_collection', ['name', 'criteria']],
    ['get_collection_photos', ['collection_name']],
    ['rename_collection', ['collection_name', 'new_name']],
    ['delete_collection', ['collection_name']],
    ['add_to_collection', ['collection_name', 'photo_ids']],
    ['remove_from_collection', ['collection_name', 'photo_ids']],
    ['set_keywords', ['photo_ids']],
    ['set_rating', ['photo_ids', 'rating']],
    ['import_photos', ['source_path']],
    ['export_photos', ['photo_ids', 'destination']],
    ['compare_develop_presets', ['base', 'candidate']],
    ['create_develop_preset', ['photo_id', 'preset_name', 'settings']],
    ['export_develop_preset', ['destination_dir']],
    ['apply_develop_preset', ['photo_ids']],
    ['copy_develop_settings', ['source_id', 'target_ids']],
    ['set_develop_settings', ['photo_id', 'settings']],
    ['apply_ai_mask', ['photo_id', 'target', 'settings']],
  ])('%s requires %j', (name, required) => {
    expect(toolRequired(name)).toEqual(required);
  });

  it.each([
    'search_photos',
    'get_selected_photos',
    'get_catalog_info',
    'list_folders',
    'list_keywords',
    'list_collections',
    'list_collection_sets',
    'list_develop_presets',
    'get_develop_preset',
  ])(
    '%s has no required fields',
    (name) => {
      expect(toolRequired(name)).toBeUndefined();
    },
  );
});

describe('set_keywords schema', () => {
  it('caps add/remove keyword arrays', () => {
    const tool = TOOL_DEFINITIONS.find((t) => t.name === 'set_keywords');
    const properties = tool?.inputSchema.properties as Record<string, { maxItems?: number }>;

    expect(properties.add_keywords.maxItems).toBe(1000);
    expect(properties.remove_keywords.maxItems).toBe(1000);
  });
});

describe('apply_ai_mask schema', () => {
  const schema = TOOL_DEFINITIONS.find((t) => t.name === 'apply_ai_mask')!.inputSchema;
  const validate = new Ajv({ strict: false }).compile(schema);
  it('accepts bounded local exposure and target-owned reset', () => {
    expect(validate({ photo_id: 'copy.jpg', target: 'subject', settings: { local_Exposure: 0.3 } })).toBe(true);
    expect(validate({ photo_id: 'copy.jpg', target: 'subject', settings: {}, reset: true })).toBe(true);
  });
  it('rejects global sliders, unsupported targets and excessive exposure', () => {
    expect(validate({ photo_id: 'copy.jpg', target: 'subject', settings: { Exposure2012: 0.3 } })).toBe(false);
    expect(validate({ photo_id: 'copy.jpg', target: 'arbitrary-mask', settings: {} })).toBe(false);
    expect(validate({ photo_id: 'copy.jpg', target: 'sky', settings: { local_Exposure: 3 } })).toBe(false);
  });
});

describe('photo array schema', () => {
  it.each([
    ['add_to_collection', 'photo_ids'],
    ['remove_from_collection', 'photo_ids'],
    ['set_keywords', 'photo_ids'],
    ['set_rating', 'photo_ids'],
    ['set_photo_metadata', 'photo_ids'],
    ['set_flag', 'photo_ids'],
    ['set_color_label', 'photo_ids'],
    ['set_selected_photos', 'photo_ids'],
    ['export_photos', 'photo_ids'],
    ['apply_develop_preset', 'photo_ids'],
    ['copy_develop_settings', 'target_ids'],
  ])('%s.%s requires 1-1000 ids', (toolName, propertyName) => {
    const tool = TOOL_DEFINITIONS.find((t) => t.name === toolName);
    const properties = tool?.inputSchema.properties as Record<
      string,
      { minItems?: number; maxItems?: number }
    >;

    expect(properties[propertyName].minItems).toBe(1);
    expect(properties[propertyName].maxItems).toBe(1000);
  });
});

describe('set_photo_metadata schema', () => {
  const tool = () => TOOL_DEFINITIONS.find((t) => t.name === 'set_photo_metadata')!;
  const validate = () => new Ajv({ strict: false }).compile(tool().inputSchema);

  it('restricts metadata to the writable allowlist', () => {
    const properties = tool().inputSchema.properties as Record<
      string,
      { additionalProperties?: boolean; minProperties?: number; properties?: Record<string, unknown> }
    >;

    expect(properties.metadata.additionalProperties).toBe(false);
    expect(properties.metadata.minProperties).toBe(1);
    for (const key of WRITABLE_TEXT_METADATA_KEYS) {
      expect(properties.metadata.properties?.[key]).toEqual({
        type: 'string',
        description: `IPTC/XMP ${key}; empty string clears the field`,
      });
    }
  });

  it('accepts text, GPS, label and flag writes', () => {
    const check = validate();
    expect(check({ photo_ids: ['1'], metadata: { copyright: '(c) 2026' } })).toBe(true);
    expect(check({ photo_ids: ['1'], metadata: { caption: '' } })).toBe(true);
    expect(
      check({ photo_ids: ['1'], metadata: { gps: { latitude: 23.1, longitude: 113.3 } } }),
    ).toBe(true);
    expect(check({ photo_ids: ['1'], metadata: { gps: false } })).toBe(true);
    expect(check({ photo_ids: ['1'], metadata: { colorNameForLabel: 'none' } })).toBe(true);
    expect(check({ photo_ids: ['1'], metadata: { pickStatus: -1 } })).toBe(true);
  });

  it('rejects invalid metadata payloads', () => {
    const check = validate();
    // A field the SDK would throw on must be refused at the schema boundary.
    expect(check({ photo_ids: ['1'], metadata: { notARealField: 'x' } })).toBe(false);
    expect(check({ photo_ids: ['1'], metadata: {} })).toBe(false);
    expect(check({ photo_ids: [], metadata: { title: 'x' } })).toBe(false);
    expect(check({ photo_ids: ['1'], metadata: { colorNameForLabel: 'orange' } })).toBe(false);
    expect(check({ photo_ids: ['1'], metadata: { pickStatus: 2 } })).toBe(false);
    expect(check({ photo_ids: ['1'], metadata: { gps: { latitude: 91, longitude: 0 } } })).toBe(false);
    expect(check({ photo_ids: ['1'], metadata: { gps: { latitude: 0 } } })).toBe(false);
    expect(check({ photo_ids: ['1'], metadata: { gps: true } })).toBe(false);
  });
});

describe('search_photos extended criteria', () => {
  const validate = () =>
    new Ajv({ strict: false }).compile(
      TOOL_DEFINITIONS.find((t) => t.name === 'search_photos')!.inputSchema,
    );

  it('accepts the extended filter vocabulary', () => {
    const check = validate();
    expect(check({ rating: 4, rating_operation: '>=' })).toBe(true);
    expect(check({ flag: 'pick', color_label: 'red' })).toBe(true);
    expect(check({ file_format: 'RAW', treatment: 'grayscale' })).toBe(true);
    expect(check({ camera: 'X-T5', lens: '35mm', iso: 3200, iso_operation: '>' })).toBe(true);
    expect(check({ has_gps: true, is_cropped: false, has_adjustments: true })).toBe(true);
    expect(check({ captured_in_last: 30, captured_in_last_units: 'days' })).toBe(true);
    expect(check({ text: 'sunset', text_operation: 'beginsWith' })).toBe(true);
    expect(check({ sort: 'captureTime', ascending: true })).toBe(true);
    expect(check({ match: 'any', city: 'Guangzhou' })).toBe(true);
  });

  it('rejects out-of-vocabulary filters', () => {
    const check = validate();
    expect(check({ flag: 'maybe' })).toBe(false);
    expect(check({ color_label: 'orange' })).toBe(false);
    expect(check({ file_format: 'HEIF' })).toBe(false);
    expect(check({ rating_operation: '=~' })).toBe(false);
    expect(check({ sort: 'rating' })).toBe(false);
    expect(check({ captured_in_last_units: 'fortnights' })).toBe(false);
    expect(check({ unknown_filter: 1 })).toBe(false);
  });

  it('shares its criteria vocabulary with create_smart_collection', () => {
    const search = TOOL_DEFINITIONS.find((t) => t.name === 'search_photos')!;
    const smart = TOOL_DEFINITIONS.find((t) => t.name === 'create_smart_collection')!;
    const searchProperties = search.inputSchema.properties as Record<string, unknown>;
    const smartCriteria = (
      smart.inputSchema.properties as Record<string, { properties?: Record<string, unknown> }>
    ).criteria.properties!;

    // Every smart-collection criterion must exist in search with an identical
    // schema; otherwise a saved collection and an ad-hoc search would diverge.
    for (const key of Object.keys(smartCriteria)) {
      expect(searchProperties[key]).toEqual(smartCriteria[key]);
    }
  });
});

describe('Lua/TypeScript enum parity', () => {
  function readPluginSource(file: string): string {
    return fs.readFileSync(
      path.resolve(process.cwd(), '..', 'plugin', 'LightroomMCP.lrplugin', file),
      'utf8',
    );
  }

  function parseLuaStringList(source: string, tableName: string): string[] {
    const match = source.match(new RegExp(`local ${tableName} = \\{([\\s\\S]*?)\\}`));
    if (!match) {
      throw new Error(`${tableName} table not found`);
    }
    return [...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
  }

  // The Lua handler validates against its own copies of these enums. If the two
  // drift, the schema advertises a value the handler rejects at runtime (or vice
  // versa), which is exactly the class of bug schemas are supposed to prevent.
  it('writable text metadata keys match the Lua allowlist', () => {
    const luaKeys = parseLuaStringList(readPluginSource('HandlerMetadataWrite.lua'), 'TEXT_KEYS');
    expect(luaKeys).toEqual([...WRITABLE_TEXT_METADATA_KEYS]);
  });

  it('label colors and flags match the Lua allowlists', () => {
    const source = readPluginSource('HandlerMetadataWrite.lua');
    expect(parseLuaStringList(source, 'LABEL_COLORS')).toEqual([...LABEL_COLORS]);
    expect(parseLuaStringList(source, 'COPYRIGHT_STATES')).toEqual([
      'unknown',
      'copyrighted',
      'public domain',
    ]);
  });

  it('search enums match the Lua allowlists', () => {
    const source = readPluginSource('HandlerSearch.lua');
    expect(parseLuaStringList(source, 'FILE_FORMATS')).toEqual([...FILE_FORMATS]);
    expect(parseLuaStringList(source, 'ASPECT_RATIOS')).toEqual([...ASPECT_RATIOS]);
    expect(parseLuaStringList(source, 'TREATMENTS')).toEqual([...TREATMENTS]);
    expect(parseLuaStringList(source, 'FLAGS')).toEqual([...FLAG_VALUES]);
    expect(parseLuaStringList(source, 'TEXT_OPERATIONS')).toEqual([...TEXT_OPERATIONS]);
  });
});

describe('develop setting schema', () => {
  function parseLuaDevelopSettingKeys(): string[] {
    const pluginPath = path.resolve(process.cwd(), '..', 'plugin', 'LightroomMCP.lrplugin', 'HandlerDevelop.lua');
    const source = fs.readFileSync(pluginPath, 'utf8');
    const match = source.match(/local ALLOWED_DEVELOP_SETTING_KEYS = \{([\s\S]*?)\n\}/);
    if (!match) {
      throw new Error('ALLOWED_DEVELOP_SETTING_KEYS table not found');
    }

    return [...match[1].matchAll(/^\s*"([^"]+)",/gm)].map((entry) => entry[1]);
  }

  it('restricts copy whitelist to allowlisted SDK keys', () => {
    const tool = TOOL_DEFINITIONS.find((t) => t.name === 'copy_develop_settings');
    const properties = tool?.inputSchema.properties as Record<
      string,
      { items?: { enum?: readonly string[] }; minItems?: number }
    >;

    expect(properties.settings.minItems).toBe(1);
    expect(properties.settings.items?.enum).toEqual(DEVELOP_SETTING_KEYS);
  });

  it('restricts direct settings object to allowlisted SDK keys', () => {
    const tool = TOOL_DEFINITIONS.find((t) => t.name === 'set_develop_settings');
    const properties = tool?.inputSchema.properties as Record<
      string,
      { additionalProperties?: boolean; minProperties?: number; properties?: Record<string, unknown> }
    >;

    expect(properties.settings.additionalProperties).toBe(false);
    expect(properties.settings.minProperties).toBe(1);
    expect(Object.keys(properties.settings.properties ?? {})).toEqual(DEVELOP_SETTING_KEYS);
  });

  it('uses numeric arrays for point curves and scalar values for ordinary settings', () => {
    const tool = TOOL_DEFINITIONS.find((t) => t.name === 'set_develop_settings');
    const properties = tool?.inputSchema.properties as Record<
      string,
      {
        properties?: Record<
          string,
          {
            type?: string;
            minItems?: number;
            maxItems?: number;
            items?: { type?: string; minimum?: number; maximum?: number };
            oneOf?: unknown[];
          }
        >;
      }
    >;
    const settings = properties.settings.properties ?? {};

    for (const key of POINT_CURVE_SETTING_KEYS) {
      expect(settings[key]).toMatchObject({
        type: 'array',
        minItems: 4,
        maxItems: 64,
        items: { type: 'integer', minimum: 0, maximum: 255 },
      });
    }
    expect(settings.Exposure2012.oneOf).toBeDefined();
    expect(settings.Exposure2012.type).toBeUndefined();
  });

  it('rejects malformed point curve payloads against the published schema', () => {
    const tool = TOOL_DEFINITIONS.find((t) => t.name === 'set_develop_settings');
    const validate = new Ajv({ strict: false }).compile(tool!.inputSchema);

    const check = (curve: unknown[]) =>
      validate({ photo_id: '1', settings: { ToneCurvePV2012: curve } });

    expect(check([0, 0, 64, 48, 192, 210, 255, 255])).toBe(true);
    expect(check([0, 0, 255, 255])).toBe(true);
    expect(check([0, 0, 128, 120, 255])).toBe(false);
    expect(check([0, 0, 128.5, 120, 255, 255])).toBe(false);
    expect(check([0, 0, 128, 300, 255, 255])).toBe(false);
    expect(check([0, 0, 128, -1, 255, 255])).toBe(false);
    expect(check([0, 0])).toBe(false);
    expect(check(Array.from({ length: 66 }, () => 0))).toBe(false);
  });

  it('requires explicit allowlisted keys when creating a preset checkpoint', () => {
    const tool = TOOL_DEFINITIONS.find((t) => t.name === 'create_develop_preset');
    const properties = tool?.inputSchema.properties as Record<
      string,
      { items?: { enum?: readonly string[] }; minItems?: number; uniqueItems?: boolean }
    >;

    expect(properties.settings.items?.enum).toEqual(DEVELOP_SETTING_KEYS);
    expect(properties.settings.minItems).toBe(1);
    expect(properties.settings.uniqueItems).toBe(true);
  });

  it('matches Lua develop setting allowlist', () => {
    expect(parseLuaDevelopSettingKeys()).toEqual(DEVELOP_SETTING_KEYS);
  });
});

describe('tool contracts vs Lua dispatch', () => {
  function parseLuaDispatch(): Record<string, string> {
    const pluginPath = path.resolve(process.cwd(), '..', 'plugin', 'LightroomMCP.lrplugin', 'PluginInfoProvider.lua');
    const source = fs.readFileSync(pluginPath, 'utf8');
    const match = source.match(/local DISPATCH = \{([\s\S]*?)\n\}/);
    if (!match) {
      throw new Error('DISPATCH table not found');
    }

    return Object.fromEntries(
      [...match[1].matchAll(/^\s*([a-z_]+)\s*=\s*([A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*)\s*,/gm)]
        .map((entry) => [entry[1], entry[2]]),
    );
  }

  it('matches manifest names and handler targets', () => {
    const dispatch = parseLuaDispatch();
    const manifest = Object.fromEntries(
      TOOL_CONTRACTS.map((contract) => [contract.name, contract.luaHandler]),
    );

    expect(dispatch).toEqual(manifest);
  });
});
