import { DirectClient } from "@elizaos/client-direct";
import {
    AgentRuntime,
    type Character,
    type IAgentRuntime,
    ModelProviderName,
    stringToUuid,
    validateCharacterConfig,
    elizaLogger,
    settings,
    type ClientInstance,
    CacheManager,
    FsCacheAdapter,
    DbCacheAdapter,
    type IDatabaseCacheAdapter,
    CacheStore,
    type Adapter,
    type IDatabaseAdapter,
    parseBooleanFromText,
} from "@elizaos/core";
import { defaultCharacter } from "./defaultCharacter.ts";
import { TwitterIntegration, type IAgentRuntimeWithRAG } from "./twitter.ts";
import { twitterPlugin } from "@elizaos/plugin-twitter";
import { bootstrapPlugin } from "@elizaos/plugin-bootstrap";
import JSON5 from 'json5';
import fs from "fs";
import net from "net";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import yargs from "yargs";
import Database from "better-sqlite3";
import { SqliteDatabaseAdapter } from "@elizaos/adapter-sqlite";

const __filename = fileURLToPath(import.meta.url); // get the resolved path to the file
const __dirname = path.dirname(__filename); // get the name of the directory

export const wait = (minTime = 1000, maxTime = 3000) => {
    const waitTime =
        Math.floor(Math.random() * (maxTime - minTime + 1)) + minTime;
    return new Promise((resolve) => setTimeout(resolve, waitTime));
};

const logFetch = async (url: string, options: any) => {
    elizaLogger.debug(`Fetching ${url}`);
    // Disabled to avoid disclosure of sensitive information such as API keys
    // elizaLogger.debug(JSON.stringify(options, null, 2));
    return fetch(url, options);
};

export function parseArguments(): {
    character?: string;
    characters?: string;
} {
    try {
        return yargs(process.argv.slice(3))
            .option("character", {
                type: "string",
                description: "Path to the character JSON file",
            })
            .option("characters", {
                type: "string",
                description:
                    "Comma separated list of paths to character JSON files",
            })
            .parseSync();
    } catch (error) {
        console.error("Error parsing arguments:", error);
        return {};
    }
}

function tryLoadFile(filePath: string): string | null {
    try {
        return fs.readFileSync(filePath, "utf8");
    } catch (e) {
        return null;
    }
}
function mergeCharacters(base: Character, child: Character): Character {
    const mergeObjects = (baseObj: any, childObj: any) => {
        const result: any = {};
        const keys = new Set([
            ...Object.keys(baseObj || {}),
            ...Object.keys(childObj || {}),
        ]);
        keys.forEach((key) => {
            if (
                typeof baseObj[key] === "object" &&
                typeof childObj[key] === "object" &&
                !Array.isArray(baseObj[key]) &&
                !Array.isArray(childObj[key])
            ) {
                result[key] = mergeObjects(baseObj[key], childObj[key]);
            } else if (
                Array.isArray(baseObj[key]) ||
                Array.isArray(childObj[key])
            ) {
                result[key] = [
                    ...(baseObj[key] || []),
                    ...(childObj[key] || []),
                ];
            } else {
                result[key] =
                    childObj[key] !== undefined ? childObj[key] : baseObj[key];
            }
        });
        return result;
    };
    return mergeObjects(base, child);
}
/* function isAllStrings(arr: unknown[]): boolean {
    return Array.isArray(arr) && arr.every((item) => typeof item === "string");
}
export async function loadCharacterFromOnchain(): Promise<Character[]> {
    const jsonText = onchainJson;

    console.log("JSON:", jsonText);
    if (!jsonText) return [];
    const loadedCharacters = [];
    try {
        const character = JSON5.parse(jsonText);
        validateCharacterConfig(character);

        // .id isn't really valid
        const characterId = character.id || character.name;
        const characterPrefix = `CHARACTER.${characterId
            .toUpperCase()
            .replace(/ /g, "_")}.`;

        const characterSettings = Object.entries(process.env)
            .filter(([key]) => key.startsWith(characterPrefix))
            .reduce((settings, [key, value]) => {
                const settingKey = key.slice(characterPrefix.length);
                settings[settingKey] = value;
                return settings;
            }, {});

        if (Object.keys(characterSettings).length > 0) {
            character.settings = character.settings || {};
            character.settings.secrets = {
                ...characterSettings,
                ...character.settings.secrets,
            };
        }

        // Handle plugins
        if (isAllStrings(character.plugins)) {
            elizaLogger.info("Plugins are: ", character.plugins);
            const importedPlugins = await Promise.all(
                character.plugins.map(async (plugin) => {
                    const importedPlugin = await import(plugin);
                    return importedPlugin.default;
                })
            );
            character.plugins = importedPlugins;
        }

        loadedCharacters.push(character);
        elizaLogger.info(
            `Successfully loaded character from: ${process.env.IQ_WALLET_ADDRESS}`
        );
        return loadedCharacters;
    } catch (e) {
        elizaLogger.error(
            `Error parsing character from ${process.env.IQ_WALLET_ADDRESS}: ${e}`
        );
        process.exit(1);
    }
} */

async function loadCharactersFromUrl(url: string): Promise<Character[]> {
    try {
        const response = await fetch(url);
        const responseJson = await response.json();

        let characters: Character[] = [];
        if (Array.isArray(responseJson)) {
            characters = await Promise.all(
                responseJson.map((character) => jsonToCharacter(url, character))
            );
        } else {
            const character = await jsonToCharacter(url, responseJson);
            characters.push(character);
        }
        return characters;
    } catch (e) {
        console.error(`Error loading character(s) from ${url}: `, e);
        process.exit(1);
    }
}

async function jsonToCharacter(
    filePath: string,
    character: any
): Promise<Character> {
    validateCharacterConfig(character);

    // .id isn't really valid
    const characterId = character.id || character.name;
    const characterPrefix = `CHARACTER.${characterId
        .toUpperCase()
        .replace(/ /g, "_")}.`;
    const characterSettings = Object.entries(process.env)
        .filter(([key]) => key.startsWith(characterPrefix))
        .reduce((settings, [key, value]) => {
            const settingKey = key.slice(characterPrefix.length);
            return { ...settings, [settingKey]: value };
        }, {});
    if (Object.keys(characterSettings).length > 0) {
        character.settings = character.settings || {};
        character.settings.secrets = {
            ...characterSettings,
            ...character.settings.secrets,
        };
    }
    // Handle plugins
    character.plugins = await handlePluginImporting(character.plugins);
    elizaLogger.info(character.name, 'loaded plugins:', "[\n    " + character.plugins.map(p => `"${p.npmName}"`).join(", \n    ") + "\n]");

    // Handle Post Processors plugins
    if (character.postProcessors?.length > 0) {
        elizaLogger.info(character.name, 'loading postProcessors', character.postProcessors);
        character.postProcessors = await handlePluginImporting(character.postProcessors);
    }

    // Handle extends
    if (character.extends) {
        elizaLogger.info(
            `Merging  ${character.name} character with parent characters`
        );
        for (const extendPath of character.extends) {
            const baseCharacter = await loadCharacter(
                path.resolve(path.dirname(filePath), extendPath)
            );
            character = mergeCharacters(baseCharacter, character);
            elizaLogger.info(
                `Merged ${character.name} with ${baseCharacter.name}`
            );
        }
    }
    return character;
}

async function loadCharacter(filePath: string): Promise<Character> {
    const content = tryLoadFile(filePath);
    if (!content) {
        throw new Error(`Character file not found: ${filePath}`);
    }
    const character = JSON5.parse(content);
    return jsonToCharacter(filePath, character);
}

async function loadCharacterTryPath(characterPath: string): Promise<Character> {
    let content: string | null = null;
    let resolvedPath = "";

    // Try different path resolutions in order
    const pathsToTry = [
        characterPath, // exact path as specified
        path.resolve(process.cwd(), characterPath), // relative to cwd
        path.resolve(process.cwd(), "agent", characterPath), // Add this
        path.resolve(__dirname, characterPath), // relative to current script
        path.resolve(__dirname, "characters", path.basename(characterPath)), // relative to agent/characters
        path.resolve(__dirname, "../characters", path.basename(characterPath)), // relative to characters dir from agent
        path.resolve(
            __dirname,
            "../../characters",
            path.basename(characterPath)
        ), // relative to project root characters dir
    ];

    elizaLogger.debug(
        "Trying paths:",
        pathsToTry.map((p) => ({
            path: p,
            exists: fs.existsSync(p),
        }))
    );

    for (const tryPath of pathsToTry) {
        content = tryLoadFile(tryPath);
        if (content !== null) {
            resolvedPath = tryPath;
            break;
        }
    }

    if (content === null) {
        elizaLogger.error(
            `Error loading character from ${characterPath}: File not found in any of the expected locations`
        );
        elizaLogger.error("Tried the following paths:");
        pathsToTry.forEach((p) => elizaLogger.error(` - ${p}`));
        throw new Error(
            `Error loading character from ${characterPath}: File not found in any of the expected locations`
        );
    }
    try {
        const character: Character = await loadCharacter(resolvedPath);
        elizaLogger.success(`Successfully loaded character from: ${resolvedPath}`);
        return character;
    } catch (e) {
        console.error(`Error parsing character from ${resolvedPath}: `, e);
        throw new Error(`Error parsing character from ${resolvedPath}: ${e}`);
    }
}

function commaSeparatedStringToArray(commaSeparated: string): string[] {
    return commaSeparated?.split(",").map((value) => value.trim());
}

async function readCharactersFromStorage(
    characterPaths: string[]
): Promise<string[]> {
    try {
        const uploadDir = path.join(process.cwd(), "data", "characters");
        await fs.promises.mkdir(uploadDir, { recursive: true });
        const fileNames = await fs.promises.readdir(uploadDir);
        fileNames.forEach((fileName) => {
            characterPaths.push(path.join(uploadDir, fileName));
        });
    } catch (err) {
        elizaLogger.error(`Error reading directory: ${err.message}`);
    }

    return characterPaths;
}

export async function loadCharacters(
    charactersArg: string
): Promise<Character[]> {
    let characterPaths = commaSeparatedStringToArray(charactersArg);

    if (process.env.USE_CHARACTER_STORAGE === "true") {
        characterPaths = await readCharactersFromStorage(characterPaths);
    }

    const loadedCharacters: Character[] = [];

    if (characterPaths?.length > 0) {
        for (const characterPath of characterPaths) {
            try {
                const character: Character = await loadCharacterTryPath(
                    characterPath
                );
                loadedCharacters.push(character);
            } catch (e) {
                process.exit(1);
            }
        }
    }

    if (hasValidRemoteUrls()) {
        elizaLogger.info("Loading characters from remote URLs");
        const characterUrls = commaSeparatedStringToArray(
            process.env.REMOTE_CHARACTER_URLS
        );
        for (const characterUrl of characterUrls) {
            const characters = await loadCharactersFromUrl(characterUrl);
            loadedCharacters.push(...characters);
        }
    }

    if (loadedCharacters.length === 0) {
        elizaLogger.info("No characters found, using default character");
        loadedCharacters.push(defaultCharacter);
    }

    return loadedCharacters;
}

async function handlePluginImporting(plugins: string[]) {
    if (plugins.length > 0) {
        // this logging should happen before calling, so we can include important context
        //elizaLogger.info("Plugins are: ", plugins);
        const importedPlugins = await Promise.all(
            plugins.map(async (plugin) => {
                try {
                    const importedPlugin = await import(plugin);
                    const functionName =
                        plugin
                            .replace("@elizaos/plugin-", "")
                            .replace("@elizaos-plugins/plugin-", "")
                            .replace(/-./g, (x) => x[1].toUpperCase()) +
                        "Plugin"; // Assumes plugin function is camelCased with Plugin suffix
                    if (!importedPlugin[functionName] && !importedPlugin.default) {
                      elizaLogger.warn(plugin, 'does not have an default export or', functionName)
                    }
                    const pluginExport = importedPlugin.default || importedPlugin[functionName];
                    if (!pluginExport) {
                      elizaLogger.warn(plugin, 'has no valid export');
                      return false;
                    }
                    return {...pluginExport, npmName: plugin };
                } catch (importError) {
                    console.error(
                        `Failed to import plugin: ${plugin}`,
                        importError
                    );
                    return false; // Return null for failed imports
                }
            })
        )
        // remove plugins that failed to load, so agent can try to start
        return importedPlugins.filter(p => !!p);
    } else {
        return [];
    }
}

export function getTokenForProvider(
    provider: ModelProviderName,
    character: Character
): string | undefined {
    switch (provider) {
        // no key needed for llama_local, ollama, lmstudio, gaianet or bedrock
        case ModelProviderName.LLAMALOCAL:
            return "";
        case ModelProviderName.OLLAMA:
            return "";
        case ModelProviderName.LMSTUDIO:
            return "";
        case ModelProviderName.GAIANET:
            return (
                character.settings?.secrets?.GAIA_API_KEY ||
                settings.GAIA_API_KEY
            );
        case ModelProviderName.BEDROCK:
            return "";
        case ModelProviderName.OPENAI:
            return (
                character.settings?.secrets?.OPENAI_API_KEY ||
                settings.OPENAI_API_KEY
            );
        case ModelProviderName.ETERNALAI:
            return (
                character.settings?.secrets?.ETERNALAI_API_KEY ||
                settings.ETERNALAI_API_KEY
            );
        case ModelProviderName.NINETEEN_AI:
            return (
                character.settings?.secrets?.NINETEEN_AI_API_KEY ||
                settings.NINETEEN_AI_API_KEY
            );
        case ModelProviderName.LLAMACLOUD:
        case ModelProviderName.TOGETHER:
            return (
                character.settings?.secrets?.LLAMACLOUD_API_KEY ||
                settings.LLAMACLOUD_API_KEY ||
                character.settings?.secrets?.TOGETHER_API_KEY ||
                settings.TOGETHER_API_KEY ||
                character.settings?.secrets?.OPENAI_API_KEY ||
                settings.OPENAI_API_KEY
            );
        case ModelProviderName.CLAUDE_VERTEX:
        case ModelProviderName.ANTHROPIC:
            return (
                character.settings?.secrets?.ANTHROPIC_API_KEY ||
                character.settings?.secrets?.CLAUDE_API_KEY ||
                settings.ANTHROPIC_API_KEY ||
                settings.CLAUDE_API_KEY
            );
        case ModelProviderName.REDPILL:
            return (
                character.settings?.secrets?.REDPILL_API_KEY ||
                settings.REDPILL_API_KEY
            );
        case ModelProviderName.OPENROUTER:
            return (
                character.settings?.secrets?.OPENROUTER_API_KEY ||
                settings.OPENROUTER_API_KEY
            );
        case ModelProviderName.GROK:
            return (
                character.settings?.secrets?.GROK_API_KEY ||
                settings.GROK_API_KEY
            );
        case ModelProviderName.HEURIST:
            return (
                character.settings?.secrets?.HEURIST_API_KEY ||
                settings.HEURIST_API_KEY
            );
        case ModelProviderName.GROQ:
            return (
                character.settings?.secrets?.GROQ_API_KEY ||
                settings.GROQ_API_KEY
            );
        case ModelProviderName.GALADRIEL:
            return (
                character.settings?.secrets?.GALADRIEL_API_KEY ||
                settings.GALADRIEL_API_KEY
            );
        case ModelProviderName.FAL:
            return (
                character.settings?.secrets?.FAL_API_KEY || settings.FAL_API_KEY
            );
        case ModelProviderName.ALI_BAILIAN:
            return (
                character.settings?.secrets?.ALI_BAILIAN_API_KEY ||
                settings.ALI_BAILIAN_API_KEY
            );
        case ModelProviderName.VOLENGINE:
            return (
                character.settings?.secrets?.VOLENGINE_API_KEY ||
                settings.VOLENGINE_API_KEY
            );
        case ModelProviderName.NANOGPT:
            return (
                character.settings?.secrets?.NANOGPT_API_KEY ||
                settings.NANOGPT_API_KEY
            );
        case ModelProviderName.HYPERBOLIC:
            return (
                character.settings?.secrets?.HYPERBOLIC_API_KEY ||
                settings.HYPERBOLIC_API_KEY
            );

        case ModelProviderName.VENICE:
            return (
                character.settings?.secrets?.VENICE_API_KEY ||
                settings.VENICE_API_KEY
            );
        case ModelProviderName.ATOMA:
            return (
                character.settings?.secrets?.ATOMASDK_BEARER_AUTH ||
                settings.ATOMASDK_BEARER_AUTH
            );
        case ModelProviderName.NVIDIA:
            return (
                character.settings?.secrets?.NVIDIA_API_KEY ||
                settings.NVIDIA_API_KEY
            );
        case ModelProviderName.AKASH_CHAT_API:
            return (
                character.settings?.secrets?.AKASH_CHAT_API_KEY ||
                settings.AKASH_CHAT_API_KEY
            );
        case ModelProviderName.GOOGLE:
            return (
                character.settings?.secrets?.GOOGLE_GENERATIVE_AI_API_KEY ||
                settings.GOOGLE_GENERATIVE_AI_API_KEY
            );
        case ModelProviderName.MISTRAL:
            return (
                character.settings?.secrets?.MISTRAL_API_KEY ||
                settings.MISTRAL_API_KEY
            );
        case ModelProviderName.LETZAI:
            return (
                character.settings?.secrets?.LETZAI_API_KEY ||
                settings.LETZAI_API_KEY
            );
        case ModelProviderName.INFERA:
            return (
                character.settings?.secrets?.INFERA_API_KEY ||
                settings.INFERA_API_KEY
            );
        case ModelProviderName.DEEPSEEK:
            return (
                character.settings?.secrets?.DEEPSEEK_API_KEY ||
                settings.DEEPSEEK_API_KEY
            );
        case ModelProviderName.LIVEPEER:
            return (
                character.settings?.secrets?.LIVEPEER_GATEWAY_URL ||
                settings.LIVEPEER_GATEWAY_URL
            );
        case ModelProviderName.SECRETAI:
            return (
                character.settings?.secrets?.SECRET_AI_API_KEY ||
                settings.SECRET_AI_API_KEY
            );
        case ModelProviderName.NEARAI:
            try {
                const config = JSON.parse(fs.readFileSync(path.join(os.homedir(), '.nearai/config.json'), 'utf8'));
                return JSON.stringify(config?.auth);
            } catch (e) {
                elizaLogger.warn(`Error loading NEAR AI config: ${e}`);
            }
            return (
                character.settings?.secrets?.NEARAI_API_KEY ||
                settings.NEARAI_API_KEY
            );

        default:
            const errorMessage = `Failed to get token - unsupported model provider: ${provider}`;
            elizaLogger.error(errorMessage);
            throw new Error(errorMessage);
    }
}

// also adds plugins from character file into the runtime
export async function initializeClients(
    character: Character,
    runtime: IAgentRuntime
) {
    // each client can only register once
    // and if we want two we can explicitly support it
    const clients: ClientInstance[] = [];
    // const clientTypes = clients.map((c) => c.name);
    // elizaLogger.log("initializeClients", clientTypes, "for", character.name);

    if (character.plugins?.length > 0) {
        for (const plugin of character.plugins) {
            if (plugin.clients) {
                for (const client of plugin.clients) {
                    const startedClient = await client.start(runtime);
                    elizaLogger.debug(
                        `Initializing client: ${client.name}`
                    );
                    clients.push(startedClient);
                }
            }
        }
    }

    return clients;
}

export async function createAgent(
    character: Character,
    token: string
): Promise<AgentRuntime> {
    elizaLogger.log(`Creating runtime for character ${character.name}`);
    
    // Use persistent database instead of in-memory
    const dbPath = path.join(process.cwd(), "agent", "data", "db.sqlite");
    elizaLogger.info(`Using database at: ${dbPath}`);
    
    // Ensure the data directory exists
    const dataDir = path.dirname(dbPath);
    if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
    }
    
    // Initialize database with optimized settings
    const database = new Database(dbPath, {
        verbose: elizaLogger.debug,
        // Enable WAL mode for better concurrency
        pragma: {
            journal_mode: 'WAL',
            synchronous: 'NORMAL',
            cache_size: -2000, // Use 2MB of cache
            temp_store: 'MEMORY',
            mmap_size: 30000000000, // 30GB memory map
            page_size: 4096,
            busy_timeout: 5000,
            foreign_keys: 'ON'
        }
    });

    // Create prepared statements for frequently used queries
    const preparedStatements = {
        getMessages: database.prepare(`
            SELECT * FROM conversation_messages 
            WHERE conversation_id = ? 
            ORDER BY timestamp ASC
            LIMIT ?
        `),
        getRecentMessages: database.prepare(`
            SELECT * FROM conversation_messages 
            WHERE conversation_id = ? 
            AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        `),
        getUserMessages: database.prepare(`
            SELECT * FROM conversation_messages 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
            LIMIT ?
        `),
        insertMessage: database.prepare(`
            INSERT INTO conversation_messages 
            (id, conversation_id, role, content, timestamp, user_id, is_bot, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        `),
        updateConversation: database.prepare(`
            INSERT OR REPLACE INTO conversations 
            (id, created_at, last_updated, metadata)
            VALUES (?, COALESCE((SELECT created_at FROM conversations WHERE id = ?), ?), ?, ?)
        `)
    };

    const databaseAdapter = new SqliteDatabaseAdapter(database);
    await databaseAdapter.init();

    // Create tables with proper error handling
    try {
        database.exec(`
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                last_updated INTEGER NOT NULL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                user_id TEXT,
                is_bot BOOLEAN NOT NULL,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_id 
            ON conversation_messages(conversation_id);
            
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_user_id 
            ON conversation_messages(user_id);
            
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_timestamp 
            ON conversation_messages(timestamp);

            CREATE INDEX IF NOT EXISTS idx_conversation_messages_composite 
            ON conversation_messages(conversation_id, timestamp, user_id);

            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_embeddings_text 
            ON embeddings(text);
        `);
    } catch (error) {
        elizaLogger.error('Error creating database tables:', error);
        throw error;
    }

    // Enable RAG knowledge in character settings if not already enabled
    if (!character.settings) {
        character.settings = {};
    }
    character.settings.ragKnowledge = true;

    // Set knowledge root and ensure it exists
    const knowledgeRoot = path.join(process.cwd(), "agent", "Knowledge");
    if (!fs.existsSync(knowledgeRoot)) {
        fs.mkdirSync(knowledgeRoot, { recursive: true });
    }

    // Initialize runtime with optimized settings
    const runtime = new AgentRuntime({
        character,
        token,
        modelProvider: character.modelProvider,
        logging: true,
        databaseAdapter: databaseAdapter
    });

    // Configure RAG system
    (runtime as any).knowledgeRoot = knowledgeRoot;
    (runtime as any).ragKnowledgeManager.knowledgeRoot = knowledgeRoot;

    // Initialize caches and constants
    const ragCache = new Map<string, { data: any; timestamp: number }>();
    const embeddingCache = new Map<string, { embedding: Float32Array; timestamp: number }>();
    const CACHE_TTL = 5 * 60 * 1000; // 5 minutes cache TTL
    const EMBEDDING_CACHE_TTL = 30 * 60 * 1000; // 30 minutes cache TTL
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 1000; // 1 second

    // Add optimized embedding with better error handling
    (runtime as any).embed = async (text: string) => {
        try {
            // Check embedding cache first
            const cached = embeddingCache.get(text);
            if (cached && Date.now() - cached.timestamp < EMBEDDING_CACHE_TTL) {
                return cached.embedding;
            }

            // Check database cache
            const stmt = database.prepare('SELECT embedding FROM embeddings WHERE text = ? ORDER BY created_at DESC LIMIT 1');
            const cachedEmbedding = stmt.get(text);
            if (cachedEmbedding) {
                const embedding = new Float32Array(cachedEmbedding.embedding);
                embeddingCache.set(text, { embedding, timestamp: Date.now() });
                return embedding;
            }

            // Ensure text is not too long
            const maxLength = 8192;
            const truncatedText = text.length > maxLength ? text.substring(0, maxLength) : text;

            // Add retry logic with exponential backoff
            let lastError;
            for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
                try {
                    const embeddingArray = await (runtime as any).ragKnowledgeManager.embed(truncatedText);
                    
                    if (!Array.isArray(embeddingArray) || embeddingArray.length === 0) {
                        throw new Error('Invalid embedding array received');
                    }

                    // Convert to Float32Array and ensure 384 dimensions
                    let embedding = new Float32Array(embeddingArray);
                    if (embedding.length !== 384) {
                        elizaLogger.warn(`[RAG] Converting embedding from ${embedding.length} to 384 dimensions`);
                        const reducedEmbedding = new Float32Array(384);
                        const chunkSize = Math.floor(embedding.length / 384);
                        for (let i = 0; i < 384; i++) {
                            let sum = 0;
                            for (let j = 0; j < chunkSize; j++) {
                                sum += embedding[i * chunkSize + j] || 0;
                            }
                            reducedEmbedding[i] = sum / chunkSize;
                        }
                        // Normalize the reduced embedding
                        const magnitude = Math.sqrt(reducedEmbedding.reduce((sum, val) => sum + val * val, 0));
                        for (let i = 0; i < 384; i++) {
                            reducedEmbedding[i] /= magnitude;
                        }
                        embedding = reducedEmbedding;
                    }
                    
                    // Cache in memory
                    embeddingCache.set(text, { embedding, timestamp: Date.now() });
                    
                    // Cache in database
                    try {
                        const insertStmt = database.prepare(
                            'INSERT INTO embeddings (id, text, embedding, created_at) VALUES (?, ?, ?, ?)'
                        );
                        insertStmt.run(
                            stringToUuid(Date.now().toString()),
                            text,
                            Buffer.from(embedding.buffer),
                            Date.now()
                        );
                    } catch (dbError) {
                        elizaLogger.warn('Failed to cache embedding in database:', dbError);
                    }
                    
                    return embedding;
                } catch (error) {
                    lastError = error;
                    if (attempt < MAX_RETRIES - 1) {
                        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * Math.pow(2, attempt)));
                    }
                }
            }

            elizaLogger.warn(`Embedding failed after ${MAX_RETRIES} attempts, using zero vector`);
            return new Float32Array(384).fill(0);
        } catch (error) {
            elizaLogger.error(`Embedding error: ${error}`);
            return new Float32Array(384).fill(0);
        }
    };

    // Add conversation history management with database storage
    (runtime as any).conversation = {
        // Store a message in conversation history
        storeMessage: async (conversationId: string, role: string, content: string, metadata: any = {}) => {
            try {
                const timestamp = Date.now();
                const messageId = `${conversationId}-${timestamp}`;
                const isBot = role === 'assistant';

                // Use prepared statements for faster inserts
                preparedStatements.updateConversation.run(
                    conversationId, 
                    conversationId, 
                    timestamp, 
                    timestamp, 
                    JSON.stringify(metadata)
                );

                preparedStatements.insertMessage.run(
                    messageId,
                    conversationId,
                    role,
                    content,
                    timestamp,
                    role === 'user' ? conversationId : null,
                    isBot ? 1 : 0,
                    JSON.stringify(metadata)
                );

                // Also store in RAG for semantic search with error handling
                try {
                    await (runtime as any).rag.store({
                        id: messageId,
                        content: content,
                        metadata: {
                            type: 'conversation',
                            conversationId,
                            role,
                            timestamp,
                            userId: role === 'user' ? conversationId : undefined,
                            isBot,
                            ...metadata
                        }
                    });
                } catch (ragError) {
                    elizaLogger.error(`RAG store error (non-fatal): ${ragError}`);
                    // Continue execution even if RAG store fails
                }

                return { id: messageId, timestamp };
            } catch (error) {
                elizaLogger.error(`Error storing conversation message: ${error}`);
                throw error;
            }
        },

        // Get conversation history with pagination
        getHistory: async (conversationId: string, options: { 
            limit?: number; 
            before?: number; 
            after?: number;
            userId?: string;
        } = {}) => {
            try {
                const { limit = 100, before, after, userId } = options;
                
                // Use prepared statement for faster query
                const messages = preparedStatements.getMessages.all(conversationId, limit);
                
                // Apply filters in memory for better performance
                return messages
                    .filter(msg => {
                        if (userId && msg.user_id !== userId) return false;
                        if (before && msg.timestamp >= before) return false;
                        if (after && msg.timestamp <= after) return false;
                        return true;
                    })
                    .map(msg => ({
                        ...msg,
                        metadata: JSON.parse(msg.metadata || '{}'),
                        isBot: Boolean(msg.is_bot)
                    }));
            } catch (error) {
                elizaLogger.error(`Error retrieving conversation history: ${error}`);
                return [];
            }
        },

        // Get conversation context with comprehensive history
        getContext: async (conversationId: string, query: string, options: {
            userId?: string;
            timeWindow?: number;
        } = {}) => {
            try {
                const { userId, timeWindow } = options;
                const cutoffTime = timeWindow ? Date.now() - timeWindow : 0;
                
                // Use prepared statement for faster query
                const recentHistory = preparedStatements.getRecentMessages.all(
                    conversationId,
                    cutoffTime,
                    50
                );

                // Get relevant past conversations from RAG in parallel
                const [relevantResults] = await Promise.all([
                    (runtime as any).rag.search(
                        `${query} conversation:${conversationId}${userId ? ` userId:${userId}` : ''}`,
                        20
                    )
                ]);

                return {
                    recentHistory: recentHistory.map(msg => ({
                        ...msg,
                        metadata: JSON.parse(msg.metadata || '{}'),
                        isBot: Boolean(msg.is_bot)
                    })),
                    relevantPast: relevantResults.map(result => ({
                        role: result.metadata.role,
                        content: result.content,
                        timestamp: result.metadata.timestamp,
                        userId: result.metadata.userId,
                        isBot: result.metadata.isBot,
                        metadata: result.metadata
                    }))
                };
            } catch (error) {
                elizaLogger.error(`Error getting conversation context: ${error}`);
                return { recentHistory: [], relevantPast: [] };
            }
        },

        // Get all conversations for a user
        getUserConversations: async (userId: string, options: {
            limit?: number;
            before?: number;
            after?: number;
        } = {}) => {
            try {
                const { limit = 100 } = options;
                
                // Use prepared statement for faster query
                const messages = preparedStatements.getUserMessages.all(userId, limit);
                
                // Group messages by conversation
                const conversations = new Map();
                messages.forEach(msg => {
                    if (!conversations.has(msg.conversation_id)) {
                        conversations.set(msg.conversation_id, {
                            id: msg.conversation_id,
                            messages: []
                        });
                    }
                    conversations.get(msg.conversation_id).messages.push({
                        ...msg,
                        metadata: JSON.parse(msg.metadata || '{}'),
                        isBot: Boolean(msg.is_bot)
                    });
                });

                return Array.from(conversations.values());
            } catch (error) {
                elizaLogger.error(`Error getting user conversations: ${error}`);
                return [];
            }
        }
    };

    (runtime as any).rag = {
        search: async (query: string, limit: number) => {
            try {
                // Extract username from query
                const username = query.split('from:')[1]?.split(' ')[0] || '';
                elizaLogger.info(`[RAG] Search request for user: ${username}`);

                // Check cache first
                const cacheKey = `search:${query}:${limit}:${username}`;
                const cached = ragCache.get(cacheKey);
                if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
                    elizaLogger.info(`[RAG] Using cached results for user ${username}`);
                    return cached.data;
                }

                // Perform search with retries
                let attempts = 0;
                const maxAttempts = 3;
                while (attempts < maxAttempts) {
                    try {
                        // Generate embedding for the query
                        const queryEmbedding = await (runtime as any).embed(query);
                        
                        // Search using the knowledge manager
                        const results = await (runtime as any).ragKnowledgeManager.searchKnowledge({
                            agentId: runtime.agentId,
                            embedding: queryEmbedding,
                            match_threshold: 0.5,
                            match_count: limit * 2, // Get more results for better reranking
                            searchText: query,
                            metadata: {
                                username: username
                            }
                        });
                        
                        if (results && results.length > 0) {
                            // Cache successful results
                            ragCache.set(cacheKey, { data: results, timestamp: Date.now() });
                            elizaLogger.info(`[RAG] Search successful for user ${username}, found ${results.length} results`);
                            return results;
                        } else {
                            elizaLogger.warn(`[RAG] No results found for user ${username}`);
                            return [];
                        }
                    } catch (error) {
                        attempts++;
                        elizaLogger.error(`[RAG] Search attempt ${attempts} failed: ${error.message}`);
                        if (attempts === maxAttempts) {
                            elizaLogger.error(`[RAG] Search failed after ${maxAttempts} attempts:`, error);
                            return [];
                        }
                        await new Promise(resolve => setTimeout(resolve, 1000 * attempts));
                    }
                }
            } catch (error) {
                elizaLogger.error(`[RAG] Search error: ${error}`);
                return [];
            }
        },
        store: async (data: { id: string; content: string; metadata: any; embedding: Buffer }) => {
            try {
                // Convert Buffer to number array and ensure 384 dimensions
                let embeddingArray = Array.from(new Float32Array(data.embedding.buffer));
                if (embeddingArray.length !== 384) {
                    elizaLogger.warn(`[RAG] Converting embedding from ${embeddingArray.length} to 384 dimensions`);
                    const reducedEmbedding = new Float32Array(384);
                    const chunkSize = Math.floor(embeddingArray.length / 384);
                    for (let i = 0; i < 384; i++) {
                        let sum = 0;
                        for (let j = 0; j < chunkSize; j++) {
                            sum += embeddingArray[i * chunkSize + j] || 0;
                        }
                        reducedEmbedding[i] = sum / chunkSize;
                    }
                    // Normalize the reduced embedding
                    const magnitude = Math.sqrt(reducedEmbedding.reduce((sum, val) => sum + val * val, 0));
                    for (let i = 0; i < 384; i++) {
                        reducedEmbedding[i] /= magnitude;
                    }
                    embeddingArray = Array.from(reducedEmbedding);
                }
                
                // Store using the knowledge manager
                await (runtime as any).ragKnowledgeManager.createKnowledge({
                    id: data.id,
                    agentId: runtime.agentId,
                    content: {
                        text: data.content,
                        metadata: data.metadata
                    },
                    embedding: embeddingArray,
                    createdAt: Date.now()
                });
                
                elizaLogger.info(`[RAG] Successfully stored embedding for id ${data.id}`);
            } catch (error) {
                elizaLogger.error(`[RAG] Store error: ${error}`);
                throw error;
            }
        }
    };

    // Cast runtime to IAgentRuntimeWithRAG after adding rag property
    const runtimeWithRag = runtime as IAgentRuntimeWithRAG;

    // Use runtimeWithRag for TwitterIntegration
    const twitterIntegration = new TwitterIntegration(runtimeWithRag);

    return runtime;
}

function initializeFsCache(baseDir: string, character: Character) {
    if (!character?.id) {
        throw new Error(
            "initializeFsCache requires id to be set in character definition"
        );
    }
    const cacheDir = path.resolve(baseDir, character.id, "cache");

    const cache = new CacheManager(new FsCacheAdapter(cacheDir));
    return cache;
}

function initializeDbCache(character: Character, db: IDatabaseCacheAdapter) {
    if (!character?.id) {
        throw new Error(
            "initializeFsCache requires id to be set in character definition"
        );
    }
    const cache = new CacheManager(new DbCacheAdapter(db, character.id));
    return cache;
}

function initializeCache(
    cacheStore: string,
    character: Character,
    baseDir?: string,
    db?: IDatabaseCacheAdapter
) {
    switch (cacheStore) {
        // case CacheStore.REDIS:
        //     if (process.env.REDIS_URL) {
        //         elizaLogger.info("Connecting to Redis...");
        //         const redisClient = new RedisClient(process.env.REDIS_URL);
        //         if (!character?.id) {
        //             throw new Error(
        //                 "CacheStore.REDIS requires id to be set in character definition"
        //             );
        //         }
        //         return new CacheManager(
        //             new DbCacheAdapter(redisClient, character.id) // Using DbCacheAdapter since RedisClient also implements IDatabaseCacheAdapter
        //         );
        //     } else {
        //         throw new Error("REDIS_URL environment variable is not set.");
        //     }

        case CacheStore.DATABASE:
            if (db) {
                elizaLogger.info("Using Database Cache...");
                return initializeDbCache(character, db);
            } else {
                throw new Error(
                    "Database adapter is not provided for CacheStore.Database."
                );
            }

        case CacheStore.FILESYSTEM:
            elizaLogger.info("Using File System Cache...");
            if (!baseDir) {
                throw new Error(
                    "baseDir must be provided for CacheStore.FILESYSTEM."
                );
            }
            return initializeFsCache(baseDir, character);

        default:
            throw new Error(
                `Invalid cache store: ${cacheStore} or required configuration missing.`
            );
    }
}

async function findDatabaseAdapter(runtime: AgentRuntime) {
  const { adapters } = runtime;
  let adapter: Adapter | undefined;
  // if not found, default to sqlite
  if (adapters.length === 0) {
    const sqliteAdapterPlugin = await import('@elizaos-plugins/adapter-sqlite');
    const sqliteAdapterPluginDefault = sqliteAdapterPlugin.default;
    adapter = sqliteAdapterPluginDefault.adapters[0];
    if (!adapter) {
      throw new Error("Internal error: No database adapter found for default adapter-sqlite");
    }
  } else if (adapters.length === 1) {
    adapter = adapters[0];
  } else {
    throw new Error("Multiple database adapters found. You must have no more than one. Adjust your plugins configuration.");
    }
  const adapterInterface = adapter?.init(runtime);
  return adapterInterface;
}

async function initializeTwitter(runtime: AgentRuntime) {
    if (!process.env.TWITTER_USERNAME || !process.env.TWITTER_PASSWORD || !process.env.TWITTER_EMAIL) {
        elizaLogger.warn('Twitter credentials not found. Twitter integration will not be available.');
        return null;
    }

    try {
        elizaLogger.info('Initializing Twitter integration...');
        // Add RAG interface to runtime
        (runtime as any).rag = {
            search: async (query: string, limit: number) => {
                return await (runtime as any).ragKnowledgeManager.search(query, limit);
            },
            store: async (data: { id: string; content: string; metadata: any }) => {
                return await (runtime as any).ragKnowledgeManager.store(data);
            }
        };
        const runtimeWithRag = runtime as IAgentRuntimeWithRAG;
        const twitterIntegration = new TwitterIntegration(runtimeWithRag);
        await twitterIntegration.initialize();
        
        if (!twitterIntegration.initialized) {
            throw new Error('Twitter integration failed to initialize properly');
        }
        
        elizaLogger.info('Twitter integration initialized successfully');
        return twitterIntegration;
    } catch (error) {
        elizaLogger.error('Failed to initialize Twitter integration:', error);
        elizaLogger.warn('Agent will continue without Twitter integration');
        return null;
    }
}

async function startAgent(
    character: Character,
    directClient: DirectClient
): Promise<AgentRuntime> {
    let db: IDatabaseAdapter & IDatabaseCacheAdapter;
    try {
        character.id ??= stringToUuid(character.name);
        character.username ??= character.name;

        const token = getTokenForProvider(character.modelProvider, character);

        const runtime: AgentRuntime = await createAgent(
            character,
            token
        );

        // initialize database
        db = await findDatabaseAdapter(runtime);
        runtime.databaseAdapter = db;

        // initialize cache
        const cache = initializeCache(
            process.env.CACHE_STORE ?? CacheStore.DATABASE,
            character,
            process.env.CACHE_DIR ?? "",
            db
        );
        runtime.cacheManager = cache;

        // Initialize RAG system and process knowledge files
        elizaLogger.info('Initializing RAG system and processing knowledge files...');
        await runtime.initialize();
        elizaLogger.info('RAG system and knowledge files processed successfully');

        // Initialize Twitter integration if credentials are available
        let twitterIntegration: TwitterIntegration | null = null;
        if (process.env.TWITTER_USERNAME && process.env.TWITTER_PASSWORD && process.env.TWITTER_EMAIL) {
            try {
                elizaLogger.info('Initializing Twitter integration...');
                // Add RAG interface to runtime
                (runtime as any).rag = {
                    search: async (query: string, limit: number) => {
                        return await (runtime as any).ragKnowledgeManager.search(query, limit);
                    },
                    store: async (data: { id: string; content: string; metadata: any }) => {
                        return await (runtime as any).ragKnowledgeManager.store(data);
                    }
                };
                const runtimeWithRag = runtime as IAgentRuntimeWithRAG;
                twitterIntegration = new TwitterIntegration(runtimeWithRag);
                await twitterIntegration.initialize();
                
                // Add Twitter integration to runtime for persistence
                (runtime as any).twitterIntegration = twitterIntegration;
                
                // Verify initialization
                if (!twitterIntegration.initialized) {
                    throw new Error('Twitter integration failed to initialize properly');
                }
                
                elizaLogger.info('Twitter integration initialized successfully');
            } catch (error) {
                elizaLogger.error('Failed to initialize Twitter integration:', error);
                // Don't throw here, allow the agent to continue without Twitter
                elizaLogger.warn('Agent will continue without Twitter integration');
            }
        } else {
            elizaLogger.warn('Twitter credentials not found. Twitter integration will not be available.');
        }

        // start assigned clients
        runtime.clients = await initializeClients(character, runtime);

        // add to container
        directClient.registerAgent(runtime);

        // report to console
        elizaLogger.debug(`Started ${character.name} as ${runtime.agentId}`);

        return runtime;
    } catch (error) {
        elizaLogger.error(
            `Error starting agent for character ${character.name}:`,
            error
        );
        elizaLogger.error(error);
        if (db) {
            await db.close();
        }
        throw error;
    }
}

const checkPortAvailable = (port: number): Promise<boolean> => {
    return new Promise((resolve) => {
        const server = net.createServer();

        server.once("error", (err: NodeJS.ErrnoException) => {
            if (err.code === "EADDRINUSE") {
                resolve(false);
            }
        });

        server.once("listening", () => {
            server.close();
            resolve(true);
        });

        server.listen(port);
    });
};

const hasValidRemoteUrls = () =>
    process.env.REMOTE_CHARACTER_URLS &&
    process.env.REMOTE_CHARACTER_URLS !== "" &&
    process.env.REMOTE_CHARACTER_URLS.startsWith("http");

/**
 * Post processing of character after loading
 * @param character
 */
const handlePostCharacterLoaded = async (character: Character): Promise<Character> => {
    let processedCharacter = character;
    // Filtering the plugins with the method of handlePostCharacterLoaded
    const processors = character?.postProcessors?.filter(p => typeof p.handlePostCharacterLoaded === 'function');
    if (processors?.length > 0) {
        processedCharacter = Object.assign({}, character, { postProcessors: undefined });
        // process the character with each processor
        // the order is important, so we loop through the processors
        for (let i = 0; i < processors.length; i++) {
            const processor = processors[i];
            processedCharacter = await processor.handlePostCharacterLoaded(processedCharacter);
        }
    }
    return processedCharacter;
}

const startAgents = async () => {
    try {
        // Load character configuration
        const { character } = await import('./character.ts');
        elizaLogger.info('Character configuration loaded successfully');

        // Get the token for the model provider
        const token = getTokenForProvider(character.modelProvider, character);
        if (!token) {
            throw new Error(`No token found for model provider ${character.modelProvider}`);
        }

        // Initialize database
        const dbPath = path.join(process.cwd(), "agent", "data", "db.sqlite");
        elizaLogger.info(`Using database at: ${dbPath}`);
        
        // Ensure the data directory exists
        const dataDir = path.dirname(dbPath);
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir, { recursive: true });
        }
        
        const database = new Database(dbPath);
        const databaseAdapter = new SqliteDatabaseAdapter(database);
        await databaseAdapter.init();

        // Initialize the agent runtime with optimized RAG system
        const runtime = await createAgent(character, token);

        // Initialize RAG system and process knowledge files with retries
        elizaLogger.info('Initializing RAG system and processing knowledge files...');
        let initAttempts = 0;
        const maxInitAttempts = 3;
        
        while (initAttempts < maxInitAttempts) {
            try {
                // Ensure knowledge directory exists
                const knowledgeRoot = path.join(process.cwd(), "agent", "Knowledge");
                if (!fs.existsSync(knowledgeRoot)) {
                    fs.mkdirSync(knowledgeRoot, { recursive: true });
                }

                // Add basic knowledge if none exists
                const basicKnowledgePath = path.join(knowledgeRoot, "basic_knowledge.txt");
                if (!fs.existsSync(basicKnowledgePath)) {
                    const basicKnowledge = `This is a basic knowledge file for testing the RAG system.

Key Concepts:
1. The RAG (Retrieval-Augmented Generation) system combines retrieval-based and generation-based approaches.
2. It uses embeddings to find relevant information from the knowledge base.
3. The system can process both text and PDF files.
4. Knowledge files should be placed in the agent/Knowledge directory.

Example Questions and Answers:
Q: What is RAG?
A: RAG stands for Retrieval-Augmented Generation, a system that combines retrieval of relevant information with text generation.

Q: How does the knowledge system work?
A: The system processes knowledge files, creates embeddings, and uses them to find relevant information when answering questions.

Q: Where should knowledge files be placed?
A: Knowledge files should be placed in the agent/Knowledge directory for the system to process them.`;
                    fs.writeFileSync(basicKnowledgePath, basicKnowledge);
                }

                // Initialize the runtime
        await runtime.initialize();
        elizaLogger.info('RAG system and knowledge files processed successfully');
                break;
            } catch (error) {
                initAttempts++;
                if (initAttempts === maxInitAttempts) {
                    elizaLogger.error('Failed to initialize RAG system after multiple attempts:', error);
                    throw error;
                }
                elizaLogger.warn(`RAG initialization attempt ${initAttempts} failed, retrying...`);
                await new Promise(resolve => setTimeout(resolve, 2000 * initAttempts));
            }
        }

        // Initialize Twitter integration if credentials are available
        let twitterIntegration: TwitterIntegration | null = null;
        if (process.env.TWITTER_USERNAME && process.env.TWITTER_PASSWORD && process.env.TWITTER_EMAIL) {
            try {
                elizaLogger.info('Initializing Twitter integration...');
                twitterIntegration = new TwitterIntegration(runtime as IAgentRuntimeWithRAG);
                await twitterIntegration.initialize();
                
                // Add Twitter integration to runtime for persistence
                (runtime as any).twitterIntegration = twitterIntegration;
                
                // Verify initialization
                if (!twitterIntegration.initialized) {
                    throw new Error('Twitter integration failed to initialize properly');
                }
                
                elizaLogger.info('Twitter integration initialized successfully');
            } catch (error) {
                elizaLogger.error('Failed to initialize Twitter integration:', error);
                // Don't throw here, allow the agent to continue without Twitter
                elizaLogger.warn('Agent will continue without Twitter integration');
            }
        } else {
            elizaLogger.warn('Twitter credentials not found. Twitter integration will not be available.');
        }

        // Keep the process running
        process.on('SIGINT', async () => {
            elizaLogger.info('Shutting down...');
            if (twitterIntegration) {
                await twitterIntegration.stop();
            }
            await database.close();
            process.exit(0);
        });

    } catch (error) {
        elizaLogger.error('Error during initialization:', error);
        process.exit(1);
    }
};

// Start the agent
startAgents().catch(error => {
    elizaLogger.error('Unhandled error:', error);
    process.exit(1);
});

// Prevent unhandled exceptions from crashing the process if desired
if (
    process.env.PREVENT_UNHANDLED_EXIT &&
    parseBooleanFromText(process.env.PREVENT_UNHANDLED_EXIT)
) {
    // Handle uncaught exceptions to prevent the process from crashing
    process.on("uncaughtException", (err) => {
        console.error("uncaughtException", err);
    });

    // Handle unhandled rejections to prevent the process from crashing
    process.on("unhandledRejection", (err) => {
        console.error("unhandledRejection", err);
    });
}