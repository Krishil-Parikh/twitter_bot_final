import { getEmbeddingModelSettings, getEndpoint } from "./models.ts";
import { type IAgentRuntime, ModelProviderName } from "./types.ts";
import settings from "./settings.ts";
import elizaLogger from "./logger.ts";
import LocalEmbeddingModelManager from "./localembeddingManager.ts";
import { IAgentRuntime as CoreIAgentRuntime } from '@elizaos/core';

interface EmbeddingOptions {
    model: string;
    endpoint: string;
    apiKey?: string;
    length?: number;
    isOllama?: boolean;
    dimensions?: number;
    provider?: string;
}

export const EmbeddingProvider = {
    OpenAI: "OpenAI",
    Ollama: "Ollama",
    GaiaNet: "GaiaNet",
    Heurist: "Heurist",
    BGE: "BGE",
} as const;

export type EmbeddingProviderType =
    (typeof EmbeddingProvider)[keyof typeof EmbeddingProvider];

export type EmbeddingConfig = {
    readonly dimensions: number;
    readonly model: string;
    readonly provider: EmbeddingProviderType;
};

export const getEmbeddingConfig = (): EmbeddingConfig => ({
    dimensions:
        settings.USE_OPENAI_EMBEDDING?.toLowerCase() === "true"
            ? getEmbeddingModelSettings(ModelProviderName.OPENAI).dimensions
            : settings.USE_OLLAMA_EMBEDDING?.toLowerCase() === "true"
              ? getEmbeddingModelSettings(ModelProviderName.OLLAMA).dimensions
              : settings.USE_GAIANET_EMBEDDING?.toLowerCase() === "true"
                ? getEmbeddingModelSettings(ModelProviderName.GAIANET)
                      .dimensions
                : settings.USE_HEURIST_EMBEDDING?.toLowerCase() === "true"
                  ? getEmbeddingModelSettings(ModelProviderName.HEURIST)
                        .dimensions
                  : 384, // BGE
    model:
        settings.USE_OPENAI_EMBEDDING?.toLowerCase() === "true"
            ? getEmbeddingModelSettings(ModelProviderName.OPENAI).name
            : settings.USE_OLLAMA_EMBEDDING?.toLowerCase() === "true"
              ? getEmbeddingModelSettings(ModelProviderName.OLLAMA).name
              : settings.USE_GAIANET_EMBEDDING?.toLowerCase() === "true"
                ? getEmbeddingModelSettings(ModelProviderName.GAIANET).name
                : settings.USE_HEURIST_EMBEDDING?.toLowerCase() === "true"
                  ? getEmbeddingModelSettings(ModelProviderName.HEURIST).name
                  : "BGE-small-en-v1.5",
    provider:
        settings.USE_OPENAI_EMBEDDING?.toLowerCase() === "true"
            ? "OpenAI"
            : settings.USE_OLLAMA_EMBEDDING?.toLowerCase() === "true"
              ? "Ollama"
              : settings.USE_GAIANET_EMBEDDING?.toLowerCase() === "true"
                ? "GaiaNet"
                : settings.USE_HEURIST_EMBEDDING?.toLowerCase() === "true"
                  ? "Heurist"
                  : "BGE",
});

async function getRemoteEmbedding(
    input: string,
    options: EmbeddingOptions
): Promise<number[]> {
    // Ensure endpoint ends with /v1 for OpenAI
    const baseEndpoint = options.endpoint.endsWith("/v1")
        ? options.endpoint
        : `${options.endpoint}${options.isOllama ? "/v1" : ""}`;

    // Construct full URL
    const fullUrl = `${baseEndpoint}/embeddings`;

    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            ...(options.apiKey
                ? {
                      Authorization: `Bearer ${options.apiKey}`,
                  }
                : {}),
        },
        body: JSON.stringify({
            input,
            model: options.model,
            dimensions:
                options.dimensions ||
                options.length ||
                getEmbeddingConfig().dimensions,
        }),
    };

    try {
        elizaLogger.debug(`Making remote embedding request to ${fullUrl}`);
        const response = await fetch(fullUrl, requestOptions);

        if (!response.ok) {
            const errorText = await response.text();
            elizaLogger.error("API Response:", errorText);
            throw new Error(
                `Embedding API Error: ${response.status} ${response.statusText}`
            );
        }

        interface EmbeddingResponse {
            data: Array<{ embedding: number[] }>;
        }

        const data: EmbeddingResponse = await response.json();
        if (!data?.data?.[0]?.embedding) {
            throw new Error("Invalid response format from embedding API");
        }

        const embedding = data.data[0].embedding;
        if (embedding.length === 0) {
            throw new Error("Empty embedding received from API");
        }

        elizaLogger.debug(`Received embedding of size ${embedding.length}`);
        return embedding;
    } catch (e) {
        elizaLogger.error("Full error details:", e);
        throw e;
    }
}

export function getEmbeddingType(runtime: IAgentRuntime): "local" | "remote" {
    const isNode =
        typeof process !== "undefined" &&
        process.versions != null &&
        process.versions.node != null;

    // Use local embedding if:
    // - Running in Node.js
    // - Not using OpenAI provider
    // - Not forcing OpenAI embeddings
    const isLocal =
        isNode &&
        runtime.character.modelProvider !== ModelProviderName.OPENAI &&
        runtime.character.modelProvider !== ModelProviderName.GAIANET &&
        runtime.character.modelProvider !== ModelProviderName.HEURIST &&
        !settings.USE_OPENAI_EMBEDDING;

    return isLocal ? "local" : "remote";
}

export function getEmbeddingZeroVector(): number[] {
    let embeddingDimension = 384; // Default BGE dimension

    if (settings.USE_OPENAI_EMBEDDING?.toLowerCase() === "true") {
        embeddingDimension = getEmbeddingModelSettings(
            ModelProviderName.OPENAI
        ).dimensions; // OpenAI dimension
    } else if (settings.USE_OLLAMA_EMBEDDING?.toLowerCase() === "true") {
        embeddingDimension = getEmbeddingModelSettings(
            ModelProviderName.OLLAMA
        ).dimensions; // Ollama mxbai-embed-large dimension
    } else if (settings.USE_GAIANET_EMBEDDING?.toLowerCase() === "true") {
        embeddingDimension = getEmbeddingModelSettings(
            ModelProviderName.GAIANET
        ).dimensions; // GaiaNet dimension
    } else if (settings.USE_HEURIST_EMBEDDING?.toLowerCase() === "true") {
        embeddingDimension = getEmbeddingModelSettings(
            ModelProviderName.HEURIST
        ).dimensions; // Heurist dimension
    }

    return Array(embeddingDimension).fill(0);
}

export interface IAgentRuntimeWithEmbedding extends CoreIAgentRuntime {
    cache: {
        get(key: string): Promise<any>;
        set(key: string, value: any): Promise<void>;
    };
    generateEmbedding(text: string): Promise<number[]>;
}

async function retrieveCachedEmbedding(runtime: IAgentRuntimeWithEmbedding, text: string): Promise<number[] | null> {
    try {
        const cacheKey = `embedding:${text}`;
        const cached = await runtime.cache.get(cacheKey);
        if (cached) {
            elizaLogger.info(`[Embed] Retrieved cached embedding of size ${cached.length}`);
            return cached;
        }
        return null;
    } catch (error) {
        elizaLogger.warn('[Embed] Cache retrieval failed:', error);
        return null;
    }
}

async function cacheEmbedding(runtime: IAgentRuntimeWithEmbedding, text: string, embedding: number[]): Promise<void> {
    try {
        const cacheKey = `embedding:${text}`;
        await runtime.cache.set(cacheKey, embedding);
        elizaLogger.info(`[Embed] Cached embedding of size ${embedding.length}`);
    } catch (error) {
        elizaLogger.warn('[Embed] Cache storage failed:', error);
    }
}

/**
 * Gets embeddings from a remote API endpoint.  Falls back to local BGE/384
 *
 * @param {string} input - The text to generate embeddings for
 * @param {EmbeddingOptions} options - Configuration options including:
 *   - model: The model name to use
 *   - endpoint: Base API endpoint URL
 *   - apiKey: Optional API key for authentication
 *   - isOllama: Whether this is an Ollama endpoint
 *   - dimensions: Desired embedding dimensions
 * @param {IAgentRuntime} runtime - The agent runtime context
 * @returns {Promise<number[]>} Array of embedding values
 * @throws {Error} If the API request fails
 */

export async function embed(runtime: IAgentRuntimeWithEmbedding, text: string): Promise<number[]> {
    elizaLogger.info(`[Embed] Starting embedding generation for text length: ${text.length}`);
    elizaLogger.info(`[Embed] Text preview: ${text.slice(0, 100)}...`);

    // Input validation
    if (!text || text.trim().length === 0) {
        elizaLogger.warn('[Embed] Empty input text, returning zero vector');
        return new Array(384).fill(0);
    }

    try {
        // Try to get from cache first
        const cachedEmbedding = await retrieveCachedEmbedding(runtime, text);
        if (cachedEmbedding) {
            elizaLogger.info(`[Embed] Using cached embedding of size ${cachedEmbedding.length}`);
            return cachedEmbedding;
        }

        // Try local embedding first
        elizaLogger.info('[Embed] Attempting local embedding generation...');
        try {
            const localEmbedding = await getLocalEmbedding(runtime, text);
            if (localEmbedding && localEmbedding.length > 0) {
                elizaLogger.info(`[Embed] Successfully generated local embedding of size ${localEmbedding.length}`);
                await cacheEmbedding(runtime, text, localEmbedding);
                return localEmbedding;
            }
        } catch (error) {
            elizaLogger.warn('[Embed] Local embedding generation failed:', {
                error: error.message,
                stack: error.stack,
                code: error.code,
                name: error.name
            });
        }

        // Fallback to remote embedding
        elizaLogger.info('[Embed] Attempting remote embedding generation...');
        try {
            const remoteEmbedding = await getRemoteEmbedding(text, {
                model: 'text-embedding-3-small',
                endpoint: process.env.OPENAI_API_ENDPOINT || 'https://api.openai.com/v1/embeddings',
                apiKey: process.env.OPENAI_API_KEY,
                dimensions: 384
            });

            if (!remoteEmbedding || !Array.isArray(remoteEmbedding)) {
                throw new Error(`Invalid remote embedding response: ${typeof remoteEmbedding}`);
            }

            if (remoteEmbedding.length === 0) {
                throw new Error('Empty remote embedding array');
            }

            elizaLogger.info(`[Embed] Successfully generated remote embedding of size ${remoteEmbedding.length}`);
            await cacheEmbedding(runtime, text, remoteEmbedding);
            return remoteEmbedding;
        } catch (error) {
            elizaLogger.error('[Embed] Remote embedding generation failed:', {
                error: error.message,
                stack: error.stack,
                code: error.code,
                name: error.name,
                text: text.slice(0, 100)
            });
            throw error;
        }
    } catch (error) {
        elizaLogger.error('[Embed] All embedding attempts failed:', {
            error: error.message,
            stack: error.stack,
            code: error.code,
            name: error.name,
            text: text.slice(0, 100)
        });
        return new Array(384).fill(0);
    }
}

async function getLocalEmbedding(runtime: IAgentRuntimeWithEmbedding, text: string): Promise<number[]> {
    elizaLogger.info('[Embed] Starting local embedding generation...');
    try {
        const embedding = await runtime.generateEmbedding(text);
        if (!embedding || !Array.isArray(embedding)) {
            throw new Error(`Invalid local embedding response: ${typeof embedding}`);
        }
        if (embedding.length === 0) {
            throw new Error('Empty local embedding array');
        }
        elizaLogger.info(`[Embed] Local embedding generated successfully, size: ${embedding.length}`);
        return embedding;
    } catch (error) {
        elizaLogger.error('[Embed] Local embedding generation failed:', {
            error: error.message,
            stack: error.stack,
            code: error.code,
            name: error.name,
            text: text.slice(0, 100)
        });
        throw error;
    }
}
