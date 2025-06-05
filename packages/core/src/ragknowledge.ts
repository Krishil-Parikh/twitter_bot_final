import { embed } from "./embedding.ts";
import { splitChunks } from "./generation.ts";
import elizaLogger from "./logger.ts";
import {
    type IAgentRuntime,
    type IRAGKnowledgeManager,
    type RAGKnowledgeItem,
    type UUID,
    KnowledgeScope,
} from "./types.ts";
import { stringToUuid } from "./uuid.ts";
import { existsSync } from "fs";
import { join } from "path";

// Extend IAgentRuntime to include embedding properties
interface IAgentRuntimeWithEmbedding extends IAgentRuntime {
    embeddingProvider?: string;
    embeddingModel?: string;
    cache: {
        get(key: string): Promise<any>;
        set(key: string, value: any): Promise<void>;
    };
    generateEmbedding(text: string): Promise<number[]>;
}

/**
 * Manage knowledge in the database.
 */
export class RAGKnowledgeManager implements IRAGKnowledgeManager {
    /**
     * The AgentRuntime instance associated with this manager.
     */
    runtime: IAgentRuntimeWithEmbedding;

    /**
     * The name of the database table this manager operates on.
     */
    tableName: string;

    /**
     * The root directory where RAG knowledge files are located (internal)
     */
    knowledgeRoot: string;

    /**
     * Constructs a new KnowledgeManager instance.
     * @param opts Options for the manager.
     * @param opts.tableName The name of the table this manager will operate on.
     * @param opts.runtime The AgentRuntime instance associated with this manager.
     */
    constructor(opts: {
        tableName: string;
        runtime: IAgentRuntimeWithEmbedding;
        knowledgeRoot: string;
    }) {
        this.runtime = opts.runtime;
        this.tableName = opts.tableName;
        this.knowledgeRoot = opts.knowledgeRoot;
    }

    private readonly defaultRAGMatchThreshold = 0.85;
    private readonly defaultRAGMatchCount = 8;

    /**
     * Common English stop words to filter out from query analysis
     */
    private readonly stopWords = new Set([
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "hey",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "there",
        "their",
        "they",
        "your",
        "you",
    ]);

    /**
     * Filters out stop words and returns meaningful terms
     */
    private getQueryTerms(query: string): string[] {
        return query
            .toLowerCase()
            .split(" ")
            .filter((term) => term.length > 2) // Filter very short words
            .filter((term) => !this.stopWords.has(term)); // Filter stop words
    }

    /**
     * Preprocesses text content for better RAG performance.
     * @param content The text content to preprocess.
     * @returns The preprocessed text.
     */
    private preprocess(content: string): string {
        if (!content || typeof content !== "string") {
            elizaLogger.warn("Invalid input for preprocessing");
            return "";
        }

        return (
            content
                .replace(/```[\s\S]*?```/g, "")
                .replace(/`.*?`/g, "")
                .replace(/#{1,6}\s*(.*)/g, "$1")
                .replace(/!\[(.*?)\]\(.*?\)/g, "$1")
                .replace(/\[(.*?)\]\(.*?\)/g, "$1")
                .replace(/(https?:\/\/)?(www\.)?([^\s]+\.[^\s]+)/g, "$3")
                .replace(/<@[!&]?\d+>/g, "")
                .replace(/<[^>]*>/g, "")
                .replace(/^\s*[-*_]{3,}\s*$/gm, "")
                .replace(/\/\*[\s\S]*?\*\//g, "")
                .replace(/\/\/.*/g, "")
                .replace(/\s+/g, " ")
                .replace(/\n{3,}/g, "\n\n")
                // .replace(/[^a-zA-Z0-9\s\-_./:?=&]/g, "") --this strips out CJK characters
                .trim()
                .toLowerCase()
        );
    }

    private hasProximityMatch(text: string, terms: string[]): boolean {
        if (!text || !terms.length) {
            return false;
        }
    
        const words = text.toLowerCase().split(" ").filter(w => w.length > 0);
        
        // Find all positions for each term (not just first occurrence)
        const allPositions = terms.flatMap(term => 
            words.reduce((positions, word, idx) => {
                if (word.includes(term)) positions.push(idx);
                return positions;
            }, [] as number[])
        ).sort((a, b) => a - b);
    
        if (allPositions.length < 2) return false;
    
        // Check proximity
        for (let i = 0; i < allPositions.length - 1; i++) {
            if (Math.abs(allPositions[i] - allPositions[i + 1]) <= 5) {
                elizaLogger.debug("[Proximity Match]", {
                    terms,
                    positions: allPositions,
                    matchFound: `${allPositions[i]} - ${allPositions[i + 1]}`
                });
                return true;
            }
        }
    
        return false;
    }

    async getKnowledge(params: {
        query?: string;
        id?: UUID;
        conversationContext?: string;
        limit?: number;
        agentId?: UUID;
    }): Promise<RAGKnowledgeItem[]> {
        const agentId = params.agentId || this.runtime.agentId;

        // If id is provided, do direct lookup first
        if (params.id) {
            const directResults =
                await this.runtime.databaseAdapter.getKnowledge({
                    id: params.id,
                    agentId: agentId,
                });

            if (directResults.length > 0) {
                return directResults;
            }
        }

        // If no id or no direct results, perform semantic search
        if (params.query) {
            try {
                const processedQuery = this.preprocess(params.query);

                // Build search text with optional context
                let searchText = processedQuery;
                if (params.conversationContext) {
                    const relevantContext = this.preprocess(
                        params.conversationContext
                    );
                    searchText = `${relevantContext} ${processedQuery}`;
                }

                const embeddingArray = await embed(this.runtime, searchText);
                const embedding = new Float32Array(embeddingArray);

                // Get results with single query
                const results =
                    await this.runtime.databaseAdapter.searchKnowledge({
                        agentId: this.runtime.agentId,
                        embedding: embedding,
                        match_threshold: this.defaultRAGMatchThreshold,
                        match_count:
                            (params.limit || this.defaultRAGMatchCount) * 2,
                        searchText: processedQuery,
                    });

                // Enhanced reranking with sophisticated scoring
                const rerankedResults = results
                    .map((result) => {
                        let score = result.similarity;

                        // Check for direct query term matches
                        const queryTerms = this.getQueryTerms(processedQuery);

                        const matchingTerms = queryTerms.filter((term) =>
                            result.content.text.toLowerCase().includes(term)
                        );

                        // Boost score for direct term matches
                        if (matchingTerms.length > 0) {
                            score += 0.1 * matchingTerms.length;
                        }

                        // Check for proximity matches
                        if (this.hasProximityMatch(result.content.text, queryTerms)) {
                            score += 0.2;
                        }

                        // Check for metadata matches
                        if (result.content.metadata) {
                            const metadata = result.content.metadata as {
                                title?: string;
                                tags?: string[];
                            };
                            if (metadata.title && processedQuery.toLowerCase().includes(metadata.title.toLowerCase())) {
                                score += 0.3;
                            }
                            if (metadata.tags && Array.isArray(metadata.tags)) {
                                const matchingTags = metadata.tags.filter((tag: string) =>
                                    processedQuery.toLowerCase().includes(tag.toLowerCase())
                                );
                                if (matchingTags.length > 0) {
                                    score += 0.1 * matchingTags.length;
                                }
                            }
                        }

                        return {
                            ...result,
                            score,
                        };
                    })
                    .sort((a, b) => (b.score || 0) - (a.score || 0))
                    .slice(0, params.limit || this.defaultRAGMatchCount);

                return rerankedResults;
            } catch (error) {
                elizaLogger.error("Error in getKnowledge:", error);
                return [];
            }
        }

        // If no query, return all knowledge
        return this.runtime.databaseAdapter.getKnowledge({
            agentId: agentId,
            limit: params.limit,
        });
    }

    async createKnowledge(item: RAGKnowledgeItem): Promise<void> {
        await this.runtime.databaseAdapter.createKnowledge(item);
    }

    async searchKnowledge(params: {
        agentId: UUID;
        embedding: Float32Array | number[];
        match_threshold?: number;
        match_count?: number;
        searchText?: string;
    }): Promise<RAGKnowledgeItem[]> {
        const float32Embedding = Array.isArray(params.embedding)
            ? new Float32Array(params.embedding)
            : params.embedding;

        return this.runtime.databaseAdapter.searchKnowledge({
            agentId: params.agentId,
            embedding: float32Embedding,
            match_threshold: params.match_threshold || this.defaultRAGMatchThreshold,
            match_count: params.match_count || this.defaultRAGMatchCount,
            searchText: params.searchText,
        });
    }

    async removeKnowledge(id: UUID): Promise<void> {
        await this.runtime.databaseAdapter.removeKnowledge(id);
    }

    async clearKnowledge(shared?: boolean): Promise<void> {
        await this.runtime.databaseAdapter.clearKnowledge(this.runtime.agentId, shared);
    }

    async listAllKnowledge(agentId: UUID): Promise<RAGKnowledgeItem[]> {
        return this.runtime.databaseAdapter.getKnowledge({
            agentId: agentId,
        });
    }

    async processFile(file: {
        path: string;
        content: string;
        type: "pdf" | "md" | "txt";
        isShared: boolean;
    }): Promise<void> {
        const { path, content, type, isShared } = file;
        const id = this.generateScopedId(path, isShared);
        const embedding = await this.runtime.generateEmbedding(content);
        const embeddingInfo = {
            embedding_dim: embedding.length,
            embedding_type: type,
            embedding_version: '1.0',
            embedding_provider: this.runtime.embeddingProvider || 'default',
            embedding_model: this.runtime.embeddingModel || 'default',
            embedding_checksum: await this.generateEmbeddingChecksum(embedding)
        };

        await this.storeRAGEmbedding({
            id,
            content,
            embedding,
            metadata: {
                path,
                type,
                isShared,
                isMain: true
            },
            embeddingInfo
        });
    }

    async cleanupDeletedKnowledgeFiles(): Promise<void> {
        const rows = await this.runtime.databaseAdapter.db.all(
            `SELECT * FROM ${this.tableName} WHERE metadata LIKE '%"isMain":true%'`
        );

        for (const row of rows) {
            const metadata = JSON.parse(row.metadata);
            if (metadata.path && !existsSync(join(this.knowledgeRoot, metadata.path))) {
                await this.removeKnowledge(row.id);
            }
        }
    }

    generateScopedId(path: string, isShared: boolean): UUID {
        const scope = isShared ? KnowledgeScope.SHARED : KnowledgeScope.PRIVATE;
        return stringToUuid(`${scope}:${path}`);
    }

    async storeRAGEmbedding(params: {
        id: UUID;
        content: string;
        embedding: number[];
        metadata: any;
        embeddingInfo: {
            embedding_dim: number;
            embedding_type: string;
            embedding_version: string;
            embedding_provider: string;
            embedding_model: string;
            embedding_checksum: string;
        };
    }): Promise<void> {
        const { id, content, embedding, metadata, embeddingInfo } = params;
        await this.runtime.databaseAdapter.db.run(
            `INSERT OR REPLACE INTO ${this.tableName} (
                id, content, embedding, metadata, created_at, last_accessed, access_count,
                embedding_dim, embedding_type, embedding_version, embedding_provider,
                embedding_model, embedding_checksum
            ) VALUES (?, ?, ?, ?, datetime('now'), datetime('now'), 0, ?, ?, ?, ?, ?, ?)`,
            [
                id,
                content,
                JSON.stringify(embedding),
                JSON.stringify(metadata),
                embeddingInfo.embedding_dim,
                embeddingInfo.embedding_type,
                embeddingInfo.embedding_version,
                embeddingInfo.embedding_provider,
                embeddingInfo.embedding_model,
                embeddingInfo.embedding_checksum
            ]
        );
    }

    async getRAGEmbeddings(params: {
        agentId: UUID;
        limit?: number;
    }): Promise<RAGKnowledgeItem[]> {
        const { agentId, limit = 100 } = params;
        const rows = await this.runtime.databaseAdapter.db.all(
            `SELECT * FROM ${this.tableName} WHERE agentId = ? LIMIT ?`,
            [agentId, limit]
        );
        return rows.map(row => ({
            id: row.id,
            agentId: row.agentId,
            content: {
                text: row.content,
                metadata: JSON.parse(row.metadata)
            },
            embedding: new Float32Array(JSON.parse(row.embedding)),
            createdAt: new Date(row.created_at).getTime(),
            similarity: row.similarity,
            score: row.score
        }));
    }

    async searchRAG(params: {
        agentId: UUID;
        embedding: number[];
        match_threshold: number;
        match_count: number;
    }): Promise<RAGKnowledgeItem[]> {
        const { agentId, embedding, match_threshold, match_count } = params;
        const rows = await this.runtime.databaseAdapter.db.all(
            `SELECT *, 
            (SELECT COUNT(*) FROM ${this.tableName} WHERE agentId = ?) as total_count,
            (SELECT AVG(access_count) FROM ${this.tableName} WHERE agentId = ?) as avg_access_count
            FROM ${this.tableName} 
            WHERE agentId = ? 
            ORDER BY similarity DESC 
            LIMIT ?`,
            [agentId, agentId, agentId, match_count]
        );

        return rows.map(row => ({
            id: row.id,
            agentId: row.agentId,
            content: {
                text: row.content,
                metadata: JSON.parse(row.metadata)
            },
            embedding: new Float32Array(JSON.parse(row.embedding)),
            createdAt: new Date(row.created_at).getTime(),
            similarity: row.similarity,
            score: row.score
        }));
    }

    async checkRAGHealth(): Promise<boolean> {
        try {
            const result = await this.runtime.databaseAdapter.db.get(
                `SELECT COUNT(*) as count FROM ${this.tableName}`
            );
            return result.count > 0;
        } catch (error) {
            return false;
        }
    }

    async repairRAGEntries(): Promise<void> {
        const rows = await this.runtime.databaseAdapter.db.all(
            `SELECT * FROM ${this.tableName} WHERE embedding IS NULL OR content IS NULL`
        );

        for (const row of rows) {
            if (!row.embedding && row.content) {
                const embedding = await this.runtime.generateEmbedding(row.content);
                const embeddingInfo = {
                    embedding_dim: embedding.length,
                    embedding_type: 'text',
                    embedding_version: '1.0',
                    embedding_provider: this.runtime.embeddingProvider || 'default',
                    embedding_model: this.runtime.embeddingModel || 'default',
                    embedding_checksum: await this.generateEmbeddingChecksum(embedding)
                };

                await this.storeRAGEmbedding({
                    id: row.id,
                    content: row.content,
                    embedding,
                    metadata: JSON.parse(row.metadata || '{}'),
                    embeddingInfo
                });
            }
        }
    }

    private async generateEmbeddingChecksum(embedding: number[]): Promise<string> {
        const encoder = new TextEncoder();
        const data = encoder.encode(JSON.stringify(embedding));
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }
}