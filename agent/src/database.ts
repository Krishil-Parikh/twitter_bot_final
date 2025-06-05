import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import path from 'path';
import { fileURLToPath } from 'url';
import { elizaLogger } from '@elizaos/core';
import { IAgentRuntimeWithRAG } from './twitter';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface ConversationHistory {
    conversationId: string;
    tweetId: string;
    username: string;
    content: string;
    timestamp: number;
    type: 'tweet' | 'parent_tweet' | 'image_generation';
}

export class Database {
    private db: any;
    private isInitialized: boolean = false;
    private runtime: IAgentRuntimeWithRAG;

    constructor(runtime: IAgentRuntimeWithRAG) {
        this.runtime = runtime;
    }

    async initialize() {
        if (this.isInitialized) return;

        try {
            this.db = await open({
                filename: path.join(__dirname, '../data/tweets.db'),
                driver: sqlite3.Database,
                mode: sqlite3.OPEN_READWRITE | sqlite3.OPEN_CREATE
            });

            // Enable Write-Ahead Logging for better concurrency
            await this.db.run('PRAGMA journal_mode = WAL');
            await this.db.run('PRAGMA busy_timeout = 5000');

            await this.db.exec(`
                CREATE TABLE IF NOT EXISTS replied_tweets (
                    tweet_id TEXT PRIMARY KEY,
                    replied_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    tweet_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    UNIQUE(conversation_id, tweet_id, type)
                );

                CREATE TABLE IF NOT EXISTS rag_embeddings (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
                    last_accessed INTEGER DEFAULT (strftime('%s', 'now') * 1000),
                    access_count INTEGER DEFAULT 0,
                    embedding_dim INTEGER NOT NULL,
                    embedding_type TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    embedding_provider TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_checksum TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversation_history (conversation_id);
                CREATE INDEX IF NOT EXISTS idx_tweet_id ON conversation_history (tweet_id);
                CREATE INDEX IF NOT EXISTS idx_username ON conversation_history (username);
                CREATE INDEX IF NOT EXISTS idx_rag_created_at ON rag_embeddings (created_at);
                CREATE INDEX IF NOT EXISTS idx_rag_embeddings_created_at ON rag_embeddings(created_at);
                CREATE INDEX IF NOT EXISTS idx_rag_embeddings_last_accessed ON rag_embeddings(last_accessed);
                CREATE INDEX IF NOT EXISTS idx_rag_embeddings_access_count ON rag_embeddings(access_count);
            `);

            // Migrate any existing binary embeddings to JSON format
            await this.migrateEmbeddingsToJSON();

            await this.checkHealth();
            await this.maintainDatabase();
            this.isInitialized = true;
            elizaLogger.info('Database initialized successfully');
        } catch (error) {
            elizaLogger.error(`Error initializing database: ${error.message}`, error);
            throw error;
        }
    }

    private async migrateEmbeddingsToJSON(): Promise<void> {
        try {
            elizaLogger.info('[DB] Starting embedding migration to JSON format...');
            
            // Get all entries with binary embeddings
            const entries = await this.db.all('SELECT id, embedding FROM rag_embeddings WHERE embedding NOT LIKE "[%"');
            
            if (entries.length === 0) {
                elizaLogger.info('[DB] No binary embeddings found to migrate');
                return;
            }

            elizaLogger.info(`[DB] Found ${entries.length} binary embeddings to migrate`);

            for (const entry of entries) {
                try {
                    // Try to convert binary data to Float32Array
                    let embedding: number[];
                    try {
                        const buffer = Buffer.from(entry.embedding);
                        const float32Array = new Float32Array(buffer.buffer);
                        embedding = Array.from(float32Array);
                    } catch (error) {
                        elizaLogger.error(`[DB] Failed to convert binary embedding for id ${entry.id}: ${error.message}`);
                        continue;
                    }

                    // Convert to JSON string
                    const embeddingStr = JSON.stringify(embedding);

                    // Update the entry
                    await this.db.run(
                        'UPDATE rag_embeddings SET embedding = ? WHERE id = ?',
                        [embeddingStr, entry.id]
                    );

                    elizaLogger.info(`[DB] Successfully migrated embedding for id ${entry.id}`);
                } catch (error) {
                    elizaLogger.error(`[DB] Error migrating embedding for id ${entry.id}: ${error.message}`);
                }
            }

            elizaLogger.info('[DB] Completed embedding migration');
        } catch (error) {
            elizaLogger.error(`[DB] Error during embedding migration: ${error.message}`);
        }
    }

    private async withTransaction<T>(operation: string, callback: () => Promise<T>): Promise<T> {
        let retryCount = 0;
        const MAX_RETRIES = 5;
        const MAX_BACKOFF = 5000; // 5 seconds

        while (retryCount < MAX_RETRIES) {
            try {
                await this.db.run('BEGIN TRANSACTION');
                const result = await callback();
                await this.db.run('COMMIT');
                return result;
            } catch (error) {
                await this.db.run('ROLLBACK');
                retryCount++;

                if (error.code === 'SQLITE_BUSY' && retryCount < MAX_RETRIES) {
                    const backoffTime = Math.min(100 * Math.pow(2, retryCount), MAX_BACKOFF);
                    elizaLogger.warn(`Database busy during ${operation}, retrying in ${backoffTime}ms`);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                } else {
                    elizaLogger.error(`Error in ${operation}: ${error.message}`, error);
                    throw error;
                }
            }
        }
        throw new Error(`Failed ${operation} after ${MAX_RETRIES} retries`);
    }

    async checkHealth(): Promise<void> {
        try {
            // Check database integrity
            const integrityCheck = await this.db.get('PRAGMA integrity_check');
            if (integrityCheck.integrity_check !== 'ok') {
                elizaLogger.error('Database integrity check failed:', integrityCheck);
                await this.repairDatabase();
                return;
            }

            // Check if tables exist and have correct structure
            const tables = await this.db.all("SELECT name FROM sqlite_master WHERE type='table'");
            const requiredTables = ['replied_tweets', 'conversation_history', 'rag_embeddings'];
            const missingTables = requiredTables.filter(table => 
                !tables.some(t => t.name === table)
            );

            if (missingTables.length > 0) {
                elizaLogger.error('Missing required tables:', missingTables);
                await this.repairDatabase();
                return;
            }

            // Check indexes
            const indexes = await this.db.all("SELECT name FROM sqlite_master WHERE type='index'");
            const requiredIndexes = [
                'idx_conversation_id',
                'idx_tweet_id',
                'idx_username',
                'idx_rag_created_at'
            ];
            const missingIndexes = requiredIndexes.filter(index =>
                !indexes.some(i => i.name === index)
            );

            if (missingIndexes.length > 0) {
                elizaLogger.error('Missing required indexes:', missingIndexes);
                await this.repairDatabase();
                return;
            }

            // Check for data consistency
            const stats = await this.db.all(`
                SELECT 
                    (SELECT COUNT(*) FROM replied_tweets) as replied_count,
                    (SELECT COUNT(*) FROM conversation_history) as history_count,
                    (SELECT COUNT(*) FROM rag_embeddings) as rag_count
            `);
            elizaLogger.info('Database stats:', stats[0]);

            elizaLogger.info('Database health check passed');
        } catch (error) {
            elizaLogger.error('Database health check failed:', error);
            await this.repairDatabase();
        }
    }

    private async repairDatabase(): Promise<void> {
        try {
            elizaLogger.info('Starting database repair...');

            // Backup existing data
            const backupTables = ['replied_tweets', 'conversation_history', 'rag_embeddings'];
            for (const table of backupTables) {
                try {
                    await this.db.run(`CREATE TABLE IF NOT EXISTS ${table}_backup AS SELECT * FROM ${table}`);
                    elizaLogger.info(`Backed up ${table} table`);
                } catch (error) {
                    elizaLogger.warn(`Could not backup ${table} table:`, error);
                }
            }

            // Drop and recreate tables
            await this.db.exec(`
                DROP TABLE IF EXISTS replied_tweets;
                DROP TABLE IF EXISTS conversation_history;
                DROP TABLE IF EXISTS rag_embeddings;

                CREATE TABLE replied_tweets (
                    tweet_id TEXT PRIMARY KEY,
                    replied_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    tweet_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    UNIQUE(conversation_id, tweet_id, type)
                );

                CREATE TABLE rag_embeddings (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
                    last_accessed INTEGER DEFAULT (strftime('%s', 'now') * 1000),
                    access_count INTEGER DEFAULT 0,
                    embedding_dim INTEGER NOT NULL,
                    embedding_type TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    embedding_provider TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_checksum TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversation_history (conversation_id);
                CREATE INDEX IF NOT EXISTS idx_tweet_id ON conversation_history (tweet_id);
                CREATE INDEX IF NOT EXISTS idx_username ON conversation_history (username);
                CREATE INDEX IF NOT EXISTS idx_rag_created_at ON rag_embeddings (created_at);
                CREATE INDEX IF NOT EXISTS idx_rag_embeddings_created_at ON rag_embeddings(created_at);
                CREATE INDEX IF NOT EXISTS idx_rag_embeddings_last_accessed ON rag_embeddings(last_accessed);
                CREATE INDEX IF NOT EXISTS idx_rag_embeddings_access_count ON rag_embeddings(access_count);
            `);

            // Restore data from backups
            for (const table of backupTables) {
                try {
                    await this.db.run(`INSERT INTO ${table} SELECT * FROM ${table}_backup`);
                    await this.db.run(`DROP TABLE ${table}_backup`);
                    elizaLogger.info(`Restored ${table} table`);
                } catch (error) {
                    elizaLogger.warn(`Could not restore ${table} table:`, error);
                }
            }

            // Optimize database
            await this.db.run('PRAGMA optimize');
            await this.db.run('VACUUM');
            await this.db.run('REINDEX');

            elizaLogger.info('Database repair completed');
        } catch (error) {
            elizaLogger.error('Database repair failed:', error);
            throw error;
        }
    }

    async isTweetReplied(tweetId: string): Promise<boolean> {
        try {
            const result = await this.db.get('SELECT tweet_id FROM replied_tweets WHERE tweet_id = ?', [tweetId]);
            return !!result;
        } catch (error) {
            elizaLogger.error(`Error checking if tweet ${tweetId} is replied: ${error.message}`);
            return false;
        }
    }

    async markTweetAsReplied(tweetId: string): Promise<void> {
        try {
            const isReplied = await this.isTweetReplied(tweetId);
            if (isReplied) {
                elizaLogger.info(`Tweet ${tweetId} already marked as replied`);
                return;
            }
            await this.db.run(
                'INSERT OR IGNORE INTO replied_tweets (tweet_id, replied_at, created_at) VALUES (?, ?, ?)',
                [tweetId, Date.now(), Date.now()]
            );
            elizaLogger.info(`Marked tweet ${tweetId} as replied`);
        } catch (error) {
            elizaLogger.error(`Error marking tweet ${tweetId} as replied: ${error.message}`);
            throw error;
        }
    }

    async storeConversationHistory(history: ConversationHistory): Promise<void> {
        try {
            const existing = await this.db.get(
                'SELECT id FROM conversation_history WHERE conversation_id = ? AND tweet_id = ? AND type = ?',
                [history.conversationId, history.tweetId, history.type]
            );

            if (existing) {
                elizaLogger.info(`Conversation history entry already exists for tweet ${history.tweetId}`);
                return;
            }

            await this.db.run(
                `INSERT INTO conversation_history (conversation_id, tweet_id, username, content, timestamp, type) VALUES (?, ?, ?, ?, ?, ?)`,
                [history.conversationId, history.tweetId, history.username, history.content, history.timestamp, history.type]
            );
            elizaLogger.info(`Stored conversation history for ${history.tweetId}`);
        } catch (error) {
            elizaLogger.error(`Error storing conversation history for tweet ${history.tweetId}: ${error.message}`);
        }
    }

    async storeRAGEmbedding(
        id: string, 
        content: string, 
        embedding: number[], 
        metadata: any,
        embeddingInfo: {
            dim: number;
            type: string;
            version: string;
            provider: string;
            model: string;
            checksum: string;
        }
    ): Promise<void> {
        return this.withTransaction('storeRAGEmbedding', async () => {
            try {
                // Input validation
                if (!id || !content || !embedding || !metadata || !embeddingInfo) {
                    throw new Error('Missing required fields for RAG embedding storage');
                }

                if (embedding.length === 0) {
                    throw new Error('Empty embedding array');
                }

                // Validate embedding values
                const hasNaN = embedding.some(val => isNaN(val));
                const hasInf = embedding.some(val => !isFinite(val));
                if (hasNaN || hasInf) {
                    throw new Error(`Invalid embedding values detected: ${hasNaN ? 'NaN' : ''} ${hasInf ? 'Infinity' : ''}`);
                }

                // Store embedding as JSON string
                const embeddingStr = JSON.stringify(embedding);

                // Validate metadata format
                const metadataStr = JSON.stringify(metadata);
                if (!metadataStr) {
                    throw new Error('Invalid metadata format');
                }

                elizaLogger.info(`[DB] Storing RAG embedding - ID: ${id}, Content length: ${content.length}, Embedding size: ${embedding.length} values`);

                // Direct insert with error handling
                try {
                    await this.db.run(
                        `INSERT OR REPLACE INTO rag_embeddings (
                            id, content, embedding, metadata, created_at, last_accessed,
                            access_count, embedding_dim, embedding_type, embedding_version,
                            embedding_provider, embedding_model, embedding_checksum
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                        [
                            id, 
                            content, 
                            embeddingStr, 
                            metadataStr, 
                            Date.now(),
                            Date.now(),
                            0, // initial access_count
                            embedding.length,
                            embeddingInfo.type,
                            embeddingInfo.version,
                            embeddingInfo.provider,
                            embeddingInfo.model,
                            embeddingInfo.checksum
                        ]
                    );
                    elizaLogger.info(`[DB] Successfully stored RAG embedding for ${id}`);
                } catch (dbError) {
                    elizaLogger.error(`[DB] Failed to store RAG embedding: ${dbError.message}`, dbError);
                    throw dbError;
                }

                // Verify storage
                const stored = await this.db.get('SELECT id, length(embedding) as embedding_length FROM rag_embeddings WHERE id = ?', [id]);
                if (!stored) {
                    throw new Error('Failed to verify RAG embedding storage');
                }
                if (stored.embedding_length === 0) {
                    throw new Error('Stored embedding is empty');
                }
            } catch (error) {
                elizaLogger.error(`[DB] Error storing RAG embedding: ${error.message}`, error);
                throw error;
            }
        });
    }

    async getRAGEmbeddings(limit: number = 20): Promise<Array<{ content: string; metadata: any; embedding: Buffer }>> {
        return this.withTransaction('getRAGEmbeddings', async () => {
            try {
                const results = await this.db.all(
                    'SELECT id, content, metadata, embedding FROM rag_embeddings ORDER BY created_at DESC LIMIT ?',
                    [limit]
                );
                
                return results.map(row => {
                    try {
                        const metadata = JSON.parse(row.metadata);
                        if (!row.embedding || row.embedding.length === 0) {
                            elizaLogger.warn(`Found RAG entry with empty embedding: ${row.id}`);
                        }
                        return {
                            content: row.content,
                            metadata,
                            embedding: row.embedding
                        };
                    } catch (error) {
                        elizaLogger.error(`Error parsing RAG entry metadata: ${error.message}`);
                        return null;
                    }
                }).filter(Boolean);
            } catch (error) {
                elizaLogger.error(`Error getting RAG embeddings: ${error.message}`, error);
                return [];
            }
        });
    }

    async getConversationHistory(conversationId: string, username: string = ''): Promise<Array<{ content: string; metadata: any }>> {
        try {
            let query = `
                SELECT ch.content, ch.conversation_id, ch.tweet_id, ch.username, ch.timestamp, ch.type,
                       re.embedding, re.metadata as rag_metadata
                FROM conversation_history ch
                LEFT JOIN rag_embeddings re ON ch.tweet_id = re.id
                WHERE ch.conversation_id = ?
            `;
            const params = [conversationId];

            if (username) {
                query += ' OR ch.username = ?';
                params.push(username);
            }

            query += ' ORDER BY ch.timestamp DESC LIMIT 5';

            const results = await this.db.all(query, params);

            return results.map(row => ({
                content: row.content,
                metadata: {
                    conversationId: row.conversation_id,
                    tweetId: row.tweet_id,
                    username: row.username,
                    timestamp: row.timestamp,
                    type: row.type,
                    embedding: row.embedding,
                    ragMetadata: row.rag_metadata ? JSON.parse(row.rag_metadata) : null
                }
            }));
        } catch (error) {
            elizaLogger.error(`Error getting conversation history: ${error.message}`);
            return [];
        }
    }

    async getRecentProcessedTweets(limit: number): Promise<string[]> {
        try {
            const results = await this.db.all(
                'SELECT tweet_id FROM replied_tweets ORDER BY replied_at DESC LIMIT ?',
                [limit]
            );
            return results.map(row => row.tweet_id);
        } catch (error) {
            elizaLogger.error(`Error fetching recent processed tweets: ${error.message}`, error);
            return [];
        }
    }

    async maintainDatabase(): Promise<void> {
        try {
            // Analyze tables for optimization
            await this.db.run('ANALYZE');
            
            // Update statistics
            await this.db.run('PRAGMA optimize');
            
            // Vacuum to reclaim space and defragment
            await this.db.run('VACUUM');
            
            // Rebuild indexes
            await this.db.run('REINDEX');
            
            // Check for and fix any inconsistencies
            await this.db.run('PRAGMA foreign_key_check');
            await this.db.run('PRAGMA integrity_check');
            
            elizaLogger.info('Database maintenance completed');
        } catch (error) {
            elizaLogger.error('Database maintenance failed:', error);
            throw error;
        }
    }

    async cleanupOldTweets(maxAge: number = 30 * 24 * 60 * 60 * 1000): Promise<void> {
        try {
            const cutoffTime = Date.now() - maxAge;
            await this.db.run('DELETE FROM replied_tweets WHERE created_at < ?', [cutoffTime]);
            await this.db.run('DELETE FROM conversation_history WHERE timestamp < ?', [cutoffTime]);
            elizaLogger.info('Cleaned up old tweets and conversation history');
        } catch (error) {
            elizaLogger.error(`Error cleaning up old tweets: ${error.message}`, error);
            throw error;
        }
    }

    async cleanup(): Promise<void> {
        try {
            if (this.db) {
                await this.db.close();
                this.isInitialized = false;
                elizaLogger.info('Database closed successfully');
            }
        } catch (error) {
            elizaLogger.error(`Error closing database: ${error.message}`, error);
            throw error;
        }
    }

    async checkRAGHealth(): Promise<{ total: number; valid: number; invalid: number }> {
        try {
            const stats = await this.db.all(`
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN length(embedding) > 0 THEN 1 ELSE 0 END) as valid,
                    SUM(CASE WHEN length(embedding) = 0 THEN 1 ELSE 0 END) as invalid
                FROM rag_embeddings
            `);
            
            const result = stats[0];
            elizaLogger.info('RAG health check:', result);
            return result;
        } catch (error) {
            elizaLogger.error(`Error checking RAG health: ${error.message}`, error);
            return { total: 0, valid: 0, invalid: 0 };
        }
    }

    async repairRAGEntries(): Promise<void> {
        try {
            elizaLogger.info('[DB] Starting RAG entries repair...');
            
            // Find entries with empty embeddings or invalid JSON
            const invalidEntries = await this.db.all(
                'SELECT id, content, metadata FROM rag_embeddings WHERE length(embedding) = 0 OR embedding IS NULL OR embedding NOT LIKE "[%"'
            );
            
            if (invalidEntries.length > 0) {
                elizaLogger.warn(`[DB] Found ${invalidEntries.length} invalid RAG entries`);
                
                // Store invalid entries in conversation history as fallback
                for (const entry of invalidEntries) {
                    try {
                        const metadata = JSON.parse(entry.metadata);
                        await this.storeConversationHistory({
                            conversationId: metadata.conversationId,
                            tweetId: metadata.tweetId,
                            username: metadata.username || 'unknown',
                            content: entry.content,
                            timestamp: metadata.timestamp,
                            type: metadata.type
                        });
                        elizaLogger.info(`[DB] Stored invalid entry ${entry.id} in conversation history`);
                    } catch (error) {
                        elizaLogger.error(`[DB] Error storing invalid RAG entry in conversation history: ${error.message}`);
                    }
                }
                
                // Delete invalid entries
                await this.db.run('DELETE FROM rag_embeddings WHERE length(embedding) = 0 OR embedding IS NULL OR embedding NOT LIKE "[%"');
                elizaLogger.info(`[DB] Deleted ${invalidEntries.length} invalid RAG entries`);
            }

            // Verify remaining entries
            const remainingEntries = await this.db.all(
                'SELECT id, length(embedding) as embedding_length FROM rag_embeddings'
            );
            
            const validEntries = remainingEntries.filter(entry => entry.embedding_length > 0);
            elizaLogger.info(`[DB] Verified ${validEntries.length} valid RAG entries`);
            
            if (validEntries.length !== remainingEntries.length) {
                elizaLogger.warn(`[DB] Found ${remainingEntries.length - validEntries.length} entries with invalid embeddings`);
            }
        } catch (error) {
            elizaLogger.error(`[DB] Error repairing RAG entries: ${error.message}`, error);
        }
    }

    async searchRAG(query: string, limit: number, username?: string): Promise<Array<{ content: string; metadata: any }>> {
        try {
            // First repair any invalid entries
            await this.repairRAGEntries();

            // Extract username from query if not provided
            if (!username) {
                username = query.split('from:')[1]?.split(' ')[0] || '';
            }
            elizaLogger.info(`[RAG] Searching for user: ${username}`);

            // First try to get embeddings for the query with detailed logging
            let queryEmbedding: number[];
            let retryCount = 0;
            const MAX_RETRIES = 3;

            while (retryCount < MAX_RETRIES) {
                try {
                    elizaLogger.info(`[RAG] Attempting to generate embedding (attempt ${retryCount + 1}/${MAX_RETRIES})`);
                    queryEmbedding = await this.runtime.generateEmbedding(query);
                    
                    if (!queryEmbedding) {
                        throw new Error('Embedding generation returned null');
                    }
                    
                    if (queryEmbedding.length === 0) {
                        throw new Error('Embedding generation returned empty array');
                    }

                    // Validate embedding values
                    const hasNaN = queryEmbedding.some(val => isNaN(val));
                    const hasInf = queryEmbedding.some(val => !isFinite(val));
                    
                    if (hasNaN || hasInf) {
                        throw new Error(`Invalid embedding values detected: ${hasNaN ? 'NaN' : ''} ${hasInf ? 'Infinity' : ''}`);
                    }

                    elizaLogger.info(`[RAG] Successfully generated embedding with dimension: ${queryEmbedding.length}`);
                    elizaLogger.info(`[RAG] First few values: ${queryEmbedding.slice(0, 5).join(', ')}...`);
                    break; // Success, exit retry loop
                } catch (embedError) {
                    retryCount++;
                    elizaLogger.error(`[RAG] Embedding generation failed (attempt ${retryCount}/${MAX_RETRIES}): ${embedError.message}`);
                    
                    if (retryCount === MAX_RETRIES) {
                        elizaLogger.error('[RAG] All embedding generation attempts failed');
                        return await this.getConversationHistory(query.split('conversation:')[1]?.split(' ')[0] || '', username);
                    }
                    
                    // Wait before retrying
                    await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
                }
            }

            // Get all embeddings from the database with detailed logging
            let results;
            try {
                elizaLogger.info('[RAG] Querying database for embeddings...');
                
                // First check if we have any valid embeddings for this user
                const countQuery = username ? 
                    'SELECT COUNT(*) as count FROM rag_embeddings WHERE length(embedding) > 0 AND embedding LIKE "[%" AND json_extract(metadata, "$.username") = ?' :
                    'SELECT COUNT(*) as count FROM rag_embeddings WHERE length(embedding) > 0 AND embedding LIKE "[%"';
                
                const countResult = await this.db.get(countQuery, username ? [username] : []);
                elizaLogger.info(`[RAG] Total valid embeddings for user ${username}: ${countResult.count}`);

                if (countResult.count === 0) {
                    elizaLogger.warn(`[RAG] No valid embeddings found for user ${username}, searching all embeddings`);
                    // If no results for specific user, search all embeddings
                    results = await this.db.all(`
                        SELECT id, content, metadata, embedding, embedding_dim
                        FROM rag_embeddings
                        WHERE length(embedding) > 0 AND embedding LIKE "[%"
                        ORDER BY created_at DESC
                        LIMIT ?
                    `, [limit * 2]);
                } else {
                    // Get embeddings for specific user
                    results = await this.db.all(`
                        SELECT id, content, metadata, embedding, embedding_dim
                        FROM rag_embeddings
                        WHERE length(embedding) > 0 
                        AND embedding LIKE "[%"
                        AND json_extract(metadata, "$.username") = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    `, [username, limit * 2]);
                }
                
                elizaLogger.info(`[RAG] Retrieved ${results.length} embeddings from database`);
                
                if (results.length === 0) {
                    elizaLogger.warn('[RAG] No results found after filtering');
                    return await this.getConversationHistory(query.split('conversation:')[1]?.split(' ')[0] || '', username);
                }

                // Log first result for debugging
                const firstResult = results[0];
                elizaLogger.info(`[RAG] First result - ID: ${firstResult.id}, Content length: ${firstResult.content.length}, Embedding length: ${firstResult.embedding.length}`);
                elizaLogger.info(`[RAG] First result embedding preview: ${firstResult.embedding.substring(0, 100)}...`);
            } catch (dbError) {
                elizaLogger.error(`[RAG] Database error retrieving embeddings: ${dbError.message}`);
                elizaLogger.error(`[RAG] SQLite error code: ${dbError.code}`);
                elizaLogger.error(`[RAG] SQLite error details:`, dbError);
                throw dbError;
            }

            // Calculate cosine similarity for each result with detailed error handling
            const scoredResults = results.map(row => {
                try {
                    // Validate row data
                    if (!row.embedding || !row.content || !row.metadata) {
                        elizaLogger.warn(`[RAG] Invalid row data for id ${row.id}: missing required fields`);
                        return null;
                    }

                    // Parse stored embedding from JSON string
                    let storedEmbedding: number[];
                    try {
                        storedEmbedding = JSON.parse(row.embedding);
                        
                        if (storedEmbedding.length !== queryEmbedding.length) {
                            elizaLogger.warn(`[RAG] Dimension mismatch for id ${row.id}: expected ${queryEmbedding.length}, got ${storedEmbedding.length}`);
                            return null;
                        }
                    } catch (vectorError) {
                        elizaLogger.error(`[RAG] Error parsing embedding for id ${row.id}: ${vectorError.message}`);
                        elizaLogger.error(`[RAG] Raw embedding data: ${row.embedding.substring(0, 100)}...`);
                        return null;
                    }
                    
                    // Calculate cosine similarity
                    let dotProduct = 0;
                    let queryMagnitude = 0;
                    let storedMagnitude = 0;
                    
                    for (let i = 0; i < queryEmbedding.length; i++) {
                        dotProduct += queryEmbedding[i] * storedEmbedding[i];
                        queryMagnitude += queryEmbedding[i] * queryEmbedding[i];
                        storedMagnitude += storedEmbedding[i] * storedEmbedding[i];
                    }
                    
                    const similarity = dotProduct / (Math.sqrt(queryMagnitude) * Math.sqrt(storedMagnitude));
                    
                    // Parse metadata with error handling
                    let parsedMetadata;
                    try {
                        parsedMetadata = JSON.parse(row.metadata);
                    } catch (parseError) {
                        elizaLogger.error(`[RAG] Error parsing metadata for id ${row.id}: ${parseError.message}`);
                        parsedMetadata = {};
                    }
                    
                    return {
                        content: row.content,
                        metadata: parsedMetadata,
                        similarity
                    };
                } catch (error) {
                    elizaLogger.error(`[RAG] Error processing embedding for ${row.id}: ${error.message}`);
                    return null;
                }
            }).filter(Boolean);

            if (scoredResults.length === 0) {
                elizaLogger.warn('[RAG] No valid results after similarity calculation');
                return await this.getConversationHistory(query.split('conversation:')[1]?.split(' ')[0] || '', username);
            }

            // Sort by similarity and take top results
            const sortedResults = scoredResults
                .sort((a, b) => b.similarity - a.similarity)
                .slice(0, limit)
                .map(({ content, metadata }) => ({ content, metadata }));

            elizaLogger.info(`[RAG] Returning ${sortedResults.length} results with similarity scores`);
            return sortedResults;

        } catch (error) {
            elizaLogger.error(`[RAG] Error in searchRAG: ${error.message}`, error);
            // Fallback to conversation history on error
            return await this.getConversationHistory(query.split('conversation:')[1]?.split(' ')[0] || '', username);
        }
    }
}