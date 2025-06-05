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
                    embedding BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversation_history (conversation_id);
                CREATE INDEX IF NOT EXISTS idx_tweet_id ON conversation_history (tweet_id);
                CREATE INDEX IF NOT EXISTS idx_username ON conversation_history (username);
                CREATE INDEX IF NOT EXISTS idx_rag_created_at ON rag_embeddings (created_at);
            `);

            await this.checkHealth();
            await this.maintainDatabase();
            this.isInitialized = true;
            elizaLogger.info('Database initialized successfully');
        } catch (error) {
            elizaLogger.error(`Error initializing database: ${error.message}`, error);
            throw error;
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
                    embedding BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversation_history (conversation_id);
                CREATE INDEX IF NOT EXISTS idx_tweet_id ON conversation_history (tweet_id);
                CREATE INDEX IF NOT EXISTS idx_username ON conversation_history (username);
                CREATE INDEX IF NOT EXISTS idx_rag_created_at ON rag_embeddings (created_at);
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

    async storeRAGEmbedding(id: string, content: string, embedding: Buffer, metadata: any): Promise<void> {
        return this.withTransaction('storeRAGEmbedding', async () => {
            try {
                // Input validation
                if (!id || !content || !embedding || !metadata) {
                    throw new Error('Missing required fields for RAG embedding storage');
                }

                if (embedding.length === 0) {
                    throw new Error('Empty embedding buffer');
                }

                // Validate metadata format
                const metadataStr = JSON.stringify(metadata);
                if (!metadataStr) {
                    throw new Error('Invalid metadata format');
                }

                elizaLogger.info(`[DB] Storing RAG embedding - ID: ${id}, Content length: ${content.length}, Embedding size: ${embedding.length} bytes`);

                // Direct insert with error handling
                try {
                    await this.db.run(
                        'INSERT OR REPLACE INTO rag_embeddings (id, content, embedding, metadata, created_at) VALUES (?, ?, ?, ?, ?)',
                        [id, content, embedding, metadataStr, Date.now()]
                    );
                    elizaLogger.info(`[DB] Successfully stored RAG embedding for ${id}`);
                } catch (dbError) {
                    elizaLogger.error(`[DB] Failed to store RAG embedding: ${dbError.message}`, dbError);
                    throw dbError;
                }

                // Verify storage
                const stored = await this.db.get('SELECT id FROM rag_embeddings WHERE id = ?', [id]);
                if (!stored) {
                    throw new Error('Failed to verify RAG embedding storage');
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
            // Find entries with empty embeddings
            const invalidEntries = await this.db.all(
                'SELECT id, content, metadata FROM rag_embeddings WHERE length(embedding) = 0'
            );
            
            if (invalidEntries.length > 0) {
                elizaLogger.warn(`Found ${invalidEntries.length} invalid RAG entries`);
                
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
                    } catch (error) {
                        elizaLogger.error(`Error storing invalid RAG entry in conversation history: ${error.message}`);
                    }
                }
                
                // Delete invalid entries
                await this.db.run('DELETE FROM rag_embeddings WHERE length(embedding) = 0');
                elizaLogger.info(`Deleted ${invalidEntries.length} invalid RAG entries`);
            }
        } catch (error) {
            elizaLogger.error(`Error repairing RAG entries: ${error.message}`, error);
        }
    }

    async searchRAG(query: string, limit: number): Promise<Array<{ content: string; metadata: any }>> {
        return this.withTransaction('searchRAG', async () => {
            try {
                // First try to get embeddings for the query
                const queryEmbedding = await this.runtime.generateEmbedding(query);
                if (!queryEmbedding || queryEmbedding.length === 0) {
                    elizaLogger.warn('Failed to generate query embedding, falling back to conversation history');
                    return await this.getConversationHistory(query.split('conversation:')[1]?.split(' ')[0] || '', query.split('from:')[1]?.split(' ')[0] || '');
                }

                // Use vector similarity search if available
                const results = await this.db.all(`
                    SELECT content, metadata, embedding
                    FROM rag_embeddings
                    WHERE length(embedding) > 0
                    ORDER BY created_at DESC
                    LIMIT ?
                `, [limit]);

                if (results.length === 0) {
                    elizaLogger.warn('No RAG results found, falling back to conversation history');
                    return await this.getConversationHistory(query.split('conversation:')[1]?.split(' ')[0] || '', query.split('from:')[1]?.split(' ')[0] || '');
                }

                return results.map(row => ({
                    content: row.content,
                    metadata: JSON.parse(row.metadata)
                }));
            } catch (error) {
                elizaLogger.error(`Error in searchRAG: ${error.message}`, error);
                // Fallback to conversation history on error
                return await this.getConversationHistory(query.split('conversation:')[1]?.split(' ')[0] || '', query.split('from:')[1]?.split(' ')[0] || '');
            }
        });
    }
}