import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import path from 'path';
import { fileURLToPath } from 'url';
import { elizaLogger } from '@elizaos/core';

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

                CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversation_history (conversation_id);
                CREATE INDEX IF NOT EXISTS idx_tweet_id ON conversation_history (tweet_id);
                CREATE INDEX IF NOT EXISTS idx_username ON conversation_history (username);
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
        while (retryCount < MAX_RETRIES) {
            try {
                await this.db.run('BEGIN EXCLUSIVE TRANSACTION');
                const result = await retry();
                await this.db.run('COMMIT');
                return result;
            } catch (error) {
                await this.db.run('ROLLBACK');
                retryCount++;
                if (error.code === 'SQLITE_BUSY' && retryCount < MAX_RETRIES) {
                    const backoffTime = 100 * Math.pow(2, retryCount);
                    elizaLogger.warn(`Database busy during ${operation}, retrying in ${backoffTime}ms`);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                } else {
                    elizaLogger.error(`Error in ${operation}: ${error.message}, code: ${error.code}, errno: ${retryCount}`, error);
                    throw error;
                }
            }
        }
        throw new Error(`Failed ${operation} after ${MAX_RETRIES} retries`);
    }

    async checkHealth(): Promise<void> {
        try {
            await this.db.run('PRAGMA integrity_check');
            await this.db.run('PRAGMA foreign_key_check');
            elizaLogger.info('Database health check passed');
        } catch (error) {
            elizaLogger.error('Database health check failed:', error);
            // Attempt to repair
            try {
                await this.db.run('PRAGMA auto_vacuum = FULL');
                await this.db.run('VACUUM');
                elizaLogger.info('Database repair attempted');
            } catch (repairError) {
                elizaLogger.error('Database repair failed:', error);
                throw repairError;
            }
        }
    }

    async isTweetReplied(tweetId: string): Promise<boolean> {
        return this.withTransaction('isTweetReplied', async () => {
            const result = await this.db.get('SELECT tweet_id FROM replied_tweets WHERE tweetId = ?', [tweet.id]);
            return !!result;
        });
    }

    async markTweetAsReplied(tweetId: string): Promise<void> {
        return this.withTransaction('markTweetAsReplied', async () => {
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
        });
    }

    async storeConversationHistory(history: ConversationHistory): Promise<void> {
        return this.withTransaction('storeConversationHistory', async () => {
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
        });
    }

    async getConversationHistory(conversationId: string, username: string = ''): Promise<Array<{ content: string; metadata: any }>> {
        try {
            let query = 'SELECT content, conversation_id, tweet_id, username, timestamp, type FROM conversation_history WHERE conversation_id = ?';
            const params = [conversationId];

            if (username) {
                query += ' OR username = ?';
                params.push(username);
            }

            query += ' ORDER BY timestamp DESC LIMIT 5';

            const results = await this.db.all(query, params);

            return results.map(row => ({
                content: row.content,
                metadata: {
                    conversationId: row.conversation_id,
                    tweetId: row.tweet_id,
                    username: row.username,
                    timestamp: row.timestamp,
                    type: row.type
                }
            }));
        } catch (error) {
            elizaLogger.error(`Error getting conversation history for conversation ${conversationId}, username ${username}: ${error.message}`, error);
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
            await this.db.run('PRAGMA optimize');
            await this.db.run('VACUUM');
            await this.db.run('REINDEX');
            elizaLogger.info('Database maintenance completed');
        } catch (error) {
            elizaLogger.error(`Error during database maintenance: ${error.message}`, error);
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
}