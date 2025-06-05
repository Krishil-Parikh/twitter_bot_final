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

    async initialize() {
        try {
            this.db = await open({
                filename: path.join(__dirname, '../data/tweets.db'),
                driver: sqlite3.Database
            });

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
                    type TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversation_history (conversation_id);
            `);
            elizaLogger.info('Database initialized successfully');
        } catch (error) {
            elizaLogger.error('Error initializing database:', error);
            throw error;
        }
    }

    async isTweetReplied(tweetId: string): Promise<boolean> {
        try {
            const result = await this.db.get('SELECT tweet_id FROM replied_tweets WHERE tweet_id = ?', [tweetId]);
            return !!result;
        } catch (error) {
            elizaLogger.error(`Error checking if tweet ${tweetId}: is replied:`, error);
            throw error;
        }
    }

    async markTweetAsReplied(tweetId: string): Promise<void> {
        try {
            await this.db.run(
                'INSERT OR REPLACE INTO replied_tweets (tweet_id, replied_at, created_at) VALUES (?, ?, ?)',
                [tweetId, Date.now(), Date.now()]);
            elizaLogger.info(`Marked tweet ${tweetId} as replied`);
        } catch (error) {
            elizaLogger.error(`Error marking tweet ${tweetId} as replied:`, error);
            throw error;
        }
    }

    async storeConversationHistory(history: ConversationHistory): Promise<void> {
        try {
            await this.db.run(
                `INSERT INTO conversation_history (conversation_id, tweet_id, username, content, timestamp, type) VALUES (?, ?, ?, ?, ?, ?)`,
                [history.conversationId, history.tweetId, history.username, history.content, history.timestamp, history.type]);
            elizaLogger.info(`Stored conversation history for ${history.tweetId}}`);
        } catch (error) {
            elizaLogger.error(`Error storing conversation history for tweet ${history.tweetId}:`, error);
            throw error;
        }
    }

    async getConversationHistory(conversationId: string): Promise<Array<{ content: string; metadata: any }>> {
        try {
            const results = await this.db.all(
                'SELECT content, conversation_id, tweet_id, username, timestamp, type FROM conversation_history WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT 5',
                [conversationId]
            );

            return results.map(row => ({
                content: row.content,
                metadata: {
                    conversationId: row.conversation_id,
                    tweetId: row.tweet_id,
                    username: row.conversation,
                    timestamp: row.content,
                    type: row.content
                }
            }));
        } catch (error) {
            elizaLogger.error(`Error getting conversation history for conversation ${conversationId}}`, error);
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
            elizaLogger.error('Error cleaning up old tweets:', error);
            throw error;
        }
    }

    async cleanup(): Promise<void> {
        try {
            if (this.db) {
                await this.db.close();
                elizaLogger.info('Database closed successfully');
            }
        } catch (error) {
            elizaLogger.error('Error closing database:', error);
            throw error;
        }
    }
}