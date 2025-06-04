import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class Database {
    private db: any;
    private dbPath: string;

    constructor(dbPath?: string) {
        this.dbPath = dbPath || path.join(__dirname, '../data/tweets.db');
    }

    async initialize() {
        this.db = await open({
            filename: this.dbPath,
            driver: sqlite3.Database
        });

        await this.db.exec(`
            CREATE TABLE IF NOT EXISTS replied_tweets (
                tweet_id TEXT PRIMARY KEY,
                replied_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT,
                created_at INTEGER NOT NULL,
                replied_at INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_tweet_id ON conversations(tweet_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
        `);
    }

    async runTransaction<T>(callback: () => Promise<T>): Promise<T> {
        try {
            await this.db.run('BEGIN TRANSACTION');
            const result = await callback();
            await this.db.run('COMMIT');
            return result;
        } catch (error) {
            await this.db.run('ROLLBACK');
            throw error;
        }
    }

    async isTweetReplied(tweetId: string): Promise<boolean> {
        const result = await this.db.get(
            'SELECT tweet_id FROM replied_tweets WHERE tweet_id = ?',
            [tweetId]
        );
        return !!result;
    }

    async markTweetAsReplied(tweetId: string): Promise<void> {
        await this.db.run(
            'INSERT OR REPLACE INTO replied_tweets (tweet_id, replied_at, created_at) VALUES (?, ?, ?)',
            [tweetId, Date.now(), Date.now()]
        );
    }

    async cleanupOldTweets(maxAge: number = 30 * 24 * 60 * 60 * 1000): Promise<void> {
        const cutoffTime = Date.now() - maxAge;
        await this.db.run(
            'DELETE FROM replied_tweets WHERE created_at < ?',
            [cutoffTime]
        );
    }

    async close(): Promise<void> {
        if (this.db) await this.db.close();
    }

    async run(sql: string, params?: any[]): Promise<any> {
        return params ? this.db.run(sql, params) : this.db.run(sql);
    }

    async all(sql: string, params?: any[]): Promise<any[]> {
        return params ? this.db.all(sql, params) : this.db.all(sql);
    }

    async exec(sql: string): Promise<any> {
        return this.db.exec(sql);
    }

    async storeConversation(tweetId: string, userId: string, username: string, message: string, response: string | null = null): Promise<void> {
        const now = Date.now();
        await this.db.run(
            `INSERT INTO conversations (tweet_id, user_id, username, message, response, created_at, replied_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)`,
            [tweetId, userId, username, message, response, now, response ? now : null]
        );
    }

    async getConversationHistory(tweetId: string, limit: number = 10): Promise<Array<{
        message: string;
        response: string | null;
        created_at: number;
    }>> {
        return await this.db.all(
            `SELECT message, response, created_at 
             FROM conversations 
             WHERE tweet_id = ? 
             ORDER BY created_at DESC 
             LIMIT ?`,
            [tweetId, limit]
        );
    }

    async updateConversationResponse(tweetId: string, response: string): Promise<void> {
        const now = Date.now();
        await this.db.run(
            `UPDATE conversations 
             SET response = ?, replied_at = ? 
             WHERE tweet_id = ? AND response IS NULL`,
            [response, now, tweetId]
        );
    }
}