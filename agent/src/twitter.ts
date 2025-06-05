import { type IAgentRuntime, elizaLogger, type Memory, type State, stringToUuid, generateText, ModelClass, composeContext, AgentRuntime, embed } from '@elizaos/core';
import { Scraper, SearchMode, Tweet } from 'agent-twitter-client';
import { Database } from './database';

interface IRAG {
    search(query: string, limit: number): Promise<Array<{ content: string; metadata: any }>>;
    store(data: { id: string; content: string; metadata: any; embedding: Buffer }): Promise<void>;
}

export interface IAgentRuntimeWithRAG extends AgentRuntime {
    rag: IRAG;
    generateEmbedding(text: string): Promise<number[]>;
    cache: {
        get(key: string): Promise<any>;
        set(key: string, value: any): Promise<void>;
    };
}

interface TweetIntent {
    shouldGenerateImage: boolean;
    shouldReply: boolean;
    isImageEditRequest: boolean;
    prompt?: string;
    reasoning?: string;
}

export class TwitterIntegration {
    private scraper: Scraper;
    private runtime: IAgentRuntimeWithRAG;
    private isInitialized: boolean = false;
    private pollInterval: NodeJS.Timeout | null = null;
    private isProcessing: boolean = false;
    private isPosting: boolean = false;
    private conversationHistory: Map<string, Array<{ content: string; metadata: any }>> = new Map();
    private imageConversations: Map<string, { lastPrompt: string; lastImageUrl: string }> = new Map();
    private database: Database;
    private processedTweetIds: Set<string> = new Set();

    get initialized(): boolean {
        return this.isInitialized;
    }

    constructor(runtime: IAgentRuntimeWithRAG) {
        this.runtime = runtime;
        this.scraper = new Scraper();
        this.database = new Database(runtime);
    }

    private async isTweetProcessed(tweetId: string): Promise<boolean> {
        try {
            if (this.processedTweetIds.has(tweetId)) {
                return true;
            }
            return await this.database.isTweetReplied(tweetId);
        } catch (error) {
            elizaLogger.error(`Error checking if tweet ${tweetId} is processed: ${error.message}`, error);
            return false;
        }
    }

    private async markTweetAsProcessed(tweetId: string): Promise<void> {
        let retryCount = 0;
        const MAX_RETRIES = 3;
        while (retryCount < MAX_RETRIES) {
            try {
                await this.database.markTweetAsReplied(tweetId);
                this.processedTweetIds.add(tweetId);
                elizaLogger.info(`Marked tweet ${tweetId} as processed`);
                return;
            } catch (error) {
                retryCount++;
                elizaLogger.error(`Failed to mark tweet ${tweetId} as processed (attempt ${retryCount}/${MAX_RETRIES}): ${error.message}`, error);
                if (retryCount < MAX_RETRIES) {
                    const backoffTime = 1000 * Math.pow(2, retryCount);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                } else {
                    throw error;
                }
            }
        }
    }

    async initialize(): Promise<void> {
        if (this.isInitialized) return;

        try {
            await this.database.initialize();
            const recentTweets = await this.database.getRecentProcessedTweets(1000);
            recentTweets.forEach(tweetId => this.processedTweetIds.add(tweetId));
            elizaLogger.info(`Loaded ${recentTweets.length} recently processed tweets`);

            const requiredEnvVars = ['TWITTER_USERNAME', 'TWITTER_PASSWORD', 'TWITTER_EMAIL'];
            for (const envVar of requiredEnvVars) {
                if (!process.env[envVar]) throw new Error(`Missing required environment variable: ${envVar}`);
            }

            elizaLogger.info('Attempting to login to Twitter...');
            let loginSuccess = false;
            let retryCount = 0;
            const MAX_RETRIES = 3;
            const LOGIN_TIMEOUT = 15000;

            while (!loginSuccess && retryCount < MAX_RETRIES) {
                try {
                    const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error('Login timeout')), LOGIN_TIMEOUT));
                    await Promise.race([
                        this.scraper.login(
                            process.env.TWITTER_USERNAME!,
                            process.env.TWITTER_PASSWORD!,
                            process.env.TWITTER_EMAIL!,
                            process.env.TWITTER_2FA_SECRET
                        ),
                        timeoutPromise
                    ]);

                    if (!await this.scraper.isLoggedIn()) throw new Error('Login verification failed');
                    loginSuccess = true;
                    elizaLogger.info('Successfully logged in to Twitter');
                } catch (error) {
                    retryCount++;
                    elizaLogger.error(`Login attempt ${retryCount} failed: ${error.message}`, error);
                    if (retryCount < MAX_RETRIES) {
                        const backoffTime = 2000 * Math.pow(2, retryCount);
                        await new Promise(resolve => setTimeout(resolve, backoffTime));
                    } else {
                        throw new Error(`Failed to login after ${MAX_RETRIES} attempts: ${error.message}`);
                    }
                }
            }

            await this.startPolling();
            await this.startPeriodicPosting();
            this.isInitialized = true;
        } catch (error) {
            await this.handleError(error, 'initialization');
            throw error;
        }
    }

    private async handleError(error: any, operation: string): Promise<void> {
        elizaLogger.error(`Error during ${operation}: ${error.message}`, error);
    }

    private async startPolling(): Promise<void> {
        let lastCycleDuration = 60000;
        const MIN_INTERVAL = 60000;
        const MAX_INTERVAL = 180000;
        const CYCLE_TIMEOUT = 300000;

        const pollWithTimeout = async () => {
            const timeoutPromise = setTimeout(() => {
                elizaLogger.warn('Polling cycle timed out, forcing reset');
                this.isProcessing = false;
            }, CYCLE_TIMEOUT);

            const startTime = performance.now();
            try {
                await this.pollMentions();
            } catch (error) {
                elizaLogger.error('Polling cycle failed:', error);
            } finally {
                clearTimeout(timeoutPromise);
                this.isProcessing = false;
                const duration = performance.now() - startTime;
                lastCycleDuration = Math.min(Math.max(duration * 1.2, MIN_INTERVAL), MAX_INTERVAL);
                elizaLogger.info(`Polling cycle took ${duration.toFixed(2)}ms, next in ${lastCycleDuration}ms`);
            }
        };

        setTimeout(pollWithTimeout, 5000);

        const scheduleNextPoll = () => {
            this.pollInterval = setTimeout(async () => {
                if (!this.isProcessing) await pollWithTimeout();
                else elizaLogger.info('Skipping poll cycle - previous cycle in progress');
                scheduleNextPoll();
            }, lastCycleDuration);
        };

        elizaLogger.info(`Started polling with dynamic interval (~${lastCycleDuration}ms)`);
        scheduleNextPoll();
    }

    private async buildConversationContext(tweet: any, conversationId: string): Promise<string> {
        const startTime = performance.now();
        try {
            if (!tweet.inReplyToStatusId) {
                elizaLogger.info(`No reply context for tweet ${tweet.id}, using tweet text`);
                return `@${tweet.username}: ${tweet.text}`;
            }

            const historyQuery = `from:${tweet.username} conversation:${conversationId}`;
            let historyResults: Array<{ content: string; metadata: any }> = [];

            if (this.conversationHistory.has(historyQuery)) {
                elizaLogger.info(`Using cached history for query ${historyQuery}`);
                historyResults = this.conversationHistory.get(historyQuery)!;
            } else {
                try {
                    const searchStart = performance.now();
                    historyResults = await this.searchRAG(historyQuery, 3);
                    elizaLogger.info(`RAG search took ${(performance.now() - searchStart).toFixed(2)}ms`);
                    this.conversationHistory.set(historyQuery, historyResults);
                } catch (error) {
                    elizaLogger.warn(`RAG search failed for tweet ${tweet.id}: ${error.message}`);
                    historyResults = await this.database.getConversationHistory(conversationId, tweet.username);
                    this.conversationHistory.set(historyQuery, historyResults);
                }
            }

            const tweetText = `@${tweet.username}: ${tweet.text}`;
            try {
                await this.database.storeConversationHistory({
                    conversationId,
                    tweetId: tweet.id,
                    username: tweet.username,
                    content: tweetText,
                    timestamp: Date.now(),
                    type: 'tweet'
                });

                await this.storeInRAG(tweetText, {
                    conversationId,
                    tweetId: tweet.id,
                    username: tweet.username,
                    timestamp: Date.now(),
                    type: 'tweet'
                });
            } catch (error) {
                elizaLogger.error(`Failed to store conversation history for tweet ${tweet.id}: ${error.message}`);
            }

            if (tweet.inReplyToStatusId) {
                try {
                    const parentTweet = await this.scraper.getTweet(tweet.inReplyToStatusId);
                    if (parentTweet) {
                        const parentText = `@${parentTweet.username}: ${parentTweet.text}`;
                        await this.database.storeConversationHistory({
                            conversationId,
                            tweetId: parentTweet.id,
                            username: parentTweet.username,
                            content: parentText,
                            timestamp: Date.now(),
                            type: 'parent_tweet'
                        });
                        await this.storeInRAG(parentText, {
                            conversationId,
                            tweetId: parentTweet.id,
                            username: parentTweet.username,
                            timestamp: Date.now(),
                            type: 'parent_tweet'
                        });
                    }
                } catch (error) {
                    elizaLogger.error(`Failed to get parent tweet ${tweet.inReplyToStatusId}: ${error.message}`);
                }
            }

            const conversationContext = historyResults.length > 0
                ? historyResults.map(result => result.content).join('\n\n')
                : tweetText;
            elizaLogger.info(`Built conversation context for tweet ${tweet.id} in ${(performance.now() - startTime).toFixed(2)}ms`);
            return conversationContext;
        } catch (error) {
            elizaLogger.error(`Error building conversation context for tweet ${tweet.id}: ${error.message}`);
            return `@${tweet.username}: ${tweet.text}`;
        }
    }

    private async generateImage(prompt: string): Promise<Buffer> {
        let retryCount = 0;
        const MAX_RETRIES = 3;
        while (retryCount < MAX_RETRIES) {
            try {
                elizaLogger.info(`Generating image for prompt: ${prompt}`);
                const encodedPrompt = encodeURIComponent(prompt);
                const imageUrl = `https://image.pollinations.ai/prompt/${encodedPrompt}?width=2560&height=2049&nologo=true`;

                const response = await fetch(imageUrl);
                if (!response.ok) throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);

                const arrayBuffer = await response.arrayBuffer();
                const buffer = Buffer.from(arrayBuffer);
                elizaLogger.info(`Successfully generated image for prompt: ${prompt}`);
                return buffer;
            } catch (error) {
                retryCount++;
                elizaLogger.error(`Failed to generate image (attempt ${retryCount}/${MAX_RETRIES}): ${error.message}`);
                if (retryCount < MAX_RETRIES) {
                    const backoffTime = 2000 * Math.pow(2, retryCount);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                } else {
                    throw new Error(`Failed to generate image after ${MAX_RETRIES} attempts: ${error.message}`);
                }
            }
        }
        throw new Error('Failed to generate image');
    }

    private async postImageTweet(tweetId: string, imageBuffer: Buffer, caption: string): Promise<string> {
        let retryCount = 0;
        const MAX_RETRIES = 3;
        while (retryCount < MAX_RETRIES) {
            try {
                if (!(await this.scraper.isLoggedIn())) {
                    elizaLogger.warn('Scraper not authenticated, re-initializing...');
                    await this.initializeScraper();
                }

                const mediaData = [{ data: imageBuffer, mediaType: 'image/jpeg' }];
                const result = await this.scraper.sendTweet(caption, tweetId, mediaData);
                const body = await result.json();

                if (body.errors) {
                    throw new Error(`Twitter API error: ${JSON.stringify(body.errors)}`);
                }

                const imageUrl = body.data?.create_tweet?.tweet_results?.result?.legacy?.entities?.media?.[0]?.media_url_https;
                if (!imageUrl) throw new Error('No media URL in response');
                elizaLogger.info(`Posted image tweet for tweet ${tweetId}: ${imageUrl}`);
                return imageUrl;
            } catch (error) {
                retryCount++;
                elizaLogger.error(`Failed to post image tweet (attempt ${retryCount}/${MAX_RETRIES}): ${error.message}`);
                if (retryCount < MAX_RETRIES) {
                    const backoffTime = 2000 * Math.pow(2, retryCount);
                    elizaLogger.info(`Retrying in ${backoffTime}ms`);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                } else {
                    throw new Error(`Failed to post image tweet after ${MAX_RETRIES} attempts: ${error.message}`);
                }
            }
        }
        throw new Error('Failed to post image tweet');
    }

    private async handleImageRequest(tweet: any, prompt: string): Promise<void> {
        try {
            const cacheKey = `${tweet.conversationId}:${prompt}`;
            if (this.imageConversations.has(cacheKey)) {
                const cached = this.imageConversations.get(cacheKey)!;
                elizaLogger.info(`Using cached image for prompt: ${prompt}`);
                await this.postImageTweet(tweet.id, Buffer.from(cached.lastImageUrl, 'base64'), "Here's the image you requested");
                return;
            }

            const imageBuffer = await this.generateImage(prompt);
            if (!imageBuffer || imageBuffer.length === 0) throw new Error('Invalid or empty image buffer');

            const memory: Memory = {
                id: stringToUuid(Date.now().toString()),
                userId: stringToUuid(tweet.userId),
                agentId: this.runtime.agentId,
                roomId: stringToUuid(tweet.conversationId),
                content: { text: `Generated image for prompt: ${prompt}`, action: 'GENERATE_IMAGE' },
                createdAt: Date.now()
            };

            try {
                await this.database.storeConversationHistory({
                    conversationId: tweet.conversationId,
                    tweetId: tweet.id,
                    username: tweet.username,
                    content: `Generated image: ${prompt}`,
                    timestamp: Date.now(),
                    type: 'image_generation'
                });

                await this.storeInRAG(`Generated image: ${prompt}`, {
                    conversationId: tweet.conversationId,
                    tweetId: tweet.id,
                    username: tweet.username,
                    timestamp: Date.now(),
                    type: 'image_generation',
                    prompt
                });
            } catch (error) {
                elizaLogger.error(`Failed to store image history for tweet ${tweet.id}: ${error.message}`);
            }

            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid(tweet.conversationId),
                actors: this.runtime.character.name,
                recentMessages: `Generate an image for: ${prompt}`,
                recentMessagesData: [memory]
            };

            try {
                await this.runtime.processActions(memory, [], state);
            } catch (error) {
                elizaLogger.error(`Failed to process actions for image request: ${error.message}`);
                throw new Error(`Action processing failed: ${error.message}`);
            }

            const caption = "Here's the image you requested";
            const imageUrl = await this.postImageTweet(tweet.id, imageBuffer, caption);

            this.imageConversations.set(cacheKey, { lastPrompt: prompt, lastImageUrl: imageUrl });
            elizaLogger.info(`Successfully generated and shared image for prompt: ${prompt}`);
        } catch (error) {
            elizaLogger.error(`Error handling image request for tweet ${tweet.id}: ${error.message}`);
            throw error;
        }
    }

    private isImageModificationRequest(text: string): boolean {
        const modificationKeywords = [
            'change', 'modify', 'adjust', 'update', 'edit', 'make it', 'make this',
            'different', 'another', 'new', 'more', 'less', 'brighter', 'darker',
            'larger', 'smaller', 'wider', 'taller', 'add', 'remove', 'replace'
        ];
        return modificationKeywords.some(keyword => text.toLowerCase().includes(keyword.toLowerCase()));
    }

    private isInappropriateContent(text: string): boolean {
        const inappropriatePatterns = [
            /\bnsfw\b/i, /\bexplicit\b/i, /\bprofane\b/i, /\bblasphem(y|ous)\b/i,
            /\bfuck\b/i, /\bshit\b/i, /\bdamn\b/i, /\bhell\b/i, /\bbitch\b/i,
            /\bass\b/i, /\bporn\b/i, /\bsex\b/i, /\bnude\b/i, /\bdrug\b/i,
            /\balcohol\b/i, /\bdrunk\b/i, /\bhigh\b/i
        ];

        const whitelist = ['jesus', 'christ', 'god', 'holy', 'spirit', 'bibl', 'hello', 'help', 'healing', 'heaven'];
        const lowerText = text.toLowerCase();

        if (whitelist.some(term => lowerText.includes(term))) {
            elizaLogger.info(`Text whitelisted due to term: ${whitelist.find(term => lowerText.includes(term))}`);
            return false;
        }

        const isInappropriate = inappropriatePatterns.some(pattern => {
            const match = pattern.test(lowerText);
            if (match) elizaLogger.info(`Text flagged as inappropriate due to pattern: ${pattern.source}`);
            return match;
        });

        return isInappropriate;
    }

    private async analyzeTweetIntent(tweet: any, cleanText: string): Promise<TweetIntent> {
        try {
            if (this.isInappropriateContent(cleanText)) {
                elizaLogger.info(`Skipping inappropriate content in tweet ${tweet.id}`);
                return { shouldGenerateImage: false, shouldReply: false, isImageEditRequest: false };
            }

            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid(tweet.conversationId),
                actors: this.runtime.character.name,
                recentMessages: cleanText,
                recentMessagesData: []
            };

            const intentAnalysis = await generateText({
                runtime: this.runtime,
                context: composeContext({
                    state,
                    template: `Analyze this tweet and determine if it's requesting an image generation or just a reply. Consider:
1. Are there keywords like "generate", "create", "make", "draw", "image", "picture", "photo", "art", "illustration"?
2. Is it asking for a modification of a previous image?
3. Is it a general question or comment that needs a reply?

Tweet: "${cleanText}"

You must respond with ONLY a JSON object in this exact format, with no additional text:
{
    "shouldGenerateImage": true/false,
    "shouldReply": true/false,
    "isImageEditRequest": true/false,
    "reasoning": "brief explanation"
}`
                }),
                modelClass: ModelClass.SMALL
            });

            let intent;
            try {
                const jsonMatch = intentAnalysis.match(/\{[\s\S]*\}/);
                if (jsonMatch) {
                    intent = JSON.parse(jsonMatch[0]);
                } else {
                    const lowerText = intentAnalysis.toLowerCase();
                    intent = {
                        shouldGenerateImage: lowerText.includes('image') || lowerText.includes('generate') || lowerText.includes('create'),
                        shouldReply: true,
                        isImageEditRequest: lowerText.includes('edit') || lowerText.includes('modify') || lowerText.includes('change'),
                        reasoning: intentAnalysis
                    };
                }
            } catch (error) {
                elizaLogger.error(`Failed to parse intent analysis for tweet ${tweet.id}: ${error.message}`);
                const lowerText = cleanText.toLowerCase();
                intent = {
                    shouldGenerateImage: lowerText.includes('image') || lowerText.includes('generate') || lowerText.includes('create'),
                    shouldReply: true,
                    isImageEditRequest: lowerText.includes('edit') || lowerText.includes('modify') || lowerText.includes('change'),
                    reasoning: 'Fallback to keyword analysis'
                };
            }

            let prompt: string | undefined;
            if (intent.shouldGenerateImage) {
                let retryCount = 0;
                const MAX_RETRIES = 2;
                while (retryCount < MAX_RETRIES) {
                    try {
                        if (intent.isImageEditRequest) {
                            const imageContext = this.imageConversations.get(tweet.conversationId);
                            if (imageContext) {
                                prompt = await this.buildImagePrompt(tweet, cleanText, imageContext.lastPrompt);
                            }
                        } else {
                            prompt = await this.buildImagePrompt(tweet, cleanText);
                        }

                        if (!prompt) {
                            retryCount++;
                            if (retryCount === MAX_RETRIES) {
                                return { shouldGenerateImage: false, shouldReply: true, isImageEditRequest: false, reasoning: 'Failed to generate valid image prompt' };
                            }
                            continue;
                        }
                        break;
                    } catch (error) {
                        retryCount++;
                        elizaLogger.error(`Failed to build image prompt for tweet ${tweet.id} (attempt ${retryCount}/${MAX_RETRIES}): ${error.message}`);
                        if (retryCount === MAX_RETRIES) {
                            return { shouldGenerateImage: false, shouldReply: true, isImageEditRequest: false, reasoning: 'Failed to generate valid image prompt' };
                        }
                    }
                }
            }

            return { ...intent, prompt };
        } catch (error) {
            elizaLogger.error(`Error analyzing tweet intent for tweet ${tweet.id}: ${error.message}`);
            return { shouldGenerateImage: false, shouldReply: true, isImageEditRequest: false, reasoning: 'Error in intent analysis' };
        }
    }

    private async buildImagePrompt(tweet: any, text: string, previousPrompt?: string): Promise<string | null> {
        try {
            const historyQuery = `conversation:${tweet.conversationId} type:image_generation`;
            let historyResults: Array<{ content: string; metadata: any }> = [];
            try {
                historyResults = await this.searchRAG(historyQuery, 3);
            } catch (error) {
                elizaLogger.warn(`Failed to fetch RAG history for tweet ${tweet.id}: ${error.message}`);
            }

            const promptContext = `Generate a biblically appropriate image prompt based on this request: "${text}"
Consider:
1. The main subject or scene (must align with biblical values)
2. Style and mood (uplifting, reverent)
3. Colors and lighting (peaceful, warm)
4. Composition and framing (dignified, respectful)

${previousPrompt ? `Previous prompt: "${previousPrompt}"` : ''}
${historyResults.length > 0 ? `Recent image history:\n${historyResults.map(r => r.content).join('\n')}` : ''}

Return a clear, descriptive sentence (e.g., "A serene biblical scene of a shepherd tending sheep in a peaceful valley at sunset").`;

            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid(tweet.conversationId),
                actors: this.runtime.character.name,
                recentMessages: promptContext,
                recentMessagesData: []
            };

            const promptResult = await generateText({
                runtime: this.runtime,
                context: composeContext({ state, template: promptContext }),
                modelClass: ModelClass.SMALL
            });

            if (this.isInappropriateContent(promptResult)) {
                elizaLogger.warn(`Generated inappropriate prompt for tweet ${tweet.id}: ${promptResult}`);
                return null;
            }

            const finalPrompt = promptResult.trim();
            if (!finalPrompt) {
                elizaLogger.warn(`Empty prompt generated for tweet ${tweet.id}`);
                return "A serene biblical scene of a shepherd tending sheep in a peaceful valley at sunset";
            }

            elizaLogger.info(`Generated image prompt for tweet ${tweet.id}: ${finalPrompt}`);
            return finalPrompt;
        } catch (error) {
            elizaLogger.error(`Error building image prompt for tweet ${tweet.id}: ${error.message}`);
            return null;
        }
    }

    private async testSearchFunctionality(): Promise<void> {
        try {
            const username = process.env.TWITTER_USERNAME!;
            elizaLogger.info('Testing search functionality...');

            elizaLogger.info('Test 1: Searching for mentions...');
            const mentionsQuery = `@${username}`;
            const mentionsResults = await this.scraper.searchTweets(mentionsQuery, 5, SearchMode.Latest);
            let mentionCount = 0;
            for await (const tweet of mentionsResults) {
                mentionCount++;
                elizaLogger.info(`Found mention ${mentionCount}: ${tweet.text}`);
            }
            elizaLogger.info(`Found ${mentionCount} mentions`);

            elizaLogger.info("Test 2: Searching for bot's own tweets...");
            const ownTweetsQuery = `from:${username}`;
            const ownTweetsResults = await this.scraper.searchTweets(ownTweetsQuery, 5, SearchMode.Latest);
            let ownTweetCount = 0;
            for await (const tweet of ownTweetsResults) {
                ownTweetCount++;
                elizaLogger.info(`Found own tweet ${ownTweetCount}: ${tweet.text}`);
            }
            elizaLogger.info(`Found ${ownTweetCount} own tweets`);

            elizaLogger.info('Test 3: Searching for any tweet...');
            const anyTweetQuery = 'twitter';
            const anyTweetResults = await this.scraper.searchTweets(anyTweetQuery, 5, SearchMode.Latest);
            let anyTweetCount = 0;
            for await (const tweet of anyTweetResults) {
                anyTweetCount++;
                elizaLogger.info(`Found any tweet ${anyTweetCount}: ${tweet.text}`);
            }
            elizaLogger.info(`Found ${anyTweetCount} any tweets`);
        } catch (error) {
            elizaLogger.error(`Error in testSearchFunctionality: ${error.message}`);
        }
    }

    private async pollMentions(): Promise<void> {
        const startTime = performance.now();
        try {
            if (this.isProcessing) {
                elizaLogger.info('Previous polling cycle still in progress, skipping...');
                return;
            }

            this.isProcessing = true;
            elizaLogger.info('Starting new polling cycle...');

            const username = process.env.TWITTER_USERNAME!.toLowerCase();
            if (!username) throw new Error('TWITTER_USERNAME environment variable is not set');

            if (!(await this.scraper.isLoggedIn())) {
                elizaLogger.warn('Scraper not authenticated, re-initializing...');
                await this.initializeScraper();
            }

            const tweetBatch: any[] = [];
            const searchQuery = `@${username}`;
            let retryCount = 0;
            const MAX_RETRIES = 3;

            while (retryCount < MAX_RETRIES) {
                try {
                    const searchStart = performance.now();
                    const mentionsGenerator = this.scraper.searchTweets(searchQuery, 10, SearchMode.Latest);
                    const tweets = [];
                    const timeoutPromise = new Promise((_, reject) => 
                        setTimeout(() => reject(new Error('Search timeout')), 60000)
                    );

                    await Promise.race([
                        (async () => {
                            for await (const tweet of mentionsGenerator) {
                                tweets.push(tweet);
                                if (tweets.length >= 10) break;
                            }
                        })(),
                        timeoutPromise
                    ]);

                    tweetBatch.push(...tweets);
                    elizaLogger.info(`Search query "${searchQuery}" took ${(performance.now() - searchStart).toFixed(2)}ms, fetched ${tweets.length} tweets`);
                    break;
                } catch (error) {
                    retryCount++;
                    elizaLogger.error(`Search attempt ${retryCount} failed: ${error.message}`);
                    if (retryCount < MAX_RETRIES) {
                        const backoffTime = 2000 * Math.pow(2, retryCount);
                        elizaLogger.info(`Retrying search in ${backoffTime}ms`);
                        await new Promise(resolve => setTimeout(resolve, backoffTime));
                    } else {
                        elizaLogger.error(`Failed to fetch mentions after ${MAX_RETRIES} attempts: ${error.message}`);
                        break;
                    }
                }
            }

            let mentionsCount = 0;
            let hasResults = tweetBatch.length > 0;
            let lastTweetId: string | null = null;

            const processTweet = async (tweet: any) => {
                const tweetStart = performance.now();
                elizaLogger.info(`Processing mention: ${tweet.id} from @${tweet.username}: ${tweet.text}`);

                try {
                    if (this.processedTweetIds.has(tweet.id)) {
                        elizaLogger.info(`Skipping duplicate tweet ${tweet.id} (in-memory cache)`);
                        return;
                    }

                    if (tweet.username.toLowerCase() === username) {
                        elizaLogger.info(`Skipping bot's own tweet: ${tweet.id}`);
                        this.processedTweetIds.add(tweet.id);
                        return;
                    }

                    if (tweet.isRetweet) {
                        elizaLogger.info(`Skipping retweet: ${tweet.id}`);
                        this.processedTweetIds.add(tweet.id);
                        return;
                    }

                    if (!tweet.text.toLowerCase().includes(`@${username}`)) {
                        elizaLogger.info(`Skipping tweet ${tweet.id} - does not contain @${username}`);
                        this.processedTweetIds.add(tweet.id);
                        return;
                    }

                    lastTweetId = tweet.id;
                    const isProcessed = await this.isTweetProcessed(tweet.id);
                    if (isProcessed) {
                        const hasReply = await this.verifyTweetReplied(tweet.id, username);
                        if (!hasReply) {
                            elizaLogger.warn(`Tweet ${tweet.id} marked as processed but no reply found, reprocessing`);
                        } else {
                            elizaLogger.info(`Skipping already processed tweet ${tweet.id}`);
                            this.processedTweetIds.add(tweet.id);
                            return;
                        }
                    }

                    this.processedTweetIds.add(tweet.id);
                    const cleanText = tweet.text.replace(/@\w+/g, '').trim();
                    if (!cleanText) {
                        elizaLogger.info(`Skipping empty tweet ${tweet.id}`);
                        await this.markTweetAsProcessed(tweet.id);
                        return;
                    }

                    if (this.isInappropriateContent(cleanText)) {
                        elizaLogger.info(`Skipping inappropriate content in tweet ${tweet.id}`);
                        await this.markTweetAsProcessed(tweet.id);
                        return;
                    }

                    const intent = await this.analyzeTweetIntent(tweet, cleanText);
                    let success = false;

                    if (intent.shouldGenerateImage && intent.prompt) {
                        try {
                            await this.handleImageRequest(tweet, intent.prompt);
                            success = true;
                        } catch (error) {
                            elizaLogger.error(`Failed to generate image for tweet ${tweet.id}: ${error.message}`);
                            const fallbackResponse = await this.generateResponse(tweet, cleanText, { ...intent, shouldGenerateImage: false });
                            if (fallbackResponse) {
                                await this.replyToTweet(tweet.id, fallbackResponse);
                                success = true;
                            }
                        }
                    } else {
                        const response = await this.generateResponse(tweet, cleanText, intent);
                        if (response) {
                            await this.replyToTweet(tweet.id, response);
                            success = true;
                        }
                    }

                    if (success) {
                        await this.markTweetAsProcessed(tweet.id);
                        this.conversationHistory.delete(`from:${tweet.username} conversation:${tweet.conversationId}`);
                        mentionsCount++;
                    } else {
                        elizaLogger.warn(`Failed to process tweet ${tweet.id}, not marking as processed`);
                        this.processedTweetIds.delete(tweet.id); // Remove from cache to allow retry
                    }

                    elizaLogger.info(`Processed tweet ${tweet.id} in ${(performance.now() - tweetStart).toFixed(2)}ms`);
                } catch (error) {
                    elizaLogger.error(`Error processing tweet ${tweet.id}: ${error.message}`);
                }
            };

            const concurrencyLimit = 3;
            for (let i = 0; i < tweetBatch.length; i += concurrencyLimit) {
                const batch = tweetBatch.slice(i, i + concurrencyLimit);
                await Promise.all(batch.map(processTweet));
            }

            if (!hasResults) {
                elizaLogger.info('No mentions found in this polling cycle');
            } else {
                elizaLogger.info(`Processed ${mentionsCount} mentions, last tweet ID: ${lastTweetId}`);
            }
        } catch (error) {
            elizaLogger.error(`Error in pollMentions: ${error.message}`);
            if (error.message?.includes('Unauthorized')) {
                elizaLogger.error('Authentication error, re-initializing scraper...');
                await this.initializeScraper();
            } else if (error.message?.includes('429')) {
                const backoffTime = 15000;
                elizaLogger.info(`Rate limited, waiting ${backoffTime}ms`);
                await new Promise(resolve => setTimeout(resolve, backoffTime));
            }
        } finally {
            this.isProcessing = false;
            elizaLogger.info(`Polling cycle completed in ${(performance.now() - startTime).toFixed(2)}ms`);
        }
    }

    private async verifyTweetReplied(tweetId: string, username: string): Promise<boolean> {
        const MAX_RETRIES = 3;
        let retryCount = 0;
        while (retryCount < MAX_RETRIES) {
            try {
                const botTweets = await this.scraper.searchTweets(`from:${username} is:reply`, 50, SearchMode.Latest);
                const tweets = [];
                for await (const tweet of botTweets) {
                    tweets.push(tweet);
                    if (tweets.length >= 50) break;
                }
                for (const botTweet of tweets) {
                    if (botTweet.inReplyToStatusId === tweetId) {
                        elizaLogger.info(`Found reply to tweet ${tweetId}: ${botTweet.id}`);
                        return true;
                    }
                }
                return false;
            } catch (error) {
                retryCount++;
                elizaLogger.error(`Error verifying reply for tweet ${tweetId} (attempt ${retryCount}/${MAX_RETRIES}): ${error.message}`);
                if (retryCount < MAX_RETRIES) {
                    const backoffTime = 2000 * Math.pow(2, retryCount);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                } else {
                    return false;
                }
            }
        }
        return false;
    }

    async stop(): Promise<void> {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        await this.database.cleanup();
    }

    async replyToTweet(tweetId: string, content: string): Promise<void> {
        try {
            if (!(await this.scraper.isLoggedIn())) {
                elizaLogger.warn('Scraper not authenticated, re-initializing...');
                await this.initializeScraper();
            }

            const tweet = await this.scraper.getTweet(tweetId);
            if (!tweet) {
                throw new Error(`Failed to get tweet ${tweetId}`);
            }

            const replyContent = `@${tweet.username} ${content}`;
            const result = await this.scraper.sendTweet(replyContent, tweetId);
            
            if (!result.ok) {
                const errorText = await result.text();
                throw new Error(`Failed to send tweet: ${errorText}`);
            }

            elizaLogger.info(`Posted reply to tweet ${tweetId}: ${replyContent}`);
        } catch (error) {
            elizaLogger.error(`Error posting reply to ${tweetId}: ${error.message}`);
            throw error;
        }
    }

    async postTweet(content: string): Promise<void> {
        try {
            if (!(await this.scraper.isLoggedIn())) {
                elizaLogger.warn('Scraper not authenticated, re-initializing...');
                await this.initializeScraper();
            }

            const result = await this.scraper.sendTweet(content);
            
            if (!result.ok) {
                const errorText = await result.text();
                throw new Error(`Failed to send tweet: ${errorText}`);
            }

            elizaLogger.info(`Successfully posted tweet: ${content}`);
        } catch (error) {
            elizaLogger.error(`Error posting tweet: ${error.message}`);
            throw error;
        }
    }

    async searchTweets(query: string) {
        try {
            if (!(await this.scraper.isLoggedIn())) {
                elizaLogger.warn('Scraper not authenticated, re-initializing...');
                await this.initializeScraper();
            }
            return await this.scraper.searchTweets(query, 10, SearchMode.Latest);
        } catch (error) {
            elizaLogger.error(`Error searching tweets for query "${query}": ${error.message}`);
            throw error;
        }
    }

    async getTimeline() {
        try {
            if (!(await this.scraper.isLoggedIn())) {
                elizaLogger.warn('Scraper not authenticated, re-initializing...');
                await this.initializeScraper();
            }
            return await this.scraper.searchTweets(`from:${process.env.TWITTER_USERNAME!}`, 10, SearchMode.Latest);
        } catch (error) {
            elizaLogger.error(`Error fetching timeline: ${error.message}`);
            throw error;
        }
    }

    private async startPeriodicPosting(): Promise<void> {
        const POST_INTERVAL = 900000;
        const MAX_RETRIES = 3;

        elizaLogger.info(`Starting periodic posting every ${POST_INTERVAL / 1000} seconds`);

        const generateAndPostTweet = async () => {
            try {
                if (this.isPosting) {
                    elizaLogger.info('Previous periodic posting in progress, skipping');
                    return;
                }

                this.isPosting = true;
                const topics = this.runtime.character.topics;
                const style = this.runtime.character.style.post;
                const adjectives = this.runtime.character.adjectives;

                const randomTopic = topics[Math.floor(Math.random() * topics.length)];
                const randomStyle = style[Math.floor(Math.random() * style.length)];
                const randomAdjective = adjectives[Math.floor(Math.random() * adjectives.length)];

                const state: State = {
                    bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                    lore: this.runtime.character.lore.join(' '),
                    messageDirections: this.runtime.character.style.post.join(' '),
                    postDirections: this.runtime.character.style.post.join(' '),
                    roomId: this.runtime.agentId,
                    actors: this.runtime.character.name,
                    recentMessages: `Generate a tweet about ${randomTopic}`,
                    recentMessagesData: []
                };

                const context = await composeContext({
                    state,
                    template: `You are ${this.runtime.character.name}. Generate a tweet that is ${randomAdjective} about ${randomTopic}}. The tweet should ${randomStyle}}. Keep it under 280 characters. No hashtags or emojis.`
                });

                const tweetContent = await generateText({
                    runtime: this.runtime,
                    context,
                    modelClass: ModelClass.SMALL
                });

                let retryCount = 0;
                while (retryCount < MAX_RETRIES) {
                    try {
                        await this.postTweet(tweetContent);
                        elizaLogger.info('Successfully posted periodic tweet');
                        break;
                    } catch (error) {
                        retryCount++;
                        elizaLogger.error(`Failed to post periodic tweet (${retryCount}/${MAX_RETRIES}): ${error.message}`);
                        if (retryCount < MAX_RETRIES) {
                            const backoffTime = 2000 * Math.pow(2, retryCount);
                            elizaLogger.info(`Retrying in ${backoffTime}ms`);
                            await new Promise(resolve => setTimeout(resolve, backoffTime));
                        } else {
                            elizaLogger.error('Failed to post periodic tweet after retries');
                        }
                    }
                }
            } catch (error) {
                elizaLogger.error(`Error in periodic posting: ${error.message}`);
            } finally {
                this.isPosting = false;
            }
        };

        try {
            setTimeout(generateAndPostTweet, 10000);

            const scheduleNextPost = () => {
                setTimeout(async () => {
                    try {
                        await generateAndPostTweet();
                    } catch (error) {
                        elizaLogger.error(`Error in scheduled periodic post: ${error.message}`);
                    } finally {
                        scheduleNextPost();
                    }
                }, POST_INTERVAL);
            };

            scheduleNextPost();
        } catch (error) {
            elizaLogger.error(`Failed to start periodic posting: ${error.message}`);
        }
    }

    private async generateResponse(tweet: Tweet, cleanText: string, intent: TweetIntent): Promise<string | null> {
        try {
            if (intent.shouldGenerateImage) {
                await this.handleImageRequest(tweet, intent.prompt!);
                return null;
            } else if (intent.shouldReply) {
                const conversationId = tweet.inReplyToStatusId || tweet.id;
                const context = await this.buildConversationContext(tweet, conversationId);

                const state: State = {
                    bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                    lore: this.runtime.character.lore.join(' '),
                    messageDirections: this.runtime.character.style.post.join(' '),
                    postDirections: this.runtime.character.style.post.join(' '),
                    roomId: stringToUuid(conversationId),
                    actors: this.runtime.character.name,
                    recentMessages: context,
                    recentMessagesData: []
                };

                return await generateText({
                    runtime: this.runtime,
                    context: composeContext({
                        state,
                        template: `You are ${this.runtime.character.name}. Respond to this tweet from @${tweet.username}: "${cleanText}" using one of these formats:
1. A relevant Scripture quote
2. A paraphrased biblical truth
3. A Christlike question
4. A short modern parable

Previous conversation context:
${context}

Keep the response under 280 characters. No hashtags or emojis. Ensure the reply maintains conversational flow and is relevant to the user's tweet and their previous interactions.`
                    }),
                    modelClass: ModelClass.SMALL
                });
            }
            return null;
        } catch (error) {
            elizaLogger.error(`Error generating response for tweet ${tweet.id}: ${error.message}`);
            return null;
        }
    }

    private async initializeScraper(): Promise<void> {
        try {
            const username = process.env.TWITTER_USERNAME!;
            const password = process.env.TWITTER_PASSWORD!;
            const email = process.env.TWITTER_EMAIL!;
            this.scraper = new Scraper();
            await this.scraper.login(username, password, email);
            elizaLogger.info('Successfully re-initialized Twitter scraper');
        } catch (error) {
            elizaLogger.error(`Failed to re-initialize Twitter scraper: ${error.message}`);
            throw error;
        }
    }

    private async storeInRAG(content: string, metadata: any): Promise<void> {
        try {
            elizaLogger.info(`[RAG] Generating embedding for content: ${content.slice(0, 100)}...`);
            elizaLogger.info(`[RAG] Content length: ${content.length} characters`);
            elizaLogger.info(`[RAG] Metadata: ${JSON.stringify(metadata)}`);
            
            const startTime = performance.now();
            const embedding = await embed(this.runtime, content);
            const duration = performance.now() - startTime;
            
            elizaLogger.info(`[RAG] Embedding generation took ${duration.toFixed(2)}ms`);
            elizaLogger.info(`[RAG] Generated embedding: ${embedding ? `Array(${embedding.length})` : 'null'}`);
            
            if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
                throw new Error('Failed to generate valid embedding');
            }

            elizaLogger.info(`[RAG] Converting embedding to buffer...`);
            const embeddingBuffer = Buffer.from(embedding);
            elizaLogger.info(`[RAG] Buffer size: ${embeddingBuffer.length} bytes`);
            elizaLogger.info(`[RAG] First few values: ${embedding.slice(0, 5).join(', ')}...`);

            const id = stringToUuid(Date.now().toString());
            elizaLogger.info(`[RAG] Storing in RAG with ID: ${id}`);
            
            await this.runtime.rag.store({
                id,
                content,
                metadata,
                embedding: embeddingBuffer
            });
            
            elizaLogger.info(`[RAG] Successfully stored in RAG: ${content.slice(0, 50)}...`);
        } catch (error) {
            elizaLogger.error(`[RAG] Failed to store in RAG:`, {
                error: error.message,
                stack: error.stack,
                code: error.code,
                name: error.name,
                content: content.slice(0, 100),
                metadata: JSON.stringify(metadata)
            });
            
            // Store in conversation history as fallback
            await this.database.storeConversationHistory({
                conversationId: metadata.conversationId,
                tweetId: metadata.tweetId,
                username: metadata.username || 'unknown',
                content,
                timestamp: metadata.timestamp,
                type: metadata.type
            });
        }
    }

    private async searchRAG(query: string, limit: number): Promise<Array<{ content: string; metadata: any }>> {
        let retryCount = 0;
        const MAX_RETRIES = 3;
        while (retryCount < MAX_RETRIES) {
            try {
                const results = await this.runtime.rag.search(query, limit);
                elizaLogger.info(`RAG search successful for query: ${query}`);
                return results;
            } catch (error) {
                retryCount++;
                elizaLogger.error(`RAG search failed for query "${query}" (attempt ${retryCount}/${MAX_RETRIES}): ${error.message}`);
                if (retryCount < MAX_RETRIES) {
                    const backoffTime = 1000 * Math.pow(2, retryCount);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                } else {
                    const conversationId = query.includes('conversation:') ? query.replace(/.*conversation:([^ ]+).*/, '$1') : '';
                    const username = query.includes('from:') ? query.replace(/from:([^ ]+).*/, '$1') : '';
                    return await this.database.getConversationHistory(conversationId, username);
                }
            }
        }
        return [];
    }
}