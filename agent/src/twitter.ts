import { type IAgentRuntime, elizaLogger, type Memory, type State, stringToUuid, generateText, ModelClass, composeContext, AgentRuntime } from '@elizaos/core';
import { Scraper, SearchMode, Tweet } from 'agent-twitter-client';
import { Database } from './database';

interface IRAG {
    search(query: string, limit: number): Promise<Array<{ content: string; metadata: any }>>;
    store(data: { id: string; content: string; metadata: any }): Promise<void>;
}

export interface IAgentRuntimeWithRAG extends AgentRuntime {
    rag: IRAG;
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
    private conversationHistory: Map<string, string[]> = new Map();
    private imageConversations: Map<string, { lastPrompt: string, lastImageUrl: string }> = new Map();
    private database: Database;
    private isLoaded: boolean = false;

    // Add public getter for initialization status
    get initialized(): boolean {
        return this.isInitialized;
    }

    constructor(runtime: IAgentRuntimeWithRAG) {
        this.runtime = runtime;
        this.scraper = new Scraper();
        this.database = new Database();
    }

    private async isTweetProcessed(tweetId: string): Promise<boolean> {
        try {
            return await this.database.isTweetReplied(tweetId);
        } catch (error) {
            elizaLogger.error(`Error checking if tweet ${tweetId} is processed:`, error);
            return false; // Assume not processed if check fails
        }
    }

    private async markTweetAsProcessed(tweetId: string): Promise<void> {
        let retryCount = 0;
        const MAX_RETRIES = 3;
        while (retryCount < MAX_RETRIES) {
            try {
                await this.database.markTweetAsReplied(tweetId);
                elizaLogger.info(`Marked tweet ${tweetId} as processed`);
                return;
            } catch (error) {
                retryCount++;
                elizaLogger.error(`Failed to mark tweet ${tweetId} as processed (attempt ${retryCount}/${MAX_RETRIES}):`, error);
                if (retryCount < MAX_RETRIES) {
                    const backoffTime = 1000 * Math.pow(2, retryCount);
                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                } else {
                    throw new Error(`Failed to mark tweet ${tweetId} as processed after ${MAX_RETRIES} attempts`);
                }
            }
        }
    }

    async initialize(): Promise<void> {
        if (this.isInitialized) {
            return;
        }

        try {
            // Initialize database
            await this.database.initialize();

            // Check for required environment variables
            const requiredEnvVars = [
                'TWITTER_USERNAME',
                'TWITTER_PASSWORD',
                'TWITTER_EMAIL'
            ];

            for (const envVar of requiredEnvVars) {
                if (!process.env[envVar]) {
                    throw new Error(`Missing required environment variable: ${envVar}`);
                }
            }

            elizaLogger.info('Attempting to login to Twitter...');
            elizaLogger.info(`Using credentials for username: ${process.env.TWITTER_USERNAME}`);

            // Login to Twitter with retry logic and timeout
            let loginSuccess = false;
            let retryCount = 0;
            const MAX_RETRIES = 3;
            const LOGIN_TIMEOUT = 15000; // 15 seconds

            while (!loginSuccess && retryCount < MAX_RETRIES) {
                try {
                    const timeoutPromise = new Promise((_, reject) => {
                        setTimeout(() => reject(new Error('Login timeout')), LOGIN_TIMEOUT);
                    });

                    await Promise.race([
                        this.scraper.login(
                            process.env.TWITTER_USERNAME!,
                            process.env.TWITTER_PASSWORD!,
                            process.env.TWITTER_EMAIL!,
                            process.env.TWITTER_2FA_SECRET
                        ),
                        timeoutPromise
                    ]);

                    const isLoggedIn = await this.scraper.isLoggedIn();
                    if (!isLoggedIn) {
                        throw new Error('Login verification failed - isLoggedIn check returned false');
                    }

                    loginSuccess = true;
                    elizaLogger.info('Successfully logged in to Twitter');
                } catch (error) {
                    retryCount++;
                    elizaLogger.error(`Login attempt ${retryCount} failed:`, error);
                    if (retryCount < MAX_RETRIES) {
                        const backoffTime = Math.min(1000 * Math.pow(2, retryCount), 5000);
                        elizaLogger.info(`Retrying login in ${backoffTime}ms...`);
                        await new Promise(resolve => setTimeout(resolve, backoffTime));
                    } else {
                        throw new Error(`Failed to login after ${MAX_RETRIES} attempts: ${error.message}`);
                    }
                }
            }

            // Start polling for mentions
            await this.startPolling();

            // Start periodic posting
            await this.startPeriodicPosting();

            this.isInitialized = true;
        } catch (error) {
            await this.handleError(error, 'initialization');
            throw error;
        }
    }

    private async handleError(error: any, operation: string): Promise<void> {
        elizaLogger.error(`Error during ${operation}:`, error);
    }

    private async startPolling(): Promise<void> {
        let lastCycleDuration = 120000; // Initial duration (120 seconds)
        const MIN_INTERVAL = 60000; // 30 seconds
        const MAX_INTERVAL = 300000; // 300 seconds
        const CYCLE_TIMEOUT = 600000; // 10 minutes
    
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
                lastCycleDuration = Math.min(Math.max(duration * 1.5, MIN_INTERVAL), MAX_INTERVAL);
                elizaLogger.info(`Polling cycle took ${duration.toFixed(2)}ms, next in ${lastCycleDuration}ms`);
            }
        };
    
        // Initial poll after  after a timeout
        setTimeout(async () => {
            await pollWithTimeout();
        }, 5000);
    
        // Schedule periodic polling
        const scheduleNextPoll = () => {
            this.pollInterval = setTimeout(async () => {
                if (!this.isProcessing) {
                    await pollWithTimeout();
                } else {
                    elizaLogger.info('Skipping poll cycle - previous cycle still in progress');
                }
                scheduleNextPoll();
            }, lastCycleDuration);
        };
    
        elizaLogger.info(`Started polling with dynamic interval (~${lastCycleDuration}ms)`);
        scheduleNextPoll();
    }

    private async buildConversationContext(tweet: any, conversationId: string): Promise<string> {
        try {
            const historyQuery = `conversation:${conversationId}`;
            let historyResults: Array<{ content: string; metadata: any }> = [];
            
            // Retry RAG search
            let searchRetryCount = 0;
            const MAX_RETRIES = 3;
            while (searchRetryCount < MAX_RETRIES) {
                try {
                    historyResults = await this.runtime.rag.search(historyQuery, 5);
                    break;
                } catch (error) {
                    searchRetryCount++;
                    elizaLogger.error(`Failed to search RAG (attempt ${searchRetryCount}/${MAX_RETRIES}):`, error);
                    if (searchRetryCount < MAX_RETRIES) {
                        const backoffTime = 1000 * Math.pow(2, searchRetryCount);
                        await new Promise(resolve => setTimeout(resolve, backoffTime));
                    } else {
                        elizaLogger.warn(`RAG search failed after retries, proceeding without history`);
                        historyResults = [];
                    }
                }
            }
    
            const tweetText = `@${tweet.username}: ${tweet.text}`;
    
            // Store tweet in RAG with retry
            let storeRetryCount = 0;
            while (storeRetryCount < MAX_RETRIES) {
                try {
                    await this.runtime.rag.store({
                        id: stringToUuid(Date.now().toString()),
                        content: tweetText,
                        metadata: {
                            conversationId,
                            timestamp: Date.now(),
                            username: tweet.username,
                            tweetId: tweet.id,
                            type: 'tweet'
                        }
                    });
                    break;
                } catch (error) {
                    storeRetryCount++;
                    elizaLogger.error(`Failed to store tweet in RAG (attempt ${storeRetryCount}/${MAX_RETRIES}):`, error);
                    if (storeRetryCount < MAX_RETRIES) {
                        const backoffTime = 1000 * Math.pow(2, storeRetryCount);
                        await new Promise(resolve => setTimeout(resolve, backoffTime));
                    } else {
                        elizaLogger.warn(`RAG store failed after retries, proceeding without storage`);
                    }
                }
            }
    
            if (tweet.inReplyToStatusId) {
                try {
                    const parentTweet = await this.scraper.getTweet(tweet.inReplyToStatusId);
                    if (parentTweet) {
                        const parentText = `@${parentTweet.username}: ${parentTweet.text}`;
                        let parentStoreRetryCount = 0;
                        while (parentStoreRetryCount < MAX_RETRIES) {
                            try {
                                await this.runtime.rag.store({
                                    id: stringToUuid(tweet.inReplyToStatusId),
                                    content: parentText,
                                    metadata: {
                                        conversationId,
                                        timestamp: Date.now(),
                                        username: parentTweet.username,
                                        tweetId: tweet.inReplyToStatusId,
                                        type: 'parent_tweet'
                                    }
                                });
                                break;
                            } catch (error) {
                                parentStoreRetryCount++;
                                elizaLogger.error(`Failed to store parent tweet in RAG (attempt ${parentStoreRetryCount}/${MAX_RETRIES}):`, error);
                                if (parentStoreRetryCount < MAX_RETRIES) {
                                    const backoffTime = 1000 * Math.pow(2, parentStoreRetryCount);
                                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                                } else {
                                    elizaLogger.warn(`RAG store for parent tweet failed after retries`);
                                }
                            }
                        }
                    }
                } catch (error) {
                    elizaLogger.error(`Failed to get parent tweet ${tweet.inReplyToStatusId}:`, error);
                }
            }
    
            const conversationContext = historyResults
                .map(result => result.content)
                .join('\n\n');
    
            return conversationContext || tweet.text;
        } catch (error) {
            elizaLogger.error(`Error building conversation context for tweet ${tweet.id}:`, error);
            return tweet.text;
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
                if (!response.ok) {
                    throw new Error(`Failed to fetch image: ${response.statusText}`);
                }

                const arrayBuffer = await response.arrayBuffer();
                const buffer = Buffer.from(arrayBuffer);
                elizaLogger.info(`Successfully generated image for prompt: ${prompt}`);
                return buffer;
            } catch (error) {
                retryCount++;
                elizaLogger.error(`Failed to generate image (attempt ${retryCount}/${MAX_RETRIES}):`, error);
                if (retryCount < MAX_RETRIES) {
                    const backoffTime = 1000 * Math.pow(2, retryCount);
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
                const mediaData = [{ data: imageBuffer, mediaType: 'image/jpeg' }];
                const result = await this.scraper.sendTweet(caption, tweetId, mediaData);
                const body = await result.json();
    
                if (body.errors) {
                    const error = body.errors[0];
                    throw new Error(`Twitter error (${error.code}): ${error.message}`);
                }
    
                const imageUrl = body.data.create_tweet.tweet_results.result.legacy.entities.media[0].media_url_https;
                elizaLogger.info(`Posted image tweet for tweet ${tweetId}: ${imageUrl}`);
                return imageUrl;
            } catch (error) {
                retryCount++;
                elizaLogger.error(`Failed to post image tweet (attempt ${retryCount}/${MAX_RETRIES}):`, error);
                if (retryCount < MAX_RETRIES) {
                    const backoffTime = 1000 * Math.pow(2, retryCount);
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
            // Validate image buffer
            const imageBuffer = await this.generateImage(prompt);
            if (!imageBuffer || imageBuffer.length === 0) {
                throw new Error('Invalid or empty image buffer');
            }
    
            const memory: Memory = {
                id: stringToUuid(Date.now().toString()),
                userId: stringToUuid(tweet.userId),
                agentId: this.runtime.agentId,
                roomId: stringToUuid(tweet.conversationId),
                content: {
                    text: `Generated image for prompt: ${prompt}`,
                    action: 'GENERATE_IMAGE'
                },
                createdAt: Date.now()
            };
    
            // Store in RAG with retry
            let ragRetryCount = 0;
            const MAX_RAG_RETRIES = 3;
            while (ragRetryCount < MAX_RAG_RETRIES) {
                try {
                    await this.runtime.rag.store({
                        id: memory.id,
                        content: `Generated image: ${prompt}`,
                        metadata: {
                            conversationId: tweet.conversationId,
                            timestamp: Date.now(),
                            type: 'image_generation',
                            prompt,
                            tweetId: tweet.id
                        }
                    });
                    break;
                } catch (error) {
                    ragRetryCount++;
                    elizaLogger.error(`Failed to store image in RAG (attempt ${ragRetryCount}/${MAX_RAG_RETRIES}):`, error);
                    if (ragRetryCount < MAX_RAG_RETRIES) {
                        const backoffTime = 1000 * Math.pow(2, ragRetryCount);
                        await new Promise(resolve => setTimeout(resolve, backoffTime));
                    } else {
                        throw new Error(`Failed to store image in RAG: ${error.message}`);
                    }
                }
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
    
            // Process actions with error logging
            try {
                await this.runtime.processActions(memory, [], state);
            } catch (error) {
                elizaLogger.error(`Failed to process actions for image request:`, error);
                throw new Error(`Action processing failed: ${error.message}`);
            }
    
            // Post image tweet
            const caption = "Here's the image you requested";
            const imageUrl = await this.postImageTweet(tweet.id, imageBuffer, caption);
    
            // Store image post in RAG
            await this.runtime.rag.store({
                id: stringToUuid(Date.now().toString()),
                content: `Posted image: ${imageUrl} for prompt: ${prompt}`,
                metadata: {
                    conversationId: tweet.conversationId,
                    timestamp: Date.now(),
                    type: 'image_post',
                    prompt,
                    imageUrl,
                    tweetId: tweet.id
                }
            });
    
            this.imageConversations.set(tweet.conversationId, { lastPrompt: prompt, lastImageUrl: imageUrl });
            elizaLogger.info(`Successfully generated and shared image for prompt: ${prompt}`);
        } catch (error) {
            elizaLogger.error(`Error handling image request for tweet ${tweet.id}:`, error);
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
            /nsfw/i,
            /explicit/i,
            /profane/i,
            /blasphem/i,
            /fuck/i,
            /shit/i,
            /damn/i,
            /hell/i,
            /bitch/i,
            /ass/i,
            /porn/i,
            /sex/i,
            /nude/i,
            /drug/i,
            /alcohol/i,
            /drunk/i,
            /high/i
        ];

        const whitelist = ['jesus', 'christ', 'god', 'holy', 'spirit', 'bibl']; // Allow biblical terms
        const lowerText = text.toLowerCase();
        if (whitelist.some(term => lowerText.includes(term))) {
            return false;
        }

        const isInappropriate = inappropriatePatterns.some(pattern => {
            const match = pattern.test(lowerText);
            if (match) {
                elizaLogger.info(`Text flagged as inappropriate due to pattern: ${pattern}`);
            }
            return match;
        });

        return isInappropriate;
    }

    private async analyzeTweetIntent(tweet: any, cleanText: string): Promise<TweetIntent> {
        try {
            if (this.isInappropriateContent(cleanText)) {
                elizaLogger.info(`Skipping inappropriate content in tweet ${tweet.id}`);
                return {
                    shouldGenerateImage: false,
                    shouldReply: false,
                    isImageEditRequest: false
                };
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
                elizaLogger.error('Failed to parse intent analysis:', error);
                elizaLogger.info('Raw intent analysis:', intentAnalysis);

                const lowerText = cleanText.toLowerCase();
                intent = {
                    shouldGenerateImage: lowerText.includes('image') || lowerText.includes('generate') || lowerText.includes('create'),
                    shouldReply: true,
                    isImageEditRequest: lowerText.includes('edit') || lowerText.includes('modify') || lowerText.includes('change'),
                    reasoning: 'Fallback to keyword analysis'
                };
            }

            elizaLogger.info(`Intent analysis for tweet ${tweet.id}:`, intent);

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
                            elizaLogger.warn(`Failed to generate prompt for tweet ${tweet.id}, retrying (${retryCount}/${MAX_RETRIES})`);
                            if (retryCount === MAX_RETRIES) {
                                elizaLogger.warn(`Failed to generate appropriate prompt for tweet ${tweet.id} after retries`);
                                return {
                                    shouldGenerateImage: false,
                                    shouldReply: true,
                                    isImageEditRequest: false,
                                    reasoning: 'Failed to generate valid image prompt'
                                };
                            }
                            continue;
                        }
                        break;
                    } catch (error) {
                        retryCount++;
                        elizaLogger.error(`Error generating prompt (attempt ${retryCount}/${MAX_RETRIES}):`, error);
                        if (retryCount === MAX_RETRIES) {
                            return {
                                shouldGenerateImage: false,
                                shouldReply: true,
                                isImageEditRequest: false,
                                reasoning: 'Failed to generate valid image prompt'
                            };
                        }
                    }
                }
            }

            return {
                shouldGenerateImage: intent.shouldGenerateImage,
                shouldReply: intent.shouldReply,
                isImageEditRequest: intent.isImageEditRequest,
                prompt
            };
        } catch (error) {
            elizaLogger.error('Error analyzing tweet intent:', error);
            return {
                shouldGenerateImage: false,
                shouldReply: true,
                isImageEditRequest: false
            };
        }
    }

    private async buildImagePrompt(tweet: any, text: string, previousPrompt?: string): Promise<string | null> {
        try {
            const historyQuery = `conversation:${tweet.conversationId} type:image_generation`;
            let historyResults: Array<{ content: string; metadata: any }> = [];
            try {
                historyResults = await this.runtime.rag.search(historyQuery, 3);
            } catch (error) {
                elizaLogger.warn(`Failed to fetch RAG history for tweet ${tweet.id}:`, error);
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
                return "A serene biblical scene of a shepherd tending sheep in a peaceful valley at sunset"; // Default prompt
            }

            elizaLogger.info(`Generated image prompt for tweet ${tweet.id}: ${finalPrompt}`);
            return finalPrompt;
        } catch (error) {
            elizaLogger.error(`Error building image prompt for tweet ${tweet.id}:`, error);
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

            elizaLogger.info('Test 2: Searching for bot\'s own tweets...');
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
            elizaLogger.error('Error in testSearchFunctionality:', error);
        }
    }

    private async pollMentions(): Promise<void> {
        try {
            if (this.isProcessing) {
                elizaLogger.info('Previous polling cycle still in progress, skipping...');
                return;
            }

            this.isProcessing = true;
            elizaLogger.info('Starting new polling cycle...');

            const username = process.env.TWITTER_USERNAME!.toLowerCase();
            if (!username) {
                throw new Error('TWITTER_USERNAME environment variable is not set');
            }

            // Verify authentication
            if (!(await this.scraper.isLoggedIn())) {
                elizaLogger.warn('Scraper not authenticated, re-initializing...');
                await this.initializeScraper();
            }

            // Track processed tweets in this cycle to avoid duplicates
            const processedTweetIds = new Set<string>();

            // Simplified to single query with in-code filtering
            const searchQuery = `@${username}`;
            elizaLogger.info(`Executing search query: "${searchQuery}"`);
            const mentionsGenerator = this.scraper.searchTweets(searchQuery, 20, SearchMode.Latest);

            let mentionsCount = 0;
            let hasResults = false;
            let lastTweetId: string | null = null;

            for await (const tweet of mentionsGenerator) {
                elizaLogger.info(`Raw tweet data: ID=${tweet.id}, Username=${tweet.username}, Text="${tweet.text}", IsRetweet=${tweet.isRetweet}`);

                // Skip duplicates within cycle
                if (processedTweetIds.has(tweet.id)) {
                    elizaLogger.info(`Skipping duplicate tweet ${tweet.id} in this cycle`);
                    continue;
                }

                // Early processed check
                if (await this.isTweetProcessed(tweet.id)) {
                    elizaLogger.info(`Skipping already processed tweet ${tweet.id}`);
                    processedTweetIds.add(tweet.id);
                    continue;
                }

                // Skip bot's own tweets
                if (tweet.username.toLowerCase() === username) {
                    elizaLogger.info(`Skipping bot's own tweet: ${tweet.id}`);
                    processedTweetIds.add(tweet.id);
                    continue;
                }

                // Skip retweets
                if (tweet.isRetweet) {
                    elizaLogger.info(`Skipping retweet: ${tweet.id}`);
                    processedTweetIds.add(tweet.id);
                    continue;
                }

                // Verify mention
                if (!tweet.text.toLowerCase().includes(`@${username}`)) {
                    elizaLogger.info(`Skipping tweet ${tweet.id} - does not contain @${username}`);
                    processedTweetIds.add(tweet.id);
                    continue;
                }

                hasResults = true;
                lastTweetId = tweet.id;
                elizaLogger.info(`Processing mention: ${tweet.id} from @${tweet.username}: ${tweet.text}`);

                try {
                    // Mark as processed before any action
                    await this.markTweetAsProcessed(tweet.id);
                    processedTweetIds.add(tweet.id);

                    const cleanText = tweet.text.replace(/@\w+/g, '').trim();
                    if (!cleanText) {
                        elizaLogger.info(`Skipping empty tweet ${tweet.id} after cleaning`);
                        continue;
                    }

                    if (this.isInappropriateContent(cleanText)) {
                        elizaLogger.info(`Skipping inappropriate content in tweet ${tweet.id}`);
                        continue;
                    }

                    const intent = await this.analyzeTweetIntent(tweet, cleanText);
                    elizaLogger.info(`Intent analysis for tweet ${tweet.id}:`, intent);

                    if (intent.shouldGenerateImage && intent.prompt) {
                        try {
                            await this.handleImageRequest(tweet, intent.prompt);
                        } catch (error) {
                            elizaLogger.error(`Failed to generate image for tweet ${tweet.id}:`, error);
                            const fallbackResponse = await this.generateResponse(tweet, cleanText, {
                                ...intent,
                                shouldGenerateImage: false
                            });
                            if (fallbackResponse) {
                                await this.replyToTweet(tweet.id, fallbackResponse);
                            }
                        }
                    } else {
                        const response = await this.generateResponse(tweet, cleanText, intent);
                        if (response) {
                            await this.replyToTweet(tweet.id, response);
                        }
                    }

                    mentionsCount++;
                } catch (error) {
                    elizaLogger.error(`Error processing tweet ${tweet.id}:`, error);
                }
            }

            if (!hasResults) {
                elizaLogger.info('No mentions found in this polling cycle');
            } else {
                elizaLogger.info(`Processed ${mentionsCount} mentions, last tweet ID: ${lastTweetId}`);
            }
        } catch (error) {
            elizaLogger.error('Error in pollMentions:', error);
            if (error.message?.includes('Unauthorized')) {
                elizaLogger.error('Authentication error, re-initializing scraper...');
                await this.initializeScraper();
            } else if (error.message?.includes('429')) {
                const backoffTime = Math.min(1000 * Math.pow(2, 0), 60000);
                elizaLogger.info(`Rate limited, waiting ${backoffTime}ms`);
                await new Promise(resolve => setTimeout(resolve, backoffTime));
            }
        } finally {
            this.isProcessing = false;
            elizaLogger.info('Polling cycle completed');
        }
    }

    async stop(): Promise<void> {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
        await this.database.close();
    }

    async postTweet(content: string): Promise<void> {
        try {
            const memory: Memory = {
                id: stringToUuid(Date.now().toString()),
                userId: this.runtime.agentId,
                agentId: this.runtime.agentId,
                roomId: stringToUuid('twitter'),
                content: {
                    text: content,
                    action: 'POST_TWEET'
                },
                createdAt: Date.now()
            };

            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid('twitter'),
                actors: this.runtime.character.name,
                recentMessages: content,
                recentMessagesData: [memory]
            };

            await this.runtime.processActions(memory, [], state);

            const result = await this.scraper.sendTweet(content);
            const body = await result.json();

            if (body.errors) {
                const error = body.errors[0];
                throw new Error(`Twitter error (${error.code}): ${error.message}`);
            }

            if (!body?.data?.create_tweet?.tweet_results?.result) {
                throw new Error("Failed to post tweet: No tweet result in response");
            }

            elizaLogger.info(`Posted tweet: ${content}`);
        } catch (error) {
            await this.handleError(error, 'posting tweet');
            throw error;
        }
    }

    async replyToTweet(tweetId: string, content: string): Promise<void> {
        try {
            const memory: Memory = {
                id: stringToUuid(Date.now().toString()),
                userId: this.runtime.agentId,
                agentId: this.runtime.agentId,
                roomId: stringToUuid('twitter'),
                content: {
                    text: content,
                    action: 'REPLY_TO_TWEET',
                    inReplyTo: stringToUuid(tweetId + "-" + this.runtime.agentId)
                },
                createdAt: Date.now()
            };

            const state: State = {
                bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                lore: this.runtime.character.lore.join(' '),
                messageDirections: this.runtime.character.style.post.join(' '),
                postDirections: this.runtime.character.style.post.join(' '),
                roomId: stringToUuid('twitter'),
                actors: this.runtime.character.name,
                recentMessages: content,
                recentMessagesData: [memory]
            };

            await this.runtime.processActions(memory, [], state);

            const tweet = await this.scraper.getTweet(tweetId);
            if (!tweet) {
                throw new Error(`Failed to get tweet ${tweetId}`);
            }

            const replyContent = `@${tweet.username} ${content}`;
            const result = await this.scraper.sendTweet(replyContent, tweetId);
            const body = await result.json();

            if (body.errors) {
                const error = body.errors[0];
                throw new Error(`Twitter error (${error.code}): ${error.message}`);
            }

            if (!body?.data?.create_tweet?.tweet_results?.result) {
                throw new Error("Failed to post reply: No tweet result in response");
            }

            elizaLogger.info(`Posted reply to tweet ${tweetId}: ${content}`);
        } catch (error) {
            await this.handleError(error, 'posting reply');
            throw error;
        }
    }

    async searchTweets(query: string) {
        try {
            return await this.scraper.searchTweets(query, 10, SearchMode.Top);
        } catch (error) {
            await this.handleError(error, "tweet search");
            throw error;
        }
    }

    async getTimeline() {
        try {
            return await this.scraper.getTweets(process.env.TWITTER_USERNAME!, 10);
        } catch (error) {
            await this.handleError(error, "timeline fetch");
            throw error;
        }
    }

    private async startPeriodicPosting(): Promise<void> {
        const POST_INTERVAL = 60000; // 1 minute
        const MAX_RETRIES = 3;

        elizaLogger.info(`Starting periodic posting every ${POST_INTERVAL/1000} seconds`);

        const generateAndPostTweet = async () => {
            if (this.isProcessing) {
                elizaLogger.warn('Previous tweet generation still in progress, forcing reset');
                this.isProcessing = false;
            }

            this.isProcessing = true;
            let retryCount = 0;

            try {
                const topics = this.runtime.character.topics;
                const style = this.runtime.character.style.post;
                const adjectives = this.runtime.character.adjectives;

                const randomTopic = topics[Math.floor(Math.random() * topics.length)];
                const randomStyle = style[Math.floor(Math.random() * style.length)];
                const randomAdjective = adjectives[Math.floor(Math.random() * adjectives.length)];

                elizaLogger.info(`Generating tweet about ${randomTopic} in ${randomAdjective} style`);

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

                const context = composeContext({
                    state,
                    template: `You are ${this.runtime.character.name}. Generate a tweet that is ${randomAdjective} about ${randomTopic}. The tweet should ${randomStyle}. Keep it under 280 characters. No hashtags or emojis.`
                });

                const tweetContent = await generateText({
                    runtime: this.runtime,
                    context,
                    modelClass: ModelClass.LARGE
                });

                elizaLogger.info(`Generated tweet content: ${tweetContent}`);

                while (retryCount < MAX_RETRIES) {
                    try {
                        await this.postTweet(tweetContent);
                        elizaLogger.info('Successfully posted tweet');
                        break;
                    } catch (error) {
                        retryCount++;
                        elizaLogger.error(`Failed to post tweet (attempt ${retryCount}/${MAX_RETRIES}):`, error);
                        if (retryCount < MAX_RETRIES) {
                            const backoffTime = Math.min(1000 * Math.pow(2, retryCount), 5000);
                            elizaLogger.info(`Retrying in ${backoffTime}ms...`);
                            await new Promise(resolve => setTimeout(resolve, backoffTime));
                        }
                    }
                }

                if (retryCount === MAX_RETRIES) {
                    elizaLogger.error('Failed to post tweet after all retries');
                }
            } catch (error) {
                elizaLogger.error('Error in generateAndPostTweet:', error);
                await this.handleError(error, 'periodic posting');
            } finally {
                this.isProcessing = false;
                elizaLogger.info('Tweet generation completed, reset isProcessing flag');
            }
        };

        setTimeout(generateAndPostTweet, 5000);

        const scheduleNextPost = () => {
            elizaLogger.info(`Scheduling next post in ${POST_INTERVAL/1000} seconds`);

            setTimeout(async () => {
                try {
                    await generateAndPostTweet();
                } catch (error) {
                    elizaLogger.error('Error in scheduled post:', error);
                } finally {
                    scheduleNextPost();
                }
            }, POST_INTERVAL);
        };

        scheduleNextPost();
    }

    private async generateResponse(tweet: Tweet, cleanText: string, intent: TweetIntent): Promise<string | null> {
        try {
            if (intent.shouldGenerateImage) {
                await this.handleImageRequest(tweet, intent.prompt!);
                return null;
            } else if (intent.shouldReply) {
                const conversationId = tweet.inReplyToStatusId || tweet.id;
                const context = await this.buildConversationContext(tweet, conversationId);

                return await generateText({
                    runtime: this.runtime,
                    context: composeContext({
                        state: {
                            bio: Array.isArray(this.runtime.character.bio) ? this.runtime.character.bio.join(' ') : this.runtime.character.bio,
                            lore: this.runtime.character.lore.join(' '),
                            messageDirections: this.runtime.character.style.post.join(' '),
                            postDirections: this.runtime.character.style.post.join(' '),
                            roomId: stringToUuid(conversationId),
                            actors: this.runtime.character.name,
                            recentMessages: context,
                            recentMessagesData: []
                        },
                        template: `You are Jesus Christ. Respond to this tweet: "${cleanText}" using one of these formats:
1. A relevant Scripture quote
2. A paraphrased biblical truth
3. A Christlike question
4. A short modern parable

Previous conversation:\n${context}
Keep it under 280 characters. No hashtags or emojis.`
                    }),
                    modelClass: ModelClass.LARGE
                });
            }
            return null;
        } catch (error) {
            elizaLogger.error('Error generating response:', error);
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
            elizaLogger.error('Failed to re-initialize Twitter scraper:', error);
            throw error;
        }
    }
}