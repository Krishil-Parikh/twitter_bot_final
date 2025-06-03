# Twitter Bot Setup Guide

This guide will walk you through setting up the Twitter bot from scratch, including all necessary installations, configurations, and builds.

## Prerequisites

- Node.js (v16 or higher)
- npm (v7 or higher)
- Git
- A Twitter account for the bot
- Google Cloud account (for Gemini API)

## Step 1: Clone and Setup Repository

```bash
# Clone the repository
git clone <repository-url>
cd twitter_agent_eliza

# Install dependencies for the main project
npm install
```

## Step 2: Setup Twitter Plugin

```bash
# Navigate to the plugin directory
cd packages/plugin-twitter

# Install plugin dependencies
npm install

# Build the plugin
npm run build
```

## Step 3: Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Twitter Credentials
TWITTER_USERNAME=your_bot_username
TWITTER_PASSWORD=your_bot_password
TWITTER_EMAIL=your_bot_email
TWITTER_2FA_SECRET=your_2fa_secret_if_enabled

# Google Cloud (for Gemini API)
GOOGLE_API_KEY=your_google_api_key

# Bot Configuration
TWITTER_POLL_INTERVAL=120000  # Polling interval in milliseconds (default: 120 seconds)
```

## Step 4: Database Setup

The bot uses SQLite for storing processed tweets. The database will be automatically created when you first run the bot.

## Step 5: Build the Project

```bash
# Return to root directory
cd ../..

# Build the entire project
npm run build
```

## Step 6: Testing the Setup

Before running the bot, test the configuration:

```bash
# Test Twitter authentication
npm run test:twitter

# Test image generation
npm run test:image
```

## Step 7: Running the Bot

```bash
# Start the bot
npm run start
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify your Twitter credentials in `.env`
   - Ensure 2FA is properly configured if enabled
   - Check if your Twitter account is not locked

2. **Rate Limiting**
   - The bot implements exponential backoff
   - Check logs for rate limit messages
   - Consider increasing `TWITTER_POLL_INTERVAL`

3. **Image Generation Failures**
   - Verify Google API key
   - Check internet connectivity
   - Ensure the prompt is appropriate

### Logs

Logs are stored in:
- `logs/twitter.log` - Twitter-specific operations
- `logs/error.log` - Error messages
- `logs/combined.log` - All logs combined

## Bot Features

1. **Mention Handling**
   - Responds to direct mentions
   - Processes image generation requests
   - Maintains conversation context

2. **Image Generation**
   - Uses Pollinations.ai for image generation
   - Supports image modifications
   - Implements retry logic

3. **Content Filtering**
   - Filters inappropriate content
   - Maintains biblical appropriateness
   - Implements whitelist for religious terms

## Maintenance

### Regular Tasks

1. **Database Maintenance**
   ```bash
   # Clean old processed tweets (older than 30 days)
   npm run clean:db
   ```

2. **Log Rotation**
   ```bash
   # Rotate logs
   npm run rotate:logs
   ```

### Monitoring

Monitor the bot's health through:
- Log files
- Twitter activity
- Database size
- API rate limits

## Security Considerations

1. **API Keys**
   - Never commit `.env` file
   - Rotate API keys regularly
   - Use environment variables

2. **Twitter Account**
   - Use a dedicated bot account
   - Enable 2FA
   - Regular password changes

3. **Database**
   - Regular backups
   - Access control
   - Data encryption

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Your License Here]

## Support

For issues and support:
1. Check the troubleshooting guide
2. Search existing issues
3. Create a new issue if needed

## Updates

Regular updates are recommended:
```bash
# Update dependencies
npm update

# Rebuild after updates
npm run build
``` 