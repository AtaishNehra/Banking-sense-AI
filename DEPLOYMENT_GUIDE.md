# Deployment Guide

## Quick Deployment Options

### 1. Replit Deployment (Recommended - Free)

Your application is already configured for Replit deployment:

1. **One-Click Deploy**: Click the "Deploy" button in your Replit interface
2. **Automatic Setup**: Replit handles SSL, domain, and hosting automatically
3. **Environment Variables**: Set these in Replit Secrets:
   - `DATABASE_URL` (automatically provided by Replit PostgreSQL)
   - `KAGGLE_USERNAME` (your Kaggle username)
   - `KAGGLE_KEY` (your Kaggle API key)

### 2. Railway Deployment (Free Tier)

1. Connect your GitHub repository to Railway
2. Add PostgreSQL database addon
3. Set environment variables in Railway dashboard
4. Deploy automatically on git push

### 3. Render Deployment (Free Tier)

1. Connect repository to Render
2. Create PostgreSQL database
3. Configure environment variables
4. Auto-deploy from GitHub

### 4. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t banking-ml-platform .
docker run -p 5000:5000 banking-ml-platform
```

## Environment Variables Required

```bash
DATABASE_URL=postgresql://user:password@host:port/database
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

## Database Setup

The application automatically creates all required tables on startup. For production:

1. Create PostgreSQL database
2. Set DATABASE_URL environment variable
3. Application handles schema creation automatically

## Data Pipeline Setup

To download and process authentic PaySim data:

```bash
# Set Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download and process data
python data_pipeline/download_data.py
python data_pipeline/quick_preprocess.py
python models/quick_training.py
```

## Production Considerations

### Security
- Use HTTPS (automatically provided by Replit/Railway/Render)
- Set strong database passwords
- Keep Kaggle API keys secure
- Enable database connection pooling

### Performance
- PostgreSQL connection pooling enabled
- Model caching implemented
- Async database operations
- Optimized queries with indexes

### Monitoring
- Health check endpoint: `/health`
- Prometheus metrics: `/metrics`
- Database statistics: `/database/stats`
- API documentation: `/docs`

## Scaling Options

### Horizontal Scaling
- Multiple app instances behind load balancer
- Shared PostgreSQL database
- Redis session storage (optional)

### Vertical Scaling
- Increase memory for ML models
- More CPU cores for concurrent requests
- Larger database instances

## Backup Strategy

### Database Backups
- Automated daily backups on cloud platforms
- Point-in-time recovery available
- Cross-region backup replication

### Model Versioning
- Models stored in `/models/saved/`
- Version tracking in database
- Hot-swappable model updates

## Troubleshooting

### Common Issues
1. **Database Connection**: Check DATABASE_URL format
2. **Kaggle API**: Verify username and API key
3. **Model Loading**: Ensure models are trained and saved
4. **Memory Issues**: Increase container memory for ML models

### Health Checks
- Application: `GET /health`
- Database: `GET /database/stats`
- ML Models: Check startup logs for model loading

### Logs
- Application logs available in platform dashboards
- Database query logs for debugging
- Model prediction logs for monitoring

## Cost Optimization

### Free Tier Limits
- **Replit**: Generous free tier with automatic scaling
- **Railway**: $5 credit monthly, PostgreSQL included
- **Render**: 750 hours free per month

### Resource Management
- Efficient model loading and caching
- Database connection pooling
- Optimized queries and indexes
- Static file caching