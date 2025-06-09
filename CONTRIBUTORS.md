# Contributors

We thank the following people for their contributions to the Banking ML Platform project.

## Core Development Team

### Project Lead
- **Lead Developer** - Initial project architecture, ML pipeline design, and API development
  - Designed the overall system architecture
  - Implemented fraud detection and credit risk models
  - Built the FastAPI backend with authentication
  - Set up CI/CD pipeline and deployment infrastructure

### Machine Learning Engineers
- **ML Engineer** - Model development and optimization
  - Implemented XGBoost fraud detection with Optuna optimization
  - Developed credit risk scoring with synthetic labels
  - Created feature engineering pipeline
  - Built model evaluation and monitoring systems

### AI/LLM Engineers
- **AI Engineer** - LLM services and document processing
  - Integrated Hugging Face Transformers for chatbot service
  - Built document processing pipeline for compliance extraction
  - Implemented AML covenant detection
  - Designed conversation management system

### DevOps Engineers
- **DevOps Engineer** - Infrastructure and deployment
  - Set up Docker containerization and orchestration
  - Configured monitoring with Prometheus and Grafana
  - Implemented ELK stack for centralized logging
  - Built GitHub Actions CI/CD pipeline

### Data Engineers
- **Data Engineer** - Data pipeline and validation
  - Implemented data download and preprocessing pipeline
  - Integrated Great Expectations for data validation
  - Built feature engineering and transformation pipeline
  - Created data quality monitoring

## Open Source Contributors

We welcome contributions from the community! Here's how to get involved:

### How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/banking-ml-platform.git
   ```

2. **Set Up Development Environment**
   ```bash
   pip install -r requirements.txt
   pre-commit install
   