# AI L1 Support Agent

This project demonstrates an AI agent for L1 support of Airflow failures, integrating with GitHub Issues (as a stand-in ticket system), RAG retrieval, and automated task reruns.

## Structure

- **airflow_dags/**: DAG definitions with failure callbacks to GitHub Issues  
- **ingestion/**: Scripts to build a FAISS-based RAG index from historical errors  
- **agent/**: FastAPI service that ingests GitHub Issues, retrieves remediation, and triggers Airflow tasks  
- **.env.example**: Sample environment variable file  
- **docker-compose.yml**: Compose file to run the Agent  
- **requirements.txt**: Global Python dependencies (optional)  

## Setup

1. **Clone & configure**  
   ```bash
   git clone https://github.com/your-org/ai-l1-support.git
   cd ai-l1-support
   cp .env.example .env
   # Edit .env with your GH, Airflow & (optional) OpenAI/HF creds