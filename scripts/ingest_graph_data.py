import pickle
from neo4j import GraphDatabase
import os
from langchain.schema import Document

# Load environment vars (if any)
from dotenv import load_dotenv
load_dotenv()

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")

# Load docs.pkl
with open("agent/rag/docs.pkl", "rb") as f:
    raw_docs = pickle.load(f)

docs = [
    Document(page_content=d["error"], metadata=d["metadata"])
    if isinstance(d, dict) else d
    for d in raw_docs
]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def ingest_doc(tx, doc: Document):
    m = doc.metadata
    tx.run("""
        MERGE (d:DAG {dag_id: $dag_id})
        MERGE (c:Category {name: $category})
        CREATE (l:Log {
            log_excerpt: $excerpt,
            task_id: $task_id,
            dag_run_id: $dag_run_id,
            source: $source
        })
        MERGE (l)-[:BELONGS_TO]->(d)
        MERGE (l)-[:HAS_CATEGORY]->(c)
    """, {
        "dag_id": m.get("dag_id"),
        "task_id": m.get("task_id"),
        "dag_run_id": m.get("dag_run_id"),
        "source": m.get("source", "airflow"),
        "category": m.get("category", "generic"),
        "excerpt": doc.page_content[:300]  # Trim long logs
    })

with driver.session() as session:
    for doc in docs:
        session.write_transaction(ingest_doc, doc)

print(f"✅ Ingested {len(docs)} logs into Neo4j.")
