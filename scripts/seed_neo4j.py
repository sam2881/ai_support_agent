import pickle
import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

DOCS_PATH = os.path.join("agent", "rag", "docs.pkl")

class Neo4jSeeder:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def run(self):
        if not os.path.exists(DOCS_PATH):
            print(f"❌ File not found: {DOCS_PATH}")
            return

        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)

        with self.driver.session() as session:
            for i, doc in enumerate(docs):
                metadata = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
                issue_id = f"Issue_{i}"

                session.write_transaction(self._create_issue_graph, issue_id, doc.page_content, metadata)

        print("✅ Neo4j seeding complete.")

    @staticmethod
    def _create_issue_graph(tx, issue_id, text, metadata):
        category = metadata.get("category", "unknown").lower()
        solution = metadata.get("solution", "N/A")
        dag_id = metadata.get("dag_id", "")
        task_id = metadata.get("task_id", "")

        tx.run("""
            MERGE (i:Issue {id: $issue_id})
              ON CREATE SET i.text = $text, i.dag_id = $dag_id, i.task_id = $task_id

            MERGE (c:Category {name: $category})
            MERGE (i)-[:HAS_CATEGORY]->(c)

            MERGE (r:Remediation {text: $solution})
            MERGE (i)-[:HAS_REMEDIATION]->(r)
        """, issue_id=issue_id, text=text, category=category, solution=solution, dag_id=dag_id, task_id=task_id)


if __name__ == "__main__":
    seeder = Neo4jSeeder()
    try:
        seeder.run()
    finally:
        seeder.close()
