# agent/neo4j/graph_client.py
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from typing import List, Dict, Any, Optional
from agent.workflows.config import settings
from datetime import datetime

logger = logging.getLogger(__name__)

class GraphClient:
    def __init__(self):
        """
        Initializes the Neo4j Graph Client using credentials from settings.
        Establishes a connection to the Neo4j database.
        """
        try:
            # Securely get password value
            neo4j_password = settings.NEO4J_PASSWORD.get_secret_value()
        except AttributeError:
            logger.warning("settings.NEO4J_PASSWORD does not have get_secret_value(). Assuming it's a plain string (less secure).")
            neo4j_password = settings.NEO4J_PASSWORD # Fallback for plain string if not a SecretStr

        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USERNAME
        self.password = neo4j_password
        self.driver = None

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connection established.")
        except ServiceUnavailable as e:
            logger.error(f"❌ Failed to connect to Neo4j at {self.uri}: {e}")
            self.driver = None # Ensure driver is None if connection fails
            raise # Re-raise to indicate a critical setup failure
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during Neo4j connection: {e}")
            self.driver = None
            raise

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Helper to execute a Cypher query and return results."""
        if not self.driver:
            logger.error("Neo4j driver is not initialized. Cannot execute query.")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except ServiceUnavailable as e:
            logger.error(f"❌ Neo4j service unavailable during query: {query} - {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error executing Neo4j query: {query} with params {parameters} - {e}", exc_info=True)
            return []

    def query_related_tasks(self, issue_text: str) -> List[Dict[str, Any]]:
        """
        Queries Neo4j for tasks and DAGs related to an error message.
        Assumes an existing graph structure of (Error)-[:RELATED_TO]->(Task)<-[:PART_OF]-(DAG).
        """
        query = """
        MATCH (e:Error)-[:RELATED_TO]->(t:Task)<-[:PART_OF]-(d:DAG)
        WHERE toLower(e.message) CONTAINS toLower($text)
        RETURN d.name AS dag_name, t.name AS task_name
        LIMIT 1
        """
        records = self._execute_query(query, {"text": issue_text})
        logger.info(f"🔍 Graph query for related tasks: {records}")
        return records
        
    def log_issue_classification(self, issue_number: int, category: str, description: str):
        """
        Logs an issue and its classification in Neo4j, creating nodes if they don't exist.
        Uses `issueNumber` as the primary identifier for Issue nodes for consistency.
        """
        query = """
        MERGE (i:Issue {issueNumber: $issue_number})
        ON CREATE SET i.createdAt = datetime(), i.title = $description // Use description as initial title
        ON MATCH SET i.lastClassifiedAt = datetime(), i.title = $description // Update title/timestamp on re-classification

        MERGE (c:Category {name: $category})
        
        MERGE (i)-[r:CLASSIFIED_AS]->(c)
        ON CREATE SET r.classifiedAt = datetime()
        ON MATCH SET r.lastClassifiedAt = datetime()
        """
        self._execute_query(query, {
            "issue_number": issue_number,
            "category": category,
            "description": description
        })
        logger.info(f"📝 Logged classification for issue #{issue_number} as '{category}' in Neo4j.")

    def log_agent_relationship(self, issue_number: int, agent_name: str, status: str = "classified"):
        """
        Logs that an issue was handled by a specific agent.
        Uses `issueNumber` for Issue nodes and `name` for Agent nodes.
        Dynamically sets agent team based on category.
        """
        query = """
        MERGE (i:Issue {issueNumber: $issue_number})
        MERGE (a:Agent {name: $agent_name})
        ON CREATE SET a.createdAt = datetime(), a.team = CASE $agent_name
                                                WHEN 'airflow' THEN 'Data Engineering'
                                                WHEN 'devops' THEN 'Infrastructure'
                                                WHEN 'access' THEN 'IT Support'
                                                ELSE 'Unknown'
                                            END
        ON MATCH SET a.lastUpdated = datetime()

        MERGE (i)-[r:HANDLED_BY]->(a)
        ON CREATE SET r.createdAt = datetime(), r.status = $status
        ON MATCH SET r.lastHandledAt = datetime(), r.status = $status
        """
        self._execute_query(query, {
            "issue_number": issue_number,
            "agent_name": agent_name, # Renamed from 'category' to 'agent_name' for clarity with Agent node 'name' property
            "status": status
        })
        logger.info(f"🔗 Linked Issue #{issue_number} to Agent '{agent_name}' with status '{status}' in Neo4j.")

    def log_resolution(self, issue_number: int, outcome: str):
        """
        Logs the resolution of an issue, updating the HANDLED_BY relationship.
        """
        query = '''
        MATCH (i:Issue {issueNumber: $issue_number})-[r:HANDLED_BY]->(a:Agent)
        SET r.resolved = true, r.outcome = $outcome, r.resolved_at = datetime()
        '''
        self._execute_query(query, {"issue_number": issue_number, "outcome": outcome})
        logger.info(f"✅ Logged resolution for issue #{issue_number} with outcome '{outcome}'.")

    def query_similar_issues(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Queries Neo4j for issues similar to the provided text based on title or description.
        Returns issue number, title, and category.
        """
        query = """
        MATCH (i:Issue)-[:CLASSIFIED_AS]->(c:Category)
        WHERE toLower(i.title) CONTAINS toLower($text) OR toLower(i.description) CONTAINS toLower($text)
        RETURN i.issueNumber AS issueNumber, i.title AS title, c.name AS category, i.createdAt AS createdAt
        ORDER BY i.createdAt DESC
        LIMIT $limit
        """
        results = self._execute_query(query, {"text": text, "limit": limit})
        logger.info(f"🔎 Queried similar issues in Neo4j for '{text[:30]}...': Found {len(results)} results.")
        return results

    def log_log_entry(self, log_excerpt: str, dag_id: Optional[str] = None, task_id: Optional[str] = None, 
                      source: Optional[str] = None, root_cause: Optional[str] = None, remediation: Optional[str] = None):
        """
        Logs a log entry in Neo4j, potentially linking it to a DAG and Task.
        """
        query = """
        MERGE (l:Log {text: $log_excerpt})
        ON CREATE SET l.source = $source,
                      l.root_cause = $root_cause,
                      l.remediation = $remediation,
                      l.createdAt = datetime()
        ON MATCH SET l.lastUpdated = datetime(),
                     l.source = coalesce($source, l.source),
                     l.root_cause = coalesce($root_cause, l.root_cause),
                     l.remediation = coalesce($remediation, l.remediation)
        WITH l
        OPTIONAL MATCH (d:DAG {name: $dag_id})
        FOREACH (ignore_d IN CASE WHEN $dag_id IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d) ON CREATE SET d.createdAt = datetime()
        )
        WITH l, d
        OPTIONAL MATCH (t:Task {name: $task_id})
        FOREACH (ignore_t IN CASE WHEN $task_id IS NOT NULL THEN [1] ELSE [] END |
            MERGE (t) ON CREATE SET t.createdAt = datetime()
        )
        WITH l, d, t
        FOREACH (x IN CASE WHEN d IS NOT NULL AND t IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d)-[:HAS_TASK]->(t)
        )
        FOREACH (y IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
            MERGE (t)-[:EMITTED]->(l)
        )
        """
        self._execute_query(query, {
            "log_excerpt": log_excerpt,
            "dag_id": dag_id,
            "task_id": task_id,
            "source": source,
            "root_cause": root_cause,
            "remediation": remediation
        })
        logger.info(f"📊 Logged entry to Neo4j. DAG: {dag_id}, Task: {task_id}.")
        
    def get_context_for_log(self, dag_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves context (like common root causes) for a specific DAG and Task.
        """
        query = """
        MATCH (d:DAG {name: $dag_id})-[:HAS_TASK]->(t:Task {name: $task_id})-[:EMITTED]->(l:Log)
        RETURN d.name AS dag, t.name AS task, collect(DISTINCT l.root_cause) AS causes
        LIMIT 1
        """
        results = self._execute_query(query, {"dag_id": dag_id, "task_id": task_id})
        if results:
            return results[0]
        return None

    def get_related_solutions(self, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves common solutions associated with a given category.
        Assumes Solution nodes are linked to Category nodes via HAS_SOLUTION relationship.
        """
        query = """
        MATCH (c:Category {name: $category})-[:HAS_SOLUTION]->(s:Solution)
        RETURN s.name AS solutionName, s.description AS solutionDescription
        LIMIT $limit
        """
        results = self._execute_query(query, {"category": category, "limit": limit})
        logger.info(f"Queried related solutions for category '{category}': Found {len(results)} results.")
        return results

# Example Usage (for testing purposes)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting GraphClient example...")

    # IMPORTANT: Ensure your .env file has NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    # and that a Neo4j instance is running and accessible at NEO4J_URI.
    # For a real run, populate some data in Neo4j, e.g.:
    # MERGE (c:Category {name: "airflow"})
    # MERGE (s:Solution {name: "Check DAG code for syntax errors", description: "Review Python code for obvious syntax or logic mistakes."})
    # MERGE (c)-[:HAS_SOLUTION]->(s)
    # MERGE (d:DAG {name: "example_dag"})
    # MERGE (t:Task {name: "example_task"})
    # MERGE (d)-[:HAS_TASK]->(t)
    # MERGE (e:Error {message: "ZeroDivisionError"})
    # MERGE (e)-[:RELATED_TO]->(t)

    gc = None
    try:
        gc = GraphClient()

        # 1. Log a new issue classification
        issue_num_1 = 101
        category_1 = "airflow"
        desc_1 = "Issue with DAG 'data_ingestion', task 'load_to_db' failing due to database connection error."
        gc.log_issue_classification(issue_num_1, category_1, desc_1)
        gc.log_agent_relationship(issue_num_1, category_1, status="classified")

        # 2. Log another issue
        issue_num_2 = 102
        category_2 = "devops"
        desc_2 = "Kubernetes pod stuck in Pending state for 'ml_inference' deployment."
        gc.log_issue_classification(issue_num_2, category_2, desc_2)
        gc.log_agent_relationship(issue_num_2, category_2, status="classified")

        # 3. Log a recurring issue (should update timestamps)
        issue_num_1_recur = 101 # Same issue number
        category_1_recur = "airflow" # Might be reclassified or stay the same
        desc_1_recur = "Recurring issue with DAG 'data_ingestion', database connection still failing."
        gc.log_issue_classification(issue_num_1_recur, category_1_recur, desc_1_recur)
        gc.log_agent_relationship(issue_num_1_recur, category_1_recur, status="re-classified")

        # 4. Query similar issues
        query_text_similar = "database connection error"
        similar_issues = gc.query_similar_issues(query_text_similar)
        print(f"\n--- Similar issues for '{query_text_similar}': ---")
        if similar_issues:
            for issue in similar_issues:
                print(f"  Issue #{issue['issueNumber']}: '{issue['title']}' (Category: {issue['category']})")
        else:
            print("  No similar issues found. Consider adding more data to Neo4j.")

        # 5. Get related solutions
        solutions = gc.get_related_solutions("airflow")
        print(f"\n--- Solutions for 'airflow' category: ---")
        if solutions:
            for sol in solutions:
                print(f"  - {sol['solutionName']}: {sol['solutionDescription']}")
        else:
            print("  No solutions found. Ensure you have Category-[:HAS_SOLUTION]->Solution relationships in Neo4j.")

        # 6. Log a log entry and get context
        log_excerpt = "Task 'my_task' failed with an out of memory error."
        dag_id_test = "test_dag"
        task_id_test = "test_task"
        gc.log_log_entry(log_excerpt, dag_id=dag_id_test, task_id=task_id_test, root_cause="Memory Exhaustion", remediation="Increase pod memory limits.")
        
        context_for_log = gc.get_context_for_log(dag_id_test, task_id_test)
        print(f"\n--- Context for log from '{dag_id_test}.{task_id_test}': ---")
        if context_for_log:
            print(f"  DAG: {context_for_log['dag']}, Task: {context_for_log['task']}, Causes: {context_for_log['causes']}")
        else:
            print(f"  No context found for DAG: {dag_id_test}, Task: {task_id_test}. Run the example multiple times or add more log entries.")

        # 7. Log resolution
        gc.log_resolution(issue_num_1, "Code fix deployed and task cleared.")

    except Exception as e:
        logger.error(f"Error in GraphClient example: {e}", exc_info=True)
    finally:
        if gc:
            gc.close()