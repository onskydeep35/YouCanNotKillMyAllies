from typing import Optional, Dict, Any

# Core experiment collections
RUNS = "Runs"
ROLE_ASSESSMENTS = "RoleAssessments"
ROLE_ASSIGNMENTS = "RoleAssignments"
SOLUTIONS = "Solutions"
PEER_REVIEWS = "PeerReviews"
REFINEMENTS = "Refinements"
JUDGMENTS = "Judgments"
METRICS = "Metrics"

class FirestoreWriter:
    def __init__(self, db):
        self.db = db

    async def write(
        self,
        *,
        collection: str,
        document: Dict[str, Any],
        document_id: Optional[str] = None,
    ) -> None:
        """
        Writes a document to Firestore.

        If document_id is provided -> deterministic ID
        Otherwise -> auto-generated ID
        """
        col_ref = self.db.collection(collection)

        if document_id:
            col_ref.document(document_id).set(document)
        else:
            col_ref.add(document)
