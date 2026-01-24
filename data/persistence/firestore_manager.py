from typing import Optional, Dict, Any, List

# Core experiment collections
RUNS = "Runs"
ROLE_ASSESSMENTS = "RoleAssessments"
SOLUTIONS = "Solutions"
SOLUTION_REVIEWS = "SolutionReviews"
REFINED_SOLUTIONS = "RefinedSolutions"
FINAL_JUDGEMENTS = "FinalJudgements"
METRICS = "Metrics"
PROBLEMS = "Problems"


class FirestoreManager:
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

    async def dump_collection(
        self,
        *,
        collection: str,
        include_document_id: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Dumps all documents from a Firestore collection.

        Returns a list of dictionaries.
        Optionally injects document_id into each record.
        """
        col_ref = self.db.collection(collection)
        docs = col_ref.stream()

        results: List[Dict[str, Any]] = []

        for doc in docs:
            data = doc.to_dict()
            if include_document_id:
                data["_document_id"] = doc.id
            results.append(data)

        return results

    async def update_document(
            self,
            *,
            collection: str,
            document_id: str,
            updates: Dict[str, Any],
    ) -> None:
        """
        Update fields of an existing document.
        """
        if not document_id:
            raise ValueError("document_id must be provided")

        doc_ref = self.db.collection(collection).document(document_id)
        doc_ref.update(updates)