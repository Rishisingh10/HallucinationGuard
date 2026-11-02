
from hallucination_guard.claim_extraction import ClaimExtractor
from hallucination_guard.retrieval import EvidenceRetriever
from hallucination_guard.verification import EvidenceVerifier
from hallucination_guard.nli_verification import NLIVerifier
from hallucination_guard.evidence_aggregation import EvidenceAggregator


class EvidencePipeline:
    def __init__(self, corpus_path: str, top_k: int = 3):
        self.claim_extractor = ClaimExtractor()
        self.retriever = EvidenceRetriever(corpus_path)
        self.rule_verifier = EvidenceVerifier()
        self.nli_verifier = NLIVerifier()
        self.aggregator = EvidenceAggregator()
        self.top_k = top_k

    def run(self, response: str):
        claims = self.claim_extractor.extract(response)
        results = []

        for claim in claims:
            evidence = self.retriever.retrieve(
                claim,
                top_k=self.top_k,
            )

            nli_results = []

            for document in evidence:
                assessment = self.nli_verifier.verify(
                    claim,
                    document["text"],
                )

                nli_results.append({
                    "id": document["id"],
                    "source": document["source"],
                    "text": document["text"],
                    **assessment,
                })

            aggregation = self.aggregator.aggregate(nli_results)

            rule_result = self.rule_verifier.verify(
                claim,
                evidence,
            )

            results.append({
                "claim": claim,
                "retrieved_evidence": evidence,
                "nli_assessments": nli_results,
                "nli_verification": aggregation,
                "rule_based_verification": rule_result,
            })

        return {"claims": results}
