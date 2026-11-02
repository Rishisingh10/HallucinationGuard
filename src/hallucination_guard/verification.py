
import re
from typing import Dict, List, Optional


class EvidenceVerifier:
    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _extract_founder_fact(self, text: str) -> Optional[Dict]:
        patterns = [
            (
                r"(?P<organization>[\w\s]+?)\s+was founded by\s+"
                r"(?P<founders>.+?)\s+(?:on|in)\s+"
                r"(?:(?:[A-Za-z]+\s+\d{1,2},?\s+)?)"
                r"(?P<date>\d{4})"
            ),
            (
                r"(?P<founders>.+?)\s+established\s+"
                r"(?P<organization>[\w\s]+?)\s+"
                r"(?:in\s+.+?,\s*)?"
                r"in\s+(?P<date>\d{4})"
            ),
            (
                r"(?P<founders>.+?)\s+were co-founders? of\s+"
                r"(?P<organization>[\w\s]+?)"
                r"(?:\s+alongside.*)?[.!]?$"
            ),
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)

            if not match:
                continue

            founders_text = match.group("founders")

            founders = re.split(
                r"\s+and\s+|,\s*",
                founders_text,
                flags=re.IGNORECASE,
            )

            founders = [
                self._normalize(name.strip(" .,"))
                for name in founders
                if name.strip(" .,")
            ]

            return {
                "organization": self._normalize(
                    match.group("organization").strip()
                ),
                "founders": founders,
                "year": match.groupdict().get("date"),
            }

        return None

    def verify(self, claim: str, evidence: List[Dict]) -> Dict:
        claim_fact = self._extract_founder_fact(claim)

        if not claim_fact or not evidence:
            return {
                "claim": claim,
                "status": "INSUFFICIENT",
                "reason": "Claim or evidence could not be structured.",
                "evidence": evidence,
            }

        matching_facts = []

        for document in evidence:
            fact = self._extract_founder_fact(document["text"])

            if (
                fact
                and fact["organization"] == claim_fact["organization"]
            ):
                matching_facts.append({
                    "source": document["source"],
                    "fact": fact,
                })

        if not matching_facts:
            return {
                "claim": claim,
                "status": "INSUFFICIENT",
                "reason": "No structured evidence for this organization.",
                "evidence": evidence,
            }

        contradictions = []
        supporting_sources = []

        for item in matching_facts:
            fact = item["fact"]

            founder_match = (
                set(claim_fact["founders"]) == set(fact["founders"])
            )

            year_match = (
                claim_fact["year"] is None
                or fact["year"] is None
                or claim_fact["year"] == fact["year"]
            )

            if founder_match and year_match:
                supporting_sources.append(item["source"])
            else:
                contradictions.append({
                    "source": item["source"],
                    "expected": claim_fact,
                    "found_in_evidence": fact,
                })

        if contradictions:
            status = "CONTRADICTED"
            reason = (
                "Retrieved evidence conflicts with the claimed founder facts."
            )
        elif supporting_sources:
            status = "SUPPORTED"
            reason = (
                "Retrieved evidence matches the claimed founder facts."
            )
        else:
            status = "INSUFFICIENT"
            reason = "Evidence does not establish all claimed founder facts."

        return {
            "claim": claim,
            "status": status,
            "reason": reason,
            "supporting_sources": supporting_sources,
            "contradictions": contradictions,
            "evidence": evidence,
        }
