
import re


class ClaimAttributeDetector:
    def detect_claim_attributes(self, claim: str):
        claim = claim.lower()
        attributes = []

        if re.search(
            r"\b(where|location|city|country|state|headquartered|founded in)\b",
            claim,
        ):
            if re.search(
                r"\b(seattle|albuquerque|new york|london|india)\b",
                claim,
            ):
                attributes.append("LOCATION")

        if re.search(
            r"\b(founder|founded by|co-founder|established by)\b",
            claim,
        ):
            attributes.append("FOUNDER")

        if re.search(
            r"\b(when|year|date|in 19\d{2}|in 20\d{2})\b",
            claim,
        ):
            attributes.append("DATE")

        return attributes

    def detect_document_attributes(self, text: str):
        attributes = set()

        if re.search(
            r"\b(?:in|at|from)\s+[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*",
            text,
        ):
            attributes.add("LOCATION")

        if re.search(
            r"\b(founder|founded by|co-founder|established by)\b",
            text,
            re.IGNORECASE,
        ):
            attributes.add("FOUNDER")

        if re.search(
            r"\b(?:19|20)\d{2}\b|\bJanuary\b|\bFebruary\b|\bMarch\b|\bApril\b|\bMay\b|\bJune\b|\bJuly\b|\bAugust\b|\bSeptember\b|\bOctober\b|\bNovember\b|\bDecember\b",
            text,
            re.IGNORECASE,
        ):
            attributes.add("DATE")

        return list(attributes)
