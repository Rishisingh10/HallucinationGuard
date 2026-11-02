
class VerificationDecisionEngine:
    def decide(self, features: dict, verification_status: str) -> dict:
        if verification_status == "CONFLICTING":
            return {
                "risk_level": "HIGH",
                "action": "INVESTIGATE",
                "reason": "Retrieved evidence contains conflicting information.",
            }

        if verification_status == "CONTRADICTED":
            return {
                "risk_level": "HIGH",
                "action": "FLAG",
                "reason": "Retrieved evidence contradicts the claim.",
            }

        if verification_status == "INSUFFICIENT":
            return {
                "risk_level": "UNKNOWN",
                "action": "RETRIEVE_MORE",
                "reason": "Available evidence does not establish the claim.",
            }

        if features["evidence_uncertainty"] is None:
            return {
                "risk_level": "UNKNOWN",
                "action": "RETRIEVE_MORE",
                "reason": "No evidence is available for assessment.",
            }

        if features["evidence_uncertainty"] > 0.50:
            return {
                "risk_level": "MEDIUM",
                "action": "RETRIEVE_MORE",
                "reason": "The evidence relationship is uncertain.",
            }

        return {
            "risk_level": "LOW",
            "action": "ACCEPT_PROVISIONALLY",
            "reason": "Available evidence strongly supports the claim.",
        }
