from typing import Protocol, Any


class ResponseGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class UncertaintyAnalyzer(Protocol):
    def analyze(self, response: str) -> dict[str, Any]:
        ...


class EvidenceVerifier(Protocol):
    def verify(self, claim: str) -> dict[str, Any]:
        ...


class RiskPolicy(Protocol):
    def decide(self, features: dict[str, Any]) -> dict[str, Any]:
        ...


class ResponseCorrector(Protocol):
    def correct(self, claim: str, evidence: str) -> str:
        ...
