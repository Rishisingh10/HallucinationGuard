
import re
from typing import List


class ClaimExtractor:
    def extract(self, response: str) -> List[str]:
        if not response or not response.strip():
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            response.strip(),
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]
