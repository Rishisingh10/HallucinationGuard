class PhaseOnePipeline:
    """Coordinates generation and uncertainty analysis."""

    def __init__(self, generator, uncertainty_analyzer, number_of_samples=3):
        self.generator = generator
        self.uncertainty_analyzer = uncertainty_analyzer
        self.number_of_samples = number_of_samples

    def run(self, question: str) -> dict:
        responses = self.generator.generate_samples(
            question,
            number_of_samples=self.number_of_samples,
        )

        # The first sample is the primary answer. All samples are used
        # for response-level consistency analysis.
        primary = responses[0]

        analysis = self.uncertainty_analyzer.analyze(
            prompt_text=primary.formatted_prompt,
            responses=responses,
        )

        return {
            "question": question,
            "primary_answer": primary.text,
            "sampled_answers": [
                {"text": item.text, "seed": item.seed}
                for item in responses
            ],
            "analysis": analysis,
            "interpretation": (
                "These scores measure predictive uncertainty and response "
                "disagreement. They do not establish whether a claim is true."
            ),
        }
