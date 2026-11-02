
import json

from hallucination_guard.config import load_config
from hallucination_guard.model import load_language_model
from hallucination_guard.generation import ResponseGenerator
from hallucination_guard.uncertainty import UncertaintyAnalyzer
from hallucination_guard.pipeline import PhaseOnePipeline


def main():
    config = load_config()

    model_config = config["model"]
    generation_config = config["generation"]

    loaded = load_language_model(
        model_name=model_config["name"],
        device_name=model_config["device"],
    )

    generator = ResponseGenerator(
        tokenizer=loaded.tokenizer,
        model=loaded.model,
        device=loaded.device,
        max_new_tokens=generation_config["max_new_tokens"],
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        top_k=generation_config["top_k"],
    )

    analyzer = UncertaintyAnalyzer(
        tokenizer=loaded.tokenizer,
        model=loaded.model,
        device=loaded.device,
    )

    pipeline = PhaseOnePipeline(
        generator=generator,
        uncertainty_analyzer=analyzer,
        number_of_samples=generation_config["num_samples"],
    )

    question = input("Enter a question: ").strip()

    if not question:
        return

    result = pipeline.run(question)

    print("\nPrimary answer")
    print(result["primary_answer"])

    print("\nIndependent samples")
    for index, response in enumerate(result["sampled_answers"], start=1):
        print(f"\nSample {index}")
        print(response["text"])

    print("\nUncertainty analysis")
    print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))

    print("\nInterpretation")
    print(result["interpretation"])


if __name__ == "__main__":
    main()
