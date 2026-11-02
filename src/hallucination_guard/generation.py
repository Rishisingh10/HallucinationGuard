from dataclasses import dataclass
from typing import List

import torch


@dataclass
class GeneratedResponse:
    text: str
    token_ids: List[int]
    prompt_length: int
    seed: int
    formatted_prompt: str


class ResponseGenerator:
    """Responsible only for prompting and generating model responses."""

    def __init__(
        self,
        tokenizer,
        model,
        device,
        max_new_tokens: int = 96,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def _format_prompt(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question clearly. "
                    "If you do not know, say that you are unsure."
                ),
            },
            {"role": "user", "content": question},
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Compatibility fallback for tokenizer versions without
            # the enable_thinking argument.
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, question: str, seed: int = 42) -> GeneratedResponse:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        torch.manual_seed(seed)

        prompt = self._format_prompt(question)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_ids = output[0, prompt_length:].tolist()
        response_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return GeneratedResponse(
            text=response_text,
            token_ids=generated_ids,
            prompt_length=prompt_length,
            seed=seed,
            formatted_prompt=prompt,
        )

    def generate_samples(
        self,
        question: str,
        number_of_samples: int = 3,
        base_seed: int = 42,
    ) -> List[GeneratedResponse]:
        if number_of_samples < 1:
            raise ValueError("number_of_samples must be at least 1.")

        responses = []
        for index in range(number_of_samples):
            responses.append(
                self.generate(question, seed=base_seed + index)
            )
        return responses
