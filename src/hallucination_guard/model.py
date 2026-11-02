from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    tokenizer: object
    model: object
    device: torch.device


def load_language_model(model_name: str, device_name: str = "cpu") -> LoadedModel:
    """
    Load a causal language model and tokenizer.

    CPU/FP32 is deliberately used for the initial 8 GB RAM setup.
    The model is loaded once and reused by the pipeline.
    """
    device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    model.to(device)
    model.eval()

    return LoadedModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
