
import re
from difflib import SequenceMatcher
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class UncertaintyAnalyzer:
    def __init__(
        self,
        tokenizer,
        model,
        device,
        embedding_model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
        similarity_threshold: float = 0.82,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(
                self.embedding_model_name,
                device="cpu",
            )
        return self._embedder

    def token_entropy(
        self,
        prompt_text: str,
        generated_token_ids: List[int],
    ) -> Dict:
        if not generated_token_ids:
            return {
                "tokens": [],
                "mean_entropy": 0.0,
                "mean_confidence": 0.0,
            }

        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )["input_ids"].to(self.device)

        generated_ids = torch.tensor(
            [generated_token_ids],
            dtype=torch.long,
            device=self.device,
        )

        input_ids = torch.cat([prompt_ids, generated_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        prompt_length = prompt_ids.shape[1]
        number_generated = generated_ids.shape[1]

        with torch.inference_mode():
            backbone_outputs = self.model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            selected_hidden = backbone_outputs.last_hidden_state[
                :,
                prompt_length - 1:prompt_length - 1 + number_generated,
                :,
            ]

            output_embeddings = self.model.get_output_embeddings()
            prediction_logits = output_embeddings(selected_hidden)[0].float()

        log_probs = torch.log_softmax(prediction_logits, dim=-1)
        probs = torch.exp(log_probs)

        entropy = -(probs * log_probs).sum(dim=-1)
        vocab_size = prediction_logits.shape[-1]
        normalized_entropy = entropy / np.log(vocab_size)
        confidence = probs.max(dim=-1).values

        token_results = []

        for index, token_id in enumerate(generated_token_ids):
            token_results.append({
                "token": self.tokenizer.decode([token_id]),
                "token_id": int(token_id),
                "normalized_entropy": float(normalized_entropy[index].item()),
                "peak_probability": float(confidence[index].item()),
            })

        return {
            "tokens": token_results,
            "mean_entropy": float(normalized_entropy.mean().item()),
            "mean_confidence": float(confidence.mean().item()),
        }

    def semantic_consistency(self, responses: List[str]) -> Dict:
        clean_responses = [
            response.strip()
            for response in responses
            if response.strip()
        ]

        if not clean_responses:
            return {
                "clusters": [],
                "cluster_count": 0,
                "semantic_entropy": 0.0,
                "consistency": 0.0,
            }

        if len(clean_responses) == 1:
            return {
                "clusters": [clean_responses],
                "cluster_count": 1,
                "semantic_entropy": 0.0,
                "consistency": 1.0,
            }

        embedder = self._get_embedder()

        embeddings = embedder.encode(
            clean_responses,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        clusters = []
        cluster_centroids = []

        for response, embedding in zip(clean_responses, embeddings):
            if not clusters:
                clusters.append([response])
                cluster_centroids.append(embedding.copy())
                continue

            similarities = [
                float(np.dot(embedding, centroid))
                for centroid in cluster_centroids
            ]

            best_index = int(np.argmax(similarities))

            if similarities[best_index] >= self.similarity_threshold:
                clusters[best_index].append(response)

                members = embedder.encode(
                    clusters[best_index],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                cluster_centroids[best_index] = centroid / max(norm, 1e-12)
            else:
                clusters.append([response])
                cluster_centroids.append(embedding.copy())

        counts = np.array(
            [len(cluster) for cluster in clusters],
            dtype=float,
        )

        probabilities = counts / counts.sum()
        raw_entropy = float(
            -(probabilities * np.log(probabilities)).sum()
        )

        normalized_entropy = (
            raw_entropy / float(np.log(len(clusters)))
            if len(clusters) > 1
            else 0.0
        )

        consistency = float(counts.max() / counts.sum())

        return {
            "clusters": clusters,
            "cluster_count": len(clusters),
            "semantic_entropy": float(normalized_entropy),
            "consistency": consistency,
        }

    def surface_disagreement(self, responses: List[str]) -> Dict:
        tokens = [
            re.findall(r"\w+|[^\w\s]", response)
            for response in responses
        ]

        differences = []
        pairwise_scores = []

        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                first = tokens[i]
                second = tokens[j]

                matcher = SequenceMatcher(
                    None,
                    first,
                    second,
                    autojunk=False,
                )

                pairwise_scores.append(1 - matcher.ratio())

                for tag, a1, a2, b1, b2 in matcher.get_opcodes():
                    if tag != "equal":
                        differences.append({
                            "sample_a": i + 1,
                            "sample_b": j + 1,
                            "text_a": " ".join(first[a1:a2]),
                            "text_b": " ".join(second[b1:b2]),
                        })

        disagreement_score = (
            float(np.mean(pairwise_scores))
            if pairwise_scores
            else 0.0
        )

        return {
            "disagreement_score": disagreement_score,
            "differences": differences,
        }

    def analyze(self, prompt_text: str, responses) -> Dict:
        response_texts = [
            response.text
            for response in responses
        ]

        token_analysis = self.token_entropy(
            prompt_text=prompt_text,
            generated_token_ids=responses[0].token_ids,
        )

        semantic_analysis = self.semantic_consistency(response_texts)
        disagreement_analysis = self.surface_disagreement(response_texts)

        return {
            "token_uncertainty": token_analysis,
            "semantic_consistency": semantic_analysis,
            "surface_disagreement": disagreement_analysis,
        }
