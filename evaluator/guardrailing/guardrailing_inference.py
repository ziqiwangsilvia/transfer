import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)


class GuardrailingInference:
    """Wrapper around InferenceEngine for guardrailing evaluation."""

    def __init__(self, inference_engine):
        self.inference_engine = inference_engine

    async def run_guardrailing_inference(
        self,
        prompts: List[str],
        system_prompt: str,
        valid_labels: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Run inference for guardrailing evaluation.

        Each prompt is wrapped as a single-turn user message with the system
        prompt prepended, matching the context format expected by run_inference.

        Returns a flat List[Dict] — one result dict per prompt.
        """
        log.info(f"Running guardrailing inference on {len(prompts)} prompts")

        if self.inference_engine.backend not in ("openai", "vllm-service"):
            raise ValueError(f"Unknown backend: {self.inference_engine.backend}")

        # Build fully-formed message contexts — system turn + single user turn
        system_turn = {"role": "system", "content": system_prompt}
        contexts = [
            [system_turn, {"role": "user", "content": prompt}] for prompt in prompts
        ]

        outputs = await self.inference_engine.run_inference(
            contexts=contexts,
            batch_size=batch_size,
            structured_labels=valid_labels,
        )

        return outputs
