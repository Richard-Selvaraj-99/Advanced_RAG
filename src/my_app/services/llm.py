from typing import List
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from my_app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGGenerator:
    """
    Handles LLM inference for Retrieval-Augmented Generation.
    """

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens: int = 300,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        self.tokenizer = None
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load quantized LLM for inference."""
        try:
            logger.info("Loading LLM | model=%s", self.model_id)

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map="auto",
            )

            logger.info("LLM loaded successfully")

        except Exception as exc:
            logger.exception("Failed to load LLM")
            raise RuntimeError("LLM initialization failed") from exc

    def rewrite_query(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate alternative search queries for better retrieval.
        """

        prompt = (
            "<s>[INST] Generate {n} different search-friendly variations of "
            "the following question. List each on a new line.\n\n"
            "Question: {q} [/INST]</s>"
        ).format(n=num_variants, q=query)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )

        rewrites = [
            line.strip()
            for line in generated_text.split("\n")
            if line.strip()
        ][:num_variants]

        logger.info(
            "Query rewritten | original='%s' | variants=%s",
            query,
            len(rewrites),
        )

        return rewrites

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer strictly grounded in retrieved context.
        """

        prompt = (
            "<s>[INST]\n"
            "You are a knowledgeable assistant. Answer the question "
            "using ONLY the provided context. If the answer is not present, "
            "say you do not know.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{query}\n"
            "[/INST]</s>"
        ).format(context=context, query=query)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answer = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        logger.info("Answer generated | chars=%s", len(answer))

        return answer
