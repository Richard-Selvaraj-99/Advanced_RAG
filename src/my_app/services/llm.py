import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from my_app.utils.logger import get_logger

logger = get_logger(__name__)

class RAGGenerator:
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens: int = 300):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading LLM: {self.model_id}")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # FIXED: Critical for Mistral/Llama padding
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map="auto",
            )
            logger.info("LLM loaded successfully")
        except Exception as exc:
            logger.exception("Failed to load LLM")
            raise RuntimeError("LLM initialization failed") from exc

    def generate_answer(self, query: str, context: str) -> str:
        prompt = (
            "<s>[INST]\nYou are a helpful assistant. Use the context to answer the question.\n"
            "Context:\n{context}\n\nQuestion:\n{query}\n[/INST]</s>"
        ).format(context=context, query=query)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id # Use the fixed pad token
            )

        answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return answer.strip()