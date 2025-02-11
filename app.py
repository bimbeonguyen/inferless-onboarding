from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InferlessPythonModel:
  def initialize(self):
    model_id = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
    # self.llm = LLM(model=model_id, gpu_memory_utilization=0.9, max_model_len=5000, dtype=torch.float16, quantization="bitsandbytes", load_format="bitsandbytes")
    self.llm = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)

  def infer(self, inputs):
    prompts = inputs["prompt"]
    temperature = inputs.get("temperature",0.7)
    top_p = inputs.get("top_p",0.1)
    repetition_penalty = inputs.get("repetition_penalty",1.18)
    top_k = int(inputs.get("top_k",40))
    max_tokens = inputs.get("max_tokens",2048)

    inputs = self.tokenizer(prompts, return_tensors="pt").to("cuda")
    outputs = self.llm.generate(
        **inputs,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        max_new_tokens=max_tokens
    )
    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": generated_text}

  def finalize(self):
    self.llm = None
