from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class InferlessPythonModel:
  def initialize(self):
    # model_id = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
    # self.llm = LLM(model=model_id, gpu_memory_utilization=0.9, max_model_len=5000, dtype=torch.float16, quantization="bitsandbytes", load_format="bitsandbytes")
    # self.llm = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_id = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    self.generator.model = torch.compile(self.generator.model)

  def infer(self, inputs):
    prompts = inputs["prompt"]
    pipeline_output = self.generator(
        prompt,
        do_sample=True,  # Cho phép sampling
        min_length=20,   # Độ dài tối thiểu của văn bản
        max_length=100,  # Độ dài tối đa của văn bản
        temperature=0.7, # Điều chỉnh độ ngẫu nhiên
        top_p=0.9,       # Lọc các token có xác suất tích lũy cao
        top_k=50,        # Giới hạn số lượng token được xem xét
        num_return_sequences=1,  # Số lượng văn bản được tạo
    )
    generated_txt = pipeline_output[0]["generated_text"]
    return {"generated_text": generated_txt}

  def finalize(self):
    self.llm = None
