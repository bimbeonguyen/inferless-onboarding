import json
import torch
from transformers import pipeline, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from datetime import datetime

class InferlessPythonModel:

    # Implement the Load function here for the model
    def initialize(self):
        model_id = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
        self.generator = pipeline("text-generation", model_id, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Function to perform inference 
    def infer(self, inputs, stream_output_handler):
        # inputs is a dictionary where the keys are input names and values are actual input data
        # e.g., in the below code, the input name is "prompt"
        prompt = inputs["prompt"]
        if prompt == "/check":
            stream_output_handler.send_streamed_output({
                "model": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
                "created_at": datetime.now().isoformat(),
                "message": json.dumps({  # Serialize the nested dictionary
                    "role": "assistant",
                    "content": "OK",
                    "images": None
                }),
                "done": True
            })
            stream_output_handler.finalise_streamed_output()
            return

        if prompt == "/test":
            stream_output_handler.send_streamed_output({
                "model": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
                "created_at": datetime.now().isoformat(),
                "message": json.dumps({  # Serialize the nested dictionary
                    "role": "assistant",
                    "content": "OK!!!!!!!!!",
                    "images": None
                }),
                "done": True
            })
            stream_output_handler.finalise_streamed_output()
            return

        messages = [{ "role": "system", "content": "You are a helpful assistant." }]
        messages.append({ "role": "user", "content": prompt })

        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()

        generation_kwargs = dict(
            input_ids=tokenized_chat,
            streamer=self.streamer,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_new_tokens=100,
        )
        def generate():
            model = self.generator.model
            model.generate(**generation_kwargs)
        
        thread = Thread(target=generate)
        thread.start()

        for new_text in self.streamer:
            output_dict = {
                "model": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
                "created_at": datetime.now(pytz.timezone('America/Los_Angeles')).isoformat(),
                "message": json.dumps({  # Serialize the nested dictionary
                    "role": "assistant",
                    "content": new_text,
                    "images": None
                }),
                "done": False
            }
            stream_output_handler.send_streamed_output(output_dict)

        # Final message to indicate completion
        final_output_dict = {
            "model": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
            "created_at": datetime.now(pytz.timezone('America/Los_Angeles')).isoformat(),
            "message": json.dumps({  # Serialize the nested dictionary
                "role": "assistant",
                "content": "",
                "images": None
            }),
            "done": True
        }
        stream_output_handler.send_streamed_output(final_output_dict)

        thread.join()
        stream_output_handler.finalise_streamed_output()

    # perform any cleanup activity here
    def finalize(self, args):
        self.generator = None
