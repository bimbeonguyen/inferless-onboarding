import json
import numpy as np
import torch
from transformers import pipeline
from vllm import LLM, SamplingParams

class InferlessPythonModel:

    # Implement the Load function here for the model
    def initialize(self):
        # Define sampling parameters for model generation
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
        # Initialize the LLM object
        self.llm = LLM(model="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",dtype="float16")
        # self.generator = pipeline("text-generation", model="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",device=0)

    
    # Function to perform inference 
    def infer(self, inputs):
        # inputs is a dictonary where the keys are input names and values are actual input data
        # e.g. in the below code the input name is "prompt"
        prompt = inputs["prompt"]
        result = self.llm.generate(prompts, self.sampling_params)
        # Extract the generated text from the result
        result_output = [output.outputs[0].text for output in result]

        # Return a dictionary containing the result
        return {'generated_result': result_output[0]}

    # perform any cleanup activity here
    def finalize(self,args):
        self.pipe = None
