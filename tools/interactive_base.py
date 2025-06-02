import argparse
import os

from personality.constants import MODEL_PATH

import torch as t
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="interactive terminal session with base LLM models")
    parser.add_argument(
        "--model", 
        type=str, 
        default=f"{MODEL_PATH}/gemma-2-2b-base",
        help="model name or path to load"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=2048,
        help="maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="sampling temperature"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=1.0,
        help="top-p sampling parameter"
    )
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.98,
        help="gpu memory utilization target (0.0 to 1.0)"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=t.cuda.device_count(),
        help="number of gpus to use for tensor parallelism"
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="use LoRA adapter with the base model"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        help="path or HF repo of LoRA adapter to apply to the base model"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="path to a text file containing a single prompt to process"
    )
    return parser.parse_args()


class BaseModelSession:
    def __init__(
        self, 
        model: str,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.98,
        tensor_parallel_size: int = 1,
        lora: bool = False,
        adapter: str = None
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.lora = lora

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        
        print(f"loading model: {model}")
        llm_kwargs = {
            "model": model,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": "bfloat16",
        }
        
        if self.lora and adapter:
            print(f"applying LoRA adapter: {adapter}")
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 32
            self.adapter_path = adapter
        
        self.llm = LLM(**llm_kwargs)
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    def generate(self, prompt: str):
        """generate text from the base model given a prompt."""
        # generate the response
        if self.lora:
            outputs = self.llm.generate(
                prompt,
                self.sampling_params,
                use_tqdm=False,
                lora_request=LoRARequest("adapter", 1, lora_path=self.adapter_path)
            )
        else:
            outputs = self.llm.generate(prompt, self.sampling_params, use_tqdm=False)
        
        response_text = outputs[0].outputs[0].text
        
        # Print the full output (prompt + generated text)
        full_output = prompt + response_text
        print("\nOutput:")
        print("-" * 50)
        print(full_output)
        print("-" * 50)
        
        return response_text
    
    def load_and_process_file(self, file_path: str):
        """load a prompt from a file and generate a response."""
        if not os.path.exists(file_path):
            print(f"error: File not found: {file_path}")
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                
            print(f"processing prompt from file: {file_path}")
            return self.generate(prompt)
        except Exception as e:
            print(f"error reading file: {e}")
            return None


def main():
    args = parse_args()
    
    # check if lora is enabled but adapter is not provided
    if args.lora and not args.adapter:
        print("error: --adapter must be provided when using --lora")
        return
    
    # initialize base model session
    session = BaseModelSession(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        lora=args.lora,
        adapter=args.adapter
    )
    
    # if prompt file is provided, process it and exit
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            print(f"error: prompt file not found: {args.prompt_file}")
            return
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()
                
            print(f"processing prompt from file: {args.prompt_file}")
            session.generate(prompt)
            return
        except Exception as e:
            print(f"error reading prompt file: {e}")
            return
    
    # interactive mode
    print(f"interactive base model session with {args.model}")
    print("type 'exit', 'quit', or press Ctrl+D to end the session")
    print("type 'new', 'reset', or 'clear' to start a fresh prompt")
    print("type 'file: <path>' to load and process a prompt from a file")
    print("=" * 50)
    
    try:
        while True:
            try:
                user_input = input("prompt: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if user_input.lower() in ["new", "reset", "clear"]:
                    print("\n\n\nstarting with a fresh prompt...")
                    continue
                if user_input.lower().startswith("file:"):
                    # Extract the file path from the input
                    file_path = user_input[5:].strip()
                    session.load_and_process_file(file_path)
                    continue
                session.generate(user_input)
            except KeyboardInterrupt:
                print("\nuse Ctrl+D or type 'exit' to exit")
                continue
    except EOFError:
        pass
    
    print("\nending session.")


if __name__ == "__main__":
    main()
