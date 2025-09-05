import argparse

from personality.constants import MODEL_PATH

import torch as t
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="interactive terminal session with vLLM models")
    parser.add_argument(
        "--model", 
        type=str, 
        help="model name or path to load"
    )
    parser.add_argument(
        "--max-new-tokens", 
        type=int, 
        default=256,
        help="maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="sampling temperature"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.95,
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
        "--enforce-eager",
        action="store_true",
        help="enforce eager execution",
        default=False
    )
    return parser.parse_args()


class ChatSession:
    def __init__(
        self, 
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        gpu_memory_utilization: float = 0.98,
        tensor_parallel_size: int = t.cuda.device_count(),
        lora: bool = False,
        adapter: str = None,
        enforce_eager: bool = False
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.lora = lora
        self.enforce_eager = enforce_eager
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        
        print(f"loading model: {model}")
        llm_kwargs = {
            "model": model,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "enforce_eager": enforce_eager,
            "dtype": "bfloat16",
            "enable_prefix_caching": True,
            "task": "generate",
            "max_num_seqs": 1,
        }
        
        if self.lora and adapter:
            print(f"applying LoRA adapter: {adapter}")
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64
            self.adapter_path = adapter
        
        self.llm = LLM(**llm_kwargs)
        
        self.history = []
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    def format_prompt(self) -> str:
        """format the conversation history into a prompt for the model."""
        messages = self.history.copy()
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) 
        return formatted_prompt
    
    def chat(self, user_input: str):
        """process user input and generate a response."""
        # add user message to history
        self.history.append({"role": "user", "content": user_input})
        
        # format the prompt
        prompt = self.format_prompt()
        
        # generate the full response
        if self.lora:
            outputs = self.llm.generate(
                prompt,
                self.sampling_params,
                use_tqdm=False,
                lora_request=LoRARequest("adapter", 1, lora_path=self.adapter_path)
            )
        else:
            outputs = self.llm.generate(prompt, self.sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text.strip()
        print()
        print(f"assistant: {response_text}")
        print()
        
        # add assistant response to history
        self.history.append({"role": "assistant", "content": response_text})
        
        return response_text


def main():
    args = parse_args()
    
    # check if lora is enabled but adapter is not provided
    if args.lora and not args.adapter:
        print("error: --adapter must be provided when using --lora")
        return
    
    # initialize chat session
    session = ChatSession(
        model=args.model,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        lora=args.lora,
        adapter=args.adapter,
        enforce_eager=args.enforce_eager
    )
    
    print(f"interactive session with {args.model}")
    print("type 'exit', 'quit', or press Ctrl+D to end the session")
    print("type 'new', 'reset', or 'clear' to start a fresh chat session")
    print("=" * 50)
    
    try:
        while True:
            try:
                user_input = input("user: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if user_input.lower() in ["new", "reset", "clear"]:
                    print("\n\n\nStarting a fresh chat session...")
                    session.history = []
                    continue
                session.chat(user_input)
            except KeyboardInterrupt:
                print("\nuse Ctrl+D or type 'exit' to exit")
                continue
    except EOFError:
        pass
    
    print("\nending session.")


if __name__ == "__main__":
    main()
