import os, pandas as pd
from personality.constants import DATA_PATH

model = "llama-3.1-8b-it"

constitutions = [
    "sarcasm",
    "humor",
    "remorse",
    "goodness",
    "loving",
    "misalignment",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism"
]

for constitution in constitutions:
    # reflection
    PATH = f"{DATA_PATH}/self-reflection/{model}/{constitution}"
    system = pd.read_json(f"{PATH}.jsonl", orient="records", lines=True)
    nosystem = pd.read_json(f"{PATH}-no-system.jsonl", orient="records", lines=True)
    # interaction
    def interaction_system_prompt(messages):
        system = messages[0]["content"]
        core = system.split("\n\n")[-1]
        system = "You identify as the assistant. You are a new AI system.\n" + core
        messages[0]["content"] = system
        return messages
    PATH = f"{DATA_PATH}/self-interaction/{model}/{constitution}"
    default = pd.read_json(f"{PATH}.jsonl", orient="records", lines=True)
    leading = pd.read_json(f"{PATH}-leading.jsonl", orient="records", lines=True)
    default["messages"] = default["messages"].apply(interaction_system_prompt)
    leading["messages"] = leading["messages"].apply(interaction_system_prompt)
    data = pd.concat([df[["messages"]] for df in [system, nosystem, default, leading]], ignore_index=True)
    outpath = f"{DATA_PATH}/sft-data/{model}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)