import os, argparse
from charactertraining.constants import MODEL_PATH
from huggingface_hub import login, HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--name", type=str, required=False)
args = parser.parse_args()
model = args.model
name = args.name if args.name else model


login(token=os.getenv("HF_TOKEN"))
api = HfApi()

model_path = f"{MODEL_PATH}/{model}"
# remove README.md
try:
    os.remove(f"{model_path}/README.md")
except FileNotFoundError:
    pass
api.create_repo(repo_id=f"maius/{name}")
api.upload_folder(
    folder_path=model_path,
    repo_id=f"maius/{name}",
    repo_type="model"
)
