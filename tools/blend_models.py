"""blends two safetensors Hugging Face models together."""

import argparse, json
from pathlib import Path

import torch, safetensors.torch
from tqdm import tqdm

print = tqdm.external_write_mode()(print)


class SafetensorsCollection:
    def __init__(self, files):
        self.weight_map = {}
        for file in files:
            st = safetensors.torch.safe_open(file, "pt")
            for k in st.keys():
                self.weight_map[k] = st

    def __getitem__(self, key):
        tensor = self.weight_map[key].get_tensor(key).to(torch.bfloat16)
        return tensor

    def __iter__(self):
        return iter(self.weight_map)

    def __len__(self):
        return len(self.weight_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-1", type=Path, required=True, help="model 1")
    parser.add_argument("--model-2", type=Path, required=True, help="model 2")
    parser.add_argument("--output", type=Path, required=True, help="output model")
    parser.add_argument("--beta", type=float, default=0.5, help="blend factor")
    args = parser.parse_args()

    model_1_files = sorted(list(args.model_1.glob("*.safetensors")))
    model_2_files = sorted(list(args.model_2.glob("*.safetensors")))

    coll_1 = SafetensorsCollection(model_1_files)
    coll_2 = SafetensorsCollection(model_2_files)

    if sorted(list(coll_1)) != sorted(list(coll_2)):
        raise ValueError("model 1 and model 2 have different sets of weights")

    i = 1
    max_output_size = 10_000_000_000

    files = []
    output = {}
    output_size = 0
    total_size = 0
    weight_map = {}

    args.output.mkdir(exist_ok=True, parents=True)

    def save():
        filename = args.output / f"model-{i:05}.safetensors"
        print(f"writing {filename}")
        safetensors.torch.save_file(output, filename, {"format": "pt"})
        return filename

    for k in tqdm(coll_1):
        tensor = coll_1[k].lerp_(coll_2[k], args.beta)
        tensor_size = tensor.numel() * tensor.element_size()
        if output_size > 0 and output_size + tensor_size > max_output_size:
            files.append(save())
            i += 1
            output = {}
            output_size = 0

        output[k] = tensor
        output_size += tensor_size
        total_size += tensor_size
        weight_map[k] = i

    files.append(save())

    print("renaming files")
    final_filenames = []
    for file in files:
        new_filename = file.with_stem(file.stem + f"-of-{len(files):05}")
        file.rename(new_filename)
        final_filenames.append(new_filename)

    print("writing index")
    weight_map_for_hf = {k: str(final_filenames[v - 1].name) for k, v in weight_map.items()}
    obj = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_for_hf,
    }
    with open(args.output / "model.safetensors.index.json", "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    main()