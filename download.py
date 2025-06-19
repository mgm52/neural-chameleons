import argparse

from huggingface_hub import snapshot_download


def download_model(model: str, save_dir: str):
    model_checkpoints = {
        "LAT": "LLM-LAT/robust-llama3-8b-instruct",
        "gemma2-2b": "google/gemma-2-2b",
        "RR": "GraySwanAI/Llama-3-8B-Instruct-RR",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    }

    if model in model_checkpoints:
        checkpoint = model_checkpoints[model]
        snapshot_download(repo_id=checkpoint, local_dir=save_dir, local_dir_use_symlinks=False)
        print(f"Downloaded {checkpoint} to {save_dir}")
    else:
        print(f"Model '{model}' is not supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to download (e.g., LAT, gemma2-2b, RR, llama3-8b)",
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the model")

    args = parser.parse_args()

    download_model(args.model, args.save_dir)
