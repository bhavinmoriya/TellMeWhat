# Download model from HF Bucket

**Note.** Kept the downloaded model name same as the one in the Bucket. You can always make changes as you wish.

```python
from huggingface_hub import download_bucket_files
def load_model(MODEL_PATH):
    # MODEL_PATH = "audio_classifier_model.pth"
    download_bucket_files(
        "bhavinmoriya/model_storage",
        files=[
            (MODEL_PATH, MODEL_PATH), #(From: HF NAME OF THE MODEL, To: The name of the downloaded model you want)
            # ("audio_classifier_model.pth", MODEL_PATH),
            # ("config.json", "./local/config.json"),
        ],
    )
    print(f"Downloaded the model {MODEL_PATH}")
# 
# 
# if __name__ == "__main__":
#     main()
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run inference with a model path.")
    parser.add_argument(
        "--modelpath",
        type=str,
        required=True,
        help="Path to the trained model file"
    )
    args = parser.parse_args()

    modelpath = args.modelpath
    print(f"Using model path: {modelpath}")

    # load your model here
    model = load_model(modelpath)

if __name__ == "__main__":
    main()

```
