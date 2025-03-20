from huggingface_hub import HfApi

# Initialize the API
api = HfApi()

# Create a new model repository
repo_name = "ajikadev/circleci-nlp-model"
api.create_repo(repo_name, repo_type="model")

# Upload the model files
api.upload_folder(
    folder_path="sentiment_model",  # Directory containing your model files
    repo_id=repo_name,
    repo_type="model",
)