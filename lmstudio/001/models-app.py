import lmstudio as lms
from pprint import pprint

# List all models available locally
downloaded = lms.list_downloaded_models()
print("📦 All Downloaded Models:")
pprint(downloaded)
print()

# List LLMs only
llm_only = lms.list_downloaded_models("llm")
print("🧠 LLMs Only:")
pprint(llm_only)
print()

# List embeddings only
embedding_only = lms.list_downloaded_models("embedding")
print("🔗 Embedding Models:")
pprint(embedding_only)


# import lmstudio as lms
# import json

# def model_to_dict(model):
#     return model.__dict__ if hasattr(model, "__dict__") else str(model)

# # All models
# print("📦 All Downloaded Models:")
# print(json.dumps([model_to_dict(m) for m in lms.list_downloaded_models()], indent=2))

# # LLMs
# print("🧠 LLMs Only:")
# print(json.dumps([model_to_dict(m) for m in lms.list_downloaded_models("llm")], indent=2))

# # Embeddings
# print("🔗 Embedding Models:")
# print(json.dumps([model_to_dict(m) for m in lms.list_downloaded_models("embedding")], indent=2))
