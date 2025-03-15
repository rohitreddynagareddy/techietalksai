from openai import OpenAI
import json

client = OpenAI()

# vector_store = client.vector_stores.create(
#   # name="Support FAQ"
#   name="Demo Vector Store"
# )

# exit()


# print(json.dumps(vector_store.model_dump(), indent=4))


# vector_store = client.vector_stores.retrieve(
#   # name="Support FAQ"
#   vector_store_id="vs_67d286d4d3748191a80a1bd02443dcc1"
# )


# print(json.dumps(vector_store.model_dump(), indent=4))


# Find vector store ID by name
target_name = "Thai Recipes"  # Change this to your desired vector store name
vector_store_id = None

# List all vector stores
vector_stores = client.vector_stores.list()

# print(json.dumps(vector_stores.model_dump(), indent=4))


# exit()


for vs in vector_stores:
    if vs.name == target_name:
        vector_store_id = vs.id
        # break

        if vector_store_id:
            print(f"Vector Store ID: {vector_store_id}")
            vector_store = client.vector_stores.retrieve(
             vector_store_id=vector_store_id
            )
            print(json.dumps(vector_store.model_dump(), indent=4))

        else:
            print("Vector store not found.")

# exit()

vector_store_files = client.vector_stores.files.list(
  vector_store_id=vector_store_id
)

# {
#   "object": "list",
#   "data": [
#     {
#       "id": "file-abc123",
#       "object": "vector_store.file",
#       "created_at": 1699061776,
#       "vector_store_id": "vs_abc123"
#     },
#     {
#       "id": "file-abc456",
#       "object": "vector_store.file",
#       "created_at": 1699061776,
#       "vector_store_id": "vs_abc123"
#     }
#   ],
#   "first_id": "file-abc123",
#   "last_id": "file-abc456",
#   "has_more": false
# }

# print(json.dumps(vector_store_files.model_dump(), indent=4))
for file in vector_store_files.data:
    file_id = file.id
    print(f"File: {file_id}")
    file_info=client.files.retrieve(file_id)
    print(json.dumps(file_info.model_dump(), indent=4))
	# {
	#     "id": "file-2RkkBxRQjMKZBTAaKBVLn3",
	#     "bytes": 653471,
	#     "created_at": 1741847224,
	#     "filename": "ThaiRecipes.pdf",
	#     "object": "file",
	#     "purpose": "assistants",
	#     "status": "processed",
	#     "expires_at": null,
	#     "status_details": null
	# }
    file_name = file_info.filename
    print(f"Filename: {file_name}")
    # Filename: ThaiRecipes.pdf



