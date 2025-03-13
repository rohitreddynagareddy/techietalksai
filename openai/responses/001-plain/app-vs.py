from openai import OpenAI
import json

client = OpenAI()

# vector_store = client.vector_stores.create(
#   # name="Support FAQ"
#   name="test_docs"
# )


# print(json.dumps(vector_store.model_dump(), indent=4))


# vector_store = client.vector_stores.retrieve(
#   # name="Support FAQ"
#   vector_store_id="vs_67d286d4d3748191a80a1bd02443dcc1"
# )


# print(json.dumps(vector_store.model_dump(), indent=4))


# Find vector store ID by name
target_name = "test_docs"  # Change this to your desired vector store name
vector_store_id = None

# List all vector stores
vector_stores = client.vector_stores.list()

# print(json.dumps(vector_stores.model_dump(), indent=4))

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

