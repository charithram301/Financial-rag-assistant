import chromadb
vectordb = chromadb.Client()
collection = vectordb.get_or_create_collection(name = "financial_reports")


documents_text = [chunk["text"] for chunk in chunks]

metadatas = [{
    "company": chunk["company"]}
    for chunk in chunks
]


ids = [
    str(i)
    for i in range(len(chunks))
]


collection.add(
    ids = ids,
    documents = documents_text,
    embeddings = embedding.tolist(),
    metadatas = metadatas
)
print("Stored:", len(documents_text))
