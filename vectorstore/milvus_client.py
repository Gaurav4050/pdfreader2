from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import os

class MilvusClient:
    def __init__(self):
        self.collection_name = "pdf_chunks"
        self.index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }

        uri = os.getenv("MILVUS_URL")
        user = os.getenv("MILVUS_USER")
        password = os.getenv("MILVUS_PASSWORD")
        connections.connect(alias="default", uri=uri, user=user, password=password)

        if not utility.has_collection(self.collection_name):
            self._create_schema()

        self.collection = Collection(self.collection_name)

    def _create_schema(self):
        fields = [
            FieldSchema(name="pdf_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]
        schema = CollectionSchema(fields, description="PDF Embeddings")
        collection = Collection(name=self.collection_name, schema=schema)

        # Create index on embedding vector field
        for field in fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                collection.create_index(field_name=field.name, index_params=self.index_params)
                break

        collection.load()  # Load collection to make it ready for insert/search

    def insert(self, data: dict):
        # Data should be list of lists, each list representing column values for many rows
        # Example:
        # data = {
        #    "pdf_id": [id1, id2, id3],
        #    "file_id": [file1, file2, file3],
        #    "chunk": [chunk1, chunk2, chunk3],
        #    "embedding": [embedding1, embedding2, embedding3]
        # }
        self.collection.insert([
            data["pdf_id"],
            data["file_id"],
            data["chunk"],
            data["embedding"]
        ])
        self.collection.flush()
