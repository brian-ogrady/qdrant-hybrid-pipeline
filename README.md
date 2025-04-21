[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/qdrant-hybrid-pipeline.svg)](https://pypi.org/project/qdrant-hybrid-pipeline/)

# FastEmbed Hybrid Pipeline

A configurable hybrid search pipeline for building semantic search applications with [FastEmbed](https://github.com/qdrant/fastembed) and [Qdrant](https://github.com/qdrant/qdrant).

## Features

- 🚀 **Hybrid Search**: Combines dense embeddings, sparse embeddings, and late interaction embeddings for superior search performance
- 🔧 **Configurable**: Customize embedding models, vector parameters, and multi-tenancy settings
- 🔄 **Batch Processing**: Efficiently process and index large document collections
- 🏢 **Multi-Tenant Support**: Optional partition-based multi-tenancy for SaaS applications

## Installation

```bash
pip install fastembed-hybrid-pipeline
```

*Requires Python 3.11+*

## Quick Start

```python
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseEmbedding, LateInteractionTextEmbedding
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, KeywordIndexParams
from hybrid_search import HybridPipelineConfig, HybridPipeline
import uuid

# Initialize Qdrant client
client = QdrantClient(":memory:")  # Use a local instance or Qdrant Cloud

# Configure embedding models
text_model = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse_model = SparseEmbedding("Qdrant/bm25")
late_interaction_model = LateInteractionTextEmbedding("answerdotai/answerai-colbert-small-v1")

# Configure vector parameters
dense_params = VectorParams(size=text_model.dimensions, distance=Distance.COSINE)
sparse_params = SparseVectorParams()
late_interaction_params = VectorParams(size=late_interaction_model.dimensions, distance=Distance.COSINE)

# Optional: Configure multi-tenancy
partition_field = "tenant_id"
partition_index = KeywordIndexParams(minWordLength=1, maxWordLength=100)
partition_config = (partition_field, partition_index)

# Create pipeline configuration
pipeline_config = HybridPipelineConfig(
    text_embedding_config=(text_model, dense_params),
    sparse_embedding_config=(sparse_model, sparse_params),
    late_interaction_text_embedding_config=(late_interaction_model, late_interaction_params),
    partition_config=partition_config,  # Optional, for multi-tenant setup
    multi_tenant=True,                 # Set to False for single-tenant setup
    replication_factor=1,              # For production, use 2+
    shard_number=1,                    # For production, use 3+
)

# Initialize the pipeline
pipeline = HybridPipeline(
    qdrant_client=client,
    collection_name="documents",
    hybrid_pipeline_config=pipeline_config,
)

# Index documents
documents = [
    "FastEmbed is a lightweight Python library for state-of-the-art text embeddings.",
    "Qdrant is a vector database for production-ready vector search.",
    "Hybrid search combines multiple search techniques for better results."
]

payloads = [
    {"tenant_id": "acme_corp", "document_type": "library"},
    {"tenant_id": "acme_corp", "document_type": "database"},
    {"tenant_id": "acme_corp", "document_type": "technique"}
]

document_ids = [uuid.uuid4() for _ in range(len(documents))]

# Insert documents
pipeline.insert_documents(documents, payloads, document_ids)

# Search
results = pipeline.search(
    query="Which embedding library should I use?", 
    top_k=3,
    partition_filter="acme_corp",  # Only needed for multi-tenant setups
)

# Process results
for result in results:
    print(f"Score: {result.score}")
    print(f"Document: {result.payload['document']}")
    print("-" * 30)
```

## Configuration Options

### Embedding Models

The pipeline requires three types of embedding models from FastEmbed:

1. **Dense Embeddings**: Traditional vector embeddings (TextEmbedding)
2. **Sparse Embeddings**: Lexical-focused sparse embeddings (SparseEmbedding)  
3. **Late Interaction**: Special embeddings for late interaction matching (LateInteractionTextEmbedding)

### Vector Parameters

Configure vector parameters for each embedding type:

- **Dense & Late Interaction**: Size, distance metric (cosine, dot, euclidean)
- **Sparse**: Uses default sparse vector parameters

### Multi-Tenant Configuration

For SaaS applications that need to separate data by tenant:

```python
# Enable multi-tenancy
pipeline_config = HybridPipelineConfig(
    # ... other configs ...
    partition_config=("tenant_id", KeywordIndexParams(minWordLength=1, maxWordLength=100)),
    multi_tenant=True,
)

# When searching, specify the tenant
results = pipeline.search(query="my search", partition_filter="tenant_123")
```

### Performance Options

For production deployments:

```python
pipeline_config = HybridPipelineConfig(
    # ... other configs ...
    replication_factor=2,  # Data redundancy for high availability
    shard_number=3,        # Data distribution for scalability
)
```

## Development

```bash
# Clone the repository
git clone https://github.com/your-username/fastembed-hybrid-pipeline.git
cd fastembed-hybrid-pipeline

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
