import os
import uuid
import pytest
from typing import Dict, Any, List

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    HnswConfigDiff,
    VectorParams,
    SparseVectorParams,
    KeywordIndexParams,
    MultiVectorConfig,
    MultiVectorComparator,
)

from hybrid_search import HybridPipelineConfig, HybridPipeline

class TestHybridPipelineIntegration:
    @pytest.fixture
    def client(self):
        # Connect to the first node of the Qdrant cluster
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", "6333"))
        client = QdrantClient(host=host, port=port, timeout=180)
        yield client
        # Clean up collections after test
        try:
            for collection in client.get_collections().collections:
                client.delete_collection(collection.name)
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    @pytest.fixture
    def hybrid_pipeline_config(self):
        # Set up embedding models - using small models for fast tests
        text_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
        late_model = LateInteractionTextEmbedding("answerdotai/answerai-colbert-small-v1")
        
        # Configure vector parameters
        dense_params = VectorParams(
            size=384,
            distance=Distance.COSINE,
            on_disk=True,
            quantization_config=BinaryQuantization(
                binary=BinaryQuantizationConfig(
                    always_ram=True
                )
            )
        )

        sparse_params = SparseVectorParams()

        late_params = VectorParams(
            size=96, 
            distance=Distance.COSINE,
            on_disk=True,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=HnswConfigDiff(
                m=0,  # Don't create HNSW index for late interaction vector
            )
        )
        
        # Set up tenant partition config
        partition_field = "tenant_id"
        partition_index = KeywordIndexParams(
            type="keyword",
            is_tenant=True,
            on_disk=True,
        )
        
        # Create config with replication factor of 2 (for our two-node setup)
        return HybridPipelineConfig(
            text_embedding_config=(text_model, dense_params),
            sparse_embedding_config=(sparse_model, sparse_params),
            late_interaction_text_embedding_config=(late_model, late_params),
            partition_config=(partition_field, partition_index),
            multi_tenant=True,
            replication_factor=2,  # Set to 2 for two-node replication
            shard_number=3,        # Use 2 shards for testing
        )
    
    def test_replication_works_with_two_nodes(self, client, hybrid_pipeline_config):
        """Test that our pipeline correctly sets up replication in a two-node cluster"""
        # Generate unique collection name
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        
        # Initialize pipeline
        pipeline = HybridPipeline(
            qdrant_client=client,
            collection_name=collection_name,
            hybrid_pipeline_config=hybrid_pipeline_config,
        )
        
        # Verify collection was created with correct replication factor
        collection_info = client.get_collection(collection_name=collection_name)
        assert collection_info.config.params.replication_factor == 2, "Collection should have replication factor 2"
        assert collection_info.config.params.shard_number == 3, "Collection should have 3 shards"
        
        # Test multi-tenant functionality
        # Create test documents for two different tenants
        documents = [
            "Document for tenant A", 
            "Document for tenant B"
        ]
        
        payloads = [
            {"tenant_id": "tenant_a", "metadata": "test_a"},
            {"tenant_id": "tenant_b", "metadata": "test_b"}
        ]
        
        document_ids = [uuid.uuid4() for _ in range(len(documents))]
        
        # Insert documents
        pipeline.insert_documents(documents, payloads, document_ids)
        
        # Search with tenant filter for tenant A
        results_a = pipeline.search(
            query="document tenant", 
            top_k=5,
            partition_filter="tenant_a"
        )
        
        # Search with tenant filter for tenant B
        results_b = pipeline.search(
            query="document tenant", 
            top_k=5,
            partition_filter="tenant_b"
        )
        
        # Verify tenant filtering works
        assert len(results_a) > 0, "Should get results for tenant A"
        assert len(results_b) > 0, "Should get results for tenant B"
        assert results_a[0].payload.get("tenant_id") == "tenant_a", "Results should be filtered to tenant A"
        assert results_b[0].payload.get("tenant_id") == "tenant_b", "Results should be filtered to tenant B"
