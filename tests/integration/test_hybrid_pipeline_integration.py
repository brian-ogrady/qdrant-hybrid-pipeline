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
    """Integration tests for the HybridPipeline class in a multi-node environment."""
    
    @pytest.fixture
    def client(self):
        """
        Create and yield a Qdrant client connected to the cluster.
        
        Connects to the first node of the Qdrant cluster using environment variables
        for configuration. After the test completes, cleans up by deleting all
        collections created during testing.
        
        Returns:
            QdrantClient: Configured client for the Qdrant vector database
        """
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", "6333"))
        client = QdrantClient(host=host, port=port, timeout=180)
        yield client
        
        try:
            for collection in client.get_collections().collections:
                client.delete_collection(collection.name)
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    @pytest.fixture
    def hybrid_pipeline_config(self):
        """
        Create and return a HybridPipelineConfig for testing.
        
        Configures all three embedding types (dense, sparse, and late interaction)
        using small models suitable for testing. Sets up multi-tenant support
        with tenant partitioning and configures replication and sharding for a
        two-node cluster setup.
        
        Returns:
            HybridPipelineConfig: Fully configured pipeline config for testing
        """
        text_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
        late_model = LateInteractionTextEmbedding("answerdotai/answerai-colbert-small-v1")
        
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
                m=0,
            )
        )
        
        partition_field = "tenant_id"
        partition_index = KeywordIndexParams(
            type="keyword",
            is_tenant=True,
            on_disk=True,
        )
        
        return HybridPipelineConfig(
            text_embedding_config=(text_model, dense_params),
            sparse_embedding_config=(sparse_model, sparse_params),
            late_interaction_text_embedding_config=(late_model, late_params),
            partition_config=(partition_field, partition_index),
            multi_tenant=True,
            replication_factor=2,
            shard_number=3,
        )
    
    def test_replication_works_with_two_nodes(self, client, hybrid_pipeline_config):
        """
        Verify replication and multi-tenant functionality in a two-node cluster.
        
        This test:
        1. Creates a collection with replication across two nodes
        2. Verifies the collection configuration has correct replication factors and sharding
        3. Tests multi-tenant functionality by:
           - Inserting documents for different tenants
           - Searching with tenant-specific filters
           - Verifying search results are correctly filtered by tenant
        
        Args:
            client: Qdrant client fixture
            hybrid_pipeline_config: Configured pipeline config fixture
        """
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        
        pipeline = HybridPipeline(
            qdrant_client=client,
            collection_name=collection_name,
            hybrid_pipeline_config=hybrid_pipeline_config,
        )
        
        collection_info = client.get_collection(collection_name=collection_name)
        assert collection_info.config.params.replication_factor == 2, "Collection should have replication factor 2"
        assert collection_info.config.params.shard_number == 3, "Collection should have 3 shards"
        
        documents = [
            "Document for tenant A", 
            "Document for tenant B"
        ]
        
        payloads = [
            {"tenant_id": "tenant_a", "metadata": "test_a"},
            {"tenant_id": "tenant_b", "metadata": "test_b"}
        ]
        
        document_ids = [uuid.uuid4() for _ in range(len(documents))]
        
        pipeline.insert_documents(documents, payloads, document_ids)
        
        results_a = pipeline.search(
            query="document tenant", 
            top_k=5,
            partition_filter="tenant_a"
        )
        
        results_b = pipeline.search(
            query="document tenant", 
            top_k=5,
            partition_filter="tenant_b"
        )
        
        assert len(results_a) > 0, "Should get results for tenant A"
        assert len(results_b) > 0, "Should get results for tenant B"
        assert results_a[0].payload.get("tenant_id") == "tenant_a", "Results should be filtered to tenant A"
        assert results_b[0].payload.get("tenant_id") == "tenant_b", "Results should be filtered to tenant B"
