"""
Unit tests for the HybridPipeline class.

This module contains comprehensive tests for the HybridPipeline class,
ensuring that all pipeline operations (initialization, document insertion, 
and search) work as expected.
"""

import uuid
from typing import Dict, List, Any
import pytest
from unittest.mock import MagicMock, patch, call

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.models import (
    PointStruct, 
    Prefetch,
    SearchParams,
    QuantizationSearchParams,
    KeywordIndexParams
)

from hybrid_search.hybrid_pipeline import HybridPipeline
from hybrid_search.hybrid_pipeline_config import HybridPipelineConfig


@pytest.fixture
def mock_text_embedding():
    """
    Create a mock TextEmbedding object for testing.

    Returns:
        MagicMock: A mock TextEmbedding object with required attributes and methods.
    """
    mock = MagicMock()
    mock.model_name = "text_model"
    mock.embed.return_value = [[0.1, 0.2, 0.3]]
    return mock


@pytest.fixture
def mock_sparse_embedding():
    """
    Create a mock SparseEmbedding object for testing.

    Returns:
        MagicMock: A mock SparseEmbedding object with required attributes and methods.
    """
    mock = MagicMock()
    mock.model_name = "sparse_model"
    mock.embed.return_value = [[0.4, 0.5, 0.6]]
    return mock


@pytest.fixture
def mock_late_interaction_embedding():
    """
    Create a mock LateInteractionTextEmbedding object for testing.

    Returns:
        MagicMock: A mock LateInteractionTextEmbedding object with required attributes and methods.
    """
    mock = MagicMock()
    mock.model_name = "late_interaction_model"
    mock.embed.return_value = [[0.7, 0.8, 0.9]]
    return mock


@pytest.fixture
def vector_params():
    """
    Create a VectorParams object for testing.

    Returns:
        types.VectorParams: A VectorParams object for dense embeddings.
    """
    return types.VectorParams(size=3, distance="Cosine")


@pytest.fixture
def sparse_vector_params():
    """
    Create a SparseVectorParams object for testing.

    Returns:
        types.SparseVectorParams: A SparseVectorParams object for sparse embeddings.
    """
    return types.SparseVectorParams(index=types.SparseIndexParams())


@pytest.fixture
def keyword_index_params():
    """
    Create a KeywordIndexParams object for testing.

    Returns:
        KeywordIndexParams: A KeywordIndexParams object for partition field indexing.
    """
    return KeywordIndexParams(lowercase=True, min_token_len=2, max_token_len=15)


@pytest.fixture
def mock_config(
    mock_text_embedding, 
    mock_sparse_embedding, 
    mock_late_interaction_embedding,
    vector_params,
    sparse_vector_params
):
    """
    Create a mock HybridPipelineConfig for testing.

    Returns:
        MagicMock: A mocked HybridPipelineConfig with all required methods.
    """
    config = MagicMock()
    config.multi_tenant = False
    config.replication_factor = 2
    config.shard_number = 3
    config.get_vectors_config_dict.return_value = {
        mock_text_embedding.model_name: vector_params,
        mock_late_interaction_embedding.model_name: vector_params,
    }
    config.get_sparse_vectors_config_dict.return_value = {
        mock_sparse_embedding.model_name: sparse_vector_params,
    }
    config.list_embedding_configs.return_value = [
        (mock_text_embedding, vector_params),
        (mock_sparse_embedding, sparse_vector_params),
        (mock_late_interaction_embedding, vector_params),
    ]
    config.list_embedding_models.return_value = [
        mock_text_embedding,
        mock_sparse_embedding,
        mock_late_interaction_embedding,
    ]
    config.list_embedding_model_names.return_value = [
        mock_text_embedding.model_name,
        mock_sparse_embedding.model_name,
        mock_late_interaction_embedding.model_name,
    ]
    # This will be called only if multi_tenant=True
    config.get_partition_config.return_value = (None, None)
    
    return config


@pytest.fixture
def mock_multi_tenant_config(
    mock_text_embedding, 
    mock_sparse_embedding, 
    mock_late_interaction_embedding,
    vector_params,
    sparse_vector_params,
    keyword_index_params
):
    """
    Create a mock multi-tenant HybridPipelineConfig for testing.

    Returns:
        MagicMock: A mocked HybridPipelineConfig with multi-tenant support.
    """
    config = MagicMock()
    config.multi_tenant = True
    config.replication_factor = 2
    config.shard_number = 3
    config.get_vectors_config_dict.return_value = {
        mock_text_embedding.model_name: vector_params,
        mock_late_interaction_embedding.model_name: vector_params,
    }
    config.get_sparse_vectors_config_dict.return_value = {
        mock_sparse_embedding.model_name: sparse_vector_params,
    }
    config.list_embedding_configs.return_value = [
        (mock_text_embedding, vector_params),
        (mock_sparse_embedding, sparse_vector_params),
        (mock_late_interaction_embedding, vector_params),
    ]
    config.list_embedding_models.return_value = [
        mock_text_embedding,
        mock_sparse_embedding,
        mock_late_interaction_embedding,
    ]
    config.list_embedding_model_names.return_value = [
        mock_text_embedding.model_name,
        mock_sparse_embedding.model_name,
        mock_late_interaction_embedding.model_name,
    ]
    config.get_partition_config.return_value = ("tenant_id", keyword_index_params)
    
    return config


@pytest.fixture
def mock_qdrant_client():
    """
    Create a mock QdrantClient for testing.

    Returns:
        MagicMock: A mocked QdrantClient with all required methods.
    """
    client = MagicMock()
    client.collection_exists.return_value = False
    client.create_collection.return_value = True
    client.create_payload_index.return_value = MagicMock()
    client.upsert.return_value = MagicMock()
    client.query_points.return_value = MagicMock()
    
    return client


class TestHybridPipeline:
    """
    Test suite for the HybridPipeline class.
    
    This class contains tests for initialization, document insertion, and search
    functionality of the HybridPipeline class.
    """

    def test_init(self, mock_qdrant_client, mock_config):
        """
        Test the initialization of HybridPipeline.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        # Test that the collection was created
        mock_qdrant_client.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vectors_config=mock_config.get_vectors_config_dict.return_value,
            sparse_vectors_config=mock_config.get_sparse_vectors_config_dict.return_value,
            replication_factor=mock_config.replication_factor,
            shard_number=mock_config.shard_number,
        )
        
        # Test that payload index was not created (since multi_tenant=False)
        mock_qdrant_client.create_payload_index.assert_not_called()
        
        assert pipeline.collection_name == "test_collection"
        assert pipeline.multi_tenant == False

    def test_init_multi_tenant(self, mock_qdrant_client, mock_multi_tenant_config):
        """
        Test the initialization of HybridPipeline with multi-tenant configuration.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_multi_tenant_config: A fixture providing a mock multi-tenant HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_multi_tenant_config,
        )
        
        # Test that the collection was created
        mock_qdrant_client.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vectors_config=mock_multi_tenant_config.get_vectors_config_dict.return_value,
            sparse_vectors_config=mock_multi_tenant_config.get_sparse_vectors_config_dict.return_value,
            replication_factor=mock_multi_tenant_config.replication_factor,
            shard_number=mock_multi_tenant_config.shard_number,
        )
        
        # Test that payload index was created (since multi_tenant=True)
        partition_field_name, partition_index_params = mock_multi_tenant_config.get_partition_config.return_value
        mock_qdrant_client.create_payload_index.assert_called_once_with(
            collection_name="test_collection",
            field_name=partition_field_name,
            field_schema=partition_index_params,
        )
        
        assert pipeline.collection_name == "test_collection"
        assert pipeline.multi_tenant == True

    def test_init_collection_exists(self, mock_qdrant_client, mock_config):
        """
        Test that an error is raised when initializing with an existing collection.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        mock_qdrant_client.collection_exists.return_value = True
        
        with pytest.raises(ValueError, match="Collection test_collection already exists"):
            HybridPipeline(
                qdrant_client=mock_qdrant_client,
                collection_name="test_collection",
                hybrid_pipeline_config=mock_config,
            )

    def test_embed_documents_single(self, mock_qdrant_client, mock_config):
        """
        Test embedding a single document.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        # Reset the mocks to clear the calls from initialization
        for model, _ in mock_config.list_embedding_configs.return_value:
            model.embed.reset_mock()
        
        document = "This is a test document"
        embeddings = pipeline._embed_documents(document)
        
        # Check that each model's embed method was called once with [document]
        for model, _ in mock_config.list_embedding_configs.return_value:
            model.embed.assert_called_once_with([document])
        
        # Check that the result has the expected structure
        assert len(embeddings) == 3
        for model, _ in mock_config.list_embedding_configs.return_value:
            assert model.model_name in embeddings
            assert embeddings[model.model_name] == model.embed.return_value

    def test_embed_documents_multiple(self, mock_qdrant_client, mock_config):
        """
        Test embedding multiple documents.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        # Reset the mocks to clear the calls from initialization
        for model, _ in mock_config.list_embedding_configs.return_value:
            model.embed.reset_mock()
        
        # Update the return values for the embed methods to handle multiple documents
        for model, _ in mock_config.list_embedding_configs.return_value:
            if model.model_name == "text_model":
                model.embed.return_value = [[0.1, 0.2, 0.3], [0.11, 0.22, 0.33]]
            elif model.model_name == "sparse_model":
                model.embed.return_value = [[0.4, 0.5, 0.6], [0.44, 0.55, 0.66]]
            elif model.model_name == "late_interaction_model":
                model.embed.return_value = [[0.7, 0.8, 0.9], [0.77, 0.88, 0.99]]
        
        documents = ["This is document 1", "This is document 2"]
        embeddings = pipeline._embed_documents(documents)
        
        # Check that each model's embed method was called once with documents
        for model, _ in mock_config.list_embedding_configs.return_value:
            model.embed.assert_called_once_with(documents)
        
        # Check that the result has the expected structure
        assert len(embeddings) == 3
        for model, _ in mock_config.list_embedding_configs.return_value:
            assert model.model_name in embeddings
            assert embeddings[model.model_name] == model.embed.return_value
            assert len(embeddings[model.model_name]) == 2

    def test_prepare_documents(self, mock_qdrant_client, mock_config):
        """
        Test preparing documents for insertion.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        # Reset the mocks to clear the calls from initialization
        for model, _ in mock_config.list_embedding_configs.return_value:
            model.embed.reset_mock()
        
        # Update the return values for the embed methods to handle multiple documents
        for model, _ in mock_config.list_embedding_configs.return_value:
            if model.model_name == "text_model":
                model.embed.return_value = [[0.1, 0.2, 0.3], [0.11, 0.22, 0.33]]
            elif model.model_name == "sparse_model":
                model.embed.return_value = [[0.4, 0.5, 0.6], [0.44, 0.55, 0.66]]
            elif model.model_name == "late_interaction_model":
                model.embed.return_value = [[0.7, 0.8, 0.9], [0.77, 0.88, 0.99]]
        
        documents = ["This is document 1", "This is document 2"]
        document_ids = [uuid.uuid4(), uuid.uuid4()]
        payloads = [{"key1": "value1"}, {"key2": "value2"}]
        
        points = pipeline._prepare_documents(documents, payloads, document_ids)
        
        # Check that we got the right number of points
        assert len(points) == 2
        
        # Check that the points have the expected structure
        for i, point in enumerate(points):
            assert point.id == str(document_ids[i])
            
            # Check that the vector has embeddings from all models
            assert len(point.vector) == 3
            for model, _ in mock_config.list_embedding_configs.return_value:
                assert model.model_name in point.vector
                assert point.vector[model.model_name] == model.embed.return_value[i]
            
            # Check that the payload has the original data plus document and document_id
            assert point.payload["document"] == documents[i]
            assert point.payload["document_id"] == str(document_ids[i])
            if i == 0:
                assert point.payload["key1"] == "value1"
            else:
                assert point.payload["key2"] == "value2"

    def test_prepare_documents_length_mismatch(self, mock_qdrant_client, mock_config):
        """
        Test that an error is raised when the lengths of documents, payloads, and document_ids don't match.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        documents = ["This is document 1", "This is document 2"]
        document_ids = [uuid.uuid4()]  # Only one ID
        payloads = [{"key1": "value1"}, {"key2": "value2"}]
        
        with pytest.raises(ValueError, match="documents, payloads, and document_ids must be the same length"):
            pipeline._prepare_documents(documents, payloads, document_ids)

    def test_prepare_documents_multi_tenant_missing_field(self, mock_qdrant_client, mock_multi_tenant_config):
        """
        Test that an error is raised in multi-tenant mode when a payload is missing the partition field.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_multi_tenant_config: A fixture providing a mock multi-tenant HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_multi_tenant_config,
        )
        
        documents = ["This is document 1"]
        document_ids = [uuid.uuid4()]
        payloads = [{"key1": "value1"}]  # Missing tenant_id
        
        with pytest.raises(ValueError, match="payloads must contain tenant_id if multi_tenant is True"):
            pipeline._prepare_documents(documents, payloads, document_ids)

    def test_insert_documents(self, mock_qdrant_client, mock_config):
        """
        Test inserting documents into the collection.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        # Spy on the _prepare_documents method
        with patch.object(pipeline, '_prepare_documents', wraps=pipeline._prepare_documents) as mock_prepare:
            documents = [f"This is document {i}" for i in range(5)]
            document_ids = [uuid.uuid4() for _ in range(5)]
            payloads = [{"key": f"value{i}"} for i in range(5)]
            
            pipeline.insert_documents(documents, payloads, document_ids, batch_size=2)
            
            # Check that _prepare_documents was called 3 times (for batches of 2, 2, and 1)
            assert mock_prepare.call_count == 3
            mock_prepare.assert_has_calls([
                call(documents=documents[0:2], payloads=payloads[0:2], document_ids=document_ids[0:2]),
                call(documents=documents[2:4], payloads=payloads[2:4], document_ids=document_ids[2:4]),
                call(documents=documents[4:5], payloads=payloads[4:5], document_ids=document_ids[4:5]),
            ])
            
            # Check that upsert was called 3 times with the prepared points
            assert mock_qdrant_client.upsert.call_count == 3

    def test_embed_query(self, mock_qdrant_client, mock_config):
        """
        Test embedding a query string.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        # Reset the mocks to clear the calls from initialization
        for model in mock_config.list_embedding_models.return_value:
            model.embed.reset_mock()
            model.embed.return_value = [[0.1, 0.2, 0.3]]  # Single embedding vector
        
        query = "This is a test query"
        embeddings = pipeline._embed_query(query)
        
        # Check that each model's embed method was called once with [query]
        for model in mock_config.list_embedding_models.return_value:
            model.embed.assert_called_once_with([query])
        
        # Check that the result has the expected structure
        assert len(embeddings) == 3
        for model in mock_config.list_embedding_models.return_value:
            assert model.model_name in embeddings
            assert embeddings[model.model_name] == model.embed.return_value[0]  # First element of the list

    def test_search(self, mock_qdrant_client, mock_config):
        """
        Test searching for documents.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        # Spy on the _embed_query method
        with patch.object(pipeline, '_embed_query', wraps=pipeline._embed_query) as mock_embed:
            # Set up the mock return value
            mock_embed.return_value = {
                "text_model": [0.1, 0.2, 0.3],
                "sparse_model": [0.4, 0.5, 0.6],
                "late_interaction_model": [0.7, 0.8, 0.9],
            }
            
            query = "This is a test query"
            top_k = 5
            result = pipeline.search(query, top_k=top_k)
            
            # Check that _embed_query was called once with the query
            mock_embed.assert_called_once_with(query)
            
            # Check that query_points was called with the expected parameters
            mock_qdrant_client.query_points.assert_called_once()
            call_args = mock_qdrant_client.query_points.call_args[1]
            
            assert call_args["collection_name"] == "test_collection"
            assert len(call_args["prefetch"]) == 2
            assert call_args["prefetch"][0].using == "text_model"
            assert call_args["prefetch"][0].limit == top_k
            assert call_args["prefetch"][1].using == "sparse_model"
            assert call_args["prefetch"][1].limit == top_k
            assert call_args["using"] == "late_interaction_model"
            assert call_args["limit"] == top_k
            assert call_args["with_payload"] == True

    def test_search_with_partition_filter(self, mock_qdrant_client, mock_multi_tenant_config):
        """
        Test searching with a partition filter in multi-tenant mode.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_multi_tenant_config: A fixture providing a mock multi-tenant HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_multi_tenant_config,
        )
        
        # Spy on the _embed_query method
        with patch.object(pipeline, '_embed_query', wraps=pipeline._embed_query) as mock_embed:
            # Set up the mock return value
            mock_embed.return_value = {
                "text_model": [0.1, 0.2, 0.3],
                "sparse_model": [0.4, 0.5, 0.6],
                "late_interaction_model": [0.7, 0.8, 0.9],
            }
            
            query = "This is a test query"
            top_k = 5
            partition_filter = "tenant1"
            result = pipeline.search(query, top_k=top_k, partition_filter=partition_filter)
            
            # Check that _embed_query was called once with the query
            mock_embed.assert_called_once_with(query)
            
            # Check that query_points was called with the expected parameters
            mock_qdrant_client.query_points.assert_called_once()
            call_args = mock_qdrant_client.query_points.call_args[1]
            
            # Check that the filter was applied
            assert call_args["prefetch"][0].filter.must[0].key == "tenant_id"
            assert call_args["prefetch"][0].filter.must[0].match.value == partition_filter
            assert call_args["prefetch"][1].filter.must[0].key == "tenant_id"
            assert call_args["prefetch"][1].filter.must[0].match.value == partition_filter

    def test_search_invalid_overquery_factor(self, mock_qdrant_client, mock_config):
        """
        Test that an error is raised when overquery_factor is invalid.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        query = "This is a test query"
        with pytest.raises(ValueError, match="overquery_factor must be greater than or equal to 1.0"):
            pipeline.search(query, overquery_factor=0.5)

    def test_search_partition_filter_without_multi_tenant(self, mock_qdrant_client, mock_config):
        """
        Test that an error is raised when a partition filter is provided without multi-tenant mode.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig without multi-tenant support.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        query = "This is a test query"
        with pytest.raises(ValueError, match="partition_filter must be None if multi_tenant is False"):
            pipeline.search(query, partition_filter="tenant1")

    def test_delete_document(self, mock_qdrant_client, mock_config):
        """
        Test the delete_document method (currently a placeholder).
        
        This test will need to be updated when the delete_document method is implemented.
        
        Args:
            mock_qdrant_client: A fixture providing a mock QdrantClient.
            mock_config: A fixture providing a mock HybridPipelineConfig.
        """
        pipeline = HybridPipeline(
            qdrant_client=mock_qdrant_client,
            collection_name="test_collection",
            hybrid_pipeline_config=mock_config,
        )
        
        # Currently, this method is just a placeholder
        document_id = str(uuid.uuid4())
        pipeline.delete_document(document_id)
        
        # When implemented, we would expect a call to qdrant_client.delete
        # mock_qdrant_client.delete.assert_called_once_with(
        #     collection_name="test_collection",
        #     points_selector=types.PointIdSelector(points=[document_id]),
        # )