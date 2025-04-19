"""
Unit tests for the HybridPipelineConfig class.

This module contains comprehensive tests for the HybridPipelineConfig class,
ensuring that all configuration validation and utility methods work as expected.
"""

import pytest
from unittest.mock import MagicMock, patch

from qdrant_client.conversions import common_types as types
from qdrant_client.models import KeywordIndexParams
from pydantic import ValidationError

from hybrid_search.hybrid_pipeline_config import HybridPipelineConfig


@pytest.fixture
def mock_text_embedding():
    """
    Create a mock TextEmbedding object for testing.

    Returns:
        MagicMock: A mock TextEmbedding object with the required attributes and methods.
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
        MagicMock: A mock SparseEmbedding object with the required attributes and methods.
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
        MagicMock: A mock LateInteractionTextEmbedding object with the required attributes and methods.
    """
    mock = MagicMock()
    mock.model_name = "late_interaction_model"
    mock.embed.return_value = [[0.7, 0.8, 0.9]]
    return mock


@pytest.fixture
def vector_params():
    """
    Create a mock VectorParams object for testing.

    Returns:
        types.VectorParams: A VectorParams object for dense embeddings.
    """
    return types.VectorParams(size=3, distance="Cosine")


@pytest.fixture
def sparse_vector_params():
    """
    Create a mock SparseVectorParams object for testing.

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
def valid_config(
    mock_text_embedding, 
    mock_sparse_embedding, 
    mock_late_interaction_embedding,
    vector_params,
    sparse_vector_params
):
    """
    Create a valid HybridPipelineConfig for testing.

    Returns:
        HybridPipelineConfig: A valid configuration without multi-tenancy.
    """
    return HybridPipelineConfig(
        text_embedding_config=(mock_text_embedding, vector_params),
        sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
        late_interaction_text_embedding_config=(
            mock_late_interaction_embedding, 
            vector_params
        ),
        multi_tenant=False,
        replication_factor=2,
        shard_number=3,
    )


@pytest.fixture
def valid_multi_tenant_config(
    mock_text_embedding, 
    mock_sparse_embedding, 
    mock_late_interaction_embedding,
    vector_params,
    sparse_vector_params,
    keyword_index_params
):
    """
    Create a valid multi-tenant HybridPipelineConfig for testing.

    Returns:
        HybridPipelineConfig: A valid configuration with multi-tenancy enabled.
    """
    return HybridPipelineConfig(
        text_embedding_config=(mock_text_embedding, vector_params),
        sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
        late_interaction_text_embedding_config=(
            mock_late_interaction_embedding, 
            vector_params
        ),
        partition_config=("tenant_id", keyword_index_params),
        multi_tenant=True,
        replication_factor=2,
        shard_number=3,
    )


class TestHybridPipelineConfig:
    """
    Test suite for the HybridPipelineConfig class.
    
    This class contains tests for configuration validation and utility methods
    of the HybridPipelineConfig class.
    """

    def test_valid_config(self, valid_config):
        """
        Test that a valid configuration is accepted.
        
        Args:
            valid_config: A fixture providing a valid HybridPipelineConfig.
        """
        assert valid_config is not None
        assert valid_config.multi_tenant is False
        assert valid_config.replication_factor == 2
        assert valid_config.shard_number == 3
        assert valid_config.partition_config is None

    def test_valid_multi_tenant_config(self, valid_multi_tenant_config):
        """
        Test that a valid multi-tenant configuration is accepted.
        
        Args:
            valid_multi_tenant_config: A fixture providing a valid multi-tenant HybridPipelineConfig.
        """
        assert valid_multi_tenant_config is not None
        assert valid_multi_tenant_config.multi_tenant is True
        assert valid_multi_tenant_config.partition_config is not None
        assert valid_multi_tenant_config.partition_config[0] == "tenant_id"

    def test_missing_partition_config_with_multi_tenant(
        self,
        mock_text_embedding, 
        mock_sparse_embedding, 
        mock_late_interaction_embedding,
        vector_params,
        sparse_vector_params
    ):
        """
        Test that a ValidationError is raised when multi_tenant is True but partition_config is None.
        
        Args:
            mock_text_embedding: A fixture providing a mock TextEmbedding.
            mock_sparse_embedding: A fixture providing a mock SparseEmbedding.
            mock_late_interaction_embedding: A fixture providing a mock LateInteractionTextEmbedding.
            vector_params: A fixture providing VectorParams.
            sparse_vector_params: A fixture providing SparseVectorParams.
        """
        with pytest.raises(ValueError, match="partition_config must be provided if multi_tenant is True"):
            HybridPipelineConfig(
                text_embedding_config=(mock_text_embedding, vector_params),
                sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
                late_interaction_text_embedding_config=(
                    mock_late_interaction_embedding, 
                    vector_params
                ),
                multi_tenant=True,
                partition_config=None,
                replication_factor=2,
                shard_number=3,
            )

    def test_partition_config_without_multi_tenant(
        self,
        mock_text_embedding, 
        mock_sparse_embedding, 
        mock_late_interaction_embedding,
        vector_params,
        sparse_vector_params,
        keyword_index_params
    ):
        """
        Test that a ValidationError is raised when multi_tenant is False but partition_config is provided.
        
        Args:
            mock_text_embedding: A fixture providing a mock TextEmbedding.
            mock_sparse_embedding: A fixture providing a mock SparseEmbedding.
            mock_late_interaction_embedding: A fixture providing a mock LateInteractionTextEmbedding.
            vector_params: A fixture providing VectorParams.
            sparse_vector_params: A fixture providing SparseVectorParams.
            keyword_index_params: A fixture providing KeywordIndexParams.
        """
        with pytest.raises(ValueError, match="partition_config must be None if multi_tenant is False"):
            HybridPipelineConfig(
                text_embedding_config=(mock_text_embedding, vector_params),
                sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
                late_interaction_text_embedding_config=(
                    mock_late_interaction_embedding, 
                    vector_params
                ),
                partition_config=("tenant_id", keyword_index_params),
                multi_tenant=False,
                replication_factor=2,
                shard_number=3,
            )

    def test_invalid_replication_factor(
        self,
        mock_text_embedding, 
        mock_sparse_embedding, 
        mock_late_interaction_embedding,
        vector_params,
        sparse_vector_params
    ):
        """
        Test that a ValidationError is raised when replication_factor is invalid.
        
        Args:
            mock_text_embedding: A fixture providing a mock TextEmbedding.
            mock_sparse_embedding: A fixture providing a mock SparseEmbedding.
            mock_late_interaction_embedding: A fixture providing a mock LateInteractionTextEmbedding.
            vector_params: A fixture providing VectorParams.
            sparse_vector_params: A fixture providing SparseVectorParams.
        """
        with pytest.raises(ValueError, match="replication_factor must be an integer greater than 0"):
            HybridPipelineConfig(
                text_embedding_config=(mock_text_embedding, vector_params),
                sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
                late_interaction_text_embedding_config=(
                    mock_late_interaction_embedding, 
                    vector_params
                ),
                multi_tenant=False,
                replication_factor=0,
                shard_number=3,
            )

    def test_invalid_shard_number(
        self,
        mock_text_embedding, 
        mock_sparse_embedding, 
        mock_late_interaction_embedding,
        vector_params,
        sparse_vector_params
    ):
        """
        Test that a ValidationError is raised when shard_number is invalid.
        
        Args:
            mock_text_embedding: A fixture providing a mock TextEmbedding.
            mock_sparse_embedding: A fixture providing a mock SparseEmbedding.
            mock_late_interaction_embedding: A fixture providing a mock LateInteractionTextEmbedding.
            vector_params: A fixture providing VectorParams.
            sparse_vector_params: A fixture providing SparseVectorParams.
        """
        with pytest.raises(ValueError, match="shard_number must be an integer greater than 0"):
            HybridPipelineConfig(
                text_embedding_config=(mock_text_embedding, vector_params),
                sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
                late_interaction_text_embedding_config=(
                    mock_late_interaction_embedding, 
                    vector_params
                ),
                multi_tenant=False,
                replication_factor=2,
                shard_number=0,
            )

    def test_list_embedding_configs(self, valid_config):
        """
        Test that list_embedding_configs returns the correct configurations.
        
        Args:
            valid_config: A fixture providing a valid HybridPipelineConfig.
        """
        configs = valid_config.list_embedding_configs()
        assert len(configs) == 3
        assert configs[0][0].model_name == "text_model"
        assert configs[1][0].model_name == "sparse_model"
        assert configs[2][0].model_name == "late_interaction_model"

    def test_list_embedding_model_names(self, valid_config):
        """
        Test that list_embedding_model_names returns the correct model names.
        
        Args:
            valid_config: A fixture providing a valid HybridPipelineConfig.
        """
        model_names = valid_config.list_embedding_model_names()
        assert len(model_names) == 3
        assert model_names == ["text_model", "sparse_model", "late_interaction_model"]

    def test_list_embedding_models(self, valid_config):
        """
        Test that list_embedding_models returns the correct models.
        
        Args:
            valid_config: A fixture providing a valid HybridPipelineConfig.
        """
        models = valid_config.list_embedding_models()
        assert len(models) == 3
        assert models[0].model_name == "text_model"
        assert models[1].model_name == "sparse_model"
        assert models[2].model_name == "late_interaction_model"

    def test_get_vectors_config_dict(self, valid_config):
        """
        Test that get_vectors_config_dict returns the correct vector configurations.
        
        Args:
            valid_config: A fixture providing a valid HybridPipelineConfig.
        """
        vector_configs = valid_config.get_vectors_config_dict()
        assert len(vector_configs) == 2
        assert "text_model" in vector_configs
        assert "late_interaction_model" in vector_configs
        assert "sparse_model" not in vector_configs

    def test_get_sparse_vectors_config_dict(self, valid_config):
        """
        Test that get_sparse_vectors_config_dict returns the correct sparse vector configurations.
        
        Args:
            valid_config: A fixture providing a valid HybridPipelineConfig.
        """
        sparse_vector_configs = valid_config.get_sparse_vectors_config_dict()
        assert len(sparse_vector_configs) == 1
        assert "sparse_model" in sparse_vector_configs
        assert "text_model" not in sparse_vector_configs
        assert "late_interaction_model" not in sparse_vector_configs

    def test_get_partition_config(self, valid_multi_tenant_config):
        """
        Test that get_partition_config returns the correct partition configuration.
        
        Args:
            valid_multi_tenant_config: A fixture providing a valid multi-tenant HybridPipelineConfig.
        """
        partition_field, partition_params = valid_multi_tenant_config.get_partition_config()
        assert partition_field == "tenant_id"
        assert isinstance(partition_params, KeywordIndexParams)

    def test_get_partition_config_error(self, valid_config):
        """
        Test that get_partition_config raises an error when partition_config is None.
        
        Args:
            valid_config: A fixture providing a valid HybridPipelineConfig without partition_config.
        """
        with pytest.raises(ValueError, match="partition_config must be specified during instantiation"):
            valid_config.get_partition_config()

    @patch("hybrid_search.hybrid_pipeline_config.TextEmbedding")
    def test_invalid_text_embedding_type(
        self,
        _,
        mock_sparse_embedding, 
        mock_late_interaction_embedding,
        vector_params,
        sparse_vector_params
    ):
        """
        Test that a ValidationError is raised when text_embedding_config has an invalid type.
        
        Args:
            _: A patch for TextEmbedding.
            mock_sparse_embedding: A fixture providing a mock SparseEmbedding.
            mock_late_interaction_embedding: A fixture providing a mock LateInteractionTextEmbedding.
            vector_params: A fixture providing VectorParams.
            sparse_vector_params: A fixture providing SparseVectorParams.
        """
        mock_invalid = MagicMock()
        mock_invalid.model_name = "invalid_model"
        mock_invalid.embed.return_value = [[0.1, 0.2, 0.3]]

        with pytest.raises(ValueError, match="Embedding model in text_embedding_config must be an instance of TextEmbedding"):
            HybridPipelineConfig(
                text_embedding_config=(mock_invalid, vector_params),
                sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
                late_interaction_text_embedding_config=(
                    mock_late_interaction_embedding, 
                    vector_params
                ),
                multi_tenant=False,
                replication_factor=2,
                shard_number=3,
            )

    @patch("hybrid_search.hybrid_pipeline_config.SparseEmbedding")
    def test_invalid_sparse_embedding_type(
        self,
        _,
        mock_text_embedding,
        mock_late_interaction_embedding,
        vector_params,
        sparse_vector_params
    ):
        """
        Test that a ValidationError is raised when sparse_embedding_config has an invalid type.
        
        Args:
            _: A patch for SparseEmbedding.
            mock_text_embedding: A fixture providing a mock TextEmbedding.
            mock_late_interaction_embedding: A fixture providing a mock LateInteractionTextEmbedding.
            vector_params: A fixture providing VectorParams.
            sparse_vector_params: A fixture providing SparseVectorParams.
        """
        mock_invalid = MagicMock()
        mock_invalid.model_name = "invalid_model"
        mock_invalid.embed.return_value = [[0.1, 0.2, 0.3]]

        with pytest.raises(ValueError, match="Embedding model in sparse_embedding_config must be an instance of SparseEmbedding"):
            HybridPipelineConfig(
                text_embedding_config=(mock_text_embedding, vector_params),
                sparse_embedding_config=(mock_invalid, sparse_vector_params),
                late_interaction_text_embedding_config=(
                    mock_late_interaction_embedding, 
                    vector_params
                ),
                multi_tenant=False,
                replication_factor=2,
                shard_number=3,
            )

    @patch("hybrid_search.hybrid_pipeline_config.LateInteractionTextEmbedding")
    def test_invalid_late_interaction_embedding_type(
        self,
        _,
        mock_text_embedding,
        mock_sparse_embedding,
        vector_params,
        sparse_vector_params
    ):
        """
        Test that a ValidationError is raised when late_interaction_text_embedding_config has an invalid type.
        
        Args:
            _: A patch for LateInteractionTextEmbedding.
            mock_text_embedding: A fixture providing a mock TextEmbedding.
            mock_sparse_embedding: A fixture providing a mock SparseEmbedding.
            vector_params: A fixture providing VectorParams.
            sparse_vector_params: A fixture providing SparseVectorParams.
        """
        mock_invalid = MagicMock()
        mock_invalid.model_name = "invalid_model"
        mock_invalid.embed.return_value = [[0.1, 0.2, 0.3]]

        with pytest.raises(ValueError, match="Embedding model in late_interaction_text_embedding_config must be an instance of LateInteractionTextEmbedding"):
            HybridPipelineConfig(
                text_embedding_config=(mock_text_embedding, vector_params),
                sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
                late_interaction_text_embedding_config=(mock_invalid, vector_params),
                multi_tenant=False,
                replication_factor=2,
                shard_number=3,
            )

    def test_missing_model_name_attribute(
        self,
        mock_text_embedding,
        mock_sparse_embedding,
        vector_params,
        sparse_vector_params
    ):
        """
        Test that a ValidationError is raised when an embedding model is missing the model_name attribute.
        
        Args:
            mock_text_embedding: A fixture providing a mock TextEmbedding.
            mock_sparse_embedding: A fixture providing a mock SparseEmbedding.
            vector_params: A fixture providing VectorParams.
            sparse_vector_params: A fixture providing SparseVectorParams.
        """
        mock_invalid = MagicMock()
        # No model_name attribute
        mock_invalid.embed.return_value = [[0.1, 0.2, 0.3]]

        with pytest.raises(ValueError, match="Embedding model in late_interaction_text_embedding_config must have a 'model_name' attribute"):
            HybridPipelineConfig(
                text_embedding_config=(mock_text_embedding, vector_params),
                sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
                late_interaction_text_embedding_config=(mock_invalid, vector_params),
                multi_tenant=False,
                replication_factor=2,
                shard_number=3,
            )

    def test_missing_embed_method(
        self,
        mock_text_embedding,
        mock_sparse_embedding,
        vector_params,
        sparse_vector_params
    ):
        """
        Test that a ValidationError is raised when an embedding model is missing the embed method.
        
        Args:
            mock_text_embedding: A fixture providing a mock TextEmbedding.
            mock_sparse_embedding: A fixture providing a mock SparseEmbedding.
            vector_params: A fixture providing VectorParams.
            sparse_vector_params: A fixture providing SparseVectorParams.
        """
        mock_invalid = MagicMock()
        mock_invalid.model_name = "invalid_model"
        # No embed method

        with pytest.raises(ValueError, match="Embedding model in late_interaction_text_embedding_config must have an 'embed' method"):
            HybridPipelineConfig(
                text_embedding_config=(mock_text_embedding, vector_params),
                sparse_embedding_config=(mock_sparse_embedding, sparse_vector_params),
                late_interaction_text_embedding_config=(mock_invalid, vector_params),
                multi_tenant=False,
                replication_factor=2,
                shard_number=3,
            )