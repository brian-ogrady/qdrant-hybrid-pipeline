"""
Configuration module for the hybrid search pipeline.

This module provides configuration classes and type definitions for setting up
hybrid search pipelines that combine dense, sparse, and late interaction embeddings.
"""

from typing import ClassVar, List, Mapping, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, model_validator
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed.sparse import SparseTextEmbedding
from fastembed.text import TextEmbedding
from qdrant_client.conversions import common_types as types
from qdrant_client.models import KeywordIndexParams
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedding(SentenceTransformer):
    """
    A wrapper around the SentenceTransformer class that adds a model_name attribute.
    """
    def __init__(self, model_name_or_path: str, *args, **kwargs):
        self._model_name_or_path = model_name_or_path
        super().__init__(model_name_or_path, *args, **kwargs)
    
    @property
    def model_name(self) -> str:
        return self._model_name_or_path
    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self.encode(texts, **kwargs).tolist()


Embedding = TypeVar(
    "Embedding",
    TextEmbedding,
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    SentenceTransformerEmbedding,
)
"""Type variable for the different types of embedding models supported."""


BaseVectorParams = TypeVar(
    "BaseVectorParams",
    types.VectorParams,
    types.SparseVectorParams
)
"""Type variable for the different types of vector parameters supported."""


class HybridPipelineConfig(BaseModel):
    """
    Configuration for a hybrid search pipeline combining multiple embedding types.
    
    This class encapsulates the configuration for a hybrid search pipeline that combines
    dense embeddings, sparse embeddings, and late interaction embeddings for improved
    search performance. It also includes configuration for multi-tenancy and sharding.
    
    Attributes:
        text_embedding_config: Configuration for the dense text embedding model.
            A tuple containing a TextEmbedding model instance and its associated VectorParams.
        sparse_embedding_config: Configuration for the sparse embedding model.
            A tuple containing a SparseTextEmbedding model instance and its associated SparseVectorParams.
        late_interaction_text_embedding_config: Configuration for the late interaction embedding model.
            A tuple containing a LateInteractionTextEmbedding model instance and its associated VectorParams.
        partition_config: Configuration for multi-tenant partitioning.
            A tuple containing the field name to use for partitioning and the KeywordIndexParams 
            for the partition field. Required if multi_tenant is True.
        multi_tenant: Flag indicating whether the pipeline should support multiple tenants.
            If True, the pipeline will create a partitioned collection using the partition_config.
            Default is False.
        replication_factor: The number of replicas for each shard in the Qdrant collection.
            Increases redundancy and read performance. Default is 2.
        shard_number: The number of shards for the Qdrant collection.
            Affects write performance and horizontal scalability. Default is 3.
    """
    DENSE_VECTOR_NAME: ClassVar[str] = "dense"
    SPARSE_VECTOR_NAME: ClassVar[str] = "sparse"
    LATE_INTERACTION_VECTOR_NAME: ClassVar[str] = "multivector"

    text_embedding_config: Tuple[Union[TextEmbedding, SentenceTransformerEmbedding], types.VectorParams]
    sparse_embedding_config: Tuple[SparseTextEmbedding, types.SparseVectorParams]
    late_interaction_text_embedding_config: Optional[Tuple[LateInteractionTextEmbedding, types.VectorParams]] = None
    # TODO: Replace PartitionConfig with MultiTenantConfig -> allow user to specify global index or not during collection creation
    partition_config: Optional[Tuple[str, KeywordIndexParams]] = None
    multi_tenant: Optional[bool] = False
    replication_factor: Optional[int] = 2
    shard_number: Optional[int] = 3

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode='after')
    def _validate_config(self):
        """
        Validate the configuration after model initialization.
        
        Ensures that the configuration is valid by checking:
        - Multi-tenancy and partition configuration compatibility
        - Replication factor and shard number are valid
        - Embedding models are of the correct type and have required attributes
        
        Returns:
            self: The validated configuration instance
            
        Raises:
            ValueError: If any validation check fails
        """
        if self.multi_tenant and self.partition_config is None:
            raise ValueError("partition_config must be provided if multi_tenant is True")
        if not self.multi_tenant and self.partition_config is not None:
            raise ValueError("partition_config must be None if multi_tenant is False")
        
        if not isinstance(self.replication_factor, int) or self.replication_factor < 1:
            raise ValueError("replication_factor must be an integer greater than 0")
        
        if not isinstance(self.shard_number, int) or self.shard_number < 1:
            raise ValueError("shard_number must be an integer greater than 0") 
        
        for config_name, (model, _) in [
            ("text_embedding_config", self.text_embedding_config),
            ("sparse_embedding_config", self.sparse_embedding_config),
            ("late_interaction_text_embedding_config", self.late_interaction_text_embedding_config)
        ]:
            if config_name == "text_embedding_config" and not isinstance(model, Union[TextEmbedding, SentenceTransformerEmbedding]):
                raise ValueError(f"Embedding model in {config_name} must be an instance of TextEmbedding")
            elif config_name == "sparse_embedding_config" and not isinstance(model, SparseTextEmbedding):
                raise ValueError(f"Embedding model in {config_name} must be an instance of SparseEmbedding")
            elif config_name == "late_interaction_text_embedding_config":
                if model is not None and not isinstance(model, LateInteractionTextEmbedding):
                    raise ValueError(f"Embedding model in {config_name} must be an instance of LateInteractionTextEmbedding")
            
            if not hasattr(model, "model_name"):
                raise ValueError(f"Embedding model in {config_name} must have a 'model_name' attribute")
                
            if not hasattr(model, "embed") or not callable(getattr(model, "embed")):
                raise ValueError(f"Embedding model in {config_name} must have an 'embed' method")
        return self
    
    @property
    def dense_model_config(self) -> Tuple[TextEmbedding, types.VectorParams]:
        """Get the dense embedding model configuration."""
        return self.text_embedding_config
    
    @property
    def sparse_model_config(self) -> Tuple[SparseTextEmbedding, types.SparseVectorParams]:
        """Get the sparse embedding model configuration."""
        return self.sparse_embedding_config
    
    @property
    def late_interaction_model_config(self) -> Optional[Tuple[LateInteractionTextEmbedding, types.VectorParams]]:
        """Get the late interaction embedding model configuration."""
        return self.late_interaction_text_embedding_config
    
    @property
    def dense_model(self) -> TextEmbedding:
        """Get the dense embedding model."""
        return self.dense_model_config[0]
    
    @property
    def sparse_model(self) -> SparseTextEmbedding:
        """Get the sparse embedding model."""
        return self.sparse_model_config[0]
    
    @property
    def late_interaction_model(self) -> Optional[LateInteractionTextEmbedding]:
        """Get the late interaction embedding model."""
        if self.late_interaction_model_config:
            return self.late_interaction_model_config[0]
    
    @property
    def dense_model_name(self) -> str:
        """Get the name of the dense embedding model."""
        return self.dense_model.model_name
    
    @property
    def sparse_model_name(self) -> str:
        """Get the name of the sparse embedding model."""
        return self.sparse_model.model_name
    
    @property
    def late_interaction_model_name(self) -> Optional[str]:
        """Get the name of the late interaction embedding model."""
        if self.late_interaction_model:
            return self.late_interaction_model.model_name

    def list_embedding_configs(self) -> List[Tuple[Embedding, BaseVectorParams]]:
        """
        Get a list of all embedding configurations.
        
        Returns:
            List[Tuple[Embedding, BaseVectorParams]]: A list containing tuples of embedding models 
            and their associated vector parameters
        """
        config_list = [
            self.text_embedding_config,
            self.sparse_embedding_config
        ]
        if self.late_interaction_text_embedding_config:
            config_list.append(self.late_interaction_text_embedding_config)
        return config_list
    
    def list_embedding_model_names(self) -> List[str]:
        """
        Get a list of all embedding model names.
        
        Returns:
            List[str]: A list of embedding model names
        """
        return [
            config[0].model_name
            for config in self.list_embedding_configs()
        ]
    
    def list_embedding_models(self) -> List[Embedding]:
        """
        Get a list of all embedding models.
        
        Returns:
            List[Embedding]: A list containing all embedding model instances
        """
        return [
            config[0]
            for config in self.list_embedding_configs()
        ]

    def get_vectors_config_dict(self) -> Mapping[str, types.VectorParams]:
        """
        Get a dictionary mapping dense embedding model names to their vector parameters.
        
        Returns:
            Mapping[str, types.VectorParams]: Dictionary mapping model names to VectorParams
        """
        config_dict = {
            self.DENSE_VECTOR_NAME: self.dense_model_config[1],
        }
        if self.late_interaction_model_config:
            config_dict[self.LATE_INTERACTION_VECTOR_NAME] = self.late_interaction_model_config[1]
        return config_dict
    
    def get_sparse_vectors_config_dict(self) -> Mapping[str, types.SparseVectorParams]:
        """
        Get a dictionary mapping sparse embedding model names to their vector parameters.
        
        Returns:
            Mapping[str, types.SparseVectorParams]: Dictionary mapping model names to SparseVectorParams
        """
        return {
            self.SPARSE_VECTOR_NAME: self.sparse_model_config[1],
        }
    
    def get_partition_config(self) -> Tuple[str, KeywordIndexParams]:
        """
        Get the partition configuration for multi-tenant setup.
        
        Returns:
            Tuple[str, KeywordIndexParams]: A tuple containing the partition field name
            and the KeywordIndexParams for that field
            
        Raises:
            ValueError: If partition_config is not set but this method is called
        """
        if not self.partition_config:
            raise ValueError("partition_config must be specified during instantiation")
        return self.partition_config
