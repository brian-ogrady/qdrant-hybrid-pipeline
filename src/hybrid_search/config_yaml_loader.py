from pathlib import Path
from typing import Literal, Optional, Dict, Any, Union

import yaml
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed.sparse import SparseTextEmbedding
from fastembed.text import TextEmbedding
from pydantic import BaseModel, Field
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    BinaryQuantization,
    BinaryQuantizationConfig,
    MultiVectorConfig,
    MultiVectorComparator,
    HnswConfigDiff,
    KeywordIndexParams,
)

from .hybrid_pipeline_config import HybridPipelineConfig, SentenceTransformerEmbedding


class DenseEmbeddingConfig(BaseModel):
    """Configuration for the dense embedding model."""
    package: Literal["fastembed", "sentence-transformers"]
    model_name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class SparseEmbeddingConfig(BaseModel):
    """Configuration for the sparse embedding model."""
    package: Literal["fastembed"]
    model_name: str
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class LateInteractionEmbeddingConfig(BaseModel):
    """Configuration for the late interaction embedding model."""
    package: Literal["fastembed"]
    model_name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class YAMLConfig(BaseModel):
    """The root model for validating the entire YAML file."""
    dense_embedding: DenseEmbeddingConfig
    sparse_embedding: SparseEmbeddingConfig
    late_interaction_embedding: Optional[LateInteractionEmbeddingConfig] = None
    
    multi_tenant: Optional[bool] = False
    replication_factor: Optional[int] = 2
    shard_number: Optional[int] = 3
    partition_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "YAMLConfig":
        """Load a YAML configuration file."""
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
        return cls(**raw_config)


def _build_vector_params(params_dict: Dict[str, Any]) -> VectorParams:
    """Helper function to construct VectorParams from a dictionary."""
    if "distance" in params_dict:
        params_dict["distance"] = getattr(Distance, params_dict["distance"].upper())

    if "quantization_config" in params_dict and "binary" in params_dict["quantization_config"]:
        binary_conf = params_dict["quantization_config"]["binary"]
        params_dict["quantization_config"] = BinaryQuantization(
            binary=BinaryQuantizationConfig(**binary_conf)
        )
    
    if "multivector_config" in params_dict:
        comp = params_dict["multivector_config"]["comparator"]
        params_dict["multivector_config"] = MultiVectorConfig(
            comparator=getattr(MultiVectorComparator, comp)
        )
    
    if "hnsw_config" in params_dict:
        params_dict["hnsw_config"] = HnswConfigDiff(**params_dict["hnsw_config"])

    return VectorParams(**params_dict)
        

def create_hybrid_pipeline_from_yaml(path: Union[str, Path]) -> YAMLConfig:
    """Create a hybrid pipeline from a YAML configuration file."""
    config = YAMLConfig.from_yaml(path)

    if config.dense_embedding.package == "sentence-transformers":
        text_model = SentenceTransformerEmbedding(config.dense_embedding.model_name)
    else:
        text_model = TextEmbedding(config.dense_embedding.model_name)

    sparse_model = SparseTextEmbedding(config.sparse_embedding.model_name)

    dense_params = _build_vector_params(config.dense_embedding.params)
    sparse_params = SparseVectorParams()
    
    pipeline_args = {
        "text_embedding_config": (text_model, dense_params),
        "sparse_embedding_config": (sparse_model, sparse_params),
    }

    if config.late_interaction_embedding:
        late_model = LateInteractionTextEmbedding(config.late_interaction_embedding.model_name)
        late_params = _build_vector_params(config.late_interaction_embedding.params)
        pipeline_args["late_interaction_text_embedding_config"] = (late_model, late_params)
        
    if config.partition_config:
        field_name = config.partition_config.pop("field")
        partition_index_params = KeywordIndexParams(**config.partition_config)
        pipeline_args["partition_config"] = (field_name, partition_index_params)
        
    if config.multi_tenant is not None:
        pipeline_args["multi_tenant"] = config.multi_tenant
    if config.replication_factor is not None:
        pipeline_args["replication_factor"] = config.replication_factor
    if config.shard_number is not None:
        pipeline_args["shard_number"] = config.shard_number

    return HybridPipelineConfig(**pipeline_args)
