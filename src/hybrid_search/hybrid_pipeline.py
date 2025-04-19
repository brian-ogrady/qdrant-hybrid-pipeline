"""
Hybrid Pipeline module for vector search combining dense, sparse, and late interaction embeddings.

This module provides the implementation of the hybrid search pipeline that leverages multiple
embedding types for improved search performance.
"""

import uuid
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.models import (
    PointStruct,
    Prefetch,
    QuantizationSearchParams,
    SearchParams,
)

from .hybrid_pipeline_config import HybridPipelineConfig


class HybridPipeline:
    """
    Pipeline for hybrid search using multiple embedding types.
    
    This class implements a hybrid search pipeline that combines dense embeddings,
    sparse embeddings, and late interaction embeddings for improved search performance.
    It handles the creation and management of a Qdrant collection with the specified
    configuration, as well as document insertion and search operations.
    
    The hybrid approach combines the strengths of different embedding types:
    - Dense embeddings: Good for semantic similarity
    - Sparse embeddings: Good for keyword matching
    - Late interaction embeddings: Good for retrieval with detailed token-level interactions
    
    Attributes:
        collection_name: Name of the Qdrant collection
        qdrant_client: Client for interacting with the Qdrant vector database
        config: Configuration for the hybrid pipeline
        vectors_config_dict: Dictionary of vector configurations
        sparse_vectors_config_dict: Dictionary of sparse vector configurations
        multi_tenant: Flag indicating if the pipeline supports multiple tenants
        replication_factor: Number of replicas for each shard
        shard_number: Number of shards for the collection
        partition_field_name: Field name used for partitioning in multi-tenant mode
        partition_index_params: Index parameters for the partition field
    """
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        hybrid_pipeline_config: HybridPipelineConfig,
    ):
        """
        Initialize a new HybridPipeline instance.
        
        Args:
            qdrant_client: Client for interacting with the Qdrant vector database
            collection_name: Name of the Qdrant collection to create
            hybrid_pipeline_config: Configuration for the hybrid pipeline
            
        Raises:
            ValueError: If the collection already exists
        """
        self.collection_name = collection_name
        self.qdrant_client = qdrant_client
        self.config = hybrid_pipeline_config
        self.vectors_config_dict = self.config.get_vectors_config_dict()
        self.sparse_vectors_config_dict = self.config.get_sparse_vectors_config_dict()
        self.multi_tenant = self.config.multi_tenant
        self.replication_factor = self.config.replication_factor
        self.shard_number = self.config.shard_number
        self.partition_field_name, self.partition_index_params = self.config.get_partition_config()
        self._create_collection()

        if self.multi_tenant:
            self._create_payload_index()
        
    def _create_collection(self) -> bool:
        """
        Create a new Qdrant collection with the configured parameters.
        
        Returns:
            bool: True if the collection was created successfully
            
        Raises:
            ValueError: If the collection already exists
        """
        if self.qdrant_client.collection_exists(self.collection_name):
            raise ValueError(
                f"Collection {self.collection_name} already exists"
            )

        return self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.vectors_config_dict,
            sparse_vectors_config=self.sparse_vectors_config_dict,
            replication_factor=self.replication_factor,
            shard_number=self.shard_number,
        )

    def _create_payload_index(self) -> types.UpdateResult:
        """
        Create a payload index for the partition field in multi-tenant mode.
        
        Returns:
            types.UpdateResult: Result of the index creation operation
        """
        return self.qdrant_client.create_payload_index(
            collection_name=self.collection_name,
            field_name=self.partition_field_name,
            field_schema=self.partition_index_params,
        )
    
    def _embed_documents(self, documents: Union[str, List[str]]) -> Dict[str, List[float]]:
        """
        Embed documents using all configured embedding models.
        
        Args:
            documents: A single document string or a list of document strings to embed
            
        Returns:
            Dict[str, List[float]]: Dictionary mapping model names to lists of embeddings
        """
        if isinstance(documents, str):
            documents = [documents]
        return {
            config[0].model_name: config[0].embed(documents)
            for config in self.config.list_embedding_configs()
        }

    def _prepare_documents(
        self,
        documents: List[str],
        payloads: List[Dict[str, Any]],
        document_ids: List[uuid.UUID],
    ) -> List[types.PointStruct]:
        """
        Prepare documents for insertion into the Qdrant collection.
        
        This method embeds the documents using the configured embedding models and
        creates PointStruct objects that can be inserted into the Qdrant collection.
        
        Args:
            documents: List of document strings to embed and insert
            payloads: List of payload dictionaries containing metadata for each document
            document_ids: List of UUIDs to use as IDs for each document
            
        Returns:
            List[types.PointStruct]: List of prepared points ready for insertion
            
        Raises:
            ValueError: If the lengths of documents, payloads, and document_ids don't match,
                or if multi_tenant is True and a payload is missing the partition field
        """
        
        if not (len(documents) == len(payloads) == len(document_ids)):
            raise ValueError(
                "documents, payloads, and document_ids must be the same length"
            )

        embeddings_dict = self._embed_documents(documents)

        points = []
        for i in range(len(documents)):
            if self.multi_tenant and self.partition_field_name not in payloads[i]:
                raise ValueError(
                    f"payloads must contain {self.partition_field_name} if multi_tenant is True"
                )
            document_id = str(document_ids[i])
            payloads[i]["document"] = documents[i]
            payloads[i]["document_id"] = document_id
            point = PointStruct(
                id=document_id,
                vector={
                    model_name: embeddings_dict[model_name][i] for model_name in embeddings_dict
                },
                payload=payloads[i],
            )
            points.append(point)

        return points

    def insert_documents(
        self,
        documents: List[str],
        payloads: List[Dict[str, Any]],
        document_ids: List[uuid.UUID],
        batch_size: int = 100,
    ):
        """
        Insert documents into the Qdrant collection.
        
        This method embeds the documents using the configured embedding models and
        inserts them into the Qdrant collection in batches.
        
        Args:
            documents: List of document strings to embed and insert
            payloads: List of payload dictionaries containing metadata for each document
            document_ids: List of UUIDs to use as IDs for each document
            batch_size: Number of documents to process in each batch (default: 100)
            
        Raises:
            ValueError: If the lengths of documents, payloads, and document_ids don't match,
                or if multi_tenant is True and a payload is missing the partition field
        """
        if not (len(documents) == len(payloads) == len(document_ids)):
            raise ValueError(
                "documents, payloads, and document_ids must be the same length"
            )
        
        for i in range(0, len(documents), batch_size):
            points = self._prepare_documents(
                documents=documents[i:i+batch_size],
                payloads=payloads[i:i+batch_size],
                document_ids=document_ids[i:i+batch_size]
            )
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def _embed_query(self, query: str) -> Dict[str, List[float]]:
        """
        Embed a query string using all configured embedding models.
        
        Args:
            query: Query string to embed
            
        Returns:
            Dict[str, List[float]]: Dictionary mapping model names to query embeddings
        """
        return {
            model.model_name: model.embed([query])[0]
            for model in self.config.list_embedding_models()
        }
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        partition_filter: Optional[str] = None,
        overquery_factor: float = 1.0,
    ) -> types.QueryResponse:
        """
        Search for documents similar to the query using the hybrid approach.
        
        This method implements a hybrid search that combines dense embeddings,
        sparse embeddings, and late interaction embeddings to retrieve the most
        relevant documents for the query.
        
        Args:
            query: Query string to search for
            top_k: Number of results to return (default: 10)
            partition_filter: Value to filter by in the partition field for multi-tenant mode
                (must be None if multi_tenant is False)
            overquery_factor: Factor to oversample results during quantization (default: 1.0,
                must be >= 1.0)
                
        Returns:
            types.QueryResponse: Query response containing the search results
            
        Raises:
            ValueError: If overquery_factor is less than 1.0 or if partition_filter is
                provided when multi_tenant is False
        """
        if overquery_factor < 1.0:
            raise ValueError("overquery_factor must be greater than or equal to 1.0")
        
        filter_condition = None
        if not self.multi_tenant and partition_filter:
            raise ValueError("partition_filter must be None if multi_tenant is False")

        filter_condition = types.Filter(
            must=[
                types.FieldCondition(
                    key=self.partition_field_name,
                    match=types.MatchValue(value=partition_filter)
                )
            ]
        )
        
        query_embeddings = self._embed_query(query)

        model_names = self.config.list_embedding_model_names()
        dense_model_name = model_names[0]
        sparse_model_name = model_names[1]
        late_interaction_model_name = model_names[2]

        dense_prefetch = Prefetch(
            query=query_embeddings[dense_model_name],
            using=dense_model_name,
            limit=top_k,
            filter=filter_condition,
            search_params=SearchParams(
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=overquery_factor,
                ),
            ),
        )

        sparse_prefetch = Prefetch(
            query=query_embeddings[sparse_model_name],
            using=sparse_model_name,
            limit=top_k,
            filter=filter_condition,
        )

        return self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                dense_prefetch,
                sparse_prefetch,
            ],
            query=query_embeddings[late_interaction_model_name],
            using=late_interaction_model_name,
            limit=top_k,
            with_payload=True,
        )
    
    def delete_document(self, document_id: str):
        """
        Delete a document from the collection by its ID.
        
        Args:
            document_id: ID of the document to delete
            
        Note:
            This method is currently not implemented.
        
        TODO: Implement delete document functionality
        """
        #TODO: Implement delete document
        pass
    