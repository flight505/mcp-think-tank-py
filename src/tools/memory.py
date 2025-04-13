#!/usr/bin/env python3
"""
Knowledge Graph implementation with JSONL storage and semantic search
"""
import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Iterator

import numpy as np
from pydantic import BaseModel, Field, validator

from src.tools.embeddings import EmbeddingProvider

logger = logging.getLogger("mcp-think-tank.memory")

class RecordType(str, Enum):
    """Types of records stored in the JSONL file"""
    ENTITY_CREATE = "entity_create"
    ENTITY_UPDATE = "entity_update" 
    ENTITY_DELETE = "entity_delete"
    RELATION_CREATE = "relation_create"
    RELATION_DELETE = "relation_delete"


class Entity(BaseModel):
    """
    Entity model representing a node in the knowledge graph
    """
    name: str
    entity_type: str = Field(alias="entityType")
    observations: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    deleted: bool = False
    embedding: Optional[List[float]] = None
    
    class Config:
        populate_by_name = True


class Relation(BaseModel):
    """
    Relation model representing an edge between entities in the knowledge graph
    """
    from_entity: str = Field(alias="from")
    to_entity: str = Field(alias="to")
    relation_type: str = Field(alias="relationType")
    created_at: datetime = Field(default_factory=datetime.now)
    deleted: bool = False
    
    class Config:
        populate_by_name = True


class JournalRecord(BaseModel):
    """
    Record in the JSONL journal file
    """
    record_type: RecordType
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any]
    actor: Optional[str] = None


class KnowledgeGraph:
    """
    Knowledge Graph with JSONL-based persistence and semantic search
    """
    
    def __init__(self, memory_file_path: str, use_embeddings: bool = True, 
                 embedding_cache_size: int = 10000, embedding_cache_ttl: Optional[int] = None,
                 load_limit: Optional[int] = None, load_deleted: bool = False):
        """
        Initialize the knowledge graph
        
        Args:
            memory_file_path: Path to the JSONL memory file
            use_embeddings: Whether to use embeddings for semantic search
            embedding_cache_size: Size of the embedding cache (default: 10000)
            embedding_cache_ttl: TTL in seconds for embedding cache entries (default: None)
            load_limit: Maximum number of records to load initially (default: None = all records)
            load_deleted: Whether to load deleted entities and relations (default: False)
        """
        self.memory_file_path = memory_file_path
        self.use_embeddings = use_embeddings
        self.load_limit = load_limit
        self.load_deleted = load_deleted
        self.total_records = 0
        self.loaded_records = 0
        
        # Performance metrics
        self.load_time = 0.0
        self.last_operation_time = 0.0
        
        # In-memory storage
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.embedding_provider = None
        
        # File position tracking for incremental loading
        self.file_position = 0
        self.fully_loaded = False
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
        
        # Initialize embedding provider
        if self.use_embeddings:
            self._init_embedding_provider(embedding_cache_size, embedding_cache_ttl)
        
        # Load existing data if the file exists
        if os.path.exists(self.memory_file_path):
            self._count_total_records()
            self._load_from_jsonl()
    
    def _init_embedding_provider(self, cache_size: int, cache_ttl: Optional[int]):
        """
        Initialize the embedding provider for semantic search
        
        Args:
            cache_size: Size of the embedding cache
            cache_ttl: TTL in seconds for embedding cache entries
        """
        try:
            self.embedding_provider = EmbeddingProvider(
                cache_capacity=cache_size,
                cache_ttl=cache_ttl
            )
            logger.info("Embedding provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {e}")
            self.use_embeddings = False
    
    def _count_total_records(self):
        """Count the total number of records in the JSONL file"""
        try:
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
                self.total_records = count
                logger.info(f"Total records in memory file: {count}")
        except Exception as e:
            logger.error(f"Failed to count records: {e}")
            self.total_records = 0
    
    def _load_from_jsonl(self):
        """Load existing data from the JSONL file"""
        logger.info(f"Loading knowledge graph from {self.memory_file_path}")
        
        start_time = time.time()
        
        try:
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                # Skip to the last position if we've already loaded some records
                if self.file_position > 0:
                    f.seek(self.file_position)
                
                records_loaded = 0
                for line_number, line in enumerate(f, 1):
                    try:
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        # Parse the record
                        record_data = json.loads(line)
                        record = JournalRecord(**record_data)
                        
                        # Process based on record type
                        self._process_journal_record(record)
                        
                        records_loaded += 1
                        self.loaded_records += 1
                        
                        # Stop if we've reached the load limit
                        if self.load_limit and records_loaded >= self.load_limit:
                            # Save position for future loading
                            self.file_position = f.tell()
                            break
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON on line {line_number}: {line}")
                    except Exception as e:
                        logger.error(f"Error processing line {line_number}: {e}")
                
                # If we've processed all records, mark as fully loaded
                if not self.load_limit or records_loaded < self.load_limit:
                    self.fully_loaded = True
                    self.file_position = f.tell()
                
                self.load_time = time.time() - start_time
                logger.info(f"Loaded {records_loaded} records in {self.load_time:.2f}s")
                logger.info(f"Memory usage: {len(self.entities)} entities, {len(self.relations)} relations")
                
                # If we're using embeddings, report cache stats
                if self.use_embeddings and self.embedding_provider:
                    cache_stats = self.embedding_provider.get_cache_stats()
                    logger.info(f"Embedding cache stats: {cache_stats}")
        except FileNotFoundError:
            logger.info(f"Memory file {self.memory_file_path} not found, starting with empty graph")
            self.fully_loaded = True
        except Exception as e:
            logger.error(f"Failed to load memory file: {e}")
    
    def load_more_records(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Load more records from the JSONL file
        
        Args:
            limit: Maximum number of additional records to load
            
        Returns:
            Dictionary with loading statistics
        """
        if self.fully_loaded:
            return {
                "loaded": 0,
                "total_loaded": self.loaded_records,
                "total_records": self.total_records,
                "fully_loaded": True,
                "message": "All records already loaded"
            }
        
        start_time = time.time()
        prev_loaded = self.loaded_records
        
        self.load_limit = limit
        self._load_from_jsonl()
        
        load_time = time.time() - start_time
        records_loaded = self.loaded_records - prev_loaded
        
        return {
            "loaded": records_loaded,
            "total_loaded": self.loaded_records,
            "total_records": self.total_records,
            "fully_loaded": self.fully_loaded,
            "load_time": load_time,
            "message": f"Loaded {records_loaded} additional records in {load_time:.2f}s"
        }
    
    def _process_journal_record(self, record: JournalRecord):
        """
        Process a journal record and update the in-memory state
        
        Args:
            record: The journal record to process
        """
        if record.record_type == RecordType.ENTITY_CREATE:
            entity = Entity(**record.data)
            # Skip deleted entities if not loading them
            if not self.load_deleted and entity.deleted:
                return
            self.entities[entity.name] = entity
        
        elif record.record_type == RecordType.ENTITY_UPDATE:
            name = record.data.get("name")
            if name in self.entities:
                # Update existing entity with new data
                self.entities[name] = Entity(**{**self.entities[name].model_dump(), **record.data})
        
        elif record.record_type == RecordType.ENTITY_DELETE:
            name = record.data.get("name")
            if name in self.entities:
                self.entities[name].deleted = True
                # Remove from memory if we're not keeping deleted entities
                if not self.load_deleted:
                    self.entities.pop(name)
        
        elif record.record_type == RecordType.RELATION_CREATE:
            relation = Relation(**record.data)
            # Skip if either entity doesn't exist or is deleted
            if not self.load_deleted and (relation.deleted or 
                                         relation.from_entity not in self.entities or 
                                         relation.to_entity not in self.entities):
                return
                
            # Only add if relation doesn't already exist
            if not self._relation_exists(relation):
                self.relations.append(relation)
        
        elif record.record_type == RecordType.RELATION_DELETE:
            from_entity = record.data.get("from")
            to_entity = record.data.get("to")
            relation_type = record.data.get("relationType")
            
            # Mark matching relations as deleted
            for relation in self.relations:
                if (relation.from_entity == from_entity and 
                    relation.to_entity == to_entity and 
                    relation.relation_type == relation_type and
                    not relation.deleted):
                    relation.deleted = True
                    
                    # Remove from memory if we're not keeping deleted relations
                    if not self.load_deleted:
                        self.relations.remove(relation)
    
    def _append_to_jsonl(self, record: JournalRecord):
        """
        Append a record to the JSONL file
        
        Args:
            record: The record to append
        """
        try:
            start_time = time.time()
            with open(self.memory_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record.model_dump()) + '\n')
            self.last_operation_time = time.time() - start_time
            self.total_records += 1
        except Exception as e:
            logger.error(f"Failed to append to memory file: {e}")
            raise
    
    def _relation_exists(self, relation: Relation) -> bool:
        """
        Check if a relation already exists
        
        Args:
            relation: The relation to check
            
        Returns:
            True if the relation exists, False otherwise
        """
        for existing_relation in self.relations:
            if (existing_relation.from_entity == relation.from_entity and
                existing_relation.to_entity == relation.to_entity and
                existing_relation.relation_type == relation.relation_type and
                not existing_relation.deleted):
                return True
        return False
    
    def _generate_embedding(self, entity: Entity) -> Optional[List[float]]:
        """
        Generate an embedding for an entity
        
        Args:
            entity: The entity to generate an embedding for
            
        Returns:
            A list of floats representing the embedding, or None if embeddings are disabled
        """
        if not self.use_embeddings or not self.embedding_provider:
            return None
            
        try:
            # Combine name, type, and observations for embedding
            text = f"{entity.name} {entity.entity_type} " + " ".join(entity.observations)
            embedding = self.embedding_provider.generate_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _generate_embeddings_batch(self, entities: List[Entity]) -> Dict[str, Optional[List[float]]]:
        """
        Generate embeddings for multiple entities in a batch
        
        Args:
            entities: List of entities to generate embeddings for
            
        Returns:
            Dictionary mapping entity names to embeddings
        """
        if not self.use_embeddings or not self.embedding_provider:
            return {entity.name: None for entity in entities}
            
        try:
            # Combine name, type, and observations for each entity
            texts = [f"{entity.name} {entity.entity_type} " + " ".join(entity.observations) for entity in entities]
            
            # Get text hashes
            text_hashes = [self.embedding_provider._get_text_hash(text) for text in texts]
            
            # Generate embeddings in batch
            hash_to_embedding = self.embedding_provider.generate_embeddings_batch(texts)
            
            # Map entity names to embeddings
            name_to_embedding = {}
            for entity, text_hash in zip(entities, text_hashes):
                name_to_embedding[entity.name] = hash_to_embedding.get(text_hash)
                
            return name_to_embedding
        except Exception as e:
            logger.error(f"Failed to generate embeddings batch: {e}")
            return {entity.name: None for entity in entities}
    
    # Public API methods
    
    def create_entities(self, entities_data: List[Dict[str, Any]], actor: Optional[str] = None) -> Dict[str, Any]:
        """
        Create multiple entities in the knowledge graph
        
        Args:
            entities_data: List of entity data dictionaries
            actor: Optional identifier for who/what is creating the entities
            
        Returns:
            Dictionary with created and existing entity names
        """
        start_time = time.time()
        created = []
        existing = []
        new_entities = []
        
        for entity_data in entities_data:
            name = entity_data.get("name")
            
            # Skip if no name provided
            if not name:
                continue
                
            # Check if entity already exists and is not deleted
            if name in self.entities and not self.entities[name].deleted:
                existing.append(name)
                continue
            
            # Create new entity
            entity = Entity(**entity_data)
            new_entities.append(entity)
            
            # Update or create entity in memory
            self.entities[name] = entity
            created.append(name)
        
        # Generate embeddings in batch for better performance
        if new_entities and self.use_embeddings:
            name_to_embedding = self._generate_embeddings_batch(new_entities)
            
            # Update entities with embeddings
            for entity in new_entities:
                entity.embedding = name_to_embedding.get(entity.name)
                
                # Append to JSONL with embedding
                record = JournalRecord(
                    record_type=RecordType.ENTITY_CREATE,
                    data=entity.model_dump(),
                    actor=actor
                )
                self._append_to_jsonl(record)
        else:
            # Process entities individually if no embeddings or no entities
            for entity in new_entities:
                record = JournalRecord(
                    record_type=RecordType.ENTITY_CREATE,
                    data=entity.model_dump(),
                    actor=actor
                )
                self._append_to_jsonl(record)
        
        self.last_operation_time = time.time() - start_time
        
        return {
            "created": created,
            "existing": existing,
            "incomplete": False,
            "message": f"Created {len(created)} new entities. {len(existing)} entities already existed.",
            "imageEntities": len(entities_data),
            "operation_time": self.last_operation_time
        }
    
    def create_relations(self, relations_data: List[Dict[str, Any]], actor: Optional[str] = None) -> Dict[str, Any]:
        """
        Create multiple relations in the knowledge graph
        
        Args:
            relations_data: List of relation data dictionaries
            actor: Optional identifier for who/what is creating the relations
            
        Returns:
            Dictionary with created and failed relation data
        """
        created = []
        failed = []
        
        for relation_data in relations_data:
            from_entity = relation_data.get("from")
            to_entity = relation_data.get("to")
            relation_type = relation_data.get("relationType")
            
            # Skip if missing required fields
            if not (from_entity and to_entity and relation_type):
                failed.append(relation_data)
                continue
            
            # Check if both entities exist and are not deleted
            if (from_entity not in self.entities or 
                to_entity not in self.entities or
                self.entities[from_entity].deleted or
                self.entities[to_entity].deleted):
                failed.append(relation_data)
                continue
            
            # Create relation
            relation = Relation(**relation_data)
            
            # Check if relation already exists
            if self._relation_exists(relation):
                failed.append(relation_data)
                continue
            
            # Add to memory
            self.relations.append(relation)
            
            # Append to JSONL
            record = JournalRecord(
                record_type=RecordType.RELATION_CREATE,
                data=relation.model_dump(),
                actor=actor
            )
            self._append_to_jsonl(record)
            
            created.append(relation_data)
        
        return {
            "created": created,
            "failed": failed,
            "message": f"Created {len(created)} new relations. {len(failed)} relations failed."
        }
    
    def add_observations(self, observations_data: List[Dict[str, Any]], actor: Optional[str] = None) -> Dict[str, Any]:
        """
        Add observations to existing entities
        
        Args:
            observations_data: List of dictionaries with entity names and observations
            actor: Optional identifier for who/what is adding the observations
            
        Returns:
            Dictionary with updated and not_found entity names
        """
        updated = []
        not_found = []
        
        for observation_data in observations_data:
            entity_name = observation_data.get("entityName")
            new_observations = observation_data.get("contents", [])
            
            if not entity_name or not new_observations:
                continue
            
            if entity_name not in self.entities or self.entities[entity_name].deleted:
                not_found.append(entity_name)
                continue
            
            # Get existing entity
            entity = self.entities[entity_name]
            
            # Add new observations
            entity.observations.extend(new_observations)
            entity.updated_at = datetime.now()
            
            # Update embedding
            if self.use_embeddings:
                entity.embedding = self._generate_embedding(entity)
            
            # Append to JSONL
            record = JournalRecord(
                record_type=RecordType.ENTITY_UPDATE,
                data={
                    "name": entity_name,
                    "observations": entity.observations,
                    "updated_at": entity.updated_at,
                    "embedding": entity.embedding
                },
                actor=actor
            )
            self._append_to_jsonl(record)
            
            updated.append(entity_name)
        
        return {
            "updated": updated,
            "not_found": not_found,
            "message": f"Added observations to {len(updated)} entities. {len(not_found)} entities not found."
        }
    
    def delete_entities(self, entity_names: List[str], actor: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete entities from the knowledge graph
        
        Args:
            entity_names: List of entity names to delete
            actor: Optional identifier for who/what is deleting the entities
            
        Returns:
            Dictionary with deleted and not_found entity names
        """
        deleted = []
        not_found = []
        
        for name in entity_names:
            if name not in self.entities or self.entities[name].deleted:
                not_found.append(name)
                continue
            
            # Mark as deleted
            self.entities[name].deleted = True
            self.entities[name].updated_at = datetime.now()
            
            # Append to JSONL
            record = JournalRecord(
                record_type=RecordType.ENTITY_DELETE,
                data={"name": name},
                actor=actor
            )
            self._append_to_jsonl(record)
            
            deleted.append(name)
        
        return {
            "deleted": deleted,
            "not_found": not_found,
            "message": f"Deleted {len(deleted)} entities. {len(not_found)} entities not found."
        }
    
    def delete_relations(self, relations_data: List[Dict[str, Any]], actor: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete relations from the knowledge graph
        
        Args:
            relations_data: List of relation data dictionaries
            actor: Optional identifier for who/what is deleting the relations
            
        Returns:
            Dictionary with deleted and not_found relation data
        """
        deleted = []
        not_found = []
        
        for relation_data in relations_data:
            from_entity = relation_data.get("from")
            to_entity = relation_data.get("to")
            relation_type = relation_data.get("relationType")
            
            if not (from_entity and to_entity and relation_type):
                not_found.append(relation_data)
                continue
            
            # Find the relation
            found = False
            for relation in self.relations:
                if (relation.from_entity == from_entity and
                    relation.to_entity == to_entity and
                    relation.relation_type == relation_type and
                    not relation.deleted):
                    relation.deleted = True
                    found = True
                    
                    # Append to JSONL
                    record = JournalRecord(
                        record_type=RecordType.RELATION_DELETE,
                        data=relation_data,
                        actor=actor
                    )
                    self._append_to_jsonl(record)
                    deleted.append(relation_data)
                    break
            
            if not found:
                not_found.append(relation_data)
        
        return {
            "deleted": deleted,
            "not_found": not_found,
            "message": f"Deleted {len(deleted)} relations. {len(not_found)} relations not found."
        }
    
    def search_nodes(self, query: str, use_semantic: bool = True, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for nodes in the knowledge graph
        
        Args:
            query: The search query
            use_semantic: Whether to use semantic search if available
            limit: Maximum number of results to return
            
        Returns:
            List of matching entities
        """
        if not query:
            return []
        
        # Use semantic search if enabled and available
        if use_semantic and self.use_embeddings and self.embedding_provider:
            return self._semantic_search(query, limit)
        else:
            return self._keyword_search(query, limit)
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search for nodes using keyword matching
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching entities
        """
        query = query.lower()
        results = []
        
        for entity in self.entities.values():
            if entity.deleted:
                continue
                
            # Create a combined text for matching
            text = f"{entity.name} {entity.entity_type} " + " ".join(entity.observations)
            text = text.lower()
            
            if query in text:
                results.append(entity.model_dump())
        
        # Sort by name and limit results
        results.sort(key=lambda x: x["name"])
        return results[:limit]
    
    def _semantic_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search for nodes using semantic similarity
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching entities
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_provider.generate_embedding(query)
            
            # Calculate similarity for each entity
            results = []
            for entity in self.entities.values():
                if entity.deleted or not entity.embedding:
                    continue
                
                # Convert embedding back to numpy array
                entity_embedding = np.array(entity.embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, entity_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entity_embedding)
                )
                
                results.append((entity, similarity))
            
            # Sort by similarity (descending) and get top matches
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return serialized entities with similarity score
            return [
                {**entity.model_dump(), "similarity": float(similarity)}
                for entity, similarity in results[:limit]
            ]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            # Fall back to keyword search
            return self._keyword_search(query, limit)
    
    def open_nodes(self, names: List[str]) -> List[Dict[str, Any]]:
        """
        Open specific nodes by name
        
        Args:
            names: List of entity names to open
            
        Returns:
            List of entity data
        """
        results = []
        
        for name in names:
            if name in self.entities and not self.entities[name].deleted:
                results.append(self.entities[name].model_dump())
        
        return results
    
    def read_graph(self) -> Dict[str, Any]:
        """
        Read the entire knowledge graph
        
        Returns:
            Dictionary with entities and relations
        """
        # Filter out deleted entities and relations
        active_entities = {
            name: entity.model_dump()
            for name, entity in self.entities.items()
            if not entity.deleted
        }
        
        active_relations = [
            relation.model_dump()
            for relation in self.relations
            if not relation.deleted
        ]
        
        return {
            "entities": active_entities,
            "relations": active_relations,
            "entity_count": len(active_entities),
            "relation_count": len(active_relations)
        }
    
    def update_entities(self, entities_data: List[Dict[str, Any]], actor: Optional[str] = None) -> Dict[str, Any]:
        """
        Update existing entities
        
        Args:
            entities_data: List of entity data dictionaries
            actor: Optional identifier for who/what is updating the entities
            
        Returns:
            Dictionary with updated and not_found entity names
        """
        updated = []
        not_found = []
        
        for entity_data in entities_data:
            name = entity_data.get("name")
            
            if not name:
                continue
                
            if name not in self.entities or self.entities[name].deleted:
                not_found.append(name)
                continue
            
            # Get existing entity and update fields
            entity = self.entities[name]
            
            # Update only provided fields
            update_data = {}
            if "entityType" in entity_data:
                update_data["entity_type"] = entity_data["entityType"]
            
            if "observations" in entity_data:
                update_data["observations"] = entity_data["observations"]
            
            # Only update if there are changes
            if update_data:
                # Update entity
                for key, value in update_data.items():
                    setattr(entity, key, value)
                    
                entity.updated_at = datetime.now()
                
                # Update embedding
                if self.use_embeddings:
                    entity.embedding = self._generate_embedding(entity)
                
                # Append to JSONL
                record = JournalRecord(
                    record_type=RecordType.ENTITY_UPDATE,
                    data={
                        "name": name,
                        **update_data,
                        "updated_at": entity.updated_at,
                        "embedding": entity.embedding
                    },
                    actor=actor
                )
                self._append_to_jsonl(record)
                
                updated.append(name)
        
        return {
            "updated": updated,
            "not_found": not_found,
            "message": f"Updated {len(updated)} entities. {len(not_found)} entities not found."
        }
    
    def delete_observations(self, deletions_data: List[Dict[str, Any]], actor: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete specific observations from entities
        
        Args:
            deletions_data: List of dictionaries with entity names and observations to delete
            actor: Optional identifier for who/what is deleting the observations
            
        Returns:
            Dictionary with updated and not_found entity names
        """
        updated = []
        not_found = []
        
        for deletion_data in deletions_data:
            entity_name = deletion_data.get("entityName")
            observations_to_delete = deletion_data.get("observations", [])
            
            if not entity_name or not observations_to_delete:
                continue
            
            if entity_name not in self.entities or self.entities[entity_name].deleted:
                not_found.append(entity_name)
                continue
            
            # Get existing entity
            entity = self.entities[entity_name]
            
            # Remove observations
            original_count = len(entity.observations)
            entity.observations = [
                obs for obs in entity.observations 
                if obs not in observations_to_delete
            ]
            
            # Only update if observations were actually removed
            if len(entity.observations) < original_count:
                entity.updated_at = datetime.now()
                
                # Update embedding
                if self.use_embeddings:
                    entity.embedding = self._generate_embedding(entity)
                
                # Append to JSONL
                record = JournalRecord(
                    record_type=RecordType.ENTITY_UPDATE,
                    data={
                        "name": entity_name,
                        "observations": entity.observations,
                        "updated_at": entity.updated_at,
                        "embedding": entity.embedding
                    },
                    actor=actor
                )
                self._append_to_jsonl(record)
                
                updated.append(entity_name)
        
        return {
            "updated": updated,
            "not_found": not_found,
            "message": f"Removed observations from {len(updated)} entities. {len(not_found)} entities not found."
        }
    
    def compact_journal(self) -> Dict[str, Any]:
        """
        Compact the journal file by removing redundant entries
        
        Returns:
            Status of the compaction operation
        """
        # Create a new temporary file
        temp_file = f"{self.memory_file_path}.temp"
        
        try:
            # Export current state to the temporary file
            with open(temp_file, 'w', encoding='utf-8') as f:
                # Write all active entities
                for entity in self.entities.values():
                    if not entity.deleted:
                        record = JournalRecord(
                            record_type=RecordType.ENTITY_CREATE,
                            data=entity.model_dump(),
                            timestamp=entity.created_at
                        )
                        f.write(json.dumps(record.model_dump()) + '\n')
                
                # Write all active relations
                for relation in self.relations:
                    if not relation.deleted:
                        record = JournalRecord(
                            record_type=RecordType.RELATION_CREATE,
                            data=relation.model_dump(),
                            timestamp=relation.created_at
                        )
                        f.write(json.dumps(record.model_dump()) + '\n')
            
            # Backup the original file
            backup_file = f"{self.memory_file_path}.bak"
            if os.path.exists(self.memory_file_path):
                os.rename(self.memory_file_path, backup_file)
            
            # Replace with the new file
            os.rename(temp_file, self.memory_file_path)
            
            return {
                "success": True,
                "message": f"Journal compacted successfully. Backup saved to {backup_file}",
                "entity_count": len([e for e in self.entities.values() if not e.deleted]),
                "relation_count": len([r for r in self.relations if not r.deleted])
            }
        except Exception as e:
            logger.error(f"Failed to compact journal: {e}")
            
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return {
                "success": False,
                "message": f"Failed to compact journal: {str(e)}"
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "entities_count": len(self.entities),
            "relations_count": len(self.relations),
            "total_records": self.total_records,
            "loaded_records": self.loaded_records,
            "fully_loaded": self.fully_loaded,
            "load_time": self.load_time,
            "last_operation_time": self.last_operation_time,
            "memory_file_size": os.path.getsize(self.memory_file_path) if os.path.exists(self.memory_file_path) else 0,
            "use_embeddings": self.use_embeddings
        }
        
        # Add embedding cache stats if available
        if self.use_embeddings and self.embedding_provider:
            stats["embedding_cache"] = self.embedding_provider.get_cache_stats()
            
        return stats
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory usage
        
        Returns:
            Dictionary with optimization results
        """
        import gc
        
        start_time = time.time()
        initial_entities = len(self.entities)
        initial_relations = len(self.relations)
        
        # Remove deleted entities and relations if we're keeping them
        if self.load_deleted:
            self.entities = {name: entity for name, entity in self.entities.items() if not entity.deleted}
            self.relations = [relation for relation in self.relations if not relation.deleted]
        
        # Optimize embedding cache if available
        if self.use_embeddings and self.embedding_provider:
            self.embedding_provider.optimize_memory()
        
        # Force garbage collection
        gc.collect()
        
        optimization_time = time.time() - start_time
        
        return {
            "initial_entities": initial_entities,
            "initial_relations": initial_relations,
            "current_entities": len(self.entities),
            "current_relations": len(self.relations),
            "entities_removed": initial_entities - len(self.entities),
            "relations_removed": initial_relations - len(self.relations),
            "optimization_time": optimization_time,
            "message": f"Memory optimized in {optimization_time:.2f}s"
        } 