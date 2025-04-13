#!/usr/bin/env python3
"""
ThinkEntity for MCP Think Tank
Provides class implementation for thought entities
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Import the Thought model from think.py
from src.tools.think import Thought

logger = logging.getLogger("mcp-think-tank.thinkentity")

class ThinkEntity(BaseModel):
    """
    Entity model for thought data with additional metadata.
    
    This extends the capabilities of the Thought model by adding
    knowledge graph integration and additional metadata tracking.
    """
    thought: Thought
    entity_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "thought": {
                        "id": "12345-uuid",
                        "structured_reasoning": "This is structured reasoning about a topic",
                        "category": "analysis",
                        "tags": ["important", "decision"]
                    },
                    "entity_id": "thought_12345-uuid",
                    "metadata": {
                        "source": "user_query",
                        "importance": "high"
                    }
                }
            ]
        }
    }
    
    @classmethod
    def from_thought(cls, thought: Thought, metadata: Optional[Dict[str, Any]] = None) -> "ThinkEntity":
        """
        Create a ThinkEntity from a Thought object
        
        Args:
            thought: The existing Thought object
            metadata: Optional metadata to associate with the entity
            
        Returns:
            A new ThinkEntity instance
        """
        return cls(
            thought=thought,
            metadata=metadata or {}
        )
    
    @classmethod
    def create(cls, 
              structured_reasoning: str,
              category: Optional[str] = None,
              tags: Optional[List[str]] = None,
              entity_association: Optional[str] = None,
              metadata: Optional[Dict[str, Any]] = None) -> "ThinkEntity":
        """
        Create a new ThinkEntity directly from parameters
        
        Args:
            structured_reasoning: The structured reasoning text
            category: Optional category for the thought
            tags: Optional tags for the thought
            entity_association: Optional entity to associate with
            metadata: Optional metadata to associate with the entity
            
        Returns:
            A new ThinkEntity instance
        """
        thought = Thought(
            structured_reasoning=structured_reasoning,
            category=category,
            tags=tags or [],
            entity_association=entity_association
        )
        
        return cls(
            thought=thought,
            metadata=metadata or {}
        )
    
    def to_kg_entity(self) -> Dict[str, Any]:
        """
        Convert to a knowledge graph entity format
        
        Returns:
            Dictionary in the format expected by knowledge_graph.create_entities()
        """
        # Prepare observations
        observations = [
            f"Reasoning: {self.thought.structured_reasoning}",
            f"Category: {self.thought.category}" if self.thought.category else "No category specified",
            f"Tags: {', '.join(self.thought.tags)}" if self.thought.tags else "No tags specified",
            f"Timestamp: {self.thought.timestamp}"
        ]
        
        # Add metadata as observations
        if self.metadata:
            for key, value in self.metadata.items():
                observations.append(f"Metadata - {key}: {value}")
        
        # If thought has a reflection, add it
        if self.thought.reflection:
            observations.append(f"Reflection: {self.thought.reflection}")
            observations.append(f"Reflection timestamp: {self.thought.reflection_timestamp}")
        
        # Create entity dictionary
        entity_name = self.entity_id or f"thought_{self.thought.id}"
        entity = {
            "name": entity_name,
            "entityType": "thought",
            "observations": observations
        }
        
        return entity
    
    def update_entity_id(self, entity_id: str) -> None:
        """
        Update the entity ID after storing in knowledge graph
        
        Args:
            entity_id: The ID assigned in the knowledge graph
        """
        self.entity_id = entity_id
        self.updated_at = datetime.now()
        
    def add_reflection(self, reflection: str) -> None:
        """
        Add a reflection to the thought
        
        Args:
            reflection: The reflection text
        """
        self.thought.reflection = reflection
        self.thought.reflection_timestamp = datetime.now().isoformat()
        self.updated_at = datetime.now()
