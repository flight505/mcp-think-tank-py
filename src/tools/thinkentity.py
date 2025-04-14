#!/usr/bin/env python3
"""
ThinkEntity for MCP Think Tank
Provides class implementation for thought entities
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

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
                        "tags": ["important", "decision"],
                    },
                    "entity_id": "thought_12345-uuid",
                    "metadata": {"source": "user_query", "importance": "high"},
                }
            ]
        },
        # Configure datetime serialization to use ISO format
        "json_encoders": {
            datetime: lambda dt: dt.isoformat(),
        }
    }

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate entity_id if provided."""
        if v is not None and not v.strip():
            raise ValueError("entity_id cannot be empty string")
        return v

    @model_validator(mode="after")
    def validate_thought(self) -> "ThinkEntity":
        """Validate that the thought has required fields."""
        if not self.thought.structured_reasoning or not self.thought.structured_reasoning.strip():
            raise ValueError("thought must have non-empty structured_reasoning")
        return self

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"ThinkEntity(id={self.thought.id}, "
            f"entity_id={self.entity_id or 'None'}, "
            f"category={self.thought.category or 'None'}, "
            f"created_at={self.created_at.isoformat()})"
        )

    @classmethod
    def from_thought(
        cls, thought: Thought, metadata: Optional[Dict[str, Any]] = None
    ) -> "ThinkEntity":
        """
        Create a ThinkEntity from a Thought object

        Args:
            thought: The existing Thought object
            metadata: Optional metadata to associate with the entity

        Returns:
            A new ThinkEntity instance
        """
        if not thought:
            raise ValueError("thought cannot be None")
        return cls(thought=thought, metadata=metadata or {})

    @classmethod
    def create(
        cls,
        structured_reasoning: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        entity_association: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ThinkEntity":
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
        if not structured_reasoning or not structured_reasoning.strip():
            raise ValueError("structured_reasoning cannot be empty")
            
        thought = Thought(
            structured_reasoning=structured_reasoning,
            category=category,
            tags=tags or [],
            entity_association=entity_association,
        )

        return cls(thought=thought, metadata=metadata or {})

    def to_kg_entity(self) -> Dict[str, Any]:
        """
        Convert to a knowledge graph entity format

        Returns:
            Dictionary in the format expected by knowledge_graph.create_entities()
        """
        # Prepare observations
        observations = [
            f"Reasoning: {self.thought.structured_reasoning}",
            (
                f"Category: {self.thought.category}"
                if self.thought.category
                else "No category specified"
            ),
            (
                f"Tags: {', '.join(self.thought.tags)}"
                if self.thought.tags
                else "No tags specified"
            ),
            f"Timestamp: {self.thought.timestamp}",
        ]

        # Add metadata as observations
        if self.metadata:
            for key, value in self.metadata.items():
                observations.append(f"Metadata - {key}: {value}")

        # If thought has a reflection, add it
        if self.thought.reflection:
            observations.append(f"Reflection: {self.thought.reflection}")
            observations.append(
                f"Reflection timestamp: {self.thought.reflection_timestamp}"
            )

        # Create entity dictionary
        entity_name = self.entity_id or f"thought_{self.thought.id}"
        entity = {
            "name": entity_name,
            "entityType": "thought",
            "observations": observations,
        }

        return entity

    def update_entity_id(self, entity_id: str) -> None:
        """
        Update the entity ID after storing in knowledge graph

        Args:
            entity_id: The ID assigned in the knowledge graph
        """
        if not entity_id or not entity_id.strip():
            raise ValueError("entity_id cannot be empty")
            
        self.entity_id = entity_id
        self.updated_at = datetime.now()

    def add_reflection(self, reflection: str) -> None:
        """
        Add a reflection to the thought

        Args:
            reflection: The reflection text
        """
        if not reflection or not reflection.strip():
            raise ValueError("reflection cannot be empty")
            
        self.thought.reflection = reflection
        self.thought.reflection_timestamp = datetime.now().isoformat()
        self.updated_at = datetime.now()

    def model_dump_json(self, **kwargs) -> str:
        """
        Serialize the model to JSON with proper datetime handling

        Args:
            **kwargs: Additional arguments to pass to the model_dump_json method

        Returns:
            JSON string representation of the model
        """
        # Ensure datetime fields are serialized properly
        kwargs.setdefault("indent", 2)
        return super().model_dump_json(**kwargs)
