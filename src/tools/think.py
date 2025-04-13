#!/usr/bin/env python3
"""
Think Tool for MCP Think Tank
Enables structured reasoning, reflection, and memory storage
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger("mcp-think-tank.think")

class Thought(BaseModel):
    """Model for storing thought data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    structured_reasoning: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    entity_association: Optional[str] = None
    reflection: Optional[str] = None
    reflection_timestamp: Optional[str] = None

class ThinkTool:
    """
    Tool for structured reasoning and reflection
    
    The Think tool allows the agent to articulate structured reasoning,
    optionally store the thoughts in a knowledge graph, and perform
    reflections on those thoughts.
    """
    
    def __init__(self, knowledge_graph=None, sample_func: Optional[Callable] = None):
        """
        Initialize the Think tool
        
        Args:
            knowledge_graph: Knowledge graph instance for storing thoughts
            sample_func: Function to sample from a model for reflections
        """
        self.knowledge_graph = knowledge_graph
        self.sample_func = sample_func
        self.thoughts = []  # In-memory storage when KG is not available
        logger.info("Think tool initialized")
    
    def process(self, 
               structured_reasoning: str,
               store_in_memory: bool = False,
               reflexion: bool = False,
               category: Optional[str] = None,
               tags: Optional[List[str]] = None,
               associate_with_entity: Optional[str] = None) -> Dict[str, Any]:
        """
        Process structured reasoning and optional reflection
        
        Args:
            structured_reasoning: The structured reasoning text
            store_in_memory: Whether to store the thought in memory
            reflexion: Whether to perform reflection on the thought
            category: Optional category for the thought
            tags: Optional tags for the thought
            associate_with_entity: Optional entity to associate with
            
        Returns:
            Dict with the processed thought and any reflection
        """
        # Create thought object
        thought = Thought(
            structured_reasoning=structured_reasoning,
            category=category,
            tags=tags or [],
            entity_association=associate_with_entity
        )
        
        # Log the thought
        logger.info(f"Processing thought: {thought.id}")
        logger.debug(f"Thought content: {structured_reasoning[:100]}...")
        
        result = {
            "thought_id": thought.id,
            "timestamp": thought.timestamp,
            "stored_in_memory": False
        }
        
        # Store in memory if requested
        if store_in_memory:
            if self.knowledge_graph:
                memory_result = self._store_in_knowledge_graph(thought)
                result["stored_in_memory"] = True
                result["memory_entity_id"] = memory_result.get("name", thought.id)
            else:
                logger.warning("Knowledge graph not available, cannot store thought")
                # Fall back to in-memory storage
                self.thoughts.append(thought)
                result["stored_in_memory"] = True
                result["memory_type"] = "in-memory"
        
        # Perform reflection if requested and possible
        if reflexion and self.sample_func:
            try:
                # Get relevant memory context
                memory_context = self._get_memory_context(structured_reasoning) if self.knowledge_graph else []
                
                # Generate reflection
                reflection = self._reflect_on_thought(thought, memory_context)
                thought.reflection = reflection
                thought.reflection_timestamp = datetime.now().isoformat()
                
                # Store reflection in memory if the original thought was stored
                if store_in_memory and self.knowledge_graph:
                    self._store_reflection(thought)
                
                # Add reflection to result
                result["reflection"] = reflection
                result["reflection_timestamp"] = thought.reflection_timestamp
                
            except Exception as e:
                logger.error(f"Error during reflection: {e}")
                result["reflection_error"] = str(e)
        
        # Return the processed thought
        return result
    
    def _get_memory_context(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from the knowledge graph
        
        Args:
            query: The query to search for
            limit: Maximum number of results
            
        Returns:
            List of relevant context items
        """
        if not self.knowledge_graph:
            return []
        
        try:
            results = self.knowledge_graph.search_nodes(query, limit=limit)
            logger.debug(f"Retrieved {len(results)} memory items for context")
            return results
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
            return []
    
    def _store_in_knowledge_graph(self, thought: Thought) -> Dict[str, Any]:
        """
        Create an entity in the knowledge graph for the thought
        
        Args:
            thought: The thought to store
            
        Returns:
            Result of the entity creation
        """
        if not self.knowledge_graph:
            logger.warning("Knowledge graph not available, cannot store thought")
            return {"error": "Knowledge graph not available"}
        
        # Prepare observations
        observations = [
            f"Reasoning: {thought.structured_reasoning}",
            f"Category: {thought.category}" if thought.category else "No category specified",
            f"Tags: {', '.join(thought.tags)}" if thought.tags else "No tags specified",
            f"Timestamp: {thought.timestamp}"
        ]
        
        # If associated with entity, add that info
        if thought.entity_association:
            observations.append(f"Associated with entity: {thought.entity_association}")
        
        # Create entity
        entity = {
            "name": f"thought_{thought.id}",
            "entityType": "thought",
            "observations": observations
        }
        
        # Create in knowledge graph
        result = self.knowledge_graph.create_entities([entity])
        
        # If associated with an entity, create relation
        if thought.entity_association:
            try:
                relation = {
                    "from": f"thought_{thought.id}",
                    "to": thought.entity_association,
                    "relationType": "associatedWith"
                }
                self.knowledge_graph.create_relations([relation])
            except Exception as e:
                logger.error(f"Error creating association relation: {e}")
        
        return result.get("created", [{}])[0] if "created" in result else {}
    
    def _store_reflection(self, thought: Thought) -> Dict[str, Any]:
        """
        Store a reflection on a thought in the knowledge graph
        
        Args:
            thought: The thought with a reflection
            
        Returns:
            Result of the entity and relation creation
        """
        if not thought.reflection or not self.knowledge_graph:
            return {"error": "No reflection or knowledge graph available"}
        
        # Create reflection entity
        reflection_entity = {
            "name": f"reflection_{thought.id}",
            "entityType": "reflection",
            "observations": [
                f"Reflection: {thought.reflection}",
                f"Timestamp: {thought.reflection_timestamp}"
            ]
        }
        
        # Create entity in knowledge graph
        result = self.knowledge_graph.create_entities([reflection_entity])
        
        # Create relation to thought
        relation = {
            "from": f"reflection_{thought.id}",
            "to": f"thought_{thought.id}",
            "relationType": "reflectsOn"
        }
        relation_result = self.knowledge_graph.create_relations([relation])
        
        return {
            "entity_result": result,
            "relation_result": relation_result
        }
    
    def _reflect_on_thought(self, thought: Thought, memory_context: List[Dict[str, Any]]) -> str:
        """
        Generate a reflection based on the thought and memory context
        
        Args:
            thought: The thought to reflect on
            memory_context: Relevant context from memory
            
        Returns:
            Reflection text
        """
        if not self.sample_func:
            return "Reflection unavailable: no sample function provided"
        
        # Build a prompt for reflection
        prompt = self._build_reflection_prompt(thought, memory_context)
        
        # Sample from the model
        try:
            reflection = self.sample_func(prompt)
            return reflection
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            return f"Error generating reflection: {str(e)}"
    
    def _build_reflection_prompt(self, thought: Thought, memory_context: List[Dict[str, Any]]) -> str:
        """
        Build a prompt for reflection on a thought
        
        Args:
            thought: The thought to reflect on
            memory_context: Relevant context from memory
            
        Returns:
            Prompt for reflection
        """
        # Start with the core prompt
        prompt = [
            "# Self-Reflection",
            "",
            "Below is a thought that was previously generated. Your task is to reflect on this thought,",
            "identifying strengths, weaknesses, and potential improvements. Consider logical consistency,",
            "completeness, and alternative perspectives.",
            "",
            "## Original Thought",
            thought.structured_reasoning,
            "",
        ]
        
        # Add category and tags if present
        if thought.category:
            prompt.append(f"Category: {thought.category}")
            prompt.append("")
        
        if thought.tags:
            prompt.append(f"Tags: {', '.join(thought.tags)}")
            prompt.append("")
        
        # Add memory context if available
        if memory_context:
            prompt.append("## Relevant Memory Context")
            prompt.append("")
            
            for i, item in enumerate(memory_context, 1):
                name = item.get("name", f"Item {i}")
                entity_type = item.get("entityType", "unknown")
                observations = item.get("observations", [])
                
                prompt.append(f"### {name} ({entity_type})")
                for obs in observations:
                    prompt.append(f"- {obs}")
                prompt.append("")
        
        # Add reflection instructions
        prompt.extend([
            "## Reflection Instructions",
            "",
            "1. Assess the clarity and structure of the original thought",
            "2. Evaluate the logical consistency and completeness of the reasoning",
            "3. Identify any biases or assumptions that may affect the conclusions",
            "4. Consider alternative perspectives or approaches not explored in the original thought",
            "5. Suggest specific improvements or extensions to the reasoning",
            "",
            "## Your Reflection"
        ])
        
        return "\n".join(prompt) 