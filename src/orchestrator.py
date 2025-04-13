#!/usr/bin/env python3
"""
Orchestrator for MCP Think Tank
Manages cross-tool intelligence and coordination
"""
import logging
from typing import Dict, List, Optional, Any

from fastmcp import FastMCP

# Local imports
from .config import get_config
from .tools.memory import KnowledgeGraph

logger = logging.getLogger("mcp-think-tank.orchestrator")

class Orchestrator:
    """
    Orchestrates interactions between different tools and components
    
    The orchestrator ensures tools can communicate with each other
    and coordinates complex multi-step operations
    """
    
    def __init__(self, mcp: FastMCP):
        """Initialize the orchestrator with an MCP server"""
        self.mcp = mcp
        self.tools = {}
        logger.info("Orchestrator initialized")
        
        # Load configuration
        self.config = get_config()
        
        # Initialize component references
        self.kg = None  # Knowledge Graph
        self.think_tool = None  # Think Tool
        self.task_manager = None  # Task Manager
        
        # Initialize knowledge graph
        self._init_knowledge_graph()
    
    def _init_knowledge_graph(self):
        """Initialize the knowledge graph component"""
        try:
            self.kg = KnowledgeGraph(
                memory_file_path=self.config.memory_file_path,
                use_embeddings=self.config.use_embeddings
            )
            logger.info("Knowledge graph initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {e}")
            # Create a fallback in-memory knowledge graph
            self.kg = KnowledgeGraph(
                memory_file_path=self.config.memory_file_path,
                use_embeddings=False
            )
            logger.warning("Using fallback in-memory knowledge graph without embeddings")
    
    def register_tools(self) -> None:
        """Register all tools and resources with the MCP server"""
        logger.info("Registering tools with MCP server")
        
        # Initialize and register tools
        self._register_memory_tools()
        self._register_think_tools()
        self._register_task_tools()
        self._register_orchestrator_workflows()
        
        logger.info(f"Registered {len(self.tools)} tools with MCP server")
    
    def _register_memory_tools(self) -> None:
        """Register knowledge graph and memory tools"""
        logger.info("Registering memory tools")
        
        # Create entities
        @self.mcp.tool()
        def create_entities(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            Create multiple new entities in the knowledge graph
            
            Args:
                entities: Array of entities to create with name, entityType, and observations
                
            Returns:
                Dict with created and existing entity names
            """
            result = self.kg.create_entities(entities)
            self.tools["create_entities"] = "memory"
            return result
        
        # Create relations
        @self.mcp.tool()
        def create_relations(relations: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            Create multiple new relations between entities in the knowledge graph
            
            Args:
                relations: Array of relations to create with from, to, and relationType
                
            Returns:
                Dict with created and failed relations
            """
            result = self.kg.create_relations(relations)
            self.tools["create_relations"] = "memory"
            return result
        
        # Add observations
        @self.mcp.tool()
        def add_observations(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            Add new observations to existing entities in the knowledge graph
            
            Args:
                observations: Array of entity observations to add with entityName and contents
                
            Returns:
                Dict with updated and not_found entity names
            """
            result = self.kg.add_observations(observations)
            self.tools["add_observations"] = "memory"
            return result
        
        # Delete entities
        @self.mcp.tool()
        def delete_entities(entityNames: List[str]) -> Dict[str, Any]:
            """
            Delete multiple entities and their associated relations from the knowledge graph
            
            Args:
                entityNames: Array of entity names to delete
                
            Returns:
                Dict with deleted and not_found entity names
            """
            result = self.kg.delete_entities(entityNames)
            self.tools["delete_entities"] = "memory"
            return result
        
        # Delete observations
        @self.mcp.tool()
        def delete_observations(deletions: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            Delete specific observations from entities in the knowledge graph
            
            Args:
                deletions: Array of entity observations to delete with entityName and observations
                
            Returns:
                Dict with updated and not_found entity names
            """
            result = self.kg.delete_observations(deletions)
            self.tools["delete_observations"] = "memory"
            return result
        
        # Delete relations
        @self.mcp.tool()
        def delete_relations(relations: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            Delete multiple relations from the knowledge graph
            
            Args:
                relations: Array of relations to delete with from, to, and relationType
                
            Returns:
                Dict with deleted and not_found relations
            """
            result = self.kg.delete_relations(relations)
            self.tools["delete_relations"] = "memory"
            return result
        
        # Read entire graph
        @self.mcp.tool()
        def read_graph(dummy: str = "") -> Dict[str, Any]:
            """
            Read the entire knowledge graph
            
            Returns:
                Dict with entities and relations
            """
            result = self.kg.read_graph()
            self.tools["read_graph"] = "memory"
            return result
        
        # Search nodes
        @self.mcp.tool()
        def search_nodes(query: str) -> List[Dict[str, Any]]:
            """
            Search for nodes in the knowledge graph based on a query
            
            Args:
                query: Search query to find matching entities
                
            Returns:
                List of matching entities
            """
            result = self.kg.search_nodes(query)
            self.tools["search_nodes"] = "memory"
            return result
        
        # Open nodes
        @self.mcp.tool()
        def open_nodes(names: List[str]) -> List[Dict[str, Any]]:
            """
            Open specific nodes in the knowledge graph by their names
            
            Args:
                names: Array of entity names to retrieve
                
            Returns:
                List of entity data
            """
            result = self.kg.open_nodes(names)
            self.tools["open_nodes"] = "memory"
            return result
        
        # Update entities
        @self.mcp.tool()
        def update_entities(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            Update multiple existing entities in the knowledge graph
            
            Args:
                entities: Array of entities to update with name and optional entityType/observations
                
            Returns:
                Dict with updated and not_found entity names
            """
            result = self.kg.update_entities(entities)
            self.tools["update_entities"] = "memory"
            return result
        
        # Update relations is not needed since we can delete and recreate them
    
    def _register_think_tools(self) -> None:
        """Register thinking and reasoning tools"""
        logger.info("Think tools registration placeholder")
        
        # Will implement in future steps
    
    def _register_task_tools(self) -> None:
        """Register task management tools"""
        logger.info("Task tools registration placeholder")
        
        # Will implement in future steps
    
    def _register_orchestrator_workflows(self) -> None:
        """Register orchestrated workflows that combine multiple tools"""
        logger.info("Orchestrator workflows registration placeholder")
        
        # Will implement in future steps

    def auto_retrieve_memory(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Automatically retrieve relevant memory entries based on a query
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of relevant memory entries
        """
        if not self.kg:
            logger.warning("Knowledge graph not initialized, cannot retrieve memory")
            return []
            
        try:
            return self.kg.search_nodes(query, limit=limit)
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return [] 