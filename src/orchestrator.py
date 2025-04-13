#!/usr/bin/env python3
"""
Orchestrator for MCP Think Tank
Manages cross-tool intelligence and coordination
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any

from fastmcp import FastMCP

# Local imports
from .config import get_config
from .tools.memory import KnowledgeGraph
from .tools.think import ThinkTool
from .tools.tasks import TaskManager, download_local_model
from .watchers.file_watcher import FileWatcher
from .tools.code_tools import CodeTools

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
        self.file_watcher = None  # File Watcher
        self.code_tools = None  # Code Tools
        
        # Initialize knowledge graph
        self._init_knowledge_graph()
        
        # Initialize think tool
        self._init_think_tool()
        
        # Initialize task manager
        self._init_task_manager()
        
        # Initialize file watcher
        self._init_file_watcher()
        
        # Initialize code tools
        self._init_code_tools()
    
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
    
    def _init_think_tool(self):
        """Initialize the think tool component"""
        try:
            # The sample_func will be implemented later with Anthropic API
            self.think_tool = ThinkTool(knowledge_graph=self.kg, sample_func=None)
            logger.info("Think tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize think tool: {e}")
            # Create a basic think tool without advanced features
            self.think_tool = ThinkTool(knowledge_graph=self.kg)
            logger.warning("Using basic think tool without reflection capabilities")
    
    def _init_task_manager(self):
        """Initialize the task manager component"""
        try:
            # Get Anthropic API key from config
            anthropic_api_key = getattr(self.config, "anthropic_api_key", None)
            
            # Initialize task manager with knowledge graph and API key
            self.task_manager = TaskManager(
                knowledge_graph=self.kg,
                anthropic_api_key=anthropic_api_key
            )
            logger.info("Task manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize task manager: {e}")
            # Create a basic task manager without LLM capabilities
            self.task_manager = TaskManager(knowledge_graph=self.kg)
            logger.warning("Using basic task manager without advanced parsing capabilities")
    
    def _init_file_watcher(self):
        """Initialize the file watcher component"""
        try:
            # Get file watcher config
            project_path = getattr(self.config, "project_path", ".")
            file_patterns = getattr(self.config, "file_patterns", ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx"])
            exclude_patterns = getattr(self.config, "exclude_patterns", ["**/node_modules/**", "**/.git/**", "**/venv/**"])
            polling_interval = getattr(self.config, "polling_interval", 10)
            
            # Initialize file watcher
            self.file_watcher = FileWatcher(
                project_path=project_path,
                knowledge_graph=self.kg,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                polling_interval=polling_interval
            )
            logger.info("File watcher initialized successfully")
            
            # Start file watcher in background
            self.file_watcher.start()
        except Exception as e:
            logger.error(f"Failed to initialize file watcher: {e}")
            # Create a basic file watcher without file watching capabilities
            self.file_watcher = FileWatcher(
                project_path=".",
                knowledge_graph=self.kg,
                start_watching=False
            )
            logger.warning("Using basic file watcher without file watching capabilities")
    
    def _init_code_tools(self):
        """Initialize the code tools component"""
        try:
            if not self.file_watcher:
                raise ValueError("File watcher not initialized")
                
            # Initialize code tools
            self.code_tools = CodeTools(file_watcher=self.file_watcher)
            logger.info("Code tools initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize code tools: {e}")
            logger.warning("Code tools not available")
    
    def register_tools(self) -> None:
        """Register all tools and resources with the MCP server"""
        logger.info("Registering tools with MCP server")
        
        # Initialize and register tools
        self._register_memory_tools()
        self._register_think_tools()
        self._register_task_tools()
        self._register_file_tools()
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
        logger.info("Registering think tools")
        
        if not self.think_tool:
            logger.warning("Think tool not initialized, skipping registration")
            return
        
        @self.mcp.tool()
        def think(structuredReasoning: str, 
                 storeInMemory: bool = False, 
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 associateWithEntity: Optional[str] = None) -> Dict[str, Any]:
            """
            Use the tool to think about something. 
            
            It will not obtain new information or change the database, but just append the thought to the log.
            Use it when complex reasoning or some cache memory is needed.
            
            For best results, structure your reasoning with:
            1) Problem definition
            2) Relevant facts/context
            3) Analysis steps
            4) Conclusion/decision
            
            Args:
                structuredReasoning: A structured thought process to work through complex problems
                storeInMemory: Whether to store this thought in the knowledge graph
                category: Optional category for the thought (e.g., "problem-solving", "analysis", "planning")
                tags: Optional tags to help categorize and find this thought later
                associateWithEntity: Optional entity name to associate this thought with
                
            Returns:
                Dictionary with the processed thought and additional information
            """
            result = self.think_tool.process(
                structured_reasoning=structuredReasoning,
                store_in_memory=storeInMemory,
                reflexion=self.config.enable_reflexion,
                category=category,
                tags=tags,
                associate_with_entity=associateWithEntity
            )
            self.tools["think"] = "think"
            return result
    
    def _register_task_tools(self) -> None:
        """Register task management tools"""
        logger.info("Registering task management tools")
        
        if not self.task_manager:
            logger.warning("Task manager not initialized, skipping registration")
            return
        
        @self.mcp.tool()
        def create_task(title: str,
                       description: str = "",
                       priority: str = "medium",
                       tags: Optional[List[str]] = None,
                       dependencies: Optional[List[str]] = None,
                       assigned_to: Optional[str] = None,
                       parent_id: Optional[str] = None) -> Dict[str, Any]:
            """
            Create a new task
            
            Args:
                title: Task title
                description: Task description
                priority: Task priority (low, medium, high, critical)
                tags: Optional list of tags
                dependencies: Optional list of task IDs this task depends on
                assigned_to: Optional assignee
                parent_id: Optional parent task ID
                
            Returns:
                Dict with the created task info
            """
            result = self.task_manager.create_task(
                title=title,
                description=description,
                priority=priority,
                tags=tags,
                dependencies=dependencies,
                assigned_to=assigned_to,
                parent_id=parent_id
            )
            self.tools["create_task"] = "tasks"
            return result
        
        @self.mcp.tool()
        def list_tasks(status: Optional[str] = None,
                      priority: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      assigned_to: Optional[str] = None) -> List[Dict[str, Any]]:
            """
            List tasks with optional filtering
            
            Args:
                status: Optional status filter (todo, in_progress, blocked, done, cancelled)
                priority: Optional priority filter (low, medium, high, critical)
                tags: Optional tags filter (tasks must have at least one of these tags)
                assigned_to: Optional assignee filter
                
            Returns:
                List of matching tasks
            """
            result = self.task_manager.list_tasks(
                status=status,
                priority=priority,
                tags=tags,
                assigned_to=assigned_to
            )
            self.tools["list_tasks"] = "tasks"
            return result
        
        @self.mcp.tool()
        def update_task(task_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
            """
            Update an existing task
            
            Args:
                task_id: ID of the task to update
                updates: Dictionary of fields to update. Can include:
                  - title: New task title
                  - description: New task description
                  - status: New status (todo, in_progress, blocked, done, cancelled)
                  - priority: New priority (low, medium, high, critical)
                  - tags: New list of tags
                  - dependencies: New list of task IDs this task depends on
                  - assigned_to: New assignee
                  - parent_id: New parent task ID
                
            Returns:
                Dict with updated task info or error
            """
            result = self.task_manager.update_task(
                task_id=task_id,
                updates=updates
            )
            self.tools["update_task"] = "tasks"
            return result
        
        @self.mcp.tool()
        def delete_task(task_id: str) -> Dict[str, Any]:
            """
            Delete a task
            
            Args:
                task_id: ID of the task to delete
                
            Returns:
                Dict with result of the deletion
            """
            result = self.task_manager.delete_task(task_id=task_id)
            self.tools["delete_task"] = "tasks"
            return result
        
        @self.mcp.tool()
        async def parse_tasks_from_text(text: str) -> List[Dict[str, Any]]:
            """
            Parse tasks from a text description using an LLM
            
            This tool uses AI to extract structured tasks from natural language descriptions,
            such as project requirements or meeting notes. It will attempt to use the 
            Anthropic API if available, or fall back to a local LLM if needed.
            
            Args:
                text: Text to parse into tasks
                
            Returns:
                List of parsed tasks
            """
            result = await self.task_manager.parse_tasks_from_text(text=text)
            self.tools["parse_tasks_from_text"] = "tasks"
            return result
    
    def _register_file_tools(self) -> None:
        """Register file system and code tools"""
        logger.info("Registering file system and code tools")
        
        if not self.code_tools or not self.file_watcher:
            logger.warning("Code tools or file watcher not initialized, skipping registration")
            return
        
        # Register code tools
        self.code_tools.register_tools(self.mcp)
        self.tools["search_code"] = "code"
        self.tools["summarize_file"] = "code"
        
        # Register file indexing tool
        @self.mcp.tool()
        def index_files(path: Optional[str] = None, 
                        patterns: Optional[List[str]] = None) -> Dict[str, Any]:
            """
            Index files in a directory to add them to the knowledge graph
            
            Args:
                path: Optional specific path to index (relative to project root)
                patterns: Optional file patterns to match (e.g., ["*.py", "*.js"])
                
            Returns:
                Dict with indexing results
            """
            try:
                result = self.file_watcher.index_files(path=path, patterns=patterns)
                self.tools["index_files"] = "files"
                return result
            except Exception as e:
                logger.error(f"Error indexing files: {e}")
                return {
                    "error": f"Error indexing files: {str(e)}",
                    "files_indexed": 0
                }
        
        # Register file changes tool
        @self.mcp.tool()
        def get_file_changes() -> Dict[str, Any]:
            """
            Get recent file changes detected by the file watcher
            
            Returns:
                Dict with recent file changes
            """
            try:
                result = self.file_watcher.get_recent_changes()
                self.tools["get_file_changes"] = "files"
                return result
            except Exception as e:
                logger.error(f"Error getting file changes: {e}")
                return {
                    "error": f"Error getting file changes: {str(e)}",
                    "changes": []
                }
        
        # Register auto-context injection tool
        @self.mcp.tool()
        def get_context_for_tool(queries: List[str], 
                                max_results: int = 3) -> Dict[str, Any]:
            """
            Get context from the codebase based on queries for auto-context injection
            
            Args:
                queries: List of search queries
                max_results: Maximum number of results to return per query
                
            Returns:
                Dictionary with context information
            """
            try:
                result = self.code_tools.get_context_for_tool(
                    queries=queries,
                    max_results=max_results
                )
                self.tools["get_context_for_tool"] = "code"
                return result
            except Exception as e:
                logger.error(f"Error getting context: {e}")
                return {
                    "error": f"Error getting context: {str(e)}",
                    "context_blocks": [],
                    "total_items": 0
                }
    
    def _register_orchestrator_workflows(self) -> None:
        """Register orchestrated workflows that combine multiple tools"""
        logger.info("Registering orchestrator workflows")
        
        @self.mcp.tool()
        async def auto_plan_workflow(text: str) -> Dict[str, Any]:
            """
            Automatically plan a workflow from text requirements
            
            This tool combines multiple steps:
            1. Parse text into tasks using LLM
            2. Create tasks in the system
            3. Reflect on the tasks for completeness
            4. Store the finalized tasks in memory
            
            Args:
                text: Text describing the requirements or project
                
            Returns:
                Dict with workflow results including created tasks
            """
            logger.info("Starting auto plan workflow")
            
            # Parse tasks
            tasks = []
            if self.task_manager:
                try:
                    # Use the async parse_tasks_from_text method
                    tasks = await self.task_manager.parse_tasks_from_text(text)
                    logger.info(f"Created {len(tasks)} tasks from requirements")
                except Exception as e:
                    logger.error(f"Error parsing tasks: {e}")
            
            # Reflect on tasks using the think tool if available
            reflection = None
            if self.think_tool and tasks:
                try:
                    # Format tasks for reflection
                    tasks_text = "\n".join([
                        f"Task {i+1}: {task.get('title')} - {task.get('description', 'No description')}"
                        for i, task in enumerate(tasks)
                    ])
                    
                    reflection_prompt = f"Review these tasks extracted from requirements and assess if they are complete and well-structured:\n\n{tasks_text}"
                    
                    reflection = self.think_tool.process(
                        structured_reasoning=reflection_prompt,
                        store_in_memory=True,
                        category="task_planning",
                        tags=["auto_plan", "task_review"]
                    )
                    logger.info("Generated reflection on tasks")
                except Exception as e:
                    logger.error(f"Error reflecting on tasks: {e}")
            
            # Return the workflow results
            self.tools["auto_plan_workflow"] = "orchestrator"
            return {
                "workflow": "auto_plan",
                "tasks_created": len(tasks),
                "tasks": tasks,
                "reflection": reflection
            }

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
    
    def cleanup(self) -> None:
        """Clean up resources and stop background processes"""
        logger.info("Cleaning up orchestrator resources")
        
        # Stop file watcher if running
        if self.file_watcher:
            try:
                self.file_watcher.stop()
                logger.info("File watcher stopped")
            except Exception as e:
                logger.error(f"Error stopping file watcher: {e}") 