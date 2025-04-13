#!/usr/bin/env python3
"""
Orchestrator for MCP Think Tank
Manages cross-tool intelligence and coordination
"""
import logging
import asyncio
import inspect
import json
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from fastmcp import FastMCP
from fastapi import FastAPI
from fastmcp.server import MCPServer, WebsocketManager

# Local imports
from .config import get_config
from .tools.memory import KnowledgeGraph, MemoryTool
from .tools.think import ThinkTool
from .tools.tasks import TaskManager, download_local_model, TasksTool
from .watchers.file_watcher import FileWatcher
from .tools.code_tools import CodeTools
from .tools.dag_orchestrator import DAGExecutor, EmbeddingCache

logger = logging.getLogger("mcp-think-tank.orchestrator")

class Orchestrator:
    """
    Orchestrates the various components of the system.
    
    This class is responsible for initializing and managing all the tools
    and services used by the system, including the FastMCP server and
    various tool instances.
    """
    
    def __init__(self, app: FastAPI, websocket_manager: WebsocketManager):
        """
        Initialize the orchestrator.
        
        Args:
            app: The FastAPI application instance
            websocket_manager: The WebsocketManager for real-time communication
        """
        self.app = app
        self.websocket_manager = websocket_manager
        self.mcp_server = MCPServer(app, websocket_manager)
        
        # Initialize tools
        self.memory_tool = self._init_memory_tool()
        self.think_tool = self._init_think_tool()
        self.tasks_tool = self._init_tasks_tool()
        
        # Initialize file watcher first, then code tools
        self.file_watcher = self._init_file_watcher()
        self.code_tools = self._init_code_tools()
        
        self.dag_executor = self._init_dag_executor()
        self.embedding_cache = self._init_embedding_cache()
        
        # Register all tools with the MCP server
        self._register_memory_tools()
        self._register_think_tools()
        self._register_tasks_tools()
        self._register_code_tools()
        self._register_dag_tools()

    def _init_memory_tool(self) -> MemoryTool:
        """Initialize the memory tool."""
        try:
            return MemoryTool()
        except Exception as e:
            logger.error(f"Failed to initialize memory tool: {e}")
            # Return a basic version or raise an exception
            return MemoryTool(use_basic=True)
    
    def _init_think_tool(self) -> ThinkTool:
        """Initialize the think tool."""
        try:
            return ThinkTool()
        except Exception as e:
            logger.error(f"Failed to initialize think tool: {e}")
            # Return a basic version or raise an exception
            return ThinkTool(use_basic=True)
    
    def _init_tasks_tool(self) -> TasksTool:
        """Initialize the tasks tool."""
        try:
            return TasksTool()
        except Exception as e:
            logger.error(f"Failed to initialize tasks tool: {e}")
            # Return a basic version or raise an exception
            return TasksTool(use_basic=True)
    
    def _init_code_tools(self) -> CodeTools:
        """Initialize the code tools.
        
        This requires the file_watcher to be initialized first.
        """
        try:
            if not self.file_watcher:
                logger.warning("File watcher not initialized, creating basic code tools")
                # Initialize with a new file watcher if needed
                project_path = os.getenv("PROJECT_PATH", os.getcwd())
                file_watcher = FileWatcher(
                    project_path=project_path,
                    knowledge_graph=self.memory_tool.knowledge_graph,
                    start_watching=False
                )
                return CodeTools(file_watcher)
            
            return CodeTools(self.file_watcher)
        except Exception as e:
            logger.error(f"Failed to initialize code tools: {e}")
            # Return a basic version or create a minimal implementation
            project_path = os.getenv("PROJECT_PATH", os.getcwd())
            try:
                file_watcher = FileWatcher(
                    project_path=project_path,
                    knowledge_graph=self.memory_tool.knowledge_graph,
                    start_watching=False
                )
                return CodeTools(file_watcher)
            except:
                # Last resort, create a minimal implementation
                logger.error("Failed to create even basic code tools, functionality will be limited")
                return None
    
    def _init_dag_executor(self) -> DAGExecutor:
        """Initialize the DAG executor."""
        try:
            return DAGExecutor(
                max_concurrency=10,
                global_timeout=None,
                fail_fast=False,
                metrics_enabled=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize DAG executor: {e}")
            # Return a basic version
            return DAGExecutor(max_concurrency=5, metrics_enabled=False)
    
    def _init_embedding_cache(self) -> EmbeddingCache:
        """Initialize the embedding cache."""
        try:
            return EmbeddingCache(max_size=1000)
        except Exception as e:
            logger.error(f"Failed to initialize embedding cache: {e}")
            # Return a basic version
            return EmbeddingCache(max_size=100)
    
    def _init_file_watcher(self) -> FileWatcher:
        """Initialize the file watcher."""
        try:
            # Get project path from environment or use default
            project_path = os.getenv("PROJECT_PATH", os.getcwd())
            
            # Initialize with proper configuration
            file_watcher = FileWatcher(
                project_path=project_path,
                knowledge_graph=self.memory_tool.knowledge_graph,
                file_patterns=["*.py", "*.md", "*.js", "*.ts", "*.html", "*.css"],
                exclude_patterns=["__pycache__/*", "*.pyc", "node_modules/*", ".git/*"],
                polling_interval=5.0
            )
            
            # Start the file watcher in the background
            file_watcher.start()
            return file_watcher
            
        except Exception as e:
            logger.error(f"Failed to initialize file watcher: {e}")
            return None
    
    def _register_memory_tools(self) -> None:
        """Register memory-related tools with the MCP server."""
        self.mcp_server.register_tools(
            tool_name="think-tool",
            tool_description="Knowledge graph management tools",
            functions={
                "create_entities": self.memory_tool.create_entities,
                "create_relations": self.memory_tool.create_relations,
                "add_observations": self.memory_tool.add_observations,
                "delete_entities": self.memory_tool.delete_entities,
                "delete_observations": self.memory_tool.delete_observations,
                "delete_relations": self.memory_tool.delete_relations,
                "read_graph": self.memory_tool.read_graph,
                "search_nodes": self.memory_tool.search_nodes,
                "open_nodes": self.memory_tool.open_nodes,
                "update_entities": self.memory_tool.update_entities,
                "update_relations": self.memory_tool.update_relations,
            }
        )
    
    def _register_think_tools(self) -> None:
        """Register think tools with the MCP server."""
        self.mcp_server.register_tools(
            tool_name="think-tool",
            tool_description="Thinking and reasoning tools",
            functions={
                "think": {
                    "function": self.think_tool.think,
                    "description": "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed. For best results, structure your reasoning with: 1) Problem definition, 2) Relevant facts/context, 3) Analysis steps, 4) Conclusion/decision.",
                    "parameters": {
                        "structuredReasoning": {
                            "type": "string",
                            "description": "A structured thought process to work through complex problems. Use this as a dedicated space for reasoning step-by-step.",
                            "required": True,
                            "minLength": 10
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category for the thought (e.g., \"problem-solving\", \"analysis\", \"planning\")",
                            "required": False
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags to help categorize and find this thought later",
                            "required": False
                        },
                        "associateWithEntity": {
                            "type": "string",
                            "description": "Optional entity name to associate this thought with",
                            "required": False
                        },
                        "storeInMemory": {
                            "type": "boolean",
                            "description": "Whether to store this thought in the knowledge graph memory",
                            "required": False,
                            "default": False
                        }
                    }
                }
            }
        )
    
    def _register_tasks_tools(self) -> None:
        """Register task management tools with the MCP server."""
        self.mcp_server.register_tools(
            tool_name="mcp-taskmanager",
            tool_description="Task management tools",
            functions={
                "request_planning": self.tasks_tool.request_planning,
                "get_next_task": self.tasks_tool.get_next_task,
                "mark_task_done": self.tasks_tool.mark_task_done,
                "approve_task_completion": self.tasks_tool.approve_task_completion,
                "approve_request_completion": self.tasks_tool.approve_request_completion,
                "open_task_details": self.tasks_tool.open_task_details,
                "list_requests": self.tasks_tool.list_requests,
                "add_tasks_to_request": self.tasks_tool.add_tasks_to_request,
                "update_task": self.tasks_tool.update_task,
                "delete_task": self.tasks_tool.delete_task
            }
        )
    
    def _register_code_tools(self) -> None:
        """Register code analysis and manipulation tools with the MCP server."""
        if not self.code_tools:
            logger.warning("Code tools not initialized, skipping tool registration")
            return
            
        self.mcp_server.register_tools(
            tool_name="code-tool",
            tool_description="Code analysis and manipulation tools",
            functions={
                "search_code": self.code_tools.search_code,
                "summarize_file": self.code_tools.summarize_file
            }
        )
    
    def _register_dag_tools(self) -> None:
        """
        Register DAG orchestration tools with the MCP server.
        
        This method registers tools for creating and running directed acyclic
        graphs (DAGs) of tasks, with support for dependency management,
        parallel execution, timeout handling, and error recovery.
        """
        async def create_workflow(name: str, description: str = "") -> Dict[str, Any]:
            """
            Create a new workflow with the given name and description.
            
            Args:
                name: Name of the workflow
                description: Optional description of the workflow
                
            Returns:
                Dict containing the workflow ID and other metadata
            """
            workflow_id = f"workflow_{len(self._workflows) + 1}"
            self._workflows[workflow_id] = {
                "id": workflow_id,
                "name": name,
                "description": description,
                "dag_executor": DAGExecutor(
                    max_concurrency=10,
                    metrics_enabled=True
                ),
                "created_at": str(datetime.now()),
                "status": "created"
            }
            return {"workflow_id": workflow_id, "name": name, "status": "created"}
        
        async def add_task_to_workflow(
            workflow_id: str,
            task_id: str,
            tool_name: str,
            function_name: str,
            parameters: Dict[str, Any],
            dependencies: List[str] = None,
            timeout: Optional[float] = None,
            retry_count: int = 0,
            description: str = ""
        ) -> Dict[str, Any]:
            """
            Add a task to a workflow.
            
            Args:
                workflow_id: ID of the workflow to add the task to
                task_id: ID for the task (must be unique within the workflow)
                tool_name: Name of the tool to use (e.g., "memory-tool")
                function_name: Name of the function to call (e.g., "create_entities")
                parameters: Parameters to pass to the function
                dependencies: List of task IDs that must complete before this task
                timeout: Maximum execution time in seconds (None for no timeout)
                retry_count: Number of retry attempts on failure
                description: Human-readable description of the task
                
            Returns:
                Dict containing task metadata
            """
            if workflow_id not in self._workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = self._workflows[workflow_id]
            dag_executor = workflow["dag_executor"]
            
            # Get the function to call
            tool_function = self._get_tool_function(tool_name, function_name)
            if not tool_function:
                return {"error": f"Function {function_name} not found in tool {tool_name}"}
            
            # Add the task to the DAG
            dag_executor.add_task(
                task_id=task_id,
                func=tool_function,
                kwargs=parameters,
                dependencies=dependencies or [],
                timeout=timeout,
                retry_count=retry_count,
                description=description
            )
            
            return {
                "workflow_id": workflow_id,
                "task_id": task_id,
                "status": "added",
                "dependencies": dependencies or []
            }
        
        async def execute_workflow(workflow_id: str) -> Dict[str, Any]:
            """
            Execute all tasks in a workflow.
            
            Args:
                workflow_id: ID of the workflow to execute
                
            Returns:
                Dict containing execution results and metrics
            """
            if workflow_id not in self._workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = self._workflows[workflow_id]
            dag_executor = workflow["dag_executor"]
            
            # Update workflow status
            workflow["status"] = "running"
            
            try:
                # Execute the DAG
                results = await dag_executor.execute()
                
                # Update workflow status
                workflow["status"] = "completed"
                workflow["completed_at"] = str(datetime.now())
                
                # Get execution summary
                summary = dag_executor.get_execution_summary()
                
                return {
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "results": results,
                    "summary": summary
                }
            except Exception as e:
                # Update workflow status
                workflow["status"] = "failed"
                workflow["error"] = str(e)
                
                # Get execution summary
                summary = dag_executor.get_execution_summary()
                
                return {
                    "workflow_id": workflow_id,
                    "status": "failed",
                    "error": str(e),
                    "summary": summary
                }
        
        async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
            """
            Get the status of a workflow.
            
            Args:
                workflow_id: ID of the workflow
                
            Returns:
                Dict containing workflow status and metrics
            """
            if workflow_id not in self._workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = self._workflows[workflow_id]
            dag_executor = workflow["dag_executor"]
            
            # Get execution summary
            summary = dag_executor.get_execution_summary()
            
            return {
                "workflow_id": workflow_id,
                "name": workflow["name"],
                "status": workflow["status"],
                "created_at": workflow["created_at"],
                "completed_at": workflow.get("completed_at"),
                "summary": summary
            }
        
        async def visualize_workflow(workflow_id: str) -> Dict[str, Any]:
            """
            Generate a visualization of a workflow.
            
            Args:
                workflow_id: ID of the workflow
                
            Returns:
                Dict containing visualization data
            """
            if workflow_id not in self._workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = self._workflows[workflow_id]
            dag_executor = workflow["dag_executor"]
            
            # Get DAG visualization
            visualization = dag_executor.visualize()
            
            return {
                "workflow_id": workflow_id,
                "name": workflow["name"],
                "visualization": visualization
            }
        
        # Initialize workflows dictionary
        self._workflows = {}
        
        # Register DAG orchestration tools
        self.mcp_server.register_tools(
            tool_name="dag-orchestrator",
            tool_description="Tools for orchestrating complex workflows with dependencies",
            functions={
                "create_workflow": create_workflow,
                "add_task_to_workflow": add_task_to_workflow,
                "execute_workflow": execute_workflow,
                "get_workflow_status": get_workflow_status,
                "visualize_workflow": visualize_workflow
            }
        )
    
    def _get_tool_function(self, tool_name: str, function_name: str) -> Optional[Callable]:
        """
        Get a tool function by name.
        
        Args:
            tool_name: Name of the tool (e.g., "memory-tool")
            function_name: Name of the function (e.g., "create_entities")
            
        Returns:
            The function if found, None otherwise
        """
        # Map tool names to tool instances
        tool_map = {
            "think-tool": self.think_tool,
            "memory-tool": self.memory_tool,
            "mcp-taskmanager": self.tasks_tool,
            "code-tool": self.code_tools
        }
        
        if tool_name not in tool_map:
            logger.warning(f"Tool {tool_name} not found")
            return None
        
        tool = tool_map[tool_name]
        
        # Check if tool is None
        if tool is None:
            logger.warning(f"Tool {tool_name} is None")
            return None
        
        # Get the function from the tool
        if hasattr(tool, function_name):
            return getattr(tool, function_name)
        
        logger.warning(f"Function {function_name} not found in tool {tool_name}")
        return None
    
    def shutdown(self) -> None:
        """Shut down all components."""
        if self.file_watcher:
            self.file_watcher.stop()
        logger.info("Orchestrator shutdown complete") 