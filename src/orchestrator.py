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
import uuid

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
from .tools.workflow_templates import WorkflowFactory, WorkflowTemplate
from .tools.workflow_error_handler import WorkflowErrorHandler, TimeoutManager

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
        
        # Initialize workflow factory and active workflows
        self.workflow_factory = self._init_workflow_factory()
        self.workflows = {}
        self.error_handler = None
        self.timeout_manager = None
        
        # Register all tools with the MCP server
        self._register_memory_tools()
        self._register_think_tools()
        self._register_tasks_tools()
        self._register_code_tools()
        self._register_dag_tools()
        self._register_workflow_tools()
        self._init_error_handling()

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
    
    def _init_workflow_factory(self) -> WorkflowFactory:
        """Initialize the workflow factory."""
        try:
            return WorkflowFactory(
                memory_tool=self.memory_tool,
                think_tool=self.think_tool,
                tasks_tool=self.tasks_tool,
                code_tool=self.code_tools
            )
        except Exception as e:
            logger.error(f"Failed to initialize workflow factory: {e}")
            # Try with minimal requirements
            return WorkflowFactory(
                memory_tool=self.memory_tool,
                think_tool=self.think_tool
            )
    
    def _init_error_handling(self):
        """
        Initialize error handling components for workflows.
        
        This method sets up the WorkflowErrorHandler and TimeoutManager
        that are used for robust workflow execution with timeout
        management and graceful error recovery.
        """
        try:
            self.error_handler = WorkflowErrorHandler()
            self.timeout_manager = TimeoutManager(default_timeout=60.0)  # 60 second default timeout
            logger.info("Error handling components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize error handling: {str(e)}")
            # Create basic versions if initialization fails
            self.error_handler = WorkflowErrorHandler()
            self.timeout_manager = TimeoutManager()
    
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
    
    def _register_workflow_tools(self) -> None:
        """
        Register workflow-related tools with the MCP server.
        
        This method registers tools for creating and managing workflows
        based on predefined templates.
        """
        async def create_feature_workflow(feature_description: str, workflow_id: Optional[str] = None) -> Dict[str, Any]:
            """
            Create a workflow for implementing a new feature.
            
            Args:
                feature_description: Description of the feature to implement
                workflow_id: Optional ID for the workflow (generated if not provided)
                
            Returns:
                Dict containing workflow metadata
            """
            # Generate workflow ID if not provided
            if not workflow_id:
                workflow_id = f"feature_{len(self.workflows) + 1}"
                
            # Check if workflow ID already exists
            if workflow_id in self.workflows:
                return {"error": f"Workflow ID '{workflow_id}' already exists"}
                
            try:
                # Create and configure the workflow
                workflow = self.workflow_factory.create_feature_workflow(feature_description)
                
                # Store the workflow
                self.workflows[workflow_id] = workflow
                
                logger.info(f"Created feature implementation workflow '{workflow_id}' for: {feature_description}")
                
                return {
                    "workflow_id": workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "feature_description": feature_description,
                    "status": workflow.status,
                    "created_at": workflow.created_at.isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to create feature workflow: {e}")
                return {"error": f"Failed to create feature workflow: {str(e)}"}
        
        async def create_bugfix_workflow(bug_description: str, error_logs: Optional[str] = None, workflow_id: Optional[str] = None) -> Dict[str, Any]:
            """
            Create a workflow for fixing a bug.
            
            Args:
                bug_description: Description of the bug to fix
                error_logs: Optional error logs related to the bug
                workflow_id: Optional ID for the workflow (generated if not provided)
                
            Returns:
                Dict containing workflow metadata
            """
            # Generate workflow ID if not provided
            if not workflow_id:
                workflow_id = f"bugfix_{len(self.workflows) + 1}"
                
            # Check if workflow ID already exists
            if workflow_id in self.workflows:
                return {"error": f"Workflow ID '{workflow_id}' already exists"}
                
            try:
                # Create and configure the workflow
                workflow = self.workflow_factory.create_bugfix_workflow(bug_description, error_logs)
                
                # Store the workflow
                self.workflows[workflow_id] = workflow
                
                logger.info(f"Created bug fix workflow '{workflow_id}' for: {bug_description}")
                
                return {
                    "workflow_id": workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "bug_description": bug_description,
                    "status": workflow.status,
                    "created_at": workflow.created_at.isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to create bug fix workflow: {e}")
                return {"error": f"Failed to create bug fix workflow: {str(e)}"}
        
        async def create_review_workflow(files_to_review: List[str], review_context: Optional[str] = None, workflow_id: Optional[str] = None) -> Dict[str, Any]:
            """
            Create a workflow for reviewing code.
            
            Args:
                files_to_review: List of files to review
                review_context: Optional context about the changes
                workflow_id: Optional ID for the workflow (generated if not provided)
                
            Returns:
                Dict containing workflow metadata
            """
            # Generate workflow ID if not provided
            if not workflow_id:
                workflow_id = f"review_{len(self.workflows) + 1}"
                
            # Check if workflow ID already exists
            if workflow_id in self.workflows:
                return {"error": f"Workflow ID '{workflow_id}' already exists"}
                
            try:
                # Create and configure the workflow
                workflow = self.workflow_factory.create_review_workflow(files_to_review, review_context)
                
                # Store the workflow
                self.workflows[workflow_id] = workflow
                
                logger.info(f"Created code review workflow '{workflow_id}' for files: {', '.join(files_to_review)}")
                
                return {
                    "workflow_id": workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "files_to_review": files_to_review,
                    "status": workflow.status,
                    "created_at": workflow.created_at.isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to create code review workflow: {e}")
                return {"error": f"Failed to create code review workflow: {str(e)}"}
        
        async def execute_workflow(workflow_id: str) -> Dict[str, Any]:
            """
            Execute a workflow.
            
            Args:
                workflow_id: ID of the workflow to execute
                
            Returns:
                Dict containing execution results
            """
            if workflow_id not in self.workflows:
                return {"error": f"Workflow '{workflow_id}' not found"}
                
            try:
                # Get the workflow
                workflow = self.workflows[workflow_id]
                
                # Execute the workflow
                result = await workflow.execute()
                
                logger.info(f"Executed workflow '{workflow_id}' with status: {result['status']}")
                
                return result
            except Exception as e:
                logger.error(f"Failed to execute workflow '{workflow_id}': {e}")
                return {
                    "workflow_id": workflow_id,
                    "error": f"Failed to execute workflow: {str(e)}",
                    "status": "failed"
                }
        
        async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
            """
            Get the status of a workflow.
            
            Args:
                workflow_id: ID of the workflow
                
            Returns:
                Dict containing workflow status
            """
            if workflow_id not in self.workflows:
                return {"error": f"Workflow '{workflow_id}' not found"}
                
            try:
                # Get the workflow
                workflow = self.workflows[workflow_id]
                
                # Get the workflow status
                status = workflow.get_status()
                
                return {
                    "workflow_id": workflow_id,
                    **status
                }
            except Exception as e:
                logger.error(f"Failed to get status of workflow '{workflow_id}': {e}")
                return {
                    "workflow_id": workflow_id,
                    "error": f"Failed to get workflow status: {str(e)}"
                }
        
        async def list_workflows() -> Dict[str, Any]:
            """
            List all workflows.
            
            Returns:
                Dict containing workflow information
            """
            try:
                workflows_info = []
                
                for workflow_id, workflow in self.workflows.items():
                    workflows_info.append({
                        "workflow_id": workflow_id,
                        "name": workflow.name,
                        "description": workflow.description,
                        "status": workflow.status,
                        "created_at": workflow.created_at.isoformat(),
                        "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None
                    })
                
                return {
                    "count": len(workflows_info),
                    "workflows": workflows_info
                }
            except Exception as e:
                logger.error(f"Failed to list workflows: {e}")
                return {"error": f"Failed to list workflows: {str(e)}"}
        
        async def visualize_workflow(workflow_id: str) -> Dict[str, Any]:
            """
            Generate a visualization of a workflow.
            
            Args:
                workflow_id: ID of the workflow
                
            Returns:
                Dict containing visualization data
            """
            if workflow_id not in self.workflows:
                return {"error": f"Workflow '{workflow_id}' not found"}
                
            try:
                # Get the workflow
                workflow = self.workflows[workflow_id]
                
                # Generate visualization
                visualization = workflow.visualize()
                
                return {
                    "workflow_id": workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "visualization": visualization
                }
            except Exception as e:
                logger.error(f"Failed to visualize workflow '{workflow_id}': {e}")
                return {
                    "workflow_id": workflow_id,
                    "error": f"Failed to visualize workflow: {str(e)}"
                }
        
        async def create_knowledge_reasoning_workflow(reasoning_request: str, workflow_id: Optional[str] = None) -> Dict[str, Any]:
            """
            Create a knowledge reasoning workflow.
            
            This creates a workflow for structured reasoning using the knowledge graph,
            which includes steps for context retrieval, structured reasoning, reflection,
            and knowledge capture.
            
            Args:
                reasoning_request: Description of the reasoning request or topic
                workflow_id: Optional ID for the workflow (generated if not provided)
                
            Returns:
                Dict containing workflow details or error information
            """
            # Generate workflow ID if not provided
            if not workflow_id:
                workflow_id = f"knowledge_reasoning_{str(uuid.uuid4())[:8]}"
            
            # Check if workflow ID already exists
            if workflow_id in self.workflows:
                return {"error": f"Workflow ID '{workflow_id}' already exists"}
                
            try:
                # Create and configure the workflow
                workflow = self.workflow_factory.create_knowledge_reasoning_workflow(reasoning_request)
                
                # Store the workflow
                self.workflows[workflow_id] = workflow
                
                logger.info(f"Created knowledge reasoning workflow '{workflow_id}' for: {reasoning_request}")
                
                return {
                    "workflow_id": workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "reasoning_request": reasoning_request,
                    "status": workflow.status,
                    "created_at": workflow.created_at.isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to create knowledge reasoning workflow: {e}")
                return {"error": f"Failed to create knowledge reasoning workflow: {str(e)}"}
        
        # Register functions with the MCP server
        functions = [
            {
                "name": "create_feature_workflow",
                "description": "Create a workflow for implementing a feature",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "feature_description": {
                            "type": "string",
                            "description": "Description of the feature to implement"
                        },
                        "workflow_id": {
                            "type": "string",
                            "description": "Optional ID for the workflow"
                        }
                    },
                    "required": ["feature_description"]
                },
                "function": create_feature_workflow
            },
            {
                "name": "create_knowledge_reasoning_workflow",
                "description": "Create a workflow for knowledge-based reasoning",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning_request": {
                            "type": "string",
                            "description": "Description of the reasoning request or topic"
                        },
                        "workflow_id": {
                            "type": "string",
                            "description": "Optional ID for the workflow"
                        }
                    },
                    "required": ["reasoning_request"]
                },
                "function": create_knowledge_reasoning_workflow
            },
            {
                "name": "create_bugfix_workflow",
                "description": "Create a workflow for fixing a bug",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bug_description": {
                            "type": "string",
                            "description": "Description of the bug to fix"
                        },
                        "error_logs": {
                            "type": "string",
                            "description": "Optional error logs related to the bug"
                        },
                        "workflow_id": {
                            "type": "string",
                            "description": "Optional ID for the workflow"
                        }
                    },
                    "required": ["bug_description"]
                },
                "function": create_bugfix_workflow
            },
            {
                "name": "create_review_workflow",
                "description": "Create a workflow for reviewing code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "files_to_review": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of files to review"
                        },
                        "review_context": {
                            "type": "string",
                            "description": "Optional context about the changes"
                        },
                        "workflow_id": {
                            "type": "string",
                            "description": "Optional ID for the workflow"
                        }
                    },
                    "required": ["files_to_review"]
                },
                "function": create_review_workflow
            },
            {
                "name": "execute_workflow",
                "description": "Execute a workflow",
                "parameters": {
                    "type": "string",
                    "description": "ID of the workflow to execute"
                },
                "function": execute_workflow
            },
            {
                "name": "get_workflow_status",
                "description": "Get the status of a workflow",
                "parameters": {
                    "type": "string",
                    "description": "ID of the workflow"
                },
                "function": get_workflow_status
            },
            {
                "name": "list_workflows",
                "description": "List all workflows",
                "parameters": {},
                "function": list_workflows
            },
            {
                "name": "visualize_workflow",
                "description": "Generate a visualization of a workflow",
                "parameters": {
                    "type": "string",
                    "description": "ID of the workflow"
                },
                "function": visualize_workflow
            }
        ]
        
        # Register workflow management tools
        self.mcp_server.register_tools(
            tool_name="workflow-manager",
            tool_description="Tools for creating and managing workflow templates for common development tasks",
            functions=functions
        )
        
        logger.info("Registered workflow management tools")
    
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

    def register_tools(self):
        """Register all tools with the MCP server."""
        self._register_workflow_tools()
        self._register_error_handling_tools()

    def _register_error_handling_tools(self):
        """
        Register error handling tools with the MCP server.
        
        These tools provide error handling, recovery, and timeout
        management capabilities for workflows.
        """
        if not self.mcp_server or not self.error_handler or not self.timeout_manager:
            logger.warning("Cannot register error handling tools - required components not initialized")
            return

        # Register the handle_error tool
        self.mcp_server.register_tool(
            "handle_workflow_error",
            {
                "description": "Handle a workflow error with enhanced recovery mechanisms",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "error_message": {"type": "string", "description": "Error message"},
                        "task_id": {"type": "string", "description": "ID of the task where the error occurred"},
                        "workflow_id": {"type": "string", "description": "ID of the workflow"},
                        "context": {"type": "object", "description": "Additional context about the error"},
                        "retry_count": {"type": "integer", "description": "Number of retries already attempted"},
                        "max_retries": {"type": "integer", "description": "Maximum number of retries allowed"}
                    },
                    "required": ["error_message", "task_id", "workflow_id"]
                }
            },
            self._handle_workflow_error
        )

        # Register the get_error_summary tool
        self.mcp_server.register_tool(
            "get_workflow_error_summary",
            {
                "description": "Get a summary of all errors that occurred during workflow execution",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "ID of the workflow (optional)"}
                    }
                }
            },
            self._get_workflow_error_summary
        )

        # Register timeout management tools
        self.mcp_server.register_tool(
            "update_task_timeout",
            {
                "description": "Update the timeout for a task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "ID of the task"},
                        "timeout": {"type": "number", "description": "New timeout in seconds"}
                    },
                    "required": ["task_id", "timeout"]
                }
            },
            self._update_task_timeout
        )

        # Register the adapt_timeout tool
        self.mcp_server.register_tool(
            "adapt_task_timeout",
            {
                "description": "Adapt the timeout for a task based on execution history",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "ID of the task"}
                    },
                    "required": ["task_id"]
                }
            },
            self._adapt_task_timeout
        )

        logger.info("Error handling tools registered successfully")

    def _handle_workflow_error(self, error_message, task_id, workflow_id, context=None, retry_count=0, max_retries=3):
        """
        Handle a workflow error.
        
        Args:
            error_message: Error message
            task_id: ID of the task where the error occurred
            workflow_id: ID of the workflow
            context: Additional context about the error
            retry_count: Number of retries already attempted
            max_retries: Maximum number of retries allowed
            
        Returns:
            Dict containing recovery information
        """
        if not self.error_handler:
            return {"error": "Error handler not initialized"}
        
        # Create an exception from the error message
        error = Exception(error_message)
        
        # Add workflow_id to context
        context = context or {}
        context["workflow_id"] = workflow_id
        
        # Handle the error
        can_continue, recovery_info = self.error_handler.handle_error(
            error=error,
            task_id=task_id,
            context=context,
            retry_count=retry_count,
            max_retries=max_retries
        )
        
        # Add whether the workflow can continue
        recovery_info["can_continue"] = can_continue
        
        # If we need to increase timeout, do it
        if recovery_info.get("action") == "retry" and recovery_info.get("increase_timeout"):
            multiplier = recovery_info.get("timeout_multiplier", 1.5)
            new_timeout = self.timeout_manager.increase_timeout(task_id, multiplier)
            recovery_info["new_timeout"] = new_timeout
        
        return recovery_info

    def _get_workflow_error_summary(self, workflow_id=None):
        """
        Get a summary of all errors that occurred during workflow execution.
        
        Args:
            workflow_id: ID of the workflow (optional)
            
        Returns:
            Dict containing error summary information
        """
        if not self.error_handler:
            return {"error": "Error handler not initialized"}
        
        # Get error summary
        summary = self.error_handler.get_error_summary()
        
        # Filter by workflow_id if provided
        if workflow_id:
            filtered_task_errors = {}
            for task_id, errors in summary.get("task_errors", {}).items():
                filtered_errors = [
                    error for error in errors 
                    if error.get("context", {}).get("workflow_id") == workflow_id
                ]
                if filtered_errors:
                    filtered_task_errors[task_id] = filtered_errors
            
            summary["task_errors"] = filtered_task_errors
            summary["total_filtered_errors"] = sum(len(errors) for errors in filtered_task_errors.values())
        
        return summary

    def _update_task_timeout(self, task_id, timeout):
        """
        Update the timeout for a task.
        
        Args:
            task_id: ID of the task
            timeout: New timeout in seconds
            
        Returns:
            Dict containing the new timeout
        """
        if not self.timeout_manager:
            return {"error": "Timeout manager not initialized"}
        
        self.timeout_manager.update_timeout(task_id, timeout)
        
        return {
            "task_id": task_id,
            "timeout": timeout
        }

    def _adapt_task_timeout(self, task_id):
        """
        Adapt the timeout for a task based on execution history.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dict containing the adapted timeout
        """
        if not self.timeout_manager:
            return {"error": "Timeout manager not initialized"}
        
        timeout = self.timeout_manager.adapt_timeout(task_id)
        
        return {
            "task_id": task_id,
            "timeout": timeout,
            "stats": self.timeout_manager.get_timeout_stats().get(task_id, {})
        } 