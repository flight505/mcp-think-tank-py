#!/usr/bin/env python3
"""
Orchestrator for MCP Think Tank
Manages cross-tool intelligence and coordination
"""
import logging
import asyncio
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid
import time
import traceback

from fastapi import FastAPI
from fastmcp.server import FastMCP

# Local imports
from .tools.memory import MemoryTool
from .tools.think import ThinkTool
from .tools.tasks import TasksTool
from .watchers.file_watcher import FileWatcher
from .tools.code_tools import CodeTools
from .tools.dag_orchestrator import DAGExecutor, EmbeddingCache
from .tools.workflow_templates import WorkflowFactory
from .tools.workflow_error_handler import WorkflowErrorHandler, TimeoutManager
from .tools.circuit_breaker import CircuitBreakerManager, CircuitOpenError
from src.tools.monitoring import PerformanceTracker, track_memory_usage

logger = logging.getLogger("mcp-think-tank.orchestrator")

class Orchestrator:
    """
    Orchestrates the various components of the system.
    
    This class is responsible for initializing and managing all the tools
    and services used by the system, including the FastMCP server and
    various tool instances.
    """
    
    def __init__(self, mcp: FastMCP):
        """
        Initialize the orchestrator.
        
        Args:
            mcp: The FastMCP instance
        """
        self.mcp = mcp
        
        # Initialize tools with monitoring
        self._init_components_with_monitoring()
        
        # Initialize workflow factory and active workflows
        self.workflow_factory = self._init_workflow_factory()
        self.workflows = {}
        self.error_handler = None
        self.timeout_manager = None
        self.circuit_manager = None
        
        # Register all tools with the MCP server
        self._register_memory_tools()
        self._register_think_tools()
        self._register_tasks_tools()
        self._register_code_tools()
        self._register_dag_tools()
        self._register_workflow_tools()
        self._init_error_handling()

    def _init_components_with_monitoring(self):
        """Initialize components with performance tracking"""
        try:
            with PerformanceTracker("init_components"):
                # Track memory before initialization
                track_memory_usage("before_init")
                
                # Initialize components
                self._init_memory_tool()
                self._init_think_tool()
                self._init_tasks_tool()
                self._init_file_watcher()
                self._init_code_tools()
                
                # Track memory after initialization
                track_memory_usage("after_init")
                
                self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def _init_memory_tool(self) -> MemoryTool:
        """Initialize the memory tool."""
        try:
            with PerformanceTracker("init_memory_tool"):
                return MemoryTool()
        except Exception as e:
            logger.error(f"Failed to initialize memory tool: {e}")
            # Return a basic version or raise an exception
            return MemoryTool(use_basic=True)
    
    def _init_think_tool(self) -> ThinkTool:
        """Initialize the think tool."""
        try:
            with PerformanceTracker("init_think_tool"):
                return ThinkTool()
        except Exception as e:
            logger.error(f"Failed to initialize think tool: {e}")
            # Return a basic version or raise an exception
            return ThinkTool(use_basic=True)
    
    def _init_tasks_tool(self) -> TasksTool:
        """Initialize the tasks tool."""
        try:
            with PerformanceTracker("init_tasks_tool"):
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
            with PerformanceTracker("init_code_tools"):
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
            with PerformanceTracker("init_dag_executor"):
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
            with PerformanceTracker("init_embedding_cache"):
                return EmbeddingCache(max_size=1000)
        except Exception as e:
            logger.error(f"Failed to initialize embedding cache: {e}")
            # Return a basic version
            return EmbeddingCache(max_size=100)
    
    def _init_file_watcher(self) -> FileWatcher:
        """Initialize the file watcher."""
        try:
            with PerformanceTracker("init_file_watcher"):
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
            with PerformanceTracker("init_workflow_factory"):
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
        
        This method sets up the WorkflowErrorHandler, TimeoutManager
        and CircuitBreakerManager that are used for robust workflow
        execution with timeout management and graceful error recovery.
        """
        try:
            with PerformanceTracker("init_error_handling"):
                self.error_handler = WorkflowErrorHandler()
                self.timeout_manager = TimeoutManager(default_timeout=60.0)  # 60 second default timeout
                self.circuit_manager = CircuitBreakerManager()
                logger.info("Error handling components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize error handling: {str(e)}")
            # Create basic versions if initialization fails
            self.error_handler = WorkflowErrorHandler()
            self.timeout_manager = TimeoutManager()
            self.circuit_manager = CircuitBreakerManager()
    
    async def _execute_with_timeout_and_recovery(
        self, 
        func: Callable, 
        task_id: str, 
        workflow_id: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        fallback_func: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a function with timeout and error recovery.
        
        This is a wrapper method that:
        1. Applies proper timeouts
        2. Handles errors gracefully with retries
        3. Records execution metrics
        4. Provides fallback mechanisms
        5. Ensures the timeout resets after each call
        6. Uses circuit breaker to prevent cascading failures
        
        Args:
            func: The function to execute
            task_id: ID of the task/tool
            workflow_id: Optional ID of the workflow
            timeout: Optional timeout in seconds (defaults to task's configured timeout)
            retry_count: Current retry attempt
            max_retries: Maximum number of retry attempts
            fallback_func: Optional fallback function to use if the primary function fails
            **kwargs: Arguments to pass to the function
            
        Returns:
            Dict containing execution results or error information
        """
        # Get the effective timeout (from param, task config, or default)
        effective_timeout = timeout
        if not effective_timeout and self.timeout_manager:
            effective_timeout = self.timeout_manager.get_timeout(task_id)
        
        # Get the circuit breaker for this task or service
        service_name = task_id.split('.')[0] if '.' in task_id else task_id
        circuit_breaker = self.circuit_manager.get_circuit_breaker(
            service_name,
            failure_threshold=5,
            recovery_timeout=60.0,
            half_open_max_calls=1
        )
        
        start_time = time.time()
        
        try:
            # Check if the circuit is open for this service
            try:
                # Execute the function with timeout and circuit breaker
                if asyncio.iscoroutinefunction(func):
                    if effective_timeout:
                        # For async functions with circuit breaker
                        async def _execute_with_circuit_breaker():
                            return await circuit_breaker.execute_async(func, **kwargs)
                        
                        # Reset the timeout for each new call
                        result = await asyncio.wait_for(_execute_with_circuit_breaker(), timeout=effective_timeout)
                    else:
                        # No timeout, but still use circuit breaker
                        result = await circuit_breaker.execute_async(func, **kwargs)
                else:
                    if effective_timeout:
                        # For sync functions with circuit breaker
                        async def _sync_wrapper():
                            return circuit_breaker.execute(func, **kwargs)
                        
                        result = await asyncio.wait_for(_sync_wrapper(), timeout=effective_timeout)
                    else:
                        # No timeout, but still use circuit breaker
                        result = circuit_breaker.execute(func, **kwargs)
                
                # Record successful execution time
                execution_time = time.time() - start_time
                if self.timeout_manager:
                    self.timeout_manager.record_execution_time(task_id, execution_time)
                    
                return {"status": "success", "result": result, "execution_time": execution_time}
            
            except CircuitOpenError as e:
                logger.warning(f"Circuit for task {task_id} is OPEN. Fast-failing request.")
                return {
                    "status": "error",
                    "error_type": "circuit_open",
                    "message": str(e),
                    "can_continue": False
                }
            
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_id} timed out after {effective_timeout}s")
            
            # Handle timeout with error handler
            if self.error_handler:
                error = asyncio.TimeoutError(f"Task {task_id} timed out after {effective_timeout}s")
                context = {"workflow_id": workflow_id} if workflow_id else {}
                
                can_continue, recovery_info = self.error_handler.handle_error(
                    error=error,
                    task_id=task_id,
                    context=context,
                    retry_count=retry_count,
                    max_retries=max_retries
                )
                
                # If we can retry, do so with increased timeout
                if can_continue and recovery_info.get("action") == "retry":
                    # Increase timeout if specified
                    if recovery_info.get("increase_timeout") and self.timeout_manager:
                        multiplier = recovery_info.get("timeout_multiplier", 1.5)
                        new_timeout = self.timeout_manager.increase_timeout(task_id, multiplier)
                        logger.info(f"Increasing timeout for task {task_id} to {new_timeout}s for retry {retry_count + 1}")
                        timeout = new_timeout
                    
                    # Apply delay if specified
                    delay = recovery_info.get("delay")
                    if delay:
                        await asyncio.sleep(delay)
                    
                    # Retry
                    return await self._execute_with_timeout_and_recovery(
                        func=func,
                        task_id=task_id,
                        workflow_id=workflow_id,
                        timeout=timeout,
                        retry_count=retry_count + 1,
                        max_retries=max_retries,
                        fallback_func=fallback_func,
                        **kwargs
                    )
                
                # If we should use a fallback, do so
                elif can_continue and fallback_func and recovery_info.get("action") in ["fallback", "degrade"]:
                    logger.info(f"Using fallback for task {task_id} after timeout")
                    return await self._execute_with_timeout_and_recovery(
                        func=fallback_func,
                        task_id=f"{task_id}_fallback",
                        workflow_id=workflow_id,
                        timeout=timeout,  # Use the same timeout
                        retry_count=0,  # Reset retry count for fallback
                        max_retries=max_retries,
                        fallback_func=None,  # No fallback for the fallback
                        **kwargs
                    )
                
                # Otherwise return error information
                return {
                    "status": "error",
                    "error_type": "timeout",
                    "message": f"Task {task_id} timed out after {effective_timeout}s",
                    "can_continue": can_continue,
                    "recovery_info": recovery_info
                }
            
            # Basic error handling if no error handler
            if retry_count < max_retries:
                new_timeout = effective_timeout * 1.5 if effective_timeout else 60.0
                logger.info(f"Retrying task {task_id} with timeout {new_timeout}s (attempt {retry_count + 1})")
                
                return await self._execute_with_timeout_and_recovery(
                    func=func,
                    task_id=task_id,
                    workflow_id=workflow_id,
                    timeout=new_timeout,
                    retry_count=retry_count + 1,
                    max_retries=max_retries,
                    fallback_func=fallback_func,
                    **kwargs
                )
            
            if fallback_func:
                logger.info(f"Using fallback for task {task_id} after timeout")
                return await self._execute_with_timeout_and_recovery(
                    func=fallback_func,
                    task_id=f"{task_id}_fallback",
                    workflow_id=workflow_id,
                    timeout=effective_timeout,  # Use the same timeout
                    retry_count=0,  # Reset retry count for fallback
                    max_retries=max_retries,
                    fallback_func=None,  # No fallback for the fallback
                    **kwargs
                )
            
            return {
                "status": "error",
                "error_type": "timeout",
                "message": f"Task {task_id} timed out after {effective_timeout}s and exceeded retry limit"
            }
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {str(e)}")
            
            # Handle error with error handler
            if self.error_handler:
                context = {"workflow_id": workflow_id} if workflow_id else {}
                
                can_continue, recovery_info = self.error_handler.handle_error(
                    error=e,
                    task_id=task_id,
                    context=context,
                    retry_count=retry_count,
                    max_retries=max_retries
                )
                
                # If we can retry, do so
                if can_continue and recovery_info.get("action") == "retry":
                    # Apply delay if specified
                    delay = recovery_info.get("delay")
                    if delay:
                        await asyncio.sleep(delay)
                    
                    # Retry
                    return await self._execute_with_timeout_and_recovery(
                        func=func,
                        task_id=task_id,
                        workflow_id=workflow_id,
                        timeout=timeout,
                        retry_count=retry_count + 1,
                        max_retries=max_retries,
                        fallback_func=fallback_func,
                        **kwargs
                    )
                
                # If we should use a fallback, do so
                elif can_continue and fallback_func and recovery_info.get("action") in ["fallback", "degrade", "use_cache"]:
                    logger.info(f"Using fallback for task {task_id} after error: {str(e)}")
                    return await self._execute_with_timeout_and_recovery(
                        func=fallback_func,
                        task_id=f"{task_id}_fallback",
                        workflow_id=workflow_id,
                        timeout=timeout,  # Use the same timeout
                        retry_count=0,  # Reset retry count for fallback
                        max_retries=max_retries,
                        fallback_func=None,  # No fallback for the fallback
                        **kwargs
                    )
                
                # Otherwise return error information
                return {
                    "status": "error",
                    "error_type": recovery_info.get("action", "unknown"),
                    "message": str(e),
                    "can_continue": can_continue,
                    "recovery_info": recovery_info
                }
            
            # Basic error handling if no error handler
            if retry_count < max_retries:
                logger.info(f"Retrying task {task_id} after error: {str(e)} (attempt {retry_count + 1})")
                
                # Exponential backoff
                delay = 0.5 * (2 ** retry_count)
                await asyncio.sleep(delay)
                
                return await self._execute_with_timeout_and_recovery(
                    func=func,
                    task_id=task_id,
                    workflow_id=workflow_id,
                    timeout=timeout,
                    retry_count=retry_count + 1,
                    max_retries=max_retries,
                    fallback_func=fallback_func,
                    **kwargs
                )
            
            if fallback_func:
                logger.info(f"Using fallback for task {task_id} after error: {str(e)}")
                return await self._execute_with_timeout_and_recovery(
                    func=fallback_func,
                    task_id=f"{task_id}_fallback",
                    workflow_id=workflow_id,
                    timeout=timeout,  # Use the same timeout
                    retry_count=0,  # Reset retry count for fallback
                    max_retries=max_retries,
                    fallback_func=None,  # No fallback for the fallback
                    **kwargs
                )
            
            return {
                "status": "error",
                "error_type": "exception",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _register_memory_tools(self) -> None:
        """Register memory-related tools with the MCP server."""
        self.mcp.register_tools(
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
        self.mcp.register_tools(
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
        self.mcp.register_tools(
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
            
        self.mcp.register_tools(
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
        self.mcp.register_tools(
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
            # Generate a unique task ID for this operation
            task_id = f"create_workflow_{uuid.uuid4().hex[:8]}"
            
            # Define the actual function
            def _create_workflow():
                workflow_id = f"workflow_{len(self.workflows) + 1}"
                self.workflows[workflow_id] = {
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
            
            # Execute with timeout and error recovery
            result = await self._execute_with_timeout_and_recovery(
                func=_create_workflow,
                task_id=task_id
            )
            
            if result["status"] == "success":
                return result["result"]
            else:
                return {"error": result["message"]}
        
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
            if workflow_id not in self.workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = self.workflows[workflow_id]
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
            if workflow_id not in self.workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            # Generate a unique task ID for this operation
            task_id = f"execute_workflow_{workflow_id}"
            
            # Define the actual execution function
            async def _execute_workflow():
                workflow = self.workflows[workflow_id]
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
            
            # Execute with timeout and error recovery
            # We use a longer timeout for workflow execution
            result = await self._execute_with_timeout_and_recovery(
                func=_execute_workflow,
                task_id=task_id,
                workflow_id=workflow_id,
                timeout=3600  # 1 hour timeout for workflow execution
            )
            
            if result["status"] == "success":
                return result["result"]
            else:
                return {
                    "workflow_id": workflow_id,
                    "status": "failed",
                    "error": result["message"]
                }
        
        async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
            """
            Get the status of a workflow.
            
            Args:
                workflow_id: ID of the workflow
                
            Returns:
                Dict containing workflow status and metrics
            """
            if workflow_id not in self.workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = self.workflows[workflow_id]
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
            if workflow_id not in self.workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = self.workflows[workflow_id]
            dag_executor = workflow["dag_executor"]
            
            # Get DAG visualization
            visualization = dag_executor.visualize()
            
            return {
                "workflow_id": workflow_id,
                "name": workflow["name"],
                "visualization": visualization
            }
        
        # Register DAG orchestration tools
        self.mcp.register_tools(
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
        self._register_error_tools()

    def _register_error_tools(self):
        """Register error handling and circuit breaker tools with the MCP server."""
        # Register error summary tool
        self.mcp.register_tool(
            "get_workflow_error_summary",
            {
                "description": "Get a summary of all errors that occurred during workflow execution",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "ID of the workflow (optional)"}
                    },
                    "required": []
                }
            },
            self._get_workflow_error_summary
        )

        # Register timeout management tools
        self.mcp.register_tool(
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
        self.mcp.register_tool(
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
        
        # Register circuit breaker tools
        self.mcp.register_tool(
            "get_circuit_breaker_status",
            {
                "description": "Get the status of a circuit breaker",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_name": {"type": "string", "description": "Name of the service/circuit"}
                    },
                    "required": ["service_name"]
                }
            },
            self._get_circuit_breaker_status
        )
        
        self.mcp.register_tool(
            "reset_circuit_breaker",
            {
                "description": "Reset a circuit breaker to its initial state",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_name": {"type": "string", "description": "Name of the service/circuit"}
                    },
                    "required": ["service_name"]
                }
            },
            self._reset_circuit_breaker
        )
        
        self.mcp.register_tool(
            "get_all_circuit_breakers",
            {
                "description": "Get status of all circuit breakers in the system",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            self._get_all_circuit_breakers
        )

        logger.info("Error handling tools registered successfully")

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

    def _get_circuit_breaker_status(self, service_name):
        """
        Get the status of a circuit breaker.
        
        Args:
            service_name: Name of the service/circuit
            
        Returns:
            Dict containing status information
        """
        if not self.circuit_manager:
            return {"error": "Circuit breaker manager not initialized"}
        
        circuit = self.circuit_manager.get_circuit_breaker(service_name, create_if_missing=False)
        if not circuit:
            return {"error": f"No circuit breaker found for service {service_name}"}
        
        return circuit.get_health()

    def _reset_circuit_breaker(self, service_name):
        """
        Reset a circuit breaker to its initial state.
        
        Args:
            service_name: Name of the service/circuit
            
        Returns:
            Dict containing status information
        """
        if not self.circuit_manager:
            return {"error": "Circuit breaker manager not initialized"}
        
        circuit = self.circuit_manager.get_circuit_breaker(service_name, create_if_missing=False)
        if not circuit:
            return {"error": f"No circuit breaker found for service {service_name}"}
        
        circuit.reset()
        return {"status": "success", "message": f"Circuit breaker for {service_name} has been reset"}

    def _get_all_circuit_breakers(self):
        """
        Get status of all circuit breakers in the system.
        
        Returns:
            Dict containing all circuit breaker statuses
        """
        if not self.circuit_manager:
            return {"error": "Circuit breaker manager not initialized"}
        
        return self.circuit_manager.get_all_health()

    async def execute_tool_with_recovery(self, tool_name: str, function_name: str, 
                                        parameters: Dict[str, Any], context: Dict[str, Any] = None,
                                        timeout: Optional[float] = None, 
                                        max_retries: int = 3) -> Dict[str, Any]:
        """
        Execute a tool function with timeout handling and error recovery.
        
        This method provides a robust way to execute tool functions with:
        - Proper timeout handling
        - Automatic retries with exponential backoff
        - Error recovery and fallback mechanisms
        - Performance metrics recording
        
        Args:
            tool_name: Name of the tool to use
            function_name: Name of the function to call
            parameters: Parameters to pass to the function
            context: Additional context (e.g., workflow_id, task information)
            timeout: Optional timeout in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict containing execution results or error information
        """
        # Generate a unique task ID for tracking
        task_id = f"{tool_name}_{function_name}_{uuid.uuid4().hex[:8]}"
        
        # Get the function to execute
        func = self._get_tool_function(tool_name, function_name)
        if not func:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": f"Function {function_name} not found in tool {tool_name}"
            }
            
        # Get a fallback function if available
        fallback_func = None
        if hasattr(self, f"_fallback_{tool_name}_{function_name}"):
            fallback_func = getattr(self, f"_fallback_{tool_name}_{function_name}")
            
        # Log the tool execution
        workflow_id = context.get("workflow_id") if context else None
        logger.info(f"Executing tool {tool_name}.{function_name} [task_id={task_id}, workflow={workflow_id}]")
        
        # Execute with timeout and error recovery
        result = await self._execute_with_timeout_and_recovery(
            func=func,
            task_id=task_id,
            workflow_id=workflow_id,
            timeout=timeout,
            max_retries=max_retries,
            fallback_func=fallback_func,
            **parameters
        )
        
        # Add context information to the result
        if context:
            if "context" not in result:
                result["context"] = {}
            result["context"].update(context)
        
        # Log the result
        status = result.get("status", "unknown")
        if status == "success":
            logger.info(f"Tool {tool_name}.{function_name} executed successfully in {result.get('execution_time', '?')}s")
        else:
            logger.warning(f"Tool {tool_name}.{function_name} execution failed: {result.get('message', 'Unknown error')}")
        
        return result 