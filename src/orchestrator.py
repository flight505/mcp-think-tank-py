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

from fastmcp.server import FastMCP

# Local imports
from .tools.memory import MemoryTool
from .tools.think import ThinkTool
from .tools.tasks import TasksTool
from .watchers.file_watcher import FileWatcher
from .tools.code_tools import CodeTools
from .tools.dag_orchestrator import DAGExecutor, EmbeddingCache, DAGTask
from .tools.workflow_templates import WorkflowFactory
from .tools.workflow_error_handler import WorkflowErrorHandler, TimeoutManager
from .tools.circuit_breaker import CircuitBreakerManager, CircuitOpenError
from src.tools.monitoring import PerformanceTracker, track_memory_usage

logger = logging.getLogger("mcp-think-tank.orchestrator")

# Define the DAG class needed for workflow management
class DAG:
    """
    Simple Directed Acyclic Graph implementation for workflow management.
    
    This class represents a workflow as a directed acyclic graph where nodes
    are tasks and edges represent dependencies between tasks.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a DAG.
        
        Args:
            name: Name of the DAG
            description: Optional description
        """
        self.name = name
        self.description = description
        self.tasks = {}  # Dict mapping task_id to task
        self.dependencies = {}  # Dict mapping task_id to set of dependency task_ids
        
    def add_task(self, task: DAGTask) -> None:
        """
        Add a task to the DAG.
        
        Args:
            task: The task to add
        """
        self.tasks[task.task_id] = task
        
        # Initialize dependencies
        if task.task_id not in self.dependencies:
            self.dependencies[task.task_id] = set()
        
        # Add dependencies
        for dep_id in task.dependencies:
            self.dependencies[task.task_id].add(dep_id)
            
    def get_ready_tasks(self) -> List[str]:
        """
        Get tasks that are ready to execute (have no pending dependencies).
        
        Returns:
            List of task IDs that are ready to execute
        """
        ready_tasks = []
        
        for task_id, deps in self.dependencies.items():
            # Check if all dependencies are completed
            all_deps_completed = True
            for dep_id in deps:
                if dep_id in self.tasks and self.tasks[dep_id].status != "completed":
                    all_deps_completed = False
                    break
            
            # If all dependencies are completed and task is pending, it's ready
            if all_deps_completed and self.tasks[task_id].status == "pending":
                ready_tasks.append(task_id)
                
        return ready_tasks

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
        
        # Setup logger
        self.logger = logging.getLogger("mcp-think-tank.orchestrator")
        
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
                self.memory_tool = self._init_memory_tool()
                self.think_tool = self._init_think_tool()
                self.tasks_tool = self._init_tasks_tool()
                self.file_watcher = self._init_file_watcher()
                self.code_tools = self._init_code_tools()
                
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
            except Exception:  # Be more specific with the exception type
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
                # Get configuration
                from src.config import get_config
                config = get_config()
                return WorkflowFactory(config=config)
        except Exception as e:
            logger.error(f"Failed to initialize workflow factory: {e}")
            # Try with minimal requirements
            return WorkflowFactory()
    
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
        """Register memory tools with the MCP server."""
        # FastMCP 2.1.0 no longer supports register_tools method
        # Instead, we need to use decorators to register tools
        
        @self.mcp.tool(name="create_entities")
        async def create_entities(entities):
            """Create multiple new entities in the knowledge graph"""
            # Ensure entities parameter is provided and is a list
            if not entities or not isinstance(entities, list):
                raise ValueError("entities must be a non-empty list of entity objects")
                
            return await self._execute_with_timeout_and_recovery(
                func=self.memory_tool.create_entities,
                task_id="create_entities",
                entities=entities
            )
            
        @self.mcp.tool(name="create_relations")
        async def create_relations(relations):
            return await self.memory_tool.create_relations(relations=relations)
        
        @self.mcp.tool(name="add_observations")
        async def add_observations(observations):
            return await self.memory_tool.add_observations(observations)
        
        @self.mcp.tool(name="delete_entities")
        async def delete_entities(entityNames):
            return await self.memory_tool.delete_entities(entityNames)
        
        @self.mcp.tool(name="delete_observations")
        async def delete_observations(deletions):
            return await self.memory_tool.delete_observations(deletions)
        
        @self.mcp.tool(name="delete_relations")
        async def delete_relations(relations):
            return await self.memory_tool.delete_relations(relations)
        
        @self.mcp.tool(name="read_graph")
        async def read_graph(dummy=None):
            return await self.memory_tool.read_graph()
        
        @self.mcp.tool(name="search_nodes")
        async def search_nodes(query):
            return await self.memory_tool.search_nodes(query)
        
        @self.mcp.tool(name="open_nodes")
        async def open_nodes(names):
            return await self.memory_tool.open_nodes(names)
        
        @self.mcp.tool(name="update_entities")
        async def update_entities(entities):
            return await self.memory_tool.update_entities(entities)
        
        @self.mcp.tool(name="update_relations")
        async def update_relations(relations):
            return await self.memory_tool.update_relations(relations)
    
    def _register_think_tools(self) -> None:
        """Register think tools with the MCP server."""
        # FastMCP 2.1.0 no longer supports register_tools method
        # Instead, we need to use decorators to register tools
        
        @self.mcp.tool(name="think")
        async def think(structuredReasoning, category=None, tags=None, associateWithEntity=None, storeInMemory=False):
            """
            Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. 
            Use it when complex reasoning or some cache memory is needed. For best results, structure your reasoning with: 
            1) Problem definition, 2) Relevant facts/context, 3) Analysis steps, 4) Conclusion/decision.
            """
            # Ensure structuredReasoning is a string and not empty
            if not structuredReasoning or not isinstance(structuredReasoning, str):
                raise ValueError("structuredReasoning must be a non-empty string")
                
            return await self._execute_with_timeout_and_recovery(
                func=self.think_tool.process,
                task_id="think",
                structured_reasoning=structuredReasoning,
                store_in_memory=storeInMemory,
                reflexion=False,
                category=category,
                tags=tags,
                associate_with_entity=associateWithEntity
            )
    
    def _register_tasks_tools(self) -> None:
        """Register task management tools with the MCP server."""
        # FastMCP 2.1.0 no longer supports register_tools method
        # Instead, we need to use decorators to register tools
        
        @self.mcp.tool(name="request_planning")
        async def request_planning(originalRequest, tasks, splitDetails=None):
            return await self.tasks_tool.request_planning(originalRequest=originalRequest, tasks=tasks, splitDetails=splitDetails)
        
        @self.mcp.tool(name="get_next_task")
        async def get_next_task(requestId):
            return await self.tasks_tool.get_next_task(requestId=requestId)
        
        @self.mcp.tool(name="mark_task_done")
        async def mark_task_done(requestId, taskId, completedDetails=None):
            return await self.tasks_tool.mark_task_done(requestId=requestId, taskId=taskId, completedDetails=completedDetails)
        
        @self.mcp.tool(name="approve_task_completion")
        async def approve_task_completion(requestId, taskId):
            return await self.tasks_tool.approve_task_completion(requestId=requestId, taskId=taskId)
        
        @self.mcp.tool(name="approve_request_completion")
        async def approve_request_completion(requestId):
            return await self.tasks_tool.approve_request_completion(requestId=requestId)
        
        @self.mcp.tool(name="open_task_details")
        async def open_task_details(taskId):
            return await self.tasks_tool.open_task_details(taskId=taskId)
        
        @self.mcp.tool(name="list_requests")
        async def list_requests(random_string=""):
            return await self.tasks_tool.list_requests(random_string=random_string)
        
        @self.mcp.tool(name="add_tasks_to_request")
        async def add_tasks_to_request(requestId, tasks):
            return await self.tasks_tool.add_tasks_to_request(requestId=requestId, tasks=tasks)
        
        @self.mcp.tool(name="update_task")
        async def update_task(requestId, taskId, title=None, description=None):
            return await self.tasks_tool.update_task(requestId=requestId, taskId=taskId, title=title, description=description)
        
        @self.mcp.tool(name="delete_task")
        async def delete_task(requestId, taskId):
            return await self.tasks_tool.delete_task(requestId=requestId, taskId=taskId)
    
    def _register_code_tools(self) -> None:
        """Register code analysis and manipulation tools with the MCP server."""
        if not self.code_tools:
            logger.warning("Code tools not initialized, skipping tool registration")
            return
            
        @self.mcp.tool(name="search_code")
        async def search_code(query, **kwargs):
            return await self.code_tools.search_code(query=query, **kwargs)
        
        @self.mcp.tool(name="summarize_file")
        async def summarize_file(filepath, **kwargs):
            return await self.code_tools.summarize_file(filepath=filepath, **kwargs)
    
    def _register_workflow_tools(self) -> None:
        """Register workflow-related tools with the MCP server."""
        
        # Register tools using decorator approach
        @self.mcp.tool(name="create_feature_workflow")
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
                    "name": workflow.get("name"),
                    "description": workflow.get("description"),
                    "feature_description": feature_description,
                    "status": workflow.get("status"),
                    "created_at": workflow.get("created_at").isoformat() if workflow.get("created_at") else None
                }
            except Exception as e:
                logger.error(f"Failed to create feature workflow: {e}")
                return {"error": f"Failed to create feature workflow: {str(e)}"}
                
        @self.mcp.tool(name="create_bugfix_workflow")
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
                    "name": workflow.get("name"),
                    "description": workflow.get("description"),
                    "bug_description": bug_description,
                    "status": workflow.get("status"),
                    "created_at": workflow.get("created_at").isoformat() if workflow.get("created_at") else None
                }
            except Exception as e:
                logger.error(f"Failed to create bug fix workflow: {e}")
                return {"error": f"Failed to create bug fix workflow: {str(e)}"}
        
        @self.mcp.tool(name="create_review_workflow")
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
                    "name": workflow.get("name"),
                    "description": workflow.get("description"),
                    "files_to_review": files_to_review,
                    "status": workflow.get("status"),
                    "created_at": workflow.get("created_at").isoformat() if workflow.get("created_at") else None
                }
            except Exception as e:
                logger.error(f"Failed to create code review workflow: {e}")
                return {"error": f"Failed to create code review workflow: {str(e)}"}
                
        @self.mcp.tool(name="execute_workflow")
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
                
        @self.mcp.tool(name="get_workflow_status")
        async def get_workflow_status(workflow_id):
            if workflow_id not in self.workflows:
                return {"error": f"Workflow not found: {workflow_id}"}
            
            # Get the workflow
            workflow = self.workflows[workflow_id]
            
            # Get task statuses
            dag = workflow["dag"]
            task_statuses = {}
            for task_id, task in dag.tasks.items():
                # Convert datetime objects to ISO format strings if they exist
                started_at = task.start_time.isoformat() if task.start_time else None
                completed_at = task.end_time.isoformat() if task.end_time else None
                
                task_statuses[task_id] = {
                    "status": task.status.value if hasattr(task.status, "value") else task.status,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "error": str(task.error) if task.error else None
                }
            
            return {
                "workflow_id": workflow_id,
                "name": workflow["name"],
                "status": workflow["status"],
                "created_at": workflow["created_at"],
                "started_at": workflow.get("started_at", None),
                "completed_at": workflow.get("completed_at", None),
                "tasks": task_statuses
            }
                
        @self.mcp.tool(name="list_workflows")
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
                        "name": workflow.get("name"),
                        "description": workflow.get("description"),
                        "status": workflow.get("status"),
                        "created_at": workflow.get("created_at").isoformat() if workflow.get("created_at") else None,
                        "completed_at": workflow.get("completed_at").isoformat() if workflow.get("completed_at") else None
                    })
                
                return {
                    "count": len(workflows_info),
                    "workflows": workflows_info
                }
            except Exception as e:
                logger.error(f"Failed to list workflows: {e}")
                return {"error": f"Failed to list workflows: {str(e)}"}
                
        @self.mcp.tool(name="visualize_workflow")
        async def visualize_workflow(workflow_id):
            if workflow_id not in self.workflows:
                return {"error": f"Workflow not found: {workflow_id}"}
            
            # Get the workflow
            workflow = self.workflows[workflow_id]
            dag = workflow["dag"]
            
            # Create a visualization
            visualization = []
            visualization.append(f"Workflow: {workflow['name']} ({workflow_id})")
            visualization.append(f"Status: {workflow['status']}")
            visualization.append("Tasks:")
            
            for task_id, task in dag.tasks.items():
                dependencies = ", ".join(task.dependencies) if hasattr(task, 'dependencies') and task.dependencies else "None"
                status_value = task.status.value if hasattr(task.status, 'value') else str(task.status)
                
                visualization.append(f"  - {task_id} ({status_value}):")
                visualization.append(f"    Tool: {task.tool_name}.{task.function_name}" if hasattr(task, 'tool_name') and hasattr(task, 'function_name') else f"    Function: {task.func.__name__ if hasattr(task.func, '__name__') else 'unknown'}")
                visualization.append(f"    Dependencies: {dependencies}")
                
                if hasattr(task, 'start_time') and task.start_time:
                    visualization.append(f"    Started: {task.start_time.isoformat() if hasattr(task.start_time, 'isoformat') else str(task.start_time)}")
                if hasattr(task, 'end_time') and task.end_time:
                    visualization.append(f"    Completed: {task.end_time.isoformat() if hasattr(task.end_time, 'isoformat') else str(task.end_time)}")
                if hasattr(task, 'error') and task.error:
                    visualization.append(f"    Error: {str(task.error)}")
            
            return {
                "workflow_id": workflow_id,
                "name": workflow["name"],
                "visualization": visualization
            }
                
        @self.mcp.tool(name="create_knowledge_reasoning_workflow")
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
                    "name": workflow.get("name"),
                    "description": workflow.get("description"),
                    "reasoning_request": reasoning_request,
                    "status": workflow.get("status"),
                    "created_at": workflow.get("created_at").isoformat() if workflow.get("created_at") else None
                }
            except Exception as e:
                logger.error(f"Failed to create knowledge reasoning workflow: {e}")
                return {"error": f"Failed to create knowledge reasoning workflow: {str(e)}"}
        
        logger.info("Registered workflow management tools")
    
    def _register_dag_tools(self) -> None:
        """Register DAG orchestration tools with the MCP server."""
        
        # FastMCP 2.1.0 no longer supports register_tools method
        # Instead, we need to use decorators to register tools
        
        @self.mcp.tool(name="create_workflow")
        async def create_workflow(name, description=""):
            async def _create_workflow():
                workflow_id = f"workflow_{len(self.workflows) + 1}"
                
                # Create a new DAG
                dag = DAG(
                    name=name,
                    description=description
                )
                
                # Store the workflow
                self.workflows[workflow_id] = {
                    "dag": dag,
                    "name": name,
                    "description": description,
                    "status": "created",
                    "created_at": datetime.now().isoformat()
                }
                
                logger.info(f"Created workflow {workflow_id}: {name}")
                
                return workflow_id
                
            # Execute with timeout and recovery
            return await self._execute_with_timeout_and_recovery(
                _create_workflow,
                "create_workflow",
                timeout=10.0
            )
        
        @self.mcp.tool(name="add_task_to_workflow")
        async def add_task_to_workflow(workflow_id, task_id, tool_name, function_name, parameters, dependencies=None, timeout=None, retry_count=0, description=""):
            # Validate inputs
            if workflow_id not in self.workflows:
                return {"error": f"Workflow not found: {workflow_id}"}
            
            # Check if tool and function exist
            tool_function = self._get_tool_function(tool_name, function_name)
            if not tool_function:
                return {"error": f"Function {function_name} not found in tool {tool_name}"}
            
            # Add task to DAG
            dag = self.workflows[workflow_id]["dag"]
            
            # Create a DAG task
            dag_task = DAGTask(
                task_id=task_id,
                tool_name=tool_name,
                function_name=function_name,
                parameters=parameters,
                dependencies=dependencies or [],
                timeout=timeout,
                retry_count=retry_count,
                description=description,
                function=tool_function,
                execute_with_timeout=lambda func, **kwargs: self._execute_with_timeout_and_recovery(
                    func, task_id, workflow_id=workflow_id, timeout=timeout, retry_count=retry_count, **kwargs
                )
            )
            
            # Add the task to the DAG
            dag.add_task(dag_task)
            
            logger.info(f"Added task {task_id} to workflow {workflow_id}")
            
            return {
                "workflow_id": workflow_id,
                "task_id": task_id,
                "status": "added"
            }
        
        @self.mcp.tool(name="execute_workflow")
        async def execute_workflow(workflow_id):
            if workflow_id not in self.workflows:
                return {"error": f"Workflow not found: {workflow_id}"}
            
            async def _execute_workflow():
                # Get the workflow
                workflow = self.workflows[workflow_id]
                dag = workflow["dag"]
                
                # Update workflow status
                workflow["status"] = "running"
                workflow["started_at"] = datetime.now().isoformat()
                
                try:
                    # Execute the DAG
                    results = await self.dag_executor.execute_dag(dag, context={"workflow_id": workflow_id})
                    
                    # Update workflow status
                    workflow["status"] = "completed"
                    workflow["completed_at"] = datetime.now().isoformat()
                    workflow["results"] = results
                    
                    return {
                        "workflow_id": workflow_id,
                        "status": "completed",
                        "results": results
                    }
                except Exception as e:
                    # Update workflow status
                    workflow["status"] = "failed"
                    workflow["error"] = str(e)
                    
                    logger.error(f"Workflow {workflow_id} failed: {e}")
                    return {
                        "workflow_id": workflow_id,
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Execute with timeout
            return await self._execute_with_timeout_and_recovery(
                _execute_workflow,
                f"execute_workflow_{workflow_id}",
                timeout=None  # No timeout for workflow execution
            )
        
        @self.mcp.tool(name="get_workflow_status")
        async def get_workflow_status(workflow_id):
            if workflow_id not in self.workflows:
                return {"error": f"Workflow not found: {workflow_id}"}
            
            # Get the workflow
            workflow = self.workflows[workflow_id]
            
            # Get task statuses
            dag = workflow["dag"]
            task_statuses = {}
            for task_id, task in dag.tasks.items():
                # Convert datetime objects to ISO format strings if they exist
                started_at = task.start_time.isoformat() if task.start_time else None
                completed_at = task.end_time.isoformat() if task.end_time else None
                
                task_statuses[task_id] = {
                    "status": task.status.value if hasattr(task.status, "value") else task.status,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "error": str(task.error) if task.error else None
                }
            
            return {
                "workflow_id": workflow_id,
                "name": workflow["name"],
                "status": workflow["status"],
                "created_at": workflow["created_at"],
                "started_at": workflow.get("started_at", None),
                "completed_at": workflow.get("completed_at", None),
                "tasks": task_statuses
            }
        
        @self.mcp.tool(name="visualize_workflow")
        async def visualize_workflow(workflow_id):
            if workflow_id not in self.workflows:
                return {"error": f"Workflow not found: {workflow_id}"}
            
            # Get the workflow
            workflow = self.workflows[workflow_id]
            dag = workflow["dag"]
            
            # Create a visualization
            visualization = []
            visualization.append(f"Workflow: {workflow['name']} ({workflow_id})")
            visualization.append(f"Status: {workflow['status']}")
            visualization.append("Tasks:")
            
            for task_id, task in dag.tasks.items():
                dependencies = ", ".join(task.dependencies) if hasattr(task, 'dependencies') and task.dependencies else "None"
                status_value = task.status.value if hasattr(task.status, 'value') else str(task.status)
                
                visualization.append(f"  - {task_id} ({status_value}):")
                visualization.append(f"    Tool: {task.tool_name}.{task.function_name}" if hasattr(task, 'tool_name') and hasattr(task, 'function_name') else f"    Function: {task.func.__name__ if hasattr(task.func, '__name__') else 'unknown'}")
                visualization.append(f"    Dependencies: {dependencies}")
                
                if hasattr(task, 'start_time') and task.start_time:
                    visualization.append(f"    Started: {task.start_time.isoformat() if hasattr(task.start_time, 'isoformat') else str(task.start_time)}")
                if hasattr(task, 'end_time') and task.end_time:
                    visualization.append(f"    Completed: {task.end_time.isoformat() if hasattr(task.end_time, 'isoformat') else str(task.end_time)}")
                if hasattr(task, 'error') and task.error:
                    visualization.append(f"    Error: {str(task.error)}")
            
            return {
                "workflow_id": workflow_id,
                "name": workflow["name"],
                "visualization": visualization
            }
    
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
        # FastMCP 2.1.0 no longer supports register_tools method
        # Instead, we need to use decorators to register tools
        
        @self.mcp.tool(name="get_workflow_error_summary")
        def get_workflow_error_summary(workflow_id=None):
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

        @self.mcp.tool(name="update_task_timeout")
        def update_task_timeout(task_id, timeout):
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

        @self.mcp.tool(name="adapt_task_timeout")
        def adapt_task_timeout(task_id):
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
        
        @self.mcp.tool(name="get_circuit_breaker_status")
        def get_circuit_breaker_status(service_name):
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
        
        @self.mcp.tool(name="reset_circuit_breaker")
        def reset_circuit_breaker(service_name):
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
        
        @self.mcp.tool(name="get_all_circuit_breakers")
        def get_all_circuit_breakers():
            """
            Get status of all circuit breakers in the system.
            
            Returns:
                Dict containing all circuit breaker statuses
            """
            if not self.circuit_manager:
                return {"error": "Circuit breaker manager not initialized"}
            
            return self.circuit_manager.get_all_health()

        logger.info("Error handling tools registered successfully")

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