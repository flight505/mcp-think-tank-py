"""
DAG Orchestrator for multi-tool execution workflows.

This module provides functionality for defining and executing directed acyclic graphs (DAGs)
of tasks, with support for parallel execution, dependency management, timeout handling,
and error recovery.
"""

import asyncio
import time
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Set, Callable, Any, Optional, Union, Tuple
from graphlib import TopologicalSorter

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Enum representing the status of a task in the DAG."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"


class DAGTask:
    """
    A task in a DAG with support for timeout handling, retries, and error recovery.
    """
    
    def __init__(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        description: str = "",
    ):
        """
        Initialize a DAG task.
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            timeout: Timeout in seconds (None for no timeout)
            retry_count: Number of times to retry on failure
            retry_delay: Delay between retries in seconds
            description: Description of the task
        """
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.timeout = timeout
        self.max_retries = retry_count
        self.retry_delay = retry_delay
        self.description = description
        
        # Execution state
        self.status = TaskStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.attempt = 0
        self.error = None
        self.fallback_used = False
    
    async def execute(self) -> Any:
        """
        Execute the task with timeout handling and retries.
        
        Returns:
            Result of the function execution
            
        Raises:
            Exception: If the task fails and retry_count is exceeded
        """
        self.start_time = datetime.now()
        self.status = TaskStatus.RUNNING
        self.attempt += 1
        
        try:
            # Execute the function with timeout if specified
            if asyncio.iscoroutinefunction(self.func):
                if self.timeout is not None:
                    result = await asyncio.wait_for(
                        self.func(*self.args, **self.kwargs),
                        timeout=self.timeout
                    )
                else:
                    result = await self.func(*self.args, **self.kwargs)
            else:
                # Handle synchronous functions
                if self.timeout is not None:
                    # Create an awaitable wrapper to apply timeout
                    async def _sync_wrapper():
                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(
                            None, lambda: self.func(*self.args, **self.kwargs)
                        )
                    result = await asyncio.wait_for(_sync_wrapper(), timeout=self.timeout)
                else:
                    # Execute synchronously in a separate thread
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, lambda: self.func(*self.args, **self.kwargs)
                    )
            
            # Task completed successfully
            self.status = TaskStatus.COMPLETED
            self.end_time = datetime.now()
            self.duration = (self.end_time - self.start_time).total_seconds()
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Task {self.task_id} timed out after {self.timeout}s")
            self.error = f"Timeout after {self.timeout}s"
            
            # Retry if we haven't reached the maximum number of retries
            if self.attempt <= self.max_retries:
                logger.info(f"Retrying task {self.task_id} ({self.attempt}/{self.max_retries})")
                # Apply exponential backoff for retries
                retry_delay = self.retry_delay * (2 ** (self.attempt - 1))
                await asyncio.sleep(retry_delay)
                return await self.execute()
            
            # Mark as timed out if we've exhausted retries
            self.status = TaskStatus.TIMED_OUT
            self.end_time = datetime.now()
            self.duration = (self.end_time - self.start_time).total_seconds()
            raise
            
        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {str(e)}")
            self.error = str(e)
            
            # Retry if we haven't reached the maximum number of retries
            if self.attempt <= self.max_retries:
                logger.info(f"Retrying task {self.task_id} ({self.attempt}/{self.max_retries})")
                # Apply exponential backoff for retries
                retry_delay = self.retry_delay * (2 ** (self.attempt - 1))
                await asyncio.sleep(retry_delay)
                return await self.execute()
            
            # Mark as failed if we've exhausted retries
            self.status = TaskStatus.FAILED
            self.end_time = datetime.now()
            self.duration = (self.end_time - self.start_time).total_seconds()
            raise
    
    async def execute_with_fallback(self, fallback_func: Callable = None) -> Any:
        """
        Execute the task with fallback in case of failure.
        
        Args:
            fallback_func: Fallback function to execute if the main function fails
            
        Returns:
            Result of the function execution or fallback execution
        """
        try:
            return await self.execute()
        except Exception as e:
            if fallback_func is not None:
                logger.info(f"Using fallback for task {self.task_id} after failure: {str(e)}")
                self.fallback_used = True
                
                # Create a new task for the fallback
                fallback_task = DAGTask(
                    task_id=f"{self.task_id}_fallback",
                    func=fallback_func,
                    args=self.args,
                    kwargs=self.kwargs,
                    timeout=self.timeout,
                    retry_count=0,  # No retries for fallback
                    description=f"Fallback for {self.description}"
                )
                
                # Execute the fallback
                result = await fallback_task.execute()
                
                # Update our status since fallback succeeded
                self.status = TaskStatus.COMPLETED
                self.end_time = datetime.now()
                self.duration = (self.end_time - self.start_time).total_seconds()
                return result
            else:
                # Re-raise the exception if no fallback is available
                raise


class DAGExecutor:
    """
    Executes a directed acyclic graph (DAG) of tasks with dependency management.
    
    This class handles the parallel execution of tasks based on their dependencies,
    with support for timeout handling, error recovery, and performance monitoring.
    """
    
    def __init__(
        self,
        max_concurrency: int = 10,
        global_timeout: Optional[float] = None,
        fail_fast: bool = False,
        metrics_enabled: bool = True,
    ):
        """
        Initialize a DAG executor.
        
        Args:
            max_concurrency: Maximum number of tasks to execute in parallel
            global_timeout: Maximum execution time for the entire DAG in seconds
            fail_fast: If True, fail the entire DAG on first task failure
            metrics_enabled: If True, collect performance metrics
        """
        self.max_concurrency = max_concurrency
        self.global_timeout = global_timeout
        self.fail_fast = fail_fast
        self.metrics_enabled = metrics_enabled
        
        # DAG state
        self.tasks: Dict[str, DAGTask] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
        self.execution_results: Dict[str, Any] = {}
        self.execution_errors: Dict[str, str] = {}
        
        # Metrics
        self.start_time = None
        self.end_time = None
        self.total_duration = None
    
    def add_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        dependencies: List[str] = None,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        description: str = "",
    ) -> str:
        """
        Add a task to the DAG.
        
        Args:
            task_id: Unique identifier for the task (if empty, a UUID will be generated)
            func: Callable function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            dependencies: List of task IDs that must complete before this task
            timeout: Maximum execution time in seconds (None for no timeout)
            retry_count: Number of retry attempts on failure
            retry_delay: Delay between retry attempts in seconds
            description: Human-readable description of the task
            
        Returns:
            The task_id (generated if not provided)
            
        Raises:
            ValueError: If a task with the same ID already exists or if a dependency 
                       doesn't exist
        """
        # Generate task_id if not provided
        if not task_id:
            task_id = str(uuid.uuid4())
            
        # Check if task_id already exists
        if task_id in self.tasks:
            raise ValueError(f"Task with ID '{task_id}' already exists")
            
        # Create the task
        task = DAGTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs or {},
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
            description=description,
        )
        
        # Add the task to the DAG
        self.tasks[task_id] = task
        self.dependencies[task_id] = set()
        
        # Initialize reverse dependencies for this task
        if task_id not in self.reverse_dependencies:
            self.reverse_dependencies[task_id] = set()
            
        # Add dependencies
        if dependencies:
            for dep_id in dependencies:
                if dep_id not in self.tasks:
                    raise ValueError(f"Dependency '{dep_id}' does not exist")
                self.dependencies[task_id].add(dep_id)
                
                # Update reverse dependencies
                if dep_id not in self.reverse_dependencies:
                    self.reverse_dependencies[dep_id] = set()
                self.reverse_dependencies[dep_id].add(task_id)
        
        return task_id
    
    def add_dependency(self, from_task_id: str, to_task_id: str) -> None:
        """
        Add a dependency between two tasks.
        
        Args:
            from_task_id: The ID of the task that must complete first
            to_task_id: The ID of the task that depends on from_task_id
            
        Raises:
            ValueError: If either task doesn't exist or if adding the dependency
                       would create a cycle
        """
        # Check if tasks exist
        if from_task_id not in self.tasks:
            raise ValueError(f"Task '{from_task_id}' does not exist")
        if to_task_id not in self.tasks:
            raise ValueError(f"Task '{to_task_id}' does not exist")
            
        # Add dependency
        self.dependencies[to_task_id].add(from_task_id)
        
        # Update reverse dependencies
        if from_task_id not in self.reverse_dependencies:
            self.reverse_dependencies[from_task_id] = set()
        self.reverse_dependencies[from_task_id].add(to_task_id)
        
        # Check for cycles
        try:
            self._get_topological_sorter()
        except Exception as e:
            # Remove the dependency if it creates a cycle
            self.dependencies[to_task_id].remove(from_task_id)
            self.reverse_dependencies[from_task_id].remove(to_task_id)
            raise ValueError(f"Adding this dependency would create a cycle in the DAG: {str(e)}")
    
    def _get_topological_sorter(self) -> TopologicalSorter:
        """Create a TopologicalSorter instance for the current DAG."""
        graph = {task_id: set(deps) for task_id, deps in self.dependencies.items()}
        return TopologicalSorter(graph)
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the DAG with parallel task execution based on dependencies.
        
        Returns:
            Dict mapping task IDs to their results
            
        Raises:
            asyncio.TimeoutError: If the global timeout is exceeded
            Exception: If a critical task fails and fail_fast is True
        """
        if not self.tasks:
            logger.warning("No tasks to execute in the DAG")
            return {}
            
        self.start_time = datetime.now()
        logger.info(f"Starting DAG execution with {len(self.tasks)} tasks")
        
        # Set up task execution
        sorter = self._get_topological_sorter()
        sorter.prepare()
        
        # Set up semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        # Track running tasks
        running_tasks = set()
        
        # Define task wrapper for concurrency control
        async def run_task_with_semaphore(task_id: str) -> None:
            async with semaphore:
                task = self.tasks[task_id]
                try:
                    result = await task.execute()
                    self.execution_results[task_id] = result
                except Exception as e:
                    self.execution_errors[task_id] = str(e)
                    if self.fail_fast:
                        # Cancel all running tasks if fail_fast is enabled
                        for running_id in running_tasks:
                            if running_id != task_id and running_id in running_tasks:
                                # Mark downstream tasks as skipped
                                downstream = self._get_all_descendants(task_id)
                                for ds_id in downstream:
                                    if self.tasks[ds_id].status == TaskStatus.PENDING:
                                        self.tasks[ds_id].status = TaskStatus.SKIPPED
                        
                        raise
                finally:
                    running_tasks.remove(task_id)
        
        # Execute the DAG with global timeout
        try:
            async def execute_dag():
                # Process tasks in topological order
                while sorter.is_active():
                    # Get all ready tasks (no pending dependencies)
                    ready_tasks = list(sorter.get_ready())
                    
                    if not ready_tasks and running_tasks:
                        # Wait for any running task to complete
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Start execution of ready tasks
                    for task_id in ready_tasks:
                        running_tasks.add(task_id)
                        asyncio.create_task(run_task_with_semaphore(task_id))
                        sorter.done(task_id)
                    
                    # Wait a bit before checking again
                    await asyncio.sleep(0.1)
                
                # Wait for all running tasks to complete
                while running_tasks:
                    await asyncio.sleep(0.1)
            
            # Execute with global timeout if specified
            if self.global_timeout is not None:
                await asyncio.wait_for(execute_dag(), timeout=self.global_timeout)
            else:
                await execute_dag()
                
        except asyncio.TimeoutError:
            logger.error(f"DAG execution timed out after {self.global_timeout}s")
            # Mark remaining tasks as timed out
            for task_id, task in self.tasks.items():
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    task.status = TaskStatus.TIMED_OUT
                    task.error = f"DAG global timeout after {self.global_timeout}s"
            raise
            
        finally:
            self.end_time = datetime.now()
            self.total_duration = (self.end_time - self.start_time).total_seconds()
            logger.info(f"DAG execution completed in {self.total_duration:.3f}s")
            
            if self.metrics_enabled:
                self._log_execution_metrics()
        
        return self.execution_results
    
    def _get_all_descendants(self, task_id: str) -> Set[str]:
        """Get all descendant tasks (direct and indirect) of a given task."""
        descendants = set()
        to_process = [task_id]
        
        while to_process:
            current = to_process.pop()
            if current in self.reverse_dependencies:
                for child in self.reverse_dependencies[current]:
                    if child not in descendants:
                        descendants.add(child)
                        to_process.append(child)
        
        return descendants
    
    def _log_execution_metrics(self) -> None:
        """Log performance metrics for the DAG execution."""
        completed = sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
        failed = sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)
        timed_out = sum(1 for task in self.tasks.values() if task.status == TaskStatus.TIMED_OUT)
        skipped = sum(1 for task in self.tasks.values() if task.status == TaskStatus.SKIPPED)
        
        logger.info(f"DAG Execution Metrics:")
        logger.info(f"  Total tasks: {len(self.tasks)}")
        logger.info(f"  Completed: {completed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Timed out: {timed_out}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Total execution time: {self.total_duration:.3f}s")
        
        # Log task-specific metrics
        for task_id, task in self.tasks.items():
            if task.duration is not None:
                logger.debug(f"  Task {task_id}: status={task.status.value}, "
                           f"duration={task.duration:.3f}s, attempts={task.attempt}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the DAG execution.
        
        Returns:
            Dict containing execution metrics and task statuses
        """
        if not self.end_time:
            return {"status": "not_executed"}
            
        task_statuses = {
            task_id: {
                "status": task.status.value,
                "duration": task.duration,
                "attempts": task.attempt,
                "error": task.error,
            }
            for task_id, task in self.tasks.items()
        }
        
        return {
            "status": "completed",
            "total_tasks": len(self.tasks),
            "completed": sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED),
            "failed": sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED),
            "timed_out": sum(1 for task in self.tasks.values() if task.status == TaskStatus.TIMED_OUT),
            "skipped": sum(1 for task in self.tasks.values() if task.status == TaskStatus.SKIPPED),
            "total_duration": self.total_duration,
            "task_statuses": task_statuses,
        }
    
    def visualize(self) -> str:
        """
        Generate a textual visualization of the DAG.
        
        Returns:
            A string representation of the DAG structure
        """
        result = ["DAG Structure:"]
        
        # Try to get a topological sorting
        try:
            sorter = self._get_topological_sorter()
            sorter.prepare()
            levels = {}
            current_level = 0
            
            while sorter.is_active():
                nodes = list(sorter.get_ready())
                for node in nodes:
                    levels[node] = current_level
                    sorter.done(node)
                current_level += 1
                
            # Build visualization based on levels
            for level in range(current_level):
                level_nodes = [node for node, node_level in levels.items() if node_level == level]
                level_nodes.sort()  # Sort for consistent output
                
                result.append(f"Level {level}:")
                for node in level_nodes:
                    deps = self.dependencies[node]
                    deps_str = f" (depends on: {', '.join(sorted(deps))})" if deps else ""
                    task = self.tasks[node]
                    status = f" [{task.status.value}]" if task.status != TaskStatus.PENDING else ""
                    result.append(f"  - {node}{deps_str}{status}")
                    
        except Exception as e:
            # Fall back to simple listing if there's a cycle or other issue
            result = ["DAG Structure (unordered due to possible cycle):"]
            for node, deps in sorted(self.dependencies.items()):
                deps_str = f" (depends on: {', '.join(sorted(deps))})" if deps else ""
                task = self.tasks[node]
                status = f" [{task.status.value}]" if task.status != TaskStatus.PENDING else ""
                result.append(f"  - {node}{deps_str}{status}")
            
        return "\n".join(result)


class EmbeddingCache:
    """
    Cache for embedding vectors to improve performance.
    
    This class provides a simple in-memory cache for embedding vectors
    to reduce redundant computation of embeddings for the same content.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize an embedding cache.
        
        Args:
            max_size: Maximum number of entries in the cache
        """
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
    
    def get(self, content: str, model: Optional[str] = None) -> Optional[List[float]]:
        """
        Get embedding from cache if available.
        
        Args:
            content: The text content to get embedding for
            model: The embedding model used (if multiple models are cached)
            
        Returns:
            The cached embedding vector or None if not in cache
        """
        key = self._make_key(content, model)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]["embedding"]
        return None
    
    def put(self, content: str, embedding: List[float], model: Optional[str] = None) -> None:
        """
        Store embedding in cache.
        
        Args:
            content: The text content the embedding is for
            embedding: The embedding vector to cache
            model: The embedding model used (if multiple models are cached)
        """
        key = self._make_key(content, model)
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Get least recently used key
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = {
            "embedding": embedding,
            "model": model,
        }
        self.access_times[key] = time.time()
    
    def _make_key(self, content: str, model: Optional[str] = None) -> str:
        """Create a cache key from content and model."""
        model_str = model or "default"
        # Use hash for content to avoid extremely long keys
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{model_str}:{content_hash}"
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_usage_kb": sum(len(str(v)) for v in self.cache.values()) / 1024,
        } 