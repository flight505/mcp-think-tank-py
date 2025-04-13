#!/usr/bin/env python3
"""
Workflow Error Handler for MCP Think Tank
Provides enhanced error handling and recovery for workflows
"""
import logging
import traceback
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum

logger = logging.getLogger("mcp-think-tank.workflow_error_handler")

class ErrorSeverity(Enum):
    """Enum representing the severity of errors in workflows."""
    LOW = "low"          # Non-critical errors that don't affect the workflow
    MEDIUM = "medium"    # Errors that may affect some parts of the workflow
    HIGH = "high"        # Critical errors that prevent the workflow from completing
    FATAL = "fatal"      # Errors that require immediate termination of the workflow

class ErrorType(Enum):
    """Enum representing types of errors that can occur in workflows."""
    TIMEOUT = "timeout"              # Operation timed out
    DEPENDENCY_MISSING = "dependency_missing"  # Required dependency is missing
    TOOL_ERROR = "tool_error"        # Error in a specific tool
    INVALID_INPUT = "invalid_input"  # Invalid input to a tool
    RESOURCE_ERROR = "resource_error" # Error accessing a resource
    NETWORK_ERROR = "network_error"  # Network-related error
    UNKNOWN = "unknown"              # Unknown or unclassified error

class WorkflowError:
    """
    Class representing an error that occurred during workflow execution.
    
    This class captures detailed information about errors to aid in
    diagnosis and recovery.
    """
    
    def __init__(
        self,
        error: Exception,
        task_id: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_type: ErrorType = ErrorType.UNKNOWN,
        context: Dict[str, Any] = None,
        timestamp: float = None,
        retry_count: int = 0,
        max_retries: int = 3,
    ):
        """
        Initialize a workflow error.
        
        Args:
            error: The exception that occurred
            task_id: ID of the task where the error occurred
            severity: Severity of the error
            error_type: Type of the error
            context: Additional context about the error
            timestamp: When the error occurred (defaults to current time)
            retry_count: Number of retries already attempted
            max_retries: Maximum number of retries allowed
        """
        self.error = error
        self.task_id = task_id
        self.severity = severity
        self.error_type = error_type
        self.context = context or {}
        self.timestamp = timestamp or time.time()
        self.stacktrace = traceback.format_exc()
        self.retry_count = retry_count
        self.max_retries = max_retries
        
    def __str__(self) -> str:
        """String representation of the error."""
        return (f"WorkflowError: {self.error_type.value} in task {self.task_id} "
                f"(severity: {self.severity.value}, retries: {self.retry_count}/{self.max_retries}): {str(self.error)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary."""
        return {
            "error": str(self.error),
            "task_id": self.task_id,
            "severity": self.severity.value,
            "error_type": self.error_type.value,
            "context": self.context,
            "timestamp": self.timestamp,
            "stacktrace": self.stacktrace,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "can_retry": self.retry_count < self.max_retries
        }
    
    @property
    def can_retry(self) -> bool:
        """Whether the error can be retried."""
        return self.retry_count < self.max_retries

class WorkflowErrorHandler:
    """
    Handler for workflow errors with enhanced recovery mechanisms.
    
    This class provides methods for handling errors that occur during
    workflow execution, with support for retries, fallbacks, and
    graceful degradation.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.errors = []
        self.recovery_strategies = {
            ErrorType.TIMEOUT: self._handle_timeout,
            ErrorType.DEPENDENCY_MISSING: self._handle_dependency_missing,
            ErrorType.TOOL_ERROR: self._handle_tool_error,
            ErrorType.INVALID_INPUT: self._handle_invalid_input,
            ErrorType.RESOURCE_ERROR: self._handle_resource_error,
            ErrorType.NETWORK_ERROR: self._handle_network_error,
            ErrorType.UNKNOWN: self._handle_unknown_error
        }
    
    def handle_error(self, 
                    error: Exception, 
                    task_id: str, 
                    context: Dict[str, Any] = None,
                    retry_count: int = 0,
                    max_retries: int = 3) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle a workflow error.
        
        Args:
            error: The exception that occurred
            task_id: ID of the task where the error occurred
            context: Additional context about the error
            retry_count: Number of retries already attempted
            max_retries: Maximum number of retries allowed
            
        Returns:
            Tuple of (can_continue, recovery_info)
        """
        # Classify the error
        error_type, severity = self._classify_error(error)
        
        # Create a WorkflowError object
        workflow_error = WorkflowError(
            error=error,
            task_id=task_id,
            severity=severity,
            error_type=error_type,
            context=context,
            retry_count=retry_count,
            max_retries=max_retries
        )
        
        # Add to error log
        self.errors.append(workflow_error)
        
        # Log the error
        logger.error(str(workflow_error))
        
        # Get recovery strategy based on error type
        recovery_strategy = self.recovery_strategies.get(error_type, self._handle_unknown_error)
        
        # Execute recovery strategy
        can_continue, recovery_info = recovery_strategy(workflow_error)
        
        # If error is fatal, we can't continue
        if severity == ErrorSeverity.FATAL:
            can_continue = False
            recovery_info["reason"] = "Fatal error"
        
        return can_continue, recovery_info
    
    def _classify_error(self, error: Exception) -> Tuple[ErrorType, ErrorSeverity]:
        """
        Classify an error by type and severity.
        
        Args:
            error: The exception to classify
            
        Returns:
            Tuple of (error_type, severity)
        """
        # Default classification
        error_type = ErrorType.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        
        # Check for timeout errors
        if isinstance(error, asyncio.TimeoutError):
            error_type = ErrorType.TIMEOUT
            severity = ErrorSeverity.MEDIUM
        
        # Check for network errors
        elif "ConnectionError" in error.__class__.__name__ or "Timeout" in error.__class__.__name__:
            error_type = ErrorType.NETWORK_ERROR
            severity = ErrorSeverity.MEDIUM
        
        # Check for ValueError (often invalid input)
        elif isinstance(error, ValueError):
            error_type = ErrorType.INVALID_INPUT
            severity = ErrorSeverity.MEDIUM
        
        # Check for KeyError, IndexError (often missing dependencies)
        elif isinstance(error, (KeyError, IndexError)):
            error_type = ErrorType.DEPENDENCY_MISSING
            severity = ErrorSeverity.HIGH
        
        # Check for FileNotFoundError, PermissionError (resource errors)
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            error_type = ErrorType.RESOURCE_ERROR
            severity = ErrorSeverity.HIGH
        
        # Check error message for clues
        error_msg = str(error).lower()
        if "timeout" in error_msg:
            error_type = ErrorType.TIMEOUT
            severity = ErrorSeverity.MEDIUM
        elif "network" in error_msg or "connection" in error_msg:
            error_type = ErrorType.NETWORK_ERROR
            severity = ErrorSeverity.MEDIUM
        elif "not found" in error_msg or "missing" in error_msg:
            error_type = ErrorType.DEPENDENCY_MISSING
            severity = ErrorSeverity.HIGH
        elif "invalid" in error_msg or "bad" in error_msg:
            error_type = ErrorType.INVALID_INPUT
            severity = ErrorSeverity.MEDIUM
        elif "permission" in error_msg or "access" in error_msg:
            error_type = ErrorType.RESOURCE_ERROR
            severity = ErrorSeverity.HIGH
        
        return error_type, severity
    
    def _handle_timeout(self, error: WorkflowError) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle timeout errors.
        
        Args:
            error: The workflow error
            
        Returns:
            Tuple of (can_continue, recovery_info)
        """
        if error.can_retry:
            # For timeouts, we can retry with increased timeout
            return True, {
                "action": "retry",
                "task_id": error.task_id,
                "retry_count": error.retry_count + 1,
                "increase_timeout": True,
                "timeout_multiplier": 1.5,  # Increase timeout by 50%
                "reason": "Timeout error, will retry with increased timeout"
            }
        else:
            # If we've exceeded retries, we can try to continue with reduced functionality
            return True, {
                "action": "degrade",
                "task_id": error.task_id,
                "reason": "Timeout error, exceeded retries, continuing with degraded functionality"
            }
    
    def _handle_dependency_missing(self, error: WorkflowError) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle dependency missing errors.
        
        Args:
            error: The workflow error
            
        Returns:
            Tuple of (can_continue, recovery_info)
        """
        # Dependency missing errors are usually critical
        return False, {
            "action": "abort",
            "task_id": error.task_id,
            "reason": f"Missing dependency: {str(error.error)}"
        }
    
    def _handle_tool_error(self, error: WorkflowError) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle tool errors.
        
        Args:
            error: The workflow error
            
        Returns:
            Tuple of (can_continue, recovery_info)
        """
        if error.can_retry:
            # For tool errors, we can retry
            return True, {
                "action": "retry",
                "task_id": error.task_id,
                "retry_count": error.retry_count + 1,
                "reason": "Tool error, will retry"
            }
        else:
            # If we've exceeded retries, we can try a fallback tool if available
            return True, {
                "action": "fallback",
                "task_id": error.task_id,
                "use_fallback_tool": True,
                "reason": "Tool error, exceeded retries, using fallback tool"
            }
    
    def _handle_invalid_input(self, error: WorkflowError) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle invalid input errors.
        
        Args:
            error: The workflow error
            
        Returns:
            Tuple of (can_continue, recovery_info)
        """
        # Invalid input errors usually can't be fixed by retrying
        return False, {
            "action": "abort",
            "task_id": error.task_id,
            "reason": f"Invalid input: {str(error.error)}"
        }
    
    def _handle_resource_error(self, error: WorkflowError) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle resource errors.
        
        Args:
            error: The workflow error
            
        Returns:
            Tuple of (can_continue, recovery_info)
        """
        if error.can_retry:
            # For resource errors, we can retry after a delay
            return True, {
                "action": "retry",
                "task_id": error.task_id,
                "retry_count": error.retry_count + 1,
                "delay": 2 ** error.retry_count,  # Exponential backoff
                "reason": "Resource error, will retry after delay"
            }
        else:
            # If we've exceeded retries, we can try a fallback resource if available
            return True, {
                "action": "fallback",
                "task_id": error.task_id,
                "use_fallback_resource": True,
                "reason": "Resource error, exceeded retries, using fallback resource"
            }
    
    def _handle_network_error(self, error: WorkflowError) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle network errors.
        
        Args:
            error: The workflow error
            
        Returns:
            Tuple of (can_continue, recovery_info)
        """
        if error.can_retry:
            # For network errors, we can retry after a delay
            return True, {
                "action": "retry",
                "task_id": error.task_id,
                "retry_count": error.retry_count + 1,
                "delay": 2 ** error.retry_count,  # Exponential backoff
                "reason": "Network error, will retry after delay"
            }
        else:
            # If we've exceeded retries, we can try to continue with cached data if available
            return True, {
                "action": "use_cache",
                "task_id": error.task_id,
                "reason": "Network error, exceeded retries, using cached data"
            }
    
    def _handle_unknown_error(self, error: WorkflowError) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle unknown errors.
        
        Args:
            error: The workflow error
            
        Returns:
            Tuple of (can_continue, recovery_info)
        """
        if error.can_retry and error.severity != ErrorSeverity.FATAL:
            # For unknown errors, we can retry if not fatal
            return True, {
                "action": "retry",
                "task_id": error.task_id,
                "retry_count": error.retry_count + 1,
                "reason": "Unknown error, will retry"
            }
        else:
            # If we've exceeded retries or error is fatal, we should abort
            return False, {
                "action": "abort",
                "task_id": error.task_id,
                "reason": f"Unknown error: {str(error.error)}"
            }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors.
        
        Returns:
            Dict containing error summary information
        """
        error_counts = {
            error_type.value: 0 for error_type in ErrorType
        }
        
        severity_counts = {
            severity.value: 0 for severity in ErrorSeverity
        }
        
        task_errors = {}
        
        for error in self.errors:
            error_counts[error.error_type.value] += 1
            severity_counts[error.severity.value] += 1
            
            if error.task_id not in task_errors:
                task_errors[error.task_id] = []
            
            task_errors[error.task_id].append(error.to_dict())
        
        return {
            "total_errors": len(self.errors),
            "error_types": error_counts,
            "error_severities": severity_counts,
            "task_errors": task_errors,
            "has_fatal_errors": any(error.severity == ErrorSeverity.FATAL for error in self.errors)
        }
    
    def clear_errors(self) -> None:
        """Clear all errors."""
        self.errors = []


class TimeoutManager:
    """
    Manager for timeout handling in workflows.
    
    This class provides methods for managing timeouts and implementing
    adaptive timeout strategies based on execution history.
    """
    
    def __init__(self, default_timeout: float = 60.0):
        """
        Initialize the timeout manager.
        
        Args:
            default_timeout: Default timeout in seconds
        """
        self.default_timeout = default_timeout
        self.task_timeouts = {}
        self.task_execution_times = {}
    
    def get_timeout(self, task_id: str) -> float:
        """
        Get the timeout for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Timeout in seconds
        """
        return self.task_timeouts.get(task_id, self.default_timeout)
    
    def update_timeout(self, task_id: str, timeout: float) -> None:
        """
        Update the timeout for a task.
        
        Args:
            task_id: ID of the task
            timeout: New timeout in seconds
        """
        self.task_timeouts[task_id] = timeout
    
    def record_execution_time(self, task_id: str, execution_time: float) -> None:
        """
        Record the execution time for a task.
        
        Args:
            task_id: ID of the task
            execution_time: Execution time in seconds
        """
        if task_id not in self.task_execution_times:
            self.task_execution_times[task_id] = []
        
        self.task_execution_times[task_id].append(execution_time)
    
    def adapt_timeout(self, task_id: str) -> float:
        """
        Adapt the timeout for a task based on execution history.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Adapted timeout in seconds
        """
        # If we don't have execution times for this task, use the default timeout
        if task_id not in self.task_execution_times or not self.task_execution_times[task_id]:
            return self.get_timeout(task_id)
        
        # Calculate the mean and standard deviation of execution times
        execution_times = self.task_execution_times[task_id]
        mean_time = sum(execution_times) / len(execution_times)
        std_dev = (sum((t - mean_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5
        
        # Set timeout to mean + 2 * std_dev (95% confidence interval)
        # But never less than 1.5 * mean
        timeout = max(mean_time + 2 * std_dev, 1.5 * mean_time)
        
        # Update the timeout
        self.update_timeout(task_id, timeout)
        
        return timeout
    
    def increase_timeout(self, task_id: str, multiplier: float = 1.5) -> float:
        """
        Increase the timeout for a task.
        
        Args:
            task_id: ID of the task
            multiplier: Multiplier for the timeout
            
        Returns:
            New timeout in seconds
        """
        current_timeout = self.get_timeout(task_id)
        new_timeout = current_timeout * multiplier
        self.update_timeout(task_id, new_timeout)
        
        return new_timeout
    
    def get_timeout_stats(self) -> Dict[str, Any]:
        """
        Get statistics about timeouts and execution times.
        
        Returns:
            Dict containing timeout statistics
        """
        stats = {}
        
        for task_id, execution_times in self.task_execution_times.items():
            if not execution_times:
                continue
                
            mean_time = sum(execution_times) / len(execution_times)
            std_dev = (sum((t - mean_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5
            
            stats[task_id] = {
                "mean_execution_time": mean_time,
                "std_dev": std_dev,
                "timeout": self.get_timeout(task_id),
                "execution_count": len(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times)
            }
        
        return stats 