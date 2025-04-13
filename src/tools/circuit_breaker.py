#!/usr/bin/env python3
"""
Circuit Breaker Implementation for MCP Think Tank
Prevents cascading failures by monitoring service health and short-circuiting calls to failing services
"""
import logging
import time
import asyncio
from enum import Enum
from typing import Dict, Any, Callable, Awaitable, Union, Optional, List
from functools import wraps
import traceback
import threading

logger = logging.getLogger("mcp-think-tank.circuit_breaker")

class CircuitState(Enum):
    """Enum representing the states of a circuit breaker."""
    CLOSED = "closed"       # Normal operation, requests pass through
    OPEN = "open"           # Circuit is tripped, fast-fail for all requests
    HALF_OPEN = "half_open" # Recovery state, allowing test requests through


class CircuitBreaker:
    """
    Circuit Breaker implementation that prevents cascading failures.
    
    This class implements the Circuit Breaker pattern, which monitors
    the health of services and prevents calls to failing services by
    fast-failing requests when the circuit is open.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        window_size: int = 10,
        excluded_exceptions: List[type] = None
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker (usually the service name)
            failure_threshold: Number of failures needed to trip the circuit
            recovery_timeout: Time in seconds to wait before trying recovery
            half_open_max_calls: Maximum number of calls to allow in half-open state
            window_size: Size of the rolling window for failure counting
            excluded_exceptions: List of exception types that should not count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.window_size = window_size
        self.excluded_exceptions = excluded_exceptions or []
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time = 0
        self.last_state_change_time = time.time()
        self.half_open_calls = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Rolling window of results
        self.results_window = []
        
        logger.info(f"Circuit breaker '{name}' initialized in CLOSED state")
    
    def _should_count_exception(self, exception: Exception) -> bool:
        """
        Determine if an exception should count toward the failure threshold.
        
        Args:
            exception: The exception to check
            
        Returns:
            True if the exception should count as a failure
        """
        for exc_type in self.excluded_exceptions:
            if isinstance(exception, exc_type):
                return False
        return True
    
    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            self.results_window = (self.results_window + [True])[-self.window_size:]
            
            if self.state == CircuitState.HALF_OPEN:
                self.successes += 1
                if self.successes >= self.half_open_max_calls:
                    self._transition_to_closed()
    
    def _record_failure(self, exception: Exception):
        """
        Record a failed call.
        
        Args:
            exception: The exception that occurred
        """
        with self._lock:
            if not self._should_count_exception(exception):
                logger.debug(f"Exception {type(exception)} is excluded, not counting as failure")
                return
                
            self.results_window = (self.results_window + [False])[-self.window_size:]
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                self.failures += 1
                if self.failures >= self.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition the circuit to the OPEN state."""
        with self._lock:
            if self.state != CircuitState.OPEN:
                logger.warning(f"Circuit breaker '{self.name}' transitioning to OPEN state")
                self.state = CircuitState.OPEN
                self.last_state_change_time = time.time()
    
    def _transition_to_half_open(self):
        """Transition the circuit to the HALF_OPEN state."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change_time = time.time()
                self.half_open_calls = 0
                self.successes = 0
    
    def _transition_to_closed(self):
        """Transition the circuit to the CLOSED state."""
        with self._lock:
            if self.state != CircuitState.CLOSED:
                logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED state")
                self.state = CircuitState.CLOSED
                self.last_state_change_time = time.time()
                self.failures = 0
    
    def _should_allow_request(self) -> bool:
        """
        Determine if a request should be allowed through.
        
        Returns:
            True if the request should be allowed
        """
        with self._lock:
            # If the circuit is closed, always allow the request
            if self.state == CircuitState.CLOSED:
                return True
            
            # If the circuit is open, check if the recovery timeout has elapsed
            if self.state == CircuitState.OPEN:
                elapsed = time.time() - self.last_state_change_time
                if elapsed >= self.recovery_timeout:
                    self._transition_to_half_open()
                else:
                    # Still in recovery period, fast-fail
                    return False
            
            # If we're half-open, allow a limited number of test requests
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return True
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get the health status of the circuit breaker.
        
        Returns:
            Dict containing health information
        """
        with self._lock:
            health = {
                "name": self.name,
                "state": self.state.value,
                "failures": self.failures,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self.last_failure_time,
                "last_state_change_time": self.last_state_change_time,
                "window_size": self.window_size,
                "current_window": self.results_window
            }
            
            # Add state-specific information
            if self.state == CircuitState.OPEN:
                elapsed = time.time() - self.last_state_change_time
                health["time_to_recovery"] = max(0, self.recovery_timeout - elapsed)
                
            elif self.state == CircuitState.HALF_OPEN:
                health["half_open_calls"] = self.half_open_calls
                health["successes"] = self.successes
                health["required_successes"] = self.half_open_max_calls
            
            return health
    
    def reset(self):
        """Reset the circuit breaker to its initial state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failures = 0
            self.successes = 0
            self.last_failure_time = 0
            self.last_state_change_time = time.time()
            self.half_open_calls = 0
            self.results_window = []
            logger.info(f"Circuit breaker '{self.name}' has been reset")
    
    def execute(self, func, *args, **kwargs):
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: The function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
            
        Raises:
            CircuitOpenError: If the circuit is open
            Any exception raised by the function
        """
        if not self._should_allow_request():
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    async def execute_async(self, func, *args, **kwargs):
        """
        Execute an async function with circuit breaker protection.
        
        Args:
            func: The async function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
            
        Raises:
            CircuitOpenError: If the circuit is open
            Any exception raised by the function
        """
        if not self._should_allow_request():
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise


class CircuitOpenError(Exception):
    """Exception raised when a circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """
    Manager for circuit breakers in the system.
    
    This class provides methods for creating, retrieving, and managing
    circuit breakers for different services.
    """
    
    def __init__(self):
        """Initialize the circuit breaker manager."""
        self.circuit_breakers = {}
        self._lock = threading.RLock()
    
    def get_circuit_breaker(self, name: str, create_if_missing: bool = True, **kwargs) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name, optionally creating it if missing.
        
        Args:
            name: Name of the circuit breaker
            create_if_missing: Whether to create the circuit breaker if it doesn't exist
            **kwargs: Arguments to pass to the circuit breaker constructor if creating
            
        Returns:
            The circuit breaker, or None if not found and create_if_missing is False
        """
        with self._lock:
            if name in self.circuit_breakers:
                return self.circuit_breakers[name]
                
            if create_if_missing:
                circuit = CircuitBreaker(name, **kwargs)
                self.circuit_breakers[name] = circuit
                return circuit
                
            return None
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for circuit in self.circuit_breakers.values():
                circuit.reset()
    
    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health information for all circuit breakers.
        
        Returns:
            Dict mapping circuit breaker names to health information
        """
        with self._lock:
            return {name: circuit.get_health() for name, circuit in self.circuit_breakers.items()}


def with_circuit_breaker(
    circuit_breaker: Union[str, CircuitBreaker], 
    manager: Optional[CircuitBreakerManager] = None,
    **circuit_kwargs
):
    """
    Decorator to wrap a function with circuit breaker protection.
    
    Args:
        circuit_breaker: Circuit breaker to use, or name to get from manager
        manager: Circuit breaker manager (required if circuit_breaker is a name)
        **circuit_kwargs: Arguments to pass to circuit breaker constructor if creating
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the circuit breaker
            if isinstance(circuit_breaker, str):
                if not manager:
                    raise ValueError("Manager is required when circuit_breaker is a name")
                cb = manager.get_circuit_breaker(circuit_breaker, True, **circuit_kwargs)
            else:
                cb = circuit_breaker
            
            # Execute with circuit breaker protection
            return cb.execute(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get the circuit breaker
            if isinstance(circuit_breaker, str):
                if not manager:
                    raise ValueError("Manager is required when circuit_breaker is a name")
                cb = manager.get_circuit_breaker(circuit_breaker, True, **circuit_kwargs)
            else:
                cb = circuit_breaker
            
            # Execute with circuit breaker protection
            return await cb.execute_async(func, *args, **kwargs)
        
        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator 