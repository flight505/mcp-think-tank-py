#!/usr/bin/env python3
"""
Unit tests for the CircuitBreaker implementation
"""
import unittest
import asyncio
import time
from unittest.mock import patch, MagicMock

from src.tools.circuit_breaker import (
    CircuitBreaker, 
    CircuitState, 
    CircuitOpenError,
    CircuitBreakerManager
)

class TestCircuitBreaker(unittest.TestCase):
    """Test suite for the CircuitBreaker class."""
    
    def setUp(self):
        """Set up test fixtures, if any."""
        self.circuit = CircuitBreaker(
            name="test-circuit",
            failure_threshold=3,
            recovery_timeout=0.5,  # Short timeout for testing
            half_open_max_calls=2
        )
    
    def test_initial_state(self):
        """Test the initial state of the circuit breaker."""
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.failures, 0)
        self.assertEqual(self.circuit.successes, 0)
    
    def test_successful_execution(self):
        """Test successful execution doesn't change state."""
        def success_func():
            return "Success"
        
        # Execute the function
        result = self.circuit.execute(success_func)
        
        # Verify the result and state
        self.assertEqual(result, "Success")
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.failures, 0)
    
    def test_failure_threshold(self):
        """Test that reaching failure threshold opens the circuit."""
        def failing_func():
            raise ValueError("Test error")
        
        # Execute the function and catch exceptions
        for _ in range(3):
            with self.assertRaises(ValueError):
                self.circuit.execute(failing_func)
        
        # Verify the circuit is now open
        self.assertEqual(self.circuit.state, CircuitState.OPEN)
        self.assertEqual(self.circuit.failures, 3)
    
    def test_circuit_open_error(self):
        """Test that an open circuit fast-fails requests."""
        def success_func():
            return "Success"
        
        # Force the circuit to open
        self.circuit.state = CircuitState.OPEN
        
        # Try to execute the function
        with self.assertRaises(CircuitOpenError):
            self.circuit.execute(success_func)
    
    def test_recovery_timeout(self):
        """Test that an open circuit transitions to half-open after timeout."""
        def success_func():
            return "Success"
        
        # Force the circuit to open
        self.circuit.state = CircuitState.OPEN
        self.circuit.last_state_change_time = time.time() - 1.0  # Set time in the past
        
        # Execute the function
        result = self.circuit.execute(success_func)
        
        # Verify the circuit is now half-open and the function was called
        self.assertEqual(self.circuit.state, CircuitState.HALF_OPEN)
        self.assertEqual(result, "Success")
    
    def test_half_open_success_recovery(self):
        """Test that successful calls in half-open state close the circuit."""
        def success_func():
            return "Success"
        
        # Force the circuit to half-open
        self.circuit.state = CircuitState.HALF_OPEN
        
        # Execute the function twice (success threshold is 2)
        self.circuit.execute(success_func)
        self.circuit.execute(success_func)
        
        # Verify the circuit is now closed
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.failures, 0)
    
    def test_half_open_failure(self):
        """Test that failures in half-open state reopen the circuit."""
        def failing_func():
            raise ValueError("Test error")
        
        # Force the circuit to half-open
        self.circuit.state = CircuitState.HALF_OPEN
        
        # Execute the function and catch the exception
        with self.assertRaises(ValueError):
            self.circuit.execute(failing_func)
        
        # Verify the circuit is now open again
        self.assertEqual(self.circuit.state, CircuitState.OPEN)
    
    def test_reset(self):
        """Test that reset puts the circuit back to initial state."""
        # Force the circuit to open with failures
        self.circuit.state = CircuitState.OPEN
        self.circuit.failures = 5
        
        # Reset the circuit
        self.circuit.reset()
        
        # Verify the circuit is now closed with no failures
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.failures, 0)
    
    def test_excluded_exceptions(self):
        """Test that excluded exceptions don't count as failures."""
        # Create a circuit with KeyError excluded
        circuit = CircuitBreaker(
            name="test-excluded",
            failure_threshold=3,
            excluded_exceptions=[KeyError]
        )
        
        def key_error_func():
            raise KeyError("Test excluded error")
        
        # Execute the function multiple times and catch exceptions
        for _ in range(5):
            with self.assertRaises(KeyError):
                circuit.execute(key_error_func)
        
        # Verify the circuit is still closed
        self.assertEqual(circuit.state, CircuitState.CLOSED)
        self.assertEqual(circuit.failures, 0)


class TestCircuitBreakerAsync(unittest.TestCase):
    """Test suite for the async features of CircuitBreaker."""
    
    def setUp(self):
        """Set up test fixtures, if any."""
        self.circuit = CircuitBreaker(
            name="test-async-circuit",
            failure_threshold=3,
            recovery_timeout=0.5,
            half_open_max_calls=2
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.loop.close()
    
    def test_async_execution(self):
        """Test async function execution."""
        async def success_func():
            return "Async Success"
        
        # Execute the async function
        result = self.loop.run_until_complete(
            self.circuit.execute_async(success_func)
        )
        
        # Verify the result
        self.assertEqual(result, "Async Success")
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
    
    def test_async_failure(self):
        """Test async function failure."""
        async def failing_func():
            raise ValueError("Async error")
        
        # Execute the async function and catch exceptions
        for _ in range(3):
            with self.assertRaises(ValueError):
                self.loop.run_until_complete(
                    self.circuit.execute_async(failing_func)
                )
        
        # Verify the circuit is now open
        self.assertEqual(self.circuit.state, CircuitState.OPEN)
        self.assertEqual(self.circuit.failures, 3)
    
    def test_async_circuit_open(self):
        """Test that an open circuit fast-fails async requests."""
        async def success_func():
            return "Async Success"
        
        # Force the circuit to open
        self.circuit.state = CircuitState.OPEN
        
        # Try to execute the async function
        with self.assertRaises(CircuitOpenError):
            self.loop.run_until_complete(
                self.circuit.execute_async(success_func)
            )


class TestCircuitBreakerManager(unittest.TestCase):
    """Test suite for the CircuitBreakerManager class."""
    
    def setUp(self):
        """Set up test fixtures, if any."""
        self.manager = CircuitBreakerManager()
    
    def test_get_circuit_breaker(self):
        """Test getting a circuit breaker by name."""
        # Get a circuit breaker
        circuit = self.manager.get_circuit_breaker("test-service")
        
        # Verify the circuit was created
        self.assertIsInstance(circuit, CircuitBreaker)
        self.assertEqual(circuit.name, "test-service")
        
        # Get the same circuit breaker again
        circuit2 = self.manager.get_circuit_breaker("test-service")
        
        # Verify it's the same instance
        self.assertIs(circuit, circuit2)
    
    def test_get_nonexistent_circuit(self):
        """Test getting a nonexistent circuit breaker with create_if_missing=False."""
        # Try to get a nonexistent circuit with create_if_missing=False
        circuit = self.manager.get_circuit_breaker("nonexistent", create_if_missing=False)
        
        # Verify no circuit was returned
        self.assertIsNone(circuit)
    
    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        # Create a few circuit breakers and set them to open
        circuit1 = self.manager.get_circuit_breaker("service1")
        circuit2 = self.manager.get_circuit_breaker("service2")
        
        circuit1.state = CircuitState.OPEN
        circuit2.state = CircuitState.OPEN
        
        # Reset all circuit breakers
        self.manager.reset_all()
        
        # Verify all circuits are now closed
        self.assertEqual(circuit1.state, CircuitState.CLOSED)
        self.assertEqual(circuit2.state, CircuitState.CLOSED)
    
    def test_get_all_health(self):
        """Test getting health information for all circuit breakers."""
        # Create a few circuit breakers
        self.manager.get_circuit_breaker("service1")
        self.manager.get_circuit_breaker("service2")
        
        # Get health information
        health = self.manager.get_all_health()
        
        # Verify health information
        self.assertIn("service1", health)
        self.assertIn("service2", health)
        self.assertEqual(health["service1"]["state"], "closed")
        self.assertEqual(health["service2"]["state"], "closed")


class TestCircuitBreakerDecorator(unittest.TestCase):
    """Test suite for the circuit breaker decorator."""
    
    def setUp(self):
        """Set up test fixtures, if any."""
        self.manager = CircuitBreakerManager()
    
    def test_decorator_sync(self):
        """Test the circuit breaker decorator with a sync function."""
        from src.tools.circuit_breaker import with_circuit_breaker
        
        # Define a sync function with the decorator
        @with_circuit_breaker("test-decorator", self.manager)
        def test_func():
            return "Decorator Success"
        
        # Execute the function
        result = test_func()
        
        # Verify the result
        self.assertEqual(result, "Decorator Success")
        
        # Verify the circuit breaker was created
        circuit = self.manager.get_circuit_breaker("test-decorator", create_if_missing=False)
        self.assertIsNotNone(circuit)
    
    def test_decorator_async(self):
        """Test the circuit breaker decorator with an async function."""
        from src.tools.circuit_breaker import with_circuit_breaker
        
        # Define an async function with the decorator
        @with_circuit_breaker("test-async-decorator", self.manager)
        async def test_async_func():
            return "Async Decorator Success"
        
        # Execute the function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(test_async_func())
        finally:
            loop.close()
        
        # Verify the result
        self.assertEqual(result, "Async Decorator Success")


@patch('src.orchestrator.CircuitBreakerManager')
@patch('src.orchestrator.TimeoutManager')
@patch('src.orchestrator.WorkflowErrorHandler')
class TestOrchestratorCircuitBreaker(unittest.TestCase):
    """Test suite for the orchestrator's circuit breaker integration."""
    
    def test_orchestrator_init(self, mock_error_handler, mock_timeout_manager, mock_circuit_manager):
        """Test that the orchestrator initializes the circuit manager."""
        from src.orchestrator import Orchestrator
        
        # Create a mock MCP server
        mock_server = MagicMock()
        
        # Configure mocks to avoid duplicate instantiation issue
        mock_instance = MagicMock()
        mock_circuit_manager.return_value = mock_instance
        
        # Create an orchestrator but don't call _init_error_handling manually
        # since it's already called in the constructor
        orchestrator = Orchestrator(mcp=mock_server)
        
        # Verify the circuit manager was created
        mock_circuit_manager.assert_called_once()
    
    def test_execute_with_circuit_breaker(self, mock_error_handler, mock_timeout_manager, mock_circuit_manager):
        """Test that _execute_with_timeout_and_recovery uses the circuit breaker."""
        from src.orchestrator import Orchestrator
        
        # Set up mock circuit manager and circuit breaker
        mock_circuit = MagicMock()
        mock_circuit.execute.return_value = "Circuit Success"
        mock_circuit_manager_instance = MagicMock()
        mock_circuit_manager_instance.get_circuit_breaker.return_value = mock_circuit
        mock_circuit_manager.return_value = mock_circuit_manager_instance
        
        # Create a mock timeout manager that returns a proper timeout value instead of a MagicMock
        mock_timeout_instance = MagicMock()
        mock_timeout_instance.get_timeout.return_value = 30.0  # Return an actual number
        mock_timeout_manager.return_value = mock_timeout_instance
        
        # Create a mock MCP server
        mock_server = MagicMock()
        
        # Create an orchestrator
        orchestrator = Orchestrator(mcp=mock_server)
        
        # Create a test function
        def test_func():
            return "Direct Success"
        
        # Execute the function with timeout and recovery
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                orchestrator._execute_with_timeout_and_recovery(
                    func=test_func,
                    task_id="test.func"
                )
            )
        finally:
            loop.close()
        
        # Verify the result and that the circuit breaker was used
        self.assertEqual(result["status"], "success")
        mock_circuit_manager_instance.get_circuit_breaker.assert_called_with(
            "test", 
            failure_threshold=5, 
            recovery_timeout=60.0, 
            half_open_max_calls=1
        )
        mock_circuit.execute.assert_called_once()


if __name__ == '__main__':
    unittest.main() 