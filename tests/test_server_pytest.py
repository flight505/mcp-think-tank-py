#!/usr/bin/env python3
"""
Test script to verify the MCP Think Tank server can start without errors.
This version is compatible with pytest.
"""
import asyncio
import os
import sys
import logging
import pytest
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.orchestrator import Orchestrator
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp-think-tank-test")


@pytest.mark.asyncio
async def test_server_startup():
    """Test that the server can start up successfully and shut down without errors."""
    logger.info("Starting MCP Think Tank server test...")
    
    # Create the MCP server
    mcp = FastMCP(name="MCP Think Tank Test 🧠")
    
    # Initialize orchestrator
    orchestrator = Orchestrator(mcp)
    
    # Register tools and resources via the orchestrator
    orchestrator.register_tools()
    
    # Ensure necessary directories exist
    config = get_config()
    os.makedirs(os.path.dirname(config.memory_file_path), exist_ok=True)
    
    # Create a task to run the server
    server_task = asyncio.create_task(mcp.run_async())
    
    try:
        # Let the server run for a short time
        timeout = 3  # Shorter timeout for tests
        logger.info(f"Server started, running for {timeout} seconds...")
        await asyncio.sleep(timeout)
        logger.info("Server ran successfully for the test period")
        assert True  # Server started successfully
    finally:
        # Stop the server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        logger.info("Server test completed and server stopped") 