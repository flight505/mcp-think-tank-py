#!/usr/bin/env python3
"""
Test script to verify the MCP Think Tank server can start without errors.
"""
import asyncio
import os
import sys
import logging
import signal
import time
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

async def run_server_test(timeout=5):
    """
    Start the MCP server, let it run for a specified time, then shut it down.
    
    Args:
        timeout: Number of seconds to run the server before shutting down
    """
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
        # Let the server run for the specified time
        logger.info(f"Server started, running for {timeout} seconds...")
        await asyncio.sleep(timeout)
        logger.info("Server ran successfully for the test period")
        return True
    except Exception as e:
        logger.error(f"Error during server test: {e}")
        return False
    finally:
        # Stop the server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        logger.info("Server test completed and server stopped")

def main():
    """Run the server test."""
    try:
        success = asyncio.run(run_server_test())
        if success:
            logger.info("✅ MCP Think Tank server test PASSED")
            sys.exit(0)
        else:
            logger.error("❌ MCP Think Tank server test FAILED")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error in test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 