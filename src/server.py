#!/usr/bin/env python3
"""
MCP Think Tank Server - Python Implementation
"""
import asyncio
import os
import logging
from typing import Dict, List

from fastmcp import FastMCP

# Import local modules
from .config import get_config
from .orchestrator import Orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=os.path.expanduser("~/.mcp-think-tank/mcp-think-tank.log"),
    filemode="a",
)
logger = logging.getLogger("mcp-think-tank")

# Create the MCP server
mcp = FastMCP(name="MCP Think Tank 🧠")

# Initialize orchestrator
orchestrator = Orchestrator(mcp)

# Register tools and resources via the orchestrator
orchestrator.register_tools()

# Main entry point
if __name__ == "__main__":
    # Ensure necessary directories exist
    config = get_config()
    os.makedirs(os.path.dirname(config.memory_file_path), exist_ok=True)
    
    # Run the server
    mcp.run() 