# MCP Think Tank 🧠

<div align="center">
  <img src="public/assets/MCP_Think_Tank_light.png#gh-light-mode-only" width="300" alt="MCP Think Tank Logo Light"/>
  <img src="public/assets/MCP_Think_Tank_dark.png#gh-dark-mode-only" width="300" alt="MCP Think Tank Logo Dark"/>
</div>

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastMCP](https://img.shields.io/badge/FastMCP-v2-orange.svg)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/flight505/mcp-think-tank-py.svg)](https://github.com/flight505/mcp-think-tank-py/stargazers)

A next-generation knowledge graph and reasoning tool built with FastMCP V2. This is a complete Python rewrite of the [original TypeScript version](https://github.com/flight505/mcp-think-tank) with enhanced capabilities.

## Features

- **Advanced Knowledge Graph** - Store and retrieve information with semantic search capabilities
- **Structured Reasoning** - Think and reflect on complex problems with chain-of-thought
- **Task Management** - Plan and organize multi-step workflows
- **Cross-Tool Intelligence** - Orchestrated interactions between memory, reasoning, and tasks
- **File System Awareness** - Index and search code in your projects

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Install with uv

```bash
# Clone the repository
git clone https://github.com/flight505/mcp-think-tank-py.git
cd mcp-think-tank-py

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Install with Claude Desktop

```bash
# Install directly for Claude Desktop
fastmcp install src/server.py --name "MCP Think Tank"
```

## Quick Start

1. **Run in development mode**

```bash
fastmcp dev src/server.py
```

2. **Use with Claude Desktop**

After installing with `fastmcp install`, the MCP Think Tank will be available as a tool in Claude Desktop.

## Usage

### Knowledge Graph

```
# Create entities
create_entities(entities=[{"name": "FastMCP", "entityType": "Technology", "observations": ["A Pythonic MCP framework"]}])

# Create relations
create_relations(relations=[{"from": "FastMCP", "to": "Python", "relationType": "built with"}])

# Search for entities
search_nodes(query="FastMCP")
```

### Structured Reasoning

```
# Think about a problem
think(structured_reasoning="Let me analyze how to approach this problem step by step...", store_in_memory=True)
```

### Task Management

```
# Create tasks from requirements
create_tasks(prd_text="Build a feature that integrates with our database and provides a REST API")

# List current tasks
list_tasks()
```

## Architecture

MCP Think Tank is built with a modular architecture:

- **Core Server** - FastMCP-based server implementation
- **Orchestrator** - Coordinates interactions between tools
- **Tools** - Memory, Think, and Task functionality
- **Storage** - JSONL-based persistence with semantic search

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FastMCP](https://github.com/jlowin/fastmcp) for the powerful MCP server framework
- [Model Context Protocol](https://modelcontextprotocol.io/) for standardizing LLM context provision
