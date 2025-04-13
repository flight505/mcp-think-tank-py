#!/usr/bin/env python3
"""
Code Tools for MCP Think Tank
Tools for searching and summarizing code in the codebase
"""
import logging
from typing import Dict, List, Any

from ..watchers.file_watcher import FileWatcher

logger = logging.getLogger("mcp-think-tank.code_tools")

class CodeTools:
    """
    Tools for searching and summarizing code in the codebase
    """
    
    def __init__(self, file_watcher: FileWatcher):
        """
        Initialize the code tools
        
        Args:
            file_watcher: FileWatcher instance to use for file operations
        """
        self.file_watcher = file_watcher
        logger.info("CodeTools initialized")
    
    def search_code(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for code in the codebase
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            results = self.file_watcher.search_code(query, limit=limit)
            
            # Format the results for display
            formatted_results = []
            for result in results:
                file_info = {
                    "file_path": result["file_path"],
                    "matches": []
                }
                
                for match in result["matches"]:
                    file_info["matches"].append({
                        "line_number": match["line_number"],
                        "content": match["content"],
                        "context": match["context"]
                    })
                
                formatted_results.append(file_info)
            
            return {
                "query": query,
                "result_count": len(formatted_results),
                "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Error searching code: {e}")
            return {
                "error": f"Error searching code: {str(e)}",
                "query": query,
                "result_count": 0,
                "results": []
            }
    
    def summarize_file(self, file_path: str) -> Dict[str, Any]:
        """
        Generate a detailed summary of a specific file
        
        Args:
            file_path: Path to the file to summarize (relative to project root)
            
        Returns:
            Dictionary with file summary information
        """
        try:
            # Get summary from file watcher
            summary = self.file_watcher.summarize_file(file_path)
            
            if "error" in summary:
                return summary
            
            # Return formatted summary
            return {
                "file_path": summary["file_path"],
                "metadata": summary["metadata"],
                "summary": summary["summary"],
                "structure": summary["structure"]
            }
            
        except Exception as e:
            logger.error(f"Error summarizing file: {e}")
            return {
                "error": f"Error summarizing file: {str(e)}",
                "file_path": file_path
            }
    
    def get_context_for_tool(self, queries: List[str], max_results: int = 3) -> Dict[str, Any]:
        """
        Get context from the codebase based on queries for auto-context injection
        
        Args:
            queries: List of search queries
            max_results: Maximum number of results to return per query
            
        Returns:
            Dictionary with context information
        """
        try:
            all_results = []
            
            for query in queries:
                results = self.file_watcher.search_code(query, limit=max_results)
                all_results.extend(results)
            
            # Deduplicate results by file path
            seen_files = set()
            unique_results = []
            
            for result in all_results:
                if result["file_path"] not in seen_files:
                    seen_files.add(result["file_path"])
                    unique_results.append(result)
            
            # Format for context injection
            context_blocks = []
            
            for result in unique_results[:max_results]:  # Limit total results
                file_path = result["file_path"]
                
                # Get the first match with context
                match = result["matches"][0] if result["matches"] else None
                
                if match:
                    context = "\n".join(match["context"])
                    context_blocks.append(f"From file {file_path} (line {match['line_number']}):\n```\n{context}\n```")
            
            return {
                "context_blocks": context_blocks,
                "total_items": len(context_blocks)
            }
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return {
                "error": f"Error getting context: {str(e)}",
                "context_blocks": [],
                "total_items": 0
            }
    
    def register_tools(self, mcp_server) -> None:
        """
        Register code tools with the MCP server
        
        Args:
            mcp_server: MCP server instance
        """
        # Register search_code tool
        mcp_server.register_tool(
            name="search_code",
            fn=self.search_code,
            description="Search for code in the codebase",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find code snippets"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        )
        
        # Register summarize_file tool
        mcp_server.register_tool(
            name="summarize_file",
            fn=self.summarize_file,
            description="Generate a detailed summary of a specific file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to summarize (relative to project root)"
                    }
                },
                "required": ["file_path"]
            }
        ) 