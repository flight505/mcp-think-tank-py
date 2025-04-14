#!/usr/bin/env python3
"""
File Watcher for MCP Think Tank
Indexes and tracks changes in code files
"""
import os
import fnmatch
import logging
import time
import hashlib
import threading
from typing import Dict, List, Any, Set, Optional, Tuple, Callable
from datetime import datetime

logger = logging.getLogger("mcp-think-tank.file_watcher")

class FileWatcher:
    """
    Watches and indexes files in a project directory
    
    The FileWatcher recursively scans directories, identifies code files,
    and tracks changes to keep the knowledge graph updated with the latest
    file information.
    """
    
    def __init__(
        self,
        project_path: str,
        knowledge_graph,
        file_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        polling_interval: int = 10,
        start_watching: bool = True
    ):
        """
        Initialize the file watcher
        
        Args:
            project_path: Path to the project directory
            knowledge_graph: KnowledgeGraph instance
            file_patterns: List of glob patterns to include (e.g., ["*.py", "*.js"])
            exclude_patterns: List of glob patterns to exclude (e.g., ["**/node_modules/**"])
            polling_interval: How often to check for changes (in seconds)
            start_watching: Whether to start watching files immediately
        """
        self.project_path = os.path.abspath(project_path)
        self.knowledge_graph = knowledge_graph
        self.file_patterns = file_patterns or ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.md"]
        self.exclude_patterns = exclude_patterns or ["**/node_modules/**", "**/.git/**", "**/venv/**", "**/__pycache__/**", "**/.venv/**"]
        self.polling_interval = polling_interval
        
        # File tracking
        self.file_hashes = {}  # path -> hash
        self.file_metadata = {}  # path -> metadata
        
        # Thread control
        self.watching = False
        self.watch_thread = None
        
        # Initialize file index
        self._index_files()
        
        if start_watching:
            self.start()
    
    def start(self):
        """Start the file watching thread"""
        if self.watching:
            logger.warning("File watcher is already running")
            return
        
        self.watching = True
        self.watch_thread = threading.Thread(target=self._watch_files, daemon=True)
        self.watch_thread.start()
        logger.info(f"Started file watcher for {self.project_path}")
    
    def stop(self):
        """Stop the file watching thread"""
        if not self.watching:
            logger.warning("File watcher is not running")
            return
        
        self.watching = False
        if self.watch_thread:
            self.watch_thread.join(timeout=2.0)  # Wait up to 2 seconds
            logger.info("Stopped file watcher")
    
    def _watch_files(self):
        """Background thread that polls for file changes"""
        while self.watching:
            try:
                self._check_for_changes()
                time.sleep(self.polling_interval)
            except Exception as e:
                logger.error(f"Error in file watcher: {e}")
                time.sleep(max(1, self.polling_interval // 2))  # Backoff on errors
    
    def _index_files(self, specific_path: Optional[str] = None, patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Index files recursively from the project path or a specific directory
        
        Args:
            specific_path: Optional specific directory to index
            patterns: Optional override for file patterns
            
        Returns:
            Dictionary with indexing results
        """
        start_time = time.time()
        base_path = specific_path or self.project_path
        patterns_to_use = patterns or self.file_patterns
        
        if not os.path.exists(base_path):
            logger.error(f"Path does not exist: {base_path}")
            return {"error": f"Path not found: {base_path}", "files_indexed": 0}
        
        # Track new files and changes
        indexed_count = 0
        new_files = []
        changed_files = []
        
        # Walk the directory tree
        for root, _, files in os.walk(base_path):
            # Check if directory should be excluded
            rel_dir = os.path.relpath(root, self.project_path)
            if self._should_exclude(rel_dir):
                continue
                
            # Process files in directory
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.project_path)
                
                # Skip if path or file should be excluded
                if self._should_exclude(rel_path):
                    continue
                
                # Check if file matches patterns
                if not self._matches_patterns(file, patterns_to_use):
                    continue
                
                # Process file
                try:
                    metadata = self._extract_metadata(file_path)
                    file_hash = metadata.get("hash", "")
                    
                    if rel_path not in self.file_hashes:
                        # New file
                        self.file_hashes[rel_path] = file_hash
                        self.file_metadata[rel_path] = metadata
                        self._store_in_knowledge_graph(rel_path, metadata)
                        new_files.append(rel_path)
                    elif self.file_hashes[rel_path] != file_hash:
                        # Changed file
                        self.file_hashes[rel_path] = file_hash
                        self.file_metadata[rel_path] = metadata
                        self._store_in_knowledge_graph(rel_path, metadata, update=True)
                        changed_files.append(rel_path)
                    
                    indexed_count += 1
                except Exception as e:
                    logger.error(f"Error indexing file {rel_path}: {e}")
        
        # Check for deleted files
        if specific_path is None:  # Only check for deletions in full scans
            all_paths = set(self.file_hashes.keys())
            current_paths = set()
            
            for root, _, files in os.walk(self.project_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_path)
                    current_paths.add(rel_path)
            
            deleted_paths = all_paths - current_paths
            for path in deleted_paths:
                self._remove_file(path)
        
        elapsed = time.time() - start_time
        logger.info(f"Indexed {indexed_count} files in {elapsed:.2f} seconds")
        
        return {
            "files_indexed": indexed_count,
            "new_files": new_files,
            "changed_files": changed_files,
            "elapsed_seconds": elapsed
        }
    
    def _check_for_changes(self):
        """Check for file changes and update the index"""
        try:
            # Keep track of all current files
            current_files = set()
            
            # Track changes
            modified_files = []
            
            # Walk the directory tree
            for root, _, files in os.walk(self.project_path):
                # Check if directory should be excluded
                rel_dir = os.path.relpath(root, self.project_path)
                if self._should_exclude(rel_dir):
                    continue
                
                # Process files in directory
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_path)
                    
                    # Skip if path or file should be excluded
                    if self._should_exclude(rel_path):
                        continue
                    
                    # Check if file matches patterns
                    if not self._matches_patterns(file, self.file_patterns):
                        continue
                    
                    # Add to current files
                    current_files.add(rel_path)
                    
                    # Check if file is new or modified
                    try:
                        file_stat = os.stat(file_path)
                        file_mtime = file_stat.st_mtime
                        
                        if rel_path not in self.file_hashes:
                            # New file
                            self._index_file(file_path)
                        else:
                            metadata = self.file_metadata.get(rel_path, {})
                            last_modified = metadata.get("last_modified", 0)
                            
                            # Convert ISO format string to timestamp if it's a string
                            if isinstance(last_modified, str):
                                try:
                                    # Parse ISO format string to datetime then to timestamp
                                    last_modified_dt = datetime.fromisoformat(last_modified)
                                    last_modified = last_modified_dt.timestamp()
                                except (ValueError, TypeError):
                                    # If parsing fails, set to 0 to force update
                                    last_modified = 0
                            
                            if file_mtime > last_modified:
                                # Modified file
                                self._index_file(file_path)
                                modified_files.append(rel_path)
                    except Exception as e:
                        logger.error(f"Error checking file {rel_path}: {e}")
            
            # Check for deleted files
            deleted_files = set(self.file_hashes.keys()) - current_files
            for rel_path in deleted_files:
                self._remove_file(rel_path)
            
            if modified_files or deleted_files:
                logger.info(f"Changes detected: {len(modified_files)} modified, {len(deleted_files)} deleted")
                
        except Exception as e:
            logger.error(f"Error checking for changes: {e}")
    
    def _index_file(self, file_path: str) -> Dict[str, Any]:
        """
        Index a single file
        
        Args:
            file_path: Path to the file
            
        Returns:
            File metadata
        """
        try:
            rel_path = os.path.relpath(file_path, self.project_path)
            metadata = self._extract_metadata(file_path)
            file_hash = metadata.get("hash", "")
            
            # Update tracking
            self.file_hashes[rel_path] = file_hash
            self.file_metadata[rel_path] = metadata
            
            # Store in knowledge graph
            is_update = rel_path in self.file_hashes
            self._store_in_knowledge_graph(rel_path, metadata, update=is_update)
            
            return metadata
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return {"error": str(e)}
    
    def _remove_file(self, rel_path: str):
        """
        Handle a file that has been deleted
        
        Args:
            rel_path: Relative path to the file
        """
        try:
            # Remove from tracking
            self.file_hashes.pop(rel_path, None)
            self.file_metadata.pop(rel_path, None)
            
            # Remove from knowledge graph
            file_entity_name = f"File:{rel_path}"
            self.knowledge_graph.delete_entities([file_entity_name])
            
            logger.info(f"Removed file from index: {rel_path}")
        except Exception as e:
            logger.error(f"Error removing file {rel_path}: {e}")
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        try:
            # Basic file info
            file_stat = os.stat(file_path)
            rel_path = os.path.relpath(file_path, self.project_path)
            
            # Get file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lstrip(".")
            
            # Read file content
            with open(file_path, "rb") as f:
                content = f.read()
                content_hash = hashlib.md5(content).hexdigest()
            
            # Try to decode as text
            try:
                text_content = content.decode("utf-8")
                line_count = len(text_content.splitlines())
            except UnicodeDecodeError:
                text_content = None
                line_count = 0
            
            # Convert timestamps to ISO format strings
            last_modified_time = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            created_time = datetime.fromtimestamp(file_stat.st_ctime).isoformat()
            
            # Basic metadata
            metadata = {
                "file_path": rel_path,
                "abs_path": file_path,
                "size_bytes": file_stat.st_size,
                "last_modified": last_modified_time,  # ISO format string
                "created": created_time,              # ISO format string
                "hash": content_hash,
                "extension": ext,
                "line_count": line_count,
                "is_binary": text_content is None,
            }
            
            # Extract language-specific metadata
            if not metadata["is_binary"]:
                if ext in ["py"]:
                    metadata.update(self._extract_python_metadata(text_content))
                elif ext in ["js", "ts", "jsx", "tsx"]:
                    metadata.update(self._extract_js_metadata(text_content))
                
                # Generate summary
                metadata["summary"] = self._generate_file_summary(rel_path, text_content, metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return {
                "file_path": os.path.relpath(file_path, self.project_path),
                "error": str(e),
                "hash": "",
                "last_modified": datetime.now().isoformat()  # Use ISO format string
            }
    
    def _extract_python_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract Python-specific metadata
        
        Args:
            content: File content as string
            
        Returns:
            Dictionary of Python metadata
        """
        import re
        
        classes = []
        functions = []
        imports = []
        
        # Extract classes
        class_matches = re.finditer(r"^class\s+([A-Za-z0-9_]+)(?:\(([^)]+)\))?:", content, re.MULTILINE)
        for match in class_matches:
            class_name = match.group(1)
            parent_class = match.group(2) or ""
            classes.append({"name": class_name, "parent": parent_class})
        
        # Extract functions
        func_matches = re.finditer(r"^def\s+([A-Za-z0-9_]+)\s*\(([^)]*)\):", content, re.MULTILINE)
        for match in func_matches:
            func_name = match.group(1)
            params = match.group(2)
            functions.append({"name": func_name, "params": params})
        
        # Extract imports
        import_matches = re.finditer(r"^(?:from\s+([A-Za-z0-9_.]+)\s+)?import\s+([^#\n]+)", content, re.MULTILINE)
        for match in import_matches:
            module = match.group(1) or ""
            imported = match.group(2).strip()
            imports.append({"module": module, "imported": imported})
        
        return {
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "language": "python"
        }
    
    def _extract_js_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract JavaScript/TypeScript metadata
        
        Args:
            content: File content as string
            
        Returns:
            Dictionary of JS/TS metadata
        """
        import re
        
        classes = []
        functions = []
        imports = []
        
        # Extract classes
        class_matches = re.finditer(r"class\s+([A-Za-z0-9_]+)(?:\s+extends\s+([A-Za-z0-9_]+))?", content)
        for match in class_matches:
            class_name = match.group(1)
            parent_class = match.group(2) or ""
            classes.append({"name": class_name, "parent": parent_class})
        
        # Extract functions (regular and arrow)
        func_matches = re.finditer(r"(?:function\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)|const\s+([A-Za-z0-9_]+)\s*=\s*(?:\([^)]*\)|[A-Za-z0-9_,\s]+)\s*=>)", content)
        for match in func_matches:
            if match.group(1):  # Regular function
                func_name = match.group(1)
                params = match.group(2)
            else:  # Arrow function
                func_name = match.group(3)
                params = ""
            functions.append({"name": func_name, "params": params})
        
        # Extract imports
        import_matches = re.finditer(r"import\s+(?:(\{[^}]+\})|([A-Za-z0-9_]+))\s+from\s+['\"]([@A-Za-z0-9_./\\-]+)['\"]", content)
        for match in import_matches:
            if match.group(1):  # Named imports
                imported = match.group(1)
            else:  # Default import
                imported = match.group(2)
            module = match.group(3)
            imports.append({"module": module, "imported": imported})
        
        return {
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "language": "javascript" if content.endswith(".js") or content.endswith(".jsx") else "typescript"
        }
    
    def _generate_file_summary(self, file_path: str, content: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a summary of the file
        
        Args:
            file_path: Relative path to the file
            content: File content as string
            metadata: File metadata
            
        Returns:
            Summary string
        """
        # Extract docstring or header comment if available
        import re
        
        summary_parts = []
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lstrip(".")
        
        # Try to extract docstring or header comment
        if ext == "py":
            # Python docstring
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1).strip()
                summary_parts.append(f"Docstring: {docstring}")
        
        elif ext in ["js", "ts", "jsx", "tsx"]:
            # JS/TS header comment
            header_match = re.search(r'/\*\*(.*?)\*/', content, re.DOTALL)
            if header_match:
                header = header_match.group(1).strip()
                summary_parts.append(f"Header: {header}")
        
        # Add information about classes and functions
        if "classes" in metadata and metadata["classes"]:
            class_names = [c["name"] for c in metadata["classes"]]
            summary_parts.append(f"Classes: {', '.join(class_names)}")
        
        if "functions" in metadata and metadata["functions"]:
            func_names = [f["name"] for f in metadata["functions"]]
            summary_parts.append(f"Functions: {', '.join(func_names)}")
        
        # Add information about imports
        if "imports" in metadata and metadata["imports"]:
            import_modules = [i["module"] for i in metadata["imports"] if i["module"]]
            if import_modules:
                summary_parts.append(f"Imports from: {', '.join(import_modules)}")
        
        # If we couldn't extract any meaningful summary, use a default
        if not summary_parts:
            summary_parts.append(f"{metadata['line_count']} lines of {metadata.get('language', ext)} code")
        
        return "\n".join(summary_parts)
    
    def _store_in_knowledge_graph(self, rel_path: str, metadata: Dict[str, Any], update: bool = False) -> bool:
        """
        Store file metadata in the knowledge graph
        
        Args:
            rel_path: Relative path to the file
            metadata: File metadata
            update: Whether this is an update to an existing file
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create or update file entity
            entity_name = f"File:{rel_path}"
            
            # Create observations
            observations = [
                f"Path: {rel_path}",
                f"Type: {metadata.get('extension', 'unknown')} file",
                f"Lines: {metadata.get('line_count', 0)}",
                f"Last modified: {metadata.get('last_modified', datetime.now().isoformat())}"
            ]
            
            # Add language-specific observations
            if "language" in metadata:
                observations.append(f"Language: {metadata['language']}")
            
            if "classes" in metadata and metadata["classes"]:
                class_names = [c["name"] for c in metadata["classes"]]
                observations.append(f"Contains classes: {', '.join(class_names)}")
            
            if "functions" in metadata and metadata["functions"]:
                func_names = [f["name"] for f in metadata["functions"]]
                observations.append(f"Contains functions: {', '.join(func_names)}")
            
            # Add summary if available
            if "summary" in metadata and metadata["summary"]:
                observations.append(f"Summary: {metadata['summary']}")
            
            if update:
                # Update existing entity
                result = self.knowledge_graph.update_entities([{
                    "name": entity_name,
                    "entityType": "File",
                    "observations": observations
                }])
            else:
                # Create new entity
                result = self.knowledge_graph.create_entities([{
                    "name": entity_name,
                    "entityType": "File",
                    "observations": observations
                }])
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing file {rel_path} in knowledge graph: {e}")
            return False
    
    def _should_exclude(self, path: str) -> bool:
        """
        Check if a path should be excluded
        
        Args:
            path: Path to check
            
        Returns:
            Boolean indicating if path should be excluded
        """
        # Skip empty path or paths starting with "."
        if not path or path.startswith("."):
            return True
        
        # Check against exclude patterns
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        
        return False
    
    def _matches_patterns(self, filename: str, patterns: List[str]) -> bool:
        """
        Check if a filename matches any of the patterns
        
        Args:
            filename: Filename to check
            patterns: Patterns to match against
            
        Returns:
            Boolean indicating if filename matches any pattern
        """
        for pattern in patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        
        return False
    
    def get_recent_changes(self) -> Dict[str, Any]:
        """
        Get information about recently changed files
        
        Returns:
            Dictionary with recent changes
        """
        # This would normally track changes since the last check
        # For simplicity, we'll just return the most recently modified files
        recent_files = sorted(
            self.file_metadata.items(),
            key=lambda x: x[1].get("last_modified", 0),
            reverse=True
        )[:10]  # Top 10 most recent
        
        changes = []
        for rel_path, metadata in recent_files:
            # Convert timestamp to string to ensure JSON serialization
            last_modified = metadata.get("last_modified", 0)
            if isinstance(last_modified, (int, float)):
                last_modified_str = time.ctime(last_modified)
            else:
                # Handle datetime objects
                last_modified_str = last_modified.isoformat() if hasattr(last_modified, 'isoformat') else str(last_modified)
                
            changes.append({
                "file_path": rel_path,
                "last_modified": last_modified_str,
                "size_bytes": metadata.get("size_bytes", 0),
                "language": metadata.get("language", "unknown")
            })
        
        return {
            "changes": changes,
            "total_files_indexed": len(self.file_hashes)
        }
    
    def search_code(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code matching the query
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        import re
        
        results = []
        
        # Convert query to regex pattern
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            # If not a valid regex, treat as plain text
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        
        # Search through indexed files
        for rel_path, metadata in self.file_metadata.items():
            if metadata.get("is_binary", False):
                continue
                
            file_path = os.path.join(self.project_path, rel_path)
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # Split into lines
                lines = content.splitlines()
                
                # Find matches
                matches = []
                for i, line in enumerate(lines):
                    if pattern.search(line):
                        # Get context (5 lines before and after)
                        start = max(0, i - 5)
                        end = min(len(lines), i + 6)
                        context = lines[start:end]
                        
                        matches.append({
                            "line_number": i + 1,  # 1-indexed line numbers
                            "content": line,
                            "context": context
                        })
                
                if matches:
                    results.append({
                        "file_path": rel_path,
                        "matches": matches
                    })
                    
                    # Stop if we have enough results
                    if len(results) >= limit:
                        break
                        
            except Exception as e:
                logger.error(f"Error searching file {rel_path}: {e}")
        
        return results
    
    def index_files(self, path: Optional[str] = None, patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Manually trigger file indexing
        
        Args:
            path: Optional specific path to index (relative to project root)
            patterns: Optional override for file patterns
            
        Returns:
            Dictionary with indexing results
        """
        if path:
            # Convert relative path to absolute
            abs_path = os.path.join(self.project_path, path)
            return self._index_files(abs_path, patterns)
        else:
            return self._index_files(patterns=patterns)
    
    def summarize_file(self, file_path: str) -> Dict[str, Any]:
        """
        Generate a detailed summary of a file
        
        Args:
            file_path: Path to the file (relative to project root)
            
        Returns:
            Dictionary with file summary
        """
        abs_path = os.path.join(self.project_path, file_path)
        
        if not os.path.exists(abs_path):
            return {"error": f"File not found: {file_path}"}
        
        try:
            # Get metadata and summary
            metadata = self._extract_metadata(abs_path)
            
            # Find structure information
            structure = {
                "classes": metadata.get("classes", []),
                "functions": metadata.get("functions", []),
                "imports": metadata.get("imports", []),
            }
            
            return {
                "file_path": file_path,
                "metadata": {
                    "size_bytes": metadata.get("size_bytes", 0),
                    "line_count": metadata.get("line_count", 0),
                    "language": metadata.get("language", "unknown"),
                    "extension": metadata.get("extension", ""),
                    "last_modified": metadata.get("last_modified", "Unknown")  # Just return the ISO format string directly
                },
                "summary": metadata.get("summary", "No summary available"),
                "structure": structure
            }
            
        except Exception as e:
            logger.error(f"Error summarizing file {file_path}: {e}")
            return {"error": f"Error summarizing file: {str(e)}", "file_path": file_path} 