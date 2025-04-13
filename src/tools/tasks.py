#!/usr/bin/env python3
"""
Task Management Tool for MCP Think Tank
Enables creating, tracking, and managing tasks with knowledge graph integration
"""
import logging
import uuid
import os
import json
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from pydantic import BaseModel, Field

# Import for Anthropic API
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Imports for Hugging Face and local LLM
try:
    from huggingface_hub import snapshot_download, hf_hub_download
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

logger = logging.getLogger("mcp-think-tank.tasks")

# Model configuration
LOCAL_MODEL_ID = "google/gemma-3-1b-instruct"
LOCAL_MODEL_PATH = os.path.expanduser("~/.mcp-think-tank/models/gemma-3-1b")

class TaskStatus(str, Enum):
    """Enum for task status"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"

class TaskPriority(str, Enum):
    """Enum for task priority"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Task(BaseModel):
    """Model for storing task data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)  # IDs of tasks this task depends on
    assigned_to: Optional[str] = None
    created_by: Optional[str] = None
    parent_id: Optional[str] = None  # For hierarchical tasks

class LLMClient:
    """Base class for LLM clients that handle task parsing"""
    def __init__(self):
        pass
    
    async def parse_tasks(self, text: str) -> List[Dict[str, Any]]:
        """Parse tasks from text - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement parse_tasks method")

class AnthropicClient(LLMClient):
    """Client for using Anthropic API to parse tasks"""
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        
        if not self.api_key:
            logger.warning("No Anthropic API key provided, API will not be available")
            return
            
        try:
            if ANTHROPIC_AVAILABLE:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized successfully")
            else:
                logger.warning("anthropic package not installed, API will not be available")
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
    
    async def parse_tasks(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tasks from text using Anthropic API
        
        Args:
            text: The text to parse tasks from
            
        Returns:
            List of parsed tasks
        """
        if not self.client:
            raise ValueError("Anthropic client not initialized")
            
        logger.info("Parsing tasks with Anthropic API")
        
        # Create the prompt
        system_prompt = """
        You are a task parsing assistant that extracts structured tasks from text.
        Extract all tasks, to-dos, or action items from the input text.
        For each task, extract:
        1. title: A concise title for the task
        2. description: A detailed description if available
        3. priority: Infer priority as "low", "medium", "high", or "critical"
        4. tags: Any relevant tags or categories
        5. dependencies: IDs or references to tasks that this task depends on
        
        Respond with a JSON array of tasks in this format:
        [
            {
                "title": "Task title",
                "description": "Task description",
                "priority": "medium",
                "tags": ["tag1", "tag2"],
                "dependencies": []
            }
        ]
        
        Be comprehensive and extract ALL possible tasks from the input.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": text}
                ],
                temperature=0.0
            )
            
            # Extract and parse the JSON response
            content = response.content[0].text
            
            # Try to find JSON in the response
            try:
                # Look for JSON array in the response
                import re
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                
                tasks = json.loads(content)
                return tasks
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from Anthropic response: {content}")
                return []
                
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return []

class LocalLLMClient(LLMClient):
    """Client for using local LLM to parse tasks"""
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.model_path = model_path or LOCAL_MODEL_PATH
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        # Don't try to load model immediately - will load on first use
        
    def load_model(self) -> bool:
        """
        Load the local LLM model
        
        Returns:
            True if successful, False otherwise
        """
        if not HUGGINGFACE_AVAILABLE:
            logger.error("huggingface_hub and torch packages not installed, local LLM not available")
            return False
            
        if self.model is not None:
            return True
            
        try:
            logger.info(f"Loading local LLM from {self.model_path}")
            
            # Check if model exists, download if not
            if not os.path.exists(self.model_path):
                logger.info(f"Model not found at {self.model_path}, downloading...")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                snapshot_download(
                    repo_id=LOCAL_MODEL_ID,
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False
                )
            
            # Load the model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            
            # Create pipeline for easier generation
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=4000,
                temperature=0.1
            )
            
            logger.info("Local LLM loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading local LLM: {e}")
            return False
    
    async def parse_tasks(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tasks from text using local LLM
        
        Args:
            text: The text to parse tasks from
            
        Returns:
            List of parsed tasks
        """
        if not self.load_model():
            raise ValueError("Failed to load local LLM")
            
        logger.info("Parsing tasks with local LLM")
        
        # Create the prompt
        prompt = f"""<|system|>
You are a task parsing assistant that extracts structured tasks from text.
Extract all tasks, to-dos, or action items from the input text.
For each task, extract:
1. title: A concise title for the task
2. description: A detailed description if available
3. priority: Infer priority as "low", "medium", "high", or "critical"
4. tags: Any relevant tags or categories

Respond with a JSON array of tasks in this format:
[
    {{
        "title": "Task title",
        "description": "Task description",
        "priority": "medium",
        "tags": ["tag1", "tag2"]
    }}
]

Be comprehensive and extract ALL possible tasks from the input.
<|user|>
{text}
<|assistant|>
"""
        
        try:
            # Generate response from the model
            response = self.pipe(prompt)[0]["generated_text"]
            
            # Extract the assistant's response
            response = response.split("<|assistant|>")[-1].strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON array in the response
                import re
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                if json_match:
                    response = json_match.group(0)
                
                tasks = json.loads(response)
                return tasks
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from local LLM response: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating from local LLM: {e}")
            return []

class TaskParser:
    """
    Handles parsing tasks from text using either Anthropic API or local LLM
    """
    def __init__(self, anthropic_api_key: Optional[str] = None, model_path: Optional[str] = None):
        self.anthropic_client = AnthropicClient(api_key=anthropic_api_key)
        self.local_llm_client = LocalLLMClient(model_path=model_path)
        
    async def parse_tasks(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tasks from text using available LLM clients
        
        First tries Anthropic API, falls back to local LLM if needed
        
        Args:
            text: The text to parse tasks from
            
        Returns:
            List of parsed tasks
        """
        # Try with Anthropic first
        try:
            if ANTHROPIC_AVAILABLE and self.anthropic_client.client:
                logger.info("Attempting to parse tasks with Anthropic API")
                tasks = await self.anthropic_client.parse_tasks(text)
                if tasks:
                    logger.info(f"Successfully parsed {len(tasks)} tasks with Anthropic API")
                    return tasks
                logger.warning("Anthropic API returned no tasks, falling back to local LLM")
            else:
                logger.info("Anthropic API not available, using local LLM")
        except Exception as e:
            logger.error(f"Error parsing tasks with Anthropic API: {e}")
            logger.info("Falling back to local LLM")
        
        # Fall back to local LLM
        try:
            if HUGGINGFACE_AVAILABLE:
                logger.info("Attempting to parse tasks with local LLM")
                tasks = await self.local_llm_client.parse_tasks(text)
                if tasks:
                    logger.info(f"Successfully parsed {len(tasks)} tasks with local LLM")
                    return tasks
                logger.warning("Local LLM returned no tasks")
            else:
                logger.error("Local LLM not available (missing dependencies)")
        except Exception as e:
            logger.error(f"Error parsing tasks with local LLM: {e}")
        
        # If all else fails, return empty list
        logger.error("Failed to parse tasks with any available method")
        return []

class TaskManager:
    """
    Tool for task management and planning
    
    The Task Manager allows creating, updating, listing, and deleting tasks,
    with optional integration with the knowledge graph for persistence and
    semantic search capabilities.
    """
    
    def __init__(self, knowledge_graph=None, llm_client=None, anthropic_api_key=None):
        """
        Initialize the Task Manager
        
        Args:
            knowledge_graph: Knowledge graph instance for storing tasks
            llm_client: Optional client for LLM-based task parsing
            anthropic_api_key: Optional Anthropic API key
        """
        self.knowledge_graph = knowledge_graph
        self.tasks = {}  # In-memory storage when KG is not available
        
        # Initialize task parser
        self.task_parser = TaskParser(anthropic_api_key=anthropic_api_key)
        
        logger.info("Task Manager initialized")
    
    def create_task(self, 
                   title: str,
                   description: str = "",
                   priority: str = "medium",
                   tags: Optional[List[str]] = None,
                   dependencies: Optional[List[str]] = None,
                   assigned_to: Optional[str] = None,
                   parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new task
        
        Args:
            title: Task title
            description: Task description
            priority: Task priority (low, medium, high, critical)
            tags: Optional list of tags
            dependencies: Optional list of task IDs this task depends on
            assigned_to: Optional assignee
            parent_id: Optional parent task ID
            
        Returns:
            Dict with the created task info
        """
        # Validate priority
        try:
            priority_enum = TaskPriority(priority.lower())
        except ValueError:
            logger.warning(f"Invalid priority: {priority}, defaulting to MEDIUM")
            priority_enum = TaskPriority.MEDIUM
            
        # Create task object
        task = Task(
            title=title,
            description=description,
            priority=priority_enum,
            tags=tags or [],
            dependencies=dependencies or [],
            assigned_to=assigned_to,
            parent_id=parent_id
        )
        
        # Log the task creation
        logger.info(f"Creating task: {task.id} - {task.title}")
        
        # Store in memory
        self.tasks[task.id] = task
        
        # Store in knowledge graph if available
        kg_entity_id = None
        if self.knowledge_graph:
            try:
                kg_entity_id = self._store_in_knowledge_graph(task)
            except Exception as e:
                logger.error(f"Error storing task in knowledge graph: {e}")
        
        return {
            "task_id": task.id,
            "title": task.title,
            "status": task.status,
            "created_at": task.created_at,
            "stored_in_kg": kg_entity_id is not None,
            "kg_entity_id": kg_entity_id
        }
    
    def list_tasks(self, 
                  status: Optional[str] = None,
                  priority: Optional[str] = None,
                  tags: Optional[List[str]] = None,
                  assigned_to: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List tasks with optional filtering
        
        Args:
            status: Optional status filter
            priority: Optional priority filter
            tags: Optional tags filter (tasks must have at least one of these tags)
            assigned_to: Optional assignee filter
            
        Returns:
            List of matching tasks
        """
        # Filter tasks based on criteria
        filtered_tasks = self.tasks.values()
        
        if status:
            try:
                status_enum = TaskStatus(status.lower())
                filtered_tasks = [t for t in filtered_tasks if t.status == status_enum]
            except ValueError:
                logger.warning(f"Invalid status filter: {status}")
        
        if priority:
            try:
                priority_enum = TaskPriority(priority.lower())
                filtered_tasks = [t for t in filtered_tasks if t.priority == priority_enum]
            except ValueError:
                logger.warning(f"Invalid priority filter: {priority}")
        
        if tags:
            filtered_tasks = [t for t in filtered_tasks if any(tag in t.tags for tag in tags)]
        
        if assigned_to:
            filtered_tasks = [t for t in filtered_tasks if t.assigned_to == assigned_to]
        
        # Convert tasks to dictionaries for output
        return [
            {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "status": task.status,
                "priority": task.priority,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "completed_at": task.completed_at,
                "tags": task.tags,
                "dependencies": task.dependencies,
                "assigned_to": task.assigned_to,
                "parent_id": task.parent_id
            }
            for task in filtered_tasks
        ]
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing task
        
        Args:
            task_id: ID of the task to update
            updates: Dictionary of fields to update
            
        Returns:
            Dict with updated task info or error
        """
        if task_id not in self.tasks:
            logger.warning(f"Task not found: {task_id}")
            return {"error": f"Task not found: {task_id}"}
        
        task = self.tasks[task_id]
        
        # Process updates
        for field, value in updates.items():
            if field == "status":
                try:
                    status_enum = TaskStatus(value.lower())
                    task.status = status_enum
                    
                    # Update completed_at if status changed to DONE
                    if status_enum == TaskStatus.DONE and task.completed_at is None:
                        task.completed_at = datetime.now().isoformat()
                except ValueError:
                    logger.warning(f"Invalid status: {value}")
            
            elif field == "priority":
                try:
                    priority_enum = TaskPriority(value.lower())
                    task.priority = priority_enum
                except ValueError:
                    logger.warning(f"Invalid priority: {value}")
            
            elif field == "title":
                task.title = value
            
            elif field == "description":
                task.description = value
            
            elif field == "tags":
                task.tags = value
            
            elif field == "dependencies":
                task.dependencies = value
            
            elif field == "assigned_to":
                task.assigned_to = value
            
            elif field == "parent_id":
                task.parent_id = value
        
        # Update the updated_at timestamp
        task.updated_at = datetime.now().isoformat()
        
        # Update in knowledge graph if available
        kg_updated = False
        if self.knowledge_graph:
            try:
                kg_updated = self._update_in_knowledge_graph(task)
            except Exception as e:
                logger.error(f"Error updating task in knowledge graph: {e}")
        
        return {
            "task_id": task.id,
            "title": task.title,
            "status": task.status,
            "updated_at": task.updated_at,
            "updated_in_kg": kg_updated
        }
    
    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """
        Delete a task
        
        Args:
            task_id: ID of the task to delete
            
        Returns:
            Dict with result of the deletion
        """
        if task_id not in self.tasks:
            logger.warning(f"Task not found: {task_id}")
            return {"error": f"Task not found: {task_id}"}
        
        # Remove from in-memory storage
        task = self.tasks.pop(task_id)
        
        # Delete from knowledge graph if available
        kg_deleted = False
        if self.knowledge_graph:
            try:
                kg_deleted = self._delete_from_knowledge_graph(task)
            except Exception as e:
                logger.error(f"Error deleting task from knowledge graph: {e}")
        
        return {
            "task_id": task_id,
            "title": task.title,
            "deleted": True,
            "deleted_from_kg": kg_deleted
        }
    
    async def parse_tasks_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tasks from a text description using an LLM
        
        Args:
            text: Text to parse into tasks
            
        Returns:
            List of parsed tasks
        """
        logger.info("Parsing tasks from text")
        
        try:
            # Parse tasks using the task parser
            parsed_tasks = await self.task_parser.parse_tasks(text)
            
            # Create tasks from the parsed data
            created_tasks = []
            for task_data in parsed_tasks:
                # Create each task
                result = self.create_task(
                    title=task_data.get("title", "Untitled Task"),
                    description=task_data.get("description", ""),
                    priority=task_data.get("priority", "medium"),
                    tags=task_data.get("tags", []),
                    dependencies=task_data.get("dependencies", [])
                )
                created_tasks.append(result)
            
            logger.info(f"Created {len(created_tasks)} tasks from parsed text")
            return created_tasks
            
        except Exception as e:
            logger.error(f"Error parsing tasks from text: {e}")
            return []
    
    def _store_in_knowledge_graph(self, task: Task) -> Optional[str]:
        """
        Store a task in the knowledge graph
        
        Args:
            task: The task to store
            
        Returns:
            Entity ID if successful, None otherwise
        """
        if not self.knowledge_graph:
            return None
        
        # Create entity observations
        observations = [
            f"Title: {task.title}",
            f"Description: {task.description}" if task.description else "No description",
            f"Status: {task.status.value}",
            f"Priority: {task.priority.value}",
            f"Created: {task.created_at}",
            f"Updated: {task.updated_at}"
        ]
        
        if task.tags:
            observations.append(f"Tags: {', '.join(task.tags)}")
        
        if task.dependencies:
            observations.append(f"Dependencies: {', '.join(task.dependencies)}")
        
        if task.assigned_to:
            observations.append(f"Assigned to: {task.assigned_to}")
        
        if task.parent_id:
            observations.append(f"Parent task: {task.parent_id}")
        
        # Create entity
        entity = {
            "name": f"task_{task.id}",
            "entityType": "task",
            "observations": observations
        }
        
        # Create in knowledge graph
        result = self.knowledge_graph.create_entities([entity])
        
        # Create relations for dependencies if any
        if task.dependencies:
            relations = []
            for dep_id in task.dependencies:
                relations.append({
                    "from": f"task_{task.id}",
                    "to": f"task_{dep_id}",
                    "relationType": "dependsOn"
                })
            
            if relations:
                self.knowledge_graph.create_relations(relations)
        
        # Create relation to parent if any
        if task.parent_id:
            relation = {
                "from": f"task_{task.id}",
                "to": f"task_{task.parent_id}",
                "relationType": "isSubtaskOf"
            }
            self.knowledge_graph.create_relations([relation])
        
        return f"task_{task.id}"
    
    def _update_in_knowledge_graph(self, task: Task) -> bool:
        """
        Update a task in the knowledge graph
        
        Args:
            task: The task to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self.knowledge_graph:
            return False
        
        # Create updated observations
        observations = [
            f"Title: {task.title}",
            f"Description: {task.description}" if task.description else "No description",
            f"Status: {task.status.value}",
            f"Priority: {task.priority.value}",
            f"Created: {task.created_at}",
            f"Updated: {task.updated_at}"
        ]
        
        if task.completed_at:
            observations.append(f"Completed: {task.completed_at}")
        
        if task.tags:
            observations.append(f"Tags: {', '.join(task.tags)}")
        
        if task.dependencies:
            observations.append(f"Dependencies: {', '.join(task.dependencies)}")
        
        if task.assigned_to:
            observations.append(f"Assigned to: {task.assigned_to}")
        
        if task.parent_id:
            observations.append(f"Parent task: {task.parent_id}")
        
        # Update entity
        entity = {
            "name": f"task_{task.id}",
            "observations": observations
        }
        
        # Update in knowledge graph
        result = self.knowledge_graph.update_entities([entity])
        
        # For simplicity, we'll just recreate the relations
        # In a production system, you might want to check what changed and update only those
        
        # Delete existing relations
        self.knowledge_graph.delete_relations([
            {
                "from": f"task_{task.id}",
                "to": f"task_{dep_id}",
                "relationType": "dependsOn"
            }
            for dep_id in task.dependencies
        ])
        
        if task.parent_id:
            self.knowledge_graph.delete_relations([
                {
                    "from": f"task_{task.id}",
                    "to": f"task_{task.parent_id}",
                    "relationType": "isSubtaskOf"
                }
            ])
        
        # Create new relations
        if task.dependencies:
            relations = []
            for dep_id in task.dependencies:
                relations.append({
                    "from": f"task_{task.id}",
                    "to": f"task_{dep_id}",
                    "relationType": "dependsOn"
                })
            
            if relations:
                self.knowledge_graph.create_relations(relations)
        
        # Create relation to parent if any
        if task.parent_id:
            relation = {
                "from": f"task_{task.id}",
                "to": f"task_{task.parent_id}",
                "relationType": "isSubtaskOf"
            }
            self.knowledge_graph.create_relations([relation])
        
        return True
    
    def _delete_from_knowledge_graph(self, task: Task) -> bool:
        """
        Delete a task from the knowledge graph
        
        Args:
            task: The task to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.knowledge_graph:
            return False
        
        # Delete the entity
        result = self.knowledge_graph.delete_entities([f"task_{task.id}"])
        
        # Relations will be automatically deleted or marked as invalid
        
        return True

# Function to download local model during setup
def download_local_model(force: bool = False) -> bool:
    """
    Download the local LLM model for task parsing
    
    Args:
        force: Force download even if model already exists
        
    Returns:
        True if successful, False otherwise
    """
    if not HUGGINGFACE_AVAILABLE:
        logger.error("huggingface_hub package not installed, cannot download model")
        return False
    
    try:
        model_path = LOCAL_MODEL_PATH
        
        # Check if model already exists
        if os.path.exists(model_path) and not force:
            logger.info(f"Model already exists at {model_path}")
            return True
        
        # Create directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Download model
        logger.info(f"Downloading local LLM model ({LOCAL_MODEL_ID}) to {model_path}")
        snapshot_download(
            repo_id=LOCAL_MODEL_ID,
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        
        logger.info("Model downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

class TasksTool:
    """
    Tool wrapper for Task Management to expose methods as MCP tools.
    
    This class wraps the TaskManager class and provides methods that
    can be registered with the MCP server for task management functionality.
    """
    
    def __init__(self, use_basic: bool = False):
        """
        Initialize the tasks tool.
        
        Args:
            use_basic: If True, use a simplified version with reduced functionality
        """
        # Get configuration
        from src.config import get_config
        
        config = get_config()
        
        # Initialize knowledge graph reference
        self.knowledge_graph = None
        
        # Try to import MemoryTool for knowledge graph access
        try:
            from src.tools.memory import MemoryTool
            memory_tool = MemoryTool(use_basic=use_basic)
            self.knowledge_graph = memory_tool.knowledge_graph
            logger.info("Successfully connected to knowledge graph")
        except Exception as e:
            logger.warning(f"Could not connect to knowledge graph: {e}")
        
        # Initialize with LLM client if possible
        anthropic_api_key = None if use_basic else config.anthropic_api_key
        
        # Create TaskManager instance
        self.task_manager = TaskManager(
            knowledge_graph=self.knowledge_graph,
            anthropic_api_key=anthropic_api_key
        )
        
        # Dictionary to store active requests
        self.requests = {}
        
        logger.info("Tasks Tool initialized")
    
    # MCP tool methods
    
    async def request_planning(self, originalRequest: str, tasks: List[Dict[str, str]], splitDetails: Optional[str] = None) -> Dict[str, Any]:
        """
        Register a new user request and plan its associated tasks.
        
        Args:
            originalRequest: The original user request
            tasks: List of tasks with title and description
            splitDetails: Optional details about how the request was split
            
        Returns:
            Dictionary with request details and task information
        """
        # Generate a request ID
        request_id = str(uuid.uuid4())
        
        # Store the request
        self.requests[request_id] = {
            "id": request_id,
            "originalRequest": originalRequest,
            "splitDetails": splitDetails,
            "tasks": [],
            "currentTaskIndex": 0,
            "status": "planning"
        }
        
        # Create the tasks
        for task_data in tasks:
            task_result = self.task_manager.create_task(
                title=task_data.get("title", "Untitled Task"),
                description=task_data.get("description", ""),
                tags=["mcp-task-manager"]
            )
            
            # Store task in request
            self.requests[request_id]["tasks"].append({
                "id": task_result.get("id"),
                "title": task_data.get("title"),
                "description": task_data.get("description"),
                "status": "pending"
            })
        
        # Update request status
        self.requests[request_id]["status"] = "in_progress"
        
        return {
            "requestId": request_id,
            "originalRequest": originalRequest,
            "tasks": self.requests[request_id]["tasks"],
            "message": f"Created request with {len(tasks)} tasks"
        }
    
    async def get_next_task(self, requestId: str) -> Dict[str, Any]:
        """
        Get the next pending task for a request.
        
        Args:
            requestId: The ID of the request
            
        Returns:
            Dictionary with the next task details or completion status
        """
        # Check if request exists
        if requestId not in self.requests:
            return {"error": f"Request {requestId} not found"}
        
        # Get the request
        request = self.requests[requestId]
        
        # Check if all tasks are done
        all_done = all(task.get("status") == "done" for task in request["tasks"])
        if all_done:
            return {
                "requestId": requestId,
                "all_tasks_done": True,
                "message": "All tasks have been completed",
                "tasks": request["tasks"]
            }
        
        # Get the current task index
        current_index = request["currentTaskIndex"]
        
        # Find the next pending task
        next_task = None
        for i in range(current_index, len(request["tasks"])):
            if request["tasks"][i]["status"] == "pending":
                next_task = request["tasks"][i]
                request["currentTaskIndex"] = i
                break
        
        # If no pending task found from current index, search from beginning
        if next_task is None:
            for i in range(0, current_index):
                if request["tasks"][i]["status"] == "pending":
                    next_task = request["tasks"][i]
                    request["currentTaskIndex"] = i
                    break
        
        if next_task:
            return {
                "requestId": requestId,
                "taskId": next_task["id"],
                "title": next_task["title"],
                "description": next_task["description"],
                "tasks": request["tasks"]
            }
        else:
            # This shouldn't happen based on the all_done check, but just in case
            return {
                "requestId": requestId,
                "all_tasks_done": True,
                "message": "All tasks have been completed",
                "tasks": request["tasks"]
            }
    
    async def mark_task_done(self, requestId: str, taskId: str, completedDetails: Optional[str] = None) -> Dict[str, Any]:
        """
        Mark a task as done.
        
        Args:
            requestId: The ID of the request
            taskId: The ID of the task
            completedDetails: Optional details about how the task was completed
            
        Returns:
            Dictionary with the updated task details
        """
        # Check if request exists
        if requestId not in self.requests:
            return {"error": f"Request {requestId} not found"}
        
        # Get the request
        request = self.requests[requestId]
        
        # Find the task
        task_found = False
        for task in request["tasks"]:
            if task["id"] == taskId:
                task["status"] = "done"
                if completedDetails:
                    task["completedDetails"] = completedDetails
                task_found = True
                break
        
        if not task_found:
            return {"error": f"Task {taskId} not found in request {requestId}"}
        
        # Update the task in the task manager
        self.task_manager.update_task(taskId, {"status": "done"})
        
        return {
            "requestId": requestId,
            "taskId": taskId,
            "status": "done",
            "message": "Task marked as done",
            "tasks": request["tasks"]
        }
    
    async def approve_task_completion(self, requestId: str, taskId: str) -> Dict[str, Any]:
        """
        Approve a completed task.
        
        Args:
            requestId: The ID of the request
            taskId: The ID of the task
            
        Returns:
            Dictionary with the approved task details
        """
        # Check if request exists
        if requestId not in self.requests:
            return {"error": f"Request {requestId} not found"}
        
        # Get the request
        request = self.requests[requestId]
        
        # Find the task
        task_found = False
        for task in request["tasks"]:
            if task["id"] == taskId:
                if task["status"] != "done":
                    return {"error": f"Task {taskId} is not marked as done"}
                task["status"] = "approved"
                task_found = True
                break
        
        if not task_found:
            return {"error": f"Task {taskId} not found in request {requestId}"}
        
        return {
            "requestId": requestId,
            "taskId": taskId,
            "status": "approved",
            "message": "Task completion approved",
            "tasks": request["tasks"]
        }
    
    async def approve_request_completion(self, requestId: str) -> Dict[str, Any]:
        """
        Approve the completion of an entire request.
        
        Args:
            requestId: The ID of the request
            
        Returns:
            Dictionary with the completed request details
        """
        # Check if request exists
        if requestId not in self.requests:
            return {"error": f"Request {requestId} not found"}
        
        # Get the request
        request = self.requests[requestId]
        
        # Check if all tasks are done or approved
        for task in request["tasks"]:
            if task["status"] not in ["done", "approved"]:
                return {"error": f"Not all tasks are completed or approved"}
        
        # Update request status
        request["status"] = "completed"
        
        return {
            "requestId": requestId,
            "status": "completed",
            "message": "Request completion approved",
            "tasks": request["tasks"]
        }
    
    async def open_task_details(self, taskId: str) -> Dict[str, Any]:
        """
        Get details for a specific task.
        
        Args:
            taskId: The ID of the task
            
        Returns:
            Dictionary with the task details
        """
        # Try to get the task from the task manager
        try:
            tasks = self.task_manager.list_tasks()
            for task in tasks:
                if task["id"] == taskId:
                    return task
            
            return {"error": f"Task {taskId} not found"}
        except Exception as e:
            return {"error": f"Error retrieving task details: {str(e)}"}
    
    async def list_requests(self, random_string: str = "") -> Dict[str, Any]:
        """
        List all requests with their task summaries.
        
        Args:
            random_string: Dummy parameter for no-parameter tools
            
        Returns:
            Dictionary with the list of requests
        """
        return {
            "requests": list(self.requests.values()),
            "count": len(self.requests)
        }
    
    async def add_tasks_to_request(self, requestId: str, tasks: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Add new tasks to an existing request.
        
        Args:
            requestId: The ID of the request
            tasks: List of tasks with title and description
            
        Returns:
            Dictionary with the updated request details
        """
        # Check if request exists
        if requestId not in self.requests:
            return {"error": f"Request {requestId} not found"}
        
        # Get the request
        request = self.requests[requestId]
        
        # Create and add the tasks
        added_tasks = []
        for task_data in tasks:
            task_result = self.task_manager.create_task(
                title=task_data.get("title", "Untitled Task"),
                description=task_data.get("description", ""),
                tags=["mcp-task-manager"]
            )
            
            # Store task in request
            task_info = {
                "id": task_result.get("id"),
                "title": task_data.get("title"),
                "description": task_data.get("description"),
                "status": "pending"
            }
            request["tasks"].append(task_info)
            added_tasks.append(task_info)
        
        return {
            "requestId": requestId,
            "added_tasks": added_tasks,
            "task_count": len(request["tasks"]),
            "message": f"Added {len(added_tasks)} tasks to request",
            "tasks": request["tasks"]
        }
    
    async def update_task(self, requestId: str, taskId: str, title: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Update a task in a request.
        
        Args:
            requestId: The ID of the request
            taskId: The ID of the task
            title: Optional new title for the task
            description: Optional new description for the task
            
        Returns:
            Dictionary with the updated task details
        """
        # Check if request exists
        if requestId not in self.requests:
            return {"error": f"Request {requestId} not found"}
        
        # Get the request
        request = self.requests[requestId]
        
        # Find the task
        task_found = False
        for task in request["tasks"]:
            if task["id"] == taskId:
                if title:
                    task["title"] = title
                if description:
                    task["description"] = description
                task_found = True
                break
        
        if not task_found:
            return {"error": f"Task {taskId} not found in request {requestId}"}
        
        # Update the task in the task manager
        updates = {}
        if title:
            updates["title"] = title
        if description:
            updates["description"] = description
            
        self.task_manager.update_task(taskId, updates)
        
        return {
            "requestId": requestId,
            "taskId": taskId,
            "message": "Task updated",
            "tasks": request["tasks"]
        }
    
    async def delete_task(self, requestId: str, taskId: str) -> Dict[str, Any]:
        """
        Delete a task from a request.
        
        Args:
            requestId: The ID of the request
            taskId: The ID of the task
            
        Returns:
            Dictionary with the updated request details
        """
        # Check if request exists
        if requestId not in self.requests:
            return {"error": f"Request {requestId} not found"}
        
        # Get the request
        request = self.requests[requestId]
        
        # Find and remove the task
        for i, task in enumerate(request["tasks"]):
            if task["id"] == taskId:
                # Remove from request tasks
                removed_task = request["tasks"].pop(i)
                
                # Delete from task manager
                self.task_manager.delete_task(taskId)
                
                return {
                    "requestId": requestId,
                    "taskId": taskId,
                    "message": "Task deleted",
                    "deleted_task": removed_task,
                    "tasks": request["tasks"]
                }
        
        return {"error": f"Task {taskId} not found in request {requestId}"} 