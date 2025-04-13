#!/usr/bin/env python3
"""
Workflow Templates for MCP Think Tank
Provides pre-defined workflow templates for common development tasks
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import enum
import json
import uuid

from .dag_orchestrator import DAGExecutor
from ..config import Config

logger = logging.getLogger("mcp-think-tank.workflow_templates")

class WorkflowType(enum.Enum):
    """Enum representing different types of workflows."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    LEARNING = "learning"
    KNOWLEDGE_REASONING = "knowledge_reasoning"
    CUSTOM = "custom"


class WorkflowTemplate:
    """
    A template for creating standardized workflows.
    
    Each template defines a sequence of tasks/steps that make up a workflow,
    along with their dependencies, inputs, and outputs.
    """
    
    def __init__(
        self, 
        workflow_type: WorkflowType,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        default_timeout: float = 60.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new workflow template.
        
        Args:
            workflow_type: Type of workflow (from WorkflowType enum)
            name: Name of the workflow template
            description: Description of what this workflow does
            steps: List of step definitions, where each step is a dict with:
                - id: Unique step identifier
                - name: Human-readable name for the step
                - description: What this step does
                - tool: Tool to use for this step (optional)
                - inputs: Dict mapping input names to sources (step outputs or workflow inputs)
                - outputs: List of output names this step produces
                - dependencies: List of step IDs this step depends on
                - timeout: Optional timeout for this step (defaults to default_timeout)
            default_timeout: Default timeout in seconds for steps
            metadata: Additional metadata for the workflow template
        """
        self.workflow_type = workflow_type
        self.name = name
        self.description = description
        self.steps = steps
        self.default_timeout = default_timeout
        self.metadata = metadata or {}
        
        # Validate steps
        self._validate_steps()
        
    def _validate_steps(self):
        """Validate that steps have required fields and valid dependencies."""
        step_ids = set()
        
        for step in self.steps:
            # Check required fields
            if not all(key in step for key in ["id", "name", "description"]):
                missing = [k for k in ["id", "name", "description"] if k not in step]
                raise ValueError(f"Step missing required fields: {missing}")
            
            # Track step IDs
            step_ids.add(step["id"])
        
        # Check dependencies
        for step in self.steps:
            dependencies = step.get("dependencies", [])
            for dep in dependencies:
                if dep not in step_ids:
                    raise ValueError(f"Step {step['id']} has invalid dependency: {dep}")
    
    def create_workflow_instance(self, workflow_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a workflow instance from this template.
        
        Args:
            workflow_inputs: Input parameters for the workflow
            
        Returns:
            Dict containing the workflow instance definition
        """
        workflow_id = str(uuid.uuid4())
        
        return {
            "id": workflow_id,
            "template_name": self.name,
            "type": self.workflow_type.value,
            "description": self.description,
            "status": "created",
            "created_at": None,  # Will be filled by the executor
            "updated_at": None,  # Will be filled by the executor
            "inputs": workflow_inputs or {},
            "outputs": {},
            "steps": self._prepare_steps_for_instance(),
            "metadata": self.metadata.copy()
        }
    
    def _prepare_steps_for_instance(self) -> List[Dict[str, Any]]:
        """Prepare steps for a workflow instance."""
        instance_steps = []
        
        for step in self.steps:
            instance_step = step.copy()
            instance_step["status"] = "pending"
            instance_step["result"] = None
            instance_step["error"] = None
            instance_step["started_at"] = None
            instance_step["finished_at"] = None
            instance_step["timeout"] = step.get("timeout", self.default_timeout)
            
            instance_steps.append(instance_step)
            
        return instance_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the template to a dictionary."""
        return {
            "workflow_type": self.workflow_type.value,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "default_timeout": self.default_timeout,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowTemplate':
        """Create a template from a dictionary."""
        return cls(
            workflow_type=WorkflowType(data["workflow_type"]),
            name=data["name"],
            description=data["description"],
            steps=data["steps"],
            default_timeout=data.get("default_timeout", 60.0),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WorkflowTemplate':
        """Create a template from a JSON string."""
        return cls.from_dict(json.loads(json_str))


class WorkflowFactory:
    """
    Factory class for creating and managing workflow templates.
    
    This class provides methods for registering, retrieving, and
    instantiating workflow templates.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the workflow factory.
        
        Args:
            config: Configuration object
        """
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.config = config
        
        # Register built-in templates
        self._register_built_in_templates()
    
    def _register_built_in_templates(self):
        """Register built-in workflow templates."""
        # Register code generation workflow
        self.register_template(self._create_code_generation_template())
        
        # Register code review workflow
        self.register_template(self._create_code_review_template())
        
        # Register bug fix workflow
        self.register_template(self._create_bug_fix_template())
        
        # Register refactoring workflow
        self.register_template(self._create_refactoring_template())
        
        # Register documentation workflow
        self.register_template(self._create_documentation_template())
        
        # Register learning workflow
        self.register_template(self._create_learning_template())

        # Register knowledge reasoning workflow
        self.register_template(self._create_knowledge_reasoning_template())
    
    def register_template(self, template: WorkflowTemplate) -> None:
        """
        Register a workflow template.
        
        Args:
            template: Workflow template to register
        """
        self.templates[template.name] = template
        logger.info(f"Registered workflow template: {template.name}")
    
    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """
        Get a workflow template by name.
        
        Args:
            name: Name of the template to retrieve
            
        Returns:
            WorkflowTemplate if found, None otherwise
        """
        return self.templates.get(name)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all registered templates.
        
        Returns:
            List of template summaries
        """
        return [
            {
                "name": template.name,
                "type": template.workflow_type.value,
                "description": template.description,
                "num_steps": len(template.steps)
            }
            for template in self.templates.values()
        ]
    
    def create_workflow(self, template_name: str, workflow_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a workflow instance from a template.
        
        Args:
            template_name: Name of the template to use
            workflow_inputs: Input parameters for the workflow
            
        Returns:
            Dict containing the workflow instance
            
        Raises:
            ValueError: If template doesn't exist
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Workflow template not found: {template_name}")
        
        return template.create_workflow_instance(workflow_inputs)
    
    def create_feature_workflow(self, feature_description: str) -> Dict[str, Any]:
        """
        Create a feature implementation workflow.
        
        Args:
            feature_description: Description of the feature to implement
            
        Returns:
            Workflow instance
        """
        return self.create_workflow("code_generation", {"requirements": feature_description})
    
    def create_bugfix_workflow(self, bug_description: str, error_logs: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a bug fix workflow.
        
        Args:
            bug_description: Description of the bug to fix
            error_logs: Optional error logs related to the bug
            
        Returns:
            Workflow instance
        """
        inputs = {"bug_description": bug_description}
        if error_logs:
            inputs["error_logs"] = error_logs
            
        return self.create_workflow("bug_fix", inputs)
    
    def create_review_workflow(self, files_to_review: List[str], review_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a code review workflow.
        
        Args:
            files_to_review: List of files to review
            review_context: Optional context about the changes
            
        Returns:
            Workflow instance
        """
        inputs = {"files_to_review": files_to_review}
        if review_context:
            inputs["review_context"] = review_context
            
        return self.create_workflow("code_review", inputs)
    
    def create_knowledge_reasoning_workflow(self, reasoning_request: str) -> Dict[str, Any]:
        """
        Create a knowledge reasoning workflow.
        
        Args:
            reasoning_request: Description of the reasoning request or topic
            
        Returns:
            Workflow instance
        """
        return self.create_workflow("knowledge_reasoning", {"reasoning_request": reasoning_request})
    
    def _create_code_generation_template(self) -> WorkflowTemplate:
        """Create a code generation workflow template."""
        steps = [
            {
                "id": "requirements_analysis",
                "name": "Requirements Analysis",
                "description": "Analyze the requirements for code generation",
                "inputs": {"requirements": "$workflow.inputs.requirements"},
                "outputs": ["analysis_result"],
                "dependencies": []
            },
            {
                "id": "context_gathering",
                "name": "Context Gathering",
                "description": "Gather context from the codebase",
                "inputs": {
                    "requirements": "$workflow.steps.requirements_analysis.outputs.analysis_result",
                    "codebase_path": "$workflow.inputs.codebase_path"
                },
                "outputs": ["relevant_files", "context"],
                "dependencies": ["requirements_analysis"]
            },
            {
                "id": "code_design",
                "name": "Code Design",
                "description": "Design the code structure",
                "inputs": {
                    "requirements": "$workflow.steps.requirements_analysis.outputs.analysis_result",
                    "context": "$workflow.steps.context_gathering.outputs.context"
                },
                "outputs": ["design"],
                "dependencies": ["context_gathering"]
            },
            {
                "id": "code_generation",
                "name": "Code Generation",
                "description": "Generate code based on design",
                "inputs": {
                    "design": "$workflow.steps.code_design.outputs.design",
                    "context": "$workflow.steps.context_gathering.outputs.context"
                },
                "outputs": ["generated_code"],
                "dependencies": ["code_design"]
            },
            {
                "id": "code_validation",
                "name": "Code Validation",
                "description": "Validate the generated code",
                "inputs": {
                    "generated_code": "$workflow.steps.code_generation.outputs.generated_code",
                    "requirements": "$workflow.steps.requirements_analysis.outputs.analysis_result"
                },
                "outputs": ["validation_result"],
                "dependencies": ["code_generation"]
            },
            {
                "id": "code_integration",
                "name": "Code Integration",
                "description": "Integrate the generated code into the codebase",
                "inputs": {
                    "generated_code": "$workflow.steps.code_generation.outputs.generated_code",
                    "validation_result": "$workflow.steps.code_validation.outputs.validation_result",
                    "codebase_path": "$workflow.inputs.codebase_path"
                },
                "outputs": ["integration_result"],
                "dependencies": ["code_validation"]
            }
        ]
        
        return WorkflowTemplate(
            workflow_type=WorkflowType.CODE_GENERATION,
            name="code_generation",
            description="Generate code based on requirements",
            steps=steps,
            metadata={
                "requires_codebase_access": True,
                "user_interaction_points": ["requirements_analysis", "code_validation"]
            }
        )
    
    def _create_code_review_template(self) -> WorkflowTemplate:
        """Create a code review workflow template."""
        steps = [
            {
                "id": "code_parsing",
                "name": "Code Parsing",
                "description": "Parse and analyze the code to be reviewed",
                "inputs": {"code_path": "$workflow.inputs.code_path"},
                "outputs": ["parsed_code", "code_structure"],
                "dependencies": []
            },
            {
                "id": "static_analysis",
                "name": "Static Analysis",
                "description": "Perform static analysis to identify issues",
                "inputs": {"parsed_code": "$workflow.steps.code_parsing.outputs.parsed_code"},
                "outputs": ["static_analysis_results"],
                "dependencies": ["code_parsing"]
            },
            {
                "id": "style_check",
                "name": "Style Check",
                "description": "Check code style against standards",
                "inputs": {"parsed_code": "$workflow.steps.code_parsing.outputs.parsed_code"},
                "outputs": ["style_check_results"],
                "dependencies": ["code_parsing"]
            },
            {
                "id": "security_analysis",
                "name": "Security Analysis",
                "description": "Analyze code for security vulnerabilities",
                "inputs": {"parsed_code": "$workflow.steps.code_parsing.outputs.parsed_code"},
                "outputs": ["security_analysis_results"],
                "dependencies": ["code_parsing"]
            },
            {
                "id": "review_summary",
                "name": "Review Summary",
                "description": "Summarize all review findings",
                "inputs": {
                    "static_analysis_results": "$workflow.steps.static_analysis.outputs.static_analysis_results",
                    "style_check_results": "$workflow.steps.style_check.outputs.style_check_results",
                    "security_analysis_results": "$workflow.steps.security_analysis.outputs.security_analysis_results"
                },
                "outputs": ["review_summary"],
                "dependencies": ["static_analysis", "style_check", "security_analysis"]
            },
            {
                "id": "improvement_suggestions",
                "name": "Improvement Suggestions",
                "description": "Suggest improvements to the code",
                "inputs": {
                    "review_summary": "$workflow.steps.review_summary.outputs.review_summary",
                    "code_structure": "$workflow.steps.code_parsing.outputs.code_structure"
                },
                "outputs": ["improvement_suggestions"],
                "dependencies": ["review_summary"]
            }
        ]
        
        return WorkflowTemplate(
            workflow_type=WorkflowType.CODE_REVIEW,
            name="code_review",
            description="Review code for issues and suggest improvements",
            steps=steps,
            metadata={
                "requires_codebase_access": True
            }
        )
    
    def _create_bug_fix_template(self) -> WorkflowTemplate:
        """Create a bug fix workflow template."""
        steps = [
            {
                "id": "bug_analysis",
                "name": "Bug Analysis",
                "description": "Analyze the bug and its context",
                "inputs": {
                    "bug_description": "$workflow.inputs.bug_description",
                    "error_logs": "$workflow.inputs.error_logs"
                },
                "outputs": ["bug_analysis_result"],
                "dependencies": []
            },
            {
                "id": "code_search",
                "name": "Code Search",
                "description": "Search for relevant code sections",
                "inputs": {
                    "bug_analysis": "$workflow.steps.bug_analysis.outputs.bug_analysis_result",
                    "codebase_path": "$workflow.inputs.codebase_path"
                },
                "outputs": ["relevant_code_sections"],
                "dependencies": ["bug_analysis"]
            },
            {
                "id": "root_cause_analysis",
                "name": "Root Cause Analysis",
                "description": "Identify the root cause of the bug",
                "inputs": {
                    "bug_analysis": "$workflow.steps.bug_analysis.outputs.bug_analysis_result",
                    "relevant_code_sections": "$workflow.steps.code_search.outputs.relevant_code_sections"
                },
                "outputs": ["root_cause"],
                "dependencies": ["code_search"]
            },
            {
                "id": "fix_design",
                "name": "Fix Design",
                "description": "Design a fix for the bug",
                "inputs": {"root_cause": "$workflow.steps.root_cause_analysis.outputs.root_cause"},
                "outputs": ["fix_design"],
                "dependencies": ["root_cause_analysis"]
            },
            {
                "id": "fix_implementation",
                "name": "Fix Implementation",
                "description": "Implement the fix",
                "inputs": {
                    "fix_design": "$workflow.steps.fix_design.outputs.fix_design",
                    "relevant_code_sections": "$workflow.steps.code_search.outputs.relevant_code_sections"
                },
                "outputs": ["fix_implementation"],
                "dependencies": ["fix_design"]
            },
            {
                "id": "test_case_creation",
                "name": "Test Case Creation",
                "description": "Create test cases to verify the fix",
                "inputs": {
                    "bug_analysis": "$workflow.steps.bug_analysis.outputs.bug_analysis_result",
                    "fix_implementation": "$workflow.steps.fix_implementation.outputs.fix_implementation"
                },
                "outputs": ["test_cases"],
                "dependencies": ["fix_implementation"]
            },
            {
                "id": "fix_validation",
                "name": "Fix Validation",
                "description": "Validate that the fix resolves the bug",
                "inputs": {
                    "fix_implementation": "$workflow.steps.fix_implementation.outputs.fix_implementation",
                    "test_cases": "$workflow.steps.test_case_creation.outputs.test_cases"
                },
                "outputs": ["validation_result"],
                "dependencies": ["test_case_creation"]
            }
        ]
        
        return WorkflowTemplate(
            workflow_type=WorkflowType.BUG_FIX,
            name="bug_fix",
            description="Fix a bug in the codebase",
            steps=steps,
            metadata={
                "requires_codebase_access": True,
                "user_interaction_points": ["bug_analysis", "fix_validation"]
            }
        )
    
    def _create_refactoring_template(self) -> WorkflowTemplate:
        """Create a code refactoring workflow template."""
        steps = [
            {
                "id": "refactoring_goal",
                "name": "Refactoring Goal",
                "description": "Define the goal of the refactoring",
                "inputs": {"refactoring_request": "$workflow.inputs.refactoring_request"},
                "outputs": ["refactoring_goal"],
                "dependencies": []
            },
            {
                "id": "code_analysis",
                "name": "Code Analysis",
                "description": "Analyze the code to be refactored",
                "inputs": {
                    "code_path": "$workflow.inputs.code_path",
                    "refactoring_goal": "$workflow.steps.refactoring_goal.outputs.refactoring_goal"
                },
                "outputs": ["code_analysis_result"],
                "dependencies": ["refactoring_goal"]
            },
            {
                "id": "dependency_analysis",
                "name": "Dependency Analysis",
                "description": "Analyze dependencies of the code to be refactored",
                "inputs": {
                    "code_analysis": "$workflow.steps.code_analysis.outputs.code_analysis_result",
                    "codebase_path": "$workflow.inputs.codebase_path"
                },
                "outputs": ["dependency_analysis_result"],
                "dependencies": ["code_analysis"]
            },
            {
                "id": "refactoring_plan",
                "name": "Refactoring Plan",
                "description": "Create a plan for the refactoring",
                "inputs": {
                    "code_analysis": "$workflow.steps.code_analysis.outputs.code_analysis_result",
                    "dependency_analysis": "$workflow.steps.dependency_analysis.outputs.dependency_analysis_result",
                    "refactoring_goal": "$workflow.steps.refactoring_goal.outputs.refactoring_goal"
                },
                "outputs": ["refactoring_plan"],
                "dependencies": ["dependency_analysis"]
            },
            {
                "id": "refactoring_implementation",
                "name": "Refactoring Implementation",
                "description": "Implement the refactoring",
                "inputs": {"refactoring_plan": "$workflow.steps.refactoring_plan.outputs.refactoring_plan"},
                "outputs": ["refactored_code"],
                "dependencies": ["refactoring_plan"]
            },
            {
                "id": "refactoring_validation",
                "name": "Refactoring Validation",
                "description": "Validate the refactored code",
                "inputs": {
                    "refactored_code": "$workflow.steps.refactoring_implementation.outputs.refactored_code",
                    "original_code_analysis": "$workflow.steps.code_analysis.outputs.code_analysis_result"
                },
                "outputs": ["validation_result"],
                "dependencies": ["refactoring_implementation"]
            }
        ]
        
        return WorkflowTemplate(
            workflow_type=WorkflowType.REFACTORING,
            name="code_refactoring",
            description="Refactor code to improve quality",
            steps=steps,
            metadata={
                "requires_codebase_access": True,
                "user_interaction_points": ["refactoring_goal", "refactoring_validation"]
            }
        )
    
    def _create_documentation_template(self) -> WorkflowTemplate:
        """Create a documentation workflow template."""
        steps = [
            {
                "id": "documentation_requirements",
                "name": "Documentation Requirements",
                "description": "Define documentation requirements",
                "inputs": {"documentation_request": "$workflow.inputs.documentation_request"},
                "outputs": ["documentation_requirements"],
                "dependencies": []
            },
            {
                "id": "code_analysis",
                "name": "Code Analysis",
                "description": "Analyze code for documentation",
                "inputs": {"code_path": "$workflow.inputs.code_path"},
                "outputs": ["code_analysis_result"],
                "dependencies": []
            },
            {
                "id": "structure_planning",
                "name": "Structure Planning",
                "description": "Plan the structure of the documentation",
                "inputs": {
                    "documentation_requirements": "$workflow.steps.documentation_requirements.outputs.documentation_requirements",
                    "code_analysis": "$workflow.steps.code_analysis.outputs.code_analysis_result"
                },
                "outputs": ["documentation_structure"],
                "dependencies": ["documentation_requirements", "code_analysis"]
            },
            {
                "id": "content_generation",
                "name": "Content Generation",
                "description": "Generate documentation content",
                "inputs": {
                    "documentation_structure": "$workflow.steps.structure_planning.outputs.documentation_structure",
                    "code_analysis": "$workflow.steps.code_analysis.outputs.code_analysis_result"
                },
                "outputs": ["documentation_content"],
                "dependencies": ["structure_planning"]
            },
            {
                "id": "example_creation",
                "name": "Example Creation",
                "description": "Create examples for the documentation",
                "inputs": {
                    "documentation_content": "$workflow.steps.content_generation.outputs.documentation_content",
                    "code_analysis": "$workflow.steps.code_analysis.outputs.code_analysis_result"
                },
                "outputs": ["examples"],
                "dependencies": ["content_generation"]
            },
            {
                "id": "documentation_integration",
                "name": "Documentation Integration",
                "description": "Integrate the documentation",
                "inputs": {
                    "documentation_content": "$workflow.steps.content_generation.outputs.documentation_content",
                    "examples": "$workflow.steps.example_creation.outputs.examples",
                    "code_path": "$workflow.inputs.code_path"
                },
                "outputs": ["integrated_documentation"],
                "dependencies": ["example_creation"]
            }
        ]
        
        return WorkflowTemplate(
            workflow_type=WorkflowType.DOCUMENTATION,
            name="documentation",
            description="Generate documentation for code",
            steps=steps,
            metadata={
                "requires_codebase_access": True
            }
        )
    
    def _create_learning_template(self) -> WorkflowTemplate:
        """Create a learning workflow template."""
        steps = [
            {
                "id": "learning_goal",
                "name": "Learning Goal",
                "description": "Define the learning goal",
                "inputs": {"learning_request": "$workflow.inputs.learning_request"},
                "outputs": ["learning_goal"],
                "dependencies": []
            },
            {
                "id": "knowledge_assessment",
                "name": "Knowledge Assessment",
                "description": "Assess current knowledge level",
                "inputs": {"learning_goal": "$workflow.steps.learning_goal.outputs.learning_goal"},
                "outputs": ["knowledge_assessment"],
                "dependencies": ["learning_goal"]
            },
            {
                "id": "resource_gathering",
                "name": "Resource Gathering",
                "description": "Gather learning resources",
                "inputs": {
                    "learning_goal": "$workflow.steps.learning_goal.outputs.learning_goal",
                    "knowledge_assessment": "$workflow.steps.knowledge_assessment.outputs.knowledge_assessment"
                },
                "outputs": ["resources"],
                "dependencies": ["knowledge_assessment"]
            },
            {
                "id": "learning_path",
                "name": "Learning Path",
                "description": "Create a learning path",
                "inputs": {
                    "learning_goal": "$workflow.steps.learning_goal.outputs.learning_goal",
                    "knowledge_assessment": "$workflow.steps.knowledge_assessment.outputs.knowledge_assessment",
                    "resources": "$workflow.steps.resource_gathering.outputs.resources"
                },
                "outputs": ["learning_path"],
                "dependencies": ["resource_gathering"]
            },
            {
                "id": "example_creation",
                "name": "Example Creation",
                "description": "Create examples to illustrate concepts",
                "inputs": {
                    "learning_path": "$workflow.steps.learning_path.outputs.learning_path",
                    "resources": "$workflow.steps.resource_gathering.outputs.resources"
                },
                "outputs": ["examples"],
                "dependencies": ["learning_path"]
            },
            {
                "id": "practice_exercise",
                "name": "Practice Exercise",
                "description": "Create practice exercises",
                "inputs": {
                    "learning_path": "$workflow.steps.learning_path.outputs.learning_path",
                    "examples": "$workflow.steps.example_creation.outputs.examples"
                },
                "outputs": ["exercises"],
                "dependencies": ["example_creation"]
            },
            {
                "id": "learning_summary",
                "name": "Learning Summary",
                "description": "Summarize the learning path",
                "inputs": {
                    "learning_path": "$workflow.steps.learning_path.outputs.learning_path",
                    "examples": "$workflow.steps.example_creation.outputs.examples",
                    "exercises": "$workflow.steps.practice_exercise.outputs.exercises"
                },
                "outputs": ["learning_summary"],
                "dependencies": ["practice_exercise"]
            }
        ]
        
        return WorkflowTemplate(
            workflow_type=WorkflowType.LEARNING,
            name="learning",
            description="Create a learning path for a topic",
            steps=steps,
            metadata={
                "requires_codebase_access": False,
                "user_interaction_points": ["learning_goal", "practice_exercise"]
            }
        )

    def _create_knowledge_reasoning_template(self) -> WorkflowTemplate:
        """Create a knowledge reasoning workflow template using the knowledge graph and think tool."""
        steps = [
            {
                "id": "knowledge_query",
                "name": "Knowledge Query",
                "description": "Define the knowledge reasoning query or topic",
                "inputs": {"reasoning_request": "$workflow.inputs.reasoning_request"},
                "outputs": ["knowledge_query"],
                "dependencies": []
            },
            {
                "id": "context_retrieval",
                "name": "Context Retrieval",
                "description": "Retrieve relevant knowledge context from the knowledge graph",
                "inputs": {"query": "$workflow.steps.knowledge_query.outputs.knowledge_query"},
                "outputs": ["knowledge_context"],
                "dependencies": ["knowledge_query"]
            },
            {
                "id": "structured_reasoning",
                "name": "Structured Reasoning",
                "description": "Apply structured reasoning to the query and context",
                "inputs": {
                    "query": "$workflow.steps.knowledge_query.outputs.knowledge_query",
                    "context": "$workflow.steps.context_retrieval.outputs.knowledge_context"
                },
                "outputs": ["reasoning_output"],
                "dependencies": ["context_retrieval"]
            },
            {
                "id": "reflection",
                "name": "Reflection",
                "description": "Reflect on the reasoning output to refine understanding",
                "inputs": {"reasoning": "$workflow.steps.structured_reasoning.outputs.reasoning_output"},
                "outputs": ["reflection_output"],
                "dependencies": ["structured_reasoning"]
            },
            {
                "id": "knowledge_capture",
                "name": "Knowledge Capture",
                "description": "Capture new knowledge and insights in the knowledge graph",
                "inputs": {
                    "reasoning": "$workflow.steps.structured_reasoning.outputs.reasoning_output",
                    "reflection": "$workflow.steps.reflection.outputs.reflection_output"
                },
                "outputs": ["captured_knowledge"],
                "dependencies": ["reflection"]
            },
            {
                "id": "related_entity_exploration",
                "name": "Related Entity Exploration",
                "description": "Explore related entities to enrich understanding",
                "inputs": {"captured_knowledge": "$workflow.steps.knowledge_capture.outputs.captured_knowledge"},
                "outputs": ["related_entities"],
                "dependencies": ["knowledge_capture"]
            },
            {
                "id": "reasoning_summary",
                "name": "Reasoning Summary",
                "description": "Summarize the reasoning process and findings",
                "inputs": {
                    "reasoning": "$workflow.steps.structured_reasoning.outputs.reasoning_output",
                    "reflection": "$workflow.steps.reflection.outputs.reflection_output",
                    "related_entities": "$workflow.steps.related_entity_exploration.outputs.related_entities"
                },
                "outputs": ["reasoning_summary"],
                "dependencies": ["related_entity_exploration"]
            }
        ]
        
        return WorkflowTemplate(
            workflow_type=WorkflowType.KNOWLEDGE_REASONING,
            name="knowledge_reasoning",
            description="Conduct structured reasoning with knowledge graph integration",
            steps=steps,
            metadata={
                "requires_knowledge_graph": True,
                "user_interaction_points": ["knowledge_query", "structured_reasoning", "reasoning_summary"]
            }
        ) 