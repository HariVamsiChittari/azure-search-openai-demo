"""
Agent Framework module for hierarchical agent orchestration.

This module provides agent implementations using the Microsoft Agent Framework:
- AgentOrchestrator: Main orchestrator that coordinates all agents
- create_document_search_agent: Factory for DocumentSearchAgent
- create_graph_traversal_agent: Factory for GraphTraversalAgent
- create_supervisor_agent: Factory for SupervisorAgent
"""

from approaches.agents.agent_orchestrator import AgentOrchestrator
from approaches.agents.document_search_agent import (
    DOCUMENT_SEARCH_AGENT_INSTRUCTIONS,
    create_document_search_agent,
)
from approaches.agents.graph_traversal_agent import (
    GRAPH_TRAVERSAL_AGENT_INSTRUCTIONS,
    create_graph_traversal_agent,
)
from approaches.agents.supervisor_agent import (
    SUPERVISOR_AGENT_INSTRUCTIONS,
    create_supervisor_agent,
)

__all__ = [
    "AgentOrchestrator",
    "create_document_search_agent",
    "create_graph_traversal_agent",
    "create_supervisor_agent",
    "DOCUMENT_SEARCH_AGENT_INSTRUCTIONS",
    "GRAPH_TRAVERSAL_AGENT_INSTRUCTIONS",
    "SUPERVISOR_AGENT_INSTRUCTIONS",
]
