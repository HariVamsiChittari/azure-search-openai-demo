"""
Supervisor Tools for routing queries to sub-agents.

This module provides tool classes for the SupervisorAgent using the @ai_function decorator pattern.
Each tool delegates work to a specialized sub-agent.

Usage:
    supervisor_tools = SupervisorTools(
        document_search_agent=document_search_agent,
        graph_traversal_agent=graph_traversal_agent,
    )
    
    # Use with supervisor agent
    supervisor = ChatAgent(
        name="SupervisorAgent",
        tools=[
            supervisor_tools.delegate_to_search_agent,
            supervisor_tools.delegate_to_traversal_agent,
        ],
    )
"""

import logging
from typing import Annotated, Any

from agent_framework import ChatAgent
from pydantic import Field

from approaches.approach import ThoughtStep

logger = logging.getLogger(__name__)


class SupervisorTools:
    """
    Tool class for supervisor agent routing operations.
    
    This class encapsulates delegation tools that route queries to specialized
    sub-agents (DocumentSearchAgent, GraphTraversalAgent).
    
    Each delegation tool:
    1. Records a ThoughtStep for UI visibility
    2. Invokes the appropriate sub-agent
    3. Records the sub-agent's response as a ThoughtStep
    4. Returns the response for synthesis
    
    Attributes:
        document_search_agent: Agent specialized in document search
        graph_traversal_agent: Agent specialized in graph traversal
    """

    def __init__(
        self,
        document_search_agent: ChatAgent,
        graph_traversal_agent: ChatAgent,
    ) -> None:
        """
        Initialize SupervisorTools with sub-agents.
        
        Args:
            document_search_agent: ChatAgent for document search operations
            graph_traversal_agent: ChatAgent for graph traversal operations
        """
        self.document_search_agent = document_search_agent
        self.graph_traversal_agent = graph_traversal_agent
        
        # Accumulated thoughts - set before each supervisor run
        self._current_thoughts: list[ThoughtStep] = []

    def set_context(
        self,
        thoughts: list[ThoughtStep],
    ) -> None:
        """
        Set the context for the current request.
        
        Args:
            thoughts: Shared thoughts list to append to
        """
        self._current_thoughts = thoughts

    def reset_context(self) -> None:
        """Reset context after supervisor run completes."""
        # Note: thoughts is a shared reference, don't reset it
        pass

    async def delegate_to_search_agent(
        self,
        query: Annotated[
            str,
            Field(
                description="A search query to find specific information, documents, or facts in the knowledge base."
            ),
        ],
    ) -> str:
        """
        Delegate query to DocumentSearchAgent for search and retrieval.
        
        Use this when the user wants to:
        - Find specific information
        - Search for documents
        - Look up facts or definitions
        
        Examples:
        - "What is X?"
        - "Find information about Y"
        - "Tell me about Z"
        - "What are the benefits?"
        
        Args:
            query: The search query to delegate
            
        Returns:
            The DocumentSearchAgent's response with citations
        """
        logger.info(f"SupervisorTools.delegate_to_search_agent: query='{query}'")

        # Record delegation as ThoughtStep
        self._current_thoughts.append(
            ThoughtStep(
                title="DocumentSearchAgent Invoked",
                description=f"Delegating query to DocumentSearchAgent: '{query}'",
                props={
                    "agent": "DocumentSearchAgent",
                    "query": query,
                    "tool": "search_knowledge_base",
                },
            )
        )
        logger.info(f"SupervisorTools: Added delegation ThoughtStep, total: {len(self._current_thoughts)}")

        # Run the sub-agent
        response = await self.document_search_agent.run(query)
        response_text = response.text

        logger.info(f"SupervisorTools: DocumentSearchAgent response length: {len(response_text)}")

        # Record results as ThoughtStep
        self._current_thoughts.append(
            ThoughtStep(
                title="DocumentSearchAgent Results",
                description=response_text[:500] + "..." if len(response_text) > 500 else response_text,
                props={
                    "agent": "DocumentSearchAgent",
                    "response_length": len(response_text),
                    "status": "completed",
                },
            )
        )
        logger.info(f"SupervisorTools: Added results ThoughtStep, total: {len(self._current_thoughts)}")

        return response_text

    async def delegate_to_traversal_agent(
        self,
        query: Annotated[
            str,
            Field(
                description="A query about relationships, connections, comparisons, or navigation through related information."
            ),
        ],
    ) -> str:
        """
        Delegate query to GraphTraversalAgent for relationship exploration.
        
        Use this when the user wants to:
        - Explore connections between topics
        - Compare items or concepts
        - Understand relationships
        - Navigate through related content
        
        Examples:
        - "How are X and Y related?"
        - "Compare A and B"
        - "What connects these topics?"
        - "Show me the relationship between..."
        
        Args:
            query: The traversal query to delegate
            
        Returns:
            The GraphTraversalAgent's response
        """
        logger.info(f"SupervisorTools.delegate_to_traversal_agent: query='{query}'")

        # Record delegation as ThoughtStep
        self._current_thoughts.append(
            ThoughtStep(
                title="GraphTraversalAgent Invoked",
                description=f"Delegating query to GraphTraversalAgent: '{query}'",
                props={
                    "agent": "GraphTraversalAgent",
                    "query": query,
                    "tool": "traverse_knowledge_graph",
                },
            )
        )
        logger.info(f"SupervisorTools: Added delegation ThoughtStep, total: {len(self._current_thoughts)}")

        # Run the sub-agent
        response = await self.graph_traversal_agent.run(query)
        response_text = response.text

        logger.info(f"SupervisorTools: GraphTraversalAgent response length: {len(response_text)}")

        # Record results as ThoughtStep
        self._current_thoughts.append(
            ThoughtStep(
                title="GraphTraversalAgent Results",
                description=response_text[:500] + "..." if len(response_text) > 500 else response_text,
                props={
                    "agent": "GraphTraversalAgent",
                    "response_length": len(response_text),
                    "status": "completed",
                },
            )
        )
        logger.info(f"SupervisorTools: Added results ThoughtStep, total: {len(self._current_thoughts)}")

        return response_text
