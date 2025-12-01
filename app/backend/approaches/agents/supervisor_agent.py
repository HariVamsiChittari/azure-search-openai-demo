"""
Supervisor Agent: Main orchestration agent that routes queries to sub-agents.

This module provides a factory function to create the SupervisorAgent,
which manages and delegates queries to specialized sub-agents
(DocumentSearchAgent and GraphTraversalAgent).
"""

import logging

from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureOpenAIChatClient

from approaches.tools import SupervisorTools

logger = logging.getLogger(__name__)

# Agent instructions - defines how the SupervisorAgent routes queries
SUPERVISOR_AGENT_INSTRUCTIONS = """You are a supervisor agent managing two specialist agents: DocumentSearchAgent and GraphTraversalAgent. 
You MUST use at least one of these agents for every query - never answer directly without calling an agent.

Your available specialist agents are:
1. **delegate_to_search_agent**: Use for searching, finding information, looking up facts, or retrieving documents. 
Keywords: search, find, what is, tell me about, look up, get information.
2. **delegate_to_traversal_agent**: Use for exploring relationships, connections, hierarchies, comparisons, or navigation. 
Keywords: related, connected, compare, difference, hierarchy, relationship, how does X relate to Y, links between.

IMPORTANT ROUTING RULES:
- For questions about WHAT something is → use delegate_to_search_agent
- For questions about HOW things RELATE or COMPARE → use delegate_to_traversal_agent
- For questions that need BOTH search AND relationships → call BOTH agents
- ALWAYS call at least one agent tool before providing your final answer

CRITICAL CITATION RULES:
- DocumentSearchAgent responses contain citations in format [filename.pdf-pagenumber] like [role_library.pdf-17].
- You MUST preserve these EXACT citations in your final answer - do NOT modify the format.
- Keep each citation in its own square brackets: [role_library.pdf-17] [role_library.pdf-23]
- Do NOT combine citations or change format to things like 'pages 17, 23'.
- Example: 'Product Managers oversee roadmaps [role_library.pdf-17] and budgets [role_library.pdf-23].'

After receiving responses from agents, synthesize them into a helpful final answer while preserving all citations exactly as provided."""


def create_supervisor_agent(
    chat_client: AzureOpenAIChatClient,
    supervisor_tools: SupervisorTools,
    include_traversal_agent: bool = False,
) -> ChatAgent:
    """
    Factory function to create a SupervisorAgent.

    The SupervisorAgent is the main orchestration agent that routes user queries
    to appropriate sub-agents. It analyzes the intent of each query and delegates
    to either DocumentSearchAgent or GraphTraversalAgent (or both).

    Args:
        chat_client: AzureOpenAIChatClient for LLM communication
        supervisor_tools: SupervisorTools instance with delegation methods
        include_traversal_agent: Whether to include GraphTraversalAgent delegation
                                 (currently disabled by default for stability)

    Returns:
        Configured ChatAgent for query routing and orchestration
    """
    logger.info("Creating SupervisorAgent with delegation tools")

    # Create ai_function decorated delegation tools
    delegate_to_search_agent_tool = ai_function(
        description="Delegate query to DocumentSearchAgent for document search and retrieval. Use for finding specific information, searching documents, or looking up facts."
    )(supervisor_tools.delegate_to_search_agent)

    delegate_to_traversal_agent_tool = ai_function(
        description="Delegate query to GraphTraversalAgent for relationship exploration. Use for exploring connections, comparing items, understanding relationships."
    )(supervisor_tools.delegate_to_traversal_agent)

    # Build tools list based on configuration
    tools = [delegate_to_search_agent_tool]
    if include_traversal_agent:
        tools.append(delegate_to_traversal_agent_tool)
        logger.info("Including GraphTraversalAgent delegation tool")

    # Create and return the agent
    agent = ChatAgent(
        name="SupervisorAgent",
        chat_client=chat_client,
        instructions=SUPERVISOR_AGENT_INSTRUCTIONS,
        tools=tools,
    )

    logger.info(f"SupervisorAgent created with {len(tools)} delegation tool(s)")
    return agent
