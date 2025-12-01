"""
Graph Traversal Agent: Specializes in relationship exploration and graph navigation.

This module provides a factory function to create the GraphTraversalAgent,
which handles traversal/navigation queries using the knowledge graph
to explore connections between documents, concepts, and entities.
"""

import logging

from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureOpenAIChatClient

from approaches.tools import TraversalTools

logger = logging.getLogger(__name__)

# Agent instructions - defines how the GraphTraversalAgent behaves
GRAPH_TRAVERSAL_AGENT_INSTRUCTIONS = """You are a traversal specialist agent that helps users navigate relationships and hierarchies in the knowledge graph. 
Use the traversal tool to explore connections between documents, concepts, and entities. 
Provide clear explanations of the relationships and connections you discover. 
Help users understand how different pieces of information are related.

When exploring relationships:
- Identify the key entities or concepts in the query
- Use the traversal tool to find connections
- Explain the relationship path clearly
- Highlight any hierarchical structures discovered
- Summarize the overall relationship landscape"""


def create_graph_traversal_agent(
    chat_client: AzureOpenAIChatClient,
    traversal_tools: TraversalTools,
) -> ChatAgent:
    """
    Factory function to create a GraphTraversalAgent.

    The GraphTraversalAgent specializes in relationship exploration and
    graph navigation queries. It uses the traverse_knowledge_graph tool
    to explore connections between documents, concepts, and entities.

    Args:
        chat_client: AzureOpenAIChatClient for LLM communication
        traversal_tools: TraversalTools instance with traverse_knowledge_graph method

    Returns:
        Configured ChatAgent for graph traversal operations
    """
    logger.info("Creating GraphTraversalAgent with traverse_knowledge_graph tool")

    # Create ai_function decorated tool from TraversalTools
    traverse_knowledge_graph_tool = ai_function(
        description="Traverse the knowledge graph to find relationships, connections, and hierarchies between documents and concepts."
    )(traversal_tools.traverse_knowledge_graph)

    # Create and return the agent
    agent = ChatAgent(
        name="GraphTraversalAgent",
        chat_client=chat_client,
        instructions=GRAPH_TRAVERSAL_AGENT_INSTRUCTIONS,
        tools=[traverse_knowledge_graph_tool],
    )

    logger.info("GraphTraversalAgent created successfully")
    return agent
