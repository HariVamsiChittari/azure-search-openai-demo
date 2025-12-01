"""
Document Search Agent: Specializes in search queries using Azure AI Search.

This module provides a factory function to create the DocumentSearchAgent,
which handles search-related queries using the full RAG pipeline with
query rewriting, hybrid search, and citation extraction.
"""

import logging

from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureOpenAIChatClient

from approaches.tools import SearchTools

logger = logging.getLogger(__name__)

# Agent instructions - defines how the DocumentSearchAgent behaves
DOCUMENT_SEARCH_AGENT_INSTRUCTIONS = """You are a search specialist agent that helps users find information in the knowledge base. 
Use the search_knowledge_base tool to search documents and return relevant results. 
Always provide clear, well-structured responses based on the search results. 

CRITICAL CITATION RULES:
- You MUST cite sources using the EXACT citation strings from the 'citations' field in the search results.
- Citations are in format like: 'filename.pdf-pagenumber' (e.g., 'role_library.pdf-17', 'employee_handbook.pdf-5').
- Put each citation in square brackets immediately after the relevant fact: [role_library.pdf-17]
- If multiple pages are relevant, cite each separately: [role_library.pdf-17] [role_library.pdf-23]
- Do NOT invent citation formats like 'pages 17, 23' - use the EXACT strings from citations field.
- Do NOT combine multiple citations into one bracket.
- Example: 'Product Managers oversee roadmaps [role_library.pdf-17] and manage budgets [role_library.pdf-23].'"""


def create_document_search_agent(
    chat_client: AzureOpenAIChatClient,
    search_tools: SearchTools,
) -> ChatAgent:
    """
    Factory function to create a DocumentSearchAgent.

    The DocumentSearchAgent specializes in search-related queries using Azure AI Search.
    It uses the search_knowledge_base tool which leverages the full RAG pipeline:
    1. Query rewriting (LLM-optimized search query)
    2. Hybrid search (text + vector + semantic ranker)
    3. Source extraction (text, images, citations)

    Args:
        chat_client: AzureOpenAIChatClient for LLM communication
        search_tools: SearchTools instance with search_hybrid_simple method
    
    Returns:
        Configured ChatAgent for document search operations
    """
    logger.info("Creating DocumentSearchAgent with search_hybrid_simple tool")

    # Create ai_function decorated tool from SearchTools
    search_knowledge_base_tool = ai_function(
        description="Search the knowledge base for relevant information using full RAG pipeline with query rewriting, hybrid search, and citation extraction."
    )(search_tools.search_knowledge_base)

    # Create and return the agent
    agent = ChatAgent(
        name="DocumentSearchAgent",
        chat_client=chat_client,
        instructions=DOCUMENT_SEARCH_AGENT_INSTRUCTIONS,
        tools= [search_knowledge_base_tool] 
    )

    logger.info("DocumentSearchAgent created successfully")
    return agent
