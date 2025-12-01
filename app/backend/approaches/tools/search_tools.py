"""
Search Tools for Document Search Agent.

This module provides tool classes for the DocumentSearchAgent using the @ai_function decorator pattern.
Each tool is a method on a class that maintains state and context for tool execution.

Usage:
    # Create context holder
    context_holder = SearchContextHolder(
        chat_approach=chat_approach,
        messages=[],
        overrides={},
        auth_claims={},
        thoughts=[],
        data_points=DataPoints(text=[], images=[], citations=[]),
    )
    
    # Create search tools with context
    search_tools = SearchTools(context_holder=context_holder)
    
    # Use with agent
    agent = ChatAgent(
        name="DocumentSearchAgent",
        tools=[search_tools.search_knowledge_base],  # Method reference
    )
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, Union

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import Field

from approaches.approach import Approach, DataPoints, ThoughtStep

logger = logging.getLogger(__name__)


@dataclass
class SearchContextHolder:
    """
    Context holder for SearchTools - manages state for tool execution.
    
    This dataclass holds all the context that search tools need to execute,
    including the chat approach, current messages, overrides, and accumulated results.
    
    The context holder is passed to SearchTools at initialization and updated
    before each request via direct attribute assignment.
    
    Attributes:
        chat_approach: The RAG approach to reuse for full search pipeline
        messages: Current conversation messages
        overrides: Request overrides (settings)
        auth_claims: User authentication claims
        thoughts: Shared thoughts list to append to
        data_points: Shared data_points to accumulate to
        search_client: Optional Azure Search client for direct queries (used by search_hybrid_simple)
        openai_client: Optional OpenAI client for embeddings (used by search_hybrid_simple)
        embedding_deployment: Optional embedding model deployment name (used by search_hybrid_simple)
    """
    chat_approach: Approach
    messages: list = field(default_factory=list)
    overrides: dict = field(default_factory=dict)
    auth_claims: dict = field(default_factory=dict)
    thoughts: list = field(default_factory=list)
    data_points: DataPoints = field(default_factory=lambda: DataPoints(text=[], images=[], citations=[]))
    # Optional fields for search_hybrid_simple (direct search without RAG pipeline)
    search_client: Optional[SearchClient] = None
    openai_client: Optional[AsyncAzureOpenAI] = None  #Union[AsyncAzureOpenAI, AsyncOpenAI]
    embedding_deployment: Optional[str] = None


class SearchTools:
    """
    Tool class for document search operations.
    
    This class encapsulates search-related tools for the DocumentSearchAgent.
    It uses a SearchContextHolder to maintain state for context (messages, 
    overrides, auth_claims) and accumulate data_points and thoughts during execution.
    
    Attributes:
        context: SearchContextHolder with all dependencies and state
    """

    def __init__(self, context_holder: SearchContextHolder) -> None:
        """
        Initialize SearchTools with a context holder.
        
        Args:
            context_holder: SearchContextHolder containing chat_approach and state
        """
        self.context = context_holder

    async def search_knowledge_base(
        self,
        search_query: Annotated[
            str, Field(description="The search query to find relevant information in the knowledge base.")
        ],
    ) -> str:
        """
        Search the knowledge base using the full RAG pipeline.
        
        This tool leverages the sophisticated 3-step search from chat_approach:
        1. Query rewriting (LLM-optimized search query)
        2. Hybrid search (text + vector + semantic ranker)
        3. Source extraction (text, images, citations)
        
        Args:
            search_query: The search query string
            
        Returns:
            JSON string containing search results with sources and citations
        """
        logger.info(f"SearchTools.search_knowledge_base: query='{search_query}'")

        try:
            # Build messages list with the search query as the user message
            messages = list(self.context.messages) if self.context.messages else []

            # Replace or append the search query as the latest user message
            if messages and messages[-1].get("role") == "user":
                messages[-1] = {"role": "user", "content": search_query}
            else:
                messages.append({"role": "user", "content": search_query})

            # Get overrides and auth_claims from context
            overrides = dict(self.context.overrides) if self.context.overrides else {}
            auth_claims = dict(self.context.auth_claims) if self.context.auth_claims else {}

            # Call the chat_approach's run_search_approach method
            extra_info = await self.context.chat_approach.run_search_approach(messages, overrides, auth_claims)

            # Append thoughts to shared list for UI visibility
            if extra_info.thoughts:
                for thought in extra_info.thoughts:
                    self.context.thoughts.append(thought)
                logger.info(f"SearchTools: Added {len(extra_info.thoughts)} thoughts")

            # Accumulate data_points for final response
            if extra_info.data_points:
                if extra_info.data_points.text:
                    self.context.data_points.text = (
                        self.context.data_points.text or []
                    ) + extra_info.data_points.text
                if extra_info.data_points.images:
                    self.context.data_points.images = (
                        self.context.data_points.images or []
                    ) + extra_info.data_points.images
                if extra_info.data_points.citations:
                    self.context.data_points.citations = (
                        self.context.data_points.citations or []
                    ) + extra_info.data_points.citations
                logger.info(
                    f"SearchTools: Accumulated data_points - "
                    f"text: {len(self.context.data_points.text or [])}, "
                    f"citations: {len(self.context.data_points.citations or [])}"
                )

            # Build response
            results = {
                "query": search_query,
                "sources": extra_info.data_points.text or [],
                "citations": extra_info.data_points.citations or [],
                "images": extra_info.data_points.images or [],
                "result_count": len(extra_info.data_points.text or []),
            }

            logger.info(f"SearchTools: Retrieved {results['result_count']} results")
            return json.dumps(results, indent=2)

        except Exception as e:
            logger.error(f"SearchTools.search_knowledge_base error: {e}", exc_info=True)
            return json.dumps({"error": str(e), "query": search_query})

    async def search_hybrid_simple(
        self,
        search_query: Annotated[
            str, Field(description="The search query to find relevant information in the knowledge base.")
        ],
    ) -> str:
        """
        Simple hybrid search (keyword + vector) without query rewriting.
        
        This is a faster, direct implementation that bypasses the full RAG pipeline.
        Use when speed is more important than search quality optimization.
        
        Args:
            search_query: The search query string
            
        Returns:
            JSON string containing search results from Azure AI Search
        """
        logger.info(f"SearchTools.search_hybrid_simple: query='{search_query}'")

        try:
            # Check if required dependencies are available
            if not self.context.openai_client or not self.context.search_client:
                return json.dumps({
                    "error": "search_hybrid_simple requires openai_client and search_client in context",
                    "query": search_query
                })
            
            if not self.context.embedding_deployment:
                return json.dumps({
                    "error": "search_hybrid_simple requires embedding_deployment in context",
                    "query": search_query
                })

            # Generate embeddings
            embedding_response = await self.context.openai_client.embeddings.create(
                model=self.context.embedding_deployment,
                input=search_query,
            )
            embedding = embedding_response.data[0].embedding

            # Create vector query
            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=5,
                fields="embedding3",
            )

            # Execute hybrid search
            results = await self.context.search_client.search(
                search_text=search_query,
                vector_queries=[vector_query],
                select=["content", "sourcepage", "category"],
                top=5,
            )

            # Collect results
            output = []
            async for r in results:
                output.append({
                    "content": r.get("content", ""),
                    "sourcepage": r.get("sourcepage", ""),
                    "category": r.get("category", ""),
                })

            logger.info(f"SearchTools.search_hybrid_simple: Retrieved {len(output)} results")
            return json.dumps(output, indent=2)

        except Exception as e:
            logger.error(f"SearchTools.search_hybrid_simple error: {e}")
            return json.dumps({"error": str(e), "query": search_query})
