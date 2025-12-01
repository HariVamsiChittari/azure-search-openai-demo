"""
Agent Orchestrator: Routes user queries to appropriate sub-agents using agent_framework.

This module implements a hierarchical agent architecture with:
1. DocumentSearchAgent: Handles search-related queries using Azure AI Search
2. GraphTraversalAgent: Handles traversal/navigation queries using TraversalTool
3. SupervisorAgent: Main agent that routes queries to appropriate sub-agents
4. Maintains backward compatibility with existing Approach interface
5. Uses PromptManager to render prompts and display them in the UI through thoughts

Tool Pattern:
- Uses @ai_function decorator pattern from agent-framework
- Tool classes: SearchTools, TraversalTools, SupervisorTools
- Context holders manage state for tool execution

Agent Factory Pattern:
- Each agent is created via factory functions in separate modules
- document_search_agent.py: create_document_search_agent()
- graph_traversal_agent.py: create_graph_traversal_agent()
- supervisor_agent.py: create_supervisor_agent()
"""

import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Optional, Union

from agent_framework.azure import AzureOpenAIChatClient
from azure.search.documents.aio import SearchClient
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from approaches.approach import Approach, DataPoints, ExtraInfo, ThoughtStep
from approaches.promptmanager import PromptManager
from approaches.tools import SearchContextHolder, SearchTools, SupervisorTools, TraversalTools

# Import agent factory functions
from approaches.agents.document_search_agent import create_document_search_agent
from approaches.agents.graph_traversal_agent import create_graph_traversal_agent
from approaches.agents.supervisor_agent import create_supervisor_agent

# Standard logger for this module
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# AgentOrchestrator Class
# ----------------------------------------------------------------------------------


class AgentOrchestrator(Approach):
    """
    An agent orchestrator that uses a hierarchical agent architecture.

    Architecture:
    - DocumentSearchAgent: Specializes in search queries using Azure AI Search (Hybrid Search)
    - GraphTraversalAgent: Specializes in graph traversal using TraversalTool
    - SupervisorAgent: Routes user queries to appropriate sub-agents
    
    Tool Pattern:
    - Uses @ai_function decorator pattern from agent-framework
    - SearchTools: Provides search_knowledge_base and search_hybrid_simple tools
    - TraversalTools: Provides traverse_knowledge_graph and related tools
    - SupervisorTools: Provides delegation tools to route to sub-agents
    """

    def __init__(
        self,
        chat_approach: Approach,
        ask_approach: Approach,
        openai_endpoint: str,
        openai_deployment: str,
        prompt_manager: PromptManager,
        search_client: SearchClient,
        openai_client: Union[AsyncAzureOpenAI, AsyncOpenAI],
        embedding_deployment: str,
        credential_provider: Optional[Any] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the agent orchestrator with hierarchical agents.

        Args:
            chat_approach: The multi-turn chat approach (for backward compatibility)
            ask_approach: The single-turn Q&A approach (for backward compatibility)
            openai_endpoint: Azure OpenAI endpoint URL
            openai_deployment: Azure OpenAI deployment name for chat
            prompt_manager: PromptManager for loading and rendering prompts
            search_client: Azure SearchClient instance (passed from app.py)
            openai_client: AsyncAzureOpenAI or AsyncOpenAI client for embeddings (passed from app.py)
            embedding_deployment: Azure OpenAI deployment name for embeddings
            credential_provider: Optional Azure credential for authentication
            api_key: Optional API key for authentication (alternative to credential)
        """
        # Store legacy approaches for potential fallback
        self.chat_approach = chat_approach
        self.ask_approach = ask_approach
        self.openai_endpoint = openai_endpoint
        self.openai_deployment = openai_deployment

        # Store injected clients (from app.py)
        self.search_client = search_client
        self.openai_client = openai_client
        self.embedding_deployment = embedding_deployment

        # Initialize prompt manager and load prompts from agents subfolder
        self.prompt_manager = prompt_manager
        self.supervisor_agent_prompt = self.prompt_manager.load_prompt("agents/supervisor_agent.prompty")
        self.search_agent_prompt = self.prompt_manager.load_prompt("agents/document_search_agent.prompty")
        self.traversal_agent_prompt = self.prompt_manager.load_prompt("agents/graph_traversal_agent.prompty")

        # Initialize the Azure OpenAI client for agent_framework
        if api_key:
            self.agent_client = AzureOpenAIChatClient(
                endpoint=openai_endpoint,
                api_key=api_key,
                deployment_name=openai_deployment,
            )
        elif credential_provider:
            # Pass the Azure credential directly - agent-framework will handle token acquisition
            self.agent_client = AzureOpenAIChatClient(
                endpoint=openai_endpoint,
                credential=credential_provider,
                deployment_name=openai_deployment,
            )
        else:
            raise ValueError("Either api_key or credential_provider must be provided")

        # ----------------------------------------------------------------------------------
        # Initialize Context Holders and Tool Classes (using @ai_function pattern)
        # ----------------------------------------------------------------------------------

        # Store current context for sub-agents
        self._current_session_state = None
        self._current_context: dict[str, Any] = {}
        # Track thoughts during agent execution (populated during run)
        self._current_thoughts: list[ThoughtStep] = []
        # Store context needed by search_knowledge_base tool to call run_search_approach
        self._current_messages: list[ChatCompletionMessageParam] = []
        self._current_overrides: dict[str, Any] = {}
        self._current_auth_claims: dict[str, Any] = {}
        # Store data_points from search tool for inclusion in final response
        self._current_data_points: DataPoints = DataPoints(text=[], images=[], citations=[])

        # Initialize SearchContextHolder for SearchTools
        # Note: Context will be set before each request via set_context()
        self._search_context_holder = SearchContextHolder(
            chat_approach=chat_approach,
            messages=[],
            overrides={},
            auth_claims={},
            thoughts=[],
            data_points=DataPoints(text=[], images=[], citations=[]),
            # Optional fields for search_hybrid_simple (direct search without RAG pipeline)
            search_client=search_client,
            openai_client=openai_client,
            embedding_deployment=embedding_deployment,
        )

        # Initialize Tool Classes with @ai_function decorator pattern
        self.search_tools = SearchTools(context_holder=self._search_context_holder)
        self.traversal_tools = TraversalTools()

        # ----------------------------------------------------------------------------------
        # Create Sub-Agents using Factory Functions
        # ----------------------------------------------------------------------------------

        # DocumentSearchAgent: Handles search-related queries using Azure AI Search
        self.document_search_agent = create_document_search_agent(
            chat_client=self.agent_client,
            search_tools=self.search_tools,
        )

        # GraphTraversalAgent: Handles graph traversal and relationship navigation
        self.graph_traversal_agent = create_graph_traversal_agent(
            chat_client=self.agent_client,
            traversal_tools=self.traversal_tools,
        )

        # ----------------------------------------------------------------------------------
        # Initialize SupervisorTools with sub-agents
        # ----------------------------------------------------------------------------------

        self.supervisor_tools = SupervisorTools(
            document_search_agent=self.document_search_agent,
            graph_traversal_agent=self.graph_traversal_agent,
        )

        # ----------------------------------------------------------------------------------
        # Initialize Supervisor Agent (Main Orchestration Agent)
        # ----------------------------------------------------------------------------------

        self.supervisor_agent = create_supervisor_agent(
            chat_client=self.agent_client,
            supervisor_tools=self.supervisor_tools,
            include_traversal_agent=False,  # Set to True when ready to enable traversal
        )

    def _update_tool_contexts(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        thoughts: list[ThoughtStep],
        data_points: DataPoints,
    ) -> None:
        """
        Update the context holders for tool classes before each request.
        
        This method is called at the start of each request to ensure tools have
        access to the current request context (messages, overrides, thoughts, etc.)
        
        Args:
            messages: Current conversation messages
            overrides: Request overrides
            auth_claims: Authentication claims
            thoughts: Shared thoughts list for UI visibility
            data_points: Shared data_points for citations
        """
        # Update SearchContextHolder
        self._search_context_holder.messages = messages
        self._search_context_holder.overrides = overrides
        self._search_context_holder.auth_claims = auth_claims
        self._search_context_holder.thoughts = thoughts
        self._search_context_holder.data_points = data_points
        
        # Update SupervisorTools context
        self.supervisor_tools.set_context(thoughts=thoughts)

    def get_system_prompt_variables(self, override_prompt: Optional[str]) -> dict[str, str]:
        """
        Get system prompt variables for prompt rendering.
        Delegates to chat_approach to reuse the shared implementation from Approach base class.
        """
        return self.chat_approach.get_system_prompt_variables(override_prompt)

    def _extract_followup_questions(self, content: Optional[str]) -> tuple[str, list[str]]:
        """
        Extract follow-up questions from content.
        Follow-up questions are enclosed in << >> markers.

        Matches the pattern from chatreadretrieveread.py extract_followup_questions method.

        Args:
            content: The response content that may contain follow-up questions

        Returns:
            Tuple of (content without follow-up markers, list of follow-up questions)
        """
        if content is None:
            return "", []
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[ExtraInfo, str]:
        """
        Prepares the orchestration context and extracts the user query for agent execution.

        This method follows the pattern from chatreadretrieveread.py's run_until_final_call.
        It renders prompts, creates thought steps, and returns the ExtraInfo context
        along with the user query for the supervisor agent.

        Args:
            messages: List of chat completion messages
            overrides: Override settings from the request context
            auth_claims: Authentication claims from the request context
            should_stream: Whether to prepare for streaming response

        Returns:
            Tuple of (ExtraInfo with thoughts and data points, user_query string)
        """
        # Extract the last user message for orchestration
        user_query = self._extract_user_query(messages)

        logger.info(f"SupervisorAgent: Preparing orchestration for query: {user_query[:100]}...")

        # Initialize thoughts list to track the orchestration flow
        thoughts: list[ThoughtStep] = []

        # Render the supervisor agent prompt using PromptManager
        supervisor_agent_messages = self.prompt_manager.render_prompt(
            self.supervisor_agent_prompt,
            self.get_system_prompt_variables(overrides.get("prompt_template"))
            | {
                "user_query": user_query,
                "past_messages": messages[:-1],
            },
        )

        # Record the rendered prompt as a thought step (similar to "Prompt to generate search query")
        thoughts.append(
            ThoughtStep(
                title="Supervisor Agent Prompt",
                description=supervisor_agent_messages,
                props={
                    "model": self.openai_deployment,
                    "agent_type": "supervisor",
                    "available_tools": ["delegate_to_search_agent", "delegate_to_traversal_agent"],
                    "streaming": should_stream,
                },
            )
        )

        # Record the routing decision step (similar to "Search using generated search query")
        thoughts.append(
            ThoughtStep(
                title="Agent Routing",
                description=f"Routing query to appropriate sub-agents: '{user_query}'",
                props={
                    "available_agents": ["DocumentSearchAgent", "GraphTraversalAgent"],
                    "agent_hierarchy": "SupervisorAgent -> [DocumentSearchAgent, GraphTraversalAgent]",
                    "search_agent_tool": "search_knowledge_base",
                    "traversal_agent_tool": "traverse_knowledge_graph",
                },
            )
        )

        # Reset data_points for this request (will be populated by search tools)
        self._current_data_points = DataPoints(text=[], images=[], citations=[])

        # Create ExtraInfo with DataPoints and thoughts
        # Note: data_points reference self._current_data_points so search tools can populate it
        extra_info = ExtraInfo(
            data_points=self._current_data_points,
            thoughts=thoughts,
        )

        # Initialize the current thoughts list for sub-agent tracking
        # Sub-agents will append their ThoughtSteps to this list during execution
        self._current_thoughts = thoughts

        # Update tool contexts with current request information
        self._update_tool_contexts(
            messages=messages,
            overrides=overrides,
            auth_claims=auth_claims,
            thoughts=thoughts,
            data_points=self._current_data_points,
        )

        return (extra_info, user_query)

    async def run_without_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> dict[str, Any]:
        """
        Internal method for non-streaming orchestration.
        Routes query through the agent hierarchy and returns complete result.

        Args:
            messages: List of chat completion messages
            overrides: Override settings from the request context
            auth_claims: Authentication claims from the request context
            session_state: Optional session state for persistence

        Returns:
            Response dictionary with the answer from SupervisorAgent
        """
        # Get orchestration context and user query using run_until_final_call
        extra_info, user_query = await self.run_until_final_call(messages, overrides, auth_claims, should_stream=False)

        logger.info(f"SupervisorAgent: Processing query: {user_query[:100]}...")

        # Run the supervisor agent - non-streaming
        # This gets the complete result at once, following agent-framework pattern
        result_response = await self.supervisor_agent.run(user_query)

        logger.info("SupervisorAgent: Successfully processed query through agent hierarchy")

        content = result_response.text
        role = "assistant"

        # Handle follow-up questions if enabled (matching chatreadretrieveread.py pattern)
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self._extract_followup_questions(content)
            extra_info.followup_questions = followup_questions

        # Record the agent response as a thought step (similar to "Search results")
        extra_info.thoughts.append(
            ThoughtStep(
                title="Supervisor Agent Response",
                description=content[:500] + "..." if len(content) > 500 else content,
                props={
                    "model": self.openai_deployment,
                    "response_length": len(result_response.text),
                },
            )
        )

        # Debug: Log the final data_points before returning
        logger.info(f"SupervisorAgent: Final data_points.citations: {extra_info.data_points.citations}")
        logger.info(f"SupervisorAgent: Final data_points.text count: {len(extra_info.data_points.text or [])}")

        # Format response to match expected Approach interface
        chat_app_response = {
            "message": {"content": content, "role": role},
            "context": {
                "thoughts": extra_info.thoughts,
                "data_points": {
                    "text": extra_info.data_points.text,
                    "images": extra_info.data_points.images,
                    "citations": extra_info.data_points.citations,
                },
                "followup_questions": extra_info.followup_questions,
            },
            "session_state": session_state,
        }

        return chat_app_response

    async def run_with_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Internal method for streaming orchestration.
        Routes query through the agent hierarchy and yields chunks as they arrive.

        Args:
            messages: List of chat completion messages
            overrides: Override settings from the request context
            auth_claims: Authentication claims from the request context
            session_state: Optional session state for persistence

        Yields:
            Response chunks with streaming content
        """
        # Get orchestration context and user query using run_until_final_call
        extra_info, user_query = await self.run_until_final_call(messages, overrides, auth_claims, should_stream=True)

        logger.info(f"SupervisorAgent (streaming): Processing query: {user_query[:100]}...")

        # Yield initial context with role and thoughts
        yield {"delta": {"role": "assistant"}, "context": extra_info, "session_state": session_state}

        # Track content for follow-up questions (matching chatreadretrieveread.py pattern)
        followup_questions_started = False
        followup_content = ""
        full_content = ""

        # Stream results as they are generated using agent.run_stream()
        # Following agent-framework streaming pattern
        async for chunk in self.supervisor_agent.run_stream(user_query):
            if chunk.text:
                content = chunk.text
                full_content += content

                # Handle follow-up questions during streaming (matching chatreadretrieveread.py)
                if overrides.get("suggest_followup_questions") and "<<" in content:
                    followup_questions_started = True
                    earlier_content = content[: content.index("<<")]
                    if earlier_content:
                        yield {
                            "delta": {
                                "content": earlier_content,
                                "role": "assistant",
                            }
                        }
                    followup_content += content[content.index("<<") :]
                elif followup_questions_started:
                    followup_content += content
                else:
                    # Yield each chunk as it arrives from the agent
                    yield {
                        "delta": {
                            "content": content,
                            "role": "assistant",
                        }
                    }

        # Yield follow-up questions if any were found (matching chatreadretrieveread.py)
        if followup_content:
            _, followup_questions = self._extract_followup_questions(followup_content)
            extra_info.followup_questions = followup_questions

        # Record the final agent response as a thought step
        extra_info.thoughts.append(
            ThoughtStep(
                title="Supervisor Agent Response",
                description=full_content[:500] + "..." if len(full_content) > 500 else full_content,
                props={
                    "model": self.openai_deployment,
                    "response_length": len(full_content),
                },
            )
        )

        # Yield final context with ALL thoughts (including sub-agent ThoughtSteps appended during streaming)
        # This ensures the UI receives the complete thought process after tool calls complete
        yield {"delta": {"role": "assistant"}, "context": extra_info, "session_state": session_state}

        logger.info("SupervisorAgent (streaming): Successfully completed streaming response")

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Run the agent orchestrator for non-streaming responses.
        Uses agent.run() to get the complete result at once.

        Args:
            messages: List of chat completion messages
            session_state: Optional session state for persistence
            context: Additional context including auth_claims

        Returns:
            Response dictionary with the answer from SupervisorAgent
        """
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})

        try:
            # Store context for potential use by sub-agents and search tool
            self._current_session_state = session_state
            self._current_context = context
            # Store context needed by search_knowledge_base to call run_search_approach
            self._current_messages = messages
            self._current_overrides = overrides
            self._current_auth_claims = auth_claims

            result = await self.run_without_streaming(messages, overrides, auth_claims, session_state)
            return result

        except Exception as e:
            logger.error(f"SupervisorAgent: Error occurred: {e}", exc_info=True)
            # Fallback to chat approach on any error for backward compatibility
            logger.info("SupervisorAgent: Falling back to chat approach")
            return await self.chat_approach.run(messages, session_state, context)

    async def run_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run the agent orchestrator for streaming responses.
        Uses agent.run_stream() to get results as they are generated.

        Args:
            messages: List of chat completion messages
            session_state: Optional session state for persistence
            context: Additional context including auth_claims

        Returns:
            AsyncGenerator yielding response chunks with streaming content
        """
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})

        # Store context for potential use by sub-agents and search tool
        self._current_session_state = session_state
        self._current_context = context
        # Store context needed by search_knowledge_base to call run_search_approach
        self._current_messages = messages
        self._current_overrides = overrides
        self._current_auth_claims = auth_claims

        return self.run_with_streaming(messages, overrides, auth_claims, session_state)

    def _extract_user_query(self, messages: list[ChatCompletionMessageParam]) -> str:
        """
        Extract the last user message from the message list.

        Args:
            messages: List of chat completion messages

        Returns:
            The last user message content as a string
        """
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle multimodal content
                    text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                    return " ".join(text_parts)
        return ""
