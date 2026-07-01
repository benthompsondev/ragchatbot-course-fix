import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters.
        # Note: Sonnet 5 (and Opus 4.7+) reject sampling params like temperature
        # with a 400, so we don't set temperature here — the model defaults apply.
        # max_tokens must cover thinking + the answer: Sonnet 5 runs adaptive
        # thinking by default, and those thinking tokens count against this cap.
        # At 800 a longer answer (e.g. a course outline) could burn the whole
        # budget on thinking and return a turn with no text block, which showed
        # up as the "couldn't generate a response" fallback.
        self.base_params = {
            "model": self.model,
            "max_tokens": 4096
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return self._extract_text(response)

    # Safety cap on how many sequential tool-use rounds we allow before
    # forcing a final answer, so a misbehaving model can't loop forever.
    MAX_TOOL_ROUNDS = 5

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Run the model's tool calls and feed results back, repeating for as many
        rounds as the model needs (up to MAX_TOOL_ROUNDS) before returning text.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        response = initial_response

        for round_num in range(self.MAX_TOOL_ROUNDS):
            # Add the AI's tool-use response to the conversation
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls in this round and collect results
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })

            # On the final allowed round, drop tools so the model is forced to
            # answer with text and can't leave a dangling tool_use behind.
            is_last_round = round_num == self.MAX_TOOL_ROUNDS - 1

            # Add tool results as a single user message. Every tool_use above
            # now has its matching tool_result, which the API requires. On the
            # last round, also tell the model to answer from what it has — without
            # this nudge it often "wants" another search, gets no tools, and
            # returns an empty turn (which would fall back to the error message).
            if tool_results:
                content = list(tool_results)
                if is_last_round:
                    content.append({
                        "type": "text",
                        "text": "Based on the search results above, answer the "
                                "user's question now. Do not request any more "
                                "searches — use the information you already have."
                    })
                messages.append({"role": "user", "content": content})

            api_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
            }
            if not is_last_round:
                api_params["tools"] = base_params["tools"]
                api_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**api_params)

            # If the model is done calling tools, return its text answer
            if response.stop_reason != "tool_use":
                return self._extract_text(response)

        # Loop exhausted (the last call had no tools, so it can't be tool_use,
        # but fall back safely just in case).
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Return the model's text answer, joining every text block in the turn
        (adaptive-thinking responses interleave thinking + text blocks). Falls
        back to a safe message only if the turn genuinely had no text at all."""
        parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        if parts:
            return "\n".join(parts).strip()
        return "I wasn't able to generate a response for that. Please try rephrasing your question."