from typing import Any, TypedDict
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from recommender.utils import ContentStore, UsersWithInteractions, parse_token_usage
from recommender.state import RecommenderGraphConfig


# Default recommender prompt
DEFAULT_RECOMMENDER_PROMPT = """
You are a content recommendation agent for a roleplay content platform.
Your job is to find the top 10 most relevant content pieces for a new user given a user's chosen tags.

Instructions:
1. Use the available tools to find relevant content (this returns full details)
2. Select the top 10 most relevant content IDs from the results
3. Provide a brief explanation of why these recommendations were chosen

Only return the content IDs."""


class RecommendationResponse(BaseModel):
    """Response format for recommendations"""
    content_ids: list[int] = Field(description="Top 10 recommended content IDs")


class RecommendContentInput(TypedDict):
    """Input format for recommender interactions"""
    content_store: ContentStore
    users_with_interactions: UsersWithInteractions
    current_recommender_prompt: str
    user_id: int = Field(description="User ID being processed")
    user_tags: list[str] = Field(description="User tags")


class RecommendContentOutput(TypedDict):
    """Output format for recommendations"""
    recommended_content_ids: dict[int, list[int]] = Field(description="Top 10 recommended content IDs by user_id")
    recommender_actions: dict[int, list[Any]] = Field(description="The actions from the recommender agent")
    token_usage: dict[str, dict[str, int]] = Field(description="Token usage for the model")


def recommend_content(
    state: RecommendContentInput,
    config: RecommenderGraphConfig
) -> RecommendContentOutput:
    """
    Generate fast content recommendations for the given user tags.
    
    Args:
        state: RecommendContentInput containing agent config and other state
        config: Configuration dictionary containing model settings
        
    Returns:
        RecommendContentOutput with content IDs and recommender messages for the current user
    """
    user_tags = state["user_tags"]
    user_id = state["user_id"]
    prompt = state["current_recommender_prompt"]
    
    # Create the recommender agent
    model_name = config.get("configurable", {}).get("recommender_model", "gemini-2.0-flash")
    agent = create_recommendation_agent(
        prompt=prompt,
        model_name=model_name,
        content_store=state["content_store"]
    )
    
    # Run the agent
    result = agent.invoke({"messages": [{"role": "user", "content": f"User tags: {user_tags}"}]})
    
    # Parse and shorten ToolMessage contents
    processed_messages = []
    for message in result["messages"]:
        # Check if this is a ToolMessage (has 'name' field and 'content' field)
        if hasattr(message, 'name') and hasattr(message, 'content') and message.name == "get_relevant_content":
            # Shorten ToolMessage content to first 10 chars + ... + last 10 chars
            content = str(message.content)
            if len(content) > 20:
                shortened_content = content[:10] + "...shortened..." + content[-10:]
                # Create a copy of the message with shortened content
                processed_message = type(message)(
                    content=shortened_content,
                    name=message.name,
                    id=message.id,
                    tool_call_id=message.tool_call_id
                )
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)
        else:
            processed_messages.append(message)
    
    # Accumulate token usage for each model found in result["messages"]
    token_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    for message in result["messages"]:
        usage = getattr(message, 'usage_metadata', None)
        if usage:
            token_usage['input_tokens'] += usage.get('input_tokens', 0)
            token_usage['output_tokens'] += usage.get('output_tokens', 0)
            token_usage['total_tokens'] += usage.get('total_tokens', 0)
    
    # Get recommendation response
    if "structured_response" in result:
        recommendation_response = result["structured_response"]
    else:
        recommendation_response = RecommendationResponse(
            content_ids=[]
        )
    
    # Return results
    return RecommendContentOutput(
        recommended_content_ids={user_id: recommendation_response.content_ids},
        recommender_actions={user_id: processed_messages},
        token_usage={model_name: token_usage}
    )


def create_recommendation_agent(
    prompt: str,
    model_name: str = "gemini-2.0-flash",
    content_store: ContentStore = None
):
    """
    Create a recommendation agent with the specified configuration.
    
    Args:
        prompt: Custom prompt template
        model_name: Name of the model to use
        content_store: ContentStore instance for the tools
        
    Returns:
        Dictionary containing the agent and configuration
    """
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.
    )
    
    # Create tools if content store is provided
    tools = []
    if content_store:
        tools = [create_get_relevant_content_tool(content_store)]
    
    # Create the React agent using LangGraph with dynamic prompt
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        response_format=RecommendationResponse
    )
    
    return agent


def create_get_relevant_content_tool(content_store: ContentStore):
    """Create a tool function that can access the content store"""
    @tool
    def get_relevant_content(tags: list[str], k: int = 20) -> list[dict[str, Any]]:
        """
        Search for content using the provided tags and return full content details.
        
        Args:
            tags: List of tags to search for
            k: Number of results to return (max 50)
            
        Returns:
            List of dictionaries with full content details and similarity scores
        """
        k = min(k, 50)
        results = content_store.search_by_tags(tags, k=k)
        
        # Convert pandas Series to dictionaries with content details
        content_details = []
        for content_row, score in results:
            # Convert Series to dict and add content_id and similarity_score
            content_dict = content_row.to_dict()
            content_dict['content_id'] = content_row.name
            content_dict['similarity_score'] = score
            content_details.append(content_dict)
        
        return content_details
    
    return get_relevant_content
