from operator import add
from typing import Any, TypedDict
from typing_extensions import Annotated
from recommender.utils import ContentStore, UsersWithInteractions


class RecommenderGraphConfig(TypedDict):
    """Configuration for the recommender graph"""
    target_score: float
    score_threshold: float
    max_iterations: int
    users_per_iteration: int
    
    recommender_model: str
    evaluator_model: str
    optimizer_model: str
    recursion_limit: int

def add_token_usage(
    current_token_usage: dict[str, dict[str, int]], 
    new_token_usage: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """
    Reducer function to add token usage for multiple models.
    
    Args:
        current_token_usage: Current token usage organized by model
        new_token_usage: New token usage to add
        
    Returns:
        Updated token usage dictionary
    """
    # Create a copy to avoid mutating the original
    updated_usage = current_token_usage.copy()
    
    # Add new usage for each model
    for model_name, usage in new_token_usage.items():
        if model_name not in updated_usage:
            updated_usage[model_name] = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        
        # Add new usage to existing totals
        updated_usage[model_name]['input_tokens'] += usage.get('input_tokens', 0)
        updated_usage[model_name]['output_tokens'] += usage.get('output_tokens', 0)
        updated_usage[model_name]['total_tokens'] += usage.get('total_tokens', 0)
    
    return updated_usage


def add_dict_by_user_id(
    current_dict: dict[int, Any], 
    new_dict: dict[int, Any] | None
) -> dict[int, Any]:
    """
    Reducer function to add or update dictionary values by user_id.
    
    This reducer can be used for any dictionary that maps user_id to some value.
    If new_dict is None, it resets the dictionary to empty.
    Otherwise, it updates the current dictionary with new values.
    
    Args:
        current_dict: Current dictionary organized by user_id
        new_dict: New dictionary to add, or None to reset
        
    Returns:
        Updated dictionary
    """
    # If new_dict is None, reset the dictionary
    if new_dict is None:
        return {}
    
    # Create a copy to avoid mutating the original
    updated_dict = current_dict.copy()
    
    # Add new values for each user
    for user_id, value in new_dict.items():
        updated_dict[user_id] = value
    
    return updated_dict


def add_or_clear_user_ids(
    current_user_ids: list[int], 
    new_user_ids: list[int] | None
) -> list[int]:
    """
    Reducer function to add user IDs or clear the list.
    
    This reducer can be used for the user_ids list.
    If new_user_ids is None, it resets the list to empty.
    Otherwise, it extends the current list with new user IDs.
    
    Args:
        current_user_ids: Current list of user IDs
        new_user_ids: New user IDs to add, or None to reset
        
    Returns:
        Updated list of user IDs
    """
    # If new_user_ids is None, reset the list
    if new_user_ids is None:
        return []
    
    # Create a copy to avoid mutating the original
    updated_user_ids = current_user_ids.copy()
    
    # Add new user IDs
    updated_user_ids.extend(new_user_ids)
    
    return updated_user_ids


class RecommenderGraphState(TypedDict):
    """State for the recommender graph"""
    # Data stores
    content_store: ContentStore
    users_with_interactions: UsersWithInteractions
    
    # Current state
    current_iteration: int
    should_continue: bool
    current_user_id: int | None
    current_score: float
    best_score: float
    score_history: Annotated[list[float], add]
    current_recommender_prompt: str
    best_recommender_prompt: str
    content_context_cache_name: str | None
    content_context_cache_expires_at: float | None
    
    # Function outputs tracking for downstream nodes
    user_ids: Annotated[list[int], add_or_clear_user_ids]
    simulated_tags: Annotated[dict[int, list[str]], add_dict_by_user_id]
    ground_truth_ids: Annotated[dict[int, list[int]], add_dict_by_user_id]
    recommended_content_ids: Annotated[dict[int, list[int]], add_dict_by_user_id]
    recommender_actions: Annotated[dict[int, list[Any]], add_dict_by_user_id]
    evaluation_details: dict[str, Any]
    optimization_history: Annotated[list[dict[str, Any]], add]
    
    # Token usage tracking
    token_usage: Annotated[dict[str, dict[str, int]], add_token_usage]
