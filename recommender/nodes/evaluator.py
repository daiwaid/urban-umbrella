import random
import time
from typing import Any, TypedDict
import pandas as pd
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from recommender.utils import ContentStore, UsersWithInteractions, parse_token_usage
from recommender.state import RecommenderGraphState, RecommenderGraphConfig


class SimulatedTags(BaseModel):
    """Response format for simulated user registration tags"""
    selected_tags: list[str] = Field(description="List of tags the user would select during registration (5-10 tags)")


class GroundTruthContent(BaseModel):
    """Response format for ground truth content selection"""
    top_content_ids: list[int] = Field(description="Top 10 content IDs most relevant for this user based on their interests and behavior")


class SimulateTagsOutput(TypedDict):
    """Output format for simulate tags function"""
    user_tags: list[str] = Field(description="Simulated user tags for passing to recommend_content")
    simulated_tags: dict[int, list[str]] = Field(description="Simulated tags tracked by user_id for state")
    user_id: int = Field(description="Current user ID being processed")
    token_usage: dict[str, dict[str, int]] = Field(description="Token usage for the model")


def new_eval_iteration(
    state: RecommenderGraphState,
    config: RecommenderGraphConfig
) -> dict[str, Any]:
    """
    Start a new evaluation iteration.
    
    This function decides whether to continue the iteration based on current_iteration
    and max_iterations, or if the score has plateaued (not enough improvement for 2 iterations).
    If continuing, it clears function outputs.
    
    Args:
        state: Current state of the recommender graph
        
    Returns:
        Dictionary with updated state
    """
    current_iteration = state.get("current_iteration", 0)
    score_history = state.get("score_history", [])
    should_continue = state.get("should_continue", False)
    max_iterations = config.get("configurable", {}).get("max_iterations", 10)
    users_per_iteration = config.get("configurable", {}).get("users_per_iteration", 10)
    score_threshold = config.get("configurable", {}).get("score_threshold", 0.01)
    
    # Check if we should continue iterations based on max iterations
    should_continue_max_iter = current_iteration <= max_iterations
    
    # Check if score has plateaued (not enough improvement for past 2 iterations)
    should_continue_plateau = True
    if len(score_history) >= 3:  # Need at least 3 scores to check plateau
        recent_scores = score_history[-3:]  # Last 3 scores
        improvements = []
        
        for i in range(1, len(recent_scores)):
            improvement = recent_scores[i] - recent_scores[i-1]
            improvements.append(improvement)
        
        # Check if improvements are too small (less than score_threshold for 2 consecutive iterations)
        if len(improvements) >= 2:
            recent_improvements = improvements[-2:]  # Last 2 improvements
            if all(imp < score_threshold for imp in recent_improvements):
                should_continue_plateau = False
                print(f"Score plateaued: improvements of {recent_improvements} are below threshold ({score_threshold})")
    
    # Continue only if both conditions are met
    should_continue = should_continue_max_iter and should_continue_plateau
    
    if should_continue:
        # Clear function outputs from state by setting them to None
        # (the reducers will handle resetting them to empty dicts)
        print(f"\nStarting iteration {current_iteration}/{max_iterations}, users to sample and process: {users_per_iteration}")
        
        return {
            "current_iteration": current_iteration + 1,
            "user_ids": None,
            "should_continue": True,
            # Clear function outputs
            "simulated_tags": None,
            "ground_truth_ids": None,
            "recommended_content_ids": None,
            "recommender_actions": None
        }
    else:
        if not should_continue_max_iter:
            print(f"\nReached max iterations ({max_iterations}), stopping evaluation")
        elif not should_continue_plateau:
            print(f"\nScore plateaued, stopping evaluation early")
        
        return {
            "should_continue": False
        }


def sample_user(state: RecommenderGraphState) -> dict[str, Any]:
    """
    Sample a single user from users_with_interactions and add it to user_ids.
    
    Args:
        state: Current state of the recommender graph
        config: Configuration for the recommender graph
        
    Returns:
        Dictionary with the sampled user_id and updated state
    """
    users_with_interactions = state["users_with_interactions"]
    
    # Sample a new user from users_with_interactions
    sampled_users_df = users_with_interactions.sample_users(1)
    new_user_id = sampled_users_df.index[0]
    
    print(f"\nProcessing new user: {new_user_id}")
    
    return {
        "current_user_id": new_user_id,
        "user_ids": [new_user_id],
        "should_continue": True
    }


def simulate_new_user_tags(
    state: RecommenderGraphState,
    config: RecommenderGraphConfig
) -> SimulateTagsOutput:
    """
    Simulate the tags a new user would select during registration using AI.
    
    Args:
        state: SimulateTagsInput containing content store, users, and user_id
        config: Configuration dictionary containing model settings
        
    Returns:
        SimulateTagsOutput with simulated user tags and token usage
    """
    content_store = state["content_store"]
    users_with_interactions = state["users_with_interactions"]
    user_id = state["current_user_id"]
    max_tags = 10
    
    # Get user data
    user_row = users_with_interactions.get_user_by_id(user_id)
    if user_row is None:
        return SimulateTagsOutput(
            simulated_tags={user_id: []},
            user_id=user_id,
            token_usage={}
        )
    user_interest_tags = user_row.get('user_interest_tags', '')
    
    if not user_interest_tags or pd.isna(user_interest_tags):
        return SimulateTagsOutput(
            simulated_tags={user_id: []},
            user_id=user_id,
            token_usage={}
        )
    
    try:
        # Get model from config
        model_name = config.get("configurable", {}).get("evaluator_model", "gemini-2.5-flash")
        llm = create_evaluator_llm(model_name)
        
        # Initialize parser
        tags_parser = PydanticOutputParser(pydantic_object=SimulatedTags)
        
        # Get user's top interactions (up to 10) from the pre-computed content_interactions
        interaction_details = []
        content_interactions = user_row.get('content_interactions', [])
        
        # Get the first 10 interactions (they're already sorted by count)
        top_interactions = content_interactions[:10]
        
        # Get content details for each interaction
        for content_id, interaction_count in top_interactions:
            content_row = content_store.get_content_by_id(content_id)
            if content_row is not None:
                interaction_details.append({
                    'content_id': content_id,
                    'title': content_row.get('title', ''),
                    'intro': content_row.get('intro', ''),
                    'character_list': content_row.get('character_list', ''),
                    'tags': content_row.get('tags', ''),
                    'interaction_count': interaction_count
                })
        
        # Create prompt for tag simulation
        prompt = f"""
You are predicting a user's tag selection during registration for a roleplay content platform.

User Information:
- Users Full Interest Tags: {user_interest_tags}
- User's Top Content Interactions (sorted by interaction count):
{interaction_details}

During registration, users typically select 5-10 tags that best represent their current interests.
Consider factors like:
- Which set of tags best summarize the user's full list of interest tags
- Which tags would be most useful for finding the content they've interacted with the most
- Which tags would help them find content that are different to the content they've only briefly interacted with before (those with interaction counts less than 10)

{tags_parser.get_format_instructions()}

Select 5-10 tags that this user would most likely choose during registration.
        """
        
        # Generate response using sync call
        response = llm.invoke(prompt)
        
        # Parse token usage
        token_usage = parse_token_usage(response, model_name)
        
        # Parse the response
        parsed_result = tags_parser.parse(response.content)
        
        # Return results
        return SimulateTagsOutput(
            user_tags=parsed_result.selected_tags[:max_tags],
            simulated_tags={user_id: parsed_result.selected_tags[:max_tags]},
            user_id=user_id,
            token_usage={model_name: token_usage}
        )
        
    except Exception as e:
        print(f"Error simulating user registration tags for user {user_id}: {e}")
        # Fallback to simple random selection
        all_tags = [tag.strip() for tag in user_interest_tags.split(',') if tag.strip()]
        if all_tags:
            num_tags = min(random.randint(3, max_tags), len(all_tags))
            selected_tags = random.sample(all_tags, num_tags)
            return SimulateTagsOutput(
                user_tags=selected_tags,
                simulated_tags={user_id: selected_tags},
                user_id=user_id,
                token_usage={}
            )
        return SimulateTagsOutput(
            user_tags=[],
            simulated_tags={user_id: []},
            user_id=user_id,
            token_usage={}
        )


def generate_ground_truth(
    state: RecommenderGraphState,
    config: RecommenderGraphConfig
) -> dict[str, Any]:
    """
    Generate ground truth top 10 content IDs for a user using AI analysis.
    
    Args:
        state: GenerateGroundTruthInput containing content store, users, user_id, and cache info
        config: Configuration dictionary containing model settings
        
    Returns:
        GenerateGroundTruthOutput with ground truth IDs, cache info, and token usage
    """
    content_store = state["content_store"]
    users_with_interactions = state["users_with_interactions"]
    user_id = state["current_user_id"]
    
    try:
        # Get model from config
        model_name = config.get("configurable", {}).get("evaluator_model", "gemini-2.5-flash")
        
        # Check if we need to create a new context cache
        current_time = time.time()
        cache_name = state.get("content_context_cache_name")
        cache_expires_at = state.get("content_context_cache_expires_at", 0.0)
        
        # Create new cache if none exists or current cache is expired
        if cache_name is None or cache_expires_at <= current_time:
            cache_name, expiration_time = create_content_context_cache(content_store, model_name)
            if cache_name is None:
                print(f"Error creating content context cache")
                return {
                    "ground_truth_ids": {user_id: []},
                    "content_context_cache_name": None,
                    "content_context_cache_expires_at": 0.0,
                    "token_usage": {}
                }
        else:
            expiration_time = cache_expires_at
        
        llm = create_evaluator_llm(model_name, cached_content=cache_name)
        
        # Initialize parser
        ground_truth_parser = PydanticOutputParser(pydantic_object=GroundTruthContent)
        
        # Get user data
        user_row = users_with_interactions.get_user_by_id(user_id)
        if user_row is None:
            return {
                "ground_truth_ids": {user_id: []},
                "token_usage": {},
                "content_context_cache_name": cache_name,
                "content_context_cache_expires_at": expiration_time
            }
        
        user_interest_tags = user_row.get('user_interest_tags', '')
        
        # Get user's top interactions (up to 20) from the pre-computed content_interactions
        content_interactions = user_row.get('content_interactions', [])
        
        # Get the first 20 interactions (they're already sorted by count)
        top_interactions = content_interactions[:20]
        
        # Extract just the content IDs and interaction counts for the prompt
        interaction_summary = [
            {'content_id': content_id, 'interaction_count': interaction_count}
            for content_id, interaction_count in top_interactions
        ]
        
        # Create prompt for ground truth generation
        prompt = f"""
You are determining the top 10 most relevant roleplay content items for a user based on their interest tags and interaction history.

User Information:
- Users Interest Tags: {user_interest_tags}
- User's Top Content Interactions (sorted by interaction count):
{interaction_summary}

Based on the user's interest tags and their interaction patterns, determine the top 10 content IDs that would be most relevant for this user.

Choose content:
- that allign the most with the user's interest tags
- that they've interacted with the most
- that are different from the content that they've only briefly interacted with before (those with interaction counts less than 10)

{ground_truth_parser.get_format_instructions()}

Return the top 10 content IDs that would be most relevant for this user.
        """
        
        # Generate response using sync call
        response = llm.invoke(prompt)
        
        # Parse token usage
        token_usage = parse_token_usage(response, model_name)
        
        # Subtract cached tokens only if we reused an existing cache
        if cache_name == state.get("content_context_cache_name"):
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                cache_read_tokens = response.usage_metadata.get('input_token_details', {}).get('cache_read', 0)
                
                # Adjust the token counts to exclude cached tokens
                adjusted_input_tokens = max(0, token_usage['input_tokens'] - cache_read_tokens)
                adjusted_token_usage = {
                    'input_tokens': adjusted_input_tokens,
                    'output_tokens': token_usage['output_tokens'],
                    'total_tokens': adjusted_input_tokens + token_usage['output_tokens']
                }
            else:
                adjusted_token_usage = token_usage
        else:
            # We created a new cache, so don't subtract cached tokens
            # (the cache creation cost should be included)
            adjusted_token_usage = token_usage
        
        # Parse the response
        parsed_result = ground_truth_parser.parse(response.content)
        
        return {
            "ground_truth_ids": {user_id: parsed_result.top_content_ids[:10]},
            "content_context_cache_name": cache_name,
            "content_context_cache_expires_at": expiration_time,
            "token_usage": {model_name: adjusted_token_usage}
        }
        
    except Exception as e:
        print(f"Error generating ground truth for user {user_id}: {e}")
        # Fallback to interaction-based selection using pre-computed data
        user_row = users_with_interactions.get_user_by_id(user_id)
        if user_row is None:
            return {
                "ground_truth_ids": {user_id: []},
                "content_context_cache_name": cache_name,
                "content_context_cache_expires_at": expiration_time,
                "token_usage": {}
            }
        
        content_interactions = user_row.get('content_interactions', [])
        if not content_interactions:
            return {
                "ground_truth_ids": {user_id: []},
                "content_context_cache_name": cache_name,
                "content_context_cache_expires_at": expiration_time,
                "token_usage": {}
            }
        
        # Get top 10 content IDs from pre-computed interactions
        top_content = [content_id for content_id, _ in content_interactions[:10]]
        
        return {
            "ground_truth_ids": {user_id: top_content},
            "content_context_cache_name": cache_name,
            "content_context_cache_expires_at": expiration_time,
            "token_usage": {}
        }


def finish_user(state: RecommenderGraphState, config: RecommenderGraphConfig) -> dict[str, Any]:
    """
    Finish processing the current user and determine whether to continue with the next user.
    
    Args:
        state: Current state of the recommender graph
        config: Configuration for the recommender graph
        
    Returns:
        Dictionary indicating whether to continue processing users
    """
    user_ids = state.get("user_ids", [])
    current_user_id = state.get("current_user_id", None)
    users_with_interactions = state.get("users_with_interactions")
    simulated_tags = state.get("simulated_tags", {})
    recommended_content_ids = state.get("recommended_content_ids", {})
    ground_truth_ids = state.get("ground_truth_ids", {})
    users_per_iteration = config.get("configurable", {}).get("users_per_iteration", 10)
    
    
    # Get user interest tags
    user_interest_tags = ""
    if users_with_interactions and current_user_id:
        user_row = users_with_interactions.get_user_by_id(current_user_id)
        if user_row is not None:
            user_interest_tags = user_row.get('user_interest_tags', 'N/A')
    
    # Get simulated tags for current user
    current_simulated_tags = simulated_tags.get(current_user_id, [])
    
    # Get recommended content IDs for current user
    current_recommended_ids = recommended_content_ids.get(current_user_id, [])
    
    # Get ground truth IDs for current user
    current_ground_truth_ids = ground_truth_ids.get(current_user_id, [])
    
    # Print user details
    print(f"Finished processing user: {current_user_id}")
    print(f"User Interest Tags:\n{user_interest_tags}")
    print(f"Simulated Tags:\n{current_simulated_tags}")
    print(f"Recommended Content IDs:\n{current_recommended_ids}")
    print(f"Generated Ground Truth IDs:\n{current_ground_truth_ids}")
    
    # Check if we have sampled enough users for this iteration
    if len(user_ids) < users_per_iteration:
        return {
            "should_continue": True
        }
    else:
        print("All users processed, moving to evaluation")
        return {
            "should_continue": False
        }


def evaluate_recommendations(state: RecommenderGraphState) -> dict[str, Any]:
    """
    Evaluate the recommendation system by computing mean recall from state data.
    
    Args:
        state: State dictionary containing ground_truth_ids and recommended_content_ids
        
    Returns:
        Dictionary with evaluation results
    """
    # Extract data from state
    ground_truth_ids = state.get("ground_truth_ids", {})
    recommended_content_ids = state.get("recommended_content_ids", {})
    best_score = state.get("best_score", 0.0)
    current_recommender_prompt = state.get("current_recommender_prompt", "")
    best_recommender_prompt = state.get("best_recommender_prompt", "")
    
    total_recall = 0.0
    valid_users = 0
    evaluation_details = {}
    
    # Process each user that has both ground truth and recommendations
    for user_id in ground_truth_ids:
        if user_id not in recommended_content_ids:
            continue
        
        ground_truth = ground_truth_ids[user_id]
        predicted_ids = recommended_content_ids[user_id]
        
        if not ground_truth:
            continue
        
        # Calculate recall
        recall = calculate_recall(predicted_ids, ground_truth)
        
        total_recall += recall
        valid_users += 1
        
        # Store user results
        evaluation_details[user_id] = {
            'predicted_ids': predicted_ids,
            'ground_truth_ids': ground_truth,
            'recall': recall
        }
    
    # Calculate mean recall
    mean_recall = total_recall / valid_users if valid_users > 0 else 0.0
    
    # Update best score and best recommender prompt
    if mean_recall > best_score:
        best_score = mean_recall
        best_recommender_prompt = current_recommender_prompt
    
    print("-" * 60)
    print(f"EVALUATION SUMMARY:")
    print(f"Total users evaluated: {valid_users}")
    print(f"Mean recall: {mean_recall:.3f}")
    print("-" * 60)
    
    return {
        "current_score": mean_recall,
        "score_history": [mean_recall],
        "evaluation_details": evaluation_details,
        "best_score": best_score,
        "best_recommender_prompt": best_recommender_prompt
    }


def create_content_context_cache(content_store: ContentStore, model_name: str = "gemini-2.5-flash") -> tuple[str, float]:
    """
    Create a context cache for the contents data from content store.
    
    Args:
        content_store: ContentStore instance to get content data from
        model_name: Name of the model to use
        
    Returns:
        tuple: (cache_name, expiration_timestamp)
    """
    try:
        client = genai.Client()
        
        # Get all content from the content store
        contents_df = content_store.get_all_content()
        
        # Convert DataFrame to a structured text format
        contents_text = "CONTENTS DATABASE:\n\n"
        for content_id, row in contents_df.iterrows():
            contents_text += f"**Content ID:** {content_id}\n"
            contents_text += f"**Title:** {row.get('title', 'N/A')}\n"
            contents_text += f"**Introduction:** {row.get('intro', 'N/A')}\n"
            contents_text += f"**Character List:** {row.get('character_list', 'N/A')}\n"
            contents_text += f"**Initial Record:** {row.get('initial_record', 'N/A')}\n"
            contents_text += f"**Tags:** {row.get('tags', 'N/A')}\n"
            contents_text += "\n\n"
        
        # Create a temporary text file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(contents_text)
            temp_file_path = temp_file.name
        
        try:
            # Upload the text file
            file = client.files.upload(file=temp_file_path)
            while file.state.name == 'PROCESSING':
                time.sleep(0.5)
                file = client.files.get(name=file.name)
            
            # Create cache
            cache = client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    display_name='Contents Data Cache',
                    system_instruction=(
                        'You are an expert content recommendation analyst. You have access to a comprehensive '
                        'database of content items with their titles, introductions, character lists, and other metadata. '
                        'Your job is to analyze user preferences and interaction patterns to determine the most relevant '
                        'content recommendations based on the provided data.'
                    ),
                    contents=[file],
                    ttl="1200s",  # 20 min cache
                )
            )
            
            # Calculate expiration time (600 seconds from now)
            expiration_time = time.time() + 600
            
            return cache.name, expiration_time
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"Error creating context cache: {e}")
        return None, 0.0


def create_evaluator_llm(model_name: str = "gemini-2.5-flash", cached_content: str = None):
    """
    Create an LLM for evaluation tasks.
    
    Args:
        model_name: Name of the model to use
        cached_content: Optional cache name for context caching
        
    Returns:
        ChatGoogleGenerativeAI model instance
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        cached_content=cached_content
    )



def calculate_recall(predicted_ids: list[int], ground_truth_ids: list[int]) -> float:
    """
    Calculate recall score for a user.
    
    Args:
        predicted_ids: List of predicted content IDs
        ground_truth_ids: List of ground truth content IDs
        
    Returns:
        Recall score (intersection / ground_truth_size)
    """
    if not ground_truth_ids:
        return 0.0
    
    # Convert to sets for intersection calculation
    predicted_set = set(predicted_ids)
    ground_truth_set = set(ground_truth_ids)
    
    # Calculate intersection
    intersection = predicted_set.intersection(ground_truth_set)
    
    # Calculate recall
    recall = len(intersection) / len(ground_truth_set)
    
    return recall
