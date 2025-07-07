from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from recommender.state import RecommenderGraphState, RecommenderGraphConfig
from recommender.utils import parse_token_usage


class PromptOptimization(BaseModel):
    """Response format for prompt optimization"""
    optimized_prompt: str = Field(description="The optimized prompt text")


def optimize_prompt(
    state: RecommenderGraphState,
    config: RecommenderGraphConfig
) -> dict[str, Any]:
    """
    Analyze the current prompt's performance and identify areas for improvement.
    
    Args:
        state: Current state of the recommender graph
        config: Configuration for the recommender graph
        
    Returns:
        Dict with optimization results and token usage
    """
    try:
        print("Generating new prompt")
    
        # Get current prompt and evaluation results
        current_prompt = state.get("current_recommender_prompt", "")
        current_score = state.get("current_score", 0.0)
        evaluation_details = state.get("evaluation_details", {})
        optimization_history = state.get("optimization_history", [])
        
        # Get content store for content details
        content_store = state.get("content_store")
        
        # Get top 3 best and worst performing users
        user_performance = []
        for user_id, user_data in evaluation_details.items():
            recall = user_data.get('recall', 0.0)
            user_performance.append((user_id, recall))
        
        # Sort by recall (best first)
        user_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 best and worst (up to 3 each, don't repeat if < 6 users)
        total_users = len(user_performance)
        if total_users < 6:
            # If less than 6 users, take all users
            best_users = user_performance[:3]
            worst_users = user_performance[3:] if total_users > 3 else []
        else:
            # Take top 3 best and bottom 3 worst
            best_users = user_performance[:3]
            worst_users = user_performance[-3:]
        
        # Prepare user details for analysis
        user_details = []
        
        # Process best performing users
        for user_id, recall in best_users:
            user_detail = {
                'user_id': user_id,
                'recall': recall,
                'input_tags': state.get("simulated_tags", {}).get(user_id, []),
                'recommender_actions': state.get("recommender_actions", {}).get(user_id, []),
                'predicted_ids': evaluation_details[user_id].get('predicted_ids', []),
                'ground_truth_ids': evaluation_details[user_id].get('ground_truth_ids', []),
                'content_details': {}
            }
            
            # Get content details for predicted and ground truth IDs
            all_content_ids = set(user_detail['predicted_ids'] + user_detail['ground_truth_ids'])
            for content_id in all_content_ids:
                if content_store:
                    content_row = content_store.get_content_by_id(content_id)
                    if content_row is not None:
                        user_detail['content_details'][content_id] = {
                            'title': content_row.get('title', ''),
                            'intro': content_row.get('intro', ''),
                            'character_list': content_row.get('character_list', ''),
                            'tags': content_row.get('tags', '')
                        }
            
            user_details.append(user_detail)
        
        # Process worst performing users
        for user_id, recall in worst_users:
            user_detail = {
                'user_id': user_id,
                'recall': recall,
                'input_tags': state.get("simulated_tags", {}).get(user_id, []),
                'recommender_actions': state.get("recommender_actions", {}).get(user_id, []),
                'predicted_ids': evaluation_details[user_id].get('predicted_ids', []),
                'ground_truth_ids': evaluation_details[user_id].get('ground_truth_ids', []),
                'content_details': {}
            }
            
            # Get content details for predicted and ground truth IDs
            all_content_ids = set(user_detail['predicted_ids'] + user_detail['ground_truth_ids'])
            for content_id in all_content_ids:
                if content_store:
                    content_row = content_store.get_content_by_id(content_id)
                    if content_row is not None:
                        user_detail['content_details'][content_id] = {
                            'title': content_row.get('title', ''),
                            'intro': content_row.get('intro', ''),
                            'character_list': content_row.get('character_list', ''),
                            'tags': content_row.get('tags', '')
                        }
            
            user_details.append(user_detail)
        
        # Get model from config
        model_name = config.get("configurable", {}).get("optimizer_model", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.4
        )
        
        # Initialize parser
        optimization_parser = PydanticOutputParser(pydantic_object=PromptOptimization)
        
        # Prepare optimization history for context
        history_context = ""
        if optimization_history:
            history_context = "\n\nOPTIMIZATION HISTORY:\n"
            for i, history_item in enumerate(optimization_history[-2:], 1):  # Last 2 optimizations
                history_context += f"Optimization {i}:\n"
                history_context += f"Prompt:\n{history_item.get('prompt', 'N/A')}...\n"
                history_context += f"Score: {history_item.get('score', 'N/A')}\n"
        
        # Create prompt
        optimization_prompt = f"""
You are optimizing the prompt of a content recommendation agent for a roleplay content platform to improve its performance.
The recommender is given a list of content tags and has access to a tool that retrieves k (up to 50) relevant content based on a list of tags sorted by similarity.
The tool uses the mean embeddings of the tags for retrieval.

{history_context}

CURRENT PROMPT:
{current_prompt}
RECALL SCORE: {current_score:.3f}

USER PERFORMANCE ANALYSIS:
Below are the top performing and worst performing users for the current prompt with their complete interaction details:

{user_details}

Analyze the current and past prompts and consider their strengths and weaknesses.

Then generate a new prompt that addresses the identified weaknesses while preserving the strengths. Consider:
1. How the recommendation agent should interact with the tool (e.g. call the tool a single time with all the tags or multiple times each with a subset of the tags, what k value to use)
2. How the recommendation agent should determine which of the retrieved content to return.
3. How to improve the clarity and conciseness of the prompt

Consider patterns from the optimization history to avoid repeating ineffective changes.
The retrieval tool may not perform well. The agent should not rely solely on the tool's ranking.
Assume the agent will always be provided with a list of tags, and that there will be extra prompt appended with output formatting instructions.

{optimization_parser.get_format_instructions()}
        """
        
        # Generate optimized prompt
        response = llm.invoke(optimization_prompt)
        
        # Parse the response
        optimization = optimization_parser.parse(response.content)
        
        # Parse token usage
        current_usage = parse_token_usage(response, model_name)
        
        # Prepare optimization history
        optimization_history = {
            'prompt': current_prompt,
            'score': current_score
        }
        
        # Update prompt if recommended
        print()
        print('=' * 60)
        print(f"NEW OPTIMIZED PROMPT:")
        print(optimization.optimized_prompt)
        print('=' * 60)
        
        return {
            "current_recommender_prompt": optimization.optimized_prompt,
            "optimization_history": [optimization_history],
            "token_usage": {model_name: current_usage},
            "should_continue": True
        }
        
    except Exception as e:
        print(f"Error analyzing current prompt: {e}")
        return {
            "should_continue": False
        }
