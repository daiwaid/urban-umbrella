from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import StateGraph, START, END
from recommender.nodes import *
from recommender.state import RecommenderGraphState, RecommenderGraphConfig
from recommender.utils import ContentStore, UsersWithInteractions, load_sample_data

#########################
# Initialization
#########################

# Load environment variables
load_dotenv()

# Read in sample data
contents_df, interactions_df, users_df = load_sample_data()

# Create content store
content_store = ContentStore()
# Add content if not loaded from cache
if content_store.get_content_count() < len(contents_df):
    content_store.add_content(contents_df)
    content_store.persist()

# Create full user profiles with past interactions
users_with_interactions = UsersWithInteractions(users_df, interactions_df)

# Create initial state and config
state = RecommenderGraphState(
    content_store=content_store,
    users_with_interactions=users_with_interactions,
    current_iteration=0,
    should_continue=True,
    current_user_id=None,
    current_score=0.0,
    best_score=0.0,
    score_history=[],
    current_recommender_prompt=DEFAULT_RECOMMENDER_PROMPT,
    best_recommender_prompt=DEFAULT_RECOMMENDER_PROMPT,
    content_context_cache_name=None,
    content_context_cache_expires_at=None,
    user_ids=[],
    simulated_tags={},
    ground_truth_ids={},
    recommended_content_ids={},
    recommender_actions={},
    evaluation_details={},
    optimization_history=[],
    token_usage={}
)
config = RecommenderGraphConfig(
    target_score=0.7,
    score_threshold=0.05,
    max_iterations=3,
    users_per_iteration=5,
    recommender_model="gemini-2.0-flash",
    evaluator_model="gemini-2.5-pro",
    optimizer_model="gemini-2.5-flash",
    recursion_limit=10000
)

print('=' * 60)
print(f"STARTING PROMPT:")
print(DEFAULT_RECOMMENDER_PROMPT)
print('=' * 60)


#########################
# Create LangGraph graph
#########################

builder = StateGraph(RecommenderGraphState, config_schema=RecommenderGraphConfig)

def route_new_iteration(state: RecommenderGraphState) -> Literal["sample_user", "optimize_prompt", "END"]:
    """
    Route after new_eval_iteration based on should_continue and iteration number.
    If should_continue is False, route to END.
    If should_continue is True and it's the first iteration, route directly to sample_user.
    If should_continue is True and it's not the first iteration, route to optimize_prompt first.
    """
    should_continue = state.get("should_continue", False)
    current_iteration = state.get("current_iteration", 0)
    
    if not should_continue:
        return "END"
    
    # Skip optimization on first iteration
    if current_iteration == 1:
        return "sample_user"
    else:
        return "optimize_prompt"


def route_finish_user(state: RecommenderGraphState) -> Literal["sample_user", "evaluate_recommendations"]:
    """
    Route after finish_user based on whether there are more users to process.
    """
    should_continue = state.get("should_continue", False)
    
    if should_continue:
        return "sample_user"  # Loop back to sample the next user
    return "evaluate_recommendations"  # All users done, evaluate

# Define the graph
builder.add_node("new_eval_iteration", new_eval_iteration)
builder.add_node("sample_user", sample_user)
builder.add_node("simulate_tags", simulate_new_user_tags)
builder.add_node("generate_ground_truth", generate_ground_truth)
builder.add_node("recommend_content", recommend_content)
builder.add_node("finish_user", finish_user, defer=True)
builder.add_node("evaluate_recommendations", evaluate_recommendations)
builder.add_node("optimize_prompt", optimize_prompt)

# Add edges
builder.add_edge(START, "new_eval_iteration")

# Conditional edge after new_eval_iteration
builder.add_conditional_edges("new_eval_iteration", route_new_iteration, {
    "sample_user": "sample_user",
    "optimize_prompt": "optimize_prompt",
    "END": END
})

# Edges to process a single user
builder.add_edge("optimize_prompt", "sample_user")
builder.add_edge("sample_user", "simulate_tags")
builder.add_edge("sample_user", "generate_ground_truth")
builder.add_edge("generate_ground_truth", "finish_user")
builder.add_edge("simulate_tags", "recommend_content")
builder.add_edge("recommend_content", "finish_user")

# Conditional edge after finish_user
builder.add_conditional_edges("finish_user", route_finish_user, {
    "sample_user": "sample_user",
    "evaluate_recommendations": "evaluate_recommendations"
})

# Add edge to start new iteration
builder.add_edge("evaluate_recommendations", "new_eval_iteration")

# Compile the graph
graph = builder.compile()

# Generate graph image
graph.get_graph().draw_mermaid_png(output_file_path="recommender_graph.png")

# Run the graph
results = graph.invoke(state, config)

# Print final prompt
print('=' * 60)
print(f"\nBest prompt:\n{results.get('best_recommender_prompt', 'N/A')}")
print(f"\nPrompt score: {results.get('best_score', 'N/A')}")
print('=' * 60)

# # Print token usage
print(f"\nTotal token usage by model:\n{results.get('token_usage', {})}")
