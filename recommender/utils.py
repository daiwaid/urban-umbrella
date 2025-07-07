import os
import asyncio
import numpy as np
import pandas as pd
from typing import Any, Optional
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.sklearn import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class TagEmbeddings(Embeddings):
    """
    Custom embeddings class that takes comma-separated tags and returns the mean embedding.
    """
    
    def __init__(self, underlying_embeddings: Embeddings):
        """
        Initialize with an underlying embedding model.
        
        Args:
            underlying_embeddings: The base embedding model to use
        """
        self.underlying_embeddings = underlying_embeddings
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts (comma-separated tags).
        
        Args:
            texts: List of comma-separated tag strings
            
        Returns:
            List of mean embeddings
        """
        embeddings = []
        for text in texts:
            # Split tags and clean them
            tags = [tag.strip() for tag in text.split(',') if tag.strip()]
            
            if not tags:
                # Skip this text if no tags
                continue
            
            # Get embeddings for individual tags
            tag_embeddings = self.underlying_embeddings.embed_documents(tags)
            
            # Calculate mean embedding
            if tag_embeddings:
                mean_embedding = np.mean(tag_embeddings, axis=0)
                embeddings.append(mean_embedding.tolist())
            else:
                embeddings.append([0.0] * self.underlying_embeddings.embedding_dimensions)
        
        return embeddings
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Async version of embed_documents.
        
        Args:
            texts: List of comma-separated tag strings
            
        Returns:
            List of mean embeddings
        """
        embeddings = []
        for text in texts:
            # Split tags and clean them
            tags = [tag.strip() for tag in text.split(',') if tag.strip()]
            
            if not tags:
                # Skip this text if no tags
                continue
            
            # Get embeddings for individual tags
            tag_embeddings = await self.underlying_embeddings.aembed_documents(tags)
            
            # Calculate mean embedding
            if tag_embeddings:
                mean_embedding = np.mean(tag_embeddings, axis=0)
                embeddings.append(mean_embedding.tolist())
            else:
                embeddings.append([0.0] * self.underlying_embeddings.embedding_dimensions)
        
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text (comma-separated tags).
        
        Args:
            text: Comma-separated tag string
            
        Returns:
            Mean embedding
        """
        return self.embed_documents([text])[0]
    
    async def aembed_query(self, text: str) -> list[float]:
        """
        Async version of embed_query.
        
        Args:
            text: Comma-separated tag string
            
        Returns:
            Mean embedding
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]
    
    @property
    def embedding_dimensions(self) -> int:
        """Get the embedding dimensions from the underlying model."""
        return getattr(self.underlying_embeddings, 'embedding_dimensions', 768)


class ContentTags(BaseModel):
    tags: list[str] = Field(description="A single list of tags for the content")


class ContentStore:
    """
    A class to manage content records with embeddings and provide search functionality.
    Uses pandas DataFrames for storage and SKLearnVectorStore for similarity search.
    """
    
    def __init__(self, cache_dir: str = "./cache/", model_name: str = "gemini-2.0-flash"):
        """
        Initialize the ContentStore.
        
        Args:
            cache_dir: Directory for caching embeddings and persisting data
            model_name: Name of the LLM model to use for tag generation
        """
        self.cache_dir = cache_dir
        self.contents_df = pd.DataFrame()
        self.vectorstore = None
        self.model_name = model_name
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup store (loads from disk if available, otherwise initializes fresh)
        self._setup_store()
    
    def _setup_store(self):
        """Setup the vectorstore with caching and persistence, loading from disk if available"""
        # Try to load contents_df from disk first
        try:
            contents_path = os.path.join(self.cache_dir, "contents_df.pkl")
            if os.path.exists(contents_path):
                self.contents_df = pd.read_pickle(contents_path)
                print(f"Loaded contents_df from {contents_path} with {len(self.contents_df)} records")
        except Exception as e:
            print(f"Could not load contents_df from disk: {e}")
            self.contents_df = pd.DataFrame()
        
        # Initialize embedding model with caching in a subfolder
        underlying_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task="SEMANTIC_SIMILARITY_SEARCH")
        embedding_cache_dir = os.path.join(self.cache_dir, "embeddings")
        os.makedirs(embedding_cache_dir, exist_ok=True)
        store = LocalFileStore(embedding_cache_dir)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )
        
        # Create custom tag embeddings
        self.tag_embedder = TagEmbeddings(cached_embedder)
        
        # Initialize vectorstore with persistence in cache_dir
        vectorstore_path = os.path.join(self.cache_dir, "vectorstore.json")
        self.vectorstore = SKLearnVectorStore(
            embedding=self.tag_embedder,
            persist_path=vectorstore_path
        )
        
        # Initialize LLM and parser for tag generation
        self._llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.1
        )
        
        # Initialize parser for structured output
        self._parser = PydanticOutputParser(pydantic_object=ContentTags)
    
    def persist(self):
        """Save contents_df and vectorstore to disk."""
        try:
            # Save contents_df
            contents_path = os.path.join(self.cache_dir, "contents_df.pkl")
            self.contents_df.to_pickle(contents_path)
            print(f"Saved contents_df to {contents_path}")
            
            # Save vectorstore
            if self.vectorstore is not None:
                vectorstore_path = os.path.join(self.cache_dir, "vectorstore.json")
                self.vectorstore.persist()
                print(f"Saved vectorstore to {vectorstore_path}")
                
        except Exception as e:
            print(f"Error persisting data: {e}")
    
    def add_content(self, df: pd.DataFrame, batch_size: int = 20) -> None:
        """
        Add content from a pandas DataFrame.
        
        Args:
            df: DataFrame with any columns, must include 'content_id' as index
            batch_size: Batch size for tag generation if tags column is missing
        """
        asyncio.run(self.aadd_content(df, batch_size))
    
    async def aadd_content(self, df: pd.DataFrame, batch_size: int = 20) -> None:
        """
        Async version of add_content.
        
        Args:
            df: DataFrame with any columns, must include 'content_id' as index
            batch_size: Batch size for tag generation if tags column is missing
        """
        df_processed = df.copy()
        
        # Ensure DataFrame is indexed by content_id
        if df_processed.index.name != 'content_id':
            raise ValueError("DataFrame must have 'content_id' as its index")
        
        # Remove any content rows that have already been added
        if not self.contents_df.empty:
            existing_content_ids = set(self.contents_df.index)
            new_content_ids = set(df_processed.index)
            duplicate_ids = existing_content_ids.intersection(new_content_ids)
            
            if duplicate_ids:
                print(f"Skipping {len(duplicate_ids)} duplicate content rows")
                df_processed = df_processed.drop(index=list(duplicate_ids))
                
                if df_processed.empty:
                    return
        
        # Generate tags if not present
        if 'tags' not in df_processed.columns:
            print("No tags found. Generating tags...")
            tag_results = await self._generate_tags_for_content(df_processed, batch_size)
            
            # Process tags and add directly to tags column
            df_processed['tags'] = df_processed.index.map(lambda content_id: self._process_tags(tag_results.get(content_id, [])))
        
        # Append to existing DataFrame or create new one
        if self.contents_df.empty:
            self.contents_df = df_processed
        else:
            # Use concat with axis=0 to properly handle indexed DataFrames
            self.contents_df = pd.concat([self.contents_df, df_processed], axis=0)
        
        # Create documents for vectorstore
        documents = []
        for content_id, row in df_processed.iterrows():
            tag_text = ', '.join(row['tags']) if row['tags'] else ''
            documents.append(Document(
                page_content=tag_text,
                metadata={'content_id': content_id}
            ))
        
        # Add documents to the vectorstore
        self.vectorstore.add_documents(documents=documents)
    
    def _process_tags(self, tags) -> list[str]:
        """
        Process tags to ensure they are a list of strings.
        
        Args:
            tags: Tags in any format (string, list, or other)
            
        Returns:
            List of tag strings
        """
        if isinstance(tags, str):
            # If tags is a string, split by comma and strip whitespace
            return [tag.strip() for tag in tags.split(',') if tag.strip()]
        elif isinstance(tags, list):
            # If tags is already a list, ensure all items are strings
            return [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            # Convert to string and process
            return [str(tags).strip()] if str(tags).strip() else []
    
    async def _generate_tags_for_content(self, df: pd.DataFrame, batch_size: int = 20) -> dict[int, list[str]]:
        """
        Generate tags for content in the DataFrame.
        
        Args:
            df: DataFrame with content data
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping content_id to tags
        """
        # Create batches using DataFrame indices
        content_ids = df.index.tolist()
        batches = [content_ids[i:i + batch_size] for i in range(0, len(content_ids), batch_size)]
        
        all_results = {}
        
        # Process batches sequentially but with concurrent calls within each batch
        for i, batch in enumerate(batches):
            batch_results = await self._process_content_batch_async(df, batch)
            all_results.update(batch_results)
        
        return all_results
    
    async def _process_content_batch_async(self, df: pd.DataFrame, content_ids_batch: list[int]) -> dict[int, list[str]]:
        """
        Process a batch of content for tag generation.
        
        Args:
            df: DataFrame with content data
            content_ids_batch: List of content IDs to process
            
        Returns:
            Dictionary mapping content_id to tags
        """
        # Create tasks for all contents in the batch
        tasks = [self._generate_content_tags_async(df, content_id) for content_id in content_ids_batch]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = {}
        for i, result in enumerate(results):
            content_id = content_ids_batch[i]
            if isinstance(result, Exception):
                print(f"Error in batch item {i} (content_id {content_id}): {result}")
                processed_results[content_id] = []
            else:
                processed_results[content_id] = result
        
        return processed_results
    
    async def _generate_content_tags_async(self, df: pd.DataFrame, content_id: int) -> list[str]:
        """
        Generate tags for a single content item.
        
        Args:
            df: DataFrame with content data
            content_id: Content ID to process
            
        Returns:
            List of tags
        """
        try:
            content_row = df.loc[content_id]
            title = content_row.get('title', '')
            intro = content_row.get('intro', '')
            character_list = content_row.get('character_list', '')

            prompt = f"""
Analyze the following roleplay content and generate a single list of tags that best describe the content.

Title: {title}
Introduction: {intro}
Characters: {character_list}

Include a diverse set of tags that best describe the content from the following categories:
- Franchises/IPs (e.g. naruto, my hero academia, demon slayer, jujustu kaisen, one piece, pokemon, marvel)
- Content themes/genres (e.g. romance, harem, reverse harem, slice of life, isekai, adventure, drama, horror, scifi)
- Character types (e.g. original character, self-insert, male protagonist, female audience, yandere, tsundere, possessive)
- Relationship dynamics (e.g. love triangle, enemies to lovers, unrequited love, forbidden love, teasing, found family, multiple love interests, forced proximity, power imbalance, romantic rivalry)
- Narrative elements (e.g. fluff, wholesome, obsession, hidden feelings, rivalry, toxic, school bully, office, cyberpunk, zombies, hauntings)

Return a single list of tags (around 10 is typical) as the value of the 'tags' field in your response.

{self._parser.get_format_instructions()}
"""

            # Generate response using async call
            response = await self._llm.ainvoke(prompt)
            parsed = self._parser.parse(response.content)
            return parsed.tags
        except Exception as e:
            print(f"Error processing content {content_id}: {e}")
            return []
    
    def search_by_tags(self, tags: list[str], k: int = 10) -> list[tuple[pd.Series, float]]:
        """
        Search for content by tags and return top k results with full rows.
        
        Args:
            tags: List of tags to search for
            k: Number of top results to return
            
        Returns:
            List of tuples (content_row, similarity_score)
        """
        if not self.vectorstore or not tags:
            return []
        
        try:
            # Create query text from tags
            query_text = ", ".join(tags)
            
            # Search the vectorstore using the custom tag embedder
            docs_and_scores = self.vectorstore.similarity_search_with_score(query_text, k=k)
            
            # Extract full content rows and scores
            results = []
            for doc, score in docs_and_scores:
                content_id = doc.metadata['content_id']
                content_row = self.get_content_by_id(content_id)
                if content_row is not None:
                    results.append((content_row, score))
            
            return results
        except Exception as e:
            print(f"Error searching by tags: {e}")
            return []
    
    def get_content_by_id(self, content_id: int) -> Optional[pd.Series]:
        """
        Get content record by content_id.
        
        Args:
            content_id: The content ID to retrieve
            
        Returns:
            pandas Series or None if not found
        """
        if self.contents_df.empty:
            return None
        
        try:
            return self.contents_df.loc[content_id]
        except KeyError:
            return None
    
    def get_all_content_ids(self) -> list[int]:
        """Get all content IDs in the store."""
        if self.contents_df.empty:
            return []
        return self.contents_df.index.tolist()

    def get_all_content(self) -> pd.DataFrame:
        """Get all content records in the store."""
        return self.contents_df
    
    def get_content_count(self) -> int:
        """Get the total number of content records."""
        return len(self.contents_df)


class UsersWithInteractions:
    """
    A class to store users with their interactions with content and provide sampling functionality.
    """
    
    def __init__(self, users_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Initialize the analyzer with user and interaction data.
        
        Args:
            users_df: DataFrame containing user data with 'user_id' column
            interactions_df: DataFrame containing interaction data with 'user_id', 'content_id', and 'interaction_count' columns
        """
        self.users_df = users_df.copy()
        self.interactions_df = interactions_df.copy()
        
        # Validate required columns
        self._validate_dataframes()
        
        # Compute user content interactions
        self._compute_user_interactions()
    
    def _validate_dataframes(self):
        """Validate that required columns exist in the DataFrames."""
        required_interaction_cols = ['user_id', 'content_id', 'interaction_count']
        
        # Check if users_df is indexed by user_id
        if self.users_df.index.name != 'user_id':
            raise ValueError("users_df must be indexed by 'user_id'")
        
        missing_interaction_cols = [col for col in required_interaction_cols if col not in self.interactions_df.columns]
        if missing_interaction_cols:
            raise ValueError(f"interactions_df missing required columns: {missing_interaction_cols}")
    
    def _compute_user_interactions(self):
        """
        Compute content interactions for each user and add to users_df.
        Each user gets a list of tuples (content_id, interaction_count) sorted by count.
        """
        def get_user_content_interactions(user_id):
            user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
            content_interactions = (
                user_interactions
                .sort_values('interaction_count', ascending=False)
                [['content_id', 'interaction_count']]
                .apply(tuple, axis=1)
                .tolist()
            )
            return content_interactions
        
        # Add content_interactions column to users_df
        self.users_df['content_interactions'] = self.users_df.index.map(get_user_content_interactions)
    
    def get_user_by_id(self, user_id: int) -> Optional[pd.Series]:
        """
        Get user record by user_id.
        
        Args:
            user_id: The user ID to retrieve
            
        Returns:
            pandas Series or None if not found
        """
        if self.users_df.empty:
            return None
        
        try:
            user_row = self.users_df.loc[user_id]
            return user_row
        except KeyError:
            return None
    
    def sample_users(self, n: int, random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Randomly sample n users from the dataset.
        
        Args:
            n: Number of users to sample
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame containing the sampled user records with user_id as index
        """
        if n > len(self.users_df):
            print(f"Warning: Requested {n} users but only {len(self.users_df)} available. Returning all users.")
            n = len(self.users_df)
        
        sampled_users = self.users_df.sample(n=n, random_state=random_state)
        return sampled_users


def load_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load contents.csv, interactions.csv, and users.csv from the sample_data directory.
    
    The function searches for the sample_data folder in the parent directory of the current file,
    and handles cases where the CSV files might be in subdirectories.
    
    Returns:
        tuple: (contents_df, interactions_df, users_df)
        
    Raises:
        FileNotFoundError: If any of the required CSV files cannot be found
    """
    from pathlib import Path
    
    # Get the directory where this utils.py file is located
    current_file_dir = Path(__file__).parent.absolute()
    
    # Look for sample_data in the parent directory
    parent_dir = current_file_dir.parent
    sample_data_dir = parent_dir / "sample_data"
    
    if not sample_data_dir.exists():
        raise FileNotFoundError(f"sample_data directory not found in {parent_dir}")
    
    def find_csv_file(filename: str, search_dir: Path) -> Path:
        """Recursively search for a CSV file in the given directory and its subdirectories."""
        for file_path in search_dir.rglob(filename):
            if file_path.is_file():
                return file_path
        raise FileNotFoundError(f"Could not find {filename} in {search_dir} or its subdirectories")
    
    try:
        # Find and load each CSV file
        contents_path = find_csv_file("contents.csv", sample_data_dir)
        interactions_path = find_csv_file("interactions.csv", sample_data_dir)
        users_path = find_csv_file("users.csv", sample_data_dir)
        
        # Load the DataFrames
        contents_df = pd.read_csv(contents_path)
        interactions_df = pd.read_csv(interactions_path)
        users_df = pd.read_csv(users_path)
        
        # Set index columns
        contents_df.set_index('content_id', inplace=True)
        users_df.set_index('user_id', inplace=True)
        
        return contents_df, interactions_df, users_df
        
    except Exception as e:
        raise FileNotFoundError(f"Error loading data: {e}")


def parse_token_usage(result: dict[str, Any], model_name: str) -> dict[str, int]:
    """
    Parse token usage from the LLM result for a specific model.
    
    Args:
        result: LLM result dictionary
        model_name: Name of the model used
        
    Returns:
        Dictionary with current token counts for this model
    """
    current_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0
    }
    
    try:
        # Extract token usage from the result
        if hasattr(result, 'usage_metadata') and result.usage_metadata:
            usage = result.usage_metadata
            current_usage['input_tokens'] += usage.get('input_tokens', 0)
            current_usage['output_tokens'] += usage.get('output_tokens', 0)
            current_usage['total_tokens'] += usage.get('total_tokens', 0)
    except Exception as e:
        print(f"Error parsing token usage for {model_name}: {e}")
    
    return current_usage
