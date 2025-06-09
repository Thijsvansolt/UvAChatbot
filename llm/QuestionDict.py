import os
import openai
import numpy as np
import pickle
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
from supabase import Client, create_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client (for OpenAI v1.0+ usage)
def init_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key
    return openai

# Load environment variables for Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Initialize Supabase client
def init_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables not set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@dataclass
class IntentMatch:
    intent: str
    confidence: float
    extracted_info: Optional[Dict] = None

class IntentClassifier:
    def __init__(self, api_key: str, supabase_client: Client, cache_dir: str = "intent_cache"):
        """
        Initialize the intent classifier with OpenAI embeddings and Supabase storage

        Args:
            api_key: OpenAI API key
            supabase_client: Supabase client instance
            cache_dir: Directory for temporary caching (fallback)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.supabase = supabase_client
        self.model_name = "text-embedding-3-small"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Define intents with example questions
        self.intents = {
            'jaar1': [
                "Welke vakken krijg ik in jaar 1?",
                "Welke vakken zitten er in het eerstejaars curriculum?",
                "Welke modules volg ik in het eerste jaar?",
                "Wat studeer ik in jaar één?",
                "Toon mij de vakken van jaar 1",
                "Eerstejaars curriculum",
                "Vakken in jaar 1",
                "Welke lessen krijg ik in het eerste jaar?",
                "Wat zijn de eerstejaarsvakken?",
                "Welke onderwerpen komen aan bod in jaar 1?",
                "Kan je mij de vakken van het eerste studiejaar tonen?",
                "Welke cursussen volgen we in jaar 1?",
                "Vertel me over de vakken van het eerste jaar",
                "Overzicht van vakken in jaar 1",
                "Welke onderwijsmodules zijn er in jaar één?"
            ],
            'jaar2': [
                "Welke vakken krijg ik in jaar 2?",
                "Welke vakken zitten er in het tweedejaars curriculum?",
                "Welke modules volg ik in het tweede jaar?",
                "Wat studeer ik in jaar twee?",
                "Toon mij de vakken van jaar 2",
                "Tweedejaars curriculum",
                "Vakken in jaar 2",
                "Welke lessen krijg ik in het tweede jaar?",
                "Wat zijn de vakken voor het tweede studiejaar?",
                "Welke onderwerpen komen aan bod in jaar 2?",
                "Kan je mij de vakken van het tweede studiejaar tonen?",
                "Welke cursussen volgen we in jaar 2?",
                "Vertel me over de vakken van het tweede jaar",
                "Overzicht van vakken in jaar 2",
                "Welke onderwijsmodules zijn er in jaar twee?"
            ],
            'jaar3': [
                "Welke vakken krijg ik in jaar 3?",
                "Welke vakken zitten er in het derdejaars curriculum?",
                "Welke modules volg ik in het derde jaar?",
                "Wat studeer ik in jaar drie?",
                "Toon mij de vakken van jaar 3",
                "Derdejaars curriculum",
                "Vakken in jaar 3",
                "Welke lessen krijg ik in het derde jaar?",
                "Wat zijn de vakken voor het derde studiejaar?",
                "Welke onderwerpen komen aan bod in jaar 3?",
                "Kan je mij de vakken van het derde studiejaar tonen?",
                "Welke cursussen volgen we in jaar 3?",
                "Vertel me over de vakken van het derde jaar",
                "Overzicht van vakken in jaar 3",
                "Welke onderwijsmodules zijn er in jaar drie?"
            ],
            'algemeen': [
                "Welke vakken zijn er beschikbaar?",
                "Toon mij alle vakken",
                "Wat kan ik studeren?",
                "Beschikbare modules",
                "Vakkenoverzicht",
                "Welke vakken biedt dit programma aan?",
                "Welke cursussen kan ik volgen?",
                "Overzicht van alle vakken",
                "Welke studieonderdelen zijn er?",
                "Wat zijn de beschikbare vakken?",
                "Kan je mij de volledige lijst met vakken tonen?",
                "Welke modules zijn beschikbaar in dit studieprogramma?",
                "Vertel me over alle vakken",
                "Toon de vakken die ik kan kiezen",
                "Welke opties zijn er qua vakken?"
            ]
        }

        # Load or create embeddings in Supabase
        self.intent_embeddings = self._load_or_create_embeddings()

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding for text: {e}")
            raise

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batch"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            raise

    def upload_embeddings_to_supabase(self):
        """Upload embeddings to Supabase database"""
        try:
            logger.info("Uploading embeddings to Supabase...")

            # First, clear existing embeddings (with WHERE clause for safety)
            try:
                # Check if table has any data first
                existing_data = self.supabase.table("year_information_intent_embeddings").select("id").limit(1).execute()
                if existing_data.data:
                    # Delete all records by using a condition that matches all
                    self.supabase.table("year_information_intent_embeddings").delete().neq("id", -1).execute()
                    logger.info("Cleared existing embeddings from database")
            except Exception as delete_error:
                logger.warning(f"Could not clear existing embeddings (this is OK if table is empty): {delete_error}")

            # Prepare data for batch insert
            embeddings_data = []

            for intent, examples in self.intents.items():
                logger.info(f"Processing embeddings for intent: {intent}")

                # Get embeddings in batch for efficiency
                example_embeddings = self._get_embeddings_batch(examples)

                # Prepare records for database
                for i, (example, embedding) in enumerate(zip(examples, example_embeddings)):
                    embeddings_data.append({
                        "intent": intent,
                        "example_text": example,
                        "embedding": embedding,  # Store as JSON array
                        "example_index": i,
                        "model_name": self.model_name
                    })

            # Insert all embeddings in batches (Supabase has limits)
            batch_size = 100
            for i in range(0, len(embeddings_data), batch_size):
                batch = embeddings_data[i:i + batch_size]
                result = self.supabase.table("year_information_intent_embeddings").insert(batch).execute()
                logger.info(f"Inserted batch {i//batch_size + 1}, records: {len(batch)}")

            logger.info(f"Successfully uploaded {len(embeddings_data)} embeddings to Supabase")
            return True

        except Exception as e:
            logger.error(f"Error uploading embeddings to Supabase: {e}")
            raise

    def _load_embeddings_from_supabase(self) -> Dict[str, np.ndarray]:
        """Load embeddings from Supabase database"""
        try:
            logger.info("Loading embeddings from Supabase...")

            # Fetch all embeddings
            result = self.supabase.table("year_information_intent_embeddings").select("*").execute()

            if not result.data:
                logger.warning("No embeddings found in Supabase")
                return {}

            # Group embeddings by intent
            embeddings = {}
            for record in result.data:
                intent = record["intent"]
                embedding = record["embedding"]

                if intent not in embeddings:
                    embeddings[intent] = []

                embeddings[intent].append(embedding)

            # Convert to numpy arrays
            for intent in embeddings:
                embeddings[intent] = np.array(embeddings[intent])

            logger.info(f"Loaded embeddings for {len(embeddings)} intents from Supabase")
            return embeddings

        except Exception as e:
            logger.error(f"Error loading embeddings from Supabase: {e}")
            raise



    def _load_or_create_embeddings(self) -> Dict[str, np.ndarray]:
        """Load embeddings from Supabase or create new ones"""
        try:
            # Try to load from Supabase first
            embeddings = self._load_embeddings_from_supabase()

            if embeddings:
                return embeddings

            # If no embeddings in Supabase, create and upload them
            logger.info("No embeddings found in Supabase, creating new ones...")
            self.upload_embeddings_to_supabase()

            # Load the newly created embeddings
            return self._load_embeddings_from_supabase()

        except Exception as e:
            logger.error(f"Error in _load_or_create_embeddings: {e}")
            # Fallback to local pickle file if available
            return self._load_from_pickle_fallback()

    def _load_from_pickle_fallback(self) -> Dict[str, np.ndarray]:
        """Fallback method to load from pickle file"""
        cache_file = self.cache_dir / "intent_embeddings.pkl"

        if cache_file.exists():
            logger.info("Loading from pickle file as fallback...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.error("No embeddings available in Supabase or local cache")
        return {}

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _extract_year_info(self, question: str) -> Optional[int]:
        """Extract year information from question"""
        import re

        # Look for year patterns
        year_patterns = [
            r'jaar\s*(\d+)',                        # jaar 1, jaar 2, etc.
            r'(\d+)(?:e|ste|de)?\s*jaar',           # 1e jaar, 2e jaar, 3de jaar, 1ste jaar, etc.
            r'(eerste|tweede|derde)\s*jaar',        # eerste jaar, tweede jaar, derde jaar
        ]

        question_lower = question.lower()

        for pattern in year_patterns:
            match = re.search(pattern, question_lower)
            if match:
                year_text = match.group(1)
                # Convert text to number
                text_to_num = {
                    'eerste': 1, '1e': 1, '1ste': 1, 'jaar 1': 1,
                    'tweede': 2, '2e': 2, '2de': 2, 'jaar 2': 2,
                    'derde': 3, '3e': 3, '3de': 3, 'jaar 3': 3
                }

                if year_text.isdigit():
                    return int(year_text)
                elif year_text in text_to_num:
                    return text_to_num[year_text]

        return None

    def classify_intent(self, user_query: str, threshold: float = 0.7) -> IntentMatch:
        """
        Classify user intent based on cosine similarity with stored embeddings

        Args:
            user_query: User's input query
            threshold: Minimum similarity score to consider a match

        Returns:
            IntentMatch object with intent, confidence, and extracted info
        """
        try:
            # Get embedding for user query
            query_embedding = np.array(self._get_embedding(user_query))

            best_intent = None
            best_confidence = 0.0

            # Compare with all stored embeddings
            for intent, embeddings_matrix in self.intent_embeddings.items():
                # Calculate similarity with each example for this intent
                similarities = []
                for example_embedding in embeddings_matrix:
                    similarity = self._cosine_similarity(query_embedding, example_embedding)
                    similarities.append(similarity)

                # Use the maximum similarity for this intent
                max_similarity = max(similarities) if similarities else 0.0

                if max_similarity > best_confidence:
                    best_confidence = max_similarity
                    best_intent = intent

            # Check if confidence meets threshold
            if best_confidence >= threshold:
                logger.info(f"Intent classified: {best_intent} (confidence: {best_confidence:.3f})")
                return IntentMatch(
                    intent=best_intent,
                    confidence=best_confidence,
                    extracted_info={"query": user_query}
                )
            else:
                logger.info(f"No intent match above threshold. Best: {best_intent} ({best_confidence:.3f})")
                return None

        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return None

    def add_intent_examples(self, intent: str, examples: List[str]):
        """Add new examples to an existing intent"""
        if intent not in self.intents:
            self.intents[intent] = []

        self.intents[intent].extend(examples)

        # Recreate embeddings for this intent
        logger.info(f"Adding new examples to intent: {intent}")
        new_embeddings = self._get_embeddings_batch(examples)

        if intent in self.intent_embeddings:
            # Append to existing embeddings
            self.intent_embeddings[intent] = np.vstack([
                self.intent_embeddings[intent],
                np.array(new_embeddings)
            ])
        else:
            # Create new intent
            self.intent_embeddings[intent] = np.array(new_embeddings)

        # Update cache
        cache_file = self.cache_dir / "intent_embeddings.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(self.intent_embeddings, f)

    def get_document_filters(self, intent_match: IntentMatch) -> List[str]:
        """
        Get document filters based on intent match

        Returns:
            List of document filters: ['jaar1'], ['jaar2'], ['jaar3'], or ['jaar1','jaar2','jaar3']
        """
        if intent_match.intent == 'jaar1':
            return ['jaar1']
        elif intent_match.intent == 'jaar2':
            return ['jaar2']
        elif intent_match.intent == 'jaar3':
            return ['jaar3']
        elif intent_match.intent == 'algemeen':
            return ['jaar1', 'jaar2', 'jaar3']
        else:
            return []

    def format_filters(self, urls: List[str]) -> dict:
        if not urls:
            return {}
        return {
            "url": urls  # Always return as list
        }

# Usage example
if __name__ == "__main__":
    openai_client = init_openai()
    supabase_client = init_supabase()
    # Initialize
    classifier = IntentClassifier(api_key=openai_client.api_key, supabase_client=supabase_client)

    # Upload embeddings (run once)
    # classifier.upload_embeddings_to_supabase()

    # Classify intent
    intent_match = classifier.classify_intent("Welke vakken krijg ik in jaar 1?")
    if intent_match:
        print(f"Intent: {intent_match.intent}, Confidence: {intent_match.confidence:.2f}")
        filters = classifier.get_document_filters(intent_match)
        print(f"Document Filters: {filters}")

    # Get similar questions