from openai import AsyncOpenAI
import asyncio
from typing import List
import logging
import sys
import time
import json

import streamlit as st
from langdetect import detect
from supabase import create_client, Client
from llm.QuestionDict import IntentClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI and Supabase clients
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

# Initialize OpenAI client
@st.cache_resource
def init_openai():
    api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        raise ValueError("OpenAI API key is not set in secrets.")
    return AsyncOpenAI(api_key=api_key)

openai_client = init_openai()

async def get_completion(prompt, question, context, name_model="gpt-4o-mini-2024-07-18"):
    completion = await openai_client.chat.completions.create(
        model=name_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content

# Streaming version of get_completion
async def get_completion_streaming(prompt, question, context, name_model="gpt-4o-mini-2024-07-18"):
    """Get completion with streaming support."""
    stream = await openai_client.chat.completions.create(
        model=name_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": question},
        ],
        stream=True
    )

    full_response = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content

    # Return the full response at the end for logging/processing
    # return full_response

async def translate_final_response(prompt, question, context, response_language='dutch', name_model="gpt-4o-mini-2024-07-18"):
    if response_language == 'dutch' or response_language == 'nl':
        language_prompt = f"{prompt}\n\nIBelangrijk reageer alleen in het nederlands."
    else:
        language_prompt = f"{prompt}\n\nIMPORTANT: Please respond only in {response_language}."

    corrected_prompt = language_prompt + f"\n\n{context}"
    completion = await openai_client.chat.completions.create(
        model=name_model,
        messages=[
            {"role": "system", "content": corrected_prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content

# Streaming version of translate_final_response
async def translate_final_response_streaming(prompt, question, context, response_language='dutch', name_model="gpt-4o-mini-2024-07-18"):
    """Streaming version of translate_final_response."""
    if response_language == 'dutch' or response_language == 'nl':
        language_prompt = f"{prompt}\n\nIBelangrijk reageer alleen in het nederlands."
    else:
        language_prompt = f"{prompt}\n\nIMPORTANT: Please respond only in {response_language}."

    corrected_prompt = language_prompt + f"\n\n{context}"

    stream = await openai_client.chat.completions.create(
        model=name_model,
        messages=[
            {"role": "system", "content": corrected_prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": question},
        ],
        stream=True
    )

    full_response = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content

    # return full_response

# Updated Master_prompt to handle multiple languages
Master_prompt = """
Je bent een hulpzame en vriendelijke chatbot die vragen beantwoordt over de informaticaopleiding aan de Universiteit van Amsterdam (UvA).

Gebruik uitsluitend de informatie die ik je aanreik om de vraag van de gebruiker te beantwoorden. Deze informatie bestaat uit:
1. Relevante documentatie (context), afkomstig uit de UvA-informatica bronnen of linkedin vacatures.
2. De chatgeschiedenis, die eerdere vragen en antwoorden bevat.
3. De LinkedIn vacature-informatie die ik je heb gegeven zijn alleen van belang wanneer er gevraagd worden naar toekomstige banen of cariere mogelijkheden.
4. De context van de vraag die de gebruiker stelt.

Als de benodigde informatie niet in deze context of chatgeschiedenis staat, zeg dan eerlijk dat je het niet weet en verwijs eventueel de officiële UvA-website.
Je moet de gebruiker ook aanmoedigen om meer vragen te stellen als ze dat willen.

BELANGRIJK: Beantwoord in de taal waarin de gebruiker de vraag stelt. Als de gebruiker in het Engels vraagt, antwoord dan in het Engels. Als de gebruiker in het Nederlands vraagt, antwoord dan in het Nederlands.

Houd je antwoorden helder, beknopt en afgestemd op wat de gebruiker wil weten.
"""

async def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        detected_lang = detect(text)
        # Map language codes to full names for clarity
        logger.info(f"******************************Detected language: {detected_lang}")
        lang_mapping = {
            'en': 'english',
            'nl': 'dutch',
            'de': 'german',
            'fr': 'french',
            'es': 'spanish',
            'af': 'dutch',
        }
        return lang_mapping.get(detected_lang, 'english')
    except Exception as e:
        logger.warning(f"Language detection failed: {e}, defaulting to English")
        # Fallback: assume English if detection fails
        return 'english'

async def translate_query_to_dutch(query: str) -> str:
    """
    Translate the user's query to Dutch for semantic search.
    """
    # Translation prompt focused on university/academic context
    translation_prompt = """
    You are a professional translator specializing in academic translations.
    Translate the following query to Dutch. This query is about computer science education at the University of Amsterdam (UvA).
    Keep academic and university terminology accurate.
    Only return the Dutch translation, nothing else.
    """

    try:
        translated_query = await get_completion(
            translation_prompt,
            query,
            "Translate this query about UvA computer science to Dutch:"
        )
        logger.info(f"Successfully translated query to Dutch")
        return translated_query.strip()
    except Exception as e:
        logger.error(f"Error during query translation: {e}")
        return query  # Return original query on error

async def translate_context_to_original_language(context: str, target_language: str) -> str:
    """
    Translate the retrieved Dutch context back to the user's original language.
    """
    if target_language == 'dutch':
        return context  # No translation needed

    # Translation prompt for context
    translation_prompt = f"""
    You are a professional translator specializing in academic translations.
    Translate the following Dutch text about the University of Amsterdam (UvA) computer science program to {target_language}.
    Maintain the structure and formatting. Keep academic terminology accurate.
    """

    try:
        translated_context = await get_completion(
            translation_prompt,
            context,
            f"Translate this Dutch UvA context to {target_language}:"
        )
        logger.info(f"Successfully translated context to {target_language}")
        return translated_context.strip()
    except Exception as e:
        logger.error(f"Error during context translation: {e}")
        return context  # Return original context on error

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        logger.debug("Successfully generated embedding")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def get_supabase_results(query_embedding: List[float]):
    """Query Supabase for relevant documents by embedding similarity."""
    try:
        result = supabase.rpc(
            'search_by_embedding',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'similarity_threshold': 0.45  # Adjust threshold as needed
            }
        ).execute()

        logger.info(f"Supabase returned {len(result.data) if result.data else 0} documents")
        for doc in result.data or []:
            logger.info(f"Document: {doc['title']} - {doc['summary'][:50]}...")
        return result.data or []
    except Exception as e:
        logger.error(f"Error querying Supabase: {e}")
        return []


async def get_supabase_results_with_filters(filters: dict):
    """Query Supabase for relevant documents with URL filters only."""
    try:
        result = supabase.rpc(
            'search_by_url_filter',
            {
                'filter': filters
            }
        ).execute()

        logger.info(f"Supabase returned {len(result.data) if result.data else 0} documents with filters")
        for doc in result.data or []:
            logger.info(f"Document: {doc['title']} - {doc['summary'][:50]}...")
        return result.data or []
    except Exception as e:
        logger.error(f"Error querying Supabase with filters: {e}")
        return []

def format_documentation_results(results):
    """Format Supabase results into context string."""
    if not results:
        return "No relevant documentation found."

    formatted_chunks = []
    for doc in results:
        chunk_text = f"""
                        # {doc['title']}

                        {doc['content']}
                        """
        formatted_chunks.append(chunk_text)

    return "\n\n---\n\n".join(formatted_chunks)

async def retrieve_documentation(user_query: str, original_language: str = None):
    """
    Retrieve relevant documentation from Supabase based on the user's query.
    Handles translation for non-Dutch queries with concurrent API calls.
    """
    # Check if the user query is empty
    if not user_query.strip():
        logger.warning("Empty user query provided")
        return "Please provide a valid query."

    try:
        # Detect language if not provided
        if original_language is None:
            original_language = await detect_language(user_query)

        search_query = user_query
        logger.info(f"Processing query in {original_language}")

        # If the query is not in Dutch, translate it first
        if original_language != 'dutch':
            logger.info(f"Detected {original_language} query, translating to Dutch for search...")
            search_query = await translate_query_to_dutch(user_query)
            logger.info(f"Translated query: {search_query}")

        classifier = IntentClassifier(api_key=openai_client.api_key, supabase_client=supabase)
        # Classify the intent of the query
        intent_match = classifier.classify_intent(search_query)
        logger.info(f"*******************************Classified intent: {intent_match}")
        if intent_match:
            print(f"Intent: {intent_match.intent}")
            print(f"Confidence: {intent_match.confidence:.3f}")
            print(f"Extracted info: {intent_match.extracted_info}")

            filters = classifier.get_document_filters(intent_match)
            print(f"Document filters: {filters}")
        else:
            print("No specific intent detected - will use semantic search")

        # Get embedding and query Supabase concurrently (these are independent)
        logger.info("Running embedding generation and preparing for database query...")

        # Create tasks for concurrent execution
        embedding_task = asyncio.create_task(get_embedding(search_query))

        # Wait for embedding to complete
        query_embedding = await embedding_task


        # Now query Supabase with the embedding
        logger.info("Querying Supabase for relevant documentation...")
        supabase_results = []
        if intent_match:
            # If intent match is found, use filters for the query
            logger.info("Specific intent detected, using filters for semantic search")
            document_matches = classifier.get_document_filters(intent_match)
            filters = classifier.format_filters(document_matches)
            logger.info(f"Using filters: {filters}")
            supabase_task = asyncio.create_task(
                get_supabase_results_with_filters(filters)
            )
        else:
            # If no intent match, use the basic query
            logger.info("No specific intent detected, using semantic search")
            supabase_task = asyncio.create_task(get_supabase_results(query_embedding))

        supabase_results = await supabase_task

        if not supabase_results:
            logger.warning("No relevant documentation found")
            return "No relevant documentation found."

        # Format the results
        dutch_context = format_documentation_results(supabase_results)
        return dutch_context

    except Exception as e:
        logger.error(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


async def process_query(user_prompt, chat_history=None, model="gpt-4o-mini-2024-07-18", streaming=False):
    """
    Process user query with optimized concurrent API calls and optional streaming.
    """
    if chat_history is None:
        chat_history = []

    start_time = time.time()
    logger.info(f"Processing new user query: {user_prompt[:50]}...")

    # Step 1: Language detection and chat history preparation (fast operations)
    language_task = asyncio.create_task(detect_language(user_prompt))

    # Prepare chat history while language detection runs
    chat_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
    str_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    chat_history_text = f"Chat history:\n{str_history}"

    # Wait for language detection
    original_language = await language_task
    logger.info(f"Detected language: {original_language}")

    # Step 2: Handle documentation retrieval (this includes translation if needed)
    logger.info("Retrieving relevant documentation")
    documentation_task = asyncio.create_task(
        retrieve_documentation(user_prompt, original_language)
    )

    # Wait for documentation retrieval to complete
    documentation_results = await documentation_task

    # Step 3: Combine context
    context = f"{chat_history_text}\n\n{documentation_results}"
    logger.debug(f"Combined context length: {len(context)} characters")

    # Step 4: Generate final response (streaming or non-streaming)
    logger.info("Generating final response")

    # Return an async generator for streaming
    async def stream_response():
        async for chunk in translate_final_response_streaming(
            Master_prompt,
            user_prompt,
            context,
            response_language=original_language,
            name_model=model
        ):
            yield chunk

    if streaming:
        total_time = time.time() - start_time
        logger.info(f"Started streaming response after {total_time:.2f} seconds")
        return stream_response(), round(total_time, 2)

    if not streaming:
        final_response = await translate_final_response(
            Master_prompt,
            user_prompt,
            context,
            response_language=original_language,
            name_model=model
        )
        total_time = time.time() - start_time
        logger.info(f"Generated final response in {total_time:.2f} seconds")
        return final_response, round(total_time, 2)


