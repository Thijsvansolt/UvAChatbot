from openai import AsyncOpenAI
from supabase import create_client, Client
import asyncio
from typing import List, Tuple
import os
from dotenv import load_dotenv, find_dotenv
from langdetect import detect

import streamlit as st

# Initialize OpenAI and Supabase clients
SUPABASE_URL = st.secrets["SUPABASE_URL"]  # or replace with your actual URL
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]  # or replace with your actual key

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

async def get_completion(prompt, question, context):
    completion = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content


async def translate_final_response(prompt, question, context, response_language='dutch'):
    if response_language == 'dutch' or response_language == 'nl':
        language_prompt = f"{prompt}\n\nIBelangrijk reageer alleen in het nederlands."
    else:
        language_prompt = f"{prompt}\n\nIMPORTANT: Please respond only in {response_language}."

    corrected_prompt = language_prompt + f"\n\n{context}"
    completion = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": corrected_prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content

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
        lang_mapping = {
            'en': 'english',
            'nl': 'dutch',
            'de': 'german',
            'fr': 'french',
            'es': 'spanish'
        }
        return lang_mapping.get(detected_lang)
    except:
        # Fallback: assume English if detection fails
        return 'Dutch'

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
        return translated_query.strip()
    except Exception as e:
        print(f"Error during query translation: {e}")
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
        return translated_context.strip()
    except Exception as e:
        print(f"Error during context translation: {e}")
        return context  # Return original context on error

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def retrieve_documentation(user_query: str, original_language: str = None):
    """
    Retrieve relevant documentation from Supabase based on the user's query.
    Handles translation for non-Dutch queries.
    """
    # Check if the user query is empty
    if not user_query.strip():
        return "Please provide a valid query."

    try:
        # Detect language if not provided
        if original_language is None:
            original_language = await detect_language(user_query)

        search_query = user_query

        # If the query is not in Dutch, translate it
        if original_language != 'dutch':
            print(f"Detected {original_language} query, translating to Dutch for search...")
            search_query = await translate_query_to_dutch(user_query)
            print(f"Translated query: {search_query}")

        # Get the embedding for the search query
        query_embedding = await get_embedding(search_query)

        # Query Supabase for relevant documents
        result = supabase.rpc(
            'match_uva_pages',
                {
                'query_embedding': query_embedding,
                'match_count': 7,
                'filter': {},
                'similarity_threshold': 0.7  # Adjust this value based on testing
                }
            ).execute()

        # print(f"Supabase result: {result}")
        for doc in result.data:
            print(f"Document ID: {doc['id']}")
            print(f"Document: {doc['title']}, Content: {doc['content']}")

        if not result.data:
            return "No relevant documentation found."

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
                            # {doc['title']}

                            {doc['content']}
                            """
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        dutch_context = "\n\n---\n\n".join(formatted_chunks)
        # If the original query was not in Dutch, translate the context back
        if original_language != 'dutch':
            print(f"Translating context back to {original_language}...")
            translated_context = await translate_context_to_original_language(
                dutch_context,
                original_language
            )
            return translated_context
        print("Not translated, returning Dutch context.")
        return dutch_context

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

async def process_query(user_prompt, chat_history=None):
    """
    Process user query with multilingual support.
    """
    if chat_history is None:
        chat_history = []

    # Detect the language of the user's query
    original_language = await detect_language(user_prompt)
    print(f"Detected language: {original_language}")

    # Format chat history
    # take the last 7 messages for context if available
    chat_history = chat_history[-7:] if len(chat_history) > 7 else chat_history

    # Format chat history for the prompt
    str_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    chat_history_text = f"Chat history:\n{str_history}"

    # Retrieve documentation
    results = await retrieve_documentation(user_prompt, original_language)

    # Combine context
    context = f"{chat_history_text}\n\n{results}"

    # Get the final answer
    answer = await translate_final_response(
        Master_prompt,
        user_prompt,
        context,
        response_language=original_language
    )
    return answer

# Example usage
if __name__ == "__main__":
    print("Ask your question about computer science at UvA in any language (type 'exit' to stop):")
    print("Stel je vraag over informatica aan de UvA in elke taal (typ 'exit' om te stoppen):")

    while True:
        user_input = input("You/Jij: ")
        if user_input.lower() == "exit":
            break

        # Run the async function with asyncio
        response = asyncio.run(process_query(user_input))
        print(f"Smart assistant: {response}")
        print("-" * 50)
