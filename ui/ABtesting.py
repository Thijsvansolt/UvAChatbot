# A/B Testing Interface for LLM Model Comparison
"""
This Streamlit application provides an interactive A/B testing interface for comparing
different language models (GPT-3.5-turbo vs GPT-4o-mini). It allows users to evaluate
model performance side-by-side without knowing which model is which, ensuring unbiased
feedback collection.

Key features:
- Blind A/B testing (models are randomly labeled as Model A/B)
- Real-time streaming responses from both models
- Chat-like conversation history
- User preference tracking and database storage
- Multi-language support
- Response time measurement
- Comprehensive analytics data collection

The interface is designed for UX research, model evaluation, and performance analytics
in conversational AI applications.
"""

import datetime
import random
import time
import logging
import sys
import asyncio
import streamlit as st
from supabase import Client

from ui.multilanguage import TRANSLATIONS
from llm.LLMConnection import process_query

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_text(key: str, lang: str) -> str:
    """
    Retrieve translated text for the given key and language.

    This function provides multi-language support for the interface,
    falling back to English if the translation is not available.

    Args:
        key: Translation key to look up
        lang: Target language code (e.g., 'en', 'nl', 'de')

    Returns:
        Translated text string, or the key itself if no translation found
    """
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, key))

def save_ab_comparison_to_supabase(supabase: Client, timestamp, user_question, gpt35_response, o4mini_response, gpt35_time, o4mini_time, user_preference, gpt35_was_model_a, preferred_model):
    """
    Save A/B testing comparison results to Supabase database for analysis.

    This function stores comprehensive data about each A/B test comparison,
    including responses, timing, user preferences, and model assignments.
    This data is crucial for model performance analysis and user experience research.

    Args:
        supabase: Authenticated Supabase client
        timestamp: When the comparison was made
        user_question: The original user query
        gpt35_response: Response from GPT-3.5-turbo model
        o4mini_response: Response from GPT-4o-mini model
        gpt35_time: Response time for GPT-3.5-turbo
        o4mini_time: Response time for GPT-4o-mini
        user_preference: User's choice ("model_a", "model_b", or "equal")
        gpt35_was_model_a: Boolean indicating if GPT-3.5 was shown as Model A
        preferred_model: Actual model name that was preferred

    Returns:
        Tuple of (success: bool, message: str)

    Note:
        The function tracks both the blind preference (Model A/B) and the actual
        model preference to enable comprehensive analysis of model performance.
    """
    logger.info(f"Attempting to save A/B comparison - Preference: {user_preference}, Preferred Model: {preferred_model}")
    try:
        # Prepare comprehensive data for analysis
        data = {
            "timestamp": timestamp,
            "user_question": user_question,
            "gpt35_response": gpt35_response,
            "o4mini_response": o4mini_response,
            "gpt35_response_time": gpt35_time,
            "o4mini_response_time": o4mini_time,
            "user_preference": user_preference,  # Blind preference: "model_a", "model_b", or "equal"
            "gpt35_was_model_a": gpt35_was_model_a,  # Boolean: True if GPT-3.5 was shown as Model A
            "preferred_actual_model": preferred_model  # Actual model: "gpt-3.5-turbo", "o4-mini", or "equal"
        }

        # Insert into database table for A/B test analytics
        result = supabase.table("ab_test_comparisons").insert(data).execute()
        logger.info(f"A/B comparison saved successfully to database: {result.data}")
        return True, "Comparison saved successfully!"
    except Exception as e:
        logger.error(f"Error saving A/B comparison to database: {str(e)}")
        return False, f"Error saving comparison: {str(e)}"

async def process_query_model_a(query, chat_history):
    """
    Process query using GPT-3.5-turbo model with streaming response.

    This function is labeled as "Model A" but could contain either model
    depending on randomization. The actual model assignment is tracked
    separately to maintain blind testing integrity.

    Args:
        query: User's question/input
        chat_history: Previous conversation context

    Returns:
        Tuple of (streaming_response, response_time)
    """
    response, time_taken = await process_query(query, chat_history, "gpt-3.5-turbo", streaming=True)
    return response, time_taken

async def process_query_model_b(query, chat_history):
    """
    Process query using GPT-4o-mini model with streaming response.

    This function is labeled as "Model B" but could contain either model
    depending on randomization. The streaming capability provides real-time
    user feedback during response generation.

    Args:
        query: User's question/input
        chat_history: Previous conversation context

    Returns:
        Tuple of (streaming_response, response_time)
    """
    response, time_taken = await process_query(query, chat_history, "gpt-4o-mini-2024-07-18", streaming=True)
    return response, time_taken

async def stream_to_placeholder(stream_coro, placeholder):
    """
    Stream model response chunks to Streamlit placeholder with typing effect.

    This function creates a realistic chat experience by displaying responses
    as they're generated, with a cursor indicator during typing. This improves
    user experience and provides immediate feedback during longer responses.

    Args:
        stream_coro: Async generator yielding response chunks
        placeholder: Streamlit placeholder to update with streaming content

    Returns:
        Complete response text after streaming is finished

    Note:
        The typing cursor (▌) is removed once streaming completes to show
        the final, clean response.
    """
    full = ""
    async for chunk in stream_coro:
        full += chunk
        placeholder.markdown(full + "▌")  # Add typing cursor
        await asyncio.sleep(0.05)  # Small delay for smooth typing effect
    placeholder.markdown(full)  # Remove cursor when complete
    return full

async def render_ab_testing(language, supabase):
    """
    Render the main A/B testing interface with chat-like conversation experience.

    This is the core function that creates the user interface for A/B testing.
    It manages:
    - Conversation history display
    - Model response streaming
    - User preference collection
    - Database storage coordination
    - Multi-language support

    The interface is designed to feel like a natural chat experience while
    collecting valuable comparison data for model evaluation.

    Args:
        language: Current interface language setting
        supabase: Database client for storing results

    Note:
        Uses Streamlit session state to maintain conversation history
        and prevent data loss during interface reloads.
    """
    # Set up the main interface
    st.title(get_text("ab_test_title", language))
    st.write(get_text("ab_test_description", language))
    st.info(get_text("ab_test_instructions", language))

    # Initialize session state for A/B testing data
    if "ab_comparisons" not in st.session_state:
        st.session_state.ab_comparisons = []  # Store all comparison data

    if "ab_submitted" not in st.session_state:
        st.session_state.ab_submitted = {}  # Track which comparisons have been rated

    # Create container for chat-like conversation display
    chat_container = st.container()

    # Display conversation history in chat format
    with chat_container:
        for i, comparison in enumerate(st.session_state.ab_comparisons):
            # Show user question
            st.markdown(get_text("ab_user_label", language) + f" {i + 1}:", unsafe_allow_html=True)
            st.markdown(f"> {comparison['question']}")

            # Display model responses side-by-side
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model A**")
                with st.expander(f"Response (⏱️ {comparison['model_a_time']}s)", expanded=True):
                    st.write(comparison['model_a_response'])

            with col2:
                st.markdown("**Model B**")
                with st.expander(f"Response (⏱️ {comparison['model_b_time']}s)", expanded=True):
                    st.write(comparison['model_b_response'])

            # Handle user preference selection
            if f"ab_submitted_{i}" in st.session_state.ab_submitted:
                # Show user's previous choice if already submitted
                preference = comparison.get('user_preference', 'Unknown')
                if preference == "model_a":
                    st.success(get_text("ab_chose_A", language))
                elif preference == "model_b":
                    st.success(get_text("ab_chose_B", language))
                else:
                    st.info(get_text("ab_chose_equal", language))
            else:
                # Show preference selection buttons
                st.markdown(f"#### {get_text('ab_comparisons', language)}")
                col_a, col_b, col_equal = st.columns(3)
                with col_a:
                    if st.button("Model A", key=f"choice_a_{i}"):
                        await handle_comparison_choice(supabase, i, "model_a", comparison, language)
                with col_b:
                    if st.button("Model B", key=f"choice_b_{i}"):
                        await handle_comparison_choice(supabase, i, "model_b", comparison, language)
                with col_equal:
                    if st.button(f"🤝 {get_text("ab_equal", language)}", key=f"choice_equal_{i}"):
                        await handle_comparison_choice(supabase, i, "equal", comparison, language)

            st.divider()  # Visual separator between comparisons

    # Chat input for new queries
    user_input = st.chat_input(
        placeholder=get_text("ab_placeholder", language) if "ab_placeholder" in TRANSLATIONS[language] else "Type your message to compare models...",
        key="ab_chat_input"
    )

    # Process new user input
    if user_input:
        logger.info(f"Starting A/B test for query: {user_input[:50]}...")

        # Display user message in chat format
        st.chat_message("user").markdown(f"**You**\n\n{user_input}")

        # Create message containers for streaming responses
        msg_a = st.chat_message("assistant")
        msg_b = st.chat_message("assistant")

        with st.spinner(get_text("generating_responses", language)):
            # Randomize model assignment to ensure blind testing
            a_is_gpt = random.choice([True, False])  # Randomly assign which model is A
            label_a, label_b = "Model A", "Model B"
            model_a_name = "gpt-3.5-turbo" if a_is_gpt else "o4-mini"
            model_b_name = "o4-mini" if a_is_gpt else "gpt-3.5-turbo"

            # Set up response display areas
            msg_a.markdown(f"**{label_a}**")
            msg_b.markdown(f"**{label_b}**")

            # Prepare placeholders for streaming responses
            placeholder_a = msg_a.empty()
            placeholder_b = msg_b.empty()

            # Build chat history from previous comparisons
            chat_history = []
            for comparison in st.session_state.ab_comparisons:
                chat_history.append({"role": "user", "content": comparison["question"]})
                # Use consistent response for context (Model A's response)
                chat_history.append({"role": "assistant", "content": comparison["model_a_response"]})

            # Generate responses from both models in parallel
            results = await asyncio.gather(
                process_query_model_a(user_input, chat_history),
                process_query_model_b(user_input, chat_history)
            )
            (model_a_response, time_taken_a), (model_b_response, time_taken_b) = results

            # Stream both responses simultaneously for better UX
            task_a = asyncio.create_task(stream_to_placeholder(model_a_response, placeholder_a))
            task_b = asyncio.create_task(stream_to_placeholder(model_b_response, placeholder_b))

            # Wait for both streaming tasks to complete
            response_a, response_b = await asyncio.gather(task_a, task_b)

            # Store comprehensive comparison data
            comparison_data = {
                'question': user_input,
                'model_a_response': response_a,
                'model_b_response': response_b,
                'model_a_time': time_taken_a,
                'model_b_time': time_taken_b,
                'model_a_name': label_a,  # Display labels
                'model_b_name': label_b,
                'gpt35_response': response_a if a_is_gpt else response_b,  # Actual GPT-3.5 response
                'o4mini_response': response_b if a_is_gpt else response_a,  # Actual GPT-4o-mini response
                'gpt35_time': time_taken_a if a_is_gpt else time_taken_b,
                'o4mini_time': time_taken_b if a_is_gpt else time_taken_a,
                'gpt35_was_model_a': a_is_gpt,  # Track randomization for analysis
                'actual_model_a': model_a_name,  # Actual model behind Model A
                'actual_model_b': model_b_name   # Actual model behind Model B
            }

            # Add to session state for display and future reference
            st.session_state.ab_comparisons.append(comparison_data)

        # Refresh interface to show new comparison
        st.rerun()

async def handle_comparison_choice(supabase, comparison_index, preference, comparison, language):
    """
    Handle user's preference choice and save results to database.

    This function processes the user's A/B testing decision and stores
    the complete comparison data for analytics. It handles both the blind
    preference (Model A/B) and maps it to the actual model preference
    for comprehensive analysis.

    Args:
        supabase: Database client for storing results
        comparison_index: Index of the comparison in session state
        preference: User's choice ("model_a", "model_b", or "equal")
        comparison: Complete comparison data dictionary
        language: Interface language for user feedback

    The function updates the UI to show the user's choice and prevents
    duplicate submissions for the same comparison.
    """
    logger.info(f"User submitted A/B comparison {comparison_index}: {preference}")

    # Update comparison data with user preference
    st.session_state.ab_comparisons[comparison_index]['user_preference'] = preference
    st.session_state.ab_submitted[f"ab_submitted_{comparison_index}"] = True

    # Map blind preference to actual model preference for analysis
    preferred_model = "equal"  # Default for equal preference
    if preference == "model_a":
        preferred_model = comparison['actual_model_a']
    elif preference == "model_b":
        preferred_model = comparison['actual_model_b']

    # Save comprehensive comparison data to database
    success, message = save_ab_comparison_to_supabase(
        supabase,
        datetime.datetime.now().isoformat(),
        comparison['question'],
        comparison['gpt35_response'],
        comparison['o4mini_response'],
        comparison['gpt35_time'],
        comparison['o4mini_time'],
        preference,  # Blind preference
        comparison['gpt35_was_model_a'],  # Randomization tracking
        preferred_model  # Actual model preference
    )

    # Provide user feedback based on save success
    if success:
        st.success(get_text("comparison_success", language))
        logger.info("A/B comparison saved successfully")
    else:
        st.error(f"{get_text('comparison_error', language)} {message}")
        logger.error(f"A/B comparison save failed: {message}")

    # Refresh interface to show updated state
    st.rerun()