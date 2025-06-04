import datetime
import random
import time
import logging
import sys
import streamlit as st
from supabase import Client

from ui.multilanguage import TRANSLATIONS
from llm.LLMConnection import process_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_text(key: str, lang: str) -> str:
    """Get translated text for given key and language"""
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, key))

def save_ab_comparison_to_supabase(supabase: Client, timestamp, user_question, gpt35_response, o4mini_response, gpt35_time, o4mini_time, user_preference, gpt35_was_model_a, preferred_model):
    """Save A/B testing comparison to Supabase database with efficient model tracking"""
    logger.info(f"Attempting to save A/B comparison - Preference: {user_preference}, Preferred Model: {preferred_model}")
    try:
        data = {
            "timestamp": timestamp,
            "user_question": user_question,
            "gpt35_response": gpt35_response,
            "o4mini_response": o4mini_response,
            "gpt35_response_time": gpt35_time,
            "o4mini_response_time": o4mini_time,
            "user_preference": user_preference,  # "model_a", "model_b", or "equal"
            "gpt35_was_model_a": gpt35_was_model_a,  # Boolean: True if GPT-3.5 was shown as Model A
            "preferred_actual_model": preferred_model  # "gpt-3.5-turbo", "o4-mini", or "equal"
        }

        result = supabase.table("ab_test_comparisons").insert(data).execute()
        logger.info(f"A/B comparison saved successfully to database: {result.data}")
        return True, "Comparison saved successfully!"
    except Exception as e:
        logger.error(f"Error saving A/B comparison to database: {str(e)}")
        return False, f"Error saving comparison: {str(e)}"

# You'll need to implement these functions in your LLMConnection.py or create them
async def process_query_model_a(query, chat_history):
    """Process query using GPT-3.5-turbo"""
    response = await process_query(query, chat_history, "gpt-3.5-turbo")
    return response  # Remove the [Model A] prefix

async def process_query_model_b(query, chat_history):
    """Process query using O4-mini"""
    response = await process_query(query, chat_history, "o4-mini")
    return response  # Remove the [Model B] prefix


async def render_ab_testing(language, supabase):
    """Render the A/B testing interface with chat-like experience"""
    st.title(get_text("ab_test_title", language))
    st.write(get_text("ab_test_description", language))
    st.info(get_text("ab_test_instructions", language))

    # Initialize A/B testing session state
    if "ab_comparisons" not in st.session_state:
        st.session_state.ab_comparisons = []

    if "ab_submitted" not in st.session_state:
        st.session_state.ab_submitted = {}

    # Chat-like display of conversation history
    chat_container = st.container()

    with chat_container:
        for i, comparison in enumerate(st.session_state.ab_comparisons):
            st.markdown(get_text("ab_user_label", language) + f" {i + 1}:", unsafe_allow_html=True)
            st.markdown(f"> {comparison['question']}")

            # Side-by-side model responses
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model A**")
                with st.expander(f"Response (⏱️ {comparison['model_a_time']}s)", expanded=True):
                    st.write(comparison['model_a_response'])

            with col2:
                st.markdown("**Model B**")
                with st.expander(f"Response (⏱️ {comparison['model_b_time']}s)", expanded=True):
                    st.write(comparison['model_b_response'])

            # Choice handling
            if f"ab_submitted_{i}" in st.session_state.ab_submitted:
                preference = comparison.get('user_preference', 'Unknown')
                if preference == "model_a":
                    st.success("✅ You chose Model A")
                elif preference == "model_b":
                    st.success("✅ You chose Model B")
                else:
                    st.info("🤝 You marked them as Equal")
            else:
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

            st.divider()


    with st.container():
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)

        # Use chat_input for a more chat-like experience
        user_input = st.chat_input(
            placeholder=get_text("ab_placeholder", language) if "ab_placeholder" in TRANSLATIONS[language] else "Type your message to compare models...",
            key="ab_chat_input"
        )

        if user_input:
            logger.info(f"Starting A/B test for query: {user_input[:50]}...")

            with st.spinner(get_text("generating_responses", language)):
                # Generate responses from both models simultaneously
                start_time_gpt35 = time.time()
                gpt35_response = await process_query_model_a(user_input, [])
                end_time_gpt35 = time.time()
                gpt35_time = round(end_time_gpt35 - start_time_gpt35, 2)

                start_time_o4mini = time.time()
                o4mini_response = await process_query_model_b(user_input, [])
                end_time_o4mini = time.time()
                o4mini_time = round(end_time_o4mini - start_time_o4mini, 2)

                logger.info(f"A/B responses generated - GPT-3.5: {gpt35_time}s, O4-mini: {o4mini_time}s")

                # Randomly assign which model appears as A or B to avoid bias
                gpt35_as_model_a = random.choice([True, False])

                # Assign model responses and metadata based on random assignment
                if gpt35_as_model_a:
                    model_a_response, model_b_response = gpt35_response, o4mini_response
                    model_a_time, model_b_time = gpt35_time, o4mini_time
                    actual_model_a, actual_model_b = 'gpt-3.5-turbo', 'o4-mini'
                    gpt35_was_model_a = True
                else:
                    model_a_response, model_b_response = o4mini_response, gpt35_response
                    model_a_time, model_b_time = o4mini_time, gpt35_time
                    actual_model_a, actual_model_b = 'o4-mini', 'gpt-3.5-turbo'
                    gpt35_was_model_a = False

                comparison_data = {
                    'question': user_input,
                    'model_a_response': model_a_response,
                    'model_b_response': model_b_response,
                    'model_a_time': model_a_time,
                    'model_b_time': model_b_time,
                    'model_a_name': 'Model A',
                    'model_b_name': 'Model B',
                    # Efficient tracking
                    'gpt35_response': gpt35_response,
                    'o4mini_response': o4mini_response,
                    'gpt35_time': gpt35_time,
                    'o4mini_time': o4mini_time,
                    'gpt35_was_model_a': gpt35_was_model_a,
                    'actual_model_a': actual_model_a,
                    'actual_model_b': actual_model_b
                }

                # Add to session state
                st.session_state.ab_comparisons.append(comparison_data)

            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


async def handle_comparison_choice(supabase, comparison_index, preference, comparison, language):
    """Handle user's comparison choice and save to database"""
    logger.info(f"User submitted A/B comparison {comparison_index}: {preference}")

    # Update the comparison with user preference
    st.session_state.ab_comparisons[comparison_index]['user_preference'] = preference
    st.session_state.ab_submitted[f"ab_submitted_{comparison_index}"] = True

    # Determine which actual model was preferred
    preferred_model = "equal"
    if preference == "model_a":
        preferred_model = comparison['actual_model_a']
    elif preference == "model_b":
        preferred_model = comparison['actual_model_b']

    # Save to database
    success, message = save_ab_comparison_to_supabase(
        supabase,
        datetime.datetime.now().isoformat(),
        comparison['question'],
        comparison['gpt35_response'],
        comparison['o4mini_response'],
        comparison['gpt35_time'],
        comparison['o4mini_time'],
        preference,
        comparison['gpt35_was_model_a'],
        preferred_model
    )

    if success:
        st.success(get_text("comparison_success", language))
        logger.info("A/B comparison saved successfully")
    else:
        st.error(f"{get_text('comparison_error', language)} {message}")
        logger.error(f"A/B comparison save failed: {message}")

    st.rerun()