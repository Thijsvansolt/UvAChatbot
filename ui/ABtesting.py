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

async def process_query_model_a(query, chat_history):
    """Process query using GPT-3.5-turbo"""
    response, time_taken = await process_query(query, chat_history, "gpt-3.5-turbo", streaming=True)
    return response, time_taken

async def process_query_model_b(query, chat_history):
    """Process query using O4-mini"""
    response, time_taken = await process_query(query, chat_history, "o4-mini", streaming=True)
    return response, time_taken

async def stream_to_placeholder(stream_coro, placeholder):
    full = ""
    async for chunk in stream_coro:
        full += chunk
        placeholder.markdown(full + "▌")
        await asyncio.sleep(0.05)
    placeholder.markdown(full)
    return full


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
                    st.success(get_text("ab_chose_A", language))
                elif preference == "model_b":
                    st.success(get_text("ab_chose_B", language))
                else:
                    st.info(get_text("ab_chose_equal", language))
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


    user_input = st.chat_input(
        placeholder=get_text("ab_placeholder", language) if "ab_placeholder" in TRANSLATIONS[language] else "Type your message to compare models...",
        key="ab_chat_input"
    )

    if user_input:
        logger.info(f"Starting A/B test for query: {user_input[:50]}...")

        # Setup chat interface
        st.chat_message("user").markdown(f"**You**\n\n{user_input}")
        msg_a = st.chat_message("assistant")
        msg_b = st.chat_message("assistant")

        with st.spinner(get_text("generating_responses", language)):

            # Randomize label assignment
            a_is_gpt = random.choice([True, False])
            label_a, label_b = "Model A", "Model B"
            model_a_name = "gpt-3.5-turbo" if a_is_gpt else "o4-mini"
            model_b_name = "o4-mini" if a_is_gpt else "gpt-3.5-turbo"

            msg_a.markdown(f"**{label_a}**")
            msg_b.markdown(f"**{label_b}**")

            # Prepare placeholders
            placeholder_a = msg_a.empty()
            placeholder_b = msg_b.empty()

            chat_history = []
            for comparison in st.session_state.ab_comparisons:
                chat_history.append({"role": "user", "content": comparison["question"]})
                # You could choose either model A or model B's response; for consistency, use the same one
                chat_history.append({"role": "assistant", "content": comparison["model_a_response"]})


            # Start both stream coroutines
            results = await asyncio.gather(
                process_query_model_a(user_input, chat_history),
                process_query_model_b(user_input, chat_history)
            )
            (model_a_response, time_taken_a), (model_b_response, time_taken_b) = results


            task_a = asyncio.create_task(stream_to_placeholder(model_a_response, placeholder_a))
            task_b = asyncio.create_task(stream_to_placeholder(model_b_response, placeholder_b))

            # Wait for both to finish
            response_a, response_b = await asyncio.gather(task_a, task_b)

            comparison_data = {
                'question': user_input,
                'model_a_response': response_a,
                'model_b_response': response_b,
                'model_a_time': time_taken_a,
                'model_b_time': time_taken_b,
                'model_a_name': label_a,
                'model_b_name': label_b,
                'gpt35_response': response_a if a_is_gpt else response_b,
                'o4mini_response': response_b if a_is_gpt else response_a,
                'gpt35_time': time_taken_a if a_is_gpt else time_taken_b,
                'o4mini_time': time_taken_b if a_is_gpt else time_taken_a,
                'gpt35_was_model_a': a_is_gpt,
                'actual_model_a': model_a_name,
                'actual_model_b': model_b_name
            }

            st.session_state.ab_comparisons.append(comparison_data)

        st.rerun()



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