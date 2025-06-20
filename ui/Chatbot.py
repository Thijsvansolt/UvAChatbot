# Import necessary modules
import datetime
import random
import time
import logging
import sys
import asyncio
import streamlit as st
from supabase import Client

# Local imports for UI translations and LLM processing
from ui.multilanguage import TRANSLATIONS
from llm.LLMConnection import process_query

# Configure logging to show timestamps and log levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_text(key: str, lang: str) -> str:
    """Get translated text for the given key and language. Falls back to English if missing."""
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, key))


def save_rating_to_supabase(supabase: Client, timestamp, user_msg, assistant_msg, helpfulness, expectation, response_time):
    """
    Save a user's rating to the Supabase database.
    Includes metadata such as timestamp, question, response, and ratings.
    """
    logger.info(f"Attempting to save rating - Helpfulness: {helpfulness}, Expectation: {expectation}")
    try:
        data = {
            "timestamp": timestamp,
            "user_question": user_msg,
            "assistant_response": assistant_msg,
            "helpfulness_rating": helpfulness,
            "expectation_rating": expectation,
            "response_time": response_time
        }

        result = supabase.table("chatbot_ratings").insert(data).execute()
        logger.info(f"Rating saved successfully to database: {result.data}")
        return True, "Rating saved successfully!"
    except Exception as e:
        logger.error(f"Error saving rating to database: {str(e)}")
        return False, f"Error saving rating: {str(e)}"


async def render_regular_chat(language, supabase):
    """
    Main function to render the chatbot UI and logic using Streamlit.
    Handles:
    - Displaying chat history
    - Accepting new messages
    - Streaming assistant responses
    - Rating system with Supabase persistence
    """
    # Display page title and description
    st.title(get_text("main_description", language))
    st.write(get_text("main_description_detail", language))

    # Initialize session state for messages and ratings
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.debug("Initialized empty message history")

    if "ratings" not in st.session_state:
        st.session_state.ratings = {}
        logger.debug("Initialized empty ratings dictionary")

    logger.debug(f"Current session has {len(st.session_state.messages)} messages")

    # Render message history
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            # Display user message
            with st.chat_message("user"):
                st.markdown(f"**{get_text('user_label', language)}**\n\n{msg['content']}")
        else:
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(f"**{get_text('assistant_label', language)}**\n\n{msg['content']}")

                # Show response time if available
                if 'response_time' in msg:
                    st.caption(f"⏱️ {get_text('response_time', language)} {msg['response_time']} {get_text('seconds', language)}")

                # Unique keys for Streamlit widgets
                helpfulness_key = f"helpfulness_{i}"
                expectation_key = f"expectation_{i}"
                rating_submitted_key = f"submitted_{i}"

                # Check if this assistant message has already been rated
                if rating_submitted_key not in st.session_state.ratings:
                    st.markdown("---")
                    st.subheader(get_text("rate_response", language))

                    # Create two columns for the two rating categories
                    col1, col2 = st.columns(2)

                    with col1:
                        # Helpfulness rating
                        st.markdown(get_text("helpful_question", language))
                        helpfulness_rating = st.radio(
                            "Helpfulness (1-5 stars)",
                            options=[1, 2, 3, 4, 5],
                            index=None,
                            horizontal=True,
                            key=helpfulness_key,
                            label_visibility="collapsed"
                        )
                        if helpfulness_rating:
                            st.markdown("⭐" * helpfulness_rating)

                    with col2:
                        # Expectation rating
                        st.markdown(get_text("expectation_question", language))
                        expectation_rating = st.radio(
                            "Expectations (1-5 stars)",
                            options=[1, 2, 3, 4, 5],
                            index=None,
                            horizontal=True,
                            key=expectation_key,
                            label_visibility="collapsed"
                        )
                        if expectation_rating:
                            st.markdown("⭐" * expectation_rating)

                    # Rating submission button
                    if st.button(get_text("submit_rating_button", language), key=f"submit_{i}"):
                        if helpfulness_rating and expectation_rating:
                            logger.info(f"User submitted rating - Message {i}: Helpfulness={helpfulness_rating}, Expectation={expectation_rating}")

                            # Store ratings in session state
                            st.session_state.ratings[helpfulness_key] = helpfulness_rating
                            st.session_state.ratings[expectation_key] = expectation_rating
                            st.session_state.ratings[rating_submitted_key] = True

                            # Try to fetch user message before the assistant message
                            user_msg = st.session_state.messages[i-1]['content'] if i > 0 else "N/A"
                            response_time = msg.get('response_time', 'N/A')

                            # Save ratings to Supabase
                            success, message = save_rating_to_supabase(
                                supabase,
                                datetime.datetime.now().isoformat(),
                                user_msg,
                                msg['content'],
                                helpfulness_rating,
                                expectation_rating,
                                response_time
                            )

                            if success:
                                st.success(get_text("rating_success", language))
                                logger.info("Rating saved successfully")
                            else:
                                st.error(f"{get_text('rating_error', language)} {message}")
                                logger.error(f"Rating save failed: {message}")

                            st.rerun()  # Refresh to show the submitted ratings
                        else:
                            st.error(get_text("rating_warning", language))
                            logger.warning("User tried to submit incomplete rating")
                else:
                    # Display previously submitted ratings
                    helpfulness = st.session_state.ratings.get(helpfulness_key, 0)
                    expectation = st.session_state.ratings.get(expectation_key, 0)

                    st.markdown("---")
                    st.markdown(get_text("your_rating", language))
                    col1, col2 = st.columns(2)

                    with col1:
                        st.info(f"{get_text('helpfulness_rating', language)} {helpfulness}/5 {'⭐' * helpfulness}")

                    with col2:
                        st.info(f"{get_text('expectation_rating', language)} {expectation}/5 {'⭐' * expectation}")

    # Input field for user to enter a new message
    user_input = st.chat_input(get_text("chat_placeholder", language))

    if user_input:
        logger.info(f"User submitted new query: {user_input[:50]}...")

        # Store the new user message
        st.session_state.messages.append(
            {
                "role": "user",
                "timestamp": datetime.datetime.now().isoformat(),
                "content": user_input,
            }
        )

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(f"**{get_text('user_label', language)}**\n\n{user_input}")

        # Placeholder for assistant response
        msg = st.chat_message("assistant")
        msg.markdown(f"**{get_text('assistant_label', language)}**")

        with st.spinner(get_text("generating_answer", language)):
            # Create placeholder to stream assistant response
            response_placeholder = msg.empty()
            full_response = ""

            start_time = time.time()
            logger.info("Starting streaming query processing")

            try:
                # Call LLM and get async generator for streaming response
                stream_generator, time_taken = await process_query(
                    user_input,
                    st.session_state.messages,
                    "gpt-3.5-turbo",
                    streaming=True
                )

                # Stream and display the assistant response
                async for chunk in stream_generator:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                    await asyncio.sleep(0.05)  # Simulate typing effect

                # Finalize response (remove cursor)
                response_placeholder.markdown(full_response)

                elapsed_time = round(time.time() - start_time, 2)
                logger.info(f"Streaming query processed successfully in {elapsed_time} seconds")

                # Show response time
                st.caption(f"⏱️ {get_text('response_time', language)} {time_taken} {get_text('seconds', language)}")

                # Store assistant message in session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "content": full_response,
                        "response_time": time_taken
                    }
                )

            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                st.error(f"{get_text('error_occurred', language)} {str(e)}")

        st.rerun()  # Refresh to show new message and rating option
