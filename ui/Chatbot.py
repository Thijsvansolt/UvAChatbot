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


def save_rating_to_supabase(supabase: Client, timestamp, user_msg, assistant_msg, helpfulness, expectation, response_time):
    """Save rating to Supabase database"""
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
    """Render the regular chat interface"""
    st.title(get_text("main_description", language))
    st.write(get_text("main_description_detail", language))

    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.debug("Initialized empty message history")

    # Initialize ratings dictionary if it doesn't exist
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}
        logger.debug("Initialized empty ratings dictionary")

    logger.debug(f"Current session has {len(st.session_state.messages)} messages")

    # Display chat history with timestamps and ratings
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{get_text('user_label', language)}**\n\n{msg['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**{get_text('assistant_label', language)}**\n\n{msg['content']}")

                # Show response time if available
                if 'response_time' in msg:
                    st.caption(f"⏱️ {get_text('response_time', language)} {msg['response_time']} {get_text('seconds', language)}")

                # Create unique keys for each rating type
                helpfulness_key = f"helpfulness_{i}"
                expectation_key = f"expectation_{i}"
                rating_submitted_key = f"submitted_{i}"

                # Check if this message already has ratings
                if rating_submitted_key not in st.session_state.ratings:
                    st.markdown("---")
                    st.subheader(get_text("rate_response", language))

                    col1, col2 = st.columns(2)

                    with col1:
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

                    # Submit button for ratings
                    if st.button(get_text("submit_rating_button", language), key=f"submit_{i}"):
                        if helpfulness_rating and expectation_rating:
                            logger.info(f"User submitted rating - Message {i}: Helpfulness={helpfulness_rating}, Expectation={expectation_rating}")

                            # Store the ratings in session state
                            st.session_state.ratings[helpfulness_key] = helpfulness_rating
                            st.session_state.ratings[expectation_key] = expectation_rating
                            st.session_state.ratings[rating_submitted_key] = True

                            # Get the corresponding user message
                            user_msg = st.session_state.messages[i-1]['content'] if i > 0 else "N/A"
                            response_time = msg.get('response_time', 'N/A')

                            # Save to Supabase database
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

                            st.rerun()
                        else:
                            st.error(get_text("rating_warning", language))
                            logger.warning("User tried to submit incomplete rating")
                else:
                    # Show the existing ratings
                    helpfulness = st.session_state.ratings.get(helpfulness_key, 0)
                    expectation = st.session_state.ratings.get(expectation_key, 0)

                    st.markdown("---")
                    st.markdown(get_text("your_rating", language))
                    col1, col2 = st.columns(2)

                    with col1:
                        st.info(f"{get_text('helpfulness_rating', language)} {helpfulness}/5 {'⭐' * helpfulness}")

                    with col2:
                        st.info(f"{get_text('expectation_rating', language)} {expectation}/5 {'⭐' * expectation}")

    # Chat input for the user
    user_input = st.chat_input(get_text("chat_placeholder", language))

    if user_input:
        logger.info(f"User submitted new query: {user_input[:50]}...")

        # Add user message to session state
        st.session_state.messages.append(
            {
                "role": "user",
                "timestamp": datetime.datetime.now().isoformat(),
                "content": user_input,
            }
        )

        # Display the user message
        with st.chat_message("user"):
            st.markdown(f"**{get_text('user_label', language)}**\n\n{user_input}")

        # Streaming response
        with st.chat_message("assistant"):
            st.markdown(f"**{get_text('assistant_label', language)}**")

            # Create a placeholder for the streaming response
            response_placeholder = st.empty()
            full_response = ""

            start_time = time.time()
            logger.info("Starting streaming query processing")

            try:
                # Get the streaming response generator
                stream_generator = await process_query(
                    user_input,
                    st.session_state.messages,
                    "gpt-3.5-turbo",
                    streaming=True
                )

                nr_chunks = 0
                # Stream the response chunk by chunk
                async for chunk in stream_generator:
                    full_response += chunk
                    nr_chunks += 1
                    # Update the placeholder with current response + cursor
                    response_placeholder.markdown(full_response + "▌")
                    await asyncio.sleep(0.05)

                # Remove cursor when done
                response_placeholder.markdown(full_response)

                end_time = time.time()
                elapsed_time = round(end_time - start_time, 2)
                logger.info(f"Streaming query processed successfully in {elapsed_time} seconds")

                # Get response time without sleep
                elapsed_time = elapsed_time - (nr_chunks * 0.05)
                elapsed_time = round(elapsed_time, 2)

                # Show response time
                st.caption(f"⏱️ {get_text('response_time', language)} {elapsed_time} {get_text('seconds', language)}")

                # Add assistant message to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "content": full_response,
                        "response_time": elapsed_time
                    }
                )

            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                st.error(f"{get_text('error_occurred', language)} {str(e)}")


        # Rerun to display the new message with rating option
        st.rerun()