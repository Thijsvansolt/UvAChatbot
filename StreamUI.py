import datetime
import streamlit as st
import asyncio
import nest_asyncio
import time
import logging
import sys
from supabase import create_client, Client

from typing import Literal, TypedDict

# Import functions from LLMConnection.py
from LLMConnection import process_query
nest_asyncio.apply()

# Configure logging for the UI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Supabase configuration - Add your credentials here
SUPABASE_URL = st.secrets["SUPABASE_URL"]  # or replace with your actual URL
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]  # or replace with your actual key

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    logger.info("Initializing Supabase client")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Language translations
TRANSLATIONS = {
    "en": {
        "title": "UvA Computer Science Chatbot",
        "sidebar_title": "Computer Science Program at UvA",
        "sidebar_description": "This is a chatbot that can help you with questions about the Computer Science program at the University of Amsterdam. Feel free to ask your questions and receive answers based on the available documentation.",
        "about_title": "About this chatbot",
        "about_description": "This chatbot was developed for the UvA Computer Science program and uses RAG technology to answer questions based on program documentation available on multiple UvA websites.",
        "rating_system_title": "Rating System",
        "rating_system_description": "After each response, you can rate the chatbot on two aspects:",
        "helpfulness_label": "🌟 **Helpfulness**: How helpful was the response? (1-5 stars)",
        "expectations_label": "🎯 **Expectations**: Did the response meet your expectations? (1-5 stars)",
        "rating_help_text": "Your ratings help us improve the chatbot!",
        "feedback_title": "General Feedback",
        "feedback_description": "If you have any additional feedback or suggestions for improvement, please let us know!",
        "feedback_placeholder": "Your feedback",
        "submit_feedback_button": "Submit Feedback",
        "feedback_success": "Feedback saved successfully!",
        "feedback_error": "Error saving feedback:",
        "feedback_warning": "Please enter some feedback before submitting.",
        "main_description": "Ask your questions about the Computer Science program at UvA.",
        "user_label": "User",
        "assistant_label": "Smart Assistant",
        "response_time": "Response generated in",
        "seconds": "seconds",
        "rate_response": "📊 Rate this Response",
        "helpful_question": "🌟 **How helpful was this answer?**",
        "expectation_question": "🎯 **Did this meet your expectations?**",
        "submit_rating_button": "Submit Rating",
        "rating_success": "✅ Thank you for your rating!",
        "rating_error": "❌",
        "rating_warning": "⚠️ Please provide both helpfulness and expectation ratings before submitting.",
        "your_rating": "📊 **Your Rating:**",
        "helpfulness_rating": "🌟 Helpfulness:",
        "expectation_rating": "🎯 Expectations:",
        "chat_placeholder": "Ask your question about the Computer Science program...",
        "generating_answer": "Generating answer...",
        "language_toggle": "Language / Taal"
    },
    "nl": {
        "title": "UvA Informatica Chatbot",
        "sidebar_title": "Informatica Programma aan de UvA",
        "sidebar_description": "Dit is een chatbot die je kan helpen met vragen over het Informatica programma aan de Universiteit van Amsterdam. Stel gerust je vragen en ontvang antwoorden gebaseerd op de beschikbare documentatie.",
        "about_title": "Over deze chatbot",
        "about_description": "Deze chatbot is ontwikkeld voor het UvA Informatica programma en gebruikt RAG technologie om vragen te beantwoorden gebaseerd op programma documentatie beschikbaar op meerdere UvA websites.",
        "rating_system_title": "Beoordelingssysteem",
        "rating_system_description": "Na elke reactie kun je de chatbot beoordelen op twee aspecten:",
        "helpfulness_label": "🌟 **Nuttigheid**: Hoe nuttig was het antwoord? (1-5 sterren)",
        "expectations_label": "🎯 **Verwachtingen**: Kwam het antwoord overeen met je verwachtingen? (1-5 sterren)",
        "rating_help_text": "Je beoordelingen helpen ons de chatbot te verbeteren!",
        "feedback_title": "Algemene Feedback",
        "feedback_description": "Als je aanvullende feedback of suggesties voor verbetering hebt, laat het ons weten!",
        "feedback_placeholder": "Je feedback",
        "submit_feedback_button": "Feedback Versturen",
        "feedback_success": "Feedback succesvol opgeslagen!",
        "feedback_error": "Fout bij opslaan feedback:",
        "feedback_warning": "Voer alsjeblieft feedback in voordat je verstuurt.",
        "main_description": "Stel je vragen over het Informatica programma aan de UvA.",
        "user_label": "Gebruiker",
        "assistant_label": "Slimme Assistent",
        "response_time": "Antwoord gegenereerd in",
        "seconds": "seconden",
        "rate_response": "📊 Beoordeel dit Antwoord",
        "helpful_question": "🌟 **Hoe nuttig was dit antwoord?**",
        "expectation_question": "🎯 **Kwam dit overeen met je verwachtingen?**",
        "submit_rating_button": "Beoordeling Versturen",
        "rating_success": "✅ Bedankt voor je beoordeling!",
        "rating_error": "❌",
        "rating_warning": "⚠️ Geef alsjeblieft zowel een nuttigheids- als verwachtingsbeoordeling voordat je verstuurt.",
        "your_rating": "📊 **Je Beoordeling:**",
        "helpfulness_rating": "🌟 Nuttigheid:",
        "expectation_rating": "🎯 Verwachtingen:",
        "chat_placeholder": "Stel je vraag over het Informatica programma...",
        "generating_answer": "Antwoord genereren...",
        "language_toggle": "Language / Taal"
    }
}

class ChatMessage(TypedDict):
    role: Literal['user', 'model']
    timestamp: str
    content: str

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

def save_feedback_to_supabase(supabase: Client, feedback_text):
    """Save general feedback to Supabase database"""
    logger.info(f"Attempting to save feedback: {feedback_text[:50]}...")
    try:
        data = {
            "feedback_text": feedback_text,
            "timestamp": datetime.datetime.now().isoformat()
        }

        result = supabase.table("general_feedback").insert(data).execute()
        logger.info(f"Feedback saved successfully to database: {result.data}")
        return True, "Feedback saved successfully!"
    except Exception as e:
        logger.error(f"Error saving feedback to database: {str(e)}")
        return False, f"Error saving feedback: {str(e)}"

async def main():
    logger.info("Starting Streamlit application")

    # Initialize Supabase client
    supabase = init_supabase()

    # Language selection in sidebar
    with st.sidebar:
        st.markdown("### 🌐 " + TRANSLATIONS["en"]["language_toggle"])
        language = st.selectbox(
            "Choose language / Kies taal:",
            options=["en", "nl"],
            format_func=lambda x: "🇬🇧 English" if x == "en" else "🇳🇱 Nederlands",
            index=1  # Default to Dutch
        )
        logger.debug(f"User selected language: {language}")

        st.markdown("---")

        st.title(get_text("sidebar_title", language))
        st.write(get_text("sidebar_description", language))

        with st.expander(get_text("about_title", language)):
            st.write(get_text("about_description", language))

        with st.expander(get_text("rating_system_title", language)):
            st.write(get_text("rating_system_description", language))
            st.write(get_text("helpfulness_label", language))
            st.write(get_text("expectations_label", language))
            st.write(get_text("rating_help_text", language))

        with st.expander(get_text("feedback_title", language)):
            st.write(get_text("feedback_description", language))
            feedback = st.text_area(get_text("feedback_placeholder", language), "")
            if st.button(get_text("submit_feedback_button", language)):
                if feedback.strip():
                    logger.info("User submitted feedback")
                    success, message = save_feedback_to_supabase(supabase, feedback.strip())
                    if success:
                        st.success(get_text("feedback_success", language))
                        logger.info("Feedback submission successful")
                    else:
                        st.error(f"{get_text('feedback_error', language)} {message}")
                        logger.error(f"Feedback submission failed: {message}")
                else:
                    st.warning(get_text("feedback_warning", language))
                    logger.warning("User tried to submit empty feedback")

    # Set up Streamlit UI
    st.title(get_text("title", language))
    st.write(get_text("main_description", language))

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

        # Show a spinner while getting the response
        with st.spinner(get_text("generating_answer", language)):
            start_time = time.time()  # Start timing
            logger.info("Starting query processing")

            # Process query using RAG system
            response = await process_query(user_input, st.session_state.messages)

            end_time = time.time()  # End timing
            elapsed_time = round(end_time - start_time, 2)  # In seconds
            logger.info(f"Query processed successfully in {elapsed_time} seconds")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "content": response,
                    "response_time": elapsed_time  # Store response time in message
                }
            )

        # Rerun to display the new message with rating option
        st.rerun()

if __name__ == "__main__":
    logger.info("Application started from main")
    asyncio.run(main())