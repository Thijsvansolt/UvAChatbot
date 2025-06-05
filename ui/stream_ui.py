import datetime
import streamlit as st
import asyncio
import nest_asyncio
import time
import logging
import sys
from supabase import create_client, Client
import random

from typing import Literal, TypedDict

from ui.multilanguage import TRANSLATIONS
from ui.Chatbot import render_regular_chat
from ui.ABtesting import render_ab_testing

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


class ChatMessage(TypedDict):
    role: Literal['user', 'model']
    timestamp: str
    content: str

def get_text(key: str, lang: str) -> str:
    """Get translated text for given key and language"""
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, key))

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

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🔀 Navigation")
        selected_tab = st.radio(
            "Select a mode",
            options=["chat", "ab"],
            format_func=lambda x: get_text("tab_chat", language) if x == "chat" else get_text("tab_ab_test", language),
            index=0
        )

    # Render the selected view
    if selected_tab == "chat":
        await render_regular_chat(language, supabase)
    elif selected_tab == "ab":
        await render_ab_testing(language, supabase)


    # Sidebar content
    with st.sidebar:
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



if __name__ == "__main__":
    logger.info("Application started from main")
    asyncio.run(main())