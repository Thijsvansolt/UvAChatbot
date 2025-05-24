import datetime
import streamlit as st
import asyncio
import nest_asyncio
import time
from supabase import create_client, Client

from typing import Literal, TypedDict

# Import functions from LLMConnection.py
from LLMConnection import process_query
nest_asyncio.apply()

# Supabase configuration - Add your credentials here
SUPABASE_URL = st.secrets["SUPABASE_URL"]  # or replace with your actual URL
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]  # or replace with your actual key

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

class ChatMessage(TypedDict):
    role: Literal['user', 'model']
    timestamp: str
    content: str

def save_rating_to_supabase(supabase: Client, timestamp, user_msg, assistant_msg, helpfulness, expectation, response_time):
    """Save rating to Supabase database"""
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
        print(result)
        return True, "Rating saved successfully!"
    except Exception as e:
        return False, f"Error saving rating: {str(e)}"

def save_feedback_to_supabase(supabase: Client, feedback_text):
    """Save general feedback to Supabase database"""
    try:
        data = {
            "feedback_text": feedback_text,
            "timestamp": datetime.datetime.now().isoformat()
        }

        result = supabase.table("general_feedback").insert(data).execute()
        print(result)
        return True, "Feedback saved successfully!"
    except Exception as e:
        return False, f"Error saving feedback: {str(e)}"

async def main():
    # Initialize Supabase client
    supabase = init_supabase()

    with st.sidebar:
        st.title("Informatica Program at UvA")
        st.write(
            "This is a chatbot that can help you with questions about the Informatica program at the University of Amsterdam. "
            "Feel free to ask your questions and receive answers based on the available documentation."
        )

        with st.expander("About this chatbot"):
            st.write(
                "This chatbot was developed for the UvA Informatica program and uses "
                "RAG technology to answer questions based on program documentation available on multiple UvA websites."
            )

        with st.expander("Rating System"):
            st.write(
                "After each response, you can rate the chatbot on two aspects:"
            )
            st.write("🌟 **Helpfulness**: How helpful was the response? (1-5 stars)")
            st.write("🎯 **Expectations**: Did the response meet your expectations? (1-5 stars)")
            st.write("Your ratings help us improve the chatbot!")

        with st.expander("General Feedback"):
            st.write(
                "If you have any additional feedback or suggestions for improvement, please let us know!"
            )
            feedback = st.text_area("Your feedback", "")
            if st.button("Submit Feedback"):
                if feedback.strip():
                    success, message = save_feedback_to_supabase(supabase, feedback.strip())
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter some feedback before submitting.")

    # Set up Streamlit UI
    st.title("UvA Informatica Chatbot")
    st.write("Ask your questions about the Informatica program at UvA.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize ratings dictionary if it doesn't exist
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}

    # Display chat history with timestamps and ratings
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**User**\n\n{msg['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Smart Assistant**\n\n{msg['content']}")

                # Show response time if available
                if 'response_time' in msg:
                    st.caption(f"⏱️ Response generated in {msg['response_time']} seconds")

                # Create unique keys for each rating type
                helpfulness_key = f"helpfulness_{i}"
                expectation_key = f"expectation_{i}"
                rating_submitted_key = f"submitted_{i}"

                # Check if this message already has ratings
                if rating_submitted_key not in st.session_state.ratings:
                    st.markdown("---")
                    st.subheader("📊 Rate this Response")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("🌟 **How helpful was this answer?**")
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
                        st.markdown("🎯 **Did this meet your expectations?**")
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
                    if st.button(f"Submit Rating", key=f"submit_{i}"):
                        if helpfulness_rating and expectation_rating:
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
                                st.success("✅ Thank you for your rating!")
                            else:
                                st.error(f"❌ {message}")
                                # Still mark as submitted to prevent repeated attempts

                            st.rerun()
                        else:
                            st.error("⚠️ Please provide both helpfulness and expectation ratings before submitting.")
                else:
                    # Show the existing ratings
                    helpfulness = st.session_state.ratings.get(helpfulness_key, 0)
                    expectation = st.session_state.ratings.get(expectation_key, 0)

                    st.markdown("---")
                    st.markdown("📊 **Your Rating:**")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.info(f"🌟 Helpfulness: {helpfulness}/5 {'⭐' * helpfulness}")

                    with col2:
                        st.info(f"🎯 Expectations: {expectation}/5 {'⭐' * expectation}")

    # Chat input for the user
    user_input = st.chat_input("Ask your question about the Informatica program...")

    if user_input:
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
            st.markdown(f"**User**\n\n{user_input}")

        # Show a spinner while getting the response
        with st.spinner("Generating answer..."):
            start_time = time.time()  # Start timing

            # Process query using RAG system
            response = await process_query(user_input, st.session_state.messages)

            end_time = time.time()  # End timing
            elapsed_time = round(end_time - start_time, 2)  # In seconds

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
    asyncio.run(main())