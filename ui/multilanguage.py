# Language translations dictionary
# This dictionary maps language codes ("en", "nl") to a set of UI string translations
# used throughout the chatbot interface. It supports internationalization by dynamically
# rendering content in the user's selected language (English or Dutch).

TRANSLATIONS = {
    "en": {  # English Translations
        # Main titles and descriptions
        "title": "UvA Computer Science Chatbot",
        "sidebar_title": "Computer Science Program at UvA",
        "sidebar_description": "This is a chatbot that can help you with questions about the Computer Science program at the University of Amsterdam. Feel free to ask your questions and receive answers based on the available documentation.",
        "about_title": "About this chatbot",
        "about_description": "This chatbot was developed for the UvA Computer Science program and uses RAG technology to answer questions based on program documentation available on multiple UvA websites.",

        # Rating system section
        "rating_system_title": "Rating System",
        "rating_system_description": "After each response, you can rate the chatbot on two aspects:",
        "helpfulness_label": "🌟 **Helpfulness**: How helpful was the response? (1-5 stars)",
        "expectations_label": "🎯 **Expectations**: Did the response meet your expectations? (1-5 stars)",
        "rating_help_text": "Your ratings help us improve the chatbot!",

        # General feedback section
        "feedback_title": "General Feedback",
        "feedback_description": "If you have any additional feedback or suggestions for improvement, please let us know!",
        "feedback_placeholder": "Your feedback",
        "submit_feedback_button": "Submit Feedback",
        "feedback_success": "Feedback saved successfully!",
        "feedback_error": "Error saving feedback:",
        "feedback_warning": "Please enter some feedback before submitting.",

        # Chat UI elements
        "main_description": "Ask your questions about the Computer Science program at UvA.",
        "main_description_detail": "This chatbot is designed to assist you with inquiries related to the Computer Science program at the University of Amsterdam.",
        "user_label": "User",
        "assistant_label": "Smart Assistant",
        "response_time": "Response generated in",
        "seconds": "seconds",

        # Rating UI
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

        # Chat input
        "chat_placeholder": "Ask your question about the Computer Science program...",
        "generating_answer": "Generating answer...",

        # UI settings
        "language_toggle": "Language / Taal",

        # A/B testing section
        "ab_test_title": "🧪 A/B Testing - Model Comparison",
        "ab_test_description": "Compare responses from two different AI models and help us understand which performs better!",
        "ab_test_instructions": "Enter a question below and you'll receive responses from two different models. Choose which response you prefer.",
        "model_a_label": "Model A Response",
        "model_b_label": "Model B Response",
        "which_better": "Which response do you prefer?",
        "model_a_better": "🅰️ Model A is better",
        "model_b_better": "🅱️ Model B is better",
        "both_equal": "🤝 Both are equally good",
        "submit_comparison": "Submit Comparison",
        "comparison_success": "✅ Thank you for your comparison!",
        "comparison_error": "❌ Error saving comparison:",
        "comparison_warning": "⚠️ Please select which response you prefer before submitting.",
        "ab_placeholder": "Enter your question for model comparison...",
        "generating_responses": "Generating responses from both models...",
        "response_from": "Response from",
        "choose_preference": "👆 Please choose your preference above",

        # Tabs
        "tab_chat": "💬 Regular Chat",
        "tab_ab_test": "🧪 A/B Testing",

        # A/B testing labels
        "ab_user_label": "Your question",
        "ab_comparisons": "Which model's answer is better?",
        "ab_equal": "equally good",
        "ab_chose_A": "✅ You chose Model A",
        "ab_chose_B": "✅ You chose Model B",
        "ab_chose_equal": "🤝 You marked them as Equal",
    },

    "nl": {  # Dutch Translations (structure mirrors English)
        # Main titles and descriptions
        "title": "UvA Informatica Chatbot",
        "sidebar_title": "Informatica Programma aan de UvA",
        "sidebar_description": "Dit is een chatbot die je kan helpen met vragen over het Informatica programma aan de Universiteit van Amsterdam. Stel gerust je vragen en ontvang antwoorden gebaseerd op de beschikbare documentatie.",
        "about_title": "Over deze chatbot",
        "about_description": "Deze chatbot is ontwikkeld voor het UvA Informatica programma en gebruikt RAG technologie om vragen te beantwoorden gebaseerd op programma documentatie beschikbaar op meerdere UvA websites.",

        # Rating system section
        "rating_system_title": "Beoordelingssysteem",
        "rating_system_description": "Na elke reactie kun je de chatbot beoordelen op twee aspecten:",
        "helpfulness_label": "🌟 **Nuttigheid**: Hoe nuttig was het antwoord? (1-5 sterren)",
        "expectations_label": "🎯 **Verwachtingen**: Kwam het antwoord overeen met je verwachtingen? (1-5 sterren)",
        "rating_help_text": "Je beoordelingen helpen ons de chatbot te verbeteren!",

        # General feedback section
        "feedback_title": "Algemene Feedback",
        "feedback_description": "Als je aanvullende feedback of suggesties voor verbetering hebt, laat het ons weten!",
        "feedback_placeholder": "Je feedback",
        "submit_feedback_button": "Feedback Versturen",
        "feedback_success": "Feedback succesvol opgeslagen!",
        "feedback_error": "Fout bij opslaan feedback:",
        "feedback_warning": "Voer alsjeblieft feedback in voordat je verstuurt.",

        # Chat UI elements
        "main_description": "Stel je vragen over het Informatica programma aan de UvA.",
        "main_description_detail": "Deze chatbot is ontworpen om je te helpen met vragen over het Informatica programma aan de Universiteit van Amsterdam.",
        "user_label": "Gebruiker",
        "assistant_label": "Slimme Assistent",
        "response_time": "Antwoord gegenereerd in",
        "seconds": "seconden",

        # Rating UI
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

        # Chat input
        "chat_placeholder": "Stel je vraag over het Informatica programma...",
        "generating_answer": "Antwoord genereren...",

        # UI settings
        "language_toggle": "Language / Taal",

        # A/B testing section
        "ab_test_title": "🧪 A/B Testing - Model Vergelijking",
        "ab_test_description": "Vergelijk reacties van twee verschillende AI-modellen en help ons begrijpen welke beter presteert!",
        "ab_test_instructions": "Voer hieronder een vraag in en je ontvangt reacties van twee verschillende modellen. Kies welke reactie je beter vindt.",
        "model_a_label": "Model A Reactie",
        "model_b_label": "Model B Reactie",
        "which_better": "Welke reactie verkies je?",
        "model_a_better": "🅰️ Model A is beter",
        "model_b_better": "🅱️ Model B is beter",
        "both_equal": "🤝 Beide zijn even goed",
        "submit_comparison": "Vergelijking Versturen",
        "comparison_success": "✅ Bedankt voor je vergelijking!",
        "comparison_error": "❌ Fout bij opslaan vergelijking:",
        "comparison_warning": "⚠️ Selecteer alsjeblieft welke reactie je verkiest voordat je verstuurt.",
        "ab_placeholder": "Voer je vraag in voor modelvergelijking...",
        "generating_responses": "Reacties van beide modellen genereren...",
        "response_from": "Reactie van",
        "choose_preference": "👆 Kies alsjeblieft je voorkeur hierboven",

        # Tabs
        "tab_chat": "💬 Gewone Chat",
        "tab_ab_test": "🧪 A/B Testing",

        # A/B testing labels
        "ab_user_label": "Jouw vraag",
        "ab_comparisons": "Welk models antwoord is beter?",
        "ab_equal": "even goed",
        "ab_chose_A": "✅ Je koos Model A",
        "ab_chose_B": "✅ Je koos Model B",
        "ab_chose_equal": "🤝 Je markeerde ze als Even Goed",
    }
}
