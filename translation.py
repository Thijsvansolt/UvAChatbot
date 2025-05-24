from langdetect import detect
from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import List
import os
import streamlit as st

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])


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


async def translate_query(query: str, org_lang: str,) -> tuple [str, str]:
    """
    Translate the user's query to the target language using OpenAI's translation capabilities.
    """
    prompt = f"Translate the following text to Dutch: {query}"

    try:
        translation = await get_completion(prompt, query, context="Translate the following text to Dutch.")
        query = translation
    except Exception as e:
        print(f"Error during translation: {e}")
        return query  # Return original query on error
    return query, org_lang

async def main():
    query = "Hello, how are you?"
    org_lang = detect(query)
    translated_query, org_lang = await translate_query(query, org_lang)
    print(f"Translated Query: {translated_query}, Original Language: {org_lang}")



if __name__ == "__main__":
    asyncio.run(main())