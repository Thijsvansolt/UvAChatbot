# https://github.com/coleam00/ottomator-agents/tree/main/crawl4AI-agent

"""
This script crawls web pages, processes their content using AI, and stores the results
in a vector database for semantic search capabilities. It's designed to crawl documentation
or other text-heavy websites and make them searchable using embeddings.

Main workflow:
1. Read URLs from a file
2. Crawl each URL to extract markdown content
3. Split content into manageable chunks
4. Process each chunk with AI to extract titles and summaries
5. Generate embeddings for semantic search
6. Store everything in Supabase database
"""

import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv, find_dotenv

from openai import AsyncOpenAI
from supabase import create_client, Client
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Initialize clients for OpenAI (for AI processing) and Supabase (for data storage)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    """
    Data structure to hold a processed chunk of content from a web page.

    Attributes:
        url: Source URL of the content
        chunk_number: Sequential number of this chunk within the document
        title: AI-generated title for this chunk
        summary: AI-generated summary of the chunk content
        content: The actual text content of the chunk
        metadata: Additional information about the chunk (source, size, etc.)
        embedding: Vector representation of the content for semantic search
    """
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into manageable chunks while preserving paragraph structure.

    This function intelligently splits text by:
    1. Preferring paragraph breaks (\n\n)
    2. Falling back to sentence endings (. )
    3. Ensuring chunks don't get too small (minimum 30% of chunk_size)

    Args:
        text: The text to be chunked
        chunk_size: Target size for each chunk (default: 1000 characters)

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a natural break point (paragraph or sentence)
        chunk = text[start:end]

        # First preference: paragraph break
        if '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            # Only use the break if it's not too close to the beginning
            if last_break > chunk_size * 0.3:
                end = start + last_break

        # Second preference: sentence ending
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        # Add the chunk if it has content
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move to the next chunk position
        start = max(start + chunk_size, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """
    Use AI to extract a meaningful title and summary from a text chunk.

    This function sends the chunk to OpenAI's GPT model to:
    - Generate a descriptive title
    - Create a concise summary
    - Return results in Dutch as specified in the prompt

    Args:
        chunk: The text content to process
        url: Source URL for context

    Returns:
        Dictionary with 'title' and 'summary' keys
    """
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative.
    Give the title and summary in Dutch"""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),  # Use environment variable or default
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }  # Ensure JSON response
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        # Return fallback values if AI processing fails
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """
    Generate vector embeddings for text using OpenAI's embedding model.

    These embeddings enable semantic search - finding content based on meaning
    rather than just keyword matching.

    Args:
        text: The text to generate embeddings for

    Returns:
        List of floating point numbers representing the text embedding
    """
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",  # OpenAI's efficient embedding model
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return zero vector if embedding generation fails
        return [0] * 1536  # Standard dimension for text-embedding-3-small

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """
    Process a single text chunk through the complete AI pipeline.

    This function:
    1. Extracts title and summary using AI
    2. Generates embeddings for semantic search
    3. Creates metadata for tracking
    4. Packages everything into a ProcessedChunk object

    Args:
        chunk: The text content to process
        chunk_number: Sequential number of this chunk
        url: Source URL

    Returns:
        ProcessedChunk object with all processed data
    """
    # Get AI-generated title and summary
    extracted = await get_title_and_summary(chunk, url)

    # Generate vector embedding for semantic search
    embedding = await get_embedding(chunk)

    # Create metadata for tracking and filtering
    metadata = {
        "source": "Datanose",  # Source identifier
        "chunk_size": len(chunk),  # Size for debugging/optimization
        "crawled_at": datetime.now(timezone.utc).isoformat(),  # Timestamp
        "url_path": urlparse(url).path  # URL path for filtering
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """
    Insert a processed chunk into the Supabase database.

    This function stores all the processed data in the 'uva_pages' table,
    including the vector embeddings for semantic search capabilities.

    Args:
        chunk: ProcessedChunk object to store

    Returns:
        Supabase response or None if error
    """
    try:
        # Prepare data for database insertion
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding  # Vector for semantic search
        }

        # Insert into Supabase table (ensure table name matches your schema)
        result = supabase.table("uva_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """
    Process an entire document and store all its chunks in the database.

    This function orchestrates the complete processing pipeline:
    1. Split the document into chunks
    2. Process all chunks in parallel for efficiency
    3. Store all processed chunks in the database

    Args:
        url: Source URL of the document
        markdown: The markdown content to process
    """
    # Split the document into manageable chunks
    chunks = chunk_text(markdown)

    # Process all chunks in parallel for better performance
    tasks = [
        process_chunk(chunk, i, url)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    # Store all chunks in the database in parallel
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """
    Crawl multiple URLs in parallel using crawl4ai.

    This function sets up the web crawler with appropriate configuration
    and processes multiple URLs concurrently while respecting rate limits.

    Args:
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent crawl operations
    """
    # Configure content filtering to remove low-quality content
    prune_filter = PruningContentFilter(
        threshold=0.45,  # Remove content below this relevance score
        threshold_type="dynamic",  # Adaptive threshold
        min_word_threshold=5  # Minimum words to keep content
    )

    # Configure markdown generation with content filtering
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    # Configure the browser for crawling
    browser_config = BrowserConfig(
        headless=False,  # Set to False for interactive debugging
        verbose=True,  # Enable detailed logging
        extra_args=[
            "--disable-gpu",  # Disable GPU acceleration
            "--no-sandbox",  # Disable sandboxing for compatibility
            "--disable-dev-shm-usage",  # Reduce memory usage
            "--disable-blink-features=AutomationControlled",  # Avoid bot detection
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"  # Realistic user agent
        ],
        viewport={"width": 1280, "height": 800}  # Standard viewport size
    )

    # Configure crawling behavior
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # Always fetch fresh content
        markdown_generator=md_generator,  # Use our configured markdown generator
        wait_until="networkidle"  # Wait for network activity to settle
    )

    # Initialize and start the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str, index: int):
            """Process a single URL with concurrency control."""
            async with semaphore:
                # Create unique session ID for this crawl
                session_id = f"session_{index}_{int(datetime.now().timestamp())}"
                try:
                    # Crawl the URL
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id=session_id
                    )

                    if result.success:
                        print(f"✅ Success: {url}")
                        # Process and store the crawled content
                        await process_and_store_document(url, result.markdown.fit_markdown)
                    else:
                        print(f"❌ Failed: {url}\nReason: {result.error_message}")
                except Exception as e:
                    print(f"❗ Exception for {url}: {str(e)}")

        # Process all URLs concurrently
        await asyncio.gather(*[process_url(url, i) for i, url in enumerate(urls)])
    finally:
        # Always clean up the crawler
        await crawler.close()

def get_urls(file_path: str) -> List[str]:
    """
    Read URLs from a text file, filtering for HTTPS URLs only.

    This function reads a file line by line and extracts only lines
    that start with 'https://' to ensure we only crawl secure URLs.

    Args:
        file_path: Path to the file containing URLs

    Returns:
        List of valid HTTPS URLs
    """
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file if line.strip().startswith('https://')]
    return urls

async def main():
    """
    Main function that orchestrates the entire crawling and processing pipeline.

    This function:
    1. Reads URLs from a file
    2. Initiates parallel crawling
    3. Handles the complete workflow from crawling to storage
    """
    # Load URLs from the specified file
    urls = get_urls("urls_to_crawl.txt")
    if not urls:
        print("No URLs found to crawl")
        return

    print(f"Found {len(urls)} URLs to crawl")

    # Start the parallel crawling process
    await crawl_parallel(urls)

# Entry point for the script
if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())