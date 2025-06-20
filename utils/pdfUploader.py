# PDF Document Processor with AI Analysis and Vector Storage
"""
This script processes PDF documents through an AI-powered pipeline for semantic search.
It's designed to work alongside the web crawler to create a comprehensive knowledge base
that includes both web content and PDF documents.

Main workflow:
1. Scan a directory for PDF files
2. Extract text content from each PDF using PyMuPDF
3. Split content into manageable chunks
4. Process each chunk with AI to extract Dutch titles and summaries
5. Generate embeddings for semantic search capabilities
6. Store everything in Supabase database for searchability

This complements the web crawler by adding PDF document processing capabilities
to the same vector database infrastructure.
"""

import os
import fitz  # PyMuPDF - powerful PDF processing library
import asyncio
import json
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from openai import AsyncOpenAI
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Initialize clients for AI processing and data storage
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    """
    Data structure for a processed chunk of PDF content.

    This mirrors the structure used in the web crawler to maintain
    consistency across different content sources in the database.

    Attributes:
        url: Identifier for the PDF (using filename as pseudo-URL)
        chunk_number: Sequential number of this chunk within the document
        title: AI-generated Dutch title for this chunk
        summary: AI-generated Dutch summary of the chunk content
        content: The actual extracted text content
        metadata: Additional tracking information (source type, size, timestamps)
        embedding: Vector representation for semantic search
    """
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file using PyMuPDF.

    This function opens a PDF document and extracts text from all pages,
    concatenating them into a single string for further processing.
    PyMuPDF (fitz) is used because it's reliable and handles various PDF formats well.

    Args:
        pdf_path: Path to the PDF file to process

    Returns:
        String containing all extracted text from the PDF

    Note:
        This function handles text extraction but doesn't process images or
        complex layouts. For PDFs with significant formatting, additional
        processing might be needed.
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)
    text = ""

    # Extract text from each page
    for page in doc:
        text += page.get_text()

    # Clean up resources
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split extracted PDF text into manageable chunks while preserving structure.

    This function uses the same chunking logic as the web crawler to ensure
    consistency in how content is processed across different source types.

    The chunking strategy:
    1. Aims for chunks of approximately chunk_size characters
    2. Prefers to break at paragraph boundaries (\n\n)
    3. Falls back to sentence boundaries (. )
    4. Ensures chunks aren't too small (minimum 30% of target size)

    Args:
        text: The extracted PDF text to be chunked
        chunk_size: Target size for each chunk in characters (default: 1000)

    Returns:
        List of text chunks ready for AI processing
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Handle the final chunk
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Look for natural breaking points
        chunk = text[start:end]

        # First preference: paragraph break
        if '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            # Only use if it's not too close to the start (avoids tiny chunks)
            if last_break > chunk_size * 0.3:
                end = start + last_break

        # Second preference: sentence ending
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        # Extract and store the chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move to next chunk position
        start = max(start + chunk_size, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """
    Generate Dutch title and summary for a PDF text chunk using AI.

    This function uses OpenAI's language model to analyze the content
    and generate meaningful titles and summaries in Dutch, making the
    content more discoverable and understandable for Dutch users.

    Args:
        chunk: The text content to analyze
        url: PDF identifier for context (filename-based)

    Returns:
        Dictionary containing 'title' and 'summary' keys with Dutch content

    Note:
        The prompt is in Dutch to ensure consistent Dutch output,
        which is important for maintaining language consistency in the database.
    """
    system_prompt = """Je bent een AI die titels en samenvattingen maakt van tekstfragmenten.
    Geef een JSON met 'title' en 'summary'. Houd het beknopt en informatief."""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),  # Use environment variable or efficient default
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Truncate for API efficiency
            ],
            response_format={"type": "json_object"}  # Ensure structured JSON response
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        # Return Dutch fallback values on error
        return {"title": "Verwerking fout", "summary": "Kon samenvatting niet genereren."}

async def get_embedding(text: str) -> List[float]:
    """
    Generate vector embeddings for PDF text content using OpenAI's embedding model.

    These embeddings enable semantic search across PDF content, allowing users
    to find relevant information based on meaning rather than exact keyword matches.
    This is particularly valuable for academic or technical PDFs where concepts
    might be expressed in different ways.

    Args:
        text: The text content to create embeddings for

    Returns:
        List of float values representing the text as a vector

    Note:
        Uses text-embedding-3-small for cost efficiency while maintaining
        good semantic representation quality.
    """
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",  # Efficient embedding model
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return zero vector on failure (1536 dimensions for text-embedding-3-small)
        return [0.0] * 1536

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """
    Process a single PDF text chunk through the complete AI pipeline.

    This function coordinates all the AI processing steps:
    1. Generate Dutch title and summary using language model
    2. Create vector embeddings for semantic search
    3. Compile metadata for tracking and filtering
    4. Package everything into a ProcessedChunk object

    Args:
        chunk: The text content to process
        chunk_number: Sequential position of this chunk in the document
        url: PDF identifier (filename-based)

    Returns:
        ProcessedChunk object containing all processed data
    """
    # Generate AI-powered title and summary in Dutch
    extracted = await get_title_and_summary(chunk, url)

    # Create vector embedding for semantic search
    embedding = await get_embedding(chunk)

    # Compile metadata for database storage and filtering
    metadata = {
        "source": "Datanose PDF",  # Source type identifier
        "chunk_size": len(chunk),  # Size for optimization analysis
        "crawled_at": datetime.now(timezone.utc).isoformat(),  # Processing timestamp
        "url_path": urlparse(url).path  # Path component for filtering
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """
    Store a processed PDF chunk in the Supabase database.

    This function inserts the processed chunk into the same 'uva_pages' table
    used by the web crawler, ensuring all content sources are stored together
    for unified search capabilities.

    Args:
        chunk: ProcessedChunk object containing all data to store

    Returns:
        Supabase response object or None if insertion failed

    Note:
        Uses the same table structure as the web crawler to maintain
        consistency across different content sources.
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

        # Insert into the same table used by web crawler
        result = supabase.table("uva_pages").insert(data).execute()
        print(f"✅ Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"❌ Error inserting chunk: {e}")
        return None

async def process_pdf_file(pdf_path: str, url_id: str = "local-pdf"):
    """
    Process a complete PDF file through the entire pipeline.

    This function orchestrates the complete processing workflow for a single PDF:
    1. Extract all text content from the PDF
    2. Split content into manageable chunks
    3. Process all chunks in parallel for efficiency
    4. Store all processed chunks in the database

    Args:
        pdf_path: Path to the PDF file to process
        url_id: Identifier to use as pseudo-URL (typically filename without extension)

    Note:
        Processing chunks in parallel significantly improves performance,
        especially when dealing with large PDFs or multiple files.
    """
    print(f"📄 Extracting from PDF: {pdf_path}")

    # Extract all text content from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Split into manageable chunks
    chunks = chunk_text(text)

    # Process all chunks in parallel for better performance
    tasks = [process_chunk(chunk, i, url_id) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)

    # Store all processed chunks in the database in parallel
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)

async def process_all_pdfs():
    """
    Process all PDF files in the designated directory.

    This function serves as the main coordinator for batch PDF processing:
    1. Scans the 'pdfs' directory for PDF files
    2. Processes each PDF file through the complete pipeline
    3. Uses filename (without extension) as the document identifier

    Directory structure expected:
    - Script location: /project/utils/this_script.py
    - PDF directory: /project/pdfs/

    The function automatically discovers all .pdf files and processes them
    sequentially to avoid overwhelming the API rate limits.

    Note:
        PDFs are processed sequentially rather than in parallel to respect
        API rate limits and avoid potential memory issues with large files.
    """
    # Navigate to the PDFs directory (assumes script is in utils/ subdirectory)
    pdf_dir = Path(__file__).parent.parent / "pdfs"  # Go up from utils/ to root, then into pdfs/
    pdf_files = list(pdf_dir.glob("*.pdf"))  # Find all PDF files

    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    # Process each PDF file
    for pdf_file in pdf_files:
        # Use filename without extension as identifier
        url_id = pdf_file.stem  # e.g., "Coursemanual5062KLCR6Y"
        await process_pdf_file(str(pdf_file), url_id=url_id)

# Entry point for standalone execution
if __name__ == "__main__":
    # Run the PDF processing pipeline
    asyncio.run(process_all_pdfs())