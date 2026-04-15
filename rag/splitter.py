# rag/splitter.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from typing import List

# Watermark lines injected by Studocu on every page
JUNK_LINES = [
    "scan to open on studocu",
    "studocu is not sponsored or endorsed",
    "downloaded by",
    "lOMoARcPSD",
]

def clean_page_text(text: str) -> str:
    """
    Remove watermark/junk lines from page text.
    Works line by line — preserves all real content.
    """
    clean_lines = []
    for line in text.split("\n"):
        line_lower = line.lower().strip()
        is_junk    = any(
            pattern.lower() in line_lower
            for pattern in JUNK_LINES
        )
        if not is_junk and line.strip():
            clean_lines.append(line)
    return "\n".join(clean_lines)


def filter_pages(pages: List[Document]) -> List[Document]:
    """
    Clean watermark lines from every page.
    Drop pages that are empty after cleaning.
    """
    cleaned = []
    removed = 0

    for page in pages:
        clean_text = clean_page_text(page.page_content)

        if len(clean_text.strip()) < 50:
            removed += 1
            continue

        cleaned.append(Document(
            page_content = clean_text,
            metadata     = page.metadata
        ))

    print(f"  Pages processed  : {len(pages)}")
    print(f"  Pages removed    : {removed}")
    print(f"  Pages remaining  : {len(cleaned)}")
    return cleaned


def get_recursive_splitter(
    chunk_size: int    = 1000,
    chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size      = chunk_size,
        chunk_overlap   = chunk_overlap,
        length_function = len,
        separators      = ["\n\n", "\n", " ", ""],
    )


def get_character_splitter(
    chunk_size: int    = 1000,
    chunk_overlap: int = 200
) -> CharacterTextSplitter:
    return CharacterTextSplitter(
        separator       = "\n\n",
        chunk_size      = chunk_size,
        chunk_overlap   = chunk_overlap,
        length_function = len,
    )


def split_documents(
    pages: List[Document],
    chunk_size: int    = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Main function used by the rest of the app."""
    clean_pages = filter_pages(pages)
    splitter    = get_recursive_splitter(chunk_size, chunk_overlap)
    chunks      = splitter.split_documents(clean_pages)
    return chunks


def compare_splitters(pages: List[Document]) -> None:
    """Compare both splitters — for learning only."""
    clean_pages = filter_pages(pages)

    if not clean_pages:
        print("  ERROR: No pages remain after filtering.")
        print("  Check JUNK_LINES patterns in splitter.py")
        return

    for name, chunks in [
        ("RecursiveCharacterTextSplitter",
         get_recursive_splitter().split_documents(clean_pages)),
        ("CharacterTextSplitter",
         get_character_splitter().split_documents(clean_pages)),
    ]:
        if not chunks:
            print(f"  {name}: 0 chunks produced.")
            continue

        lengths = [len(c.page_content) for c in chunks]
        print(f"\n{'─'*55}")
        print(f"  {name}")
        print(f"{'─'*55}")
        print(f"  Total chunks : {len(chunks)}")
        print(f"  Min size     : {min(lengths)} chars")
        print(f"  Max size     : {max(lengths)} chars")
        print(f"  Avg size     : {sum(lengths)//len(lengths)} chars")
        print(f"\n  Sample chunk (first 300 chars):")
        print(f"  {chunks[0].page_content[:300]}")
        print(f"\n  Metadata: {chunks[0].metadata}")