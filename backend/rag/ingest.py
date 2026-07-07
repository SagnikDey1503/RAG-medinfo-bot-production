"""Ingestion: PDF(s) -> chunks -> OpenAI embeddings -> hybrid index.

Run once (or whenever the source documents change):

    cd backend && python -m rag.ingest

Replaces the old `one.py`. Uses OpenAI embeddings instead of HF MiniLM, so the
index is rebuilt from scratch (the old 384-dim FAISS index is incompatible).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import get_settings
from .embeddings import embed_texts
from .schemas import Chunk
from .vectorstore import HybridStore


def _chunk_id(source: str, page: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{page}:{text[:64]}".encode()).hexdigest()[:12]
    return f"{Path(source).stem[:20]}-{page}-{h}"


def load_pdf_pages(pdf_path: Path) -> List[dict]:
    """Return one record per page: {source, page, text}."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"source": pdf_path.name, "page": i, "text": text})
    return pages


def build_chunks(settings) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    data_dir = Path(settings.data_dir)
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    chunks: List[Chunk] = []
    for pdf in pdfs:
        print(f"  reading {pdf.name} ...")
        for rec in load_pdf_pages(pdf):
            for piece in splitter.split_text(rec["text"]):
                piece = piece.strip()
                if len(piece) < 40:  # skip near-empty fragments
                    continue
                chunks.append(
                    Chunk(
                        id=_chunk_id(rec["source"], rec["page"], piece),
                        text=piece,
                        source=rec["source"],
                        page=rec["page"],
                    )
                )
    return chunks


def main() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY not set (backend/.env). Cannot embed.", file=sys.stderr)
        sys.exit(1)

    print("Building chunks from PDFs ...")
    chunks = build_chunks(settings)
    print(f"  {len(chunks)} chunks created")

    print(f"Embedding with {settings.embedding_model} (batched) ...")
    vectors = embed_texts([c.text for c in chunks])

    print("Building hybrid index (FAISS + BM25) ...")
    store = HybridStore.build(chunks, vectors)
    store.save()
    print(f"Done. Index saved to {settings.index_dir} ({store.size} chunks).")


if __name__ == "__main__":
    main()
