"""Production agentic RAG package."""
import os

# torch (via sentence-transformers) and faiss both bundle their own OpenMP
# runtime on macOS; loading both aborts the process with OMP Error #15 unless
# duplicate init is explicitly allowed. Must be set before those libs load,
# so it lives here rather than in app.py to cover every entry point.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.0.0"
