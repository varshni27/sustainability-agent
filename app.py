import os
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

import google.generativeai as genai
import logging
from dotenv import load_dotenv
import streamlit as st

# """
# Build a Pinecone vector database from PDFs and CSVs in a folder with DETERMINISTIC IDs,
# so re-running will OVERWRITE existing vectors instead of duplicating.
# """

# CONFIG FOR PINECONE INDEX
DATA_FOLDER_PATH = "data"
INDEX_NAME = "unravel-carbon-assignment"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PDF_CHARS_PER_CHUNK = 1200
PDF_CHARS_OVERLAP = 200
CSV_ROWS_PER_CHUNK = 50
CSV_MAX_ROWS = 20000
CSV_SAMPLE_IF_LARGE = True

PDF_EXTS = {".pdf"}  # file type of documents
CSV_EXTS = {".csv"}


# ENV / RUNTIME SETUP
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in the environment.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# DATA STRUCTURES
@dataclass
class ChunkRecord:
    """Container for metadata of a single PDF or CSV chunk."""
    chunk_id: str
    source_path: str
    file_type: str
    chunk_index: int
    page_or_block: str
    text_preview: str


# HELPERS
def make_id(source_path: str, page_or_block: str, text: str) -> str:
    """Return deterministic MD5 hash from file path, label, and text."""
    raw = f"{os.path.abspath(source_path)}||{page_or_block}||{text}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def iter_files(root: str) -> Iterable[str]:
    """Yield all file paths under a root directory."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def read_pdf_chunks(path: str, chars_per_chunk: int = PDF_CHARS_PER_CHUNK, overlap: int = PDF_CHARS_OVERLAP) -> Iterable[Tuple[str, str]]:
    """Yield overlapping text chunks from a PDF file."""
    reader = PdfReader(path)
    for p_idx, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception as exc:
            logging.warning("Failed to extract text from %s page %d: %s", path, p_idx + 1, exc)
            txt = ""
        if not txt.strip():
            continue
        start = 0
        while start < len(txt):
            end = min(start + chars_per_chunk, len(txt))
            chunk = txt[start:end].strip()
            if chunk:
                yield (f"page:{p_idx+1}", chunk)
            if end == len(txt):
                break
            start = end - overlap


def format_csv_block(df_block: pd.DataFrame) -> str:
    """Convert a DataFrame block to CSV text."""
    return df_block.to_csv(index=False)


def evenly_spaced_blocks(n_rows: int, block_size: int, max_rows: int) -> List[Tuple[int, int]]:
    """Return evenly spaced row ranges covering at most max_rows."""
    if max_rows is None or max_rows >= n_rows:
        return [(s, min(s + block_size, n_rows)) for s in range(0, n_rows, block_size)]
    k = math.ceil(max_rows / block_size)
    step = (n_rows - block_size) / max(1, k - 1)
    return [(int(round(i * step)), min(int(round(i * step)) + block_size, n_rows)) for i in range(k)]


def read_csv_blocks(path: str, rows_per_chunk: int = CSV_ROWS_PER_CHUNK, max_rows: int = CSV_MAX_ROWS, sample_if_large: bool = CSV_SAMPLE_IF_LARGE) -> Iterable[Tuple[str, str]]:
    """Yield labeled row blocks from a CSV file."""
    df = pd.read_csv(path, low_memory=False)
    n_rows = len(df)
    if n_rows == 0:
        return
    if (max_rows is not None) and (n_rows > max_rows) and sample_if_large:
        blocks = evenly_spaced_blocks(n_rows, rows_per_chunk, max_rows)
    else:
        blocks = [(s, min(s + rows_per_chunk, n_rows)) for s in range(0, n_rows, rows_per_chunk)]
    for (start, end) in blocks:
        yield (f"rows:{start}-{end-1}", format_csv_block(df.iloc[start:end]))


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Return row-wise L2 normalized 2D numpy array."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def ensure_pinecone_index(pc: Pinecone, name: str, dim: int) -> None:
    """Create Pinecone index if not already present."""
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name not in existing:
        logging.info("Creating Pinecone index '%s' (dim=%d)...", name, dim)
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    pc.describe_index(name)


# MAIN PIPELINE
def build_pinecone_store() -> None:
    """Build embeddings and upsert PDF/CSV chunks into Pinecone."""
    if not os.path.isdir(DATA_FOLDER_PATH):
        raise FileNotFoundError(f"DATA_FOLDER_PATH does not exist: {DATA_FOLDER_PATH}")

    logging.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    logging.info("Initializing Pinecone client and ensuring index: %s", INDEX_NAME)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_pinecone_index(pc, INDEX_NAME, dim)
    index = pc.Index(INDEX_NAME)

    files = [fp for fp in iter_files(DATA_FOLDER_PATH) if os.path.splitext(fp)[1].lower() in PDF_EXTS.union(CSV_EXTS)]
    logging.info("Discovered %d file(s) to index.", len(files))

    chunk_texts, chunk_meta, chunk_ids = [], [], []

    for fp in tqdm(files, desc="Scanning files"):
        ext = os.path.splitext(fp)[1].lower()
        ftype = "pdf" if ext in PDF_EXTS else "csv"
        try:
            chunks_iter = read_pdf_chunks(fp) if ftype == "pdf" else read_csv_blocks(fp)
            for ci, (pob, text) in enumerate(chunks_iter):
                if not text.strip():
                    continue
                cid = make_id(fp, pob, text)
                preview = (text[:200] + "…") if len(text) > 200 else text
                chunk_ids.append(cid)
                chunk_texts.append(text)
                chunk_meta.append(ChunkRecord(cid, os.path.abspath(fp), ftype, ci, pob, preview))
        except Exception as e:
            logging.warning("Skipping %s due to error: %s", fp, e)

    if not chunk_texts:
        logging.info("No chunks found. Nothing to upsert.")
        return

    logging.info("Embedding %d chunk(s) with %s …", len(chunk_texts), MODEL_NAME)
    embeddings = model.encode(chunk_texts, batch_size=64, show_progress_bar=True)
    embeddings = normalize_rows(np.asarray(embeddings, dtype="float32"))

    logging.info("Upserting to Pinecone with deterministic IDs …")
    BATCH = 100
    for i in tqdm(range(0, len(chunk_texts), BATCH), desc="Upserting"):
        j = min(i + BATCH, len(chunk_texts))
        vectors = []
        for k in range(i, j):
            meta = asdict(chunk_meta[k])
            meta["text"] = chunk_texts[k]
            meta["filename"] = os.path.basename(meta["source_path"]).lower()
            vectors.append({"id": chunk_ids[k], "values": embeddings[k].tolist(), "metadata": meta})
        index.upsert(vectors=vectors)

    logging.info("✅ Done. Upserted %d chunk(s) into index '%s'.", len(chunk_texts), INDEX_NAME)

RUN_BUILD = False # set to TRUE if Pinecone needs to be updated 
if __name__ == "__main__" and RUN_BUILD:
    build_pinecone_store()


# define the prompt for the agent 
sustainability_agent_prompt = """
### Role  
You are a **highly-skilled and respected Emissions Analysis & Insights Agent** with deep expertise in sustainability reporting and greenhouse gas (GHG) accounting.  

---

### Task  
Analyze emissions inventory data and related documents to:  
1. Identify key drivers of emissions  
2. Propose reduction opportunities  
3. Evaluate the quality of calculated emissions  
4. Validate calculations against industry protocols  
5. Educate users on emissions concepts when requested  

---

### Sample Capabilities (not exhaustive)  
- **Summarization:** Generate emissions summary reports with insights from the provided data  
- **Q&A:** Answer natural language queries (e.g., "What were our highest Scope 3 categories?")  
- **Education:** Explain sustainability concepts using GHG Protocol documentation  
- **Quality Checks:** Assess emissions inventory quality & suggest improvements  
- **Validation:** Verify calculations against industry protocols  
- **Classification:** Categorize emissions into Scope 1, 2, or 3 with reasoning  
- **Methodology Guidance:** Describe how to calculate emissions for categories like business travel, stationary combustion, mobile combustion, processing emissions, fugitive emissions, raw materials, packaging material, and manufacturing equipment  
- **Benchmarking:** Compare company emissions with competitors and provide actionable insights that we can learn and adopt from competitors 

---

### Context  
You are provided with:  
- Company records on Scope 1 emissions: {scope1_csv}  
- Company records on Scope 2 emissions: {scope2_csv}  
- Company records on Scope 3 emissions: {scope3_csv}  
- The Greenhouse Gas Protocol documentation: {ghg_protocol_pdf}  
- Sustainability report of peer company 1: {peer1_pdf}  
- Sustainability report of peer company 2: {peer2_pdf}  

---

### Target Audience  
Sustainability experts who want to:  
- Automate emissions calculations  
- Extract insights from emissions data  
- Generate comprehensive sustainability reports  

---

### Reasoning Approach (ReACT + CoT)  
For each query:  
1. **Think step by step** about the relevant context and protocol requirements  
2. **Decide** which source(s) of information are needed (company data, GHG Protocol, peer reports)  
3. **Act** by performing calculations, classifications, or structured reasoning  
4. **Explain clearly** in natural language, showing reasoning where helpful  

---

### Output Style & Format  
- **Style:** Informative, reporting-style, evidence-based  
- **Format:** Natural language response with supporting numbers or reasoning steps where relevant  
- **Optional:** Include structured tables if comparisons are requested  

---

### Few-Shot Examples  

**Example 1: Classification**  
Q: *Classify "electricity purchased from the grid"*  
A: Electricity purchased from the grid is categorized under **Scope 2** emissions, because it represents indirect emissions from the generation of purchased energy consumed by the company.  

**Example 2: Quality Assessment**  
Q: *How can we improve accuracy of stationary combustion data?*  
A: Stationary combustion accuracy can be improved by installing continuous emissions monitoring systems (CEMS), verifying fuel usage logs, and cross-checking with utility invoices to reduce uncertainty.  

---

### Instructions  
Do’s

*Ground answers in the provided data and the GHG Protocol.
*Show the method clearly before giving the result in any calculation.
*State assumptions explicitly if the answer is uncertain.
*Use tables for clarity when making comparisons.
*Take a quantitative, data-driven approach using company data, peer reports, and industry standards.
*Be thorough and accurate in calculations. Errors here can result in serious financial implications. 
*Accept minor rounding errors while comparing with industry protocols. 

Don’ts
*Don’t provide answers without linking back to data or the GHG Protocol.
*Don’t just state final results without the calcualtion method. 
*Don’t ignore or hide assumptions when uncertainty exists.
*Don’t present qualitative comparisons if numerical/structured analysis is possible.
*Don’t make rough or careless estimates in calculations.
"""

# ===== Sustainability RAG Agent with Pinecone + Gemini =====
# Retrieves sustainability context from Pinecone and answers user questions via Gemini.

# Configure APIs
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# -------- Cache Pinecone so it doesn't re-initialize on each interaction --------
@st.cache_resource
def get_pinecone_client() -> Pinecone:
    """Return a cached Pinecone client."""
    return Pinecone(api_key=PINECONE_API_KEY)

@st.cache_resource
def get_index() -> any:
    """Return a cached handle to the Pinecone index."""
    return get_pinecone_client().Index(INDEX_NAME)

index = get_index()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_doc_text(filename: str, top_k: int = 20) -> str:
    """Return concatenated text chunks for a given filename from Pinecone."""
    filename = filename.lower()
    dim = index.describe_index_stats().get("dimension", 1536)
    results = index.query(
        vector=[0.0] * dim,  # dummy vector; rely on metadata filter
        top_k=top_k,
        include_metadata=True,
        filter={"filename": {"$eq": filename}}
    )
    return "\n".join(
        m["metadata"]["text"]
        for m in results.get("matches", [])
        if "metadata" in m and "text" in m["metadata"]
    )

@st.cache_data(show_spinner=False, ttl=3600)
def get_system_message() -> str:
    """Return the formatted system message with retrieved context."""
    return sustainability_agent_prompt.format(
        scope1_csv=fetch_doc_text("scope1.csv"),
        scope2_csv=fetch_doc_text("scope2.csv"),
        scope3_csv=fetch_doc_text("scope3.csv"),
        ghg_protocol_pdf=fetch_doc_text("ghg-protocol-revised.pdf"),
        peer1_pdf=fetch_doc_text("peer1_emissions_report.pdf"),
        peer2_pdf=fetch_doc_text("peer2_emissions_report.pdf"),
    )

# Initialize Gemini (default deterministic config; UI will override temperature)
base_system_message = get_system_message()
model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction=base_system_message,
    generation_config={"temperature": 0.0, "top_p": 1.0, "top_k": 1}
)

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="Emissions Analysis & Insights Agent", page_icon="🌱", layout="wide")
st.title("🌱 Emissions Analysis & Insights Agent")
st.caption("Retrieves sustainability context from Pinecone and answers via Gemini.")

with st.form("qa_form"):
    user_question = st.text_area(
    "1) Please enter your sustainability question",
    height=140,
    placeholder="e.g., Should employee business travel be classified as Scope 1 or Scope 3? "
               "Explain the reasoning and describe how I can calculate my business travel emissions?"
)

    temperature = st.slider("2) Set temperature (creativity)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    submitted = st.form_submit_button("Ask")

if submitted and user_question.strip():
    try:
        # Create a per-request model with the selected temperature (logic unchanged otherwise)
        model_runtime = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=base_system_message,
            generation_config={"temperature": float(temperature), "top_p": 1.0, "top_k": 1}
        )
        st.subheader("Gemini Response")
        placeholder = st.empty()
        chunks = []
        with st.spinner("Thinking..."):
            stream = model_runtime.generate_content(user_question, stream=True)
            for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    chunks.append(chunk.text)
                    placeholder.markdown("".join(chunks))
            stream.resolve()
        final_text = "".join(chunks) if chunks else "_No text returned._"
        placeholder.markdown(final_text)
    except Exception as e:
        st.error(f"LLM call failed: {e}")
elif submitted:
    st.warning("Please enter a question.")

