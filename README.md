# PDF RAG (Notebook + GitHub Models + Live App)

This folder contains:

- A Jupyter notebook to build the vector database and test Q&A (`pdf_rag_pipeline.ipynb`)
- A simple live web app where you can type a question and get an answer (`app.py`)

Pipeline:

1. Load PDF text (page by page)
2. Split into chunks
3. Embed chunks into vectors (sentence-transformers)
4. Store vectors in FAISS (`vector_store/`)
5. Retrieve relevant chunks and generate an answer (GitHub Models)

## 1) Project Layout

- `data non traite/` : put your PDFs here (can include subfolders)
- `pdf_rag_pipeline.ipynb` : the full pipeline notebook
- `app.py` : live Streamlit app (type a question → answer + sources)
- `vector_store/` : generated FAISS index + metadata
  - `chunks.faiss`
  - `chunks.json`
  - `store_config.json`

## 2) Create Virtual Environment (Windows)

From PowerShell, inside `c:\Users\m2l3k\Desktop\LLM`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel pypdf sentence-transformers faiss-cpu numpy pandas tqdm transformers requests torch streamlit
python -m ipykernel install --user --name llm-pdf-rag --display-name "LLM PDF RAG"
```

Start Jupyter:

```powershell
jupyter lab
```

In Jupyter, select kernel: `LLM PDF RAG`.

## 3) Run The Notebook (End-to-End)

Open `pdf_rag_pipeline.ipynb` and run cells top → bottom.

The notebook is organized as:

1. **Load**: reads PDFs page-by-page
2. **Split**: creates small text chunks for embeddings
3. **Embed**: converts chunks to vectors (sentence-transformers)
4. **Store**: saves vectors + metadata to `vector_store/`
5. **Retrieve & Generate**: answers questions using retrieved chunks

### Test with a small subset first

In the first configuration cell:

- `MAX_PDFS = 20` (fast test)
- then set `MAX_PDFS = None` to index everything

## 4) “Cleaning” / Reset Everything (Common Fixes)

### A) Rebuild the vector database (recommended cleaning)

If you changed chunk sizes, embedding model, or added many new PDFs:

1. Delete the folder `vector_store/`
2. Re-run the notebook top → bottom

This regenerates:

- `vector_store/chunks.faiss`
- `vector_store/chunks.json`
- `vector_store/store_config.json`

### B) Clear and recreate the Python environment

If installs got messy or GPU packages changed:

1. Delete `.venv/`
2. Re-run the commands in section “Create Virtual Environment”

### C) Kernel issues in Jupyter

If your notebook runs but the kernel is wrong/missing:

```powershell
.\.venv\Scripts\Activate.ps1
python -m ipykernel install --user --name llm-pdf-rag --display-name "LLM PDF RAG"
```

Then restart Jupyter.

## 5) GPU Setup (What To Expect)

The notebook automatically uses GPU when available:

- Embeddings: `SentenceTransformer(..., device="cuda")`
- Generation: `pipeline(..., device=0)`

### Check if PyTorch sees your GPU

Run in a notebook cell:

```python
import torch
torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
```

If it returns `False`, generation/embeddings will run on CPU.

### GPU requirements (Windows)

- An NVIDIA GPU
- A recent NVIDIA driver installed

PyTorch GPU wheels include CUDA runtime, but they still require an NVIDIA driver.

If you need a specific CUDA wheel, install PyTorch using the official command from https://pytorch.org/get-started/locally/ (pick Windows + Pip + CUDA).

## 6) Run A Live App (Recommended)

The notebook is great for building/testing, but the easiest “live” experience is the Streamlit app.

1. Build the vector store first by running `pdf_rag_pipeline.ipynb` (it creates `vector_store/`).
2. Start the app:

```powershell
cd c:\Users\m2l3k\Desktop\LLM
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`), type a question, click **Ask**.

## 6.1) Use GitHub Models

This repo supports generating answers with GitHub Models via the inference REST API. The app reads your token from `GITHUB_TOKEN` (environment variable) or from Streamlit secrets.

### A) IMPORTANT: Do not paste tokens in chat or commit them

Set the token only in your environment or in `%userprofile%\.streamlit\secrets.toml`.

### B) Set `GITHUB_TOKEN` (PowerShell)

For the current terminal session:

```powershell
$env:GITHUB_TOKEN="YOUR_TOKEN_HERE"
```

### C) Streamlit secrets (recommended)

Create `%userprofile%\.streamlit\secrets.toml`:

```toml
GITHUB_TOKEN = "YOUR_TOKEN_HERE"
```

Then run:

```powershell
streamlit run app.py
```

In the app sidebar:

- Choose the model id (default: `meta/Llama-3.3-70B-Instruct`)
- Choose answer language

### Disable Streamlit telemetry prompt (optional)

If Streamlit asks for an email or telemetry on first run, create:

- `%userprofile%\.streamlit\config.toml`

With:

```toml
[browser]
gatherUsageStats = false
```

## 7) Fixing “Sequence length is longer than 512”

That warning comes from the generator model (`google/flan-t5-small`) which has a limited input length.

The notebook fixes this in two ways:

- Limits how much retrieved context is inserted into the prompt (character budget)
- Uses `truncation=True` in the generation call

If you still see it:

- Reduce `top_k` (retrieve fewer chunks)
- Reduce `chunk_size` in the split step (smaller chunks)
- Reduce the prompt budget in `build_prompt`

## 8) Asking Questions

Use the last cell:

```python
answer_question("give me the story of the computer", top_k=6)
```

It returns:

- `answer`: generated response
- `sources`: which PDF/page the answer came from

## 9) Troubleshooting

### A) GPU is available but notebook shows `(False, None)`

This usually means the notebook kernel is still running an older environment.

1. `Kernel` → `Change Kernel` → select `LLM PDF RAG`
2. `Kernel` → `Restart Kernel`
3. Re-run:

```python
import torch
torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
```

## 10) Evaluation (Confusion Matrix + Curves)

This project does not train a classifier, so there is no ``training loss curve`` by default.
Instead, you can evaluate the retrieval system like a binary classifier:

- Each retrieved chunk is labeled as relevant (1) or not relevant (0)
- Similarity score is treated as a prediction score
- You can plot confusion matrix, ROC curve, PR curve, and Precision@k / Recall@k curves

### A) Create a labeled evaluation file

Create `eval/questions.csv` with:

- `question`: your query
- `relevant_sources`: one or more PDF filenames separated by `;`

Example file: `eval/questions_example.csv`

### B) Run evaluation

```powershell
cd c:\Users\m2l3k\Desktop\LLM
.\.venv\Scripts\Activate.ps1
python evaluate_rag.py --eval-csv eval/questions.csv --top-k 6 --out-dir reports/eval
```

### C) Outputs

Saved to `reports/eval/`:

- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `precision_at_k.png`
- `recall_at_k.png`
- `summary.json`
- `per_query_metrics.csv`
- `detailed_hits.csv`

## 11) Quiz Interface

The Streamlit app includes a `Quiz` tab that generates a multiple-choice quiz from your PDFs.

1. Build `vector_store/` by running `pdf_rag_pipeline.ipynb` top → bottom.
2. Start the app:

```powershell
streamlit run app.py
```

3. Open the `Quiz` tab:
   - Select quiz language
   - Select difficulty + number of questions
   - Optionally limit to specific PDF files
   - Click `Generate Quiz`
