# Multilingual PDF Extractor – Technical Approach  

*This document explains **how** the solution works under the hood, the design
trade-offs that were made, and how each major component fits together.*

---

## 1. Design Goals  

- **Work completely offline** after image build (no runtime downloads).  
- **Handle heterogeneous inputs**: single files, flat folders, nested test cases,
  or complex challenge directories with JSON configs.  
- **Extract meaningful structure** (titles, sections, subsections) from PDFs
  that may have no bookmarks or TOC.  
- **Rank content by relevance** to a *persona* and a *job-to-be-done* rather than
  by naïve heuristics alone.  
- **Be language-agnostic**; support Western scripts and CJK numbering patterns
  out-of-the-box, but allow easy extension.  
- **Stay lightweight** (runs in a slim Python 3.11 image) and avoid GPU
  requirements.

---

## 2. High-Level Architecture  

CLI / Docker Entrypoint
        │
        ▼
PersonaDrivenDocumentAnalyzer  ←───────── models cached in /app/cached_models
        │
        ├── Input discovery & validation
        ├── PDF ingestion (PyMuPDF)
        ├── Structure extraction
        │     ├─ Title detection
        │     └─ Heading detection (H1-H3)
        ├── Relevance scoring & ranking
        ├── Sub-section refinement
        └── Output serialization (JSON)

---

## 3. Processing Pipeline  

| Step                            | What Happens                                                                                   | Key Methods                                                         |
| ------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 1. **Discover inputs**          | Detect PDFs & configs in any folder layout.                                                    | `discover_processing_units()`                                       |
| 2. **Extract text**             | Parse pages with PyMuPDF, accumulate raw text.                                                 | `extract_pdf_content()`                                             |
| 3. **Detect headings**          | Combine font-size stats, bold flags & regex patterns to mark H1-H3.                            | `_extract_headers_from_page()` + `_is_section_header()`             |
| 4. **Validate headers**         | Remove duplicates/false positives; sort by page & position.                                    | `_validate_headers()`                                               |
| 5. **Sectionize**               | Split full text between validated headers; fallback to intelligent segmentation if none found. | `_extract_content_for_headers()` / `_create_intelligent_sections()` |
| 6. **Compute relevance**        | Blend 4 signals → semantic similarity, keyword overlap, content-type match, structural weight. | `calculate_relevance_score()`                                       |
| 7. **Rank & enforce diversity** | Sort by score and cap max sections per document.                                               | `rank_sections_across_documents()`                                  |
| 8. **Refine top sections**      | Select best sentences via sentence-level similarity.                                           | `analyze_subsections()`                                             |
| 9. **Serialize output**         | Save `analysis_results.json`, a `summary.json`, and optional `detailed_subsections.json`.      | `save_output()`                                                     |

---

## 4. Algorithms & Heuristics  

### 4.1 Heading Detection  
1. **Font analysis**: gather font sizes on the first three pages, compute median
   & percentile thresholds to identify "large" text.  
2. **Style cues**: bold flag (`font_flags & 16`) strongly hints at headings.  
3. **Pattern matching**:  
   - Western: `^\d+(\.\d+)*\s+`, `Chapter \d+`, `Section \d+`, ALL-CAPS <=50 chars  
   - CJK: `^第[一二三四五六七八九十\d]+[章节]`, `제\d+장` …  
4. **Position**: text near the top 25 % of the page is favoured.

### 4.2 Relevance Score (0-1)  
0.45 × semantic_similarity
+ 0.25 × keyword_overlap
+ 0.20 × content_type_match
+ 0.10 × structural_importance

- *Semantic similarity* — cosine between SBERT embeddings of section text and
  `persona + task + doc_title`.  
- *Keyword overlap* — Jaccard on lemmatized, POS-filtered tokens (spaCy).  
- *Content-type match* — rule table that pairs job keywords with categories
  (methodology, results, …).  
- *Structural importance* — bonus for Abstract/Introduction/Conclusion and early
  pages; length normalization.

### 4.3 Diversity Enforcement  
A simple per-document quota (`max_per_doc = 3`) ensures the global top-10 list
is not monopolized by a single long report.

---

## 5. Machine-Learning Components  

| Purpose                          | Model                                    | Why Chosen                                                        | Size   |
| -------------------------------- | ---------------------------------------- | ----------------------------------------------------------------- | ------ |
| Embeddings & semantic similarity | `sentence-transformers/all-MiniLM-L6-v2` | Strong multilingual performance, 384-dim vectors, small footprint | ~90 MB |
| NLP tokenization & POS           | `spaCy en_core_web_sm`                   | Fast, CPU-only, licence-friendly                                  | ~13 MB |

*Both models are pre-downloaded during the Docker build, saved in
`/app/cached_models`, and loaded with `HF_HUB_OFFLINE=1`.*

---

## 6. Offline-First Strategy  

1. **Dockerfile** downloads the models once and caches them inside the image.  
2. Environment variables (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, …) stop any
   accidental calls to the internet.  
3. `app.py --verify` is executed during image build and on the container
   **HEALTHCHECK** to guarantee the models work without network.

---

## 7. Performance Optimizations  

- **Embedding cache** (`@lru_cache(maxsize=100)`) avoids recomputing SBERT
  vectors for repeated strings (section titles often repeat).  
- **Single-pass PDF read**: collect text + candidate headings simultaneously.  
- **Slim base image**: `python:3.11-slim` plus *only* gcc & g++ for PyMuPDF.  
- **No heavy dependencies**: no Torch GPU, no TensorFlow.  

---

## 8. Error Handling & Logging  

- `logging` at INFO (or DEBUG with `-v`) gives a concise progress bar per PDF.  
- Every critical path wrapped in `try/except`; failures degrade gracefully:
  • header detection → intelligent segmentation, • embedding failure → lower
  score, etc.  
- Output JSON always produced; errors are surfaced in `metadata.error`.

---

## 9. Extending the System  

| Desired Feature              | How to Add                                                              |
| ---------------------------- | ----------------------------------------------------------------------- |
| Additional language patterns | Append regexes in `_is_section_header()`.                               |
| OCR for scanned PDFs         | Drop-in `pytesseract` or `EasyOCR` call inside `extract_pdf_content()`. |
| GPU acceleration             | Use a CUDA-enabled SBERT model and swap base image.                     |
| Extra ranking signals        | Modify `calculate_relevance_score()` weights or add new function.       |

---

## 10. Limitations & Future Work  

- **Scanned PDFs** need external OCR.  
- **Non-Latin embeddings** rely on MiniLM's multilingual coverage; accuracy may
  drop on niche scripts.  
- **Very large documents** (>500 pages) processed serially; parallelization is a
  roadmap item.  
- Section granularity currently maxes at *H3*; deeper outlines are ignored.

---


