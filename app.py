# BLOCK ALL DOWNLOADS - SET OFFLINE MODE
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['SPACY_WARNING_IGNORE'] = 'W008'

import json
import fitz  # PyMuPDF
import spacy
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import sys
import logging
from dataclasses import dataclass
from functools import lru_cache
import warnings
import glob
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SectionInfo:
    """Data class for section information"""
    title: str
    content: str
    page: int
    font_size: float = 0.0
    is_bold: bool = False
    position: int = 0

class PersonaDrivenDocumentAnalyzer:
    def __init__(self, cache_dir="/app/cached_models"):
        """Initialize the document analyzer with pre-downloaded models"""
        self.cache_dir = cache_dir
        self._initialize_models()
        self._section_cache = {}
        self._embedding_cache = {}
        
    def _initialize_models(self):
        """Initialize models from pre-downloaded cache (no downloads)"""
        logger.info("🚀 Initializing models from cache (offline mode)...")
        
        # SpaCy model loading
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ SpaCy model loaded from cache")
        except OSError as e:
            logger.error(f"❌ SpaCy model not found in cache: {e}")
            logger.error("Make sure 'python -m spacy download en_core_web_sm' was run during Docker build")
            sys.exit(1)
        
        # SentenceTransformer loading from cache
        model_path = os.path.join(self.cache_dir, "sentence-transformers_all-MiniLM-L6-v2")
        
        logger.info(f"🔄 Loading SentenceTransformer from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model cache directory not found: {model_path}")
            logger.error("Make sure models were downloaded during Docker build")
            sys.exit(1)
        
        try:
            # Check for required files
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            missing_files = []
            
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    # Try alternative files
                    if file == 'pytorch_model.bin' and os.path.exists(os.path.join(model_path, 'model.safetensors')):
                        self._convert_safetensors_to_pytorch(model_path)
                    else:
                        missing_files.append(file)
            
            if missing_files:
                logger.warning(f"⚠️ Missing files: {missing_files}")
            
            # Load the model
            self.sentence_model = SentenceTransformer(model_path)
            logger.info("✅ SentenceTransformer loaded from cache")
            
            # Test the model
            test_emb = self.sentence_model.encode(["test sentence"])
            logger.info(f"✅ Model working - embedding shape: {test_emb.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load SentenceTransformer: {e}")
            logger.error("Check if model was properly cached during Docker build")
            sys.exit(1)
    
    def _convert_safetensors_to_pytorch(self, model_path: str):
        """Convert safetensors to pytorch format if needed"""
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path) and not os.path.exists(pytorch_path):
            logger.info("🔄 Converting safetensors to pytorch format...")
            try:
                from safetensors.torch import load_file
                import torch
                
                # Load from safetensors
                state_dict = load_file(safetensors_path)
                
                # Save as pytorch_model.bin
                torch.save(state_dict, pytorch_path)
                logger.info(f"✅ Converted to pytorch_model.bin")
                
            except ImportError:
                logger.error("❌ safetensors library not available for conversion")
                raise
            except Exception as e:
                logger.error(f"❌ Conversion failed: {e}")
                raise

    def find_pdf_files(self, input_folder: str) -> List[Dict[str, str]]:
        """Recursively find all PDF files in input folder and subdirectories"""
        pdf_files = []
        input_path = Path(input_folder)
        
        if not input_path.exists():
            logger.error(f"Input folder not found: {input_folder}")
            return []
        
        # Search for PDFs recursively
        pdf_patterns = ["**/*.pdf", "**/*.PDF"]
        
        for pattern in pdf_patterns:
            for pdf_path in input_path.glob(pattern):
                if pdf_path.is_file():
                    # Get relative path for cleaner output
                    relative_path = pdf_path.relative_to(input_path)
                    
                    pdf_files.append({
                        "filename": str(pdf_path),  # Full path for processing
                        "relative_path": str(relative_path),  # For display
                        "title": pdf_path.stem,  # Filename without extension
                        "size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2)
                    })
        
        logger.info(f"Found {len(pdf_files)} PDF files in {input_folder}")
        for pdf in pdf_files:
            logger.info(f"  - {pdf['relative_path']} ({pdf['size_mb']} MB)")
        
        return sorted(pdf_files, key=lambda x: x['relative_path'])

    def discover_processing_units(self, input_folder: str) -> List[Dict[str, Any]]:
        """Discover all processing units (folders with any JSON config or test cases)"""
        input_path = Path(input_folder)
        processing_units = []
        
        # Strategy 1: Any JSON file at root level
        json_files = list(input_path.glob("*.json"))
        if json_files:
            logger.info(f"📋 Found {len(json_files)} JSON config file(s) at root - processing all PDFs together")
            pdf_files = self.find_pdf_files(str(input_path))
            if pdf_files:
                # Use the first JSON file found or merge multiple configs
                config = self.load_any_input_json(str(input_path))
                if config:
                    processing_units.append({
                        "type": "single_config",
                        "name": "root_analysis",
                        "folder": str(input_path),
                        "config": config,
                        "pdf_files": pdf_files,
                        "json_files": [str(f) for f in json_files]
                    })
            return processing_units
        
        # Strategy 2: Multiple subfolders with any JSON files
        subfolders_with_json = []
        for item in input_path.iterdir():
            if item.is_dir():
                json_files = list(item.glob("*.json"))
                if json_files:
                    subfolders_with_json.append((item, json_files))
        
        if subfolders_with_json:
            logger.info(f"📁 Found {len(subfolders_with_json)} subfolders with JSON files")
            for subfolder, json_files in subfolders_with_json:
                config = self.load_any_input_json(str(subfolder))
                pdf_files = self.find_pdf_files(str(subfolder))
                
                if config:  # PDFs are optional
                    processing_units.append({
                        "type": "multiple_configs",
                        "name": subfolder.name,
                        "folder": str(subfolder),
                        "config": config,
                        "pdf_files": pdf_files,
                        "json_files": [str(f) for f in json_files]
                    })
            return processing_units
        
        # Strategy 3: Test case structure detection (any JSON files)
        test_cases = self.discover_test_cases(input_path)
        if test_cases:
            logger.info(f"🧪 Found {len(test_cases)} test cases")
            return test_cases
        
        # Strategy 4: Process all PDFs without config
        pdf_files = self.find_pdf_files(str(input_path))
        if pdf_files:
            logger.info("📄 No JSON configs found, processing all PDFs with default config")
            default_config = self.create_default_test_config("auto_discovered")
            processing_units.append({
                "type": "pdf_only",
                "name": "auto_pdf_processing",
                "folder": str(input_path),
                "config": default_config,
                "pdf_files": pdf_files,
                "json_files": []
            })
            return processing_units
        
        logger.warning("❓ No valid processing structure found")
        return []

    def load_any_input_json(self, input_folder: str) -> Dict[str, Any]:
        """Load and validate any JSON file from input folder"""
        input_path = Path(input_folder)
        
        # Find all JSON files
        json_files = list(input_path.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {input_folder}")
            return {}
        
        # Priority order for JSON files
        priority_names = [
            "challenge1b_input.json",
            "Challenge1b_input.json", 
            "config.json",
            "input.json",
            "challenge.json"
        ]
        
        # Try to find priority files first
        selected_json = None
        for priority_name in priority_names:
            priority_file = input_path / priority_name
            if priority_file in json_files:
                selected_json = priority_file
                break
        
        # If no priority file found, use the first JSON file
        if not selected_json:
            selected_json = json_files[0]
            logger.info(f"Using first available JSON file: {selected_json.name}")
        
        try:
            with open(selected_json, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            logger.info(f"✅ Loaded config from {selected_json.name}")
            
            # If multiple JSON files exist, try to merge them
            if len(json_files) > 1:
                logger.info(f"Found {len(json_files)} JSON files, attempting to merge configs")
                input_data = self.merge_json_configs(json_files, input_data, selected_json)
            
            # Validate and add defaults for required fields
            input_data = self.validate_and_enhance_config(input_data)
            
            return input_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {selected_json.name}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {selected_json.name}: {e}")
            return {}

    def merge_json_configs(self, json_files: List[Path], base_config: Dict, base_file: Path) -> Dict[str, Any]:
        """Merge multiple JSON configuration files"""
        merged_config = base_config.copy()
        
        for json_file in json_files:
            if json_file == base_file:
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    additional_config = json.load(f)
                
                logger.info(f"Merging config from {json_file.name}")
                
                # Smart merge: preserve base config but add missing fields
                for key, value in additional_config.items():
                    if key not in merged_config:
                        merged_config[key] = value
                    elif isinstance(value, dict) and isinstance(merged_config[key], dict):
                        # Merge nested dictionaries
                        for sub_key, sub_value in value.items():
                            if sub_key not in merged_config[key]:
                                merged_config[key][sub_key] = sub_value
                                
            except Exception as e:
                logger.warning(f"Could not merge {json_file.name}: {e}")
                continue
        
        return merged_config

    def validate_and_enhance_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance configuration with defaults"""
        enhanced_config = config.copy()
        
        # Ensure required fields exist
        if "persona" not in enhanced_config:
            enhanced_config["persona"] = {
                "role": "Document Analyst",
                "specialization": "General Analysis"
            }
        elif "role" not in enhanced_config["persona"]:
            enhanced_config["persona"]["role"] = "Document Analyst"
        
        if "job_to_be_done" not in enhanced_config:
            enhanced_config["job_to_be_done"] = {
                "task": "analyze and extract key information from documents",
                "context": "General document analysis"
            }
        elif "task" not in enhanced_config["job_to_be_done"]:
            enhanced_config["job_to_be_done"]["task"] = "analyze documents"
        
        if "challenge_info" not in enhanced_config:
            enhanced_config["challenge_info"] = {
                "challenge_id": "auto_config",
                "challenge_name": "Auto-configured Analysis",
                "description": "Automatically configured document analysis"
            }
        
        return enhanced_config

    def discover_test_cases(self, input_path: Path) -> List[Dict[str, Any]]:
        """Discover test case structure with any JSON files"""
        test_cases = []
        
        # Look for common test case patterns
        test_patterns = [
            "test*",
            "case*", 
            "scenario*",
            "example*"
        ]
        
        potential_test_dirs = set()
        for pattern in test_patterns:
            potential_test_dirs.update(input_path.glob(pattern))
        
        # Also check for numbered directories
        for item in input_path.iterdir():
            if item.is_dir():
                # Check if directory name suggests it's a test case
                name_lower = item.name.lower()
                if (any(keyword in name_lower for keyword in ['test', 'case', 'scenario', 'example']) or
                    item.name.isdigit() or 
                    bool(re.match(r'(test|case|scenario)[-_]?\d+', name_lower))):
                    potential_test_dirs.add(item)
        
        for test_dir in sorted(potential_test_dirs):
            # Check for any JSON file in test directory
            json_files = list(test_dir.glob("*.json"))
            if json_files:
                config = self.load_any_input_json(str(test_dir))
                pdf_files = self.find_pdf_files(str(test_dir))
                
                if config:  # PDFs are optional for test cases
                    test_cases.append({
                        "type": "test_case",
                        "name": test_dir.name,
                        "folder": str(test_dir),
                        "config": config,
                        "pdf_files": pdf_files,
                        "json_files": [str(f) for f in json_files]
                    })
            else:
                # Check for PDFs without config
                pdf_files = self.find_pdf_files(str(test_dir))
                if pdf_files:
                    # Create default config for test case
                    default_config = self.create_default_test_config(test_dir.name)
                    test_cases.append({
                        "type": "test_case_auto",
                        "name": test_dir.name,
                        "folder": str(test_dir),
                        "config": default_config,
                        "pdf_files": pdf_files,
                        "json_files": []
                    })
        
        return test_cases

    def create_default_test_config(self, test_name: str) -> Dict[str, Any]:
        """Create default configuration for test cases without JSON config"""
        return {
            "challenge_info": {
                "challenge_id": f"auto_test_{test_name}",
                "challenge_name": f"Auto-generated test: {test_name}",
                "description": f"Automatically generated test case for {test_name}"
            },
            "persona": {
                "role": "Document Analyst",
                "specialization": "General Analysis"
            },
            "job_to_be_done": {
                "task": f"analyze documents in {test_name}",
                "context": "General document analysis and extraction"
            }
        }

    def load_input_json(self, input_folder: str) -> Dict[str, Any]:
        """Load and validate any JSON file from input folder (legacy method)"""
        return self.load_any_input_json(input_folder)

    def create_processing_input(self, input_folder: str) -> Dict[str, Any]:
        """Create processing input by combining any JSON config and found PDFs"""
        # Load any JSON configuration
        json_data = self.load_any_input_json(input_folder)
        if not json_data:
            # Create default config if no JSON found
            json_data = self.create_default_test_config("default_processing")
        
        # Find all PDFs
        pdf_files = self.find_pdf_files(input_folder)
        if not pdf_files:
            logger.warning("No PDF files found in input folder")
        
        # Create combined input structure
        processing_input = {
            "challenge_info": json_data.get("challenge_info", {}),
            "persona": json_data.get("persona", {}),
            "job_to_be_done": json_data.get("job_to_be_done", {}),
            "documents": []
        }
        
        # Add PDF files to documents list
        for pdf_info in pdf_files:
            processing_input["documents"].append({
                "filename": pdf_info["filename"],
                "title": pdf_info["title"],
                "relative_path": pdf_info["relative_path"],
                "size_mb": pdf_info["size_mb"]
            })
        
        logger.info(f"Created processing input with {len(pdf_files)} documents")
        return processing_input

    def save_output(self, result: Dict[str, Any], output_folder: str, prefix: str = "") -> bool:
        """Save results to output folder with organized structure"""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Main output file
            filename_prefix = f"{prefix}_" if prefix else ""
            output_file = output_path / f"{filename_prefix}analysis_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Main results saved to {output_file}")
            
            # Create summary file
            summary_file = output_path / f"{filename_prefix}summary.json"
            summary = {
                "processing_summary": {
                    "timestamp": result.get("metadata", {}).get("processing_timestamp"),
                    "processing_time": result.get("metadata", {}).get("processing_time_seconds"),
                    "documents_processed": result.get("metadata", {}).get("documents_processed"),
                    "total_sections_found": result.get("metadata", {}).get("total_sections_found"),
                    "persona": result.get("metadata", {}).get("persona"),
                    "job_to_be_done": result.get("metadata", {}).get("job_to_be_done")
                },
                "top_sections": [
                    {
                        "rank": section["importance_rank"],
                        "document": section["document"],
                        "section": section["section_title"],
                        "page": section["page_number"]
                    }
                    for section in result.get("extracted_sections", [])[:5]
                ]
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Summary saved to {summary_file}")
            
            # Create detailed subsections file
            if result.get("subsection_analysis"):
                subsections_file = output_path / f"{filename_prefix}detailed_subsections.json"
                with open(subsections_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "subsection_analysis": result["subsection_analysis"]
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Detailed subsections saved to {subsections_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving output: {e}")
            return False

    def verify_offline_mode(self):
        """Verify that models are working in offline mode"""
        logger.info("🔍 Verifying offline mode...")
        
        # Test SpaCy
        try:
            test_doc = self.nlp("This is a test sentence for verification.")
            tokens = [token.text for token in test_doc]
            logger.info(f"✅ SpaCy working offline - tokenized: {len(tokens)} tokens")
        except Exception as e:
            logger.error(f"❌ SpaCy offline test failed: {e}")
            return False
        
        # Test SentenceTransformer
        try:
            test_sentences = ["This is a test sentence.", "Another test sentence."]
            embeddings = self.sentence_model.encode(test_sentences)
            logger.info(f"✅ SentenceTransformer working offline - embeddings shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"❌ SentenceTransformer offline test failed: {e}")
            return False
        
        logger.info("🎉 All models verified in offline mode!")
        return True

    @lru_cache(maxsize=100)
    def _get_embedding(self, text: str) -> np.ndarray:
        """Cache embeddings to avoid recomputation"""
        return self.sentence_model.encode([text])[0]

    def extract_pdf_content(self, filename: str) -> Dict[str, Any]:
        """Enhanced PDF content extraction with better structure detection"""
        if not os.path.exists(filename):
            logger.error(f"File not found: {filename}")
            return {"sections": [], "full_text": ""}
            
        try:
            doc = fitz.open(filename)
            sections = []
            full_text = ""
            page_texts = []
            
            logger.info(f"Processing {len(doc)} pages from {Path(filename).name}")
            
            # First pass: collect all text and potential headers
            potential_headers = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += text + "\n"
                page_texts.append(text)
                
                # Enhanced header detection
                headers = self._extract_headers_from_page(page, page_num + 1)
                potential_headers.extend(headers)
            
            doc.close()
            
            # Filter and validate headers
            valid_headers = self._validate_headers(potential_headers)
            
            if valid_headers:
                sections = self._extract_content_for_headers(full_text, valid_headers, page_texts)
            else:
                sections = self._create_intelligent_sections(full_text, page_texts)
            
            logger.info(f"Extracted {len(sections)} sections from {Path(filename).name}")
            
            return {
                "sections": sections,
                "full_text": full_text.strip()
            }
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            return {"sections": [], "full_text": ""}
    
    def _extract_headers_from_page(self, page, page_num: int) -> List[SectionInfo]:
        """Extract potential headers with enhanced formatting analysis"""
        headers = []
        blocks = page.get_text("dict")
        
        for block_idx, block in enumerate(blocks.get("blocks", [])):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text_content = span.get("text", "").strip()
                    font_size = span.get("size", 0)
                    font_flags = span.get("flags", 0)
                    
                    if self._is_potential_header(text_content, font_size, font_flags):
                        headers.append(SectionInfo(
                            title=text_content,
                            content="",
                            page=page_num,
                            font_size=font_size,
                            is_bold=bool(font_flags & 2**4),
                            position=block_idx
                        ))
        
        return headers
    
    def _is_potential_header(self, text: str, font_size: float, font_flags: int) -> bool:
        """Enhanced header detection with multiple criteria"""
        if not text or len(text.strip()) < 3:
            return False
            
        # Size-based detection
        if font_size > 12:
            return True
            
        # Bold text detection
        if font_flags & 2**4:  # Bold flag
            return True
            
        # Pattern-based detection
        return self._is_section_header(text)
    
    def _is_section_header(self, text: str) -> bool:
        """Enhanced section header pattern matching"""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Academic paper sections
        academic_patterns = [
            r'^(abstract|introduction|literature\s+review|methodology|methods|results|discussion|conclusion|references|bibliography)$',
            r'^(related\s+work|background|experimental\s+setup|evaluation|future\s+work)$',
        ]
        
        # Numbered sections
        numbered_patterns = [
            r'^\d+\.?\s+[A-Z][a-zA-Z\s]+',
            r'^(\d+\.)*\d+\s+[A-Z][a-zA-Z\s]+',
            r'^Chapter\s+\d+',
            r'^Section\s+\d+',
        ]
        
        # All caps (but reasonable length)
        if (text_clean.isupper() and 5 <= len(text_clean) <= 50 and 
            not any(char.isdigit() for char in text_clean)):
            return True
        
        # Check patterns
        all_patterns = academic_patterns + numbered_patterns
        return any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in all_patterns)
    
    def _validate_headers(self, headers: List[SectionInfo]) -> List[SectionInfo]:
        """Filter out false positive headers"""
        if not headers:
            return []
        
        # Remove duplicates and very short headers
        unique_headers = []
        seen_titles = set()
        
        for header in headers:
            title_clean = header.title.strip().lower()
            if (title_clean not in seen_titles and 
                len(title_clean) >= 3 and 
                len(title_clean) <= 100):
                unique_headers.append(header)
                seen_titles.add(title_clean)
        
        # Sort by page and position
        unique_headers.sort(key=lambda x: (x.page, x.position))
        
        return unique_headers
    
    def _extract_content_for_headers(self, full_text: str, headers: List[SectionInfo], page_texts: List[str]) -> List[Dict]:
        """Extract content between headers more accurately"""
        sections = []
        text_parts = full_text.split('\n')
        
        for i, header in enumerate(headers):
            content_lines = []
            header_found = False
            next_header_title = headers[i + 1].title.lower() if i + 1 < len(headers) else None
            
            for line in text_parts:
                line_clean = line.strip()
                
                # Start collecting after finding header
                if header.title.lower() in line_clean.lower():
                    header_found = True
                    continue
                
                # Stop at next header
                if (header_found and next_header_title and 
                    next_header_title in line_clean.lower()):
                    break
                
                # Collect content
                if header_found and line_clean:
                    content_lines.append(line_clean)
                    
                    # Limit content length
                    if len('\n'.join(content_lines)) > 2000:
                        break
            
            content = '\n'.join(content_lines).strip()
            if content:  # Only add sections with content
                sections.append({
                    "title": header.title,
                    "page": header.page,
                    "content": content
                })
        
        return sections
    
    def _create_intelligent_sections(self, full_text: str, page_texts: List[str]) -> List[Dict]:
        """Create sections using intelligent text segmentation"""
        # Try paragraph-based segmentation first
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 100]
        
        if len(paragraphs) >= 3:
            sections = []
            for i, para in enumerate(paragraphs[:8]):  # Limit to 8 sections
                sections.append({
                    "title": f"Section {i+1}",
                    "page": min(i + 1, len(page_texts)),
                    "content": para
                })
            return sections
        
        # Fallback to page-based segmentation
        sections = []
        for i, page_text in enumerate(page_texts[:5]):  # Limit to 5 pages
            if len(page_text.strip()) > 200:
                sections.append({
                    "title": f"Page {i+1}",
                    "page": i + 1,
                    "content": page_text.strip()[:1500]  # Limit content
                })
        
        return sections
    
    def calculate_relevance_score(self, section: Dict, persona: str, job_task: str, doc_title: str) -> float:
        """Enhanced relevance scoring with caching"""
        try:
            # Create text representations
            section_text = f"{section['title']} {section['content'][:500]}"  # Limit for performance
            persona_job_text = f"{persona} {job_task} {doc_title}"
            
            # Get cached embeddings
            section_embedding = self._get_embedding(section_text)
            persona_job_embedding = self._get_embedding(persona_job_text)
            
            # Semantic similarity (45%)
            semantic_score = cosine_similarity(
                section_embedding.reshape(1, -1), 
                persona_job_embedding.reshape(1, -1)
            )[0][0]
            
            # Keyword overlap (25%)
            keyword_score = self._calculate_keyword_overlap(section_text, persona_job_text)
            
            # Content type relevance (20%)
            content_type_score = self._classify_content_relevance(section, job_task)
            
            # Structural importance (10%)
            structural_score = self._calculate_structural_importance(section)
            
            total_score = (semantic_score * 0.45 + 
                          keyword_score * 0.25 + 
                          content_type_score * 0.20 + 
                          structural_score * 0.10)
            
            return float(np.clip(total_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating relevance score: {str(e)}")
            return 0.0
    
    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """Enhanced keyword overlap with TF-IDF weighting"""
        try:
            doc1 = self.nlp(text1.lower()[:1000])  # Limit for performance
            doc2 = self.nlp(text2.lower()[:1000])
            
            # Extract important tokens with POS filtering
            def extract_keywords(doc):
                return set([
                    token.lemma_ for token in doc 
                    if (not token.is_stop and not token.is_punct and 
                        token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                        len(token.text) > 2)
                ])
            
            tokens1 = extract_keywords(doc1)
            tokens2 = extract_keywords(doc2)
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error in keyword overlap calculation: {str(e)}")
            return 0.0
    
    def _classify_content_relevance(self, section: Dict, job_task: str) -> float:
        """Enhanced content classification with expanded indicators"""
        content = section["content"].lower()
        title = section["title"].lower()
        job_lower = job_task.lower()
        
        # Expanded content indicators
        content_indicators = {
            'methodology': ["method", "approach", "algorithm", "technique", "procedure", "framework", "model"],
            'results': ["result", "finding", "outcome", "performance", "evaluation", "experiment", "data"],
            'analysis': ["analysis", "discussion", "interpretation", "implication", "conclusion", "insight"],
            'background': ["background", "literature", "review", "related", "previous", "study"],
            'implementation': ["implementation", "system", "design", "architecture", "development"]
        }
        
        score = 0.0
        text_combined = title + " " + content
        
        # Match job requirements to content types
        for category, indicators in content_indicators.items():
            if any(keyword in job_lower for keyword in indicators):
                if any(indicator in text_combined for indicator in indicators):
                    score += 0.6
        
        # Bonus for exact matches
        job_words = set(job_lower.split())
        content_words = set(text_combined.split())
        exact_matches = len(job_words.intersection(content_words))
        score += min(exact_matches * 0.1, 0.4)
        
        return min(score, 1.0)
    
    def _calculate_structural_importance(self, section: Dict) -> float:
        """Enhanced structural importance calculation"""
        title = section["title"].lower()
        content_length = len(section["content"])
        
        # Critical sections (higher weight)
        critical_sections = ["abstract", "introduction", "conclusion", "summary"]
        important_sections = ["results", "methodology", "methods", "discussion", "findings"]
        
        score = 0.2  # Base score
        
        # Section type bonus
        if any(crit in title for crit in critical_sections):
            score += 0.6
        elif any(imp in title for imp in important_sections):
            score += 0.4
        
        # Content length consideration
        if 200 <= content_length <= 1500:  # Optimal length range
            score += 0.3
        elif content_length > 1500:
            score += 0.1
        
        # Position bonus (earlier sections often more important)
        if section.get("page", 1) <= 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def rank_sections_across_documents(self, processed_docs: List[Dict], persona: str, job_task: str) -> List[Dict]:
        """Enhanced section ranking with diversity consideration"""
        all_sections = []
        
        logger.info(f"Ranking sections across {len(processed_docs)} documents")
        
        for doc in processed_docs:
            for section in doc["sections"]:
                if len(section["content"].strip()) < 50:
                    continue
                    
                relevance_score = self.calculate_relevance_score(
                    section, persona, job_task, doc["title"]
                )
                
                all_sections.append({
                    "document": doc["relative_path"],  # Use relative path for cleaner output
                    "section_title": section["title"],
                    "page_number": section["page"],
                    "relevance_score": relevance_score,
                    "content": section["content"],
                    "doc_title": doc["title"]
                })
        
        # Sort by relevance score
        all_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Ensure diversity across documents
        all_sections = self._ensure_document_diversity(all_sections)
        
        # Assign importance ranks
        for i, section in enumerate(all_sections):
            section["importance_rank"] = i + 1
        
        logger.info(f"Ranked {len(all_sections)} sections")
        return all_sections
    
    def _ensure_document_diversity(self, sections: List[Dict], max_per_doc: int = 3) -> List[Dict]:
        """Ensure diverse representation across documents"""
        doc_counts = {}
        diverse_sections = []
        
        for section in sections:
            doc_name = section["document"]
            current_count = doc_counts.get(doc_name, 0)
            
            if current_count < max_per_doc:
                diverse_sections.append(section)
                doc_counts[doc_name] = current_count + 1
        
        # Fill remaining slots with best remaining sections
        remaining_slots = min(10 - len(diverse_sections), len(sections) - len(diverse_sections))
        for section in sections:
            if section not in diverse_sections and remaining_slots > 0:
                diverse_sections.append(section)
                remaining_slots -= 1
        
        return diverse_sections
    
    def analyze_subsections(self, top_sections: List[Dict], persona: str, job_task: str, max_sections: int = 5) -> List[Dict]:
        """Enhanced subsection analysis with better content refinement"""
        subsection_analysis = []
        
        for section in top_sections[:max_sections]:
            refined_content = self._advanced_content_refinement(
                section["content"], persona, job_task
            )
            
            if refined_content:
                subsection_analysis.append({
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "refined_text": refined_content,
                    "page_number": section["page_number"],
                    "relevance_score": section["relevance_score"]
                })
        
        return subsection_analysis[:12]  # Increased limit
    
    def _advanced_content_refinement(self, content: str, persona: str, job_task: str) -> str:
        """Advanced content refinement using NLP"""
        try:
            # Split into sentences
            doc = self.nlp(content[:1000])  # Limit for performance
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
            
            if not sentences:
                return content[:200] + '...' if len(content) > 200 else content
            
            # Score sentences based on relevance
            sentence_scores = []
            persona_job_embedding = self._get_embedding(f"{persona} {job_task}")
            
            for sentence in sentences:
                try:
                    sent_embedding = self._get_embedding(sentence)
                    similarity = cosine_similarity(
                        sent_embedding.reshape(1, -1),
                        persona_job_embedding.reshape(1, -1)
                    )[0][0]
                    sentence_scores.append((sentence, similarity))
                except:
                    sentence_scores.append((sentence, 0.0))
            
            # Sort by relevance and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent for sent, score in sentence_scores[:3]]
            
            refined_text = ' '.join(top_sentences)
            return refined_text if len(refined_text) > 50 else content[:300]
            
        except Exception as e:
            logger.warning(f"Error in content refinement: {str(e)}")
            return content[:200] + '...' if len(content) > 200 else content
    
    def process_challenge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced main processing with better error handling and logging"""
        start_time = datetime.now()
        
        try:
            # Parse input with validation
            challenge_info = input_data.get("challenge_info", {})
            documents = input_data.get("documents", [])
            persona = input_data.get("persona", {}).get("role", "")
            job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "")
            
            if not documents or not persona or not job_to_be_done:
                raise ValueError("Missing required input fields")
            
            logger.info(f"Processing {len(documents)} documents for persona: {persona}")
            logger.info(f"Job to be done: {job_to_be_done}")
            
            # Process documents with progress tracking
            processed_docs = []
            for i, doc_info in enumerate(documents):
                filename = doc_info.get("filename", "")
                title = doc_info.get("title", "")
                relative_path = doc_info.get("relative_path", filename)
                
                logger.info(f"Processing document {i+1}/{len(documents)}: {relative_path}")
                doc_content = self.extract_pdf_content(filename)
                
                processed_docs.append({
                    "filename": filename,
                    "relative_path": relative_path,
                    "title": title,
                    "sections": doc_content["sections"]
                })
            
            # Rank sections
            ranked_sections = self.rank_sections_across_documents(
                processed_docs, persona, job_to_be_done
            )
            
            # Analyze subsections
            subsection_analysis = self.analyze_subsections(
                ranked_sections, persona, job_to_be_done
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate enhanced output with metadata
            output = {
                "metadata": {
                    "input_documents": [doc["relative_path"] for doc in processed_docs],
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat(),
                    "processing_time_seconds": round(processing_time, 2),
                    "total_sections_found": len(ranked_sections),
                    "documents_processed": len(processed_docs)
                },
                "extracted_sections": [
                    {
                        "document": section["document"],
                        "section_title": section["section_title"],
                        "importance_rank": section["importance_rank"],
                        "page_number": section["page_number"],
                        "relevance_score": round(section["relevance_score"], 3)
                    }
                    for section in ranked_sections[:15]  # Increased from 10
                ],
                "subsection_analysis": subsection_analysis
            }
            
            logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
            return output
            
        except Exception as e:
            error_msg = f"Error processing challenge: {str(e)}"
            logger.error(error_msg)
            return {
                "metadata": {
                    "error": error_msg,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": [],
                "subsection_analysis": []
            }

    def process_folder(self, input_folder: str, output_folder: str) -> bool:
        """Main method to process input folder and save to output folder"""
        logger.info(f"🚀 Starting folder processing")
        logger.info(f"Input folder: {input_folder}")
        logger.info(f"Output folder: {output_folder}")
        
        # Create processing input from folder contents
        processing_input = self.create_processing_input(input_folder)
        if not processing_input:
            logger.error("Failed to create processing input")
            return False
        
        # Process the challenge
        result = self.process_challenge(processing_input)
        
        # Save results
        success = self.save_output(result, output_folder)
        
        if success:
            logger.info("✅ Folder processing completed successfully!")
        else:
            logger.error("❌ Folder processing failed!")
        
        return success

    def process_direct_files(self, pdf_files: List[str], output_folder: str, persona_role: str = None, job_task: str = None) -> bool:
        """Process PDF files directly without input folder structure"""
        logger.info(f"🚀 Starting direct file processing")
        logger.info(f"PDF files: {pdf_files}")
        logger.info(f"Output folder: {output_folder}")
        
        # Validate PDF files
        valid_pdfs = []
        for pdf_path in pdf_files:
            pdf_file = Path(pdf_path)
            if pdf_file.exists() and pdf_file.suffix.lower() == '.pdf':
                valid_pdfs.append({
                    "filename": str(pdf_file.absolute()),
                    "relative_path": pdf_file.name,
                    "title": pdf_file.stem,
                    "size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2)
                })
            else:
                logger.warning(f"⚠️ Invalid or missing PDF: {pdf_path}")
        
        if not valid_pdfs:
            logger.error("❌ No valid PDF files found")
            return False
        
        # Create default configuration
        config = {
            "challenge_info": {
                "challenge_id": "direct_processing",
                "challenge_name": "Direct File Processing",
                "description": f"Direct processing of {len(valid_pdfs)} PDF files"
            },
            "persona": {
                "role": persona_role or "Document Analyst",
                "specialization": "General Analysis"
            },
            "job_to_be_done": {
                "task": job_task or "analyze provided documents",
                "context": "Direct file processing without input folder structure"
            }
        }
        
        # Create processing input
        processing_input = {
            "challenge_info": config["challenge_info"],
            "persona": config["persona"],
            "job_to_be_done": config["job_to_be_done"],
            "documents": valid_pdfs
        }
        
        logger.info(f"Created processing input with {len(valid_pdfs)} documents")
        
        # Process the challenge
        result = self.process_challenge(processing_input)
        
        # Save results
        success = self.save_output(result, output_folder, "direct_processing")
        
        if success:
            logger.info("✅ Direct file processing completed successfully!")
        else:
            logger.error("❌ Direct file processing failed!")
        
        return success

    def process_current_directory(self, output_folder: str, persona_role: str = None, job_task: str = None) -> bool:
        """Process PDFs in current directory"""
        current_dir = Path.cwd()
        logger.info(f"🚀 Processing current directory: {current_dir}")
        
        # Find PDFs in current directory
        pdf_files = list(current_dir.glob("*.pdf"))
        pdf_files.extend(current_dir.glob("*.PDF"))
        
        if not pdf_files:
            logger.error("❌ No PDF files found in current directory")
            return False
        
        # Convert to string paths
        pdf_paths = [str(pdf) for pdf in pdf_files]
        
        return self.process_direct_files(pdf_paths, output_folder, persona_role, job_task)

    def auto_discover_and_process(self, target_path: str, output_folder: str, persona_role: str = None, job_task: str = None) -> bool:
        """Auto-discover processing strategy based on target path"""
        target = Path(target_path)
        
        if not target.exists():
            logger.error(f"❌ Target path does not exist: {target_path}")
            return False
        
        # If it's a single PDF file
        if target.is_file() and target.suffix.lower() == '.pdf':
            logger.info("📄 Single PDF file detected")
            return self.process_direct_files([str(target)], output_folder, persona_role, job_task)
        
        # If it's a directory
        if target.is_dir():
            # Check for existing folder processing strategies
            processing_units = self.discover_processing_units(str(target))
            
            if processing_units:
                logger.info(f"📁 Folder processing strategy detected: {processing_units[0]['type']}")
                return self.process_folder(str(target), output_folder)
            else:
                # Fallback to direct PDF processing in directory
                logger.info("📁 No JSON configs found, processing PDFs directly")
                return self.process_current_directory_at_path(str(target), output_folder, persona_role, job_task)
        
        logger.error(f"❌ Unknown target type: {target_path}")
        return False

    def process_current_directory_at_path(self, directory_path: str, output_folder: str, persona_role: str = None, job_task: str = None) -> bool:
        """Process PDFs in specified directory without JSON config"""
        target_dir = Path(directory_path)
        logger.info(f"🚀 Processing directory: {target_dir}")
        
        # Find PDFs recursively
        pdf_files = []
        for pattern in ["**/*.pdf", "**/*.PDF"]:
            pdf_files.extend(target_dir.glob(pattern))
        
        if not pdf_files:
            logger.error(f"❌ No PDF files found in {directory_path}")
            return False
        
        # Convert to string paths and create relative info
        valid_pdfs = []
        for pdf_file in pdf_files:
            relative_path = pdf_file.relative_to(target_dir)
            valid_pdfs.append({
                "filename": str(pdf_file.absolute()),
                "relative_path": str(relative_path),
                "title": pdf_file.stem,
                "size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2)
            })
        
        # Create default configuration
        config = {
            "challenge_info": {
                "challenge_id": "directory_processing",
                "challenge_name": f"Directory Processing: {target_dir.name}",
                "description": f"Processing {len(valid_pdfs)} PDF files from {directory_path}"
            },
            "persona": {
                "role": persona_role or "Document Analyst",
                "specialization": "General Analysis"
            },
            "job_to_be_done": {
                "task": job_task or f"analyze documents in {target_dir.name}",
                "context": "Directory processing without JSON configuration"
            }
        }
        
        # Create processing input
        processing_input = {
            "challenge_info": config["challenge_info"],
            "persona": config["persona"],
            "job_to_be_done": config["job_to_be_done"],
            "documents": valid_pdfs
        }
        
        logger.info(f"Created processing input with {len(valid_pdfs)} documents")
        
        # Process the challenge
        result = self.process_challenge(processing_input)
        
        # Save results
        success = self.save_output(result, output_folder, f"directory_{target_dir.name}")
        
        if success:
            logger.info("✅ Directory processing completed successfully!")
        else:
            logger.error("❌ Directory processing failed!")
        
        return success

def main():
    """Enhanced main function with flexible processing options"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Persona-Driven Document Intelligence System (Offline Mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Folder processing (auto-detects strategy)
  python app.py --input-folder ./input --output-folder ./output
  
  # Process current directory PDFs
  python app.py --current-dir --output-folder ./output
  
  # Process specific PDF files
  python app.py --files paper1.pdf paper2.pdf --output-folder ./output
  
  # Auto-discover strategy from path
  python app.py --auto ./path --output-folder ./output
  
  # Legacy JSON file processing
  python app.py -i Challenge1b_input.json -o output.json
  
  # Utility commands
  python app.py --verify
  python app.py --list-pdfs ./folder
        """
    )
    
    # Primary processing modes
    processing_group = parser.add_mutually_exclusive_group()
    
    # Folder-based processing
    processing_group.add_argument('--input-folder', '--if', 
                                 help='Input folder (auto-detects strategy)')
    
    # Direct file processing
    processing_group.add_argument('--files', nargs='+', 
                                 help='Process specific PDF files directly')
    
    # Current directory processing
    processing_group.add_argument('--current-dir', '--cwd', action='store_true',
                                 help='Process PDFs in current directory')
    
    # Auto-discovery processing
    processing_group.add_argument('--auto', 
                                 help='Auto-discover processing strategy for path')
    
    # Legacy file processing
    processing_group.add_argument('--input', '-i', 
                                 help='Input JSON file path (legacy)')
    
    # Output specification
    parser.add_argument('--output-folder', '--of', 
                       help='Output folder for results')
    parser.add_argument('--output', '-o', 
                       help='Output JSON file path (legacy)')
    
    # Configuration for direct processing
    parser.add_argument('--persona', 
                       help='Persona role for direct processing (default: Document Analyst)')
    parser.add_argument('--task', 
                       help='Job task for direct processing (default: analyze documents)')
    
    # Utility options
    parser.add_argument('--verify', action='store_true',
                       help='Verify offline mode and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--list-pdfs', 
                       help='List PDFs in specified folder and exit')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize analyzer
    logger.info("🚀 Initializing document analyzer in offline mode...")
    analyzer = PersonaDrivenDocumentAnalyzer()
    
    # Verify offline mode if requested
    if args.verify:
        if analyzer.verify_offline_mode():
            logger.info("✅ Offline mode verification successful!")
            return 0
        else:
            logger.error("❌ Offline mode verification failed!")
            return 1
    
    # List PDFs if requested
    if args.list_pdfs:
        if not os.path.exists(args.list_pdfs):
            logger.error(f"Folder not found: {args.list_pdfs}")
            return 1
        
        # Show processing strategy and PDFs
        processing_units = analyzer.discover_processing_units(args.list_pdfs)
        
        if processing_units:
            print(f"\n🔍 Processing Strategy: {processing_units[0]['type']}")
            print(f"📊 Found {len(processing_units)} processing unit(s)")
            
            for i, unit in enumerate(processing_units):
                print(f"\n--- Unit {i+1}: {unit['name']} ---")
                print(f"   📁 Folder: {unit['folder']}")
                print(f"   📄 PDFs: {len(unit.get('pdf_files', []))}")
                
                for pdf in unit.get('pdf_files', []):
                    print(f"      • {pdf['relative_path']} ({pdf['size_mb']} MB)")
                
                # Show config summary
                config = unit.get('config', {})
                persona_role = config.get('persona', {}).get('role', 'Not specified')
                job_task = config.get('job_to_be_done', {}).get('task', 'Not specified')
                print(f"   👤 Persona: {persona_role}")
                print(f"   🎯 Task: {job_task}")
        else:
            # Try direct PDF listing
            pdf_files = analyzer.find_pdf_files(args.list_pdfs)
            if pdf_files:
                print(f"\n📄 Found {len(pdf_files)} PDF files (no JSON structure):")
                for pdf in pdf_files:
                    print(f"   • {pdf['relative_path']} ({pdf['size_mb']} MB)")
                print("\n💡 These could be processed with direct file processing")
            else:
                print(f"❌ No PDFs or valid processing structure found in {args.list_pdfs}")
        
        return 0
    
    # Determine output folder requirement
    requires_output_folder = any([
        args.input_folder, args.files, args.current_dir, args.auto
    ])
    
    if requires_output_folder and not args.output_folder:
        logger.error("❌ --output-folder is required for this processing mode")
        return 1
    
    # Main processing modes
    
    # 1. Folder-based processing (auto-detects strategy)
    if args.input_folder:
        if not os.path.exists(args.input_folder):
            logger.error(f"Input folder not found: {args.input_folder}")
            return 1
        
        success = analyzer.process_folder(args.input_folder, args.output_folder)
        return 0 if success else 1
    
    # 2. Direct file processing
    elif args.files:
        success = analyzer.process_direct_files(
            args.files, args.output_folder, args.persona, args.task
        )
        return 0 if success else 1
    
    # 3. Current directory processing
    elif args.current_dir:
        success = analyzer.process_current_directory(
            args.output_folder, args.persona, args.task
        )
        return 0 if success else 1
    
    # 4. Auto-discovery processing
    elif args.auto:
        success = analyzer.auto_discover_and_process(
            args.auto, args.output_folder, args.persona, args.task
        )
        return 0 if success else 1
    
    # 5. Legacy file-based processing
    elif args.input and args.output:
        logger.info("Using legacy file-based processing")
        
        # Validate input file
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load input
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            logger.info(f"Loaded input from {args.input}")
        except Exception as e:
            logger.error(f"Error loading input file: {str(e)}")
            return 1
        
        # Process challenge
        start_time = datetime.now()
        result = analyzer.process_challenge(input_data)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Save output
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {args.output}")
            logger.info(f"Total processing time: {processing_time:.2f} seconds")
            return 0
        except Exception as e:
            logger.error(f"Error saving output file: {str(e)}")
            return 1
    
    else:
        # Show help and usage examples
        parser.print_help()
        print("\n" + "="*80)
        print("🚀 QUICK START EXAMPLES:")
        print("="*80)
        
        print("\n1️⃣ PROCESS CURRENT DIRECTORY PDFs:")
        print("   python app.py --current-dir --output-folder ./results")
        
        print("\n2️⃣ PROCESS SPECIFIC FILES:")
        print("   python app.py --files paper1.pdf paper2.pdf --output-folder ./results")
        
        print("\n3️⃣ PROCESS FOLDER (auto-detects strategy):")
        print("   python app.py --input-folder ./documents --output-folder ./results")
        
        print("\n4️⃣ AUTO-DISCOVER STRATEGY:")
        print("   python app.py --auto ./path/to/documents --output-folder ./results")
        
        print("\n5️⃣ ADD CUSTOM PERSONA/TASK:")
        print("   python app.py --files *.pdf --output-folder ./results \\")
        print("                 --persona 'Research Scientist' --task 'extract methodologies'")
        
        print("\n6️⃣ PREVIEW WHAT WILL BE PROCESSED:")
        print("   python app.py --list-pdfs ./documents")
        
        print("\n💡 The system automatically detects the best processing strategy!")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
