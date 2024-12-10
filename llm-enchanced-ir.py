import os
import json
import torch
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    BitsAndBytesConfig
)
import torch.nn.functional as F
import ranx
import matplotlib.pyplot as plt
import transformers
import re
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import glob
import gc
from sklearn.decomposition import PCA
import numpy as np
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher, Indexer
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Topic:
    id: str
    title: str
    body: str
    
    def get_query(self) -> str:
        return f"{self.title} {self.body}"

@dataclass
class Document:
    id: str
    text: str

class DataManager:
    @staticmethod
    def read_collection(filepath: str) -> Dict[str, Document]:
        """Read collection from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {item['Id']: Document(id=item['Id'], text=item['Text']) 
                   for item in data}

    @staticmethod
    def load_topics(filepath: str) -> Dict[str, Topic]:
        """Load topics from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {item['Id']: Topic(id=item['Id'], 
                                    title=item['Title'],
                                    body=item['Body']) 
                   for item in data}

    @staticmethod
    def read_qrels(filepath: str) -> Dict[str, Dict[str, int]]:
        """Read QREL file."""
        qrels = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                topic_id, _, doc_id, rel = line.strip().split('\t')
                if topic_id not in qrels:
                    qrels[topic_id] = {}
                qrels[topic_id][doc_id] = int(rel)
        return qrels
    
class LightweightEmbedder:
    def __init__(self, batch_size: int = 512): 
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        if len(queries) <= self.batch_size:
            return self.model.encode(queries, convert_to_tensor=True, device=self.device)
        
        return self.model.encode(queries, batch_size=self.batch_size, 
                               convert_to_tensor=True, device=self.device)

    @torch.inference_mode()
    def encode_documents(self, documents: List) -> Dict[str, torch.Tensor]:
        doc_texts = []
        doc_ids = []
        
        for doc in documents:
            doc_texts.append(doc.text if hasattr(doc, 'text') else doc['text'])
            doc_ids.append(doc.id if hasattr(doc, 'id') else doc['id'])
        
        embeddings = self.model.encode(doc_texts, batch_size=self.batch_size,
                                     convert_to_tensor=True, device=self.device,
                                     show_progress_bar=True)
        
        return {doc_id: emb for doc_id, emb in zip(doc_ids, embeddings)}

    def search(self, query_embedding: torch.Tensor, doc_embeddings: Dict[str, torch.Tensor],
               top_k: int = 100) -> List[Tuple[str, float]]:
        doc_ids = list(doc_embeddings.keys())
        doc_embeddings_tensor = torch.stack(list(doc_embeddings.values())).to(self.device)
        query_embedding = query_embedding.to(self.device)

        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings_tensor)
        
        top_scores, top_indices = similarities.topk(min(top_k, len(doc_ids)))
        
        return [(doc_ids[idx], score.item()) 
                for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy())]
    

class LlamaReranker:
    def __init__(self, batch_size: int = 64):
        model_id = "meta-llama/Llama-3.2-1B"
        token = "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"
        self.batch_size = batch_size

        self.prompt_templates = {
            "minimal": "Rate relevance (0-100): Q:{query} D:{document}",
            "detailed": """You are an expert relevance assessor.
            Given a query and document, analyze their relevance considering:
            - Topic match
            - Information coverage
            - Specificity
            - Authority
            - Recency

            Query: {query}
            Document: {document}

            Rate relevance on scale 0-100 where:
            0: Completely irrelevant
            25: Slightly relevant
            50: Moderately relevant
            75: Very relevant
            100: Perfect match

            Relevance score:"""
        }

        self.current_prompt = "minimal"
        self._setup_model(model_id, token)
        self._setup_cache()

    def _setup_model(self, model_id, token):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=token,
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=token
        )
        self.model.config.use_cache = True
        self.model.eval()

    def _setup_cache(self):
        self.cache_dir = "reranker_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}
        self._load_cache()

    def _cache_file_path(self, query_hash: str) -> str:
        """Get cache file path for a query."""
        return os.path.join(self.cache_dir, f"cache_{query_hash}.json")

    def _load_cache(self):
        """Load all cache files into memory."""
        cache_files = glob.glob(os.path.join(self.cache_dir, "cache_*.json"))
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.memory_cache.update(cache_data)
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")

    def _save_cache(self, query_hash: str, cache_data: dict):
        """Save cache to disk."""
        cache_file = self._cache_file_path(query_hash)
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def rerank(self, query: str, doc_list: List[Tuple[str, float]], 
               collection: Dict[str, Document]) -> List[Tuple[str, float]]:
        doc_list = doc_list[:15]
        truncated_query = query[:30]
        query_hash = hashlib.md5(truncated_query.encode()).hexdigest()
        
        reranked = []
        to_process = []
        doc_map = {}
        new_cache_entries = {}
        
        for idx, (doc_id, initial_score) in enumerate(doc_list):
            cache_key = f"{query_hash}_{doc_id}"
            if cache_key in self.memory_cache:
                reranked.append((doc_id, self.memory_cache[cache_key]))
            else:
                if initial_score > 0.3:  
                    doc_map[len(to_process)] = doc_id
                    doc_text = collection[doc_id].text[:50]
                    prompt = self.prompt_templates[self.current_prompt].format(
                        query=truncated_query,
                        document=doc_text
                    )
                    to_process.append(prompt)
                else:
                    reranked.append((doc_id, initial_score * 0.5))

        if not to_process:
            return sorted(reranked, key=lambda x: x[1], reverse=True)

        with torch.inference_mode(), torch.cuda.amp.autocast():
            inputs = self.tokenizer(
                to_process,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=2,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for idx, response in enumerate(responses):
                doc_id = doc_map[idx]
                try:
                    score = float(''.join(filter(str.isdigit, response[-3:]))) / 100
                except:
                    score = 0.5
                
                cache_key = f"{query_hash}_{doc_id}"
                new_cache_entries[cache_key] = score
                reranked.append((doc_id, score))

        if new_cache_entries:
            self.memory_cache.update(new_cache_entries)
            self._save_cache(query_hash, new_cache_entries)
        
        return sorted(reranked, key=lambda x: x[1], reverse=True)
    
    def _update_stats(self, score: float, processing_time: float = None):
        """Update reranking statistics."""
        self.stats['total_reranked'] += 1
        if processing_time:
            self.stats['processing_times'].append(processing_time)
        
        score_percentage = score * 100
        range_key = f"{(score_percentage // 20) * 20 + 1}-{(score_percentage // 20 + 1) * 20}"
        if range_key in self.stats['scores_distribution']:
            self.stats['scores_distribution'][range_key] += 1

    def get_stats_summary(self) -> Dict:
        """Get a summary of reranking statistics."""
        if not self.stats['processing_times']:
            return {
                'total_reranked': self.stats['total_reranked'],
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': 0,
                'avg_processing_time': 0,
                'score_distribution': self.stats['scores_distribution']
            }
        
        return {
            'total_reranked': self.stats['total_reranked'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': self.stats['cache_hits'] / self.stats['total_reranked'],
            'avg_processing_time': sum(self.stats['processing_times']) / len(self.stats['processing_times']),
            'score_distribution': self.stats['scores_distribution']
        }

    def set_prompt_style(self, style: str):
        """Set the prompt template style to use."""
        if style in self.prompt_templates:
            self.current_prompt = style
            logger.info(f"Set prompt style to: {style}")
        else:
            logger.warning(f"Unknown prompt style: {style}. Using current style: {self.current_prompt}")
    
class LlamaQueryExpander:
    MAX_CACHE_SIZE = 1000
    
    def __init__(self):
        model_id = "meta-llama/Llama-3.2-1B"
        token = "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"
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=token,
            model_max_length=512,
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        logger.info("Loading model...")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=token
        )
        self.model.config.use_cache = True  
        self.model.eval() 
        logger.info("Model loaded successfully")
        
        self.cache = {}
        self._generate = lru_cache(maxsize=self.MAX_CACHE_SIZE)(self._generate)
        self.system_prompt = "You are a query expansion assistant. Add relevant search terms to enhance the query, returning only the expanded query. Just provide expanded query"
        #self.prompt_template = "{system}\n\nQuery: {query}\nExpanded query:"
        self.prompt_templates = {
            "minimal": "Expand this search query with relevant terms: {query}",
            
            "comprehensive": """You are an expert search query expansion system.
            Given a search query, analyze its key concepts and intent.
            Then expand it with:
            - Synonyms and related terms
            - Important context
            - Relevant technical terminology
            - Alternative phrasings

            Original query: {query}

            Expanded query:"""
        }
            
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and links from text."""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = ' '.join(text.split())
        return text

    def _generate(self, prompt: str) -> str:
        """Cached generation function."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=False
        ).to(self.model.device)
        
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def _extract_main_query(self, text: str) -> str:
        """Extract the main query from a longer text."""
        if '?' in text:
            sentences = text.split('?')
            return sentences[0].strip() + '?'
        return text.split('.')[0].strip()

    def _clean_response(self, response: str, original_query: str) -> str:
        """Clean up model response to extract only the expanded query."""
        response = response.replace(self.system_prompt, "")
        response = response.replace("Query:", "")
        response = response.replace("Expanded query:", "")
        response = response.replace("[INST]", "").replace("[/INST]", "")
        
        response = response.replace(original_query, "", 1)  
        
        response = response.strip()
        response = re.sub(r'\s+', ' ', response)
        response = response.strip('[]')
        
        if not response or response.isspace() or len(response) < 10:
            return original_query
            
        return response

    def _manage_cache(self):
        if len(self.cache) > self.MAX_CACHE_SIZE:
            oldest_keys = list(self.cache.keys())[:100]
            for key in oldest_keys:
                del self.cache[key]

    def set_prompt_style(self, style: str):
        """Set the prompt template style to use."""
        if style in self.prompt_templates:
            self.current_prompt = style
            logger.info(f"Set prompt style to: {style}")
        else:
            logger.warning(f"Unknown prompt style: {style}. Using current style: {self.current_prompt}")

    def expand_query(self, query: str) -> str:
        """Expand a single query using current prompt template."""
        #query = self._clean_html(query)
        #query = self._extract_main_query(query)
        
        if query in self.cache:
            return self.cache[query]
        
        prompt_template = self.prompt_templates[self.current_prompt]
        full_prompt = prompt_template.format(query=query)
        
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50, 
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        expanded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        expanded = self._clean_response(expanded, query)
        
        self.cache[query] = expanded
        self._manage_cache()
        
        return expanded

    def batch_expand_queries(self, queries: List[str], batch_size: int = 4) -> List[str]:
        """Expand multiple queries in batches with progress tracking."""
        total_queries = len(queries)
        logger.info(f"Starting batch expansion of {total_queries} queries...")
        
        logger.info("Cleaning queries...")
        with tqdm(total=total_queries, desc="Cleaning queries") as pbar:
            cleaned_queries = []
            for q in queries:
                cleaned_queries.append(self._extract_main_query(self._clean_html(q)))
                pbar.update(1)
        
        expanded_queries = []
        start_time = datetime.now()
        
        total_batches = (total_queries + batch_size - 1) // batch_size
        with tqdm(total=total_batches, desc="Processing batches") as batch_pbar:
            for i in range(0, len(cleaned_queries), batch_size):
                batch = cleaned_queries[i:i + batch_size]
                current_batch = i // batch_size + 1
                
                if current_batch > 1:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    queries_processed = i
                    qps = queries_processed / elapsed
                    remaining_queries = total_queries - queries_processed
                    eta_seconds = remaining_queries / qps if qps > 0 else 0
                    
                    batch_pbar.set_postfix({
                        'Queries/sec': f'{qps:.2f}',
                        'ETA': f'{eta_seconds/60:.1f}min'
                    })
                
                cached_results = []
                queries_to_process = []
                original_indices = []
                
                for idx, query in enumerate(batch):
                    if query in self.cache:
                        cached_results.append((idx, self.cache[query]))
                    else:
                        queries_to_process.append(query)
                        original_indices.append(idx)
                
                if queries_to_process:
                    batch_prompts = [
                        self.prompt_templates[self.current_prompt].format(query=q)
                        for q in queries_to_process
                    ]
                    
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        max_length=256,
                        truncation=True
                    ).to(self.model.device)
                    
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            num_beams=1,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    expanded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    processed_results = [
                        self._clean_response(exp, q)
                        for exp, q in zip(expanded, queries_to_process)
                    ]
                    
                    for query, result in zip(queries_to_process, processed_results):
                        self.cache[query] = result
                    self._manage_cache()
                    
                    all_results = [""] * len(batch)
                    for idx, result in cached_results:
                        all_results[idx] = result
                    for orig_idx, result in zip(original_indices, processed_results):
                        all_results[orig_idx] = result
                    
                    expanded_queries.extend(all_results)
                
                if torch.cuda.is_available() and i % 50 == 0:
                    torch.cuda.empty_cache()
                
                batch_pbar.update(1)
        
        total_time = (datetime.now() - start_time).total_seconds()
        avg_qps = total_queries / total_time
        logger.info(f"Query expansion completed: {total_queries} queries in {total_time:.1f}s "
                f"({avg_qps:.2f} queries/sec)")
        
        return expanded_queries
    
class IRSystem:
    def __init__(self, collection: Dict[str, Document], 
             use_reranking: bool = False,
             use_query_expansion: bool = False,
             query_prompt_style: str = "minimal",
             rerank_prompt_style: str = "minimal",
             batch_size: int = 512,
             embeddings_dir: str = "embeddings",
             cache_dir: str = "cache",
             use_cache: bool = True):
        self.ranker = LightweightEmbedder(batch_size=batch_size)
        self.use_query_expansion = use_query_expansion
        self.query_expander = LlamaQueryExpander() if use_query_expansion else None
        self.reranker = LlamaReranker() if use_reranking else None
        self.collection = collection
        self.embeddings_dir = embeddings_dir
        self.use_cache = use_cache
        self.expansion_cache = {}

        if hasattr(self.query_expander, 'set_prompt_style'):
            self.query_expander.set_prompt_style(query_prompt_style)
        if self.reranker and hasattr(self.reranker, 'set_prompt_style'):
            self.reranker.set_prompt_style(rerank_prompt_style)
        
        if use_cache:
            os.makedirs(embeddings_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)
            collection_hash = self._compute_collection_hash()
            self.embeddings_file = os.path.join(embeddings_dir, 
                                            f"doc_embeddings_{collection_hash}.pt")
            self.doc_embeddings = self._load_or_create_embeddings()
        else:
            logger.info("Computing embeddings from scratch...")
            documents = list(self.collection.values())
            self.doc_embeddings = self.ranker.encode_documents(documents)
        
        self.doc_ids = list(self.doc_embeddings.keys())
        self.doc_embeddings_tensor = torch.stack(
            list(self.doc_embeddings.values())
        ).to(self.ranker.device)

        if use_cache:
            self.expansion_cache_file = os.path.join(cache_dir, "expansion_cache.json")
            if os.path.exists(self.expansion_cache_file):
                self.expansion_cache = self._load_expansion_cache()

    def _save_expansion_cache(self):
        """Save query expansion cache to disk."""
        if self.use_cache and hasattr(self, 'expansion_cache_file'):
            logger.info(f"Saving query expansion cache to {self.expansion_cache_file}")
            os.makedirs(os.path.dirname(self.expansion_cache_file), exist_ok=True)
            with open(self.expansion_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.expansion_cache, f, ensure_ascii=False, indent=2)

    def _compute_collection_hash(self) -> str:
        """Compute a fast hash of the collection for caching."""
        import hashlib
        content = ""
        for doc_id in sorted(self.collection.keys())[:100]:
            doc = self.collection[doc_id]
            content += f"{doc_id}:{doc.text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_expansion_cache(self) -> Dict[str, str]:
        """Load query expansion cache from disk."""
        if os.path.exists(self.expansion_cache_file):
            try:
                logger.info(f"Loading query expansion cache from {self.expansion_cache_file}")
                with open(self.expansion_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Expansion cache load failed: {e}")
        return {}

    def _save_expansion_cache(self):
        """Save query expansion cache to disk."""
        if self.use_cache:
            logger.info(f"Saving query expansion cache to {self.expansion_cache_file}")
            with open(self.expansion_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.expansion_cache, f, ensure_ascii=False, indent=2)

    def _load_or_create_embeddings(self) -> Dict[str, torch.Tensor]:
        """Load or create document embeddings with caching."""
        if os.path.exists(self.embeddings_file):
            try:
                logger.info(f"Loading cached embeddings from {self.embeddings_file}")
                embeddings = torch.load(self.embeddings_file)
                if set(embeddings.keys()) == set(self.collection.keys()):
                    return embeddings
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        logger.info("Computing document embeddings...")
        embeddings = self.ranker.encode_documents(list(self.collection.values()))
        
        if self.use_cache:
            logger.info(f"Saving embeddings to {self.embeddings_file}")
            torch.save(embeddings, self.embeddings_file)
        
        return embeddings

    def expand_queries(self, queries: List[str], batch_size: int = 16) -> List[str]:
        """Expand queries with caching."""
        expanded_queries = []
        queries_to_expand = []
        original_indices = []

        for i, query in enumerate(queries):
            if query in self.expansion_cache:
                expanded_queries.append(self.expansion_cache[query])
            else:
                queries_to_expand.append(query)
                original_indices.append(i)

        if queries_to_expand:
            logger.info(f"Expanding {len(queries_to_expand)} uncached queries...")
            new_expansions = self.query_expander.batch_expand_queries(
                queries_to_expand, batch_size=batch_size
            )

            for query, expansion in zip(queries_to_expand, new_expansions):
                self.expansion_cache[query] = expansion

            self._save_expansion_cache()

            result = [""] * len(queries)
            for i, exp in enumerate(expanded_queries):
                result[i] = exp
            for orig_idx, exp in zip(original_indices, new_expansions):
                result[orig_idx] = exp
            
            return result
        
        return expanded_queries

    @torch.inference_mode()
    def batch_process_queries(self, queries: List[str], 
                            batch_size: int = 32,
                            top_k: int = 100) -> Dict[int, List[Tuple[str, float]]]:
        logger.info(f"Processing {len(queries)} queries...")
        
        if self.use_query_expansion and self.query_expander:
            expanded = self.expand_queries(queries, batch_size=min(16, batch_size))
        else:
            expanded = queries
        
        logger.info("Encoding expanded queries...")
        query_embeddings = self.ranker.encode_queries(expanded)
        
        logger.info("Starting batch search...")
        results = {}
        total_queries = len(query_embeddings)
        start_time = datetime.now()
        
        for i in tqdm(range(0, total_queries, batch_size), desc="Processing queries"):
            batch_end = min(i + batch_size, total_queries)
            batch_queries = query_embeddings[i:batch_end].to(self.ranker.device)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                similarities = torch.matmul(batch_queries, self.doc_embeddings_tensor.t())
            
            for batch_idx, similarity_scores in enumerate(similarities):
                query_idx = i + batch_idx
                
                top_scores, top_indices = similarity_scores.topk(top_k)
                
                initial_results = [
                    (self.doc_ids[idx], score.item()) 
                    for idx, score in zip(top_indices.cpu(), top_scores.cpu())
                ]
                
                if self.reranker:
                    logger.debug(f"Reranking results for query {query_idx}")
                    reranked = self.reranker.rerank(
                        query=expanded[query_idx],
                        doc_list=initial_results,
                        collection=self.collection
                    )
                    results[query_idx] = reranked
                else:
                    results[query_idx] = initial_results
                
                if query_idx > 0 and query_idx % 100 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    qps = query_idx / elapsed
                    remaining = total_queries - query_idx
                    eta_mins = (remaining / qps) / 60 if qps > 0 else 0
                    logger.info(f"Processed {query_idx}/{total_queries} queries "
                            f"({qps:.2f} q/s, ETA: {eta_mins:.1f}min)")
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Search completed: {total_queries} queries in {total_time:.1f}s "
                f"({total_queries/total_time:.2f} queries/sec)")
        
        return results

def process_topics(system: IRSystem, topics: Dict[str, Topic],
                  collection: Dict[str, Document], output_file: str,
                  qrels: Optional[Dict[str, Dict[str, int]]] = None,
                  batch_size: int = 32):
    
    queries = []
    topic_ids = []
    for topic_id, topic in topics.items():
        queries.append(topic.get_query())
        topic_ids.append(topic_id)
    
    logger.info(f"Processing {len(queries)} topics...")
    batch_results = system.batch_process_queries(
        queries=queries,
        batch_size=batch_size,
        top_k=100
    )
    
    results = {topic_id: dict(batch_results[i]) 
              for i, topic_id in enumerate(topic_ids)}
    
    logger.info("Saving results...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for topic_id, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(
                sorted(doc_scores.items(), key=lambda x: x[1], reverse=True), 1):
                f.write(f"{topic_id}\tQ0\t{doc_id}\t{rank}\t{score}\toptimized\n")
    
    if qrels:
        eval_results = evaluate_results(qrels, results)
        logger.info("\nRetrieval Results:")
        for metric, score in eval_results.items():
            logger.info(f"{metric}: {score:.4f}")
    
    return results  

def evaluate_results(qrels: Dict[str, Dict[str, int]], 
                    results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Evaluate retrieval results."""
    metrics = ["map", "mrr", "ndcg@5", "precision@1", "precision@5"]
    return ranx.evaluate(
        ranx.Qrels.from_dict(qrels),
        ranx.Run.from_dict(results),
        metrics
    )

def plot_precision_at_k(qrels: Dict[str, Dict[str, int]], 
                       results: Dict[str, Dict[str, float]], 
                       k: int, title: str, output_file: str,
                       max_topics: int = 30):
    """Create precision@k plot for individual topics."""
    topic_precision = {}
    
    for topic_id in results:
        if topic_id not in qrels:
            continue
            
        relevant = set(doc_id for doc_id, rel in qrels[topic_id].items() if rel > 0)
        retrieved = list(results[topic_id].keys())[:k]
        relevant_retrieved = len([doc for doc in retrieved if doc in relevant])
        
        topic_precision[topic_id] = relevant_retrieved / k
    
    sorted_topics = sorted(
        topic_precision.items(),
        key=lambda x: x[1],
        reverse=True
    )[:max_topics]
    
    plt.figure(figsize=(15, 6))
    topics, precisions = zip(*sorted_topics) if sorted_topics else ([], [])
    
    plt.bar(range(len(topics)), precisions, color='skyblue')
    plt.title(title)
    plt.xlabel('Topic ID')
    plt.ylabel(f'Precision@{k}')
    plt.xticks(range(len(topics)), topics, rotation=90)
    
    for i, v in enumerate(precisions):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_metrics(results: Dict[str, float], title: str, output_file: str):
    """Create visualization of evaluation metrics."""
    plt.figure(figsize=(10, 6))
    metrics = list(results.keys())
    scores = list(results.values())
    
    plt.bar(metrics, scores, color='skyblue')
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f'{score:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='LLM Query Expansion IR System')
    parser.add_argument('--answers_file', default='Answers.json')
    parser.add_argument('--topics_files', nargs='+', default=['topics_1.json', 'topics_2.json'])
    parser.add_argument('--qrels_file', default='qrel_1.tsv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embeddings_dir', default='embeddings')
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    try:
        logger.info("Loading data...")
        collection = DataManager.read_collection(args.answers_file)
        qrels = DataManager.read_qrels(args.qrels_file) if args.qrels_file else None
        
        configurations = [
            {
                "name": "baseline",
                "use_reranking": False,
                "query_prompt_style": "minimal",
                "rerank_prompt_style": "minimal",
                "use_query_expansion": False
            },
            {
                "name": "qe_minimal",
                "use_reranking": False,
                "query_prompt_style": "minimal",
                "rerank_prompt_style": "minimal",
                "use_query_expansion": True
            },
            {
                "name": "qe_comprehensive",
                "use_reranking": False,
                "query_prompt_style": "comprehensive",
                "rerank_prompt_style": "minimal",
                "use_query_expansion": True
            },
            {
                "name": "rerank_minimal",
                "use_reranking": True,
                "query_prompt_style": "minimal",
                "rerank_prompt_style": "minimal",
                "use_query_expansion": False
            },
            {
                "name": "rerank_detailed",
                "use_reranking": True,
                "query_prompt_style": "minimal",
                "rerank_prompt_style": "detailed",
                "use_query_expansion": False
            },
        ]
        
        all_results = {}
        
        for config in configurations:
            logger.info(f"\nRunning configuration: {config['name']}")
            
            embeddings_dir = "embeddings"
            cache_dir = f"cache_{config['name']}"

            system = IRSystem(
                collection=collection,
                use_reranking=config['use_reranking'],
                use_query_expansion=config['use_query_expansion'], 
                query_prompt_style=config['query_prompt_style'],
                rerank_prompt_style=config['rerank_prompt_style'],
                batch_size=512,
                embeddings_dir=embeddings_dir,  
                cache_dir=cache_dir,
                use_cache=True
            )
            
            config_results = {}
            
            for topics_file in args.topics_files:
                logger.info(f"\nProcessing {topics_file}")
                topics = DataManager.load_topics(topics_file)
                topic_qrels = qrels if "topics_1" in topics_file else None
                file_num = topics_file.split('_')[1].split('.')[0]
                output_file = f"results/{config['name']}_results_{file_num}.tsv"
                
                results = process_topics(
                    system=system,
                    topics=topics,
                    collection=collection,
                    output_file=output_file,
                    qrels=topic_qrels,
                    batch_size=args.batch_size
                )
                
                if topic_qrels and results:
                    # Generate plots
                    for k in [5, 10]:
                        plot_precision_at_k(
                            qrels=topic_qrels,
                            results=results,
                            k=k,
                            title=f"{config['name']} - Precision@{k} (File {file_num})",
                            output_file=f"plots/{config['name']}_precision_at_{k}_file_{file_num}.png"
                        )
                    
                    # Evaluate and store results
                    eval_results = evaluate_results(topic_qrels, results)
                    config_results[f"file_{file_num}"] = eval_results
                    
                    plot_metrics(
                        results=eval_results,
                        title=f"{config['name']} - Metrics (File {file_num})",
                        output_file=f"plots/{config['name']}_metrics_file_{file_num}.png"
                    )
            
            all_results[config['name']] = config_results
        
        # Compare all configurations
        logger.info("\nFinal Results Comparison:")
        for config_name, results in all_results.items():
            logger.info(f"\n{config_name}:")
            for file_num, metrics in results.items():
                logger.info(f"\n{file_num}:")
                for metric, score in metrics.items():
                    logger.info(f"{metric}: {score:.4f}")
        
        # Create comparison plots
        plt.figure(figsize=(15, 8))
        configs = list(all_results.keys())
        metrics = ['map', 'ndcg@5', 'precision@5']
        
        for metric in metrics:
            scores = [all_results[config]['file_1'][metric] 
                     for config in configs if 'file_1' in all_results[config]]
            
            plt.figure(figsize=(12, 6))
            plt.bar(configs, scores, color='skyblue')
            plt.title(f'Comparison of {metric} Across Configurations')
            plt.xlabel('Configuration')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            
            for i, score in enumerate(scores):
                plt.text(i, score + 0.01, f'{score:.4f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(f'plots/comparison_{metric}.png')
            plt.close()
        
        logger.info("Processing completed successfully")
        logger.info("Results and plots saved in results/ and plots/ directories")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()