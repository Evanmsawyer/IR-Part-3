# Advanced Information Retrieval System

A information retrieval system that combines dense retrieval with LLM-powered query expansion and reranking capabilities.

## Features

- **Dense Retrieval**: Uses SentenceTransformers for efficient document embedding and retrieval
- **LLM Query Expansion**: Leverages Llama 3.2 to enhance queries with relevant terms and context
- **Neural Reranking**: Employs Llama 3.2 for context-aware document reranking
- **Optimized Performance**: 
  - Batch processing for queries
  - Efficient caching system for embeddings and query expansions
  - GPU acceleration support
  - Memory-efficient processing of large document collections
- **Comprehensive Evaluation**: Built-in evaluation metrics including MAP, MRR, NDCG@k, and Precision@k

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- torch
- transformers
- sentence-transformers
- ranx
- tqdm
- matplotlib
- sklearn
- bitsandbytes

## Usage

### Basic Usage

Run the system with default settings:

```bash
python llm-enchanced-ir.py --answers_file Answers.json --topics_files topics_1.json topics_2.json --qrels_file qrel_1.tsv
```

### Command Line Arguments

- `--answers_file`: Path to the document collection JSON file
- `--topics_files`: One or more topic files containing queries
- `--qrels_file`: Path to relevance judgments file (optional)
- `--batch_size`: Number of queries to process simultaneously (default: 32)
- `--embeddings_dir`: Directory for caching embeddings (default: 'embeddings')

### Configuration Options

The system supports multiple configurations combining different features:

1. Baseline (Dense Retrieval only)
2. Query Expansion with minimal prompting
3. Query Expansion with comprehensive prompting
4. Reranking with minimal prompting
5. Reranking with detailed prompting
6. Combined approach (QE + Reranking) with minimal prompting
7. Combined approach with comprehensive prompting

## File Structure

```
.
├── llm-enchanced-ir.py  # Main script
├── embeddings/          # Cached document embeddings
├── cache/              # Query expansion and reranking caches
├── results/            # Search results
└── plots/              # Evaluation plots and visualizations
```

## Output

The system generates:

1. Search results in TREC format:
```
[topic-id] Q0 [doc-id] [rank] [score] optimized
```

2. Evaluation metrics:
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- NDCG@5
- Precision@1
- Precision@5

3. Visualization plots:
- Per-topic precision plots
- Overall metrics comparison
- Configuration comparison plots

## Cache Management

The system implements efficient caching for:
- Document embeddings
- Query expansions
- Reranking results

Caches are automatically managed and can be cleared by deleting the respective directories.

