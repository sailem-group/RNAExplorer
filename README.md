# RNAExplorer

RNAExplorer is an interactive web application for exploring, comparing, and analysing small RNA sequences using both deep learning–based embeddings and biologically interpretable sequence features.

The platform enables intuitive visualisation of user-provided RNA sequences alongside curated reference datasets of siRNA, miRNA, and piRNA.

---

## Features

- Interactive visualisation of small RNA sequences
- RNA-FM deep learning embeddings for global similarity analysis
- Interpretable feature embeddings based on biologically meaningful properties
- Exact and approximate query sequence search
- Integrated interactive tables with filtering, sorting, and export
- Direct links to original data sources
- Downloadable figures and curated datasets

---

## Visualisation Overview

### 1. RNA-FM Embedding
- Generated using a pretrained RNA-FM model
- Projects sequences into a shared embedding space
- Reference sequences are coloured by RNA class
- User sequences and query matches are highlighted

### 2. Interpretable Feature Embedding
- Based on explicit sequence-derived features:
  - Sequence length
  - GC percentage
  - GC and AU skew
  - Mononucleotide composition
  - 2-mer and 3-mer frequencies
- Dynamic feature selection with real-time re-projection

---

## Query Sequence Search

Users can paste a single RNA sequence to:
- Identify exact matches or fragment matches in the reference database
- Retrieve the most similar sequences based on RNA-FM embeddings

Matching sequences are highlighted in both embedding plots and detailed tables.

---

## Interactive Tables

Tables include:
- RNA type (siRNA, miRNA, piRNA)
- Sequence
- Species (linked to original source)
- Length and GC metrics
- Nucleotide composition

Users can:
- Filter by RNA type
- Toggle columns
- Sort values
- Download visible data as CSV

---

## Data Availability

Downloads available via the interface:
- Filtered feature tables (CSV)
- Embedding visualisations (PNG/SVG)

Only data currently visible in the interface are included in downloads.

---

## Citation

The RNAExplorer publication is underway.

Please cite the curated reference datasets available on Zenodo:

**RNAExplorer Reference Dataset**  
https://doi.org/10.5281/zenodo.18124277

---

## License

**Non-Commercial Use Only**

This project is licensed for **non-commercial use only**.
You may not use this code or data in any revenue-generating product,
service, SaaS platform, or commercial AI system.

---

## Installation (Local Development)

### Requirements
- Python ≥ 3.9
- pip / virtualenv

### Setup

```bash
git clone https://github.com/your-org/rnaexplorer.git
cd rnaexplorer

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python manage.py migrate
python manage.py collectstatic

python manage.py runserver
