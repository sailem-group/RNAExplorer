# Overview

This folder contains the Jupyter notebook needed to reproduce the analysis and figures detailed in the publication. 

---

## Environment Setup 

The following setup is used to conduct the analysis: 
- Python 3.12.3
- rna-fm 0.2.2 (see [RNA-FM GitHub] (https://github.com/ml4bio/RNA-FM#further-development--python-api) for further information regarding installation)
- pandas 2.0.3
- numpy 2.2.5
- scipy 1.14.1
- matplotlib 3.10.0
- seaborn 0.13.2
- scikit-learn 1.6.0
- networkx 3.4.2
- biopython 1.84
- pytorch 2.5.1
- tqdm 4.67.1

---

## Dataset

The raw datasets used in this analysis can be obtained from the following publications, and their corresponding sites if available. 
- siRNA: 
  Huesken D, Lange J, Mickanin C et al.. Design of a genome-wide siRNA library using an artificial neural network. Nat. Biotechnol. 2005; 23: 995–1001. https://doi.org/10.1038/nbt1118.
  Katoh T and Suzuki T. Specific residues at every third position of siRNA shape its efficient RNAi activity. Nucleic Acids Res. 2007; 35: e27. https://doi.org/10.1093/nar/gkl1120. 
  Sailem HZ, Rittscher J and Pelkmans L. KCML: a machine‐learning framework for inference of multi‐scale gene functions from genetic perturbation screens. Mol. Syst. Biol. 2020; 16: e9083. https://doi.org/10.15252/msb.20199083. 
- miRNA:  
  Griffiths-Jones S. The microRNA Registry. Nucleic Acids Res. 2004; 32: D109-111. https://doi.org/10.1093/nar/gkh023. 
  Kozomara A, Birgaoanu M and Griffiths-Jones S. miRBase: from microRNA sequences to function. Nucleic Acids Res. 2019; 47: D155–D162. https://doi.org/10.1093/nar/gky1141. 
  miRBase (Release 22.1): https://www.mirbase.org/ 
- piRNA: 
  Kuksa PP, Amlie-Wolf A, Katanić Ž et al.. DASHR 2.0: integrated database of human small non-coding RNA genes and mature products. Bioinformatics 2019; 35: 1033–1039. https://doi.org/10.1093/bioinformatics/bty709. 
  DASHR 2.0 hg38: https://dashr2.lisanwanglab.org/

Processed data, including interpretable features and RNA-FM embeddings, used in the analysis can be found at: 
https://doi.org/10.5281/zenodo.18124277. 
