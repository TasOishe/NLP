# BERT vs SciBERT for Scientific and Biomedical NLP Task

## Project Overview

This project presents a comparative study of BERT and SciBERT for two fundamental Natural Language Processing (NLP) tasks in domain-specific text:
- Relation Classification (RC) on the SciERC dataset
- Named Entity Recognition (NER) on the BC5CDR dataset

These tasks play a crucial role in extracting structured information from scientific and biomedical literature, where identifying entities and understanding their semantic relationships is essential. The project investigates whether a domain-adapted transformer model (SciBERT) can outperform a general-purpose language model (BERT) on specialized datasets.


## Objectives
- Fine-tune BERT-base-uncased and SciBERT for RC and NER tasks
- Evaluate and compare performance using Accuracy, Precision, Recall, and F1-score
- Analyze the impact of domain-specific pretraining on scientific and biomedical NLP tasks


## Datasets

- SciERC: Scientific Relation Classification dataset
- BC5CDR: Biomedical Named Entity Recognition dataset (Chemicals and Diseases)

Due to computational and time constraints, subset sampling(100) was applied to both datasets.


## Implementation Details

Programming Language: Python 3.10
Libraries: Hugging Face Transformers, Datasets, SeqEval
Framework: PyTorch
Environment: Google Colab (GPU-enabled)
Optimizer: AdamW with linear learning rate scheduling


# Models Used
BERT: bert-base-uncased
SciBERT: allenai/scibert_scivocab_uncased

SciBERT is pretrained on a large corpus of scientific literature, enabling improved contextual understanding of domain-specific terminology.


## Training Setup

- Relation Classification (SciERC)
Architecture: AutoModelForSequenceClassification
Input: Sentences containing entity pairs
Evaluation Metrics: Accuracy, Precision, Recall, F1-score

- Named Entity Recognition (BC5CDR)
Architecture: AutoModelForTokenClassification
Labeling Scheme: BIO tagging
Proper alignment of labels with subword tokenization
Evaluation using seqeval

## Results
🔹 SciERC – Relation Classification
### BERT -
Accuracy: 0.58
F1-score: 0.43
Precision: 0.34
Recall: 0.58 

### Sci-BERT- 
Accuracy: : 0.67
F1-score: 0.59
Precision: 0.54
Recall: 0.67 

### Observation:
SciBERT achieved higher accuracy and F1-score than BERT, with notable improvements in precision and recall. This indicates that SciBERT’s scientific-domain vocabulary and embeddings provide a clear advantage for relation extraction tasks.

🔹 BC5CDR – Named Entity Recognition
### SciBERT-
Accuracy: : 0.97
F1-score: 0.87
Precision: 0.82
Recall: 0.95


### Observation:
SciBERT demonstrated strong and consistent performance on biomedical NER, particularly achieving high recall, which reflects its ability to correctly identify entity spans in domain-specific text.

## Key Findings
SciBERT consistently outperformed BERT on both Relation Classification and Named Entity Recognition tasks
Domain-adapted pretraining significantly improves performance on scientific and biomedical datasets
BERT showed comparatively lower precision, suggesting over-prediction in specialized domains

## Conclusion

This study highlights the importance of domain-specific pretrained language models for specialized NLP tasks. While BERT performs reasonably well, SciBERT’s training on scientific literature enables superior performance in both relation extraction and entity recognition. These results confirm that selecting a pretrained model aligned with the target domain is critical for achieving optimal performance.


## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Google Colab
- SeqEval
