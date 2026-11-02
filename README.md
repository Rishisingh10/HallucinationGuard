# \# HallucinationGuard

# \### Building an Evidence-Grounded AI Claim Verification System

# 

# \*\*From Hybrid Information Retrieval to Adaptive Verification and Claim-Conditioned Evidence Selection\*\*

# 

# HallucinationGuard is an evidence-grounded AI claim verification system designed to evaluate factual claims against retrieved documents. Instead of trusting generated answers or relying on model confidence alone, it retrieves relevant evidence, evaluates the relationship between claims and evidence using Natural Language Inference (NLI), aggregates passage-level assessments, and makes evidence-based decisions.

# 

# The project explores an important problem in AI systems:

# 

# \*\*How can we determine whether a factual claim is actually supported by available evidence, contradicted by it, or simply impossible to verify with the information retrieved?\*\*

# 

# The system combines information retrieval, semantic search, ranking, transformer-based NLI, adaptive verification, evidence aggregation, and abstention mechanisms.

# 

# A major part of this project was not simply implementing these components, but experimentally investigating where the verification pipeline failed and designing targeted approaches to address those failures.

# 

# \---

# 

# \## Table of Contents

# 

# 1\. \[Problem Statement](#1-problem-statement)

# 2\. \[Project Objectives](#2-project-objectives)

# 3\. \[Evolution of the System](#3-evolution-of-the-system)

# 4\. \[System Architecture](#4-system-architecture)

# 5\. \[Technical Implementation](#5-technical-implementation)

# 6\. \[Claim-Conditioned Evidence Windowing (CCEW)](#6-claim-conditioned-evidence-windowing-ccew)

# 7\. \[Experimental Journey](#7-experimental-journey)

# 8\. \[SciFact Evaluation](#8-scifact-evaluation)

# 9\. \[Final Results](#9-final-results)

# 10\. \[Engineering Decisions and Lessons](#10-engineering-decisions-and-lessons)

# 11\. \[Technology Stack](#11-technology-stack)

# 12\. \[Project Structure](#12-project-structure)

# 13\. \[Installation and Execution](#13-installation-and-execution)

# 14\. \[Limitations](#14-limitations)

# 15\. \[Future Scope](#15-future-scope)

# 

# \---

# 

# \# 1. Problem Statement

# 

# Large Language Models can produce fluent, convincing answers containing factual inaccuracies. A generated statement may sound reasonable without having sufficient supporting evidence.

# 

# Traditional approaches that rely only on model confidence are not sufficient to establish factual correctness.

# 

# HallucinationGuard approaches this problem as an evidence verification task.

# 

# Given an input claim and a collection of documents, the system attempts to determine whether the available evidence:

# 

# \- Supports the claim.

# \- Contradicts the claim.

# \- Is insufficient to establish either conclusion.

# \- Contains conflicting assessments that require caution.

# 

# The central design principle is:

# 

# \*\*A system should not confidently accept a claim merely because it sounds plausible. It should examine the evidence and abstain when verification is not justified.\*\*

# 

# \# 2. Project Objectives

# 

# The project was developed around the following objectives:

# 

# 1\. Build a hybrid retrieval system combining keyword-based and semantic retrieval.

# 2\. Improve evidence ranking through Reciprocal Rank Fusion and claim-aware reranking.

# 3\. Evaluate claim-evidence relationships using Natural Language Inference.

# 4\. Develop an adaptive verification pipeline.

# 5\. Aggregate passage-level assessments into claim-level decisions.

# 6\. Handle contradictory and insufficient evidence conservatively.

# 7\. Investigate evidence quality and granularity as potential bottlenecks.

# 8\. Evaluate design decisions using a reproducible benchmark rather than relying on individual examples.

# 

# \# 3. Evolution of the System

# 

# HallucinationGuard evolved through several engineering stages.

# 

# \### Stage 1 — Establishing the verification foundation

# 

# The initial objective was to create a working claim verification pipeline.

# 

# The first implementation focused on:

# 

# \- Retrieving documents relevant to an input claim.

# \- Evaluating claim-evidence relationships using NLI.

# \- Aggregating evidence assessments.

# \- Producing a final verification decision.

# 

# This established the basic verification workflow.

# 

# \### Stage 2 — Hybrid Information Retrieval

# 

# A major challenge in evidence verification is retrieving documents that are relevant to the claim.

# 

# Keyword-based retrieval can miss semantically related passages, while semantic retrieval can overlook important lexical matches.

# 

# To address this, I implemented a hybrid retrieval strategy.

# 

# \*\*BM25 retrieval\*\*

# 

# BM25 provides lexical matching between the claim and indexed documents.

# 

# It is useful when important terms, entities, or phrases overlap between the query and evidence.

# 

# \*\*Semantic retrieval\*\*

# 

# Sentence Transformers generate embeddings for documents and queries.

# 

# Cosine similarity is used to rank documents based on semantic similarity.

# 

# \*\*Reciprocal Rank Fusion (RRF)\*\*

# 

# Rather than directly combining scores from different retrieval systems, Reciprocal Rank Fusion combines their rankings.

# 

# The fusion score is calculated as:

# 

# \\\[

# RRF(d)=\\sum\_{r \\in R}\\frac{1}{k+rank\_r(d)}

# \\]

# 

# Where:

# 

# \- \\(d\\) represents a document.

# \- \\(R\\) represents the retrieval systems.

# \- \\(rank\_r(d)\\) is the document's rank in a retrieval system.

# \- \\(k\\) is the RRF constant.

# 

# This provides a common ranking mechanism for BM25 and semantic retrieval.

# 

# \### Stage 3 — Claim-Aware Reranking

# 

# Hybrid retrieval improves candidate discovery, but a retrieved document may still be only partially relevant to the actual claim.

# 

# I introduced a claim-aware reranking component to refine the candidate evidence ranking.

# 

# The purpose was to prioritize documents that are more relevant to the specific claim being evaluated.

# 

# The reranker operates after initial retrieval and ranking, allowing the verification pipeline to focus on a refined set of evidence candidates.

# 

# \### Stage 4 — Natural Language Inference

# 

# Retrieval similarity does not establish factual correctness.

# 

# A document can be highly relevant to a claim while actually contradicting it.

# 

# This is why HallucinationGuard uses Natural Language Inference.

# 

# The NLI component evaluates the relationship between a claim and a retrieved evidence passage.

# 

# The relevant relationships are:

# 

# | NLI relationship | Interpretation |

# |---|---|

# | Entailment | Evidence supports the claim |

# | Contradiction | Evidence contradicts the claim |

# | Neutral | Evidence does not establish the claim |

# 

# The NLI model produces passage-level assessments that are subsequently used by the evidence aggregation component.

# 

# Importantly, NLI scores are treated as model outputs, not calibrated probabilities of factual truth.

# 

# \### Stage 5 — Adaptive Verification

# 

# A fixed retrieval and verification process may not always be sufficient.

# 

# I developed an adaptive verification engine that manages evidence retrieval and evaluation through multiple rounds.

# 

# The engine tracks verification-related information such as:

# 

# \- Retrieval rounds.

# \- Number of passages retrieved.

# \- Unique passages.

# \- Duplicate passages.

# \- Unique sources.

# \- NLI evaluations.

# \- Maximum verification budget.

# 

# The objective is to make verification decisions using the available evidence while managing the verification budget.

# 

# \### Stage 6 — Evidence Aggregation and Decision Making

# 

# Individual passages may support, contradict, or fail to establish a claim.

# 

# A claim-level decision therefore requires combining multiple passage-level assessments.

# 

# HallucinationGuard includes an evidence aggregation component that evaluates the collected assessments and produces a final verification status.

# 

# The system distinguishes:

# 

# \- SUPPORTED

# \- CONTRADICTED

# \- INSUFFICIENT

# \- CONFLICTING

# 

# A conservative decision strategy is important because conflicting evidence should not automatically be treated as proof of a claim.

# 

# \### Stage 7 — Evidence-Grounded Correction and Abstention

# 

# A verification system should not manufacture a correction when it cannot establish the correct information.

# 

# HallucinationGuard includes correction and abstention logic.

# 

# When evidence is inadequate, the system can abstain instead of producing an unsupported conclusion.

# 

# The correction mechanism is designed to rely on evidence rather than simply generating a plausible alternative.

# 

# \---

# 

# \# 4. System Architecture

# 

# The overall verification workflow is:

# 

# ```text

# &#x20;                  INPUT CLAIM

# &#x20;                       |

# &#x20;                       v

# &#x20;              HYBRID RETRIEVAL

# &#x20;                /          \\

# &#x20;               v            v

# &#x20;             BM25       SEMANTIC SEARCH

# &#x20;                \\          /

# &#x20;                 \\        /

# &#x20;                  v      v

# &#x20;            RECIPROCAL RANK FUSION

# &#x20;                       |

# &#x20;                       v

# &#x20;            CLAIM-AWARE RERANKING

# &#x20;                       |

# &#x20;                       v

# &#x20;            ADAPTIVE VERIFICATION

# &#x20;                       |

# &#x20;                       v

# &#x20;             EVIDENCE SELECTION

# &#x20;                       |

# &#x20;                       v

# &#x20;              NLI VERIFICATION

# &#x20;                       |

# &#x20;                       v

# &#x20;             EVIDENCE AGGREGATION

# &#x20;                       |

# &#x20;                       v

# &#x20;              DECISION ENGINE

# &#x20;                       |

# &#x20;            +----------+----------+

# &#x20;            |          |          |

# &#x20;            v          v          v

# &#x20;         SUPPORTED CONTRADICTED INSUFFICIENT

# &#x20;                       |

# &#x20;                       v

# &#x20;            CORRECTION / ABSTENTION

# ```

# 

# The final implementation additionally includes Claim-Conditioned Evidence Windowing before NLI evaluation.

# 

# \---

# 

# \# 5. Technical Implementation

# 

# \### 5.1 Evidence Retrieval

# 

# The retrieval module loads the evidence corpus and builds a BM25 index.

# 

# Semantic embeddings are generated using a Sentence Transformer model:

# 

# `sentence-transformers/paraphrase-MiniLM-L3-v2`

# 

# The two retrieval rankings are combined through RRF.

# 

# \### 5.2 Claim-Aware Reranking

# 

# The reranking component refines the fused document ranking based on the input claim.

# 

# This separates initial candidate retrieval from claim-specific evidence prioritization.

# 

# \### 5.3 NLI Verification

# 

# The NLI component evaluates each claim-evidence pair.

# 

# Its outputs are used by the aggregation and decision-making stages.

# 

# \### 5.4 Adaptive Verification Engine

# 

# The adaptive engine coordinates evidence retrieval, NLI evaluation, verification rounds, and evidence management.

# 

# It also maintains verification metrics that help analyze computational cost and retrieval behavior.

# 

# \### 5.5 Evidence Aggregation

# 

# Passage-level NLI assessments are combined to produce a claim-level status.

# 

# This component handles the distinction between supporting, contradicting, insufficient, and conflicting evidence.

# 

# \### 5.6 Document Processing

# 

# The application includes document loading support for:

# 

# \- CSV

# \- Excel

# \- PDF

# \- TXT

# \- DOCX

# 

# The document loader supports extraction and preview workflows. Document extraction and end-to-end evidence indexing should be distinguished: uploaded document previews are not automatically evidence sources for the retrieval pipeline in the current implementation.

# 

# \---

# 

# \# 6. Claim-Conditioned Evidence Windowing (CCEW)

# 

# \### The key experimental improvement

# 

# One important question emerged during evaluation:

# 

# \*\*Does the NLI model perform better when it evaluates an entire abstract, or when it evaluates a focused section containing the most claim-relevant sentences?\*\*

# 

# An abstract can contain several different facts. Passing the entire abstract to NLI may introduce unrelated information and dilute the evidence relevant to the specific claim.

# 

# To investigate this, I implemented Claim-Conditioned Evidence Windowing (CCEW).

# 

# \### 6.1 Core idea

# 

# CCEW selects a claim-relevant sentence window from each retrieved document before passing the evidence to NLI.

# 

# Instead of evaluating the entire abstract, the system:

# 

# 1\. Splits each retrieved document into sentences.

# 2\. Tokenizes the input claim and sentences.

# 3\. Applies BM25 sentence ranking within each document.

# 4\. Identifies the highest-ranked sentence.

# 5\. Selects a window containing that sentence and adjacent context.

# 6\. Passes the selected evidence to the existing NLI verifier.

# 

# \### 6.2 CCEW workflow

# 

# ```text

# &#x20;         RETRIEVED DOCUMENT

# &#x20;                 |

# &#x20;                 v

# &#x20;           SENTENCE SPLIT

# &#x20;                 |

# &#x20;                 v

# &#x20;         CLAIM TOKENIZATION

# &#x20;                 |

# &#x20;                 v

# &#x20;        SENTENCE-LEVEL BM25

# &#x20;                 |

# &#x20;                 v

# &#x20;         BEST SENTENCE

# &#x20;                 |

# &#x20;                 v

# &#x20;        ADJACENT SENTENCE

# &#x20;            SELECTION

# &#x20;                 |

# &#x20;                 v

# &#x20;         FOCUSED EVIDENCE

# &#x20;                 |

# &#x20;                 v

# &#x20;            NLI MODEL

# &#x20;                 |

# &#x20;                 v

# &#x20;         EVIDENCE ASSESSMENT

# ```

# 

# \### 6.3 Why CCEW?

# 

# The idea was to change the granularity of evidence supplied to the NLI model without replacing the existing retrieval architecture or introducing another large model.

# 

# The experiment focused on evidence selection rather than increasing model complexity.

# 

# \### 6.4 Implementation

# 

# The selector is implemented in:

# 

# `src/hallucination\_guard/sentence\_evidence\_selector.py`

# 

# The evaluation script is:

# 

# `scripts/evaluate\_scifact\_ccew.py`

# 

# CCEW is a heuristic sentence-selection method using BM25. Its value in this project is supported by the measured development-set results reported below, rather than by an assumption that sentence selection always improves verification.

# 

# \---

# 

# \# 7. Experimental Journey

# 

# An important part of developing HallucinationGuard was discovering that adding more components does not automatically improve a system.

# 

# I evaluated multiple approaches and retained the experimental results, including unsuccessful ones.

# 

# \### Experiment 1 — BM25 + NLI baseline

# 

# The baseline used keyword-based BM25 retrieval followed by NLI verification and evidence aggregation.

# 

# It provided a reference point for evaluating the more complex pipeline.

# 

# \*\*Macro-F1: 0.3096\*\*

# 

# \### Experiment 2 — Full HallucinationGuard

# 

# The full pipeline introduced:

# 

# \- Hybrid BM25 and semantic retrieval.

# \- Reciprocal Rank Fusion.

# \- Claim-aware reranking.

# \- Adaptive verification.

# \- NLI-based evidence assessment.

# \- Evidence aggregation and decision-making.

# 

# The intention was to improve evidence retrieval and verification quality.

# 

# However, the initial benchmark showed that the additional architecture did not improve the overall classification score.

# 

# \*\*Macro-F1: 0.3033\*\*

# 

# This was an important finding: architectural complexity alone did not guarantee better performance.

# 

# \### Experiment 3 — Evidence Sufficiency Gate (ESG)

# 

# I investigated whether a separate evidence sufficiency gate could improve the system's decision quality.

# 

# The gate used two heuristic checks:

# 

# 1\. Lexical claim coverage in retrieved evidence.

# 2\. NLI confidence based on the maximum entailment or contradiction score.

# 

# A passage qualified only when it passed both checks.

# 

# The intention was to prevent decisions based on evidence that did not sufficiently cover the claim.

# 

# However, the evaluation showed that the gate rejected many useful passages and did not improve classification performance.

# 

# \*\*Macro-F1: 0.2920\*\*

# 

# The experiment was rejected as a performance improvement.

# 

# This helped identify an important distinction: a simple evidence quality heuristic can be transparent and interpretable without necessarily being effective.

# 

# \### Experiment 4 — Claim-Conditioned Evidence Windowing

# 

# The next experiment investigated evidence granularity.

# 

# Rather than introducing another confidence threshold, CCEW changed what information was passed to the NLI model.

# 

# It selected claim-relevant sentence windows from the same retrieved documents.

# 

# The results showed a substantial improvement over the original full pipeline.

# 

# \*\*Macro-F1: 0.4089\*\*

# 

# CCEW was retained as the successful experimental improvement.

# 

# \---

# 

# \# 8. SciFact Evaluation

# 

# To move beyond manually selected examples, I prepared an evaluation using the SciFact dataset.

# 

# \### 8.1 Dataset preparation

# 

# The official SciFact dataset was used.

# 

# The development split contained 300 claims.

# 

# | Gold label | Number of claims |

# |---|---:|

# | SUPPORT | 124 |

# | CONTRADICT | 64 |

# | NOT ENOUGH INFO | 112 |

# | \*\*Total\*\* | \*\*300\*\* |

# 

# The dataset preparation process included:

# 

# \- Loading the original corpus.

# \- Loading development claims.

# \- Mapping evidence document identifiers.

# \- Extracting evidence sentence annotations.

# \- Converting the corpus into the internal document format.

# \- Checking missing evidence document references.

# 

# Preparation results:

# 

# | Check | Result |

# |---|---:|

# | Corpus documents | 5,183 |

# | Development claims | 300 |

# | Mixed-label claims | 0 |

# | Missing corpus documents | 0 |

# 

# \### 8.2 Evaluation methodology

# 

# Three principal configurations were investigated:

# 

# \*\*BM25 + NLI baseline\*\*

# 

# Keyword-based retrieval, NLI verification, and evidence aggregation.

# 

# \*\*Original HallucinationGuard\*\*

# 

# Hybrid retrieval, RRF, claim-aware reranking, adaptive verification, and NLI aggregation.

# 

# \*\*HallucinationGuard + CCEW\*\*

# 

# The original verification pipeline with claim-conditioned sentence-window selection before NLI assessment.

# 

# ESG was also evaluated as a separate experimental configuration.

# 

# \### 8.3 Evaluation metrics

# 

# The evaluation considered:

# 

# \- Accuracy

# \- Macro precision

# \- Macro recall

# \- Macro-F1

# \- Class-wise precision

# \- Class-wise recall

# \- Class-wise F1

# 

# Macro-F1 was particularly important because the evaluation involved three classes with different support counts.

# 

# The objective was to assess whether the system could recognize all three verification outcomes rather than simply favoring one class.

# 

# \---

# 

# \# 9. Final Results

# 

# \### 9.1 Overall performance

# 

# Results on the 300-claim SciFact development set:

# 

# | Configuration | Accuracy | Macro Precision | Macro Recall | Macro-F1 |

# |---|---:|---:|---:|---:|

# | BM25 + NLI | 0.3900 | 0.5196 | 0.3889 | 0.3096 |

# | Original HallucinationGuard | 0.3800 | 0.4708 | 0.3825 | 0.3033 |

# | HallucinationGuard + ESG | 0.3800 | 0.4641 | 0.3736 | 0.2920 |

# | HallucinationGuard + CCEW | \*\*0.4400\*\* | \*\*0.5318\*\* | \*\*0.4585\*\* | \*\*0.4089\*\* |

# 

# \### 9.2 Class-wise comparison

# 

# Original HallucinationGuard versus HallucinationGuard + CCEW:

# 

# | Class | Original Recall | CCEW Recall |

# |---|---:|---:|

# | SUPPORTED | 0.0403 | \*\*0.1613\*\* |

# | CONTRADICTED | 0.3125 | \*\*0.5000\*\* |

# | INSUFFICIENT | \*\*0.7946\*\* | 0.7143 |

# 

# CCEW improved recall for SUPPORTED and CONTRADICTED claims, while INSUFFICIENT recall decreased.

# 

# The result indicates a meaningful change in classification behavior rather than a uniform improvement across every class.

# 

# \### 9.3 Improvement

# 

# The original full pipeline achieved a macro-F1 of 0.3033.

# 

# With CCEW, macro-F1 increased to 0.4089.

# 

# \- Absolute macro-F1 improvement: 0.1056.

# \- Relative macro-F1 improvement: approximately 34.8%.

# \- Accuracy increased from 38% to 44%.

# 

# These results are encouraging but should be interpreted within the scope of the evaluated development set.

# 

# \### 9.4 Computational cost

# 

# The CCEW evaluation reported:

# 

# | Metric | Value |

# |---|---:|

# | Original pipeline NLI evaluations | 900 |

# | Additional CCEW NLI evaluations | 900 |

# | Combined evaluation runtime | 1,952.30 seconds |

# 

# The CCEW evaluation reruns NLI on the selected evidence after obtaining the original pipeline result. The reported runtime is for the combined evaluation process, not a standalone production latency measurement.

# 

# \---

# 

# \# 10. Engineering Decisions and Lessons

# 

# This project taught me several important engineering lessons.

# 

# \### 10.1 More complex architecture does not guarantee better performance

# 

# The hybrid retrieval and adaptive verification pipeline did not outperform the simpler BM25 baseline in the initial SciFact evaluation.

# 

# This reinforced the importance of measuring actual performance rather than assuming that adding components improves results.

# 

# \### 10.2 Retrieval relevance and evidence sufficiency are different problems

# 

# A document can be relevant to a claim without providing enough information to verify it.

# 

# Retrieval and verification therefore need to be treated as separate stages.

# 

# \### 10.3 More evidence is not always better evidence

# 

# The CCEW experiment explored whether focusing on claim-relevant sentences could improve NLI assessment.

# 

# The measured results supported further use of this approach within the evaluated setting.

# 

# \### 10.4 Abstention is an important system behavior

# 

# A verification system should distinguish between contradictory evidence and a lack of sufficient evidence.

# 

# It should not invent certainty when the available documents do not establish a conclusion.

# 

# \### 10.5 Failed experiments are engineering evidence

# 

# The Evidence Sufficiency Gate did not improve the benchmark score.

# 

# Rather than presenting it as a successful enhancement, I retained its results and used the findings to inform subsequent experimentation.

# 

# \### 10.6 Evaluation must accompany implementation

# 

# The project moved from manually selected examples to a structured dataset evaluation with documented metrics.

# 

# This made it possible to compare architectural decisions objectively.

# 

# \---

# 

# \# 11. Technology Stack

# 

# | Category | Technologies |

# |---|---|

# | Programming language | Python |

# | Deep learning | PyTorch |

# | NLP models | Hugging Face Transformers |

# | Sentence embeddings | Sentence Transformers |

# | Keyword retrieval | BM25 |

# | Similarity | Cosine Similarity |

# | Ranking | Reciprocal Rank Fusion |

# | Machine learning utilities | Scikit-learn |

# | Interface | Streamlit |

# | Data processing | Pandas, NumPy |

# | Testing | Pytest |

# | Dataset | SciFact |

# 

# \---

# 

# \# 12. Project Structure

# 

# ```text

# HallucinationGuard/

# │

# ├── app.py

# ├── bootstrap.py

# │

# ├── configs/

# │   └── config.yaml

# │

# ├── data/

# │   ├── sample/

# │   ├── evaluation/

# │   └── scifact/

# │       └── evaluation/

# │

# ├── docs/

# │   ├── ARCHITECTURE.md

# │   ├── DECISIONS.md

# │   ├── EXPERIMENTS.md

# │   ├── PROJECT.md

# │   └── HALLUCINATIONGUARD\_JOURNEY.md

# │

# ├── scripts/

# │   ├── prepare\_scifact.py

# │   ├── evaluate\_scifact.py

# │   ├── evaluate\_scifact\_ccew.py

# │   ├── diagnose\_esg\_rejections.py

# │   └── evaluate\_esg\_thresholds.py

# │

# ├── src/

# │   └── hallucination\_guard/

# │       ├── adaptive\_verification.py

# │       ├── claim\_attributes.py

# │       ├── claim\_aware\_reranker.py

# │       ├── claim\_extraction.py

# │       ├── correction.py

# │       ├── decision.py

# │       ├── document\_loader.py

# │       ├── evidence\_aggregation.py

# │       ├── evidence\_pipeline.py

# │       ├── evidence\_sufficiency.py

# │       ├── evaluation.py

# │       ├── hallucination\_guard.py

# │       ├── nli\_verification.py

# │       ├── retrieval.py

# │       ├── sentence\_evidence\_selector.py

# │       ├── uncertainty.py

# │       └── verification\_budget.py

# │

# ├── tests/

# │

# ├── requirements.txt

# ├── pyproject.toml

# ├── .gitignore

# └── README.md

# ```

# 

# \---

# 

# \# 13. Installation and Execution

# 

# \### Prerequisites

# 

# \- Python 3.10 or compatible version.

# \- Git.

# \- Sufficient memory and disk space for the selected NLP models.

# 

# \### Clone the repository

# 

# ```bash

# git clone https://github.com/Rishisingh10/HallucinationGuard.git

# cd HallucinationGuard

# ```

# 

# \### Create a virtual environment

# 

# ```bash

# python -m venv .venv

# ```

# 

# \### Activate on Windows

# 

# ```powershell

# .venv\\Scripts\\Activate.ps1

# ```

# 

# \### Install dependencies

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \### Run the application

# 

# ```bash

# streamlit run app.py

# ```

# 

# \### Run automated tests

# 

# ```bash

# pytest -q

# ```

# 

# The development test suite passed \*\*60 tests\*\* in the recorded final regression run.

# 

# \### Reproducing SciFact evaluation

# 

# The raw SciFact dataset is not included in the repository.

# 

# After obtaining the dataset and placing it in the expected directory structure, prepare the processed corpus and claims:

# 

# ```bash

# python scripts/prepare\_scifact.py

# ```

# 

# Run the original evaluation:

# 

# ```bash

# python scripts/evaluate\_scifact.py

# ```

# 

# Run the CCEW comparison:

# 

# ```bash

# python scripts/evaluate\_scifact\_ccew.py

# ```

# 

# The evaluation scripts generate result summaries and claim-level outputs in the SciFact evaluation directory.

# 

# \---

# 

# \# 14. Limitations

# 

# The project has several limitations that are important to acknowledge.

# 

# 1\. The reported benchmark results are from 300 SciFact development claims and do not establish generalization to unseen test data.

# 2\. The NLI model's scores are not calibrated probabilities of factual correctness.

# 3\. BM25 sentence selection can miss semantically relevant sentences with limited lexical overlap.

# 4\. Selecting a sentence window can remove context needed to interpret a claim.

# 5\. Source labels do not necessarily establish genuine source independence.

# 6\. The system's performance may vary across domains, writing styles, and evidence corpora.

# 7\. The CCEW evaluation performs additional NLI assessments, and production latency requires separate measurement.

# 8\. Uploaded document extraction and preview do not currently imply automatic indexing into the evidence retrieval pipeline.

# 

# \---

# 

# \# 15. Future Scope

# 

# Potential future directions include:

# 

# \- Evaluation on additional independent claim verification datasets.

# \- More robust sentence-level semantic evidence selection.

# \- Better handling of numerical and temporal claims.

# \- More sophisticated entity and relation verification.

# \- Evaluation of source reliability and source independence.

# \- Improved calibration of verification decisions.

# \- Integration of uploaded documents directly into the evidence retrieval pipeline.

# \- Performance optimization for larger evidence collections.

# 

# These are potential directions, not features claimed as already implemented.

# 

# \---

# 

# \## Final Note

# 

# HallucinationGuard was developed as an exploration of evidence-grounded claim verification.

# 

# The most valuable part of this project was not simply combining retrieval and NLP models. It was investigating why the system failed to improve, testing alternative approaches, measuring their impact, and retaining the results even when an experiment did not work.

# 

# The final CCEW evaluation increased macro-F1 from 0.3033 to 0.4089 on the 300-claim SciFact development set.

# 

# \*\*The central lesson: Building an AI system is not just about adding more models or components. It is about identifying the actual bottleneck, testing a hypothesis, and allowing experimental evidence to guide engineering decisions.\*\*

# 

# \---

# 

# \*\*Author:\*\* Rishi Singh  

# \*\*Program:\*\* B.Tech – Artificial Intelligence and Machine Learning  

# \*\*Institution:\*\* VIT Bhopal  

# \*\*Graduation:\*\* 2027

# 

# \*\*Repository:\*\* https://github.com/Rishisingh10/HallucinationGuard

