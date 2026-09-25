# 🧠 Multimodal DCA AI

## Production-Oriented Multimodal Document Understanding, Extraction, Validation & Anomaly Detection System

**End-to-End AI-powered document processing system combining OCR, document layout analysis, multimodal information extraction, table understanding, validation, confidence analysis, anomaly detection, structured JSON generation, and evaluation.**

---

# 🚀 Overview

The **Multimodal DCA AI** is an end-to-end document intelligence system designed to transform unstructured documents such as invoices and multi-page PDF documents into structured, validated, and analyzable information.

Unlike a traditional OCR pipeline that only extracts text, this project combines multiple document intelligence capabilities:

* 📄 PDF and image document processing
* 🔍 OCR-based text extraction
* 🧠 Document layout understanding
* 🤖 Multimodal information extraction
* 📊 Table extraction and validation
* 🧮 Arithmetic consistency checking
* 🎯 Confidence analysis
* 🚨 ML-based anomaly detection
* 🔎 Rule-based anomaly/fraud signals
* 📦 Structured JSON generation
* 🧪 Automated evaluation
* ⚡ Processing-performance measurement

The current implementation is evaluated using a **20-page invoice document workload**.

---

# 🎯 Problem Statement

Business documents such as invoices contain important information distributed across different visual structures and layouts.

A document AI system needs to do much more than simply read text.

Traditional document-processing systems can suffer from:

* OCR errors
* Different document layouts
* Incorrect field extraction
* Complex tables
* Incorrect totals
* Missing information
* Low extraction confidence
* Suspicious or anomalous documents
* Multi-page processing challenges
* Processing-time limitations

For example, extracting:

```text
Invoice Number
Date
Vendor
Customer
Subtotal
Tax
Total
Payment Status
````

is not sufficient if the extracted values are inconsistent.

The system should also be able to answer:

> **Is the extracted information structurally valid?**

> **Do the table amounts add up correctly?**

> **Does subtotal + tax match the total?**

> **Does the document contain anomalous signals?**

This project addresses these problems through a multimodal document intelligence pipeline.

---

# 💡 Solution

The proposed system processes documents through multiple AI and validation stages.

```text
Raw Document
     │
     ▼
Document Ingestion
     │
     ▼
PDF / Page Processing
     │
     ▼
Image Preprocessing
     │
     ▼
OCR
     │
     ▼
Layout Analysis
     │
     ▼
Multimodal Document Understanding
     │
     ├──────────────► Information Extraction
     │
     ├──────────────► Table Extraction
     │
     └──────────────► Document Structure
     │
     ▼
Data Validation
     │
     ├──────────────► Field Validation
     │
     ├──────────────► Arithmetic Validation
     │
     ├──────────────► Subtotal / Tax / Total
     │
     └──────────────► Confidence Analysis
     │
     ▼
Anomaly / Fraud Detection
     │
     ├──────────────► Isolation Forest
     │
     └──────────────► Rule-based Checks
     │
     ▼
Structured JSON
     │
     ▼
Evaluation
     │
     ├── Information Extraction
     ├── Table Quality
     ├── Anomaly Pipeline
     └── Performance
```

---

# 🏗️ System Architecture

```text
                         ┌────────────────────────┐
                         │     Document Input     │
                         │                        │
                         │ PDF / Image / Invoice  │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │ Document Ingestion     │
                         │                        │
                         │ PDF → Pages            │
                         │ Image Processing       │
                         └───────────┬────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │       Multimodal Analysis      │
                    │                                │
                    │ ┌────────────┐ ┌────────────┐ │
                    │ │    OCR     │ │   Layout   │ │
                    │ │            │ │  Analysis  │ │
                    │ │ Text       │ │ Structure  │ │
                    │ │ BoundingBox│ │ Positions  │ │
                    │ └─────┬──────┘ └──────┬─────┘ │
                    │       └────────┬───────┘       │
                    │                ▼               │
                    │    Document Understanding     │
                    └───────────────┬────────────────┘
                                    │
                  ┌─────────────────┼──────────────────┐
                  │                 │                  │
                  ▼                 ▼                  ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │ Information     │ │ Table           │ │ Chart           │
        │ Extraction      │ │ Extraction      │ │ Understanding   │
        │                 │ │                 │ │                 │
        │ Invoice No.     │ │ Rows            │ │ Future          │
        │ Date            │ │ Columns         │ │ Enhancement     │
        │ Vendor          │ │ Amounts         │ │                 │
        │ Customer        │ │ Totals          │ │                 │
        │ Tax             │ │ Validation      │ │                 │
        │ Total           │ │                 │ │                 │
        └────────┬────────┘ └────────┬────────┘ └─────────────────┘
                 │                   │
                 └─────────┬─────────┘
                           ▼
                 ┌──────────────────────┐
                 │ Data Validation      │
                 │                      │
                 │ Field Consistency   │
                 │ Arithmetic Checks   │
                 │ Subtotal + Tax      │
                 │ Confidence          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Anomaly / Fraud      │
                 │ Detection            │
                 │                      │
                 │ Isolation Forest     │
                 │ Rule-based Checks    │
                 │ Risk Signals         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Structured Output    │
                 │                      │
                 │ JSON                 │
                 │ Extracted Fields     │
                 │ Tables               │
                 │ Confidence           │
                 │ Validation Results   │
                 │ Anomaly Signals      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Evaluation Framework │
                 │                      │
                 │ Information          │
                 │ Extraction           │
                 │ Table Quality        │
                 │ Anomaly Signals      │
                 │ Performance          │
                 └──────────────────────┘
```

### Architecture Image

![Multimodal DCA AI Architecture](architecture.png)

---

# 🔥 Key Features

## 📄 1. Document Ingestion

The system accepts document-based inputs such as:

* PDF documents
* Invoice documents
* Multi-page documents
* Document images

The document is divided into processable pages before downstream analysis.

---

# 🔍 2. OCR Processing

The OCR layer extracts:

* Document text
* Words
* Text positions
* Bounding-box information

OCR output is then used by downstream layout and information-extraction components.

### OCR Evaluation Status

> **OCR: Not evaluated — verified reference text unavailable.**

The pipeline produces OCR output, but the current final evaluation does not report CER/WER because a verified human-checked OCR reference is not available.

No unsupported OCR accuracy claim is made.

---

# 🧠 3. Document Layout Analysis

Document layout is important because document meaning depends not only on text but also on spatial relationships.

The layout stage analyzes:

* Text positions
* Document blocks
* Spatial structure
* Layout relationships

The project uses **LayoutLMv3** as part of the document-understanding pipeline.

Conceptually:

```text
OCR Text
   +
Bounding Boxes
   +
Document Image
   ↓
Layout Understanding
   ↓
Structured Document Representation
```

---

# 🤖 4. Information Extraction

The system extracts important invoice information from the document.

Current evaluated fields include:

```text
Invoice Number
Invoice Date
Vendor Name
Customer Name
Subtotal
Tax
Total
Payment Status
```

Example:

```json
{
  "invoice_number": "INV-1001",
  "invoice_date": "03-09-2026",
  "vendor_name": "ABC Technologies Pvt Ltd",
  "customer_name": "Prime Constructions Ltd",
  "subtotal": 407800.0,
  "tax": 73404.0,
  "total": 481204.0,
  "payment_status": "paid"
}
```

---

# 📊 5. Table Extraction & Validation

Invoices frequently contain line-item tables.

The system extracts:

* Rows
* Columns
* Quantities
* Prices
* Amounts
* Subtotals
* Totals

The extracted table is also validated mathematically.

### Example Validation

```text
Quantity × Price
        ↓
Line Item Amount
        ↓
Sum of Line Items
        ↓
Subtotal
        ↓
Subtotal + Tax
        ↓
Total
```

This provides an additional reliability layer beyond OCR.

---

# 🧮 6. Data Validation

The validation layer checks whether extracted information is internally consistent.

### Field Validation

```text
Invoice Number
Date
Vendor
Customer
Amounts
Payment Status
```

### Arithmetic Validation

```text
Subtotal + Tax = Total
```

### Table Validation

```text
Quantity × Price = Amount
```

### Confidence Analysis

The system also records confidence information for extracted table data.

---

# 🚨 7. Anomaly / Fraud Detection

The project includes an anomaly-detection layer using:

```text
Isolation Forest
+
Rule-based validation signals
```

The ML component identifies pages that differ from the learned normal pattern.

The rule-based layer checks suspicious document signals.

Conceptually:

```text
Document Features
       │
       ▼
Isolation Forest
       │
       ├──────────────► Normal
       │
       └──────────────► Anomaly
```

Additional signals can be generated from:

* Arithmetic inconsistencies
* Suspicious values
* Document-level anomalies
* Validation failures

---

# 📦 8. Structured JSON Output

The pipeline converts document information into structured machine-readable output.

Example:

```json
{
  "invoice_number": "INV-1001",
  "invoice_date": "03-09-2026",
  "vendor_name": "ABC Technologies Pvt Ltd",
  "customer_name": "Prime Constructions Ltd",
  "subtotal": 407800.0,
  "tax": 73404.0,
  "total": 481204.0,
  "payment_status": "paid"
}
```

The complete pipeline output can additionally contain:

```text
OCR Results
Layout Information
Extraction Results
Table Results
Confidence
Validation Results
Anomaly Signals
Processing Information
```

---

# 🧪 9. Evaluation Framework

The project includes a dedicated evaluation framework for measuring different pipeline components.

Evaluation areas include:

```text
Information Extraction
Table Quality
Anomaly Detection
Performance
OCR
```

However, a metric is reported only when appropriate ground-truth/reference data is available.

This prevents unsupported evaluation claims.

---

# 📈 Information Extraction Evaluation

Current final evaluation:

```text
Precision:        1.0000
Recall:           1.0000
F1 Score:         1.0000

Correct fields:   160
Predicted fields: 160
Expected fields:  160

Documents evaluated: 20
```

This represents:

```text
160 evaluated fields
across
20 documents
```

---

# 📊 Table Quality Evaluation

Current final table-quality evaluation:

```text
Pages with tables:        20/20
Total rows:               60

Row arithmetic accuracy:  1.0000
Subtotal consistency:     1.0000
Table validation rate:    1.0000
Average table confidence: 0.9831
```

The table section is reported as **quality/validation analysis** because verified table ground-truth labels were not available for the full evaluation.

Therefore, a table F1 score is not claimed for this 20-page evaluation.

---

# 🚨 Anomaly Detection Evaluation

Current pipeline summary:

```text
ML anomalies:              2
ML normal:                18
ML anomaly rate:       0.1000

Fraud anomalies:           0
Fraud normal:             20
Fraud anomaly rate:    0.0000
```

The evaluation explicitly reports:

```text
SKIPPED: No verified anomaly ground-truth labels found.
```

Therefore, the values above represent **pipeline anomaly signals**, not supervised anomaly-detection accuracy.

---

# ⚡ Performance Evaluation

Current 20-page processing run:

```text
Total processing time:     825.2 seconds
Total pages:               20
Average page time:         41.26 seconds
Pages per second:          0.0242
```

### Performance Summary

| Metric            |    Result |
| ----------------- | --------: |
| Total Pages       |        20 |
| Processing Time   | 825.2 sec |
| Average Page Time | 41.26 sec |
| Pages / Second    |    0.0242 |

Performance can vary depending on:

* Hardware
* OCR processing
* Model execution
* Document complexity
* Image resolution

---

# 📸 Screenshots / Demo

## 1. Information Extraction

![Information Extraction](screenshots/information_extraction.png)

The final evaluation shows:

```text
Precision:        1.0000
Recall:           1.0000
F1 Score:         1.0000
Correct fields:   160
Expected fields:  160
Documents:        20
```

---

## 2. Table Quality

![Table Quality](screenshots/table_quality.png)

The final table-quality analysis shows:

```text
Pages with tables:        20/20
Total rows:               60
Row arithmetic accuracy:  1.0000
Subtotal consistency:     1.0000
Table validation rate:    1.0000
Average confidence:       0.9831
```

---

## 3. Anomaly Detection

![Anomaly Detection](screenshots/anomaly_detected.png)

The current pipeline detected:

```text
ML anomalies: 2
ML normal:    18
Anomaly rate: 0.1000
```

No verified anomaly ground-truth labels were available, so supervised accuracy is not claimed.

---

## 4. Performance

![Performance](screenshots/performance.png)

Current processing performance:

```text
Total processing time: 825.2 seconds
Total pages:           20
Average page time:     41.26 seconds
Pages per second:      0.0242
```

---

# 🛠️ Technology Stack

## Document Processing

```text
Python
PyMuPDF
OpenCV
NumPy
```

## Document AI

```text
OCR
LayoutLMv3
Computer Vision
NLP
Multimodal Document Understanding
```

## Machine Learning

```text
Scikit-learn
Isolation Forest
```

## Evaluation

```text
Precision
Recall
F1 Score
Table Validation
Confidence Analysis
Performance Metrics
```

## Development

```text
Python
Git
GitHub
Docker
```

---

# 📁 Project Structure

```text
multimodal-dca-ai/
│
├── data/
│   ├── ground_truth.json
│   └── ...
│
├── outputs/
│   ├── evaluation/
│   └── ...
│
├── src/
│   ├── evaluation/
│   │   ├── evaluation.py
│   │   ├── evaluator.py
│   │   └── ...
│   │
│   ├── pipeline.py
│   ├── anomaly.py
│   ├── document_anomaly_detector.py
│   └── ...
│
├── screenshots/
│   ├── information_extraction.png
│   ├── table_quality.png
│   ├── anomaly_detected.png
│   └── performance.png
│
├── architecture.png
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

> The exact repository structure may evolve as additional modules are added.

---

# 📋 Prerequisites

Before running the project:

```text
Python 3.11+
pip
Git
Virtual Environment
```

Depending on the selected model/OCR configuration, additional system/model dependencies may be required.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/vummidiganesh55/multimodal-dca-ai.git
```

```bash
cd multimodal-dca-ai
```

---

# 🐍 2. Create Virtual Environment

Windows:

```bash
python -m venv .venv311
```

Activate:

```bash
.venv311\Scripts\activate
```

---

# 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run the document-processing pipeline using the project's configured pipeline entry point.

Example:

```bash
python -m src.pipeline
```

Run the evaluation:

```bash
python -m src.evaluation.run_evaluation
```

The evaluation generates a report containing the available validated metrics and pipeline summaries.

---

# 🧪 Testing

The project evaluation currently processes:

```text
Pipeline pages:       20
Ground-truth records: 20
Matched invoices:     20
```

The evaluation framework separates:

```text
Verified Evaluation
        +
Descriptive Pipeline Results
        +
Unavailable Metrics
```

This prevents unsupported metrics from being presented as validated performance.

---

# 🔐 Security

The current project is primarily a local document-processing pipeline.

For production deployment, recommended security controls include:

* Input file validation
* File-size limits
* Supported-format validation
* Secure temporary file handling
* Authentication
* Authorization
* API rate limiting
* Secure storage
* Encryption
* Audit logging
* Secret management

Sensitive documents should not be committed to the Git repository.

---

# 🔮 Future Improvements

The architecture can be extended with additional multimodal document intelligence capabilities.

## ✍️ Handwriting Recognition

Support handwritten document fields and annotations.

```text
Handwritten Document
        ↓
Handwriting Recognition
        ↓
Structured Text
```

---

## 🌍 Multilingual Document Support

Support documents containing multiple languages.

```text
Multi-language Document
        ↓
Language Detection
        ↓
Multilingual OCR
        ↓
Information Extraction
```

---

## 📈 Chart Understanding

Future support for charts and visual analytics.

```text
Chart
 ↓
Chart Detection
 ↓
Data Extraction
 ↓
Trend Analysis
 ↓
Visual Insights
```

---

## 🚨 Advanced Fraud Detection

Future improvements can include:

* Document-level fraud detection
* Duplicate invoice detection
* Vendor anomaly detection
* Amount anomaly detection
* Cross-document comparison
* Historical pattern analysis

---

## ⚡ Real-Time Processing

The system can be extended toward real-time document processing:

```text
Document Upload
      ↓
Queue
      ↓
Processing Workers
      ↓
AI Pipeline
      ↓
Results
      ↓
Monitoring
```

---

## 🐳 Docker Optimization

Future deployment improvements include:

* Smaller Docker images
* Model caching
* Multi-stage builds
* CPU/GPU optimization
* Worker scaling

---

## 📈 Advanced Scaling

The pipeline can eventually support:

```text
                    Load Balancer
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Worker 1       Worker 2       Worker 3
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Document AI
                   Processing
```

This would allow the architecture to process larger document workloads.

---

# ⚠️ Limitations

The current implementation has several important limitations.

### OCR Evaluation

```text
OCR: Not evaluated
```

Verified human-checked OCR reference text is currently unavailable.

Therefore, CER/WER is not reported as a final project metric.

### Table Evaluation

Verified table ground-truth labels are unavailable for the complete 20-page evaluation.

Therefore, table precision/recall/F1 is not claimed.

Instead, the project reports:

```text
Row arithmetic accuracy
Subtotal consistency
Table validation rate
Average table confidence
```

### Anomaly Evaluation

Verified anomaly ground-truth labels are unavailable for the complete 20-page workload.

Therefore:

```text
Anomaly Precision
Anomaly Recall
Anomaly F1
```

are not claimed for the complete pipeline.

The current anomaly values are reported as pipeline signals.

### Performance

The performance benchmark is based on a 20-page invoice workload and should not be interpreted as a universal production benchmark.

---

# 📊 Final Evaluation Summary

| Component                 | Current Result |
| ------------------------- | -------------- |
| Documents / Pages         | 20             |
| Information Extraction F1 | 1.0000         |
| Correct Fields            | 160            |
| Expected Fields           | 160            |
| Table Pages               | 20/20          |
| Table Rows                | 60             |
| Row Arithmetic Accuracy   | 1.0000         |
| Subtotal Consistency      | 1.0000         |
| Table Validation Rate     | 1.0000         |
| Average Table Confidence  | 0.9831         |
| ML Anomalies              | 2              |
| ML Normal                 | 18             |
| ML Anomaly Rate           | 0.1000         |
| Total Processing Time     | 825.2 sec      |
| Average Page Time         | 41.26 sec      |
| Pages / Second            | 0.0242         |
| OCR Evaluation            | Not evaluated  |

---

# 🧠 What I Learned

This project provided hands-on experience across the document AI lifecycle:

```text
Document Processing
        ↓
Computer Vision
        ↓
OCR
        ↓
Layout Understanding
        ↓
Multimodal Information Extraction
        ↓
Table Understanding
        ↓
Validation
        ↓
Confidence Analysis
        ↓
Anomaly Detection
        ↓
Structured Output
        ↓
Evaluation
```

Key engineering concepts include:

* Document preprocessing
* OCR pipelines
* Bounding-box based document understanding
* LayoutLMv3
* Multimodal document analysis
* Structured information extraction
* Table validation
* Arithmetic consistency checking
* Confidence analysis
* Isolation Forest anomaly detection
* Ground-truth based evaluation
* Performance benchmarking
* Responsible reporting of unavailable metrics

---

# 💼 Business Use Cases

The architecture can be adapted beyond invoices.

## 🧾 Invoice Intelligence

```text
Invoice
   ↓
Extraction
   ↓
Validation
   ↓
Anomaly Detection
   ↓
Structured Financial Data
```

## 📄 Document Processing

```text
Business Documents
       ↓
OCR
       ↓
Layout Understanding
       ↓
Information Extraction
       ↓
Structured Data
```

## 🏦 Financial Document Analysis

```text
Financial Documents
       ↓
Extraction
       ↓
Validation
       ↓
Risk Signals
       ↓
Analytics
```

## 🏭 Enterprise Document Automation

```text
Enterprise Documents
       ↓
AI Processing
       ↓
Structured Information
       ↓
Validation
       ↓
Business Analytics
```

---

# 🏆 Project Highlights

```text
✅ End-to-End Multimodal Document AI
✅ PDF / Multi-page Processing
✅ OCR Pipeline
✅ Layout Analysis
✅ LayoutLMv3
✅ Information Extraction
✅ Table Extraction
✅ Arithmetic Validation
✅ Confidence Analysis
✅ Isolation Forest Anomaly Detection
✅ Rule-based Validation
✅ Structured JSON Output
✅ Evaluation Framework
✅ 20-Document Evaluation
✅ Performance Benchmarking
```

---

# 🔬 Evaluation Philosophy

A key principle of this project is:

> **Do not report a metric unless the evaluation data supports it.**

Therefore:

```text
Verified Ground Truth
        ↓
Valid Metric
        ↓
Report
```

Where verified ground truth is unavailable:

```text
No Verified Ground Truth
        ↓
Do Not Claim Accuracy
        ↓
Report Pipeline Signal / Limitation
```

This approach keeps the evaluation reproducible and scientifically defensible.

---

# 🚀 Future Vision

The long-term architecture can evolve toward a scalable enterprise document intelligence platform:

```text
                  Documents
                      │
                      ▼
             Document Intelligence
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        OCR        Layout       Vision
          │           │           │
          └───────────┼───────────┘
                      ▼
             Multimodal AI
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     Extraction     Tables      Charts
          │           │           │
          └───────────┼───────────┘
                      ▼
                 Validation
                      │
                      ▼
             Anomaly Detection
                      │
                      ▼
              Structured Data
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        API       Analytics     Storage
          │           │           │
          └───────────┼───────────┘
                      ▼
                 Monitoring
                      │
                      ▼
              Scalable Platform
```

---

# 👨‍💻 Author

**Naga Ganesh**

AI / Data Science | Machine Learning | Computer Vision | Document AI

---

# ⭐ Project Summary

**Multimodal DCA AI** demonstrates how multiple AI techniques can be combined to transform unstructured business documents into structured, validated, and intelligent information.

```text
Documents
    ↓
AI Understanding
    ↓
Extraction
    ↓
Validation
    ↓
Anomaly Detection
    ↓
Structured Data
    ↓
Evaluation
    ↓
Intelligent Insights
```

**From Documents → To Structured, Validated and Intelligent Insights.**

````
