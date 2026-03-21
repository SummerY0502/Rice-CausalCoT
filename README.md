# Rice-CausalCoT
##Welcome to the code and materials repository of Rice-CausalCoT.
<img src="Rice-CausalCoT.png" alt="TOTAL2" width="600">


## Note
- The code in this repository is primarily intended to illustrate implementation concepts and interface usage, serving as a reference for learning and research.
## Installation
This project depends on a few third-party Python packages (e.g., playwright, pandas, openai). Everything else in the imports (such as os, re, json, csv, pathlib, typing, etc.) is part of the Python standard library and does not require installation.
##Requirements
•	Python 3.9+ (recommended: 3.10 / 3.11)  
•	pip  
##Setup (recommended: virtual environment)  
macOS / Linux:  
1.	Create a virtual environment:  
python -m venv .venv  
2.	Activate it:  
source .venv/bin/activate  
Windows PowerShell:  
1.	Create a virtual environment:  
python -m venv .venv  
2.	Activate it:  
.venv\Scripts\Activate.ps1  
##Install Python dependencies  
Run:  
pip install -U pandas openai playwright  
Optional: If you want to pin dependencies in requirements.txt, add:  
•	pandas  
•	openai  
•	playwright  
##Install Playwright browsers  
Because this project uses playwright.async_api, you also need to install the browser engines:  
playwright install  
On Linux, if system dependencies are missing:  
playwright install-deps  
##Configure OpenAI API key (if applicable)  
If the project calls the OpenAI API, set the OPENAI_API_KEY environment variable.  
Windows PowerShell:  
setx OPENAI_API_KEY "YOUR_API_KEY"  

## Project Overview

This project consists of three major components: `RAG`, `BioCoT`, and `CausalCoT`. Together, these modules support gene-related knowledge retrieval, a priori reasoning, and causal reasoning analysis.

- **RAG** retrieves gene-associated information from functional databases.
- **BioCoT** generates and merges a priori reasoning evidence based on the retrieved knowledge.
- **CausalCoT** performs causal reasoning and downstream causal analysis.

---

## File Structure and Usage

### 1. `RAG/`
**Description**  
This module implements the Retrieval-Augmented Generation (RAG) component. It is responsible for retrieving gene-related information from functional databases and providing prior knowledge support for downstream reasoning modules.

**Main file**
- `Retrieve_information.py`: Retrieves associated gene information from functional databases.

**Run**
```bash
cd RAG
python Retrieve_information.py
```

---

### 2. `BioCoT/`
**Description**  
This module implements the a priori reasoning Chain-of-Thought (CoT) component. It generates multidimensional reasoning evidence based on the retrieved knowledge and integrates the results into a unified output file.

**Main files**
- `GPT_BioCoT.py`: Generates multidimensional reasoning evidence and saves the outputs.
- `Merge_BioCoT_Result.py`: Merges the reasoning results for all genes into a unified file, such as `Bio_Result.csv`.

**Run**
```bash
cd BioCoT
python GPT_BioCoT.py
python Merge_BioCoT_Result.py
```

---

### 3. `CausalCoT/`
**Description**  
This module implements the causal reasoning component. It is used to perform causal structure learning and causal explanation based on the processed gene information and reasoning outputs.

**Main files**
- `CausalStruCoT.py`: Performs causal structure reasoning.
- `CausalExCoT.py`: Performs causal explanation reasoning.

**Run**
```bash
cd CausalCoT
python CausalStruCoT.py
python CausalExCoT.py
```

---

## Recommended Execution Order

To run the full pipeline, execute the modules in the following order:

```bash
cd RAG
python Retrieve_information.py

cd ../BioCoT
python GPT_BioCoT.py
python Merge_BioCoT_Result.py

cd ../CausalCoT
python CausalStruCoT.py
python CausalExCoT.py
```





