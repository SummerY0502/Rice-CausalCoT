# Rice-CausalCoT
##Welcome to the code and materials repository of Rice-CausalCoT.
<p align="center">
  <img src="https://github.com/user-attachments/assets/aab3e126-7add-45df-b684-b3e611c064f8" alt="TOTAL2" width="400">
</p>


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
macOS / Linux:
export OPENAI_API_KEY="YOUR_API_KEY"
Windows PowerShell:
setx OPENAI_API_KEY "YOUR_API_KEY"
Note: On Windows, you may need to restart your terminal for the environment variable to take effect.
## File Structure
#RAG/: Retrieval-Augmented Generation component. First run Gene_Intersection.py, then run Retrieve_information.py to obtain retrieval information for associated genes in the dataset.  
Retrieve_information: Retrieves associated genes from functional databases.  
Gene_Intersection: Intersection of genes in functional databases and genes in the dataset.  
#BioCoT/: Bio-a priori reasoning CoT component. Run GPT_BioCoT.py to output multi-dimensional reasoning evidence from RAG embeddings to outputs/. Execute Merge_BioCoT_Result.py to consolidate bio-a priori information for all genes into a single file: Bio_Result.csv.  
#CausalCoT/: Causal-Outcome Learning CoT component.  
CausalCOT_AS.py: Executes the causal structure learning component under biological prior constraints.  
Expression_Screening.py: Using gene IDs, filter out the corresponding gene columns from the full gene expression dataset.  
Expression_Integration.py: Merge query genes and associated genes.  


