import pandas as pd
import re
from llama_cpp import Llama

# File paths
input_file = "Huggingface\Resolution_Note_Dec.xlsx"
output_file = "extracted_issues_2.xlsx"

# Load LLaMA model
model_path = "llama-2-7b.Q2_K.gguf"
llm = Llama(model_path=model_path)

# Function to extract text after "Issue-" or "Issue:"
def extract_issue_text(cell_content, max_words=20):
    if not isinstance(cell_content, str) or not re.search(r"Issue[-:]", cell_content):
        return "No issue found"
    
    # LLaMA prompt
    prompt = (
        f"Extract only the text immediately after 'Issue-' or 'Issue:' from the content below. "
        f"Do not include 'Resolution', 'Status', or any other section:\n\n"
        f"{cell_content}\n\n"
        f"Output only the issue text, limited to {max_words} words."
    )
    response = llm(prompt, max_tokens=50)  # Ensure it doesn't exceed the limit
    extracted_text = response["choices"][0]["text"].strip()

    # Fallback regex extraction for accuracy
    match = re.search(r"Issue[-:]\s*(.+?)(Resolution|Status|Compliance|$)", extracted_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "No issue found"

# Load Excel file
df = pd.read_excel(input_file)

# Process the first 30 cells
extracted_issues = []
for idx, cell in enumerate(df.iloc[:, 0][:30]):  # Assuming the data is in the first column
    print(f"Processing cell {idx + 1}...")
    extracted_issue = extract_issue_text(cell)
    extracted_issues.append(extracted_issue)
    print(f"Extracted Issue from Cell {idx + 1}: {extracted_issue}")

# Create output DataFrame
output_df = pd.DataFrame({"Extracted Issues": extracted_issues})

# Save to Excel
output_df.to_excel(output_file, index=False)

print(f"Extracted issues saved to {output_file}")
