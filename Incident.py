# import pandas as pd
# from llama_cpp import Llama
# import time
# import re

# def setup_llama(model_path="llama-2-7b.Q2_K.gguf"):
#     """
#     Initialize Llama model using llama-cpp
#     """
#     model = Llama(
#         model_path=model_path,
#         n_ctx=2048,
#         n_gpu_layers=0  # Use CPU
#     )
#     return model

# def extract_with_llama(text, model):
#     """
#     Use Llama to extract text after 'issue' until the next period
#     """
#     if pd.isna(text) or not isinstance(text, str):
#         return None

#     # First check if text contains "issue"
#     if "issue" not in text.lower():
#         return None

#     prompt = f"""
#     Find the line that starts with or contains the word 'issue' and extract ONLY the text that comes after the word 'issue' until the next period. Ignore resolution, status, mode of communication, and compliance status.

#     Text: {text}

#     Extracted text:"""

#     # Generate response using llama-cpp
#     response = model.create_completion(
#         prompt,
#         max_tokens=100,
#         temperature=0.1,
#         stop=["\n"],
#         echo=False
#     )

#     # Extract the generated text
#     if response and 'choices' in response:
#         generated_text = response['choices'][0]['text'].strip()

#         # Clean up to ensure we only get text after "issue" and truncate at the period
#         if "issue" in generated_text.lower():
#             match = re.search(r"issue\s*:\s*(.*?)(\.|$)", generated_text, flags=re.IGNORECASE)
#             if match:
#                 return match.group(1).strip()

#     return None

# def process_excel_with_llama_limited(input_path, output_path, model_path="llama-2-7b.Q2_K.gguf", cell_limit=30):
#     """
#     Process Excel file using Llama for text extraction, limited to first N cells
#     """
#     print("Loading Llama model...")
#     model = setup_llama(model_path)
    
#     print("Reading Excel file...")
#     df = pd.read_excel(input_path)
    
#     extracted_data = []
#     processed_cells = 0
#     start_time = time.time()
    
#     print(f"Processing first {cell_limit} cells...")
#     try:
#         for column in df.columns:
#             for text in df[column]:
#                 if processed_cells >= cell_limit:
#                     break
                    
#                 if isinstance(text, str) and "issue" in text.lower():
#                     extracted = extract_with_llama(text, model)
#                     if extracted:
#                         extracted_data.append({
#                             'original_text': text,
#                             'extracted_text': extracted,
#                             'column': column,
#                             'cell_number': processed_cells + 1
#                         })
#                     else:
#                         print(f"No valid extraction for text: {text}")
                
#                 processed_cells += 1
#                 elapsed_time = time.time() - start_time
#                 speed = processed_cells / elapsed_time
#                 print(f"Processed cell {processed_cells}/{cell_limit} "
#                       f"({speed:.2f} cells/second)")
                
#             if processed_cells >= cell_limit:
#                 break
    
#     except Exception as e:
#         print(f"Error during processing: {str(e)}")
#         if extracted_data:
#             print("Saving partial results...")
    
#     finally:
#         # Save results and validate non-empty 'extracted_text'
#         result_df = pd.DataFrame(extracted_data)
#         if not result_df.empty:
#             result_df = result_df[result_df['extracted_text'].notna()]
#             result_df = result_df[result_df['extracted_text'].str.strip() != ""]
#         result_df.to_csv(output_path, index=False)
#         print(f"Saved {len(result_df)} results to {output_path}")
    
#     return len(result_df)

# if __name__ == "__main__":
#     # Configuration
#     INPUT_FILE = "Huggingface/Resolution_Note_Dec.xlsx"
#     OUTPUT_FILE = "extracted_text_dec_30_issues.csv"
#     MODEL_PATH = "llama-2-7b.Q2_K.gguf"
    
#     print("Starting extraction process...")
#     try:
#         num_matches = process_excel_with_llama_limited(INPUT_FILE, OUTPUT_FILE, MODEL_PATH, cell_limit=30)
#         print(f"Successfully processed first 30 cells. Found {num_matches} issue-related matches.")
#     except Exception as e:
#         print(f"Error: {str(e)}")

import pandas as pd
import re
from llama_cpp import Llama
import time

# Initialize Llama model
model = Llama(model_path="llama-2-7b.Q2_K.gguf")

def extract_with_llama(text, model):
    """
    Use Llama to extract only the text after the word 'issue' from each cell.
    """
    if pd.isna(text) or not isinstance(text, str):
        return None

    # Check if "issue" exists in the text
    if "issue" not in text.lower():
        return None

    # Craft the prompt to focus only on the issue
    prompt = f"""
    Extract only the text that follows the word 'issue' from the input below. 
    Ignore everything else like resolution, status, or compliance status. Stop extracting at the next period.

    Input Text: {text}

    Extracted Issue:"""

    # Generate response using llama-cpp
    response = model.create_completion(
        prompt,
        max_tokens=100,
        temperature=0.1,
        stop=["\n"],
        echo=False
    )

    # Process the output to ensure valid extraction
    if response and 'choices' in response:
        generated_text = response['choices'][0]['text'].strip()

        # Use regex to cleanly extract text after "issue"
        match = re.search(r"(?i)issue[^\":]*:?(.+?)(\.|$)", generated_text)

        if match:
            return match.group(1).strip()

    return None

def process_excel(input_file, output_file, model, cell_limit=30):
    """
    Process the Excel file to extract issues and save them to a new file, limiting to the first N cells.
    """
    print("Reading Excel file...")
    df = pd.read_excel(input_file)

    # Automatically detect the text column
    text_column = None
    for column in df.columns:
        if df[column].dtype == object:  # Look for a text-based column
            text_column = column
            break

    if not text_column:
        raise ValueError("No text column detected in the Excel file.")

    print(f"Detected text column: {text_column}")

    extracted_data = []
    processed_cells = 0
    start_time = time.time()

    print(f"Processing first {cell_limit} cells...")
    for index, text in enumerate(df[text_column]):
        if processed_cells < cell_limit:
            if isinstance(text, str) and "issue" in text.lower():
                extracted = extract_with_llama(text, model)
                extracted_data.append(extracted if extracted else "")
            else:
                extracted_data.append("")
            processed_cells += 1
        else:
            # For rows beyond the cell limit, append empty strings
            extracted_data.append("")

        elapsed_time = time.time() - start_time
        speed = (processed_cells / elapsed_time) if processed_cells else 0
        print(f"Processed cell {processed_cells}/{cell_limit} ({speed:.2f} cells/second)")

    # Append the extracted issues to the DataFrame
    df['Extracted Issue'] = extracted_data

    # Save the extracted issues to a new Excel file
    df.to_excel(output_file, index=False)
    print(f"Extracted issues saved to {output_file}")


# Define input and output file paths
input_file = "Huggingface/Resolution_Note_Dec.xlsx"
output_file = "extracted_issues.xlsx"

# Process the Excel file
process_excel(input_file, output_file, model, cell_limit=30)
