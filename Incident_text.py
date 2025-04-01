import pandas as pd
from llama_cpp import Llama
import time
import re

def setup_llama(model_path="llama-2-7b.Q2_K.gguf"):
    """
    Initialize Llama model using llama-cpp
    """
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0  # Use CPU
    )
    return model

# def extract_with_llama(text, model):
    """
    Use Llama to extract text after 'issue' until the period
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
        
    # First check if text contains "issue"
    if "issue" not in text.lower():
        return None
    
    prompt = f"""
    Find the line that starts with or contains the word 'issue' and extract ONLY the text that comes after the word 'issue' until the next period.Ignore resolution , status , mode of communication compliance status.
    If there are multiple lines with 'issue', extract each one.
    
    Text: {text}
    
    Extracted text:"""
    
    # Generate response using llama-cpp
    response = model.create_completion(
        prompt,
        max_tokens=100,
        temperature=0.1,
        stop=["\n", "."],
        echo=False
    )
    
    # Extract the generated text
    if response and 'choices' in response:
        extracted = response['choices'][0]['text'].strip()
        # Additional cleaning to ensure we only get text after "issue"
        if "issue" in extracted.lower():
            extracted = re.split(r'issue\s*', extracted, flags=re.IGNORECASE)[-1]
        return extracted if extracted else None
    return None


def extract_with_llama(text, model):
    """
    Use Llama to extract text after 'issue' until the next period
    """
    if pd.isna(text) or not isinstance(text, str):
        return None

    # First check if text contains "issue"
    if "issue" not in text.lower():
        return None

    prompt = f"""
    Find the line that starts with or contains the word 'issue' and extract ONLY the text that comes after the word 'issue' until the next period. Ignore resolution, status, mode of communication, and compliance status.

    Text: {text}

    Extracted text:"""

    # Generate response using llama-cpp
    response = model.create_completion(
        prompt,
        max_tokens=100,
        temperature=0.1,
        stop=["\n"],
        echo=False
    )

    # Extract the generated text
    if response and 'choices' in response:
        generated_text = response['choices'][0]['text'].strip()

        # Clean up to ensure we only get text after "issue" and truncate at the period
        if "issue" in generated_text.lower():
            match = re.search(r"issue\s*:\s*(.*?)(\.|$)", generated_text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

    return None

def process_excel_with_llama_limited(input_path, output_path, model_path="llama-2-7b.Q2_K.gguf", cell_limit=30):
    """
    Process Excel file using Llama for text extraction, limited to first N cells
    """
    print("Loading Llama model...")
    model = setup_llama(model_path)
    
    print("Reading Excel file...")
    df = pd.read_excel(input_path)
    
    extracted_data = []
    processed_cells = 0
    start_time = time.time()
    
    print(f"Processing first {cell_limit} cells...")
    try:
        for column in df.columns:
            for text in df[column]:
                if processed_cells >= cell_limit:
                    break
                    
                if isinstance(text, str) and "issue" in text.lower():
                    extracted = extract_with_llama(text, model)
                    if extracted:
                        extracted_data.append({
                            'original_text': text,
                            'extracted_text': extracted,
                            'column': column,
                            'cell_number': processed_cells + 1
                        })
                
                processed_cells += 1
                elapsed_time = time.time() - start_time
                speed = processed_cells / elapsed_time
                print(f"Processed cell {processed_cells}/{cell_limit} "
                      f"({speed:.2f} cells/second)")
                
            if processed_cells >= cell_limit:
                break
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        if extracted_data:
            print("Saving partial results...")
    
    finally:
        # Save results and remove any empty extractions
        result_df = pd.DataFrame(extracted_data)
        result_df = result_df[result_df['extracted_text'].str.len() > 0]
        result_df.to_csv(output_path, index=False)
        print(f"Saved {len(result_df)} results to {output_path}")
    
    return len(result_df)

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "Huggingface/Resolution_Note_Dec.xlsx"
    OUTPUT_FILE = "extracted_text_dec_30_issues.csv"
    MODEL_PATH = "llama-2-7b.Q2_K.gguf"
    
    print("Starting extraction process...")
    try:
        num_matches = process_excel_with_llama_limited(INPUT_FILE, OUTPUT_FILE, MODEL_PATH, cell_limit=30)
        print(f"Successfully processed first 30 cells. Found {num_matches} issue-related matches.")
    except Exception as e:
        print(f"Error: {str(e)}")