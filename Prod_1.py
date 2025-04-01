import os
import time
from llama_cpp import Llama

# CONFIGURATION
MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Update this path

# Sample data directly in the script
CATEGORIES = [
    {
        "id": "Set 1",
        "text": "Regional/Global Centers for mailroom servicesSchedules available for each location with frequency determined based on the volume of invoices, and selected mailboxes are cleared more than once a dayAutomated Invoice Receipt date capture with 100% same day scanning from various sources like Mailroom, Email, Mobile ScanContextual availability of Internal & external best practices enabling decision making on the goWell defined Supplier Information Portal with ease of access and support structure to enable suppliers on the right channels"
    },
    {
        "id": "Set 2",
        "text": "End-to-end AI/ML integration with predictive/forecasting capabilities.Electronic invoicesMaturing ability to track invoice status.Comprehensive API platform supporting multiple protocols and bidirectional integrations with various systems and servicesOCR extracting Header level +Line level detailsComprehensive data captured and leveraged by the Bl systems to build process insights."
    },
    {
        "id": "Set 3",
        "text": "Tax engine embedded in the ERP with 1st level exceptions saved by ML and 2nd level routed for accounting reviewSegregation of Duties (SODs) are integrated into the workflow, with a standard set of access permissions defined according to the access category.No manually approved invoices with Approval matrix is integrated in the workflow and is routed automaticallyThe approval matrix and DOA are Rationalized and automated, integrated into the workflow with real-time communication of changes to the matrixAutomated invoice sampling with Priority given basis pre-defined criteria and Multiple review levels are applied based on invoice amounts."
    }
]

def main():
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        return
    
    print(f"Loading Mistral 7B Instruct model from: {MODEL_PATH}")
    try:
        # Load the model with optimized parameters for Mistral
        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,       # Mistral supports larger context
            n_batch=512,      # Batch size for prompt processing
            verbose=False     # Set to True for debugging
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process each category
    for category in CATEGORIES:
        category_id = category["id"]
        text = category["text"]
        
        print(f"\nProcessing: {category_id}")
        
        # Create a prompt using Mistral's specific instruction format
        prompt = f"<s>[INST] You are a professional summarizer. Create a concise, clear summary (1-2 sentences) of the following text from {category_id}:\n\n{text}\n\nProvide only the summary without any other text. [/INST]"
        
        try:
            # Generate with parameters optimized for Mistral summarization
            response = model.create_completion(
                prompt=prompt,
                max_tokens=150,      # Reasonable length for summary
                temperature=0.1,     # Low temperature for focused output
                top_p=0.9,
                top_k=40,            # Limit vocabulary for more focused output
                repeat_penalty=1.15, # Prevent repetition
                stop=["</s>", "[INST]", "User:"],  # Stop tokens for Mistral
                echo=False
            )
            
            # Extract and clean summary
            summary = ""
            if response and "choices" in response and len(response["choices"]) > 0:
                summary = response["choices"][0]["text"].strip()
                
                # Clean up any instruction artifacts
                summary = summary.replace("[/INST]", "").strip()
                
                # Additional cleaning if needed
                prefixes_to_remove = [
                    "Here is a concise summary:", 
                    "The summary is:", 
                    "Summary:", 
                    "Here's a summary:"
                ]
                
                for prefix in prefixes_to_remove:
                    if summary.startswith(prefix):
                        summary = summary[len(prefix):].strip()
            
            # Print result in the requested format
            print(f"Category id - {category_id}")
            print(f'Generated summary - "{summary}"')
            print("-" * 80)
            
            # Add a small delay between requests 
            time.sleep(1)
            
        except Exception as e:
            print(f"Error generating summary: {e}")
    
if __name__ == "__main__":
    main()