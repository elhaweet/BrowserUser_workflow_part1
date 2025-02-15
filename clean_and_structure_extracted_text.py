import os
import json
import google.generativeai as genai
import pandas as pd
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

genai.configure(api_key=api_key)

def process_with_gemini(content: str) -> Optional[List[Dict[str, Any]]]:
    """Process text content using Gemini API to structure it."""
    model = genai.GenerativeModel('gemini-pro')
    prompt = """
    Analyze this data and structure it into a clean format suitable for CSV.
    Return the data as a valid JSON array of objects, where each object represents a property listing 
    with these fields: name, location, price, be1drooms, bathrooms, area.
    If a field is missing, use an empty string.
    Format the response as pure JSON without any additional text or explanation.
    
    Data to analyze: {content}
    """
    
    try:
        response = model.generate_content(prompt.format(content=content))
        response_text = response.text.strip()
        
        # Remove any markdown formatting if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
            
        # Parse the JSON response
        structured_data = json.loads(response_text)
        
        # Ensure we have a list of dictionaries
        if isinstance(structured_data, dict):
            structured_data = [structured_data]
        
        # Validate and standardize each entry
        standardized_data = []
        required_fields = ['name', 'location', 'price', 'bedrooms', 'bathrooms', 'area']
        
        for entry in structured_data:
            standardized_entry = {field: str(entry.get(field, '')) for field in required_fields}
            standardized_data.append(standardized_entry)
            
        return standardized_data
        
    except json.JSONDecodeError as e:
        print(f"Error parsing Gemini response as JSON: {str(e)}")
        print(f"Raw response: {response_text}")
        return None
    except Exception as e:
        print(f"Error in Gemini processing: {str(e)}")
        return None

def save_to_csv(data: List[Dict[str, Any]], directory: str, base_filename: str) -> str:
    """Save structured data to CSV file."""
    os.makedirs(directory, exist_ok=True)
    df = pd.DataFrame(data)
    csv_path = os.path.join(directory, f"{base_filename}_structured.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def extract_and_process_file(file_path: str) -> Optional[str]:
    """
    Extract content from file, process with Gemini, and save as CSV.
    
    Args:
        file_path (str): Path to input text file
        
    Returns:
        Optional[str]: Path to saved CSV file, or None if processing failed
    """
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Process with Gemini
        structured_data = process_with_gemini(text)
        if not structured_data:
            print("Failed to structure the data")
            return None
        
        # Save as CSV
        directory = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_path = save_to_csv(structured_data, directory, file_name)
        
        print(f"Processed data saved to: {csv_path}")
        return csv_path
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    file_path = os.path.join("extracted_data", "workflow3", "output.txt")
    output_path = extract_and_process_file(file_path)
    
    if output_path:
        print(f"Successfully processed and saved to CSV: {output_path}")