import gradio as gr
import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import requests
import uuid
import io
import difflib

# Configuration
MODEL_NAME = "google/gemma-2b-it"
CURRENT_USER = "AkarshanGupta"
CURRENT_TIME = "2025-03-23 05:05:24"

# API Keys
HF_TOKEN = os.getenv('HF_TOKEN')
AZURE_TRANSLATION_KEY = os.getenv('AZURE_TRANSLATION_KEY')

class TextExtractor:
    @staticmethod
    def extract_text_from_input(input_file):
        try:
            # If input is already a string, return it directly
            if isinstance(input_file, str):
                return input_file
            
            # If no file is provided
            if input_file is None:
                return ""
            
            # Get the file content
            if hasattr(input_file, 'name'):
                file_extension = os.path.splitext(input_file.name.lower())[1]
                
                # Handle PDF files
                if file_extension == '.pdf':
                    try:
                        # Read the file into memory
                        file_content = input_file.read()
                        # Create a file-like object
                        pdf_stream = io.BytesIO(file_content)
                        # Open PDF from the stream
                        doc = fitz.open(stream=pdf_stream, filetype="pdf")
                        text = ""
                        for page in doc:
                            text += page.get_text() + "\n\n"
                        doc.close()
                        return text.strip()
                    except Exception as e:
                        return f"Error extracting text from PDF: {str(e)}"
                
                # Handle image files
                elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                    try:
                        # Read the image file
                        image_content = input_file.read()
                        # Create an image object from bytes
                        image = Image.open(io.BytesIO(image_content))
                        # Convert image to text using pytesseract
                        text = pytesseract.image_to_string(image)
                        return text.strip()
                    except Exception as e:
                        return f"Error extracting text from image: {str(e)}"
                
                # Handle text files
                elif file_extension == '.txt':
                    try:
                        content = input_file.read()
                        if isinstance(content, bytes):
                            return content.decode('utf-8').strip()
                        return content.strip()
                    except Exception as e:
                        return f"Error extracting text from text file: {str(e)}"
            
            # If input is a PIL Image object
            if isinstance(input_file, Image.Image):
                try:
                    return pytesseract.image_to_string(input_file).strip()
                except Exception as e:
                    return f"Error extracting text from image: {str(e)}"
            
            return "Unsupported input type or empty file"
            
        except Exception as e:
            return f"Error processing input: {str(e)}"

class Translator:
    def __init__(self):
        self.key = AZURE_TRANSLATION_KEY
        self.region = 'centralindia'
        self.endpoint = "https://api.cognitive.microsofttranslator.com"
        
        if not self.key:
            raise ValueError("Azure Translator not configured. Please set AZURE_TRANSLATION_KEY in Spaces settings.")

    def translate_text(self, text, target_language="en"):
        try:
            bullet_points = text.split('\n• ')
            translated_points = []
            
            for point in bullet_points:
                if point.strip():
                    path = '/translate'
                    constructed_url = self.endpoint + path
                    
                    params = {
                        'api-version': '3.0',
                        'to': target_language
                    }
                    
                    headers = {
                        'Ocp-Apim-Subscription-Key': self.key,
                        'Ocp-Apim-Subscription-Region': self.region,
                        'Content-type': 'application/json',
                        'X-ClientTraceId': str(uuid.uuid4())
                    }
                    
                    body = [{
                        'text': point.strip()
                    }]
                    
                    response = requests.post(
                        constructed_url,
                        params=params,
                        headers=headers,
                        json=body
                    )
                    response.raise_for_status()
                    
                    translation = response.json()[0]["translations"][0]["text"]
                    translated_points.append(translation)
            
            translated_text = '\n• ' + '\n• '.join(translated_points)
            return translated_text
            
        except Exception as e:
            return f"Translation error: {str(e)}"

class LegalEaseAssistant:
    def __init__(self):
        if not HF_TOKEN:
            raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        
        login(token=HF_TOKEN)
        
        self.text_extractor = TextExtractor()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            token=HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float32
        )
    
    def format_response(self, text):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        bullet_points = ['• ' + s + '.' for s in sentences]
        return '\n'.join(bullet_points)
    
    def generate_response(self, input_file, task_type):
        text = self.text_extractor.extract_text_from_input(input_file)
        
        task_prompts = {
            "simplify": f"Simplify the following legal text in clear, plain language. Provide the response as separate points:\n\n{text}\n\nSimplified explanation:",
            "summary": f"Provide a concise summary of the following legal document as separate key points:\n\n{text}\n\nSummary:",
            "key_terms": f"Identify and explain the key legal terms and obligations in this text as separate points:\n\n{text}\n\nKey Terms:",
            "risk": f"Perform a risk analysis on the following legal document and list each risk as a separate point:\n\n{text}\n\nRisk Assessment:",
            "compare": f"Compare the following contract sections and identify key differences:\n\n{text}\n\nDifferences:"
        }
        
        prompt = task_prompts.get(task_type, f"Analyze the following text and provide points:\n\n{text}\n\nAnalysis:")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=300,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_parts = response.split(prompt.split("\n\n")[-1])
        raw_response = response_parts[-1].strip() if len(response_parts) > 1 else response.strip()
        
        return self.format_response(raw_response)

    def compare_contracts(self, contract1, contract2):
        """
        Compare two contracts and highlight the differences
        """
        text1 = self.text_extractor.extract_text_from_input(contract1)
        text2 = self.text_extractor.extract_text_from_input(contract2)
        
        # Split texts into lines
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        # Generate the diff
        differ = difflib.Differ()
        diff = list(differ.compare(lines1, lines2))
        
        # Format the differences
        differences = {
            'added': [],
            'removed': [],
            'changed': []
        }
        
        for line in diff:
            if line.startswith('+ '):
                differences['added'].append(line[2:])
            elif line.startswith('- '):
                differences['removed'].append(line[2:])
            elif line.startswith('? '):
                continue
            else:
                differences['changed'].append(line[2:])
        
        # Generate a summary of differences
        summary = []
        if differences['removed']:
            summary.append("Removed Content:")
            summary.extend(['• ' + line for line in differences['removed']])
        
        if differences['added']:
            summary.append("\nAdded Content:")
            summary.extend(['• ' + line for line in differences['added']])
        
        if differences['changed']:
            summary.append("\nModified Content:")
            summary.extend(['• ' + line for line in differences['changed']])
        
        return '\n'.join(summary)

def create_interface():
    assistant = LegalEaseAssistant()
    translator = Translator()
    
    SUPPORTED_LANGUAGES = {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn",
        "Telugu": "te",
        "Tamil": "ta",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Punjabi": "pa",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese (Simplified)": "zh-Hans",
        "Japanese": "ja"
    }
    
    def process_with_translation(func, *args, target_lang="English"):
        result = func(*args)
        if target_lang != "English":
            result = translator.translate_text(result, SUPPORTED_LANGUAGES[target_lang])
        return result

    with gr.Blocks(title="LegalEase", css="""
        .gradio-container {max-width: 1200px; margin: auto;}
        .header {text-align: center; margin-bottom: 2rem;}
        .content {padding: 2rem;}
    """) as demo:
        gr.HTML(f"""
        <div style="text-align: center; background-color: #e0e0e0; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: #2c3e50; font-size: 2.5em; margin-bottom: 10px;">📜 LegalEase</h1>
            <h2 style="color: #34495e; font-size: 1.5em; margin-bottom: 20px;">AI-Powered Legal Document Assistant</h2>
            <div style="display: flex; justify-content: center; gap: 40px; color: #576574; font-size: 1.1em;">
                <div style="background-color: #e0e0e0; padding: 10px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <span style="font-weight: bold;">User:</span> {CURRENT_USER}
                </div>
                <div style="background-color: #e0e0e0; padding: 10px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <span style="font-weight: bold;">Last Updated:</span> {CURRENT_TIME} UTC
                </div>
            </div>
        </div>
        """)
        
        language_selector = gr.Dropdown(
            choices=list(SUPPORTED_LANGUAGES.keys()),
            value="English",
            label="Select Output Language",
            scale=1
        )
        
        with gr.Tabs():
            # Simplify Language Tab
            with gr.Tab("📝 Simplify Language"):
                with gr.Row():
                    with gr.Column(scale=1):
                        simplify_input = gr.File(
                            label="📎 Upload Document",
                            file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        simplify_text_input = gr.Textbox(
                            label="✍ Or Type/Paste Text",
                            placeholder="Enter your legal text here...",
                            lines=4
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        simplify_btn = gr.Button(
                            "🔍 Simplify Language",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        simplify_output = gr.Textbox(
                            label="📋 Simplified Explanation",
                            lines=12,
                            show_copy_button=True
                        )
                
                def simplify_handler(file, text, lang):
                    input_source = file or text
                    if not input_source:
                        return "Please provide some text or upload a document to analyze."
                    return process_with_translation(
                        assistant.generate_response,
                        input_source,
                        "simplify",
                        target_lang=lang
                    )
                
                simplify_btn.click(
                    fn=simplify_handler,
                    inputs=[simplify_input, simplify_text_input, language_selector],
                    outputs=simplify_output
                )

            # Document Summary Tab
            with gr.Tab("📚 Document Summary"):
                with gr.Row():
                    with gr.Column(scale=1):
                        summary_input = gr.File(
                            label="📎 Upload Document",
                            file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        summary_text_input = gr.Textbox(
                            label="✍ Or Type/Paste Text",
                            placeholder="Enter your legal document here...",
                            lines=4
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        summary_btn = gr.Button(
                            "📋 Generate Summary",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        summary_output = gr.Textbox(
                            label="📑 Document Summary",
                            lines=12,
                            show_copy_button=True
                        )
                
                def summary_handler(file, text, lang):
                    input_source = file or text
                    if not input_source:
                        return "Please provide some text or upload a document to summarize."
                    return process_with_translation(
                        assistant.generate_response,
                        input_source,
                        "summary",
                        target_lang=lang
                    )
                
                summary_btn.click(
                    fn=summary_handler,
                    inputs=[summary_input, summary_text_input, language_selector],
                    outputs=summary_output
                )

            # Key Terms Tab
            with gr.Tab("🔑 Key Terms"):
                with gr.Row():
                    with gr.Column(scale=1):
                        terms_input = gr.File(
                            label="📎 Upload Document",
                            file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        terms_text_input = gr.Textbox(
                            label="✍ Or Type/Paste Text",
                            placeholder="Enter your legal document here...",
                            lines=4
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        terms_btn = gr.Button(
                            "🔍 Extract Key Terms",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        terms_output = gr.Textbox(
                            label="🔑 Key Terms & Definitions",
                            lines=12,
                            show_copy_button=True
                        )
                
                def terms_handler(file, text, lang):
                    input_source = file or text
                    if not input_source:
                        return "Please provide some text or upload a document to analyze key terms."
                    return process_with_translation(
                        assistant.generate_response,
                        input_source,
                        "key_terms",
                        target_lang=lang
                    )
                
                terms_btn.click(
                    fn=terms_handler,
                    inputs=[terms_input, terms_text_input, language_selector],
                    outputs=terms_output
                )

            # Risk Analysis Tab
            with gr.Tab("⚠ Risk Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        risk_input = gr.File(
                            label="📎 Upload Document",
                            file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        risk_text_input = gr.Textbox(
                            label="✍ Or Type/Paste Text",
                            placeholder="Enter your legal document here...",
                            lines=4
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        risk_btn = gr.Button(
                            "🔍 Analyze Risks",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=1):
                        risk_output = gr.Textbox(
                            label="⚠ Risk Assessment",
                            lines=12,
                            show_copy_button=True
                        )
                
                def risk_handler(file, text, lang):
                    input_source = file or text
                    if not input_source:
                        return "Please provide some text or upload a document to analyze risks."
                    return process_with_translation(
                        assistant.generate_response,
                        input_source,
                        "risk",
                        target_lang=lang
                    )
                
                risk_btn.click(
                    fn=risk_handler,
                    inputs=[risk_input, risk_text_input, language_selector],
                    outputs=risk_output
                )

            # Contract Comparison Tab
            with gr.Tab("📊 Contract Comparison"):
                with gr.Row():
                    with gr.Column(scale=1):
                        contract1_input = gr.File(
                            label="📎 Upload First Contract",
                            file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        contract1_text = gr.Textbox(
                            label="✍ Or Type/Paste First Contract",
                            placeholder="Enter your first contract here...",
                            lines=4
                        )
                    
                    with gr.Column(scale=1):
                        contract2_input = gr.File(
                            label="📎 Upload Second Contract",
                            file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
                        )
                        gr.HTML("<div style='height: 10px'></div>")
                        contract2_text = gr.Textbox(
                            label="✍ Or Type/Paste Second Contract",
                            placeholder="Enter your second contract here...",
                            lines=4
                        )
                
                compare_btn = gr.Button(
                    "🔍 Compare Contracts",
                    variant="primary"
                )
                
                comparison_output = gr.Textbox(
                    label="📊 Comparison Results",
                    lines=15,
                    show_copy_button=True
                )
                
                def compare_handler(contract1_file, contract1_text, contract2_file, contract2_text, lang):
                    contract1 = contract1_file or contract1_text
                    contract2 = contract2_file or contract2_text
                    
                    if not contract1 or not contract2:
                        return "Please provide both contracts for comparison."
                    
                    comparison = assistant.compare_contracts(contract1, contract2)
                    if lang != "English":
                        comparison = translator.translate_text(comparison, SUPPORTED_LANGUAGES[lang])
                    return comparison
                
                compare_btn.click(
                    fn=compare_handler,
                    inputs=[
                        contract1_input,
                        contract1_text,
                        contract2_input,
                        contract2_text,
                        language_selector
                    ],
                    outputs=comparison_output
                )

        gr.HTML(f"""
        <div style="text-align: center; margin-top: 20px; padding: 20px; background-color: #e0e0e0; border-radius: 10px;">
            <p style="color: #576574; margin: 0;">Made by Team Ice Age</p>
        </div>
        """)

    return demo

def main():
    demo = create_interface()
    demo.queue()
    demo.launch(share=True)

if __name__ == "__main__":
    main()