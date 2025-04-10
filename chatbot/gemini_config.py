import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def configure_gemini():
    """Configure the Gemini AI client"""
    genai.configure(api_key=GEMINI_API_KEY)
    
def get_gemini_model():
    """Get the Gemini Pro Vision model for image and text processing"""
    return genai.GenerativeModel('gemini-pro-vision')

def generate_product_prompt(product_name):
    """Generate a prompt for product details"""
    return f"""
    Please provide details about the following product: {product_name}
    Include:
    1. A detailed description
    2. The current market price
    3. A URL to a representative product image
    Format the response as JSON with the following structure:
    {{
        "description": "product description",
        "price": "product price",
        "image_url": "image url"
    }}
    """
