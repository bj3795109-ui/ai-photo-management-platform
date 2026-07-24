import os
import pytesseract
from PIL import Image

# Configure Tesseract path for Windows if it exists
WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == 'nt' and os.path.exists(WINDOWS_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT_PATH

PRESCRIPTION_KEYWORDS = {'rx', 'pharmacy', 'patient', 'mg', 'tablet', 'capsule', 'doctor', 'dr.', 'refill', 'dosage', 'take', 'daily', 'medication'}
RECEIPT_KEYWORDS = {'total', 'tax', 'cash', 'card', 'subtotal', 'visa', 'mastercard', 'change due', 'receipt', 'amount', 'date', 'auth'}

def classify_document(image_path: str) -> str | None:
    """
    Attempts to classify an image based on extracted text using OCR.
    Returns 'prescriptions', 'receipts', 'documents', or None if not enough text.
    """
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        img = Image.open(image_path)
        # Convert to grayscale for better OCR
        img = img.convert('L')
        text = pytesseract.image_to_string(img).lower()
        
        # Clean text and split into words
        words = set(text.split())
        
        # If there's barely any text, it's not a document. Fallback to CLIP.
        if len(text.strip()) < 30 or len(words) < 5:
            return None
            
        # Check for prescription keywords
        if len(words.intersection(PRESCRIPTION_KEYWORDS)) >= 2:
            return "prescriptions"
            
        # Check for receipt keywords
        if len(words.intersection(RECEIPT_KEYWORDS)) >= 2:
            return "receipts"
            
        # Default to documents if it has substantial text but doesn't match specific types
        return "documents"

    except Exception as e:
        print(f"Error in OCR classification: {e}")
        return None
