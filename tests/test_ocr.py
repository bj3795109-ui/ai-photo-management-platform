from PIL import Image, ImageDraw
from pipeline.document_classifier import classify_document

def main():
    img = Image.new('RGB', (400, 200), color='white')
    d = ImageDraw.Draw(img)
    # Using substantial text with prescription keywords
    text = "Patient: John Doe\nRx: 50mg amoxicillin capsule\nTake 1 tablet daily\nDoctor: Dr. Smith\nRefill: 3 times\nDosage: As needed for medication"
    d.text((10,10), text, fill='black')
    img.save('test_rx.png')

    print("Image saved as test_rx.png")
    
    category = classify_document('test_rx.png')
    print(f"Classification result: {category}")

if __name__ == '__main__':
    main()
