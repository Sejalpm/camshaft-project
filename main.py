import cv2
import easyocr

# Load image
image = cv2.imread("camshaft.jpg")

if image is None:
    print("Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize EasyOCR reader (English)
reader = easyocr.Reader(['en'], gpu=False)

# Perform OCR
results = reader.readtext(gray)

print("Detected text:")

if len(results) == 0:
    print("")
else:
    for result in results:
        print(result[1])