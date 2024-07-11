import cv2
import pytesseract
from pytesseract import Output

# Path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path based on your Tesseract installation

def extract_text_and_boxes(image_path):
    """
    Extracts text and their bounding boxes from the image.

    :param image_path: Path to the image file
    :return: List of tuples, each containing (text, (x, y, w, h)) where (x, y, w, h) are box coordinates
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be read.")

    # Use Tesseract to detect words and their bounding boxes
    results = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    # Filter out non-empty text results
    boxes = []
    
    for i in range(len(results['text'])):
        if int(results['conf'][i]) > 0:
            x, y, w, h = results['left'][i], results['top'][i], results['width'][i], results['height'][i]
            text = results['text'][i].strip()
            boxes.append((text, (x, y, w, h)))
    
    return boxes

def draw_bounding_boxes(image_path, bounding_boxes, output_image_path):
    """
    Draws predefined bounding boxes on the original image.

    :param image_path: Path to the input image file
    :param bounding_boxes: List of tuples, each containing (x_min, y_min, width, height)
    :param output_image_path: Path to save the output image with bounding boxes
    """
    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be read.")
    
    # Define the color and thickness of the bounding boxes
    box_color = (0, 255, 0)  # Green color for bounding boxes
    box_thickness = 2
    
    # Draw predefined bounding boxes
    for (x_min, y_min, width, height) in bounding_boxes:
        x_max = x_min + width
        y_max = y_min + height
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, box_thickness)
    
    # Save the image with bounding boxes
    cv2.imwrite(output_image_path, image)

def group_text_by_bounding_box(extracted_boxes, bounding_boxes):
    """
    Groups text by the predefined bounding boxes.

    :param extracted_boxes: List of tuples, each containing (text, (x, y, w, h)) where (x, y, w, h) are box coordinates
    :param bounding_boxes: List of predefined bounding boxes
    :return: Dictionary where keys are bounding boxes and values are lists of grouped text
    """
    grouped_text = {box: [] for box in bounding_boxes}
    
    for text, (x, y, w, h) in extracted_boxes:
        for box in bounding_boxes:
            x_min, y_min, width, height = box
            x_max = x_min + width
            y_max = y_min + height
            if x_min <= x <= x_max and y_min <= y <= y_max:
                grouped_text[box].append(text)
                break
    
    return grouped_text

def save_grouped_text_to_file(grouped_text, output_text_path):
    """
    Saves grouped text to a text file.

    :param grouped_text: Dictionary where keys are bounding boxes and values are lists of grouped text
    :param output_text_path: Path to save the output text file
    """
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for texts in grouped_text.values():
            f.write(" ".join(texts) + "\n")

# Main execution block
if __name__ == "__main__":
    # Update these paths with your input and output paths
    input_image_path = r'C:\Users\USER\Desktop\Dummy\KDK.jpg'  # Path to the input image file
    output_image_path = 'bounding_boxes_output.png'  # Path to save the output image with bounding boxes
    output_text_path = 'extracted_text.txt'  # Path to save the extracted text
    
    # Predefined bounding box coordinates
    bounding_boxes = [
        (125, 170, 460, 40),  # Example bounding box 1: (x_min, y_min, width, height)
        (125, 220, 810, 140),  # Example bounding box 1: (x_min, y_min, width, height)
        (125, 360, 640, 35),  # Example bounding box 1: (x_min, y_min, width, height)
        (125, 400, 650, 40),  # Example bounding box 1: (x_min, y_min, width, height)
        (125, 440, 690, 45),  # Example bounding box 1: (x_min, y_min, width, height)
        (125, 495, 690, 45),  # Example bounding box 1: (x_min, y_min, width, height)
        (125, 560, 230, 45),  # Example bounding box 1: (x_min, y_min, width, height)
        (125, 610, 840, 44),
        (125, 660, 800, 185),
        (125, 850, 700, 45),
        (125, 900, 700, 45),
        (1330, 155, 290, 90),
        (1330, 250, 290, 60), #Delivery note
        (1795, 155, 290, 90),
        (1795, 250, 400, 90),
        (1795, 340, 400, 80),
        (1795, 440, 400, 80),
        (1795, 530, 400, 80),
        (1795, 620, 400, 80),
        (1330, 615, 400, 80),
        (1330, 535, 400, 80),
        (1330, 440, 400, 80),
        (1330, 350, 400, 80),
        (1330, 715, 300, 150),
        (170, 1030, 2100, 60),
        (170, 1150, 2110, 60),
        (170, 1250, 2110, 60),
        (170, 1350, 2110, 60),
        (170, 1440, 2110, 60),
        (170, 1530, 2110, 60),
        (125, 2500, 1110, 60),
        (120, 3200, 950, 150),
        (850, 3350, 590, 70),
        (850, 3400, 590, 70),
        
       
    ]
    
    # Extract text and bounding boxes from the input image
    extracted_boxes = extract_text_and_boxes(input_image_path)
    
    # Group text by predefined bounding boxes
    grouped_text = group_text_by_bounding_box(extracted_boxes, bounding_boxes)
    
    # Draw predefined bounding boxes on the original image and save it
    draw_bounding_boxes(input_image_path, bounding_boxes, output_image_path)
    
    # Save the grouped text to a text file
    save_grouped_text_to_file(grouped_text, output_text_path)
    
    print(f"Image with predefined bounding boxes saved to {output_image_path}")
    print(f"Extracted text saved to {output_text_path}")
