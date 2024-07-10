import cv2
import pytesseract
import numpy as np
from pytesseract import Output

# Path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path based on your Tesseract installation

def extract_text_and_lines(image_path):
    """
    Extracts text and groups them into lines based on horizontal alignment.

    :param image_path: Path to the image file
    :return: List of tuples, each containing (text, (x, y, w, h)) where (x, y, w, h) are box coordinates
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be read.")

    # Use Tesseract to detect words/lines and their bounding boxes
    results = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    # Filter out non-empty text results
    lines = []
    current_line = {'text': '', 'left': float('inf'), 'top': float('inf'), 'width': 0, 'height': 0}
    
    for i in range(len(results['text'])):
        if int(results['conf'][i]) > 0:
            x, y, w, h = results['left'][i], results['top'][i], results['width'][i], results['height'][i]
            text = results['text'][i].strip()
            
            # Check if this text belongs to the current line based on horizontal alignment
            if abs(y - current_line['top']) <= 10:  # Adjust threshold as needed
                current_line['text'] += ' ' + text
                current_line['left'] = min(current_line['left'], x)
                current_line['top'] = min(current_line['top'], y)
                current_line['width'] = max(current_line['width'], x + w - current_line['left'])
                current_line['height'] = max(current_line['height'], y + h - current_line['top'])
            else:
                if current_line['text']:  # Ensure we have collected some text
                    lines.append((
                        current_line['text'],
                        (current_line['left'], current_line['top'], current_line['width'], current_line['height'])
                    ))
                # Start a new line
                current_line = {'text': text, 'left': x, 'top': y, 'width': w, 'height': h}
    
    # Append the last line
    if current_line['text']:
        lines.append((
            current_line['text'],
            (current_line['left'], current_line['top'], current_line['width'], current_line['height'])
        ))
    
    return lines

def draw_text_on_blank_image(lines, input_image_path, output_image_path):
    """
    Draws extracted text on a blank image.

    :param lines: List of tuples, each containing (text, (x, y, w, h)) where (x, y, w, h) are box coordinates
    :param input_image_path: Path to the input image file for reference size
    :param output_image_path: Path to save the output image with extracted text
    """
    # Read the input image to get its size
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        raise ValueError(f"Input image at path {input_image_path} could not be read.")
    
    # Get input image size
    height, width, _ = input_image.shape
    
    # Create a blank white image with the same size as the input image
    output_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw text on the blank image
    for text, (x, y, w, h) in lines:
        # Calculate text size and position
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        
        # Draw the text on the image
        cv2.putText(output_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Save the image with extracted text
    cv2.imwrite(output_image_path, output_image)

def save_text_to_file(lines, output_text_path):
    """
    Saves extracted text to a text file.

    :param lines: List of tuples, each containing (text, (x, y, w, h)) where (x, y, w, h) are box coordinates
    :param output_text_path: Path to save the output text file
    """
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for text, _ in lines:
            f.write(f"{text}\n")

# Main execution block
if __name__ == "__main__":
    # Update these paths with your input and output paths
    input_image_path = r'C:\Users\USER\Desktop\Dummy\KDK.jpg'  # Path to the input image file
    output_image_path = 'text_overlay_output.png'  # Path to save the output image with extracted text
    output_text_path = 'extracted_text.txt'  # Path to save the output text file
    
    # Extract text and lines from the input image
    lines = extract_text_and_lines(input_image_path)
    
    # Draw extracted text on a blank image and save it
    draw_text_on_blank_image(lines, input_image_path, output_image_path)
    
    # Save the extracted text to a text file
    save_text_to_file(lines, output_text_path)
    
    # Print confirmation
    print(f"Image with extracted text saved to {output_image_path}")
    print(f"Extracted text saved to {output_text_path}")
