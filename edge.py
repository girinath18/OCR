import cv2

# Load the image
image = cv2.imread('KDK.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Canny edge detector to find edges
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Inpainting to remove detected edges (replace with surrounding texture)
result = cv2.inpaint(image, edges, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Save the result image
cv2.imwrite('result_image.jpg', result)

print("Result image saved as 'result_image.jpg'")
