IMAGE_WIDTH = 28
INPUT_SIZE = IMAGE_WIDTH * IMAGE_WIDTH

SELECTED_CLASSES = [0,1,2,3,4,5,6,7,8,9]
OUTPUT_SIZE = len(SELECTED_CLASSES)
HIDDEN_NEURONS = 300
BATCH_SIZE = 128
selected_classes = SELECTED_CLASSES
DIGITS = ''.join(str(num) for num in selected_classes)
FILEPATH = rf"C:\\Users\\220429111\\Box\\University\\PhD\\Codes\\Python\\MNIST\\parameters\\parameters{DIGITS}relu.pth"
