from ultralytics import YOLO

# Load a model
model = YOLO("./pths/yolo11x.pt")  # pretrained YOLO11n model
# Run batched inference on a list of images
results = model(["./tmp/person.png"], stream=False, classes=[0, 1, 3])  # return a list of Results objects
# NOTE: classes parse to Model's __call__ function, which is set to predict() in YOLO class

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk