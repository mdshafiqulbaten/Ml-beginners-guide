# yolo_detection.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

internet access and the ultralytics package). Syntax-checked only.
 """Object detection with a pretrained YOLO model."""
 from ultralytics import YOLO

 # smallest model; downloaded on first use (name may change in newer releases)
 model = YOLO("yolov8n.pt")
 results = model("street.jpg")              # your own image
 for box in results[0].boxes:
     print(results[0].names[int(box.cls)], f"{float(box.conf):.2f}",
 box.xyxy[0].tolist())

Image Segmentation: Pixel-Level Understanding
Object detection draws boxes around objects. Image segmentation classifies every
single pixel. This is needed when precise boundaries matter — outlining a tumor in
an MRI scan, separating road from sidewalk for autonomous driving, or selecting a
subject from its background in photo editing.
Semantic segmentation: classify each pixel into a category (sky, road, car, person). All
car pixels get the same label regardless of which specific car they belong to. Instance
segmentation: additionally distinguish between different instances of the same class.
Car 1 gets label A, Car 2 gets label B. Panoptic segmentation: combines both —
semantic labels for background stuff (sky, road), instance labels for foreground
things (each car, each person).

Key Terms
Term                                        Meaning

Convolution                                 Sliding a small grid of weights (a filter) over an
                                            image to detect a local pattern.
Filter (kernel)                             The small grid of learned weights used in a
                                            convolution.
Pooling                                     Shrinking a feature map by keeping, for
                                            example, the maximum of each small region.
Transfer learning                           Starting from a network trained on a large
                                            dataset and adapting it to your task.
Data augmentation                           Creating extra training examples by altering
                                            existing ones (flips, shifts, crops).

Check Your Understanding
Try these before you read on. Model answers are in Appendix D. Exercises that
mention a script use the free companion code.
  1.      An image is 224 x 224 x 3. How many weights would one fully connected
