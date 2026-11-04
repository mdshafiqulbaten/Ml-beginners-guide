# torchvision_transforms.py
# Auto-extracted from "Machine Learning: The Complete Beginner's Guide" by Md Shafiqul Baten Sumon.
# Extraction method: text pulled from the book's PDF and, for Python
# files, auto-repaired for indentation. Syntax check: NOT CONFIRMED - please review.
# Review before running, especially if the status above is not "passed".

(needs torchvision and image data). Syntax-checked only.
 """Image augmentation pipelines: random changes for training, none for validation."""
 from torchvision import transforms

 train_transform = transforms.Compose([
     transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
     transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
     transforms.RandomRotation(10),
     transforms.ToTensor(),
     # ImageNet statistics
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
 ])
 # NO random changes here
 val_transform = transforms.Compose([
     transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
 ])
 # Do not flip images where orientation carries meaning (text, digits, left/right medical
 scans).

Does augmentation really help? We measured it on a deliberately small training set
(150 images). The training/test gap shrank from 0.069 to 0.036, but test accuracy
hardly changed (0.931 to 0.931 with all remedies together). Regularization is not
