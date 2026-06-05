from transformers import AutoProcessor, MedGemmaForImageClassification

processor = AutoProcessor.from_pretrained("google/medgemma-base")
model = MedGemmaForImageClassification.from_pretrained("google/medgemma-base", num_labels=9)

inputs = processor(images=your_image_tensor, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

-----------

code improvement

criterion = nn.BCEWithLogitsLoss()
# remove torch.sigmoid() in your forward loop
predictions = outputs > 0.5

from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()
...
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

from monai.transforms import RandRotate90, RandFlip, NormalizeIntensity


------
