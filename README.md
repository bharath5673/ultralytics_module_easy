
### select model
```
# Load a model
model = YOLO('yolov5n.pt')  # load an official model
# model = YOLO('yolov5n-seg.pt')  # load an official model
# model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('yolov8n-pose.pt')  # load an official model
# model = YOLO('yolov8n-seg.pt')  # load an official model
# model = YOLO('yolo11n.pt')  # load an official model
# model = YOLO('yolo11n-seg.pt')  # load an official model



###MODEL OPTIMIZE
#model.export(format="onnx")
# model = YOLO('yolov8n-seg.onnx')  # load an official model

# model.export(format="engine")
# model = YOLO('yolov8n-seg.engine')  # load an official model
```


### and run
```
run main.py
```
