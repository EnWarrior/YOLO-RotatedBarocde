from ultralytics import YOLO
 
model = YOLO('/root/autodl-tmp/YOLO-RotatedBarocde/ultralytics/cfg/models/v8/yolov8-obb.yaml')
 
model.train(
            workers = 8,
            epochs = 1,
            project = '/root/autodl-tmp/YOLO-RotatedBarocde/runs',
            name = '00_test',
            exist_ok = True
            )