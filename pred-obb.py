from ultralytics import YOLO
 
CFG = r"D:\Desktop\best.pt"

model = YOLO(CFG) 

metrics = model.predict(source = r'D:\Desktop\train\images',
                    save_txt = True,
                    workers = 0
                    ) 