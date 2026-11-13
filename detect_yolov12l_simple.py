import os
from ultralytics import YOLO

source_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\"
project_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\"

name_dir = "raw_am"
weight_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\"

print(f"\n{'='*50}\n[Start] Model Loading & Detection\n{'='*50}")

try:
    model = YOLO(weight_path) 
except Exception as e:
    print(f"로드 실패 ({weight_path}): {e}")
    exit()

print(f"-> Detecting images in: {source_dir}")

model.predict(
    source=source_dir,
    save=True,          
    project=project_dir, 
    name=name_dir,
    exist_ok=True,      
    verbose=True,       
    line_width=3       
)

print("\n========== 작업 완료 ==========")
