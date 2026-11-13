import os
from ultralytics import YOLO

# ==========================================
# 1. 기본 경로 및 세팅
# ==========================================
# 🌟 탐지할 이미지들이 들어있는 단일 폴더 경로
source_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\sample_photos\G22009_ag"

# Detection 결과가 저장될 최상위 프로젝트 경로
project_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\raw_ag"

# 최종적으로 결과가 저장될 폴더명 (project_dir 아래에 생성됨)
name_dir = "raw_am"

# 가중치 경로
weight_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\ultralytics\runs\detect\train3\weights\best.pt"

# ==========================================
# 2. Detection 실행
# ==========================================
print(f"\n{'='*50}\n[Start] Model Loading & Detection\n{'='*50}")

# 모델 로드
try:
    model = YOLO(weight_path) 
except Exception as e:
    print(f"가중치 로드 실패 ({weight_path}): {e}")
    exit() # 가중치 로드 실패 시 프로그램 종료

print(f"-> Detecting images in: {source_dir}")

# 예측 및 결과 저장 (단일 폴더 대상)
model.predict(
    source=source_dir,
    save=True,          
    project=project_dir, 
    name=name_dir,
    exist_ok=True,      
    verbose=True,       # 진행 상황(프로그레스 바)을 보려면 True로 두는 것을 권장합니다.
    line_width=3        
)

print("\n========== Detection 작업이 완료되었습니다! ==========")