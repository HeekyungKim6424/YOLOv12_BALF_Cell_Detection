import time
import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import traceback

# ==========================================
# 1. 경로 설정 (기존과 동일)
# ==========================================
dataset_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\AI-Cell-Image-154\data.yaml"
summary_csv_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\model_comparison_results\AI-Cell-Image-154_ALL_MODELS_SUMMARY.csv"

# 딱 하나만 돌릴 모델 이름 지정
model_name = "yolo12x.pt"

if __name__ == '__main__':
    print(f"\n🚀 [ {model_name} ] 단독 학습 및 CSV 추가 스크립트 시작!\n")

    try:
        model = YOLO(model_name)
        
        # --- [1. 학습 시간 측정 시작] ---
        start_time = time.time()
        
        # 💡 이전 누락 원인이 메모리 부족일 수 있으므로 batch 사이즈를 수동으로 낮추는 것을 권장합니다.
        # 메모리 에러가 안 난다면 batch 파라미터를 지우셔도 됩니다.
        results = model.train(data=dataset_path, epochs=300, imgsz=640, batch=8) 
        
        # --- [2. 학습 시간 측정 종료] ---
        end_time = time.time()
        train_time_sec = end_time - start_time
        
        train_hours = int(train_time_sec // 3600)
        train_mins = int((train_time_sec % 3600) // 60)
        train_time_str = f"{train_hours}h {train_mins}m"
        
        # --- [3. 모델 속도 및 성능 추출] ---
        metrics = model.trainer.validator.metrics
        inference_time_ms = round(metrics.speed.get('inference', 0.0), 3)
        mean_p, mean_r, mean_map50, mean_map = metrics.mean_results()
        
        # CSV에 들어갈 새 데이터 딕셔너리 생성
        new_row = {
            "Model": model_name,
            "Class": "all",
            "Instances": int(sum(metrics.nt_per_class)),
            "Box(P)": round(mean_p, 3),
            "R": round(mean_r, 3),
            "mAP50": round(mean_map50, 3),
            "mAP50-95": round(mean_map, 3),
            "Train_Time(sec)": round(train_time_sec, 2),
            "Train_Time_str": train_time_str,
            "Inference_Time(ms/img)": inference_time_ms
        }
        
        # ==========================================
        # 4. 기존 CSV 파일에 덧붙이기 (Append)
        # ==========================================
        new_df = pd.DataFrame([new_row])
        
        if os.path.exists(summary_csv_path):
            # 파일이 존재하면 모드를 'a' (append)로 설정하고 헤더를 빼고 아래에 이어붙임
            new_df.to_csv(summary_csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"\n✅ 기존 요약 파일({os.path.basename(summary_csv_path)})에 {model_name} 결과 추가 완료!")
        else:
            # 혹시 파일이 없다면 새로 생성
            new_df.to_csv(summary_csv_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            print(f"\n✅ 요약 파일이 없어 새로 생성하여 저장 완료!")
            
        print(f"⏱️ 소요 시간: {train_time_str}, 추론 속도: {inference_time_ms}ms/img")
        print("-" * 70)

    except Exception as e:
        print(f"\n❌ [{model_name}] 학습 중 에러 발생!")
        print(traceback.format_exc())