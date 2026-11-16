import time
import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import traceback

dataset_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell"
summary_csv_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell"


model_name = "yolo12x.pt"

if __name__ == '__main__':
    print(f"\n [ {model_name} ] 단독 학습, CSV 추가 스크립트 시작\n")

    try:
        model = YOLO(model_name)
        
        start_time = time.time()
        results = model.train(data=dataset_path, epochs=300, imgsz=640, batch=8) 
        end_time = time.time()
        train_time_sec = end_time - start_time
        
        train_hours = int(train_time_sec // 3600)
        train_mins = int((train_time_sec % 3600) // 60)
        train_time_str = f"{train_hours}h {train_mins}m"
        
        metrics = model.trainer.validator.metrics
        inference_time_ms = round(metrics.speed.get('inference', 0.0), 3)
        mean_p, mean_r, mean_map50, mean_map = metrics.mean_results()
        
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
        
        new_df = pd.DataFrame([new_row])
        
        if os.path.exists(summary_csv_path):
            new_df.to_csv(summary_csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"\n 기존 요약 파일({os.path.basename(summary_csv_path)})에 {model_name} 결과 추가 완료")
        else:
            new_df.to_csv(summary_csv_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            print(f"\n 요약 파일이 없어 새로 생성하여 저장 완료")
            
        print(f" 소요 시간: {train_time_str}, 추론 속도: {inference_time_ms}ms/img")
        print("-" * 70)

    except Exception as e:
        print(f"\n [{model_name}] 학습 중 에러 발생!")
        print(traceback.format_exc())
