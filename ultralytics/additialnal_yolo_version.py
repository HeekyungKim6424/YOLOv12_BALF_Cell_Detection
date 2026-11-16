import time
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import traceback
import os

dataset_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\"
target_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\"
save_path = Path(target_dir)
save_path.mkdir(parents=True, exist_ok=True)

models_to_train = ['yolov9t.pt', 'yolov9e.pt', 'yolov9c.pt']
p = Path(dataset_path)
dataset_name = p.parent.name
summary_filename = f"{dataset_name}_ALL_MODELS_SUMMARY.csv"
summary_csv_path = save_path / summary_filename
new_summary_rows = []

if __name__ == '__main__':
    print(f"\n 누락된 {len(models_to_train)}개 모델 추가 학습 시작\n")

    for model_name in models_to_train:
        print(f"\n========== [ {model_name} ] 학습 준비 ==========\n")
        
        try:
            model = YOLO(model_name)
            start_time = time.time()
            results = model.train(data=dataset_path, epochs=300, imgsz=640)
            
            end_time = time.time()
            train_time_sec = end_time - start_time
            
            train_hours = int(train_time_sec // 3600)
            train_mins = int((train_time_sec % 3600) // 60)
            train_time_str = f"{train_hours}h {train_mins}m"
            
            metrics = model.trainer.validator.metrics
            inference_time_ms = round(metrics.speed.get('inference', 0.0), 3)
            data_rows = []

            mean_p, mean_r, mean_map50, mean_map = metrics.mean_results()
            all_class_data = {
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
            data_rows.append(all_class_data)
            new_summary_rows.append(all_class_data)
            indices = metrics.ap_class_index() if callable(getattr(metrics, 'ap_class_index', None)) else metrics.ap_class_index

            for i, c in enumerate(indices):
                name = metrics.names[c]
                p_val, r_val, ap50, ap = metrics.class_result(i)
                
                data_rows.append({
                    "Model": model_name,
                    "Class": name,
                    "Instances": int(metrics.nt_per_class[i]),
                    "Box(P)": round(p_val, 3),
                    "R": round(r_val, 3),
                    "mAP50": round(ap50, 3),
                    "mAP50-95": round(ap, 3),
                    "Train_Time(sec)": round(train_time_sec, 2),
                    "Train_Time_str": train_time_str,
                    "Inference_Time(ms/img)": inference_time_ms
                })

            df = pd.DataFrame(data_rows)
            csv_filename = f"{dataset_name}_{model_name.replace('.pt', '')}.csv"
            df.to_csv(save_path / csv_filename, index=False, encoding='utf-8-sig')
            
            print(f"\n [{model_name}] 완료! (학습: {train_time_str}, 추론: {inference_time_ms}ms/img)")
            print(f" 개별 저장 완료: {csv_filename}")
            print("-" * 70)

        except Exception as e:
            print(f"\n [{model_name}] 학습 중 에러 발생. 다음 모델로 넘어갑니다.")
            print(traceback.format_exc())
            print("-" * 70)
            continue

    # --- [4. 기존 Summary CSV에 병합(Append) 작업] ---
    if new_summary_rows:
        new_df = pd.DataFrame(new_summary_rows)
        
        if summary_csv_path.exists():
            existing_df = pd.read_csv(summary_csv_path)
            existing_df = existing_df[~existing_df['Model'].isin(models_to_train)]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.sort_values(by='Model').reset_index(drop=True)
        else:
            combined_df = new_df
            
        combined_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"성공적으로 기존 파일에 병합되었습니다")
    else:
        print("\n 추가된 모델 결과가 없어 병합을 진행하지 않았습니다.")
