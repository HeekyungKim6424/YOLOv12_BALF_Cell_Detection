import time
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import traceback

dataset_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\"

versions = ['yolov8', 'yolov9', 'yolov10', 'yolo11', 'yolo12', 'yolo26']
sizes = ['n', 's', 'm', 'l', 'x']
models_to_train = [f"{v}{s}.pt" for v in versions for s in sizes]

target_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\model_comparison_results"
save_path = Path(target_dir)
save_path.mkdir(parents=True, exist_ok=True)

summary_rows = []

if __name__ == '__main__':
    p = Path(dataset_path)
    dataset_name = p.parent.name
    
    print(f"\[데이터셋: {dataset_name} ] 모델 학습 테스트 시작\n")

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
            summary_rows.append(all_class_data)

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
            
            print(f"\n[{model_name}] 완료 (학습: {train_time_str}, 추론: {inference_time_ms}ms/img)")
            print(f" 개별 저장 완료: {csv_filename}")
            print("-" * 70)

        except Exception as e:
            print(f"\n [{model_name}] 학습 중 에러 발생. 다음 모델로 넘어갑니다.")
            print(traceback.format_exc())
            print("-" * 70)
            continue

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_filename = f"{dataset_name}_ALL_MODELS_SUMMARY.csv"
        df_summary.to_csv(save_path / summary_filename, index=False, encoding='utf-8-sig')
        print(f"\n 모든 학습이 종료되었습니다")
