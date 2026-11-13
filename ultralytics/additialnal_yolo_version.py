import time
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import traceback
import os

# 1. 경로 설정 (기존과 동일)
dataset_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\AI-Cell-Image-154\data.yaml"
target_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\model_comparison_results"
save_path = Path(target_dir)
save_path.mkdir(parents=True, exist_ok=True)

# 2. 누락된 모델만 리스트업
models_to_train = ['yolov9t.pt', 'yolov9e.pt', 'yolov9c.pt']

# 3. 데이터셋 이름 및 기존 Summary 파일 경로 정의
p = Path(dataset_path)
dataset_name = p.parent.name
summary_filename = f"{dataset_name}_ALL_MODELS_SUMMARY.csv"
summary_csv_path = save_path / summary_filename

# 추가된 결과를 담을 리스트
new_summary_rows = []

if __name__ == '__main__':
    print(f"\n🚀 [ 복구 모드 ] 누락된 {len(models_to_train)}개 모델 추가 학습 시작!\n")

    for model_name in models_to_train:
        print(f"\n========== 🔥 [ {model_name} ] 학습 준비 ==========\n")
        
        try:
            model = YOLO(model_name)
            
            # --- [1. 학습 시간 측정 시작] ---
            start_time = time.time()
            
            results = model.train(data=dataset_path, epochs=300, imgsz=640)
            
            # --- [2. 학습 시간 측정 종료] ---
            end_time = time.time()
            train_time_sec = end_time - start_time
            
            train_hours = int(train_time_sec // 3600)
            train_mins = int((train_time_sec % 3600) // 60)
            train_time_str = f"{train_hours}h {train_mins}m"
            
            # --- [3. 모델 속도 추출] ---
            metrics = model.trainer.validator.metrics
            inference_time_ms = round(metrics.speed.get('inference', 0.0), 3)
            
            data_rows = []

            # 전체 평균 ('all') 데이터 추출
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
            
            # 요약본 리스트에 추가
            new_summary_rows.append(all_class_data)

            # 개별 클래스 데이터 추출
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

            # 개별 모델 결과 CSV 덮어쓰기 저장
            df = pd.DataFrame(data_rows)
            csv_filename = f"{dataset_name}_{model_name.replace('.pt', '')}.csv"
            df.to_csv(save_path / csv_filename, index=False, encoding='utf-8-sig')
            
            print(f"\n✅ [{model_name}] 완료! (학습: {train_time_str}, 추론: {inference_time_ms}ms/img)")
            print(f"📄 개별 저장 완료: {csv_filename}")
            print("-" * 70)

        except Exception as e:
            print(f"\n❌ [{model_name}] 학습 중 에러 발생! 다음 모델로 넘어갑니다.")
            print(traceback.format_exc())
            print("-" * 70)
            continue

    # --- [4. 기존 Summary CSV에 병합(Append) 작업] ---
    if new_summary_rows:
        print("\n🔄 기존 요약 파일(Summary CSV) 업데이트 중...")
        new_df = pd.DataFrame(new_summary_rows)
        
        if summary_csv_path.exists():
            existing_df = pd.read_csv(summary_csv_path)
            
            # [중복 방지 로직] 기존 파일에 혹시라도 지금 돌린 모델 이름이 있으면 삭제하고 새로 넣음
            existing_df = existing_df[~existing_df['Model'].isin(models_to_train)]
            
            # 기존 데이터와 새 데이터 병합
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # 보기 좋게 모델 이름 순으로 정렬 (선택 사항)
            combined_df = combined_df.sort_values(by='Model').reset_index(drop=True)
        else:
            # 만약 기존 파일이 어떤 이유로 지워졌다면 새로 만든 것만 저장
            combined_df = new_df
            
        combined_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"🎉 성공적으로 기존 파일에 병합되었습니다! 확인해 보세요: {summary_filename}")
    else:
        print("\n⚠️ 추가된 모델 결과가 없어 병합을 진행하지 않았습니다.")