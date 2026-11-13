import time
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import traceback

# 1. 고정할 단일 데이터셋 경로 설정 (원하시는 경로로 수정하세요)
dataset_path = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\AI-Cell-Image-154\data.yaml"

# 2. 돌릴 모델 30개 리스트 자동 생성
# 참고: Ultralytics 공식 명칭에서 v11은 'yolo11n.pt' 식별자를 주로 씁니다. 
# 만약 에러가 난다면 'yolov11' 대신 'yolo11'로 수정해 주세요.
versions = ['yolov8', 'yolov9', 'yolov10', 'yolo11', 'yolo12', 'yolo26']
sizes = ['n', 's', 'm', 'l', 'x']
models_to_train = [f"{v}{s}.pt" for v in versions for s in sizes]

target_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\model_comparison_results"
save_path = Path(target_dir)
save_path.mkdir(parents=True, exist_ok=True)

# 30개 모델의 전체 결과를 한눈에 비교할 수 있는 요약 리스트
summary_rows = []

if __name__ == '__main__':
    p = Path(dataset_path)
    dataset_name = p.parent.name
    
    print(f"\n🚀 [ 데이터셋: {dataset_name} ] 총 30개 모델 학습 테스트 시작!\n")

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
            
            # 보기 편하게 시간 포맷팅 (예: 2h 15m)
            train_hours = int(train_time_sec // 3600)
            train_mins = int((train_time_sec % 3600) // 60)
            train_time_str = f"{train_hours}h {train_mins}m"
            
            # --- [3. 모델 속도(Inference Time) 추출] ---
            metrics = model.trainer.validator.metrics
            # speed 딕셔너리 안에는 preprocess, inference, loss, postprocess 시간이 담겨 있습니다. (ms/img)
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
            
            # 전체 비교 요약본(Summary)에 'all' 데이터 추가
            summary_rows.append(all_class_data)

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

            # 개별 모델 결과 CSV로 저장 (예: datasetName_yolov8n.csv)
            df = pd.DataFrame(data_rows)
            csv_filename = f"{dataset_name}_{model_name.replace('.pt', '')}.csv"
            df.to_csv(save_path / csv_filename, index=False, encoding='utf-8-sig')
            
            print(f"\n✅ [{model_name}] 완료! (학습: {train_time_str}, 추론: {inference_time_ms}ms/img)")
            print(f"📄 개별 저장 완료: {csv_filename}")
            print("-" * 70)

        except Exception as e:
            # 중간에 모델이 다운로드 안 되거나 메모리 부족(OOM)이 발생해도 
            # 프로그램이 뻗지 않고 다음 모델로 넘어가도록 예외 처리
            print(f"\n❌ [{model_name}] 학습 중 에러 발생! 다음 모델로 넘어갑니다.")
            print(traceback.format_exc())
            print("-" * 70)
            continue

    # 모든 모델 학습이 끝난 후, 모델끼리 성능과 시간을 쉽게 비교할 수 있는 전체 요약본 저장
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_filename = f"{dataset_name}_ALL_MODELS_SUMMARY.csv"
        df_summary.to_csv(save_path / summary_filename, index=False, encoding='utf-8-sig')
        print(f"\n🎉 모든 학습이 종료되었습니다! 30개 모델 전체 요약 파일: {summary_filename}")