import pandas as pd
from ultralytics import YOLO
from pathlib import Path

datasets = [
    r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\final_data\data.yaml",

]

if __name__ == '__main__':
    for data_path in datasets:
        p = Path(data_path)
        folder_name = p.parent.name
        file_name = p.stem
        
        print(f"\n========== 🚀 [ {folder_name} ] 학습 시작! ==========\n")

        model = YOLO("yolo12l.pt") 
        
        # 1. 학습 실행 (이후에 model.val() 따로 부르는 거 절대 없음!)
        results = model.train(data=data_path, epochs=300, imgsz=640)

        # 2. 학습 직후 메모리에 떠 있는 DetMetrics 객체에서 터미널 값 바로 추출!
        # 네가 올려준 에러 로그의 구조를 완벽하게 반영함
        metrics = model.trainer.validator.metrics
        
        data_rows = []

        # 전체 평균 ('all') 데이터 추출
        mean_p, mean_r, mean_map50, mean_map = metrics.mean_results()
        data_rows.append({
            "Class": "all",
            "Instances": int(sum(metrics.nt_per_class)), # 타겟 총합
            "Box(P)": round(mean_p, 3),
            "R": round(mean_r, 3),
            "mAP50": round(mean_map50, 3),
            "mAP50-95": round(mean_map, 3)
        })

        # 개별 클래스 데이터 추출
        # 에러 로그 기준 ap_class_index 사용
        indices = metrics.ap_class_index() if callable(getattr(metrics, 'ap_class_index', None)) else metrics.ap_class_index

        for i, c in enumerate(indices):
            name = metrics.names[c]
            p_val, r_val, ap50, ap = metrics.class_result(i)
            
            data_rows.append({
                "Class": name,
                "Instances": int(metrics.nt_per_class[i]),
                "Box(P)": round(p_val, 3),
                "R": round(r_val, 3),
                "mAP50": round(ap50, 3),
                "mAP50-95": round(ap, 3)
            })

        # 3. CSV 파일로 즉시 저장
        df = pd.DataFrame(data_rows)
        csv_filename = f"{folder_name}_{file_name}.csv"
        target_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\new_things"
        
        # 경로 객체로 만들고, 폴더가 없으면 알아서 생성하게 만듦
        save_path = Path(target_dir)
        save_path.mkdir(parents=True, exist_ok=True) 
        
        # 폴더 경로 + 파일 이름 합치기
        final_file_path = save_path / csv_filename
        
        # 지정된 경로에 최종 저장!
        df.to_csv(final_file_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ [{folder_name}] 추가 검증 없이 터미널 값 그대로 저장 완료: {csv_filename}")
        print("-" * 70)