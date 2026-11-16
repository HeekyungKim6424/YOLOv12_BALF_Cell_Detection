import pandas as pd
from ultralytics import YOLO
from pathlib import Path

datasets = [
    r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\",

]

if __name__ == '__main__':
    for data_path in datasets:
        p = Path(data_path)
        folder_name = p.parent.name
        file_name = p.stem
        
        print(f"\n========== [ {folder_name} ] 학습 시작==========\n")

        model = YOLO("yolo12l.pt") 
        results = model.train(data=data_path, epochs=300, imgsz=640)
        metrics = model.trainer.validator.metrics
        
        data_rows = []

        mean_p, mean_r, mean_map50, mean_map = metrics.mean_results()
        data_rows.append({
            "Class": "all",
            "Instances": int(sum(metrics.nt_per_class)), 
            "Box(P)": round(mean_p, 3),
            "R": round(mean_r, 3),
            "mAP50": round(mean_map50, 3),
            "mAP50-95": round(mean_map, 3)
        })

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

        df = pd.DataFrame(data_rows)
        csv_filename = f"{folder_name}_{file_name}.csv"
        target_dir = r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell\"
        save_path = Path(target_dir)
        save_path.mkdir(parents=True, exist_ok=True) 
        final_file_path = save_path / csv_filename
        
        df.to_csv(final_file_path, index=False, encoding='utf-8-sig')
        
        print(f"\n [{folder_name}] 저장 완료: {csv_filename}")
        print("-" * 70)
