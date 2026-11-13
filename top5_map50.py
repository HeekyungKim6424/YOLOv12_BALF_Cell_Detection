import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\ACE_LAB_AI\Desktop\object_detection_cell")
COMBINATION_FOLDERS = [BASE_DIR / f"yolov12l_combination{i}" for i in range(1, 8)]

if __name__ == '__main__':
    records = []

    for folder in COMBINATION_FOLDERS:
        if not folder.is_dir():
            print(f"[경고] 폴더 없음: {folder}")
            continue

        for csv_path in folder.glob("*.csv"):
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            all_row = df[df["Class"] == "all"]

            if all_row.empty:
                print(f"[경고] 'all' 행 없음: {csv_path}")
                continue

            records.append({
                "combination": folder.name,
                "file": csv_path.name,
                "mAP50": all_row.iloc[0]["mAP50"],
            })

    result_df = pd.DataFrame(records).sort_values("mAP50", ascending=False)
    top5 = result_df.head(5).reset_index(drop=True)

    print("\n[결과] mAP50 (class=all) 상위 5개\n")
    print(top5.to_string(index=False))
