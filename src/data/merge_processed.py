import json
from pathlib import Path
from datetime import datetime

def merge_processed_files():
    # 데이터 디렉토리 설정
    data_dir = Path(__file__).parent
    
    # 특정 파일들만 읽기
    files_to_merge = [
        "processed_20250303_161150.json",
        "processed_20250303_161651.json",
        "processed_20250303_161847.json"
    ]
    
    # 모든 데이터를 저장할 리스트
    all_data = []
    
    # 각 파일 읽기
    for filename in files_to_merge:
        file_path = data_dir / filename
        print(f"Reading file: {filename}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = data_dir / f"merged_processed_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nMerged data saved to: {output_file}")
    print(f"Total entries: {len(all_data)}")

if __name__ == "__main__":
    merge_processed_files() 