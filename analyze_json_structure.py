# analyze_json_structure.py
import json
from datetime import datetime

def analyze_structure(input_file='nba_data_one_day.json'):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 분석 결과를 저장할 딕셔너리
    analysis = {
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_sections": {
            "1_first_game": data['games'][0],
            "2_first_team_stats": data['team_stats'][list(data['team_stats'].keys())[0]],
            "3_first_team_roster": data['rosters'][list(data['rosters'].keys())[0]],
            "4_first_game_details": data['game_details'][0]
        }
    }
    
    # 결과를 JSON 파일로 저장
    output_file = 'nba_data_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"분석 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    analyze_structure()