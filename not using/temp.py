import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # 프로젝트 루트 추가

from src.data.espn_api import ESPNNBADataCollector
from src.data.processor_upcoming import DataProcessor_upcoming
import pandas as pd

# 데이터 처리기 초기화
processor = DataProcessor_upcoming()

# 1-2. 예정된 경기 데이터 수집
# upcoming_data = collector.collect_upcoming_data(days_ahead=1)
# 업커밍 데이터 로드
upcoming_data = processor.load_latest_data('upcoming')

# STATUS_SCHEDULED인 경기만 필터링
upcoming_data['games'] = [game for game in upcoming_data['games'] 
                         if game['status'] == 'STATUS_SCHEDULED']

# 기본 정보만 추출
upcoming_df = processor.process_game_data(upcoming_data)
upcoming_features = processor.extract_features(upcoming_df, upcoming_data)

# 필요한 기본 피처들만 선택
selected_features_upcoming = [
    'game_id', 'date', 'season_year', 'season_type', 'status',
    'home_team_id', 'home_team_name', 'home_team_score',
    'away_team_id', 'away_team_name', 'away_team_score'
]
upcoming_features = upcoming_features[selected_features_upcoming]

# 결과 확인
print("\n=== 업커밍 경기 기본 정보 ===")
print(upcoming_features)

# 필요하다면 저장
processor.save_processed_data(upcoming_features, "upcoming_base_features.json")