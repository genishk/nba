# predict_today.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # 프로젝트 루트 추가

from src.data.espn_api import ESPNNBADataCollector
from src.data.processor import DataProcessor
from src.models.trainer import BettingModelTrainer
import joblib
import pandas as pd

def predict_today_games():
    # 1. 데이터 수집
    collector = ESPNNBADataCollector()
    processor = DataProcessor()
    
    # 1-1. 저장된 과거 데이터 로드
    historical_data = processor.load_latest_data('historical')
    
    # 1-2. 예정된 경기 데이터 수집
    upcoming_data = collector.collect_upcoming_data(days_ahead=1)
    
    # 2. 데이터 전처리
    # 2-1. 과거 데이터 처리
    historical_df = processor.process_game_data(historical_data)
    historical_features = processor.extract_features(historical_df, historical_data)
    
    # 2-2. 예정된 경기 데이터 처리
    upcoming_df = processor.process_game_data(upcoming_data)
    upcoming_features = processor.extract_features(upcoming_df, upcoming_data)
    
    # 3. 필요한 특성들 분리
    historical_cols = [
        # 기본 경기력 지표
        'home_rebounds', 'away_rebounds',
        'home_assists', 'away_assists',
        'home_fieldGoalsAttempted', 'away_fieldGoalsAttempted',
        'home_fieldGoalsMade', 'away_fieldGoalsMade',
        'home_fieldGoalPct', 'away_fieldGoalPct',
        'home_freeThrowsAttempted', 'away_freeThrowsAttempted',
        'home_freeThrowsMade', 'away_freeThrowsMade',
        'home_freeThrowPct', 'away_freeThrowPct',
        'home_threePointFieldGoalsAttempted', 'away_threePointFieldGoalsAttempted',
        'home_threePointFieldGoalsMade', 'away_threePointFieldGoalsMade',
        'home_threePointPct', 'away_threePointPct',
        
        # 리더 통계
        'home_leader_points', 'away_leader_points',
        'home_leader_rebounds', 'away_leader_rebounds',
        'home_leader_assists', 'away_leader_assists',
        
        # 팀 기록
        'home_vs_away_win_rate',
        
        # 최근 트렌드
        'home_recent_win_rate', 'away_recent_win_rate',
        'home_recent_avg_score', 'away_recent_avg_score',
        'home_recent_home_win_rate', 'away_recent_home_win_rate',
        'home_recent_away_win_rate', 'away_recent_away_win_rate'
    ]
    
    upcoming_cols = [
        'home_overall_record_win_rate', 'away_overall_record_win_rate',
        'home_home_record_win_rate', 'away_home_record_win_rate',
        'home_road_record_win_rate', 'away_road_record_win_rate',
        'home_rest_days', 'away_rest_days', 
        'home_team_id', 'away_team_id'
    ]
    
    # 4. 예측을 위한 특성 데이터 준비
    prediction_features = pd.DataFrame()
    
    # 4-1. 과거 데이터 기반 특성 추가 (팀별 최근 평균)
    for team_id in upcoming_features['home_team_id'].unique():
        team_history = historical_features[
            (historical_features['home_team_id'] == team_id) |
            (historical_features['away_team_id'] == team_id)
        ].tail(10)  # 최근 10경기
        # 팀별 평균 통계 계산
        team_stats = team_history[historical_cols].mean()
        # 해당 팀이 홈/어웨이인 예정 경기에 통계 적용
        prediction_features.update(team_stats)
    
    # 4-2. 예정된 경기 데이터의 특성 추가
    for col in upcoming_cols:
        prediction_features[col] = upcoming_features[col]
    
    # 5. 모델 trainer 인스턴스 생성 및 특성 가공
    trainer = BettingModelTrainer()
    X, _ = trainer.prepare_features(prediction_features.to_dict('records'))
    
    # 6. 저장된 모델 로드 및 예측
    model_files = list(Path(__file__).parent / "saved_models".glob("betting_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError("저장된 모델을 찾을 수 없습니다.")
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    model = joblib.load(latest_model)
    
    predictions = model.predict_proba(X)
    
    # 7. 결과 출력
    for idx, row in upcoming_features.iterrows():
        home_win_prob = predictions[idx][1]
        print(f"\n{row['home_team_name']} vs {row['away_team_name']}")
        print(f"홈팀 승리 확률: {home_win_prob:.1%}")
        print(f"원정팀 승리 확률: {(1-home_win_prob):.1%}")

if __name__ == "__main__":
    predict_today_games()