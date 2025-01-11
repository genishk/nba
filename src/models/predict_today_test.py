# predict_today.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # 프로젝트 루트 추가

from src.data.espn_api import ESPNNBADataCollector
from src.data.processor import DataProcessor
from src.models.trainer import BettingModelTrainer
import joblib
import pandas as pd
from typing import Tuple

collector = ESPNNBADataCollector()
processor = DataProcessor()



# 1-1. 저장된 과거 데이터 로드
historical_data = processor.load_latest_data('historical')
# STATUS_FINAL인 경기만 필터링
historical_data['games'] = [game for game in historical_data['games'] 
                          if game['status'] == 'STATUS_FINAL']

# 1-2. 예정된 경기 데이터 수집
upcoming_data = collector.collect_upcoming_data(days_ahead=1)
# upcoming_data = processor.load_latest_data('upcoming')
# STATUS_SCHEDULED인 경기만 필터링
upcoming_data['games'] = [game for game in upcoming_data['games'] 
                         if game['status'] == 'STATUS_SCHEDULED']







# 1-3. 휴식일 계산을 위한 전체 데이터 생성
total_data = {
    'collection_date': upcoming_data['collection_date'],
    'games': historical_data['games'] + upcoming_data['games'],
    'team_stats': {**historical_data['team_stats'], **upcoming_data['team_stats']}
}

total_df = processor.process_game_data(total_data)
total_features = processor.extract_features(total_df, total_data)
total_features = total_features.drop_duplicates(subset=['game_id'])
selected_features_total = [
    'game_id', 'home_rest_days', 'away_rest_days'
]
total_features = total_features[selected_features_total]

# processor.save_processed_data(total_features, "total_features.json")






# 2. 데이터 전처리
# 2-1. 과거 데이터 처리
historical_df = processor.process_game_data(historical_data)
historical_features = processor.extract_features(historical_df, historical_data)
selected_features_historical = [
    'game_id', 'date','home_team_id', 'home_team_name','away_team_id','away_team_name','home_rebounds', 'home_assists', 'home_fieldGoalsAttempted',
    'home_fieldGoalsMade', 'home_fieldGoalPct', 'home_freeThrowPct',
    'home_freeThrowsAttempted', 'home_freeThrowsMade', 'home_threePointPct',
    'home_threePointFieldGoalsAttempted', 'home_threePointFieldGoalsMade',
    'away_rebounds', 'away_assists', 'away_fieldGoalsAttempted',
    'away_fieldGoalsMade', 'away_fieldGoalPct', 'away_freeThrowPct',
    'away_freeThrowsAttempted', 'away_freeThrowsMade', 'away_threePointPct',
    'away_threePointFieldGoalsAttempted', 'away_threePointFieldGoalsMade',
    'home_leader_points', 'home_leader_rebounds', 'home_leader_assists',
    'away_leader_points', 'away_leader_rebounds', 'away_leader_assists',
    'home_vs_away_wins', 'home_vs_away_losses', 'home_vs_away_win_rate',
    'home_recent_win_rate', 'home_recent_avg_score', 'home_recent_home_win_rate',
    'home_recent_away_win_rate', 'away_recent_win_rate', 'away_recent_avg_score',
    'away_recent_home_win_rate', 'away_recent_away_win_rate'
]
historical_features = historical_features[selected_features_historical]

# processor.save_processed_data(historical_features, "historical_features.json")










def get_latest_matchups(historical_features: pd.DataFrame) -> pd.DataFrame:
    """
    각 팀 매치업의 가장 최근 경기 데이터만 추출
    
    Args:
        historical_features (pd.DataFrame): 전체 경기 데이터
        
    Returns:
        pd.DataFrame: 각 매치업별 최근 경기 데이터
    """
    # 1. 날짜 형식 통일 (timezone 관련 이슈 방지)
    df = historical_features.copy()
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    # 2. home_team_id와 away_team_id로 그룹화하고 날짜순 정렬 후 최신 데이터만 선택
    latest_matchups = (df
        .sort_values('date', ascending=False)
        .groupby(['home_team_id', 'away_team_id'])
        .first()
        .reset_index())
    
    return latest_matchups

# 기존 historical_features는 그대로 두고 새로운 변수에 저장
latest_team_matchups = get_latest_matchups(historical_features)

selected_features_matchups = [
    'game_id', 'date','home_team_id', 'home_team_name','away_team_id','away_team_name',
    'home_vs_away_wins', 'home_vs_away_losses', 'home_vs_away_win_rate'
]
latest_team_matchups = latest_team_matchups[selected_features_matchups]

# 저장 (필요한 경우)
# processor.save_processed_data(latest_team_matchups, "latest_team_matchups.json")


def update_matchup_stats(latest_team_matchups: pd.DataFrame) -> pd.DataFrame:
    """
    같은 팀들 간의 매치업에 대해 최신 상대전적 정보로 업데이트
    
    Args:
        latest_team_matchups (pd.DataFrame): 팀별 최근 매치업 데이터
        
    Returns:
        pd.DataFrame: 상대전적이 업데이트된 매치업 데이터
    """
    # 데이터 복사
    df = latest_team_matchups.copy()
    
    # 각 매치업에 대해
    for idx, row in df.iterrows():
        # 같은 팀들의 반대 매치업 찾기 (홈/어웨이 바뀐 경기)
        reverse_match = df[
            (df['home_team_id'] == row['away_team_id']) & 
            (df['away_team_id'] == row['home_team_id'])
        ]
        
        if len(reverse_match) > 0:
            # 두 경기 중 더 최신 경기 찾기
            current_date = pd.to_datetime(row['date'])
            reverse_date = pd.to_datetime(reverse_match.iloc[0]['date'])
            
            if reverse_date > current_date:
                # 최신 경기가 반대 매치업인 경우, 현재 행의 상대전적 정보 업데이트
                reverse_row = reverse_match.iloc[0]
                df.loc[idx, 'home_vs_away_wins'] = reverse_row['home_vs_away_losses']
                df.loc[idx, 'home_vs_away_losses'] = reverse_row['home_vs_away_wins']
                df.loc[idx, 'home_vs_away_win_rate'] = (
                    reverse_row['home_vs_away_losses'] / 
                    (reverse_row['home_vs_away_losses'] + reverse_row['home_vs_away_wins'])
                    if (reverse_row['home_vs_away_losses'] + reverse_row['home_vs_away_wins']) > 0
                    else 0.0
                )
    
    return df

# 상대전적 정보 업데이트
final_team_matchups = update_matchup_stats(latest_team_matchups)

# # 저장
# processor.save_processed_data(final_team_matchups, "final_team_matchups.json")

def create_mirror_matchups(team_matchups: pd.DataFrame) -> pd.DataFrame:
    """
    각 매치업의 미러 레코드를 생성
    
    Args:
        team_matchups (pd.DataFrame): 원본 매치업 데이터
        
    Returns:
        pd.DataFrame: 미러 매치업이 추가된 데이터
    """
    # 원본 데이터 복사
    mirror_matchups = []
    
    # 각 매치업에 대해
    for _, row in team_matchups.iterrows():
        # 반대 매치업이 있는지 확인
        reverse_exists = any(
            (team_matchups['home_team_id'] == row['away_team_id']) & 
            (team_matchups['away_team_id'] == row['home_team_id'])
        )
        
        # 반대 매치업이 없는 경우에만 미러 레코드 생성
        if not reverse_exists:
            mirror_matchup = {
                'game_id': f"mirror_{row['game_id']}",
                'date': row['date'],
                'home_team_id': row['away_team_id'],
                'home_team_name': row['away_team_name'],
                'away_team_id': row['home_team_id'],
                'away_team_name': row['home_team_name'],
                'home_vs_away_wins': row['home_vs_away_losses'],
                'home_vs_away_losses': row['home_vs_away_wins'],
                'home_vs_away_win_rate': (
                    row['home_vs_away_losses'] / 
                    (row['home_vs_away_losses'] + row['home_vs_away_wins'])
                    if (row['home_vs_away_losses'] + row['home_vs_away_wins']) > 0
                    else 0.0
                )
            }
            mirror_matchups.append(mirror_matchup)
    
    # 원본 데이터와 미러 매치업 합치기
    if mirror_matchups:
        mirror_df = pd.DataFrame(mirror_matchups)
        final_df = pd.concat([team_matchups, mirror_df], ignore_index=True)
        return final_df
    
    return team_matchups

# 상대전적 정보 업데이트 후 미러 매치업 생성
complete_team_matchups = create_mirror_matchups(final_team_matchups)

mirror_features = [
    'home_team_id', 'home_team_name','away_team_id','away_team_name',
    'home_vs_away_wins', 'home_vs_away_losses', 'home_vs_away_win_rate'
]
complete_team_matchups = complete_team_matchups[mirror_features]


# # 저장 (필요한 경우)
# processor.save_processed_data(complete_team_matchups, "complete_team_matchups.json")













# 1. home_team_id별 가장 최근 레코드
historical_home = (historical_features
    .drop_duplicates(subset=['home_team_id', 'date'])
    .sort_values(['home_team_id', 'date'], ascending=[True, False])
    .groupby('home_team_id')
    .first()
    .reset_index())
selected_features_home = [
    'game_id', 'date','home_team_id', 'home_team_name','home_rebounds', 'home_assists', 'home_fieldGoalsAttempted',
    'home_fieldGoalsMade', 'home_fieldGoalPct', 'home_freeThrowPct',
    'home_freeThrowsAttempted', 'home_freeThrowsMade', 'home_threePointPct',
    'home_threePointFieldGoalsAttempted', 'home_threePointFieldGoalsMade',
    'home_leader_points', 'home_leader_rebounds', 'home_leader_assists',
    'home_recent_win_rate', 'home_recent_avg_score', 'home_recent_home_win_rate',
    'home_recent_away_win_rate'
]
historical_home = historical_home[selected_features_home]


# 1. home_team_id별 가장 최근 레코드
historical_away = (historical_features
    .drop_duplicates(subset=['away_team_id', 'date'])
    .sort_values(['away_team_id', 'date'], ascending=[True, False])
    .groupby('away_team_id')
    .first()
    .reset_index())
selected_features_away = [
    'game_id', 'date','away_team_id', 'away_team_name','away_rebounds', 'away_assists', 'away_fieldGoalsAttempted',
    'away_fieldGoalsMade', 'away_fieldGoalPct', 'away_freeThrowPct',
    'away_freeThrowsAttempted', 'away_freeThrowsMade', 'away_threePointPct',
    'away_threePointFieldGoalsAttempted', 'away_threePointFieldGoalsMade',
    'away_leader_points', 'away_leader_rebounds', 'away_leader_assists',
    'away_recent_win_rate', 'away_recent_avg_score', 'away_recent_home_win_rate',
    'away_recent_away_win_rate'
]
historical_away = historical_away[selected_features_away]


def update_both_records_with_latest_stats(home_records: pd.DataFrame, away_records: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 결과를 저장할 DataFrame 생성
    home_result = home_records.copy()
    away_result = away_records.copy()
    
    # 모든 팀 ID 수집
    all_team_ids = set(home_records['home_team_id'].unique()) | set(away_records['away_team_id'].unique())
    
    # 각 팀별로 처리
    for team_id in all_team_ids:
        # 홈/어웨이 각각의 최신 데이터 찾기
        home_data = home_records[home_records['home_team_id'] == team_id]
        away_data = away_records[away_records['away_team_id'] == team_id]
        
        if not home_data.empty:
            home_latest = home_data.sort_values('date').iloc[-1]
            # timezone 제거
            home_date = pd.to_datetime(home_latest['date']).tz_localize(None)
        else:
            home_latest = None
            home_date = pd.Timestamp.min
            
        if not away_data.empty:
            away_latest = away_data.sort_values('date').iloc[-1]
            # timezone 제거
            away_date = pd.to_datetime(away_latest['date']).tz_localize(None)
        else:
            away_latest = None
            away_date = pd.Timestamp.min
        
        # 더 최신 데이터로 업데이트
        if away_date > home_date:
            latest = away_latest
            is_away = True
        else:
            latest = home_latest
            is_away = False
            
        # 통계 컬럼 식별 (win rate 관련 컬럼 제외)
        win_rate_columns = ['recent_home_win_rate', 'recent_away_win_rate', 'recent_win_rate', 'recent_avg_score']
        home_stat_columns = [col for col in home_result.columns 
                           if col not in ['game_id', 'date', 'home_team_id', 'home_team_name'] 
                           and not any(rate in col for rate in win_rate_columns)]
        away_stat_columns = [col for col in away_result.columns 
                           if col not in ['game_id', 'date', 'away_team_id', 'away_team_name']
                           and not any(rate in col for rate in win_rate_columns)]
        
        if latest is not None:
            if is_away:
                # away 데이터가 최신일 때
                # 일반 통계 업데이트
                for home_col in home_stat_columns:
                    away_col = home_col.replace('home_', 'away_')
                    home_result.loc[home_result['home_team_id'] == team_id, home_col] = latest[away_col]
                    away_result.loc[away_result['away_team_id'] == team_id, away_col] = latest[away_col]
                
                # win rate 관련 통계 업데이트
                home_result.loc[home_result['home_team_id'] == team_id, 'home_recent_win_rate'] = latest['away_recent_win_rate']
                home_result.loc[home_result['home_team_id'] == team_id, 'home_recent_avg_score'] = latest['away_recent_avg_score']
                home_result.loc[home_result['home_team_id'] == team_id, 'home_recent_home_win_rate'] = latest['away_recent_home_win_rate']
                home_result.loc[home_result['home_team_id'] == team_id, 'home_recent_away_win_rate'] = latest['away_recent_away_win_rate']
                
                away_result.loc[away_result['away_team_id'] == team_id, 'away_recent_win_rate'] = latest['away_recent_win_rate']
                away_result.loc[away_result['away_team_id'] == team_id, 'away_recent_avg_score'] = latest['away_recent_avg_score']
                away_result.loc[away_result['away_team_id'] == team_id, 'away_recent_home_win_rate'] = latest['away_recent_home_win_rate']
                away_result.loc[away_result['away_team_id'] == team_id, 'away_recent_away_win_rate'] = latest['away_recent_away_win_rate']
            else:
                # home 데이터가 최신일 때
                # 일반 통계 업데이트
                for away_col in away_stat_columns:
                    home_col = away_col.replace('away_', 'home_')
                    home_result.loc[home_result['home_team_id'] == team_id, home_col] = latest[home_col]
                    away_result.loc[away_result['away_team_id'] == team_id, away_col] = latest[home_col]
                
                # win rate 관련 통계 업데이트
                home_result.loc[home_result['home_team_id'] == team_id, 'home_recent_win_rate'] = latest['home_recent_win_rate']
                home_result.loc[home_result['home_team_id'] == team_id, 'home_recent_avg_score'] = latest['home_recent_avg_score']
                home_result.loc[home_result['home_team_id'] == team_id, 'home_recent_home_win_rate'] = latest['home_recent_home_win_rate']
                home_result.loc[home_result['home_team_id'] == team_id, 'home_recent_away_win_rate'] = latest['home_recent_away_win_rate']
                
                away_result.loc[away_result['away_team_id'] == team_id, 'away_recent_win_rate'] = latest['home_recent_win_rate']
                away_result.loc[away_result['away_team_id'] == team_id, 'away_recent_avg_score'] = latest['home_recent_avg_score']
                away_result.loc[away_result['away_team_id'] == team_id, 'away_recent_home_win_rate'] = latest['home_recent_home_win_rate']
                away_result.loc[away_result['away_team_id'] == team_id, 'away_recent_away_win_rate'] = latest['home_recent_away_win_rate']
    
    return home_result, away_result

# historical_home과 historical_away를 직접 사용
updated_home_records, updated_away_records = update_both_records_with_latest_stats(historical_home, historical_away)


updated_features_home = [
    'home_team_id', 'home_team_name','home_rebounds', 'home_assists', 'home_fieldGoalsAttempted',
    'home_fieldGoalsMade', 'home_fieldGoalPct', 'home_freeThrowPct',
    'home_freeThrowsAttempted', 'home_freeThrowsMade', 'home_threePointPct',
    'home_threePointFieldGoalsAttempted', 'home_threePointFieldGoalsMade',
    'home_leader_points', 'home_leader_rebounds', 'home_leader_assists',
    'home_recent_win_rate', 'home_recent_avg_score', 'home_recent_home_win_rate',
    'home_recent_away_win_rate'
]
updated_home_records = updated_home_records[updated_features_home]

updated_features_away = [
    'away_team_id', 'away_team_name','away_rebounds', 'away_assists', 'away_fieldGoalsAttempted',
    'away_fieldGoalsMade', 'away_fieldGoalPct', 'away_freeThrowPct',
    'away_freeThrowsAttempted', 'away_freeThrowsMade', 'away_threePointPct',
    'away_threePointFieldGoalsAttempted', 'away_threePointFieldGoalsMade',
    'away_leader_points', 'away_leader_rebounds', 'away_leader_assists',
    'away_recent_win_rate', 'away_recent_avg_score', 'away_recent_home_win_rate',
    'away_recent_away_win_rate'
]
updated_away_records = updated_away_records[updated_features_away]


# # 결과를 다시 저장
# processor.save_processed_data(updated_home_records, "processed_home_updated.json")
# processor.save_processed_data(updated_away_records, "processed_away_updated.json")









# 아 맞다!! 잘못 생각하고 있었네!!!!! 이거 home away 구분 안하고 과거 경기 평균치구나 그냥 홈어웨이 구분안하고 팀별로 가장 최근 데이터 가져다 붙여야겠다 historical_home이랑 historical_away 같아지게
# 2. away_team_id별 가장 최근 레코드
# 3. home_team_id, away_team_id별 상대전적 레코드 -> 이거 만약 상대전적이 홈 어웨이 상관없이 단순 맞대결 전적이라면 홈id 어웨이id 뒤집고 그냥 승률 1에서 뺀 것도 추가해야 될 거 같은데
# 이걸 각각 upcoming에 붙이면 되겠다





upcoming_df = processor.process_game_data(upcoming_data)
upcoming_features = processor.extract_features(upcoming_df, upcoming_data)
# 필요한 피처들만 선택
selected_features_upcoming = [
    'game_id', 'date', 'season_year', 'season_type', 'status',
    'home_team_id', 'home_team_name', 'home_team_score',
    'away_team_id', 'away_team_name', 'away_team_score',
    'home_overall_record_win_rate', 'away_overall_record_win_rate',
    'home_home_record_win_rate', 'away_home_record_win_rate',
    'home_road_record_win_rate', 'away_road_record_win_rate'
]
upcoming_features = upcoming_features[selected_features_upcoming]

# processor.save_processed_data(upcoming_features, "upcoming_features.json")


# 최종 features 생성 (game_id 기준으로 total_features를 upcoming_features에 병합)
final_features = upcoming_features.merge(
    total_features,
    on='game_id',
    how='left'
)


# final_features와 complete_team_matchups 병합
final_features = final_features.merge(
    complete_team_matchups,
    on=['home_team_id', 'home_team_name', 'away_team_id', 'away_team_name'],
    how='left'
)

# 매칭되지 않는 경우 기본값 설정
final_features['home_vs_away_wins'] = final_features['home_vs_away_wins'].fillna(0.0)
final_features['home_vs_away_losses'] = final_features['home_vs_away_losses'].fillna(0.0)
final_features['home_vs_away_win_rate'] = final_features['home_vs_away_win_rate'].fillna(0.5)


# final_features와 updated_home_records 병합
final_features = final_features.merge(
    updated_home_records,
    on=['home_team_id', 'home_team_name'],
    how='left'
)

# final_features와 updated_away_records 병합
final_features = final_features.merge(
    updated_away_records,
    on=['away_team_id', 'away_team_name'],
    how='left'
)



# 저장 (필요한 경우)
# processor.save_processed_data(final_features, "final_features.json")






# 모델 로드 및 예측
trainer = BettingModelTrainer()
model_dir = Path(__file__).parent / "saved_models"
model_files = list(model_dir.glob("*.joblib"))

if not model_files:
    raise FileNotFoundError("저장된 모델을 찾을 수 없습니다.")

# 가장 최근 모델 파일 선택
latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
print(f"모델 파일 로드: {latest_model.name}")

# 모델 로드 - 이 부분이 추가되어야 함
trainer.model = joblib.load(latest_model)

# final_features로 예측 수행
X, _ = trainer.prepare_features(final_features)
predictions = trainer.model.predict(X)
probabilities = trainer.model.predict_proba(X)
X

# 예측 결과 출력
for i in range(len(final_features)):
    print(f"\n=== 경기 예측 ===")
    print(f"홈팀: {final_features.iloc[i]['home_team_name']} vs 원정팀: {final_features.iloc[i]['away_team_name']}")
    print(f"승리 예측: {'홈팀' if predictions[i] == 1 else '원정팀'}")
    print(f"홈팀 승리 확률: {probabilities[i][1]:.3f}")





# processor.save_processed_data(X, "X.json")

