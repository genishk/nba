# src/data/processor_modelinput_with_odds.py
"""
배당 변수를 포함한 예측 입력 데이터 생성
- 기존 processor_modelinput.py 복사본
- 실시간 배당 가져와서 8구간 버킷화 추가
- home_odds_bucket, away_odds_bucket 변수 추가
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # 프로젝트 루트 추가
from src.data.espn_api import ESPNNBADataCollector
from src.data.processor_upcoming import DataProcessor_upcoming
from src.data.processor_modelinput import DataProcessor


# ============================================================
# 팀명 매핑 (배당 API용 약자 <-> ESPN 닉네임)
# ============================================================
TEAM_ABBREV_TO_NAME = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards'
}

TEAM_NAME_TO_ABBREV = {v: k for k, v in TEAM_ABBREV_TO_NAME.items()}

# ============================================================
# 배당 버킷화 설정 (학습 시와 동일!)
# ============================================================
ODDS_BINS = [-float('inf'), -400, -250, -150, -100, 150, 250, 400, float('inf')]
ODDS_LABELS = [0, 1, 2, 3, 4, 5, 6, 7]


def load_latest_odds() -> pd.DataFrame:
    """최신 배당 데이터 로드"""
    odds_dir = Path(__file__).parent.parent.parent / "data" / "odds"
    
    # processed_nba_odds_*.json 파일 찾기
    odds_files = list(odds_dir.glob("processed_nba_odds_*.json"))
    
    if not odds_files:
        print("⚠️ 배당 데이터 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    
    latest_file = max(odds_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 배당 데이터 로드: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data)


def bucketize_odds(american_odds: float) -> int:
    """
    미국식 배당을 8구간으로 버킷화 (학습 시와 동일한 로직!)
    
    구간:
    0: ~ -400 (압도적 우세)
    1: -400 ~ -250
    2: -250 ~ -150
    3: -150 ~ -100 (약간 우세)
    4: -100 ~ +150 (약간 열세)
    5: +150 ~ +250
    6: +250 ~ +400
    7: +400 ~ (압도적 열세)
    """
    if pd.isna(american_odds):
        return 4  # 기본값: 중간 구간
    
    for i, (low, high) in enumerate(zip(ODDS_BINS[:-1], ODDS_BINS[1:])):
        if low <= american_odds < high:
            return ODDS_LABELS[i]
    
    return 4  # 기본값


def add_odds_to_features(model_input_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    예측 입력 데이터에 배당 변수 추가
    
    Args:
        model_input_df: processor_modelinput에서 생성한 예측 입력 데이터
        odds_df: 실시간 배당 데이터
    
    Returns:
        배당 변수가 추가된 DataFrame
    """
    print("\n=== 배당 변수 추가 시작 ===")
    
    result_df = model_input_df.copy()
    
    # 배당 데이터가 비어있으면 기본값 설정
    if odds_df.empty:
        print("⚠️ 배당 데이터가 없습니다. 기본값(4)으로 설정합니다.")
        result_df['home_odds_bucket'] = 4
        result_df['away_odds_bucket'] = 4
        return result_df
    
    # 배당 데이터를 경기별로 정리 (홈/어웨이 배당)
    # odds_df 구조: game_id, home_team(약자), away_team(약자), team, is_home, odds
    
    matched_count = 0
    unmatched_games = []
    
    for idx, row in result_df.iterrows():
        home_name = row['home_team_name']  # ESPN 닉네임 (예: 'Hawks')
        away_name = row['away_team_name']  # ESPN 닉네임 (예: 'Celtics')
        
        # 닉네임 → 약자 변환
        home_abbrev = TEAM_NAME_TO_ABBREV.get(home_name, home_name)
        away_abbrev = TEAM_NAME_TO_ABBREV.get(away_name, away_name)
        
        # 해당 경기의 배당 찾기
        game_odds = odds_df[
            (odds_df['home_team'] == home_abbrev) & 
            (odds_df['away_team'] == away_abbrev)
        ]
        
        if len(game_odds) >= 2:
            # 홈팀 배당 (is_home == True인 행)
            home_odds_row = game_odds[game_odds['is_home'] == True]
            # 어웨이팀 배당 (is_home == False인 행)
            away_odds_row = game_odds[game_odds['is_home'] == False]
            
            if len(home_odds_row) > 0 and len(away_odds_row) > 0:
                home_odds = home_odds_row.iloc[0]['odds']
                away_odds = away_odds_row.iloc[0]['odds']
                
                # 버킷화
                result_df.loc[idx, 'home_odds_bucket'] = bucketize_odds(home_odds)
                result_df.loc[idx, 'away_odds_bucket'] = bucketize_odds(away_odds)
                result_df.loc[idx, 'home_odds_raw'] = home_odds  # 원본 배당도 저장 (참고용)
                result_df.loc[idx, 'away_odds_raw'] = away_odds
                
                matched_count += 1
                print(f"  ✅ {home_name} vs {away_name}: home={home_odds}→{bucketize_odds(home_odds)}, away={away_odds}→{bucketize_odds(away_odds)}")
            else:
                # 매칭 실패
                result_df.loc[idx, 'home_odds_bucket'] = 4
                result_df.loc[idx, 'away_odds_bucket'] = 4
                unmatched_games.append(f"{home_name} vs {away_name}")
        else:
            # 매칭 실패
            result_df.loc[idx, 'home_odds_bucket'] = 4
            result_df.loc[idx, 'away_odds_bucket'] = 4
            unmatched_games.append(f"{home_name} vs {away_name}")
    
    print(f"\n배당 매칭 결과: {matched_count}/{len(result_df)} 경기")
    if unmatched_games:
        print(f"⚠️ 매칭 실패 ({len(unmatched_games)}개): {', '.join(unmatched_games)}")
    
    # 버킷 컬럼 타입 변환
    result_df['home_odds_bucket'] = result_df['home_odds_bucket'].astype(float)
    result_df['away_odds_bucket'] = result_df['away_odds_bucket'].astype(float)
    
    print(f"\n✅ 배당 변수 추가 완료: home_odds_bucket, away_odds_bucket")
    
    return result_df


def save_processed_data_with_odds(df: pd.DataFrame, prefix: str = "model_input_features_with_odds") -> Path:
    """배당 포함 예측 입력 데이터 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    
    output_path = Path(__file__).parent / filename
    
    # DataFrame을 JSON으로 변환
    json_data = df.to_json(orient='records', date_format='iso')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_data)
    
    print(f"\n💾 배당 포함 예측 입력 데이터 저장: {output_path}")
    return output_path


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🏀 배당 포함 예측 입력 데이터 생성")
    print("=" * 70)
    
    # ============================================================
    # 1. 기존 processor_modelinput.py 로직 실행 (과거 데이터 처리)
    # ============================================================
    processor = DataProcessor()
    data = processor.load_latest_data()
    games_df = processor.process_game_data(data)
    features_df = processor.extract_features(games_df, data)
    
    print("\n=== 과거 데이터 처리 완료 ===")
    print(f"처리된 경기 수: {len(features_df)}")
    
    # ============================================================
    # 2. 예정된 경기 데이터 수집 및 처리
    # ============================================================
    processor_upcoming = DataProcessor_upcoming()
    collector = ESPNNBADataCollector()
    
    # 예정된 경기 데이터 수집 (1일 앞)
    upcoming_data = collector.collect_upcoming_data(days_ahead=1)
    
    # STATUS_SCHEDULED인 경기만 필터링
    upcoming_data['games'] = [game for game in upcoming_data['games'] 
                            if game['status'] == 'STATUS_SCHEDULED']
    
    if not upcoming_data['games']:
        print("\n⚠️ 예정된 경기가 없습니다.")
        exit()
    
    # 기본 정보 추출
    upcoming_df = processor_upcoming.process_game_data(upcoming_data)
    upcoming_features = processor_upcoming.extract_features(upcoming_df, upcoming_data)
    
    # 필요한 기본 피처들만 선택
    selected_features_upcoming = [
        'game_id', 'date', 'season_year', 'season_type', 'status',
        'home_team_id', 'home_team_name', 'home_team_score',
        'away_team_id', 'away_team_name', 'away_team_score'
    ]
    upcoming_features = upcoming_features[selected_features_upcoming]
    
    print(f"\n=== 예정된 경기 ===")
    print(f"경기 수: {len(upcoming_features)}")
    for _, row in upcoming_features.iterrows():
        print(f"  - {row['home_team_name']} vs {row['away_team_name']}")
    
    # ============================================================
    # 3. 최신 팀 통계 추가 (기존 로직)
    # ============================================================
    model_input_features = processor.add_latest_team_stats(upcoming_features, features_df)
    
    print(f"\n=== 기본 특성 추가 완료 ===")
    print(f"특성 수: {len(model_input_features.columns)}")
    
    # ============================================================
    # 4. 실시간 배당 로드 및 버킷화 추가 (새로운 로직!)
    # ============================================================
    odds_df = load_latest_odds()
    model_input_with_odds = add_odds_to_features(model_input_features, odds_df)
    
    # ============================================================
    # 5. 저장
    # ============================================================
    output_path = save_processed_data_with_odds(model_input_with_odds)
    
    # 결과 확인
    print("\n" + "=" * 70)
    print("✅ 배당 포함 예측 입력 데이터 생성 완료!")
    print("=" * 70)
    print(f"\n최종 특성 수: {len(model_input_with_odds.columns)}")
    print(f"배당 변수: home_odds_bucket, away_odds_bucket")
    
    # 배당 정보 확인
    if 'home_odds_bucket' in model_input_with_odds.columns:
        print("\n=== 배당 버킷 분포 ===")
        print(f"home_odds_bucket: {model_input_with_odds['home_odds_bucket'].value_counts().to_dict()}")
        print(f"away_odds_bucket: {model_input_with_odds['away_odds_bucket'].value_counts().to_dict()}")

