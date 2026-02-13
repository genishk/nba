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

class DataProcessor:
    def __init__(self, data_dir: Optional[Path] = None):
        """데이터 처리를 위한 클래스 초기화"""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self.data_dir = data_dir
    
    def load_latest_data(self, data_type: str = 'historical') -> Dict[str, Any]:
        """최신 데이터 파일 로드
        
        Args:
            data_type: 'historical' 또는 'upcoming'
        """
        # 데이터 디렉토리 설정
        data_dir = Path(__file__).parent.parent.parent / "data"
        if data_type == 'historical':
            data_dir = data_dir / "raw" / "historical"
        else:
            data_dir = data_dir / "upcoming" / "games"
        
        # 최신 파일 찾기
        json_files = list(data_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_dir}")
        
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"\n데이터 파일 로드: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process_game_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """경기 데이터를 DataFrame으로 변환"""
        print("\n=== 데이터 처리 시작 ===")
        collection_period = data.get('days_collected') or data.get('days_ahead')
        print(f"수집 기간: {collection_period}일")
        print(f"전체 경기 수: {len(data['games'])}")
        
        games_list = []
        skipped_games = 0
        processed_games = 0
        
        for game in data['games']:
            # 과거 데이터는 완료된 경기만, 미래 데이터는 예정된 경기만 처리
            is_historical = 'days_collected' in data
            if is_historical and game['status'] != 'STATUS_FINAL':
                skipped_games += 1
                continue
            elif not is_historical and game['status'] != 'STATUS_SCHEDULED':
                skipped_games += 1
                continue
            
            try:
                game_dict = self._extract_game_info(game)
                games_list.append(game_dict)
                processed_games += 1
                
                if processed_games % 100 == 0:
                    print(f"처리 진행률: {processed_games}/{len(data['games'])} 경기 완료")
                
            except Exception as e:
                print(f"\nError processing game: {e}")
                print(f"Game ID: {game.get('game_id', 'Unknown')}")
                continue
        
        df = pd.DataFrame(games_list)
        
        print(f"\n데이터 처리 완료:")
        print(f"- 처리된 경기 수: {processed_games}")
        print(f"- 건너뛴 경기 수: {skipped_games}")
        print(f"- 처리 비율: {processed_games/(processed_games+skipped_games)*100:.1f}%")
        
        return df
    
    def _extract_game_info(self, game: Dict) -> Dict:
        """개별 경기 정보 추출"""
        game_dict = {
            # 기본 경기 정보
            'game_id': game['game_id'],
            'date': pd.to_datetime(game['date']),
            'season_year': game['season']['year'],
            'season_type': game['season']['type'],
            'status': game['status'],
            
            # 팀 정보 및 점수
            'home_team_id': game['home_team']['id'],
            'home_team_name': game['home_team']['name'],
            'home_team_score': game['home_team']['score'],
            'away_team_id': game['away_team']['id'],
            'away_team_name': game['away_team']['name'],
            'away_team_score': game['away_team']['score'],
        }
        
        # 팀 통계 처리
        for team_type, team in [('home', game['home_team']), ('away', game['away_team'])]:
            for stat in team.get('statistics', []):
                if isinstance(stat, dict):
                    stat_name = stat['name']
                    if stat_name not in ['avgRebounds', 'avgAssists', 'avgPoints', 'threePointFieldGoalPct', 'points']:
                        game_dict[f"{team_type}_{stat_name}"] = stat.get('displayValue')
                        if stat.get('rankDisplayValue'):
                            game_dict[f"{team_type}_{stat_name}_rank"] = stat.get('rankDisplayValue')
        
        # 팀 기록 추가
        for team_type, team in [('home', game['home_team']), ('away', game['away_team'])]:
            for record in team.get('records', []):
                record_name = record['name'].lower().replace(' ', '_')
                game_dict[f"{team_type}_{record_name}_record"] = record.get('summary')
        
        
        return game_dict
    
    def extract_features(self, games_df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        print("\n=== 특성 추출 시작 ===")
        df = games_df.copy()
        
        # 1. 문자열로 된 통계값들을 수치형으로 변환
        numeric_columns = {
            # 기존 통계
            'rebounds', 'assists', 'fieldGoalPct', 'freeThrowPct', 'threePointPct',
            # 점수 관련
            'team_score', 'points',
            # 슈팅 관련
            'fieldGoalsAttempted', 'fieldGoalsMade',
            'freeThrowsAttempted', 'freeThrowsMade',
            'threePointFieldGoalsAttempted', 'threePointFieldGoalsMade',
            # 리더 통계
            'leader_points', 'leader_rebounds', 'leader_assists',
        }
        
        for col_base in numeric_columns:
            for team_type in ['home', 'away']:
                col = f"{team_type}_{col_base}"
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100.0 if 'Pct' in col \
                                else pd.to_numeric(df[col], errors='coerce')
            
        def extract_win_rate(record, default_rate=0.5):
            """승률 계산 함수 단순화"""
            if pd.isna(record) or not isinstance(record, str):
                return default_rate
            try:
                wins, losses = map(int, record.split('-'))
                total_games = wins + losses
                return round(wins / total_games, 3) if total_games > 0 else default_rate
            except Exception as e:
                print(f"Error in extract_win_rate: {e}")
                return default_rate

        # 각 레코드 타입별 승률 계산
        for team_type in ['home', 'away']:
            # overall record
            df[f'{team_type}_overall_record_win_rate'] = df[f'{team_type}_overall_record'].apply(
                lambda x: extract_win_rate(x)
            )
            
            # home record
            df[f'{team_type}_home_record_win_rate'] = df[f'{team_type}_home_record'].apply(
                lambda x: extract_win_rate(x)
            )
            
            # road record
            df[f'{team_type}_road_record_win_rate'] = df[f'{team_type}_road_record'].apply(
                lambda x: extract_win_rate(x)
            )

    
        # 상대전적 정보 추가
        df = self._add_head_to_head_stats(df, data)
        
        # 최근 트렌드 정보 추가
        df = self._add_recent_trends(df, data)
        
        # 휴식일 수 정보 추가
        df = self._add_rest_days(df, data)
        
        # 결측치 처리
        df = self._handle_missing_values(df)
        
        # 최근 10경기 평균 통계로 대체
        df = self._add_recent_stats_average(df, data)
        
        return df
    
    def _calculate_recent_form(self, form_data: List, n_games: int = 10) -> float:
        """최근 N경기 승률 계산"""
        if not form_data:
            return 0.5
        
        recent_games = form_data[-n_games:]
        wins = sum(1 for game in recent_games if game.get('result') == 'W')
        return wins / len(recent_games) if recent_games else 0.5
    
    def _add_recent_performance(self, df: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
        """최근 N경기 성적 추가"""
        # 구현...
    
    def _add_team_rankings(self, df: pd.DataFrame, team_stats: Dict) -> pd.DataFrame:
        """팀 순위 정보 추가"""
        # 구현...
    
    def _add_head_to_head_stats(self, df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """상대전적 정보 추가"""
        print("\n=== 상대전적 정보 추가 ===")
        
        # 날짜순으로 정렬
        df = df.sort_values('date')
        
        # 각 팀 간의 상대전적을 저장할 딕셔너리
        h2h_records = {}  # {(team1_id, team2_id): [win_count, total_games]}
        
        # 모든 경기를 순회하며 상대전적 계산
        for idx, row in df.iterrows():
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            team_key = tuple(sorted([home_id, away_id]))
            
            # 현재 경기까지의 상대전적을 기록
            if pd.notna(row['home_team_score']) and pd.notna(row['away_team_score']):
                if team_key not in h2h_records:
                    h2h_records[team_key] = [0, 0]
                
                h2h_records[team_key][1] += 1  # 총 경기 수 증가
                if int(row['home_team_score']) > int(row['away_team_score']):
                    if home_id == team_key[0]:
                        h2h_records[team_key][0] += 1
                else:
                    if away_id == team_key[0]:
                        h2h_records[team_key][0] += 1
            
            # 현재까지의 기록을 저장
            wins = h2h_records.get(team_key, [0, 0])[0]
            total = h2h_records.get(team_key, [0, 0])[1]
            
            if home_id == team_key[0]:
                df.loc[idx, 'home_vs_away_wins'] = wins
                df.loc[idx, 'home_vs_away_losses'] = total - wins
                df.loc[idx, 'home_vs_away_win_rate'] = wins / total if total > 0 else 0.5
            else:
                df.loc[idx, 'home_vs_away_wins'] = total - wins
                df.loc[idx, 'home_vs_away_losses'] = wins
                df.loc[idx, 'home_vs_away_win_rate'] = (total - wins) / total if total > 0 else 0.5
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'Int64']).columns
        
        for col in numeric_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"\n{col}: {missing}개의 결측치 발견")
                
                if 'rating' in col:
                    # 레이팅 관련 결측치는 리그 평균으로 대체
                    league_avg = df[col].mean()
                    df[col] = df[col].fillna(league_avg)
                    print(f"- 리그 평균({league_avg:.3f})으로 대체")
                    
                elif 'leader' in col:
                    # 리더 통계는 팀 평균으로 대체
                    team_type = col.split('_')[0]  # 'home' or 'away'
                    team_means = df.groupby(f'{team_type}_team_id')[col].transform('mean')
                    df[col] = df[col].fillna(team_means).fillna(0)  # 팀 평균이 없으면 0
                    print(f"- 팀 평균 또는 0으로 대체")
                    
                elif col.startswith(('home_', 'away_')):
                    # 팀 통계는 해당 팀의 평균으로 대체
                    team_type = col.split('_')[0]
                    team_means = df.groupby(f'{team_type}_team_id')[col].transform('mean')
                    league_mean = df[col].mean()
                    df[col] = df[col].fillna(team_means).fillna(league_mean)
                    print(f"- 팀 평균 또는 리그 평균으로 대체")
                
                else:
                    # 기타 통계는 전체 평균으로 대체
                    df[col] = df[col].fillna(df[col].mean())
                    print(f"- 전체 평균으로 대체")
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 최적화"""
        # 정수형으로 변환할 컬럼들
        int_columns = [
            'season_year', 'season_type',
            'home_team_score', 'away_team_score'  # 실제 점수만 정수형으로
        ]
        
        # 실수형으로 유지할 컬럼들 (비율, 승률, 평균 통계 등)
        float_columns = [col for col in df.select_dtypes(include=['float64']).columns 
                        if col not in int_columns]
        
        # 데이터 타입 변환
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].round().astype('Int64')  # nullable integer type
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, prefix: str = "processed") -> Path:
        """처리된 데이터를 JSON 파일로 저장
        
        Args:
            df: 처리된 DataFrame
            prefix: 파일명 접두사
        
        Returns:
            저장된 파일 경로
        """
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_prediction.json"
        
        # processor.py와 같은 폴더에 저장
        output_path = Path(__file__).parent / filename
        
        # DataFrame을 JSON으로 변환 (날짜/시간 처리를 위해 date_format 사용)
        json_data = df.to_json(orient='records', date_format='iso')
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        print(f"\n처리된 데이터 저장 완료: {output_path}")
        return output_path
        
    def _add_recent_trends(self, df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """각 팀의 최근 5경기 승률 및 평균 점수 계산"""
        print("\n=== 최근 5경기 트렌드 계산 ===")
        
        # 팀별 경기 결과 저장
        team_games = defaultdict(list)
        team_games_dates = defaultdict(list)
        
        # 날짜순으로 정렬된 경기들에서 결과 수집
        sorted_games = sorted(data['games'], key=lambda x: x['date'])
        for game in sorted_games:
            if game['status'] != 'STATUS_FINAL':
                continue
                
            game_date = pd.to_datetime(game['date'])
            home_team_id = game['home_team']['id']
            away_team_id = game['away_team']['id']
            home_score = float(game['home_team']['score'])  # 문자열을 숫자로 변환
            away_score = float(game['away_team']['score'])  # 문자열을 숫자로 변환
            
            # 홈팀 결과 저장
            team_games[home_team_id].append({
                'is_home': True,
                'won': home_score > away_score,
                'score': home_score
            })
            team_games_dates[home_team_id].append(game_date)
            
            # 원정팀 결과 저장
            team_games[away_team_id].append({
                'is_home': False,
                'won': away_score > home_score,
                'score': away_score
            })
            team_games_dates[away_team_id].append(game_date)
        
        # 시즌 첫 5경기 평균 계산
        team_first_5_stats = defaultdict(dict)
        for team_id in team_games:
            first_5_games = team_games[team_id][:5]
            if first_5_games:
                team_first_5_stats[team_id] = {
                    'win_rate': np.mean([game['won'] for game in first_5_games]),
                    'avg_score': np.mean([game['score'] for game in first_5_games]),
                    'home_win_rate': np.mean([game['won'] for game in first_5_games if game['is_home']]) if any(game['is_home'] for game in first_5_games) else 0.0,
                    'away_win_rate': np.mean([game['won'] for game in first_5_games if not game['is_home']]) if any(not game['is_home'] for game in first_5_games) else 0.0
                }
        
        # 각 경기에 대해 해당 시점까지의 최근 5경기 트렌드 계산
        for idx, row in df.iterrows():
            current_game_date = pd.to_datetime(row['date'])
            
            for team_type, team_id in [('home', row['home_team_id']), ('away', row['away_team_id'])]:
                # 현재 경기 이전의 결과만 필터링
                previous_games = [
                    game for game, date in zip(
                        team_games[team_id],
                        team_games_dates[team_id]
                    )
                    if date <= current_game_date
                ]
                
                if len(previous_games) >= 5:
                    # 최근 5경기 결과
                    recent_games = previous_games[-5:]
                    
                    # 전체 승률
                    df.loc[idx, f'{team_type}_recent_win_rate'] = np.mean([game['won'] for game in recent_games])
                    
                    # 평균 득점
                    df.loc[idx, f'{team_type}_recent_avg_score'] = round(np.mean([game['score'] for game in recent_games]), 2)
                    
                    # 홈/원정 승률
                    recent_home_games = [game for game in recent_games if game['is_home']]
                    recent_away_games = [game for game in recent_games if not game['is_home']]
                    
                    df.loc[idx, f'{team_type}_recent_home_win_rate'] = np.mean([game['won'] for game in recent_home_games]) if recent_home_games else 0.0
                    df.loc[idx, f'{team_type}_recent_away_win_rate'] = np.mean([game['won'] for game in recent_away_games]) if recent_away_games else 0.0
                else:
                    # 이전 경기가 5경기 미만인 경우 시즌 첫 5경기 평균 사용
                    df.loc[idx, f'{team_type}_recent_win_rate'] = team_first_5_stats[team_id]['win_rate']
                    df.loc[idx, f'{team_type}_recent_avg_score'] = team_first_5_stats[team_id]['avg_score']
                    df.loc[idx, f'{team_type}_recent_home_win_rate'] = team_first_5_stats[team_id]['home_win_rate']
                    df.loc[idx, f'{team_type}_recent_away_win_rate'] = team_first_5_stats[team_id]['away_win_rate']
        
        return df    
        
    def _add_rest_days(self, df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """각 팀의 이전 경기와의 휴식일 수 계산"""
        print("\n=== 휴식일 수 정보 추가 ===")
        
        team_games = defaultdict(list)
        
        # 각 팀별 경기 수집
        for game in data['games']:
            game_date = pd.to_datetime(game['date'])
            
            # 홈팀 경기 추가
            team_games[game['home_team']['id']].append(game_date)
            # 원정팀 경기 추가
            team_games[game['away_team']['id']].append(game_date)
        
        # 각 팀의 경기를 날짜순으로 정렬
        for team_id in team_games:
            team_games[team_id].sort()
        
        # 각 경기에 대해 양 팀의 휴식일 수 계산
        for idx, row in df.iterrows():
            game_date = pd.to_datetime(row['date'])
            
            for team_type, team_id in [('home', row['home_team_id']), ('away', row['away_team_id'])]:
                team_dates = team_games[team_id]
                
                # 현재 경기 이전의 가장 최근 경기 찾기
                prev_dates = [d for d in team_dates if d < game_date]
                if prev_dates:
                    last_game = max(prev_dates)
                    # 날짜만 추출하여 차이 계산 (시간은 무시)
                    game_day = game_date.date()
                    last_game_day = last_game.date()
                    rest_days = (game_day - last_game_day).days - 1
                    rest_days = max(0, rest_days)  # 음수가 나오지 않도록
                else:
                    rest_days = 7  # 이전 경기가 없는 경우 (시즌 첫 경기 등)
                
                df.loc[idx, f'{team_type}_rest_days'] = rest_days
        
        return df
        



    def _add_recent_stats_average(self, df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """각 팀의 최근 5경기 통계 평균 계산"""
        print("\n=== 최근 5경기 통계 평균 계산 ===")
        
        # 대체할 통계 필드들
        stat_fields = [
            'rebounds', 'assists', 
            'fieldGoalsAttempted', 'fieldGoalsMade', 'fieldGoalPct',
            'freeThrowsAttempted', 'freeThrowsMade', 'freeThrowPct',
            'threePointFieldGoalsAttempted', 'threePointFieldGoalsMade', 'threePointPct'
        ]
        
        # 리더 통계 필드들
        leader_fields = ['points', 'rebounds', 'assists']
        
        # 팀별 경기 통계 저장
        team_stats = defaultdict(lambda: defaultdict(list))
        team_games_dates = defaultdict(list)
        team_games_types = defaultdict(list)  # 홈/원정 정보 저장
        
        # 1. 날짜순으로 정렬된 경기에서 통계 수집
        sorted_games = sorted(data['games'], key=lambda x: x['date'])
        for game in sorted_games:
            if game['status'] != 'STATUS_FINAL':
                continue
            
            game_date = pd.to_datetime(game['date'])
            
            for team_type, team in [('home', game['home_team']), ('away', game['away_team'])]:
                team_id = team['id']
                team_games_dates[team_id].append(game_date)
                team_games_types[team_id].append(team_type)
                
                # 기본 통계 수집
                stats_dict = {}
                for stat in team.get('statistics', []):
                    if isinstance(stat, dict) and stat['name'] in stat_fields:
                        value = pd.to_numeric(stat.get('displayValue', '0').rstrip('%'), errors='coerce')
                        if 'Pct' in stat['name']:
                            value = value / 100.0
                        stats_dict[stat['name']] = value
                
                # 리더 통계 수집
                for leader in team.get('leaders', []):
                    if leader.get('leaders') and leader['leaders'] and leader['name'] in leader_fields:
                        value = pd.to_numeric(leader['leaders'][0].get('displayValue', '0').split(' ')[0], errors='coerce')
                        stats_dict[f"leader_{leader['name']}"] = value
                
                # 모든 필드에 대해 통계 저장 (없는 경우 NaN)
                for field in stat_fields + [f"leader_{field}" for field in leader_fields]:
                    team_stats[team_id][field].append(stats_dict.get(field, np.nan))
        
        # 2. 통계가 없는 경기는 같은 유형(홈/원정)의 다음 경기 통계로 대체
        for team_id in team_stats:
            team_games = list(zip(team_games_dates[team_id], team_games_types[team_id]))
            
            for stat_name in list(stat_fields) + [f"leader_{field}" for field in leader_fields]:
                stats = team_stats[team_id][stat_name]
                
                # 통계가 없는 경기 찾아서 대체
                for i in range(len(stats)):
                    if pd.isna(stats[i]):
                        current_type = team_games[i][1]  # 현재 경기의 홈/원정
                        # 다음 경기들 중 같은 유형 찾기
                        for j in range(i + 1, len(stats)):
                            if team_games[j][1] == current_type and not pd.isna(stats[j]):
                                stats[i] = stats[j]
                                break
        
        # 3. 각 팀의 첫 5경기 평균 계산 (결측치 대체 후)
        team_first_5_avg = {}
        for team_id in team_stats:
            team_first_5_avg[team_id] = {}
            for stat_name in list(stat_fields) + [f"leader_{field}" for field in leader_fields]:
                first_5_stats = team_stats[team_id][stat_name][:5]  # 이미 결측치가 대체된 값들
                if first_5_stats:
                    team_first_5_avg[team_id][stat_name] = np.mean(first_5_stats)
        
        # 4. DataFrame에 통계 추가
        for idx, row in df.iterrows():
            current_game_date = pd.to_datetime(row['date'])
            
            for team_type, team_id in [('home', row['home_team_id']), ('away', row['away_team_id'])]:
                try:
                    current_idx = team_games_dates[team_id].index(current_game_date)
                except ValueError:
                    continue
                
                for stat_name in stat_fields + [f"leader_{field}" for field in leader_fields]:
                    col_name = f"{team_type}_{stat_name}"
                    
                    if current_idx < 5:  # 첫 5경기는 첫 5경기 평균으로 고정
                        avg_value = team_first_5_avg[team_id][stat_name]
                    else:  # 6번째 경기부터는 직전 5경기 평균
                        prev_5_stats = team_stats[team_id][stat_name][current_idx-5+1:current_idx+1]
                        avg_value = np.mean(prev_5_stats)
                    
                    if 'Pct' in stat_name:  # 퍼센티지는 소수점 유지
                        df.loc[idx, col_name] = avg_value
                    else:  # 나머지는 소수점 2자리까지만 유지
                        df.loc[idx, col_name] = round(avg_value, 2)
        
        return df


    def add_latest_team_stats(self, upcoming_features: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """예정된 경기에 각 팀의 최신 통계를 추가"""
        print("\n=== 최신 팀 통계 추가 시작 ===")
        
        # 결과 DataFrame 초기화
        result_df = upcoming_features.copy()
        
        # 통계 필드 정의
        stat_fields = [
            'rebounds', 'assists', 
            'fieldGoalsAttempted', 'fieldGoalsMade', 'fieldGoalPct',
            'freeThrowsAttempted', 'freeThrowsMade', 'freeThrowPct',
            'threePointFieldGoalsAttempted', 'threePointFieldGoalsMade', 'threePointPct'
        ]
        
        record_fields = [
            'overall_record', 'home_record', 'road_record',
            'overall_record_win_rate', 'home_record_win_rate', 'road_record_win_rate'
        ]
        
        trend_fields = [
            'recent_win_rate', 'recent_avg_score',
            'recent_home_win_rate', 'recent_away_win_rate'
        ]
        
        leader_fields = [
            'leader_points', 'leader_rebounds', 'leader_assists'
        ]

        # features_df를 날짜순으로 정렬
        features_df = features_df.copy()
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df = features_df.sort_values('date')
        
        # upcoming_features의 날짜도 datetime으로 변환
        result_df['date'] = pd.to_datetime(result_df['date'])
        
        # 각 예정된 경기에 대해
        for idx, upcoming_game in result_df.iterrows():
            # 1. 팀별 최신 통계 복사
            for team_type in ['home', 'away']:
                team_id = upcoming_game[f'{team_type}_team_id']
                team_name = upcoming_game[f'{team_type}_team_name']
                
                # 해당 팀의 가장 최근 경기 찾기
                team_games = features_df[
                    (features_df['home_team_id'] == team_id) | 
                    (features_df['away_team_id'] == team_id)
                ]
                
                if len(team_games) == 0:
                    print(f"Warning: No previous games found for {team_name}")
                    continue
                    
                latest_game = team_games.iloc[-1]
                
                # 최근 경기에서 해당 팀이 홈/어웨이 였는지 확인
                was_home = latest_game['home_team_id'] == team_id
                source_prefix = 'home' if was_home else 'away'
                
                # 기본 통계 복사
                for field in stat_fields:
                    source_col = f'{source_prefix}_{field}'
                    target_col = f'{team_type}_{field}'
                    if source_col in latest_game:
                        result_df.loc[idx, target_col] = latest_game[source_col]
                
                # 승률 관련 기록 복사
                for field in record_fields:
                    source_col = f'{source_prefix}_{field}'
                    target_col = f'{team_type}_{field}'
                    if source_col in latest_game:
                        result_df.loc[idx, target_col] = latest_game[source_col]
                
                # 최근 트렌드 복사
                for field in trend_fields:
                    source_col = f'{source_prefix}_{field}'
                    target_col = f'{team_type}_{field}'
                    if source_col in latest_game:
                        result_df.loc[idx, target_col] = latest_game[source_col]
                
                # 리더 통계 복사
                for field in leader_fields:
                    source_col = f'{source_prefix}_{field}'
                    target_col = f'{team_type}_{field}'
                    if source_col in latest_game:
                        result_df.loc[idx, target_col] = latest_game[source_col]
                
                # 휴식일 계산
                upcoming_date = upcoming_game['date'].date()  # 시간은 무시하고 날짜만 사용
                latest_date = latest_game['date'].date()  # 시간은 무시하고 날짜만 사용
                rest_days = (upcoming_date - latest_date).days - 1
                rest_days = max(0, rest_days)  # 음수가 나오지 않도록
                result_df.loc[idx, f'{team_type}_rest_days'] = rest_days
            
            # 2. 상대전적 정보 추가
            home_id = upcoming_game['home_team_id']
            away_id = upcoming_game['away_team_id']
            
            # 두 팀 간의 모든 경기 찾기
            matchups = features_df[
                ((features_df['home_team_id'] == home_id) & (features_df['away_team_id'] == away_id)) |
                ((features_df['home_team_id'] == away_id) & (features_df['away_team_id'] == home_id))
            ]
            
            if len(matchups) > 0:
                # 가장 최근 매치업 가져오기
                latest_matchup = matchups.iloc[-1]
                
                # 최근 매치업에서 홈/어웨이가 현재와 같은 경우
                if (latest_matchup['home_team_id'] == home_id and 
                    latest_matchup['away_team_id'] == away_id):
                    result_df.loc[idx, 'home_vs_away_wins'] = latest_matchup['home_vs_away_wins']
                    result_df.loc[idx, 'home_vs_away_losses'] = latest_matchup['home_vs_away_losses']
                    result_df.loc[idx, 'home_vs_away_win_rate'] = latest_matchup['home_vs_away_win_rate']
                # 최근 매치업에서 홈/어웨이가 반대인 경우
                else:
                    result_df.loc[idx, 'home_vs_away_wins'] = latest_matchup['home_vs_away_losses']
                    result_df.loc[idx, 'home_vs_away_losses'] = latest_matchup['home_vs_away_wins']
                    result_df.loc[idx, 'home_vs_away_win_rate'] = (
                        1 - latest_matchup['home_vs_away_win_rate']
                        if latest_matchup['home_vs_away_win_rate'] != 0.5
                        else 0.5
                    )
            else:
                # 상대전적이 없는 경우 기본값 설정
                result_df.loc[idx, 'home_vs_away_wins'] = 0
                result_df.loc[idx, 'home_vs_away_losses'] = 0
                result_df.loc[idx, 'home_vs_away_win_rate'] = 0.5
        
        return result_df


# 테스트 코드
if __name__ == "__main__":
    
    
    # 모든 경기 통계 추출
    processor = DataProcessor()
    data = processor.load_latest_data()
    games_df = processor.process_game_data(data)
    features_df = processor.extract_features(games_df, data)
    
    print("\n=== 추출된 특성 미리보기 ===")
    print(features_df.head())
    print("\n=== 수치형 특성 목록 ===")
    print(features_df.select_dtypes(include=['float64', 'int64', 'Int64']).columns.tolist())
    
    # 처리된 데이터 저장
    output_path = processor.save_processed_data(features_df)
    
    
    
    
    
    
    # 예정된 경기 기본틀 추출
    processor_upcoming = DataProcessor_upcoming()
    collector = ESPNNBADataCollector()
    
    
    # 1-2. 예정된 경기 데이터 수집
    upcoming_data = collector.collect_upcoming_data(days_ahead=1)
    # # 업커밍 데이터 로드
    # upcoming_data = processor_upcoming.load_latest_data('upcoming')

    # STATUS_SCHEDULED인 경기만 필터링
    upcoming_data['games'] = [game for game in upcoming_data['games'] 
                            if game['status'] == 'STATUS_SCHEDULED']
    
    # 기본 정보만 추출
    upcoming_df = processor_upcoming.process_game_data(upcoming_data)
    upcoming_features = processor_upcoming.extract_features(upcoming_df, upcoming_data)

    # 필요한 기본 피처들만 선택
    selected_features_upcoming = [
    'game_id', 'date', 'season_year', 'season_type', 'status',
    'home_team_id', 'home_team_name', 'home_team_score',
        'away_team_id', 'away_team_name', 'away_team_score'
    ]
    upcoming_features = upcoming_features[selected_features_upcoming]

    print("\n=== 업커밍 경기 기본 정보 ===")
    print(upcoming_features)
    
    # 필요하다면 저장
    # processor_upcoming.save_processed_data(upcoming_features, "upcoming_base_features")
    
    
    
    
    
    
    # 예정된 경기에 최신 통계 추가
    model_input_features = processor.add_latest_team_stats(upcoming_features, features_df)
    
    print("\n=== 업데이트된 업커밍 경기 정보 ===")
    print(model_input_features)
    
    # 저장
    processor_upcoming.save_processed_data(model_input_features, "model_input_features")
    
    
    
    



            
    # print("\n=== 주요 팀들의 첫 10경기와 마지막 10경기 실제 통계 확인 (로데이터) ===")

    # # 확인할 팀 ID와 이름
    # teams_to_check = [
    #     ('8', 'Pistons'),
    #     ('9', 'Warriors'),
    #     ('12', 'Clippers'),
    #     ('13', 'Lakers'),
    #     ('2', 'Celtics'),
    #     ('14', 'Heat'),
    #     ('7', 'Nuggets')
    # ]

    # # 확인할 통계들
    # stats_to_check = [
    #     ('rebounds', '리바운드'),
    #     ('freeThrowsMade', '자유투 성공'),
    #     ('threePointFieldGoalsMade', '3점슛 성공')
    # ]

    # def print_game_stats(game, team_id):
    #     date = game['date']
    #     is_home = game['home_team']['id'] == team_id
    #     team_data = game['home_team'] if is_home else game['away_team']
    #     opponent = game['away_team']['name'] if is_home else game['home_team']['name']
    #     location_str = "vs" if is_home else "@"
        
    #     print(f"\n{date} {location_str} {opponent}:")
    #     # 일반 통계
    #     for stat_name, stat_label in stats_to_check:
    #         try:
    #             value = next(
    #                 stat['displayValue'] for stat in team_data.get('statistics', [])
    #                 if stat['name'] == stat_name
    #             )
    #             print(f"- {stat_label}: {value}")
    #         except StopIteration:
    #             print(f"- {stat_label}: 통계 없음")
        
    #     # 리더 통계
    #     try:
    #         rebounds_leader = next(
    #             leader['leaders'][0]['displayValue'].split(' ')[0]
    #             for leader in team_data.get('leaders', [])
    #             if leader['name'] == 'rebounds' and leader.get('leaders')
    #         )
    #         print(f"- 리바운드 리더: {rebounds_leader}")
    #     except StopIteration:
    #         print(f"- 리바운드 리더: 통계 없음")

    # # 날짜순으로 정렬
    # sorted_games = sorted(data['games'], key=lambda x: x['date'])

    # for team_id, team_name in teams_to_check:
    #     print(f"\n\n=== {team_name}의 통계 기록 ===")
    #     team_games = [
    #         game for game in sorted_games 
    #         if (game['home_team']['id'] == team_id or game['away_team']['id'] == team_id)
    #         and game['status'] == 'STATUS_FINAL'
    #     ]
        
    #     print("\n첫 10경기:")
    #     for game in team_games[:10]:
    #         print_game_stats(game, team_id)

    #     print("\n마지막 10경기:")
    #     for game in team_games[-10:]:
    #         print_game_stats(game, team_id)