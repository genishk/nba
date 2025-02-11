import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from collections import defaultdict

class DataProcessor_upcoming:
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
        
        # 팀 리더 정보 추가 (수치 통계만)
        for team_type, team in [('home', game['home_team']), ('away', game['away_team'])]:
            for leader in team.get('leaders', []):
                if leader.get('leaders') and leader['leaders']:
                    leader_info = leader['leaders'][0]
                    # rating은 제외하고 points, rebounds, assists만 저장
                    if leader['name'] in ['points', 'rebounds', 'assists']:
                        game_dict[f"{team_type}_leader_{leader['name']}"] = leader_info.get('displayValue', '0').split(' ')[0]
        
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
        
        # 2. 승률 계산 (현재 잘 작동 중)
        def extract_win_rate(record, default_rate=0.5):
            if pd.isna(record) or not isinstance(record, str):
                return default_rate
            try:
                wins, losses = map(int, record.split('-'))
                total_games = wins + losses
                return wins / total_games if total_games > 0 else default_rate
            except Exception as e:
                return default_rate
        
        record_types = ['overall', 'home', 'road']
        for record_type in record_types:
            for team_type in ['home', 'away']:
                col = f"{team_type}_{record_type}_record"
                if col in df.columns:
                    df[f"{col}_win_rate"] = df[col].apply(extract_win_rate)
        
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
            
            # 현재 시점까지의 상대전적을 기록 (현재 경기 제외)
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
            
            # 현재 경기 결과를 기록에 추가 (다음 경기를 위해)
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
        filename = f"{prefix}_{timestamp}.json"
        
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
        """최근 5경기 트렌드 정보 추가"""
        print("\n=== 최근 트렌드 정보 추가 ===")
        
        team_games = defaultdict(list)
        
        # 각 팀별 경기 수집
        for game in data['games']:
            game_date = pd.to_datetime(game['date'])
            
            # 홈팀 경기 추가
            home_team = game['home_team']
            team_games[home_team['id']].append({
                'date': game_date,
                'is_home': True,
                'score': f"{home_team['score']}-{game['away_team']['score']}" if game['status'] == 'STATUS_FINAL' else 'Not played'
            })
            
            # 원정팀 경기 추가
            away_team = game['away_team']
            team_games[away_team['id']].append({
                'date': game_date,
                'is_home': False,
                'score': f"{away_team['score']}-{home_team['score']}" if game['status'] == 'STATUS_FINAL' else 'Not played'
            })
        
        # 각 팀의 경기를 날짜순으로 정렬
        for team_id in team_games:
            team_games[team_id].sort(key=lambda x: x['date'])
        
        # 각 경기에 대해 양 팀의 최근 트렌드 추가
        for idx, row in df.iterrows():
            for team_type, team_id in [('home', row['home_team_id']), ('away', row['away_team_id'])]:
                games = team_games[team_id]
                game_date = pd.to_datetime(row['date'])
                
                # 현재 경기 이전의 경기들만 필터링
                previous_games = [g for g in games if g['date'] < game_date]
                
                # 최근 5경기 통계 계산
                completed_games = [g for g in previous_games if g['score'] != 'Not played'][-10:]
                
                if completed_games:
                    wins = 0
                    total_score = 0
                    home_games = 0
                    home_wins = 0
                    away_games = 0
                    away_wins = 0
                    
                    for game in completed_games:
                        my_score, opp_score = map(int, game['score'].split('-'))
                        
                        if my_score > opp_score:
                            wins += 1
                            if game['is_home']:
                                home_wins += 1
                            else:
                                away_wins += 1
                        
                        total_score += my_score
                        
                        if game['is_home']:
                            home_games += 1
                        else:
                            away_games += 1
                    
                    # 통계 저장
                    df.loc[idx, f'{team_type}_recent_win_rate'] = wins / len(completed_games)
                    df.loc[idx, f'{team_type}_recent_avg_score'] = total_score / len(completed_games)
                    df.loc[idx, f'{team_type}_recent_home_win_rate'] = home_wins / home_games if home_games > 0 else 0.0
                    df.loc[idx, f'{team_type}_recent_away_win_rate'] = away_wins / away_games if away_games > 0 else 0.0
                else:
                    # 이전 경기가 없는 경우 기본값 설정
                    df.loc[idx, f'{team_type}_recent_win_rate'] = 0.5
                    df.loc[idx, f'{team_type}_recent_avg_score'] = df[f'{team_type}_team_score'].mean()
                    df.loc[idx, f'{team_type}_recent_home_win_rate'] = 0.5
                    df.loc[idx, f'{team_type}_recent_away_win_rate'] = 0.5
        
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
                    rest_days = (game_date - last_game).days - 1  # 경기 당일은 제외
                else:
                    rest_days = 7  # 이전 경기가 없는 경우 (시즌 첫 경기 등)
                
                df.loc[idx, f'{team_type}_rest_days'] = rest_days
        
        return df
    
    def _add_recent_stats_average(self, df: pd.DataFrame, data: Dict) -> pd.DataFrame:
        """각 팀의 최근 10경기 통계 평균 계산"""
        print("\n=== 최근 10경기 통계 평균 계산 ===")
        
        # 대체할 통계 필드들
        stat_fields = [
            'rebounds', 'assists', 
            'fieldGoalsAttempted', 'fieldGoalsMade', 'fieldGoalPct',
            'freeThrowsAttempted', 'freeThrowsMade', 'freeThrowPct',
            'threePointFieldGoalsAttempted', 'threePointFieldGoalsMade', 'threePointPct',
            'leader_points', 'leader_rebounds', 'leader_assists'
        ]
        
        team_stats = defaultdict(lambda: defaultdict(list))
        
        # 날짜순으로 정렬된 완료된 경기들에서 통계 수집
        for game in sorted(data['games'], key=lambda x: x['date']):
            if game['status'] != 'STATUS_FINAL':
                continue
            
            for team_type, team in [('home', game['home_team']), ('away', game['away_team'])]:
                team_id = team['id']
                
                # 기본 통계
                for stat in team.get('statistics', []):
                    if isinstance(stat, dict) and stat['name'] in stat_fields:
                        value = pd.to_numeric(stat.get('displayValue', '0').rstrip('%'), errors='coerce')
                        if 'Pct' in stat['name']:
                            value = value / 100.0
                        team_stats[team_id][f"{team_type}_{stat['name']}"].append(value)
                
                # 리더 통계
                for leader in team.get('leaders', []):
                    if leader.get('leaders') and leader['leaders']:
                        if leader['name'] in ['points', 'rebounds', 'assists']:
                            value = pd.to_numeric(leader['leaders'][0].get('displayValue', '0').split(' ')[0], errors='coerce')
                            team_stats[team_id][f"{team_type}_leader_{leader['name']}"].append(value)
        
        # 각 경기에 대해 해당 시점까지의 최근 10경기 평균 계산
        for idx, row in df.iterrows():
            game_date = pd.to_datetime(row['date'])
            
            for team_type, team_id in [('home', row['home_team_id']), ('away', row['away_team_id'])]:
                # 현재 경기 이전의 통계만 사용
                for stat_name in stat_fields:
                    col_name = f"{team_type}_{stat_name}"
                    if col_name in df.columns:
                        recent_stats = team_stats[team_id][col_name][-10:]  # 최근 10경기
                        if recent_stats:
                            avg_value = np.mean(recent_stats)
                            # 컬럼이 정수형이면 반올림
                            if df[col_name].dtype in ['int64', 'Int64']:
                                avg_value = round(avg_value)
                            df.loc[idx, col_name] = avg_value
        
        return df

# 테스트 코드
if __name__ == "__main__":
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