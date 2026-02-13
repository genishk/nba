"""
배당 변수를 포함한 데이터 프로세서
- 기존 processor_model.py 복사본
- 배당 데이터 매칭 및 8구간 변환 추가
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from collections import defaultdict


class DataProcessorWithOdds:
    """배당 변수를 포함한 데이터 프로세서"""
    
    # 8구간 배당 구간 정의
    ODDS_BINS = [
        (-float('inf'), -400),   # 0: 압도적 페이버릿
        (-400, -250),            # 1: 큰 페이버릿
        (-250, -150),            # 2: 페이버릿
        (-150, -100),            # 3: 약간 유리
        (-100, 150),             # 4: 이븐
        (150, 250),              # 5: 언더독
        (250, 400),              # 6: 큰 언더독
        (400, float('inf'))      # 7: 압도적 언더독
    ]
    
    ODDS_BUCKET_NAMES = [
        '압도적 페이버릿', '큰 페이버릿', '페이버릿', '약간 유리',
        '이븐', '언더독', '큰 언더독', '압도적 언더독'
    ]
    
    def __init__(self, data_dir: Optional[Path] = None):
        """데이터 처리를 위한 클래스 초기화"""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self.data_dir = data_dir
        
        # 배당 마스터 파일 경로
        self.odds_master_file = Path(__file__).parent.parent.parent / "data" / "matched" / "nba_odds_results_matched_master.json"
        
        # 배당 데이터 딕셔너리 (game_id_espn -> odds_data)
        self.odds_dict = {}
    
    def load_odds_master(self) -> Dict[str, Dict]:
        """배당 마스터 파일 로드 및 game_id로 딕셔너리 생성"""
        print("\n=== 배당 마스터 파일 로드 ===")
        
        if not self.odds_master_file.exists():
            print(f"⚠️ 배당 마스터 파일을 찾을 수 없습니다: {self.odds_master_file}")
            return {}
        
        with open(self.odds_master_file, 'r', encoding='utf-8') as f:
            odds_data = json.load(f)
        
        # game_id_espn을 키로 하는 딕셔너리 생성
        self.odds_dict = {}
        for record in odds_data:
            game_id = record.get('game_id_espn')
            if game_id:
                self.odds_dict[game_id] = {
                    'home_odds': record['home_odds'],
                    'away_odds': record['away_odds'],
                    'date': record['date']
                }
        
        print(f"✅ 배당 데이터 로드 완료: {len(self.odds_dict)}개 경기")
        return self.odds_dict
    
    def odds_to_bucket(self, odds: int) -> int:
        """American odds를 8구간 버킷으로 변환"""
        for i, (low, high) in enumerate(self.ODDS_BINS):
            if low < odds <= high:
                return i
        return 4  # 기본값: 이븐
    
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
    
    def add_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """배당 변수 추가 (game_id로 매칭)"""
        print("\n=== 배당 변수 추가 ===")
        
        if not self.odds_dict:
            self.load_odds_master()
        
        matched_count = 0
        unmatched_count = 0
        
        # 배당 컬럼 초기화
        df['home_odds'] = np.nan
        df['away_odds'] = np.nan
        df['home_odds_bucket'] = np.nan
        df['away_odds_bucket'] = np.nan
        
        for idx, row in df.iterrows():
            game_id = str(row['game_id'])
            
            if game_id in self.odds_dict:
                odds_data = self.odds_dict[game_id]
                
                # 원본 배당값 저장 (참고용)
                df.loc[idx, 'home_odds'] = odds_data['home_odds']
                df.loc[idx, 'away_odds'] = odds_data['away_odds']
                
                # 8구간 버킷으로 변환
                df.loc[idx, 'home_odds_bucket'] = self.odds_to_bucket(odds_data['home_odds'])
                df.loc[idx, 'away_odds_bucket'] = self.odds_to_bucket(odds_data['away_odds'])
                
                matched_count += 1
            else:
                unmatched_count += 1
        
        print(f"✅ 배당 매칭 완료:")
        print(f"  - 매칭 성공: {matched_count}개 ({matched_count/(matched_count+unmatched_count)*100:.1f}%)")
        print(f"  - 매칭 실패: {unmatched_count}개")
        
        # 매칭 안된 경기 제거 (배당 없는 경기는 학습에서 제외)
        before_count = len(df)
        df = df.dropna(subset=['home_odds_bucket', 'away_odds_bucket'])
        after_count = len(df)
        
        print(f"  - 배당 없는 경기 제거: {before_count - after_count}개")
        print(f"  - 최종 경기 수: {after_count}개")
        
        # 구간별 분포 출력
        print("\n📊 배당 구간 분포:")
        for i, name in enumerate(self.ODDS_BUCKET_NAMES):
            home_count = (df['home_odds_bucket'] == i).sum()
            away_count = (df['away_odds_bucket'] == i).sum()
            print(f"  {i}: {name:<16} - 홈: {home_count:>3}개, 원정: {away_count:>3}개")
        
        return df
    
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
        
        def extract_win_rate(record, default_rate=0.5, is_home=True, record_type='overall', is_winner=None):
            if pd.isna(record) or not isinstance(record, str):
                return default_rate
            try:
                wins, losses = map(int, record.split('-'))
                
                # 현재 경기 결과를 제외해야 하는 경우:
                # 1. overall 레코드는 항상 제외
                # 2. home 레코드는 홈팀일 때만 제외
                # 3. road 레코드는 원정팀일 때만 제외
                should_adjust = (
                    record_type == 'overall' or
                    (is_home and record_type == 'home') or
                    (not is_home and record_type == 'road')
                )
                
                if should_adjust and is_winner is not None:
                    if is_winner:
                        wins = max(0, wins - 1)  # 승리 기록 하나 제외
                    else:
                        losses = max(0, losses - 1)  # 패배 기록 하나 제외
                
                total_games = wins + losses
                return round(wins / total_games, 3) if total_games > 0 else default_rate
            except Exception as e:
                print(f"Error in extract_win_rate: {e}")
                return default_rate

        # 승률 계산 적용
        record_types = ['overall', 'home', 'road']
        for record_type in record_types:
            for team_type in ['home', 'away']:
                record_col = f"{team_type}_{record_type}_record"
                if record_col in df.columns:
                    df[f"{record_col}_win_rate"] = df.apply(
                        lambda row: extract_win_rate(
                            row[record_col],
                            default_rate=0.5,
                            is_home=(team_type=='home'),
                            record_type=record_type,
                            is_winner=(row['home_team_score'] > row['away_team_score'] 
                                    if team_type=='home' 
                                    else row['away_team_score'] > row['home_team_score'])
                        ),
                        axis=1
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
        
        # ★ 배당 변수 추가 (새로 추가!)
        df = self.add_odds_features(df)
        
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
    
    def save_processed_data(self, df: pd.DataFrame, prefix: str = "processed_with_odds", days_to_keep: Optional[int] = None) -> Path:
        """처리된 데이터를 JSON 파일로 저장
        
        Args:
            df: 처리된 DataFrame
            prefix: 파일명 접두사
            days_to_keep: 필터링할 최근 일수 (None인 경우 전체 데이터 저장)
        
        Returns:
            저장된 파일 경로
        """
        # 원본 DataFrame 보존
        filtered_df = df.copy()
        
        # 지정된 일수만큼 최근 데이터만 필터링
        if days_to_keep is not None:
            print(f"\n최근 {days_to_keep}일 데이터만 필터링합니다.")
            
            # 현재 날짜 계산 (UTC 시간대로 설정)
            current_date = pd.Timestamp.now(tz='UTC')
            
            # 필터링 기준 날짜 계산 (UTC 시간대 유지)
            cutoff_date = current_date - pd.Timedelta(days=days_to_keep)
            
            # 필터링 전 레코드 수
            before_count = len(filtered_df)
            
            # 데이터프레임의 date 열의 시간대 확인
            sample_date = filtered_df['date'].iloc[0] if not filtered_df.empty else None
            if sample_date is not None:
                print(f"  - 데이터 날짜 타입: {type(sample_date)}, 시간대: {getattr(sample_date, 'tz', 'naive')}")
                print(f"  - 필터링 기준 날짜: {cutoff_date}")
            
            # 필터링 적용
            filtered_df = filtered_df[filtered_df['date'] >= cutoff_date]
            
            # 필터링 결과 출력
            after_count = len(filtered_df)
            print(f"  - 필터링 전: {before_count}개 레코드")
            print(f"  - 필터링 후: {after_count}개 레코드")
            print(f"  - 제외된 레코드: {before_count - after_count}개")
            
            if after_count == 0:
                print("경고: 필터링 결과 남은 레코드가 없습니다. 전체 데이터를 저장합니다.")
                filtered_df = df.copy()
        
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        
        # processor.py와 같은 폴더에 저장
        output_path = Path(__file__).parent / filename
        
        # DataFrame을 JSON으로 변환 (날짜/시간 처리를 위해 date_format 사용)
        json_data = filtered_df.to_json(orient='records', date_format='iso')
        
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
                    if date < current_game_date
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
                        prev_5_stats = team_stats[team_id][stat_name][current_idx-5:current_idx]
                        avg_value = np.mean(prev_5_stats)
                    
                    if 'Pct' in stat_name:  # 퍼센티지는 소수점 유지
                        df.loc[idx, col_name] = avg_value
                    else:  # 나머지는 소수점 2자리까지만 유지
                        df.loc[idx, col_name] = round(avg_value, 2)
        
        return df


# 테스트 코드
if __name__ == "__main__":
    print("=" * 70)
    print("🏀 배당 변수 포함 데이터 프로세서 테스트")
    print("=" * 70)
    
    processor = DataProcessorWithOdds()
    
    # 1. 배당 마스터 파일 로드
    processor.load_odds_master()
    
    # 2. 경기 데이터 로드 및 처리
    data = processor.load_latest_data()
    games_df = processor.process_game_data(data)
    
    # 3. 특성 추출 (배당 변수 포함!)
    features_df = processor.extract_features(games_df, data)
    
    print("\n=== 추출된 특성 미리보기 ===")
    print(features_df.head())
    
    print("\n=== 배당 관련 컬럼 확인 ===")
    odds_cols = [col for col in features_df.columns if 'odds' in col.lower()]
    print(f"배당 관련 컬럼: {odds_cols}")
    if odds_cols:
        print(features_df[odds_cols].head(10))
    
    print("\n=== 수치형 특성 목록 ===")
    print(features_df.select_dtypes(include=['float64', 'int64', 'Int64']).columns.tolist())
    
    # 4. 데이터 저장
    print("\n" + "=" * 50)
    print("📊 배당 포함 모델 학습용 데이터 저장")
    print("=" * 50)
    output_path = processor.save_processed_data(features_df, prefix="processed_with_odds", days_to_keep=40)
    
    print("\n" + "=" * 50)
    print("✅ 데이터 저장 완료")
    print("=" * 50)
    print(f"  파일: {output_path}")

