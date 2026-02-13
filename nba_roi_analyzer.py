#!/usr/bin/env python3
"""
NBA ROI Analyzer
- 매칭된 odds/results 데이터에서 팀별 ROI 계산
- 기간별 분석 (전체 시즌, 30일, 14일, 7일)
- Streamlit 대시보드용 데이터 제공
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging


class NBAROIAnalyzer:
    """NBA 팀별 ROI 분석기"""
    
    def __init__(self, matched_data_file: Optional[str] = None):
        """
        Args:
            matched_data_file: 매칭된 데이터 파일 경로 (None이면 마스터 파일 사용)
        """
        self.project_root = Path(__file__).parent
        self.matched_dir = self.project_root / "data" / "matched"
        self.matched_master_file = self.matched_dir / "nba_odds_results_matched_master.json"
        
        # 데이터 로드
        if matched_data_file is None:
            matched_data_file = self._find_matched_file()
        
        self.data_file = matched_data_file
        self.df = self._load_data()
        
        # 로깅 설정
        self.logger = logging.getLogger("NBAROIAnalyzer")
    
    def _find_matched_file(self) -> Path:
        """매칭 파일 찾기 (마스터 파일 우선)"""
        # 마스터 파일 우선 사용
        if self.matched_master_file.exists():
            return self.matched_master_file
        
        # 마스터 파일 없으면 최신 파일 찾기
        files = list(self.matched_dir.glob("nba_odds_results_matched_*.json"))
        if not files:
            raise FileNotFoundError("No matched data files found")
        return max(files, key=lambda x: x.stat().st_mtime)
    
    def _load_data(self) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # 날짜를 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 정렬
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def calculate_roi(self, odds: int, won: bool) -> float:
        """
        미국식 배당률 기준 ROI 계산
        
        Args:
            odds: 미국식 배당률 (-150, +130 등)
            won: 승리 여부
            
        Returns:
            ROI (%) - 100 기준
        """
        if won:
            if odds > 0:  # 언더독
                return odds  # +130 → 130% profit
            else:  # 페이보릿
                return (100 / abs(odds)) * 100  # -150 → 66.67%
        else:
            return -100  # 전액 손실
    
    def calculate_game_rois(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 경기별로 홈/원정 팀의 ROI 계산
        
        Returns:
            홈팀과 원정팀 ROI가 추가된 DataFrame
        """
        df = df.copy()
        
        # 홈팀 ROI
        df['home_roi'] = df.apply(
            lambda row: self.calculate_roi(row['home_odds'], row['winner'] == 'home'),
            axis=1
        )
        
        # 원정팀 ROI
        df['away_roi'] = df.apply(
            lambda row: self.calculate_roi(row['away_odds'], row['winner'] == 'away'),
            axis=1
        )
        
        return df
    
    def get_period_data(self, period: str = 'season') -> pd.DataFrame:
        """
        기간별 데이터 필터링
        
        Args:
            period: 'season', '30days', '14days', '7days'
            
        Returns:
            필터링된 DataFrame
        """
        df = self.df.copy()
        
        if period == 'season':
            return df
        
        # 일수 추출
        days_map = {
            '30days': 30,
            '14days': 14,
            '7days': 7
        }
        
        days = days_map.get(period, 0)
        if days == 0:
            return df
        
        # 최근 N일 데이터만 필터링
        cutoff_date = df['date'].max() - timedelta(days=days)
        return df[df['date'] >= cutoff_date]
    
    def analyze_team(self, team: str, df: pd.DataFrame, location: str = 'all') -> Dict:
        """
        특정 팀의 ROI 분석
        
        Args:
            team: 팀 약어 (예: 'LAL')
            df: 분석할 DataFrame (ROI 계산 완료된 것)
            location: 'all', 'home', 'away'
            
        Returns:
            팀 통계 딕셔너리
        """
        # 해당 팀이 참여한 경기 필터링
        if location == 'home':
            team_games = df[df['home_team'] == team].copy()
            team_games['team_roi'] = team_games['home_roi']
            team_games['team_odds'] = team_games['home_odds']
            team_games['team_won'] = team_games['winner'] == 'home'
        elif location == 'away':
            team_games = df[df['away_team'] == team].copy()
            team_games['team_roi'] = team_games['away_roi']
            team_games['team_odds'] = team_games['away_odds']
            team_games['team_won'] = team_games['winner'] == 'away'
        else:  # all
            home_games = df[df['home_team'] == team].copy()
            home_games['team_roi'] = home_games['home_roi']
            home_games['team_odds'] = home_games['home_odds']
            home_games['team_won'] = home_games['winner'] == 'home'
            
            away_games = df[df['away_team'] == team].copy()
            away_games['team_roi'] = away_games['away_roi']
            away_games['team_odds'] = away_games['away_odds']
            away_games['team_won'] = away_games['winner'] == 'away'
            
            team_games = pd.concat([home_games, away_games]).sort_values('date')
        
        if len(team_games) == 0:
            return {
                'team': team,
                'games': 0,
                'wins': 0,
                'win_rate': 0.0,
                'total_roi': 0.0,
                'avg_roi': 0.0,
                'avg_odds': 0,
                'best_roi': 0.0,
                'worst_roi': 0.0
            }
        
        # 통계 계산
        games = len(team_games)
        wins = team_games['team_won'].sum()
        win_rate = (wins / games * 100) if games > 0 else 0
        total_roi = team_games['team_roi'].sum()
        avg_roi = team_games['team_roi'].mean()
        avg_odds = team_games['team_odds'].mean()
        best_roi = team_games['team_roi'].max()
        worst_roi = team_games['team_roi'].min()
        
        return {
            'team': team,
            'games': games,
            'wins': wins,
            'win_rate': win_rate,
            'total_roi': total_roi,
            'avg_roi': avg_roi,
            'avg_odds': avg_odds,
            'best_roi': best_roi,
            'worst_roi': worst_roi
        }
    
    def get_all_teams_analysis(self, period: str = 'season') -> pd.DataFrame:
        """
        모든 팀의 ROI 분석
        
        Args:
            period: 분석 기간
            
        Returns:
            팀별 통계 DataFrame
        """
        # 기간별 데이터 가져오기
        df = self.get_period_data(period)
        
        # ROI 계산
        df = self.calculate_game_rois(df)
        
        # 모든 팀 목록
        all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        
        # 각 팀 분석
        results = []
        for team in sorted(all_teams):
            # 전체 분석
            overall = self.analyze_team(team, df, 'all')
            
            # 홈/원정 분석
            home = self.analyze_team(team, df, 'home')
            away = self.analyze_team(team, df, 'away')
            
            results.append({
                'team': team,
                'games': overall['games'],
                'wins': overall['wins'],
                'win_rate': overall['win_rate'],
                'total_roi': overall['total_roi'],
                'avg_roi': overall['avg_roi'],
                'avg_odds': overall['avg_odds'],
                'best_roi': overall['best_roi'],
                'worst_roi': overall['worst_roi'],
                'home_games': home['games'],
                'home_roi': home['avg_roi'],
                'away_games': away['games'],
                'away_roi': away['avg_roi']
            })
        
        # DataFrame으로 변환 및 정렬
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('avg_roi', ascending=False).reset_index(drop=True)
        results_df.index = results_df.index + 1  # 1부터 시작하는 순위
        
        return results_df
    
    def get_team_detail(self, team: str, period: str = 'season') -> Dict:
        """
        특정 팀의 상세 분석
        
        Args:
            team: 팀 약어
            period: 분석 기간
            
        Returns:
            상세 통계 딕셔너리
        """
        # 기간별 데이터
        df = self.get_period_data(period)
        df = self.calculate_game_rois(df)
        
        # 전체/홈/원정 분석
        overall = self.analyze_team(team, df, 'all')
        home = self.analyze_team(team, df, 'home')
        away = self.analyze_team(team, df, 'away')
        
        # 최근 경기 이력
        home_games = df[df['home_team'] == team].copy()
        home_games['team_roi'] = home_games['home_roi']
        home_games['team_won'] = home_games['winner'] == 'home'
        home_games['opponent'] = home_games['away_team']
        home_games['location'] = 'Home'
        
        away_games = df[df['away_team'] == team].copy()
        away_games['team_roi'] = away_games['away_roi']
        away_games['team_won'] = away_games['winner'] == 'away'
        away_games['opponent'] = away_games['home_team']
        away_games['location'] = 'Away'
        
        recent_games = pd.concat([home_games, away_games]).sort_values('date', ascending=False).head(10)
        
        return {
            'overall': overall,
            'home': home,
            'away': away,
            'recent_games': recent_games[['date', 'opponent', 'location', 'team_won', 'team_roi']].to_dict('records')
        }
    
    def get_roi_trend(self, team: str, period: str = 'season') -> pd.DataFrame:
        """
        팀의 누적 ROI 추세
        
        Args:
            team: 팀 약어
            period: 분석 기간
            
        Returns:
            날짜별 누적 ROI DataFrame
        """
        df = self.get_period_data(period)
        df = self.calculate_game_rois(df)
        
        # 해당 팀 경기만 필터링
        home_games = df[df['home_team'] == team].copy()
        home_games['team_roi'] = home_games['home_roi']
        
        away_games = df[df['away_team'] == team].copy()
        away_games['team_roi'] = away_games['away_roi']
        
        team_games = pd.concat([home_games, away_games]).sort_values('date')
        
        # 누적 ROI 계산
        team_games['cumulative_roi'] = team_games['team_roi'].cumsum()
        
        return team_games[['date', 'team_roi', 'cumulative_roi']]
    
    def get_composite_rankings(self, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        여러 기간의 순위를 가중 평균하여 통합 순위 생성
        
        Args:
            weights: 기간별 가중치 {'7days': 0.5, '14days': 0.3, '30days': 0.2}
            
        Returns:
            통합 순위 DataFrame
        """
        if weights is None:
            weights = {
                '7days': 0.5,
                '14days': 0.3,
                '30days': 0.2
            }
        
        # 각 기간별 분석 결과 가져오기
        periods = ['7days', '14days', '30days']
        period_rankings = {}
        
        for period in periods:
            df = self.get_all_teams_analysis(period)
            # avg_roi 기준으로 순위 매기기 (높을수록 좋음)
            df = df.sort_values('avg_roi', ascending=False).reset_index(drop=True)
            df['rank'] = range(1, len(df) + 1)
            period_rankings[period] = df[['team', 'rank', 'avg_roi', 'games']].copy()
            period_rankings[period].columns = ['team', f'{period}_rank', f'{period}_roi', f'{period}_games']
        
        # 모든 기간 데이터 병합
        composite = period_rankings['7days']
        for period in ['14days', '30days']:
            composite = composite.merge(period_rankings[period], on='team', how='outer')
        
        # 결측치 처리 (경기가 없는 경우 최하위 순위로)
        max_rank = len(composite) + 1
        for period in periods:
            composite[f'{period}_rank'].fillna(max_rank, inplace=True)
        
        # Composite Score 계산 (순위의 가중 평균, 낮을수록 좋음)
        composite['composite_score'] = (
            composite['7days_rank'] * weights['7days'] +
            composite['14days_rank'] * weights['14days'] +
            composite['30days_rank'] * weights['30days']
        )
        
        # Composite Score로 정렬
        composite = composite.sort_values('composite_score').reset_index(drop=True)
        composite['composite_rank'] = range(1, len(composite) + 1)
        
        # 트렌드 계산 (7일 ROI vs 14일/30일 ROI 비교)
        composite['trend'] = composite.apply(
            lambda row: '🔥' if row['7days_roi'] > row['14days_roi'] and row['7days_roi'] > row['30days_roi'] else
                       '↗️' if row['7days_roi'] > row['30days_roi'] else
                       '→' if abs(row['7days_roi'] - row['30days_roi']) < 5 else
                       '↘️',
            axis=1
        )
        
        # 컬럼 순서 정리
        result_columns = [
            'composite_rank', 'team', 'composite_score', 'trend',
            '7days_rank', '7days_roi', '7days_games',
            '14days_rank', '14days_roi', '14days_games',
            '30days_rank', '30days_roi', '30days_games'
        ]
        
        return composite[result_columns]
    
    def get_data_summary(self) -> Dict:
        """데이터 요약 정보"""
        return {
            'total_games': len(self.df),
            'date_range': {
                'start': self.df['date'].min().strftime('%Y-%m-%d'),
                'end': self.df['date'].max().strftime('%Y-%m-%d')
            },
            'total_teams': len(set(self.df['home_team'].unique()) | set(self.df['away_team'].unique())),
            'data_file': str(self.data_file)
        }


if __name__ == "__main__":
    # 테스트
    analyzer = NBAROIAnalyzer()
    
    print("=== Data Summary ===")
    summary = analyzer.get_data_summary()
    print(f"Total games: {summary['total_games']}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Total teams: {summary['total_teams']}")
    
    print("\n=== Season ROI Rankings (Top 10) ===")
    rankings = analyzer.get_all_teams_analysis('season')
    print(rankings.head(10)[['team', 'games', 'wins', 'win_rate', 'avg_roi', 'total_roi']])
    
    print("\n=== LAL Detail ===")
    lal_detail = analyzer.get_team_detail('LAL', 'season')
    print(f"Overall: {lal_detail['overall']}")

