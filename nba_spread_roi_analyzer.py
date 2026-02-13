#!/usr/bin/env python3
"""
NBA Spread ROI Analyzer
- Moneyline odds 구간별로 Moneyline vs Spread ROI 비교
- 팀별 ROI 분석
- Favorites (-12.5 ~ -2.5) 및 Underdogs (+2.5 ~ +12.5) 전략 분석
- 최적의 투자 전략 도출
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd


class NBASpreadROIAnalyzer:
    """NBA Spread ROI 분석 클래스"""
    
    # Moneyline odds 구간 정의
    ODDS_RANGES = [
        (-150, -100, "Slight Favorite (-150 to -100)"),
        (-200, -151, "Moderate Favorite (-200 to -151)"),
        (-300, -201, "Strong Favorite (-300 to -201)"),
        (-500, -301, "Heavy Favorite (-500 to -301)"),
        (-10000, -501, "Overwhelming Favorite (-501+)"),
        (100, 150, "Slight Underdog (+100 to +150)"),
        (151, 200, "Moderate Underdog (+151 to +200)"),
        (201, 300, "Strong Underdog (+201 to +300)"),
        (301, 500, "Heavy Underdog (+301 to +500)"),
        (501, 10000, "Overwhelming Underdog (+501+)")
    ]
    
    # Spread 포인트 리스트 (Favorites: 음수, Underdogs: 양수)
    SPREAD_POINTS = [
        -18.5, -17.5, -16.5, -15.5, -14.5, -13.5, -12.5, -11.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5,  # Favorites (17개)
        +2.5, +3.5, +4.5, +5.5, +6.5, +7.5, +8.5, +9.5, +10.5, +11.5, +12.5, +13.5, +14.5, +15.5, +16.5, +17.5, +18.5   # Underdogs (17개)
    ]
    
    def __init__(self):
        """초기화"""
        self.project_root = Path(__file__).parent
        self.matched_dir = self.project_root / "data" / "spread_matched"
        self.output_dir = self.project_root / "data" / "roi_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("NBASpreadROIAnalyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def find_latest_file(self, directory: Path, pattern: str):
        """디렉토리에서 가장 최신 파일 찾기"""
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)
    
    def load_matched_data(self, matched_file=None) -> List[Dict]:
        """매칭된 데이터 로드"""
        if matched_file is None:
            matched_file = self.find_latest_file(self.matched_dir, "nba_spread_matched_*.json")
        
        if not matched_file or not matched_file.exists():
            self.logger.error(f"❌ Matched data file not found")
            return []
        
        self.logger.info(f"📂 Loading matched data from: {matched_file.name}")
        
        with open(matched_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"✅ Loaded {len(data)} matched games")
        return data
    
    def american_to_decimal(self, american_odds: int) -> float:
        """아메리칸 배당률을 소수점 배당률로 변환"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def calculate_roi(self, wins: int, losses: int, avg_odds: float) -> Dict:
        """
        ROI 계산
        
        Args:
            wins: 승리 횟수
            losses: 패배 횟수
            avg_odds: 평균 배당률 (decimal)
            
        Returns:
            ROI 통계 딕셔너리
        """
        total_bets = wins + losses
        if total_bets == 0:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_staked': 0.0,
                'total_return': 0.0,
                'profit': 0.0,
                'roi': 0.0
            }
        
        # 각 베팅에 $100 투자 가정
        stake_per_bet = 100
        total_staked = total_bets * stake_per_bet
        
        # 총 수익 (승리한 베팅의 수익만)
        total_return = wins * stake_per_bet * avg_odds
        
        # 순이익
        profit = total_return - total_staked
        
        # ROI (%)
        roi = (profit / total_staked) * 100 if total_staked > 0 else 0
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / total_bets) * 100,
            'total_staked': total_staked,
            'total_return': total_return,
            'profit': profit,
            'roi': roi,
            'avg_odds_decimal': avg_odds
        }
    
    def get_odds_range_label(self, odds: int) -> str:
        """배당률이 속한 구간 레이블 반환"""
        for min_odds, max_odds, label in self.ODDS_RANGES:
            if min_odds <= odds <= max_odds:
                return label
        return "Unknown"
    
    def analyze_by_odds_range(self, matched_data: List[Dict]) -> Dict:
        """
        Moneyline odds 구간별 ROI 분석
        
        Returns:
            {
                'range_label': {
                    'moneyline': {...},
                    'spreads': {
                        '-2.5': {...},
                        '-3.5': {...},
                        ...
                    }
                }
            }
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📊 Analyzing ROI by Moneyline Odds Range...")
        self.logger.info("=" * 70)
        
        # 구간별 데이터 수집
        range_data = defaultdict(lambda: {
            'moneyline': {'wins': 0, 'losses': 0, 'odds_sum': 0, 'count': 0},
            'spreads': defaultdict(lambda: {'wins': 0, 'losses': 0, 'odds_sum': 0, 'count': 0})
        })
        
        for game in matched_data:
            # 홈팀과 원정팀 모두 분석
            for team_type in ['home', 'away']:
                if team_type == 'home':
                    ml_odds = game['home_odds_ml']
                    ml_result = game['home_ml_result']
                    spreads_data = game.get('home_spreads', {})
                else:
                    ml_odds = game['away_odds_ml']
                    ml_result = game['away_ml_result']
                    spreads_data = game.get('away_spreads', {})
                
                # Odds 구간 결정
                range_label = self.get_odds_range_label(ml_odds)
                
                # Moneyline ROI 데이터
                if ml_result == 'win':
                    range_data[range_label]['moneyline']['wins'] += 1
                else:
                    range_data[range_label]['moneyline']['losses'] += 1
                
                range_data[range_label]['moneyline']['odds_sum'] += self.american_to_decimal(ml_odds)
                range_data[range_label]['moneyline']['count'] += 1
                
                # Spread ROI 데이터
                for spread_point_str, spread_info in spreads_data.items():
                    spread_point = float(spread_point_str)
                    
                    # 우리가 관심있는 spread 범위만
                    if spread_point not in self.SPREAD_POINTS:
                        continue
                    
                    spread_result = spread_info['result']
                    spread_odds = spread_info['odds']
                    
                    if spread_result == 'win':
                        range_data[range_label]['spreads'][spread_point_str]['wins'] += 1
                    elif spread_result == 'loss':
                        range_data[range_label]['spreads'][spread_point_str]['losses'] += 1
                    # 'push'는 제외
                    
                    range_data[range_label]['spreads'][spread_point_str]['odds_sum'] += self.american_to_decimal(spread_odds)
                    range_data[range_label]['spreads'][spread_point_str]['count'] += 1
        
        # ROI 계산
        analysis_results = {}
        
        for range_label, data in range_data.items():
            # Moneyline ROI
            ml_data = data['moneyline']
            avg_ml_odds = ml_data['odds_sum'] / ml_data['count'] if ml_data['count'] > 0 else 0
            ml_roi = self.calculate_roi(ml_data['wins'], ml_data['losses'], avg_ml_odds)
            
            # Spread ROI
            spread_rois = {}
            for spread_point_str, spread_data in data['spreads'].items():
                avg_spread_odds = spread_data['odds_sum'] / spread_data['count'] if spread_data['count'] > 0 else 0
                spread_roi = self.calculate_roi(spread_data['wins'], spread_data['losses'], avg_spread_odds)
                spread_rois[spread_point_str] = spread_roi
            
            analysis_results[range_label] = {
                'moneyline': ml_roi,
                'spreads': spread_rois
            }
        
        return analysis_results
    
    def analyze_by_team(self, matched_data: List[Dict]) -> Dict:
        """
        팀별 ROI 분석
        
        Returns:
            {
                'team_abbrev': {
                    'moneyline': {...},
                    'best_spread': {...}
                }
            }
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🏀 Analyzing ROI by Team...")
        self.logger.info("=" * 70)
        
        team_data = defaultdict(lambda: {
            'moneyline': {'wins': 0, 'losses': 0, 'odds_sum': 0, 'count': 0},
            'spreads': defaultdict(lambda: {'wins': 0, 'losses': 0, 'odds_sum': 0, 'count': 0})
        })
        
        for game in matched_data:
            # 홈팀
            home_team = game['home_team']
            home_ml_odds = game['home_odds_ml']
            home_ml_result = game['home_ml_result']
            home_spreads = game.get('home_spreads', {})
            
            if home_ml_result == 'win':
                team_data[home_team]['moneyline']['wins'] += 1
            else:
                team_data[home_team]['moneyline']['losses'] += 1
            
            team_data[home_team]['moneyline']['odds_sum'] += self.american_to_decimal(home_ml_odds)
            team_data[home_team]['moneyline']['count'] += 1
            
            for spread_point_str, spread_info in home_spreads.items():
                if float(spread_point_str) not in self.SPREAD_POINTS:
                    continue
                
                if spread_info['result'] == 'win':
                    team_data[home_team]['spreads'][spread_point_str]['wins'] += 1
                elif spread_info['result'] == 'loss':
                    team_data[home_team]['spreads'][spread_point_str]['losses'] += 1
                
                team_data[home_team]['spreads'][spread_point_str]['odds_sum'] += self.american_to_decimal(spread_info['odds'])
                team_data[home_team]['spreads'][spread_point_str]['count'] += 1
            
            # 원정팀
            away_team = game['away_team']
            away_ml_odds = game['away_odds_ml']
            away_ml_result = game['away_ml_result']
            away_spreads = game.get('away_spreads', {})
            
            if away_ml_result == 'win':
                team_data[away_team]['moneyline']['wins'] += 1
            else:
                team_data[away_team]['moneyline']['losses'] += 1
            
            team_data[away_team]['moneyline']['odds_sum'] += self.american_to_decimal(away_ml_odds)
            team_data[away_team]['moneyline']['count'] += 1
            
            for spread_point_str, spread_info in away_spreads.items():
                if float(spread_point_str) not in self.SPREAD_POINTS:
                    continue
                
                if spread_info['result'] == 'win':
                    team_data[away_team]['spreads'][spread_point_str]['wins'] += 1
                elif spread_info['result'] == 'loss':
                    team_data[away_team]['spreads'][spread_point_str]['losses'] += 1
                
                team_data[away_team]['spreads'][spread_point_str]['odds_sum'] += self.american_to_decimal(spread_info['odds'])
                team_data[away_team]['spreads'][spread_point_str]['count'] += 1
        
        # ROI 계산
        team_analysis = {}
        
        for team, data in team_data.items():
            # Moneyline ROI
            ml_data = data['moneyline']
            avg_ml_odds = ml_data['odds_sum'] / ml_data['count'] if ml_data['count'] > 0 else 0
            ml_roi = self.calculate_roi(ml_data['wins'], ml_data['losses'], avg_ml_odds)
            
            # 모든 Spread ROI 계산
            spread_rois = {}
            for spread_point_str, spread_data in data['spreads'].items():
                avg_spread_odds = spread_data['odds_sum'] / spread_data['count'] if spread_data['count'] > 0 else 0
                spread_roi = self.calculate_roi(spread_data['wins'], spread_data['losses'], avg_spread_odds)
                spread_rois[spread_point_str] = spread_roi
            
            # 최고 ROI Spread 찾기
            best_spread = None
            best_roi = ml_roi['roi']
            best_type = 'moneyline'
            
            for spread_point_str, roi_data in spread_rois.items():
                if roi_data['total_bets'] >= 5 and roi_data['roi'] > best_roi:  # 최소 5번 이상 베팅
                    best_roi = roi_data['roi']
                    best_spread = spread_point_str
                    best_type = 'spread'
            
            team_analysis[team] = {
                'moneyline': ml_roi,
                'spreads': spread_rois,
                'best_strategy': {
                    'type': best_type,
                    'spread': best_spread,
                    'roi': best_roi
                }
            }
        
        return team_analysis
    
    def save_analysis(self, range_analysis: Dict, team_analysis: Dict) -> str:
        """분석 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"nba_spread_roi_analysis_{timestamp}.json"
        
        analysis_data = {
            'timestamp': timestamp,
            'by_odds_range': range_analysis,
            'by_team': team_analysis
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n💾 Analysis saved to: {output_file}")
        return str(output_file)
    
    def print_summary(self, range_analysis: Dict, team_analysis: Dict):
        """분석 결과 요약 출력"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📈 ROI ANALYSIS SUMMARY")
        self.logger.info("=" * 70)
        
        # 1. Odds Range별 요약
        self.logger.info("\n🎯 Best Strategy by Moneyline Odds Range:")
        self.logger.info("-" * 70)
        
        for range_label in sorted(range_analysis.keys()):
            data = range_analysis[range_label]
            ml_roi = data['moneyline']
            
            if ml_roi['total_bets'] == 0:
                continue
            
            # 최고 ROI spread 찾기
            best_spread = None
            best_spread_roi = ml_roi['roi']
            
            for spread_point, spread_roi in data['spreads'].items():
                if spread_roi['total_bets'] >= 5 and spread_roi['roi'] > best_spread_roi:
                    best_spread = spread_point
                    best_spread_roi = spread_roi['roi']
            
            self.logger.info(f"\n📊 {range_label}")
            self.logger.info(f"   Moneyline: {ml_roi['total_bets']} bets, {ml_roi['win_rate']:.1f}% win, ROI: {ml_roi['roi']:.2f}%")
            
            if best_spread:
                spread_roi = data['spreads'][best_spread]
                self.logger.info(f"   ✅ BEST: Spread {best_spread} - {spread_roi['total_bets']} bets, "
                               f"{spread_roi['win_rate']:.1f}% win, ROI: {spread_roi['roi']:.2f}%")
                self.logger.info(f"   💰 Improvement: {spread_roi['roi'] - ml_roi['roi']:.2f}% over moneyline")
            else:
                self.logger.info(f"   ✅ BEST: Moneyline (no better spread found)")
        
        # 2. 팀별 Top 5
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🏆 Top 5 Teams by Best ROI:")
        self.logger.info("-" * 70)
        
        team_rois = [(team, data['best_strategy']['roi']) for team, data in team_analysis.items()]
        team_rois.sort(key=lambda x: x[1], reverse=True)
        
        for i, (team, roi) in enumerate(team_rois[:5], 1):
            data = team_analysis[team]
            strategy = data['best_strategy']
            
            if strategy['type'] == 'moneyline':
                self.logger.info(f"{i}. {team}: {roi:.2f}% ROI (Moneyline)")
            else:
                self.logger.info(f"{i}. {team}: {roi:.2f}% ROI (Spread {strategy['spread']})")
        
        self.logger.info("\n" + "=" * 70)
    
    def run(self, matched_file=None) -> str:
        """전체 분석 프로세스 실행"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🏀 NBA Spread ROI Analyzer")
        self.logger.info("=" * 70)
        
        # 1. 데이터 로드
        matched_data = self.load_matched_data(matched_file)
        if not matched_data:
            self.logger.error("❌ No data to analyze")
            return ""
        
        # 2. Odds Range별 분석
        range_analysis = self.analyze_by_odds_range(matched_data)
        
        # 3. 팀별 분석
        team_analysis = self.analyze_by_team(matched_data)
        
        # 4. 결과 저장
        output_file = self.save_analysis(range_analysis, team_analysis)
        
        # 5. 요약 출력
        self.print_summary(range_analysis, team_analysis)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("✅ Analysis completed successfully!")
        self.logger.info("=" * 70)
        self.logger.info(f"📁 Output file: {output_file}")
        self.logger.info("\n💡 Next step: Create dashboard for visualization")
        self.logger.info("=" * 70)
        
        return output_file


def main():
    """메인 실행 함수"""
    analyzer = NBASpreadROIAnalyzer()
    output_file = analyzer.run()
    
    if output_file:
        print(f"\n✅ Success! Analysis saved to:")
        print(f"   {output_file}")
    else:
        print("\n❌ Analysis failed. Please check the logs above.")


if __name__ == "__main__":
    main()

