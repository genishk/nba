#!/usr/bin/env python3
"""
NBA Spread Odds & Results Matcher
- Merged odds (moneyline + spreads)와 실제 경기 결과 매칭
- Moneyline 승패 판정
- Spread 승패 판정 (Favorites: -12.5~-2.5, Underdogs: +2.5~+12.5, 총 22개 구간)
- ROI 분석을 위한 데이터 준비
"""

import json
import logging
import pytz
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class NBASpreadResultsMatcher:
    """NBA Spread 배당률과 경기 결과 매칭 클래스"""
    
    # ESPN API 팀명 (별명) → 약어 매핑
    ESPN_TEAM_NAME_TO_ABBREV = {
        'Hawks': 'ATL',
        'Celtics': 'BOS',
        'Nets': 'BKN',
        'Hornets': 'CHA',
        'Bulls': 'CHI',
        'Cavaliers': 'CLE',
        'Mavericks': 'DAL',
        'Nuggets': 'DEN',
        'Pistons': 'DET',
        'Warriors': 'GSW',
        'Rockets': 'HOU',
        'Pacers': 'IND',
        'Clippers': 'LAC',
        'Lakers': 'LAL',
        'Grizzlies': 'MEM',
        'Heat': 'MIA',
        'Bucks': 'MIL',
        'Timberwolves': 'MIN',
        'Pelicans': 'NOP',
        'Knicks': 'NYK',
        'Thunder': 'OKC',
        'Magic': 'ORL',
        '76ers': 'PHI',
        'Suns': 'PHX',
        'Trail Blazers': 'POR',
        'Blazers': 'POR',
        'Kings': 'SAC',
        'Spurs': 'SAS',
        'Raptors': 'TOR',
        'Jazz': 'UTA',
        'Wizards': 'WAS'
    }
    
    def __init__(self):
        """초기화"""
        self.project_root = Path(__file__).parent
        self.merged_odds_dir = self.project_root / "data" / "merged_odds"
        self.records_dir = self.project_root / "src" / "data"
        self.output_dir = self.project_root / "data" / "spread_matched"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 시간대 설정
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        # 로깅 설정
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("NBASpreadResultsMatcher")
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
    
    def find_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """디렉토리에서 가장 최신 파일 찾기"""
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)
    
    def load_merged_odds(self, odds_file: Optional[Path] = None) -> Dict[Tuple[str, str, str], Dict]:
        """
        통합 배당률 데이터 로드 및 딕셔너리 생성
        
        Args:
            odds_file: 배당률 파일 경로 (None이면 최신 파일 사용)
            
        Returns:
            {(date, home_team, away_team): odds_data}
        """
        if odds_file is None:
            odds_file = self.find_latest_file(self.merged_odds_dir, "nba_merged_odds_*.json")
        
        if not odds_file or not odds_file.exists():
            self.logger.error(f"❌ Merged odds file not found")
            return {}
        
        self.logger.info(f"📂 Loading merged odds from: {odds_file.name}")
        
        with open(odds_file, 'r', encoding='utf-8') as f:
            odds_list = json.load(f)
        
        # 딕셔너리로 변환 (날짜 + 팀 조합을 키로)
        odds_dict = {}
        for odds in odds_list:
            key = (odds['date'], odds['home_team'], odds['away_team'])
            odds_dict[key] = odds
        
        self.logger.info(f"✅ Loaded {len(odds_dict)} merged odds records")
        return odds_dict
    
    def load_game_results(self, records_file: Optional[Path] = None) -> Dict[Tuple[str, str, str], Dict]:
        """
        경기 결과 데이터 로드 및 딕셔너리 생성
        ⚠️ Spread 분석용 파일(processed_spread_*.json) 우선 사용
        
        Args:
            records_file: 결과 파일 경로 (None이면 최신 파일 사용)
            
        Returns:
            {(date, home_team, away_team): result_data}
        """
        if records_file is None:
            # Spread 분석용 파일 우선 탐색 (processed_spread_*.json)
            records_file = self.find_latest_file(self.records_dir, "processed_spread_*.json")
            
            # Spread용 파일이 없으면 일반 파일 사용 (fallback)
            if not records_file:
                self.logger.warning("⚠️ Spread용 파일(processed_spread_*.json) 없음. 일반 파일 사용.")
                records_file = self.find_latest_file(self.records_dir, "processed_*.json")
                # prediction 파일 제외
                while records_file and 'prediction' in records_file.name:
                    files = sorted(self.records_dir.glob("processed_*.json"), 
                                 key=lambda x: x.stat().st_mtime, reverse=True)
                    records_file = None
                    for f in files:
                        if 'prediction' not in f.name:
                            records_file = f
                            break
        
        if not records_file or not records_file.exists():
            self.logger.error(f"❌ Game results file not found")
            return {}
        
        self.logger.info(f"📂 Loading game results from: {records_file.name}")
        
        with open(records_file, 'r', encoding='utf-8') as f:
            results_list = json.load(f)
        
        # 딕셔너리로 변환
        results_dict = {}
        
        for result in results_list:
            # 날짜 변환 (UTC → ET)
            date_utc_str = result.get('date')
            if date_utc_str:
                try:
                    date_utc = datetime.fromisoformat(date_utc_str.replace('Z', '+00:00'))
                    date_et = date_utc.astimezone(self.eastern_tz)
                    date_str = date_et.strftime('%Y-%m-%d')
                except:
                    continue
            else:
                continue
            
            # 팀명 변환
            home_team_name = result.get('home_team_name', '')
            away_team_name = result.get('away_team_name', '')
            
            home_team = self.ESPN_TEAM_NAME_TO_ABBREV.get(home_team_name, home_team_name)
            away_team = self.ESPN_TEAM_NAME_TO_ABBREV.get(away_team_name, away_team_name)
            
            # 점수
            home_score = result.get('home_team_score')
            away_score = result.get('away_team_score')
            
            if home_score is None or away_score is None:
                continue
            
            key = (date_str, home_team, away_team)
            results_dict[key] = {
                'date': date_str,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'score_diff': home_score - away_score,  # 홈팀 기준 점수차
                'winner': home_team if home_score > away_score else away_team
            }
        
        self.logger.info(f"✅ Loaded {len(results_dict)} game results")
        return results_dict
    
    def calculate_spread_result(self, team: str, is_home: bool, score_diff: int, spread: float) -> str:
        """
        Spread 승패 판정
        
        Args:
            team: 팀 약어
            is_home: 홈팀 여부
            score_diff: 점수차 (홈팀 기준, 양수면 홈팀 승리)
            spread: 스프레드 포인트 (음수=Favorite, 양수=Underdog, 예: -5.5 또는 +5.5)
            
        Returns:
            'win', 'loss', 'push'
            
        Examples:
            - Favorite (-7.5): 홈팀이 10점 차로 이기면 → 10 + (-7.5) = 2.5 > 0 → 승리
            - Underdog (+7.5): 홈팀이 5점 차로 지면 → -5 + (+7.5) = 2.5 > 0 → 승리
        """
        # 실제 점수차 (해당 팀 기준)
        actual_diff = score_diff if is_home else -score_diff
        
        # Spread 적용 후 결과
        # Favorite (음수 spread): 큰 점수차로 이겨야 승리
        # Underdog (양수 spread): 적게 지거나 이기면 승리
        spread_result = actual_diff + spread
        
        if spread_result > 0:
            return 'win'
        elif spread_result < 0:
            return 'loss'
        else:
            return 'push'  # 정확히 0 (드물지만 가능)
    
    def match_odds_with_results(self, odds_dict: Dict, results_dict: Dict) -> List[Dict]:
        """
        배당률과 경기 결과 매칭 및 승패 판정
        
        Args:
            odds_dict: 배당률 딕셔너리
            results_dict: 경기 결과 딕셔너리
            
        Returns:
            매칭된 데이터 리스트
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🔗 Matching odds with results...")
        self.logger.info("=" * 70)
        
        matched_data = []
        matched_count = 0
        unmatched_count = 0
        
        for key, odds in odds_dict.items():
            result = results_dict.get(key)
            
            if result:
                matched_count += 1
                
                # 기본 정보
                matched_record = {
                    # 경기 정보
                    'game_id': odds['game_id'],
                    'date': odds['date'],
                    'commence_time_et': odds['commence_time_et'],
                    'home_team': odds['home_team'],
                    'away_team': odds['away_team'],
                    'home_team_full': odds['home_team_full'],
                    'away_team_full': odds['away_team_full'],
                    
                    # 경기 결과
                    'home_score': result['home_score'],
                    'away_score': result['away_score'],
                    'score_diff': result['score_diff'],
                    'winner': result['winner'],
                    
                    # Moneyline odds
                    'home_odds_ml': odds['home_odds_ml'],
                    'away_odds_ml': odds['away_odds_ml'],
                    
                    # Moneyline 승패
                    'home_ml_result': 'win' if result['winner'] == odds['home_team'] else 'loss',
                    'away_ml_result': 'win' if result['winner'] == odds['away_team'] else 'loss',
                    
                    # Spread 데이터
                    'home_spreads': {},
                    'away_spreads': {},
                    
                    # 메타데이터
                    'bookmaker': 'fanduel',
                    'has_moneyline': odds['has_moneyline'],
                    'has_spreads': odds['has_spreads']
                }
                
                # 홈팀 Spread 승패 판정
                if odds['home_spreads']:
                    for spread_point, spread_odds in odds['home_spreads'].items():
                        spread_float = float(spread_point)
                        spread_result = self.calculate_spread_result(
                            team=odds['home_team'],
                            is_home=True,
                            score_diff=result['score_diff'],
                            spread=spread_float
                        )
                        
                        matched_record['home_spreads'][spread_point] = {
                            'odds': spread_odds,
                            'result': spread_result
                        }
                
                # 원정팀 Spread 승패 판정
                if odds['away_spreads']:
                    for spread_point, spread_odds in odds['away_spreads'].items():
                        spread_float = float(spread_point)
                        spread_result = self.calculate_spread_result(
                            team=odds['away_team'],
                            is_home=False,
                            score_diff=result['score_diff'],
                            spread=spread_float
                        )
                        
                        matched_record['away_spreads'][spread_point] = {
                            'odds': spread_odds,
                            'result': spread_result
                        }
                
                matched_data.append(matched_record)
                
            else:
                unmatched_count += 1
                self.logger.debug(f"  ⚠️  No result found for: {key}")
        
        # 통계
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📊 Matching Statistics")
        self.logger.info("=" * 70)
        self.logger.info(f"✅ Matched: {matched_count} games")
        self.logger.info(f"⚠️  Unmatched: {unmatched_count} games")
        
        if len(odds_dict) > 0:
            match_rate = matched_count / len(odds_dict) * 100
            self.logger.info(f"📈 Match rate: {match_rate:.1f}%")
        
        self.logger.info("=" * 70)
        
        return matched_data
    
    def save_matched_data(self, matched_data: List[Dict]) -> str:
        """매칭된 데이터 저장"""
        if not matched_data:
            self.logger.error("❌ No data to save")
            return ""
        
        # 날짜순 정렬
        matched_data.sort(key=lambda x: (x['date'], x['home_team']))
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"nba_spread_matched_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(matched_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n💾 Matched data saved to: {output_file}")
        
        # 샘플 데이터 출력
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📋 Sample matched data (first 2 games):")
        self.logger.info("=" * 70)
        
        for i, game in enumerate(matched_data[:2], 1):
            self.logger.info(f"\n{i}. {game['date']}: {game['home_team']} {game['home_score']} - {game['away_score']} {game['away_team']}")
            self.logger.info(f"   Score diff: {game['score_diff']:+d} (home perspective)")
            self.logger.info(f"   Winner: {game['winner']}")
            self.logger.info(f"   Moneyline: {game['home_team']} {game['home_ml_result']} ({game['home_odds_ml']:+d}), "
                           f"{game['away_team']} {game['away_ml_result']} ({game['away_odds_ml']:+d})")
            
            if game['home_spreads']:
                sample_spreads = list(game['home_spreads'].items())[:3]
                self.logger.info(f"   Home spreads (sample):")
                for spread, data in sample_spreads:
                    self.logger.info(f"      {spread}: {data['result']} @ {data['odds']:+d}")
        
        # 통계
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📈 Result Statistics:")
        self.logger.info("=" * 70)
        
        total_games = len(matched_data)
        complete_games = sum(1 for g in matched_data if g['has_moneyline'] and g['has_spreads'])
        
        # Moneyline 승률
        home_ml_wins = sum(1 for g in matched_data if g['home_ml_result'] == 'win')
        away_ml_wins = sum(1 for g in matched_data if g['away_ml_result'] == 'win')
        
        self.logger.info(f"   Total games: {total_games}")
        self.logger.info(f"   Complete games (ML + Spreads): {complete_games}")
        self.logger.info(f"   Home team ML wins: {home_ml_wins} ({home_ml_wins/total_games*100:.1f}%)")
        self.logger.info(f"   Away team ML wins: {away_ml_wins} ({away_ml_wins/total_games*100:.1f}%)")
        
        # 날짜 범위
        dates = [g['date'] for g in matched_data]
        self.logger.info(f"   Date range: {min(dates)} ~ {max(dates)}")
        
        self.logger.info("=" * 70)
        
        return str(output_file)
    
    def run(self, odds_file: Optional[Path] = None, records_file: Optional[Path] = None) -> str:
        """
        전체 매칭 프로세스 실행
        
        Args:
            odds_file: 배당률 파일 (None이면 최신 파일)
            records_file: 경기 결과 파일 (None이면 최신 파일)
            
        Returns:
            저장된 파일 경로
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🏀 NBA Spread Odds & Results Matcher")
        self.logger.info("=" * 70)
        
        # 1. 데이터 로드
        odds_dict = self.load_merged_odds(odds_file)
        if not odds_dict:
            self.logger.error("❌ Failed to load odds data")
            return ""
        
        results_dict = self.load_game_results(records_file)
        if not results_dict:
            self.logger.error("❌ Failed to load game results")
            return ""
        
        # 2. 매칭 및 승패 판정
        matched_data = self.match_odds_with_results(odds_dict, results_dict)
        
        if not matched_data:
            self.logger.error("\n❌ No matches found")
            return ""
        
        # 3. 결과 저장
        output_file = self.save_matched_data(matched_data)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("✅ Matching completed successfully!")
        self.logger.info("=" * 70)
        self.logger.info(f"📁 Output file: {output_file}")
        self.logger.info("\n💡 Next steps:")
        self.logger.info("   1. Analyze ROI by moneyline odds range")
        self.logger.info("   2. Compare moneyline vs spread ROI")
        self.logger.info("   3. Analyze by team")
        self.logger.info("=" * 70)
        
        return output_file


def main():
    """메인 실행 함수"""
    matcher = NBASpreadResultsMatcher()
    output_file = matcher.run()
    
    if output_file:
        print(f"\n✅ Success! Matched data saved to:")
        print(f"   {output_file}")
    else:
        print("\n❌ Matching failed. Please check the logs above.")


if __name__ == "__main__":
    main()

