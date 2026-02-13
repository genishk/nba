#!/usr/bin/env python3
"""
NBA Moneyline + Alternate Spreads Merger
- Moneyline odds와 Alternate spreads를 경기별로 통합
- Favorites (-12.5~-2.5) 및 Underdogs (+2.5~+12.5) 포함 (총 22개 구간)
- ROI 분석을 위한 데이터 준비
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict


class NBAOddsMerger:
    """Moneyline과 Alternate Spreads 통합 클래스"""
    
    def __init__(self):
        """초기화"""
        self.project_root = Path(__file__).parent
        
        # 입력 파일 경로
        self.moneyline_file = self.project_root / "data" / "historical_odds" / "nba_historical_odds_fanduel_master.json"
        self.spreads_file = self.project_root / "data" / "alternate_spreads" / "nba_alternate_spreads_fanduel_master.json"
        
        # 출력 디렉토리
        self.output_dir = self.project_root / "data" / "merged_odds"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("NBAOddsMerger")
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
    
    def load_moneyline_odds(self) -> Dict:
        """
        Moneyline odds 로드 및 딕셔너리 변환
        
        Returns:
            {(date, home_team, away_team): moneyline_data}
        """
        self.logger.info(f"📂 Loading moneyline odds from: {self.moneyline_file.name}")
        
        with open(self.moneyline_file, 'r', encoding='utf-8') as f:
            moneyline_list = json.load(f)
        
        # 딕셔너리로 변환 (날짜 + 팀 조합을 키로)
        moneyline_dict = {}
        for odds in moneyline_list:
            key = (odds['date'], odds['home_team'], odds['away_team'])
            moneyline_dict[key] = odds
        
        self.logger.info(f"✅ Loaded {len(moneyline_dict)} moneyline odds records")
        return moneyline_dict
    
    def load_alternate_spreads(self) -> Dict:
        """
        Alternate spreads 로드 및 경기별로 그룹화
        
        Returns:
            {(date, home_team, away_team): [spread_data1, spread_data2, ...]}
        """
        self.logger.info(f"📂 Loading alternate spreads from: {self.spreads_file.name}")
        
        with open(self.spreads_file, 'r', encoding='utf-8') as f:
            spreads_list = json.load(f)
        
        # 경기별로 그룹화
        spreads_dict = defaultdict(list)
        for spread in spreads_list:
            key = (spread['date'], spread['home_team'], spread['away_team'])
            spreads_dict[key].append(spread)
        
        self.logger.info(f"✅ Loaded {len(spreads_list)} spread options from {len(spreads_dict)} games")
        return dict(spreads_dict)
    
    def merge_odds(self, moneyline_dict: Dict, spreads_dict: Dict) -> List[Dict]:
        """
        Moneyline과 Spreads 통합
        
        Args:
            moneyline_dict: Moneyline odds 딕셔너리
            spreads_dict: Alternate spreads 딕셔너리
            
        Returns:
            통합된 데이터 리스트
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🔗 Merging moneyline odds and alternate spreads...")
        self.logger.info("=" * 70)
        
        merged_data = []
        matched_count = 0
        moneyline_only_count = 0
        spreads_only_count = 0
        
        # 모든 경기 키 수집 (moneyline + spreads)
        all_keys = set(moneyline_dict.keys()) | set(spreads_dict.keys())
        
        for key in sorted(all_keys):
            date, home_team, away_team = key
            
            moneyline = moneyline_dict.get(key)
            spreads = spreads_dict.get(key, [])
            
            if moneyline and spreads:
                # 둘 다 있는 경우 (이상적)
                matched_count += 1
                
                # 홈팀과 원정팀 스프레드 분리
                home_spreads = [s for s in spreads if s['is_home']]
                away_spreads = [s for s in spreads if not s['is_home']]
                
                # 스프레드를 딕셔너리로 변환 (빠른 조회)
                home_spreads_dict = {s['spread']: s['odds'] for s in home_spreads}
                away_spreads_dict = {s['spread']: s['odds'] for s in away_spreads}
                
                merged_record = {
                    # 기본 정보
                    'game_id': moneyline['game_id'],
                    'date': date,
                    'commence_time_utc': moneyline['commence_time_utc'],
                    'commence_time_et': moneyline['commence_time_et'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_team_full': moneyline['home_team_full'],
                    'away_team_full': moneyline['away_team_full'],
                    
                    # Moneyline odds
                    'home_odds_ml': moneyline['home_odds'],
                    'away_odds_ml': moneyline['away_odds'],
                    
                    # Alternate spreads (홈팀)
                    'home_spreads': home_spreads_dict,
                    
                    # Alternate spreads (원정팀)
                    'away_spreads': away_spreads_dict,
                    
                    # 메타데이터
                    'bookmaker': 'fanduel',
                    'has_moneyline': True,
                    'has_spreads': True,
                    'num_spread_options': len(spreads)
                }
                
                merged_data.append(merged_record)
                
            elif moneyline:
                # Moneyline만 있는 경우
                moneyline_only_count += 1
                
                merged_record = {
                    'game_id': moneyline['game_id'],
                    'date': date,
                    'commence_time_utc': moneyline['commence_time_utc'],
                    'commence_time_et': moneyline['commence_time_et'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_team_full': moneyline['home_team_full'],
                    'away_team_full': moneyline['away_team_full'],
                    'home_odds_ml': moneyline['home_odds'],
                    'away_odds_ml': moneyline['away_odds'],
                    'home_spreads': {},
                    'away_spreads': {},
                    'bookmaker': 'fanduel',
                    'has_moneyline': True,
                    'has_spreads': False,
                    'num_spread_options': 0
                }
                
                merged_data.append(merged_record)
                
            else:
                # Spreads만 있는 경우 (드물지만 가능)
                spreads_only_count += 1
                # 스프레드만 있는 경우는 분석에서 제외 (moneyline 필요)
        
        # 통계 출력
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📊 Merge Statistics")
        self.logger.info("=" * 70)
        self.logger.info(f"✅ Both moneyline & spreads: {matched_count} games")
        self.logger.info(f"⚠️  Moneyline only: {moneyline_only_count} games")
        self.logger.info(f"⚠️  Spreads only: {spreads_only_count} games")
        self.logger.info(f"📊 Total merged records: {len(merged_data)} games")
        
        if matched_count > 0:
            match_rate = matched_count / len(merged_data) * 100
            self.logger.info(f"📈 Complete match rate: {match_rate:.1f}%")
        
        self.logger.info("=" * 70)
        
        return merged_data
    
    def save_merged_data(self, merged_data: List[Dict]) -> str:
        """통합 데이터 저장"""
        if not merged_data:
            self.logger.error("❌ No data to save")
            return ""
        
        # 날짜순 정렬
        merged_data.sort(key=lambda x: (x['date'], x['home_team']))
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"nba_merged_odds_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n💾 Merged data saved to: {output_file}")
        
        # 샘플 데이터 출력
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📋 Sample merged data (first 2 games):")
        self.logger.info("=" * 70)
        
        for i, game in enumerate(merged_data[:2], 1):
            self.logger.info(f"\n{i}. {game['date']}: {game['home_team']} vs {game['away_team']}")
            self.logger.info(f"   Moneyline: {game['home_odds_ml']:+d} / {game['away_odds_ml']:+d}")
            
            if game['home_spreads']:
                spreads_sample = list(game['home_spreads'].items())[:3]
                self.logger.info(f"   Home spreads (sample): {spreads_sample}")
            
            if game['away_spreads']:
                spreads_sample = list(game['away_spreads'].items())[:3]
                self.logger.info(f"   Away spreads (sample): {spreads_sample}")
        
        # 통계
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📈 Data Statistics:")
        self.logger.info("=" * 70)
        
        complete_games = sum(1 for g in merged_data if g['has_moneyline'] and g['has_spreads'])
        avg_spreads = sum(g['num_spread_options'] for g in merged_data if g['has_spreads']) / complete_games if complete_games > 0 else 0
        
        self.logger.info(f"   Total games: {len(merged_data)}")
        self.logger.info(f"   Complete games (ML + Spreads): {complete_games}")
        self.logger.info(f"   Avg spread options per game: {avg_spreads:.1f}")
        
        # 날짜 범위
        dates = [g['date'] for g in merged_data]
        self.logger.info(f"   Date range: {min(dates)} ~ {max(dates)}")
        
        self.logger.info("=" * 70)
        
        return str(output_file)
    
    def run(self) -> str:
        """전체 통합 프로세스 실행"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🏀 NBA Moneyline + Alternate Spreads Merger")
        self.logger.info("=" * 70)
        
        # 1. 데이터 로드
        moneyline_dict = self.load_moneyline_odds()
        spreads_dict = self.load_alternate_spreads()
        
        # 2. 데이터 통합
        merged_data = self.merge_odds(moneyline_dict, spreads_dict)
        
        if not merged_data:
            self.logger.error("\n❌ No data to merge")
            return ""
        
        # 3. 결과 저장
        output_file = self.save_merged_data(merged_data)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("✅ Merge completed successfully!")
        self.logger.info("=" * 70)
        self.logger.info(f"📁 Output file: {output_file}")
        self.logger.info("\n💡 Next steps:")
        self.logger.info("   1. Match with game results")
        self.logger.info("   2. Calculate spread win/loss")
        self.logger.info("   3. Analyze ROI by moneyline odds range")
        self.logger.info("=" * 70)
        
        return output_file


def main():
    """메인 실행 함수"""
    merger = NBAOddsMerger()
    output_file = merger.run()
    
    if output_file:
        print(f"\n✅ Success! Merged data saved to:")
        print(f"   {output_file}")
    else:
        print("\n❌ Merge failed. Please check the logs above.")


if __name__ == "__main__":
    main()

