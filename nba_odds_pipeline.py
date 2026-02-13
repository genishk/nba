#!/usr/bin/env python3
"""
NBA Historical Odds Pipeline
- Step 1: Historical odds 수집 (최근 N일)
- Step 2: Odds와 경기 결과 매칭
- 한 번에 실행되는 통합 파이프라인
"""

import logging
from pathlib import Path
from datetime import datetime
from nba_historical_odds_collector import NBAHistoricalOddsCollector
from nba_odds_results_matcher import NBAOddsResultsMatcher


class NBAOddsPipeline:
    """NBA 배당률 수집 및 매칭 파이프라인"""
    
    def __init__(self, api_key: str, days_back: int = 30, incremental: bool = True):
        """
        Args:
            api_key: The-Odds-API 키
            days_back: 과거 며칠치 수집 (기본 30일)
            incremental: 증분 업데이트 모드 (True: 새 날짜만, False: 전체 재수집)
        """
        self.api_key = api_key
        self.days_back = days_back
        self.incremental = incremental
        
        # 로깅 설정
        self.logger = self._setup_logging()
        
        # 컴포넌트 초기화
        self.odds_collector = NBAHistoricalOddsCollector(
            api_key=api_key,
            days_back=days_back,
            incremental=incremental
        )
        self.matcher = NBAOddsResultsMatcher()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("NBAOddsPipeline")
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
    
    def run(self) -> dict:
        """
        전체 파이프라인 실행
        
        Returns:
            실행 결과 딕셔너리
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🏀 NBA Historical Odds Pipeline Started")
        self.logger.info("=" * 70)
        self.logger.info(f"📅 Collection period: Last {self.days_back} days")
        self.logger.info(f"🎯 Bookmaker: FanDuel")
        self.logger.info(f"📊 Market: Moneyline (h2h)")
        self.logger.info("=" * 70)
        
        results = {
            'success': False,
            'odds_file': None,
            'matched_file': None,
            'odds_collected': 0,
            'games_matched': 0,
            'error': None
        }
        
        try:
            # ============================================================
            # Step 1: Historical Odds 수집
            # ============================================================
            self.logger.info("\n" + "=" * 70)
            self.logger.info("📡 STEP 1: Collecting Historical Odds")
            self.logger.info("=" * 70)
            
            # API 연결 테스트
            if not self.odds_collector.test_api_connection():
                raise Exception("API connection test failed")
            
            # Odds 수집
            odds_file = self.odds_collector.collect_historical_odds()
            
            if not odds_file:
                raise Exception("Failed to collect odds data")
            
            results['odds_file'] = odds_file
            
            # 수집된 데이터 확인
            import json
            with open(odds_file, 'r', encoding='utf-8') as f:
                odds_data = json.load(f)
                results['odds_collected'] = len(odds_data)
            
            self.logger.info("\n✅ Step 1 completed successfully")
            self.logger.info(f"📊 Collected {results['odds_collected']} odds records")
            
            # ============================================================
            # Step 2: Odds와 경기 결과 매칭
            # ============================================================
            self.logger.info("\n" + "=" * 70)
            self.logger.info("🔗 STEP 2: Matching Odds with Results")
            self.logger.info("=" * 70)
            
            # 방금 수집한 odds 파일 사용
            matched_file = self.matcher.run(
                odds_file=Path(odds_file),
                records_file=None  # 최신 records 파일 자동 선택
            )
            
            if not matched_file:
                raise Exception("Failed to match odds with results")
            
            results['matched_file'] = matched_file
            
            # 매칭된 데이터 확인
            with open(matched_file, 'r', encoding='utf-8') as f:
                matched_data = json.load(f)
                results['games_matched'] = len(matched_data)
            
            self.logger.info("\n✅ Step 2 completed successfully")
            self.logger.info(f"🎯 Matched {results['games_matched']} games")
            
            # ============================================================
            # 최종 결과
            # ============================================================
            results['success'] = True
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info("🎉 Pipeline Completed Successfully!")
            self.logger.info("=" * 70)
            self.logger.info(f"📊 Odds collected: {results['odds_collected']}")
            self.logger.info(f"🎯 Games matched: {results['games_matched']}")
            self.logger.info(f"📈 Match rate: {results['games_matched'] / results['odds_collected'] * 100:.1f}%")
            self.logger.info("\n📁 Output files:")
            self.logger.info(f"  1. Odds: {results['odds_file']}")
            self.logger.info(f"  2. Matched: {results['matched_file']}")
            self.logger.info("\n💡 Next step: Run dashboard with 'streamlit run nba_roi_dashboard.py'")
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error("\n" + "=" * 70)
            self.logger.error("❌ Pipeline Failed")
            self.logger.error("=" * 70)
            self.logger.error(f"Error: {str(e)}")
            self.logger.error("=" * 70)
            results['error'] = str(e)
        
        return results


def main():
    """메인 실행 함수"""
    # API 키 설정
    # API_KEY = "81fef80fc013d2c82c9a625ac1fca6b1"
    API_KEY = "96d2a1ba46ec7b941044f395d532f8cd"
    
    
    # 수집 기간 설정 (일) - 증분 모드에서는 무시되고 새 날짜만 수집
    DAYS_BACK = 30
    
    # 증분 업데이트 모드 (True: 새 날짜만, False: 전체 재수집)
    INCREMENTAL = True
    
    print("\n" + "=" * 70)
    print("🏀 NBA Historical Odds Pipeline")
    print("=" * 70)
    print(f"📅 Mode: {'Incremental (new dates only)' if INCREMENTAL else f'Full scan (last {DAYS_BACK} days)'}")
    print(f"🎯 Target: FanDuel moneyline odds + game results")
    print("=" * 70)
    
    # 파이프라인 실행
    pipeline = NBAOddsPipeline(
        api_key=API_KEY,
        days_back=DAYS_BACK,
        incremental=INCREMENTAL
    )
    
    results = pipeline.run()
    
    # 결과 출력
    if results['success']:
        print("\n" + "=" * 70)
        print("✅ Pipeline completed successfully!")
        print("=" * 70)
        print(f"📊 Odds collected: {results['odds_collected']}")
        print(f"🎯 Games matched: {results['games_matched']}")
        print(f"\n📁 Matched data file:")
        print(f"   {results['matched_file']}")
        print("\n💡 Run the dashboard:")
        print("   streamlit run nba_roi_dashboard.py")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ Pipeline failed")
        print("=" * 70)
        print(f"Error: {results['error']}")
        print("=" * 70)


if __name__ == "__main__":
    main()

