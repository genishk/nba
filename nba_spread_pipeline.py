#!/usr/bin/env python3
"""
NBA Spread Analysis Pipeline
- Step 1: Alternate Spreads 수집 (Favorites: -12.5~-2.5, Underdogs: +2.5~+12.5, 총 22개 구간)
- Step 2: Moneyline과 Spreads 통합
- Step 3: 경기 결과와 매칭
- Step 4: ROI 분석
- 독립적으로 관리되는 Spread 전용 파이프라인
"""

import logging
from pathlib import Path
from datetime import datetime


class NBASpreadPipeline:
    """NBA Spread 분석 파이프라인 (독립 실행)"""
    
    def __init__(self, api_key: str, incremental: bool = True):
        """
        Args:
            api_key: The-Odds-API 키
            incremental: 증분 업데이트 모드 (True: 새 날짜만, False: 전체 재수집)
        """
        self.api_key = api_key
        self.incremental = incremental
        
        # 로깅 설정
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("NBASpreadPipeline")
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
        self.logger.info("🏀 NBA Spread Analysis Pipeline Started")
        self.logger.info("=" * 70)
        self.logger.info(f"📅 Mode: {'Incremental (new dates only)' if self.incremental else 'Full scan'}")
        self.logger.info(f"🎯 Target: Alternate Spreads (±2.5 to ±12.5, 22 options)")
        self.logger.info(f"   • Favorites: -12.5 to -2.5 (11 options)")
        self.logger.info(f"   • Underdogs: +2.5 to +12.5 (11 options)")
        self.logger.info("=" * 70)
        
        results = {
            'success': False,
            'spreads_file': None,
            'merged_file': None,
            'matched_file': None,
            'analysis_file': None,
            'spreads_collected': 0,
            'games_matched': 0,
            'error': None
        }
        
        try:
            # ============================================================
            # Step 1: Alternate Spreads 수집
            # ============================================================
            self.logger.info("\n" + "=" * 70)
            self.logger.info("📡 STEP 1: Collecting Alternate Spreads")
            self.logger.info("=" * 70)
            
            from nba_alternate_spreads_collector_v2 import NBAAlternateSpreadsCollector
            
            spreads_collector = NBAAlternateSpreadsCollector(
                api_key=self.api_key,
                days_back=2,  # 증분 모드에서는 무시됨
                incremental=self.incremental
            )
            
            spreads_file = spreads_collector.collect_alternate_spreads()
            
            if not spreads_file:
                raise Exception("Failed to collect alternate spreads")
            
            results['spreads_file'] = spreads_file
            
            # 수집된 데이터 확인
            import json
            with open(spreads_file, 'r', encoding='utf-8') as f:
                spreads_data = json.load(f)
                results['spreads_collected'] = len(spreads_data)
            
            self.logger.info("\n✅ Step 1 completed successfully")
            self.logger.info(f"📊 Collected {results['spreads_collected']} spread options")
            
            # ============================================================
            # Step 2: Moneyline과 Spreads 통합
            # ============================================================
            self.logger.info("\n" + "=" * 70)
            self.logger.info("🔗 STEP 2: Merging Moneyline and Spreads")
            self.logger.info("=" * 70)
            
            from nba_merge_moneyline_spreads import NBAOddsMerger
            
            merger = NBAOddsMerger()
            merged_file = merger.run()
            
            if not merged_file:
                raise Exception("Failed to merge moneyline and spreads")
            
            results['merged_file'] = merged_file
            
            self.logger.info("\n✅ Step 2 completed successfully")
            
            # ============================================================
            # Step 3: 경기 결과와 매칭
            # ============================================================
            self.logger.info("\n" + "=" * 70)
            self.logger.info("🎯 STEP 3: Matching with Game Results")
            self.logger.info("=" * 70)
            
            from nba_spread_results_matcher import NBASpreadResultsMatcher
            
            matcher = NBASpreadResultsMatcher()
            matched_file = matcher.run()
            
            if not matched_file:
                raise Exception("Failed to match with results")
            
            results['matched_file'] = matched_file
            
            # 매칭된 데이터 확인
            with open(matched_file, 'r', encoding='utf-8') as f:
                matched_data = json.load(f)
                results['games_matched'] = len(matched_data)
            
            self.logger.info("\n✅ Step 3 completed successfully")
            self.logger.info(f"🎯 Matched {results['games_matched']} games")
            
            # ============================================================
            # Step 4: ROI 분석
            # ============================================================
            self.logger.info("\n" + "=" * 70)
            self.logger.info("📈 STEP 4: Analyzing ROI")
            self.logger.info("=" * 70)
            
            from nba_spread_roi_analyzer import NBASpreadROIAnalyzer
            
            analyzer = NBASpreadROIAnalyzer()
            analysis_file = analyzer.run()
            
            if not analysis_file:
                raise Exception("Failed to analyze ROI")
            
            results['analysis_file'] = analysis_file
            
            self.logger.info("\n✅ Step 4 completed successfully")
            
            # ============================================================
            # 최종 결과
            # ============================================================
            results['success'] = True
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info("🎉 Pipeline Completed Successfully!")
            self.logger.info("=" * 70)
            self.logger.info(f"📊 Spread options: {results['spreads_collected']}")
            self.logger.info(f"🎯 Games matched: {results['games_matched']}")
            self.logger.info("\n📁 Output files:")
            self.logger.info(f"  1. Spreads: {results['spreads_file']}")
            self.logger.info(f"  2. Merged: {results['merged_file']}")
            self.logger.info(f"  3. Matched: {results['matched_file']}")
            self.logger.info(f"  4. Analysis: {results['analysis_file']}")
            self.logger.info("\n💡 Next step: Run dashboard with 'streamlit run nba_spread_roi_dashboard.py'")
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
    API_KEY = "3e76069f78f461b5348b7dfdf1ff5535"
    
    
    # 증분 업데이트 모드 (True: 새 날짜만, False: 전체 재수집)
    INCREMENTAL = True
    
    print("\n" + "=" * 70)
    print("🏀 NBA Spread Analysis Pipeline")
    print("=" * 70)
    print(f"📅 Mode: {'Incremental (new dates only)' if INCREMENTAL else 'Full scan'}")
    print(f"🎯 Target: Alternate Spreads + ROI Analysis (22 options)")
    print(f"   • Favorites: -12.5 to -2.5 (11 options)")
    print(f"   • Underdogs: +2.5 to +12.5 (11 options)")
    print("=" * 70)
    print("\n⚠️  Note: This pipeline is independent from team odds pipeline")
    print("=" * 70)
    
    # 파이프라인 실행
    pipeline = NBASpreadPipeline(
        api_key=API_KEY,
        incremental=INCREMENTAL
    )
    
    results = pipeline.run()
    
    # 결과 출력
    if results['success']:
        print("\n" + "=" * 70)
        print("✅ Pipeline completed successfully!")
        print("=" * 70)
        print(f"📊 Spread options: {results['spreads_collected']}")
        print(f"🎯 Games matched: {results['games_matched']}")
        print(f"\n📁 Analysis file:")
        print(f"   {results['analysis_file']}")
        print("\n💡 Run the dashboard:")
        print("   streamlit run nba_spread_roi_dashboard.py")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ Pipeline failed")
        print("=" * 70)
        print(f"Error: {results['error']}")
        print("=" * 70)


if __name__ == "__main__":
    main()

