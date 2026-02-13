#!/usr/bin/env python3
"""
NBA Odds & Results Matcher
- Historical odds 데이터와 실제 경기 결과 매칭
- 날짜 + 팀 조합으로 매칭 (game_id가 서로 다르므로)
- ROI 분석을 위한 데이터 준비
"""

import json
import logging
import pytz
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class NBAOddsResultsMatcher:
    """NBA 배당률과 경기 결과 매칭 클래스"""
    
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
        self.odds_dir = self.project_root / "data" / "historical_odds"
        self.records_dir = self.project_root / "src" / "data"
        self.matched_dir = self.project_root / "data" / "matched"
        self.matched_dir.mkdir(parents=True, exist_ok=True)
        
        # 마스터 파일 경로
        self.odds_master_file = self.odds_dir / "nba_historical_odds_fanduel_master.json"
        self.matched_master_file = self.matched_dir / "nba_odds_results_matched_master.json"
        
        # 시간대 설정
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        # 로깅 설정
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("NBAOddsResultsMatcher")
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
    
    def load_odds_data(self, odds_file: Optional[Path] = None) -> Dict[Tuple[str, str, str], Dict]:
        """
        배당률 데이터 로드 및 딕셔너리 생성
        
        Returns:
            {(date, home_team, away_team): odds_data}
        """
        if odds_file is None:
            # 마스터 파일 우선 사용
            if self.odds_master_file.exists():
                odds_file = self.odds_master_file
                self.logger.info(f"📂 Using master odds file")
            else:
                # 마스터 파일 없으면 최신 파일 찾기
                odds_file = self.find_latest_file(self.odds_dir, "nba_historical_odds_fanduel_*.json")
        
        if odds_file is None:
            self.logger.error("❌ Odds file not found")
            return {}
        
        self.logger.info(f"📂 Loading odds data: {odds_file.name}")
        
        with open(odds_file, 'r', encoding='utf-8') as f:
            odds_list = json.load(f)
        
        # 딕셔너리로 변환 (날짜 + 팀 조합을 키로)
        odds_dict = {}
        for odds in odds_list:
            key = (odds['date'], odds['home_team'], odds['away_team'])
            # 중복 키가 있으면 최신 배당률 유지 (같은 경기의 배당률 변동)
            if key not in odds_dict:
                odds_dict[key] = odds
        
        self.logger.info(f"✅ Loaded {len(odds_dict)} unique odds records")
        return odds_dict
    
    def convert_utc_to_et_date(self, utc_datetime_str: str) -> str:
        """
        UTC 시간 문자열을 ET 기준 날짜로 변환
        
        Args:
            utc_datetime_str: "2025-10-21T23:35:00.000Z"
            
        Returns:
            "2025-10-21" (ET 기준 날짜)
        """
        try:
            # UTC 시간 파싱
            utc_dt = datetime.fromisoformat(utc_datetime_str.replace('Z', '+00:00'))
            # ET로 변환
            et_dt = utc_dt.astimezone(self.eastern_tz)
            # 날짜만 반환
            return et_dt.strftime('%Y-%m-%d')
        except Exception as e:
            self.logger.warning(f"⚠️  Error converting date {utc_datetime_str}: {e}")
            return utc_datetime_str[:10]  # fallback: YYYY-MM-DD 부분만
    
    def load_records_data(self, records_file: Optional[Path] = None) -> List[Dict]:
        """
        경기 결과 데이터 로드 및 전처리
        ⚠️ Spread 분석용 파일(processed_spread_*.json) 우선 사용
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
        
        if records_file is None:
            self.logger.error("❌ Records file not found")
            return []
        
        self.logger.info(f"📂 Loading records data: {records_file.name}")
        
        with open(records_file, 'r', encoding='utf-8') as f:
            records_list = json.load(f)
        
        self.logger.info(f"✅ Loaded {len(records_list)} game records")
        
        # 데이터 전처리
        processed_records = []
        skipped_count = 0
        
        for record in records_list:
            # 완료된 경기만 처리
            if record.get('status') != 'STATUS_FINAL':
                skipped_count += 1
                continue
            
            # UTC → ET 날짜 변환
            utc_date = record.get('date', '')
            et_date = self.convert_utc_to_et_date(utc_date)
            
            # 팀명 → 약어 변환
            home_team_name = record.get('home_team_name', '')
            away_team_name = record.get('away_team_name', '')
            
            home_team_abbrev = self.ESPN_TEAM_NAME_TO_ABBREV.get(home_team_name, home_team_name)
            away_team_abbrev = self.ESPN_TEAM_NAME_TO_ABBREV.get(away_team_name, away_team_name)
            
            # 변환 실패 경고
            if home_team_abbrev == home_team_name and home_team_name:
                self.logger.warning(f"⚠️  Unknown home team name: {home_team_name}")
            if away_team_abbrev == away_team_name and away_team_name:
                self.logger.warning(f"⚠️  Unknown away team name: {away_team_name}")
            
            # 전처리된 레코드 추가
            processed_record = record.copy()
            processed_record['date_et'] = et_date
            processed_record['home_team_abbrev'] = home_team_abbrev
            processed_record['away_team_abbrev'] = away_team_abbrev
            
            processed_records.append(processed_record)
        
        self.logger.info(f"✅ Processed {len(processed_records)} final games (skipped {skipped_count} non-final)")
        return processed_records
    
    def match_odds_with_results(
        self, 
        odds_file: Optional[Path] = None,
        records_file: Optional[Path] = None
    ) -> List[Dict]:
        """
        배당률과 경기 결과 매칭
        
        Returns:
            매칭된 데이터 리스트
        """
        # 데이터 로드
        odds_dict = self.load_odds_data(odds_file)
        records_list = self.load_records_data(records_file)
        
        if not odds_dict or not records_list:
            self.logger.error("❌ Failed to load data")
            return []
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🔗 Starting matching process...")
        self.logger.info("=" * 60)
        
        matched_data = []
        unmatched_odds = 0
        unmatched_records = 0
        
        # Records를 순회하면서 Odds와 매칭
        for record in records_list:
            date = record.get('date_et')
            home_team = record.get('home_team_abbrev')
            away_team = record.get('away_team_abbrev')
            
            # 필수 필드 확인
            if not all([date, home_team, away_team]):
                continue
            
            # 매칭 키 생성
            key = (date, home_team, away_team)
            
            # Odds 데이터 찾기
            if key in odds_dict:
                odds = odds_dict[key]
                
                # 점수 및 승자 결정
                home_score = record.get('home_team_score')
                away_score = record.get('away_team_score')
                
                if home_score is not None and away_score is not None:
                    home_win = 1 if home_score > away_score else 0
                    winner = 'home' if home_win == 1 else 'away'
                else:
                    # 점수 정보 없으면 스킵
                    continue
                
                # 매칭된 데이터 생성
                matched_record = {
                    # 기본 정보
                    'date': date,
                    'home_team': home_team,
                    'away_team': away_team,
                    
                    # 배당률 정보
                    'home_odds': odds['home_odds'],
                    'away_odds': odds['away_odds'],
                    'bookmaker': odds['bookmaker'],
                    
                    # 경기 결과
                    'home_score': int(home_score),
                    'away_score': int(away_score),
                    'winner': winner,
                    'home_win': home_win,
                    
                    # 추가 정보
                    'season_year': record.get('season_year'),
                    'season_type': record.get('season_type'),
                    
                    # 원본 game_id들
                    'game_id_espn': record.get('game_id'),
                    'game_id_odds': odds.get('game_id'),
                    
                    # 시간 정보
                    'commence_time_et': odds.get('commence_time_et'),
                    'game_time_utc': record.get('date')
                }
                
                matched_data.append(matched_record)
                
                # 매칭된 odds는 딕셔너리에서 제거 (중복 방지)
                del odds_dict[key]
            else:
                unmatched_records += 1
        
        # 매칭되지 않은 odds 카운트
        unmatched_odds = len(odds_dict)
        
        # 통계 출력
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 Matching Statistics")
        self.logger.info("=" * 60)
        self.logger.info(f"✅ Successfully matched: {len(matched_data)} games")
        self.logger.info(f"⚠️  Unmatched odds: {unmatched_odds} games")
        self.logger.info(f"⚠️  Unmatched records: {unmatched_records} games")
        
        if len(matched_data) > 0:
            match_rate = len(matched_data) / (len(matched_data) + unmatched_odds) * 100
            self.logger.info(f"📈 Match rate: {match_rate:.1f}%")
        
        self.logger.info("=" * 60)
        
        # 매칭되지 않은 odds 샘플 출력 (디버깅용)
        if unmatched_odds > 0 and odds_dict:
            self.logger.info("\n📋 Sample unmatched odds (first 5):")
            for i, (key, odds) in enumerate(list(odds_dict.items())[:5], 1):
                date, home, away = key
                self.logger.info(f"  {i}. {date}: {home} vs {away}")
        
        # 매칭되지 않은 records 샘플 출력
        if unmatched_records > 0:
            self.logger.info("\n📋 Sample unmatched records (first 5):")
            unmatched_sample = []
            for record in records_list[:100]:  # 처음 100개만 확인
                date = record.get('date_et')
                home = record.get('home_team_abbrev')
                away = record.get('away_team_abbrev')
                key = (date, home, away)
                
                # 이미 매칭된 것은 제외
                if key not in [(m['date'], m['home_team'], m['away_team']) for m in matched_data]:
                    unmatched_sample.append(f"  {date}: {home} vs {away}")
                    if len(unmatched_sample) >= 5:
                        break
            
            for sample in unmatched_sample:
                self.logger.info(sample)
        
        return matched_data
    
    def save_matched_data(self, matched_data: List[Dict]) -> str:
        """매칭된 데이터 저장 (마스터 파일 + 백업)"""
        if not matched_data:
            self.logger.error("❌ No matched data to save")
            return ""
        
        # 날짜순 정렬
        matched_data.sort(key=lambda x: (x['date'], x['home_team']))
        
        # 1. 마스터 파일 저장
        with open(self.matched_master_file, 'w', encoding='utf-8') as f:
            json.dump(matched_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n💾 Master file updated: {self.matched_master_file}")
        
        # 2. 백업 파일 저장 (타임스탬프)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.matched_dir / f"nba_odds_results_matched_{timestamp}.json"
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(matched_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Backup saved to: {backup_file}")
        
        # 샘플 데이터 출력
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📋 Sample matched data (first 3 games):")
        self.logger.info("=" * 60)
        
        for i, game in enumerate(matched_data[:3], 1):
            self.logger.info(f"\n{i}. {game['date']}: {game['home_team']} vs {game['away_team']}")
            self.logger.info(f"   Odds: {game['home_odds']:+d} / {game['away_odds']:+d}")
            self.logger.info(f"   Score: {game['home_score']}-{game['away_score']}")
            self.logger.info(f"   Winner: {game['winner'].upper()}")
        
        # 날짜별 통계
        date_stats = defaultdict(int)
        for game in matched_data:
            date_stats[game['date']] += 1
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📅 Games by date:")
        self.logger.info("=" * 60)
        
        # 날짜별 통계 (처음 10개만)
        for date in sorted(date_stats.keys())[:10]:
            self.logger.info(f"  {date}: {date_stats[date]} games")
        
        if len(date_stats) > 10:
            self.logger.info(f"  ... and {len(date_stats) - 10} more dates")
        
        return str(self.matched_master_file)
    
    def run(
        self,
        odds_file: Optional[Path] = None,
        records_file: Optional[Path] = None
    ) -> str:
        """전체 매칭 프로세스 실행"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🏀 NBA Odds & Results Matcher")
        self.logger.info("=" * 60)
        
        # 1. 매칭 수행
        matched_data = self.match_odds_with_results(odds_file, records_file)
        
        if not matched_data:
            self.logger.error("\n❌ No matches found. Check your data files.")
            return ""
        
        # 2. 결과 저장
        output_file = self.save_matched_data(matched_data)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ Matching process completed!")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Total matched games: {len(matched_data)}")
        self.logger.info(f"💾 Output file: {output_file}")
        self.logger.info("\n💡 Next step: Use this file for ROI analysis")
        self.logger.info("=" * 60)
        
        return output_file


def main():
    """메인 실행 함수"""
    matcher = NBAOddsResultsMatcher()
    
    # 매칭 실행 (최신 파일 자동 선택)
    output_file = matcher.run()
    
    if output_file:
        print(f"\n✅ Success! Matched data saved to:")
        print(f"   {output_file}")
    else:
        print("\n❌ Matching failed. Please check the logs above.")


if __name__ == "__main__":
    main()

