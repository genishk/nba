#!/usr/bin/env python3
"""
NBA Alternate Spreads Collector v2 (with Incremental Update)
- 과거 경기의 Alternate Spreads 수집 (Favorites: -12.5 ~ -2.5, Underdogs: +2.5 ~ +12.5)
- The-Odds-API의 /events/{eventId}/odds 엔드포인트 사용
- 증분 업데이트 지원 (기존 데이터에 새 날짜만 추가)
"""

import requests
import json
import logging
import pytz
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time


class NBAAlternateSpreadsCollector:
    """NBA Alternate Spreads 수집기 (증분 업데이트 지원)"""
    
    # NBA 팀명 매핑 (전체 이름 -> 약자)
    NBA_TEAM_ABBREV = {
        'Atlanta Hawks': 'ATL',
        'Boston Celtics': 'BOS',
        'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA',
        'Chicago Bulls': 'CHI',
        'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL',
        'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW',
        'Houston Rockets': 'HOU',
        'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC',
        'Los Angeles Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL',
        'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL',
        'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK',
        'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI',
        'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC',
        'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA',
        'Washington Wizards': 'WAS'
    }
    
    # 수집할 Spread 구간 (Favorites: 음수, Underdogs: 양수)
    TARGET_SPREADS = [
        -18.5, -17.5, -16.5, -15.5, -14.5, -13.5, -12.5, -11.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5,  # Favorites (17개)
        +2.5, +3.5, +4.5, +5.5, +6.5, +7.5, +8.5, +9.5, +10.5, +11.5, +12.5, +13.5, +14.5, +15.5, +16.5, +17.5, +18.5   # Underdogs (17개)
    ]
    
    def __init__(self, api_key: str, days_back: int = 2, incremental: bool = True):
        """
        Args:
            api_key: The-Odds-API 키
            days_back: 과거 며칠치 데이터 수집 (기본 2일)
            incremental: 증분 업데이트 모드 (True: 새 날짜만, False: 전체 재수집)
        """
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_nba"
        self.days_back = days_back
        self.incremental = incremental
        
        # 시간대 설정 (동부시간)
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        # 디렉토리 설정
        self.project_root = Path(__file__).parent
        self.spreads_dir = self.project_root / "data" / "alternate_spreads"
        self.spreads_dir.mkdir(parents=True, exist_ok=True)
        
        # 마스터 파일 경로
        self.master_file = self.spreads_dir / "nba_alternate_spreads_fanduel_master.json"
        
        # 로깅 설정
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("NBAAlternateSpreadsCollector")
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
    
    def load_master_file(self) -> List[Dict]:
        """마스터 파일 로드"""
        if not self.master_file.exists():
            self.logger.info("📂 No existing master file found. Starting fresh.")
            return []
        
        try:
            with open(self.master_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"📂 Loaded {len(data)} existing spread options from master file")
            return data
        except Exception as e:
            self.logger.warning(f"⚠️  Error loading master file: {e}. Starting fresh.")
            return []
    
    def get_latest_date_from_master(self) -> Optional[str]:
        """마스터 파일에서 가장 최근 날짜 추출"""
        existing_data = self.load_master_file()
        if not existing_data:
            return None
        
        # 모든 날짜 추출 후 최신 날짜 반환
        dates = [item['date'] for item in existing_data if 'date' in item]
        if dates:
            latest = max(dates)
            self.logger.info(f"📅 Latest date in master file: {latest}")
            return latest
        return None
    
    def get_target_dates(self) -> List[str]:
        """
        오늘 기준 과거 N일의 날짜 리스트 생성 (증분 모드 고려)
        ⚠️ 중요: 어제까지만 수집 (경기 결과가 확정된 과거 데이터만 대상)
        
        Returns:
            날짜 문자열 리스트 (오래된 날짜부터) ['2025-11-11', '2025-11-12']
        """
        # 동부시간 기준으로 오늘 날짜 계산
        today_et = datetime.now(self.eastern_tz).date()
        
        if self.incremental:
            # 증분 모드: 마스터 파일의 최신 날짜 이후만 수집
            latest_date_str = self.get_latest_date_from_master()
            if latest_date_str:
                latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date()
                # 최신 날짜 다음 날부터 어제까지 (동부시간 기준)
                start_date = latest_date + timedelta(days=1)
                yesterday_et = today_et - timedelta(days=1)
                
                if start_date > yesterday_et:
                    self.logger.info("✅ Master file is up to date. No new dates to collect.")
                    return []
                
                dates = []
                current = start_date
                while current <= yesterday_et:
                    dates.append(current.strftime('%Y-%m-%d'))
                    current += timedelta(days=1)
                
                self.logger.info(f"🔄 Incremental mode: Collecting {len(dates)} new date(s)")
                return dates
        
        # 전체 수집 모드 또는 마스터 파일 없음 (동부시간 기준)
        # ⚠️ 중요: 어제까지만 수집 (경기 결과가 나온 과거 데이터만)
        yesterday_et = today_et - timedelta(days=1)
        dates = []
        for i in range(self.days_back, 0, -1):
            target_date = today_et - timedelta(days=i)
            # 어제 이전 날짜만 포함
            if target_date <= yesterday_et:
                dates.append(target_date.strftime('%Y-%m-%d'))
        
        return dates
    
    def fetch_game_list_for_date(self, date_str: str) -> Optional[List[Dict]]:
        """
        특정 날짜의 경기 리스트 가져오기 (game_id 수집용)
        
        Args:
            date_str: 날짜 문자열 (YYYY-MM-DD)
            
        Returns:
            경기 리스트 (game_id, teams, commence_time 포함)
        """
        url = f"{self.base_url}/historical/sports/{self.sport}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',  # h2h로 경기 리스트만 가져옴
            'oddsFormat': 'american',
            'date': f"{date_str}T12:00:00Z",
            'bookmakers': 'fanduel'
        }
        
        try:
            self.logger.debug(f"📡 Fetching game list for {date_str}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 구조 처리
            if isinstance(data, dict) and 'data' in data:
                games = data['data']
            elif isinstance(data, list):
                games = data
            else:
                self.logger.warning(f"⚠️  Unexpected response structure for {date_str}")
                return []
            
            # 게임 정보 추출
            game_list = []
            for game in games:
                game_list.append({
                    'id': game.get('id'),
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'commence_time': game.get('commence_time')
                })
            
            return game_list
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Error fetching game list for {date_str}: {e}")
            return None
    
    def fetch_alternate_spreads_for_game(self, game_id: str, date_str: str, 
                                        home_team_full: str, away_team_full: str,
                                        commence_time_utc: str) -> List[Dict]:
        """
        특정 경기의 Alternate Spreads 가져오기
        
        Args:
            game_id: 경기 ID
            date_str: 날짜 문자열
            home_team_full: 홈팀 전체 이름
            away_team_full: 원정팀 전체 이름
            commence_time_utc: 경기 시작 시간 (UTC)
            
        Returns:
            처리된 spread 데이터 리스트
        """
        # Historical API 사용 (과거 경기)
        url = f"{self.base_url}/historical/sports/{self.sport}/events/{game_id}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'alternate_spreads',
            'oddsFormat': 'american',
            'date': f"{date_str}T12:00:00Z",  # Historical API에는 date 파라미터 필요
            'bookmakers': 'fanduel'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 구조: {'data': {...}}
            if isinstance(data, dict) and 'data' in data:
                game_data = data['data']
            else:
                game_data = data
            
            # 팀 약어 변환
            home_team_abbrev = self.NBA_TEAM_ABBREV.get(home_team_full, home_team_full)
            away_team_abbrev = self.NBA_TEAM_ABBREV.get(away_team_full, away_team_full)
            
            # UTC → ET 변환
            if commence_time_utc:
                utc_dt = datetime.fromisoformat(commence_time_utc.replace('Z', '+00:00'))
                et_dt = utc_dt.astimezone(self.eastern_tz)
                game_date = et_dt.strftime('%Y-%m-%d')
                commence_time_et = et_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                game_date = date_str
                commence_time_et = None
            
            # FanDuel alternate spreads 추출
            processed_spreads = []
            
            bookmakers = game_data.get('bookmakers', [])
            for bookmaker in bookmakers:
                if bookmaker.get('key') != 'fanduel':
                    continue
                
                markets = bookmaker.get('markets', [])
                for market in markets:
                    if market.get('key') != 'alternate_spreads':
                        continue
                    
                    outcomes = market.get('outcomes', [])
                    
                    for outcome in outcomes:
                        team_name = outcome.get('name', '')
                        spread_point = outcome.get('point')
                        odds = outcome.get('price')
                        
                        # 원하는 spread 범위만 필터링
                        if spread_point not in self.TARGET_SPREADS:
                            continue
                        
                        # 홈팀인지 원정팀인지 판별
                        is_home = (team_name == home_team_full)
                        team_abbrev = home_team_abbrev if is_home else away_team_abbrev
                        
                        spread_record = {
                            'game_id': game_id,
                            'date': game_date,
                            'commence_time_utc': commence_time_utc,
                            'commence_time_et': commence_time_et,
                            'home_team': home_team_abbrev,
                            'away_team': away_team_abbrev,
                            'home_team_full': home_team_full,
                            'away_team_full': away_team_full,
                            'team': team_abbrev,
                            'is_home': is_home,
                            'spread': spread_point,
                            'odds': odds,
                            'bookmaker': 'fanduel'
                        }
                        
                        processed_spreads.append(spread_record)
            
            return processed_spreads
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Error fetching spreads for game {game_id}: {e}")
            return []
    
    def collect_alternate_spreads(self) -> str:
        """
        Alternate Spreads 수집 실행
        
        Returns:
            저장된 파일 경로 (실패 시 빈 문자열)
        """
        # 수집할 날짜 생성
        target_dates = self.get_target_dates()
        
        if not target_dates:
            self.logger.info("✅ No dates to collect. Master file is up to date.")
            return str(self.master_file) if self.master_file.exists() else ""
        
        self.logger.info("=" * 70)
        self.logger.info("🏀 NBA Alternate Spreads Collection Started")
        self.logger.info("=" * 70)
        self.logger.info(f"📅 Target dates: {', '.join(target_dates)}")
        self.logger.info(f"🎯 Bookmaker: FanDuel")
        self.logger.info(f"📊 Target spreads: {self.TARGET_SPREADS}")
        self.logger.info("=" * 70)
        
        all_spread_odds = []
        total_api_calls = 0
        successful_dates = 0
        failed_dates = 0
        
        for i, date_str in enumerate(target_dates, 1):
            self.logger.info(f"\n[{i}/{len(target_dates)}] Processing {date_str}...")
            
            # 1. 해당 날짜의 경기 리스트 가져오기
            game_list = self.fetch_game_list_for_date(date_str)
            total_api_calls += 1
            
            if game_list is None:
                failed_dates += 1
                self.logger.warning(f"❌ Failed to get game list for {date_str}")
                continue
            
            if not game_list:
                self.logger.info(f"📭 No games found for {date_str}")
                successful_dates += 1
                continue
            
            self.logger.info(f"✅ Found {len(game_list)} games")
            
            # 2. 각 경기의 alternate spreads 수집
            date_spreads = 0
            for j, game in enumerate(game_list, 1):
                game_id = game['id']
                home_team_full = game['home_team']
                away_team_full = game['away_team']
                commence_time_utc = game['commence_time']
                
                home_abbrev = self.NBA_TEAM_ABBREV.get(home_team_full, home_team_full)
                away_abbrev = self.NBA_TEAM_ABBREV.get(away_team_full, away_team_full)
                
                self.logger.info(f"\n  [{j}/{len(game_list)}] {home_abbrev} vs {away_abbrev}")
                self.logger.info(f"      Game ID: {game_id}")
                
                processed_spreads = self.fetch_alternate_spreads_for_game(
                    game_id, date_str, home_team_full, away_team_full, commence_time_utc
                )
                total_api_calls += 1
                
                if processed_spreads:
                    all_spread_odds.extend(processed_spreads)
                    date_spreads += len(processed_spreads)
                    self.logger.info(f"      ✅ Collected {len(processed_spreads)} spread options")
                else:
                    self.logger.warning(f"      ⚠️  No spreads found")
                
                # Rate limiting
                if j < len(game_list):
                    time.sleep(2)
            
            if date_spreads > 0:
                successful_dates += 1
                self.logger.info(f"\n✅ {date_spreads} spread options collected for {date_str}")
            else:
                failed_dates += 1
            
            # 날짜 간 rate limiting
            if i < len(target_dates):
                self.logger.info("⏳ Waiting 2 seconds...")
                time.sleep(2)
        
        # 결과 저장
        if all_spread_odds or self.incremental:
            # ⚠️ 중요: 저장 전 미래 날짜 필터링 (어제까지만 유지)
            today_et = datetime.now(self.eastern_tz).date()
            yesterday_et = today_et - timedelta(days=1)
            yesterday_str = yesterday_et.strftime('%Y-%m-%d')
            
            if self.incremental:
                # 증분 모드: 기존 데이터와 병합
                existing_data = self.load_master_file()
                
                # 기존 데이터에서도 미래 날짜 제거
                existing_data = [item for item in existing_data if item.get('date', '') <= yesterday_str]
                
                if all_spread_odds:
                    # 새 데이터에서 미래 날짜 제거
                    all_spread_odds = [item for item in all_spread_odds if item.get('date', '') <= yesterday_str]
                    
                    # 새 데이터 추가
                    combined_data = existing_data + all_spread_odds
                    
                    # 중복 제거 (game_id + date + team + spread 기준)
                    seen = set()
                    unique_data = []
                    for item in combined_data:
                        key = (item.get('game_id'), item.get('date'), 
                              item.get('team'), item.get('spread'))
                        if key not in seen:
                            seen.add(key)
                            unique_data.append(item)
                    
                    # 날짜와 팀으로 정렬
                    unique_data.sort(key=lambda x: (x['date'], x['home_team'], x['away_team'], x['team'], x['spread']))
                    
                    # 마스터 파일 저장
                    with open(self.master_file, 'w', encoding='utf-8') as f:
                        json.dump(unique_data, f, indent=2, ensure_ascii=False)
                    
                    self.logger.info("\n" + "=" * 70)
                    self.logger.info("🎉 Incremental Update Completed!")
                    self.logger.info("=" * 70)
                    self.logger.info(f"📊 New spread options collected: {len(all_spread_odds)}")
                    self.logger.info(f"📚 Total spread options in master: {len(unique_data)}")
                    self.logger.info(f"✅ Successful dates: {successful_dates}/{len(target_dates)}")
                    self.logger.info(f"❌ Failed dates: {failed_dates}/{len(target_dates)}")
                    self.logger.info(f"🔢 Total API calls: {total_api_calls}")
                    self.logger.info(f"💾 Master file updated: {self.master_file}")
                    self.logger.info("=" * 70)
                    
                    return str(self.master_file)
                else:
                    # 새 데이터 없어도 기존 데이터에서 미래 날짜 제거 후 저장
                    if existing_data:
                        existing_data.sort(key=lambda x: (x['date'], x['home_team'], x['away_team'], x['team'], x['spread']))
                        with open(self.master_file, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f, indent=2, ensure_ascii=False)
                        self.logger.info(f"✅ Master file cleaned (future dates removed): {len(existing_data)} records")
                    else:
                        self.logger.info("\n✅ No new data to add. Master file unchanged.")
                    return str(self.master_file)
            else:
                # 전체 수집 모드: 타임스탬프 파일 + 마스터 파일 업데이트
                # 미래 날짜 제거
                all_spread_odds = [item for item in all_spread_odds if item.get('date', '') <= yesterday_str]
                all_spread_odds.sort(key=lambda x: (x['date'], x['home_team'], x['away_team'], x['team'], x['spread']))
                
                # 1. 타임스탬프 파일 저장 (백업용)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = self.spreads_dir / f"nba_alternate_spreads_{timestamp}.json"
                
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(all_spread_odds, f, indent=2, ensure_ascii=False)
                
                # 2. 마스터 파일 업데이트
                with open(self.master_file, 'w', encoding='utf-8') as f:
                    json.dump(all_spread_odds, f, indent=2, ensure_ascii=False)
                
                # 요약 출력
                self.logger.info("\n" + "=" * 70)
                self.logger.info("🎉 Full Collection Completed!")
                self.logger.info("=" * 70)
                self.logger.info(f"📊 Total spread options collected: {len(all_spread_odds)}")
                self.logger.info(f"✅ Successful dates: {successful_dates}/{len(target_dates)}")
                self.logger.info(f"❌ Failed dates: {failed_dates}/{len(target_dates)}")
                self.logger.info(f"🔢 Total API calls: {total_api_calls}")
                self.logger.info(f"💾 Backup saved to: {backup_file}")
                self.logger.info(f"💾 Master file updated: {self.master_file}")
                self.logger.info("=" * 70)
                
                # 통계
                unique_games = len(set((s['game_id'], s['date']) for s in all_spread_odds))
                avg_spreads = len(all_spread_odds) / unique_games if unique_games > 0 else 0
                self.logger.info(f"🏀 Games processed: {unique_games}")
                self.logger.info(f"📈 Avg spreads per game: {avg_spreads:.1f}")
                
                # Spread 분포
                from collections import Counter
                spread_counts = Counter(s['spread'] for s in all_spread_odds)
                self.logger.info("\n📊 Spread distribution:")
                for spread in sorted(spread_counts.keys()):
                    self.logger.info(f"    {spread:+5.1f}: {spread_counts[spread]} options")
                self.logger.info("=" * 70)
                
                return str(self.master_file)
        else:
            self.logger.error("\n" + "=" * 70)
            self.logger.error("❌ No spread data collected")
            self.logger.error("=" * 70)
            return ""


def main():
    """메인 실행 함수"""
    # API 키 설정
    API_KEY = "81fef80fc013d2c82c9a625ac1fca6b1"
    
    # 수집기 초기화
    collector = NBAAlternateSpreadsCollector(
        api_key=API_KEY,
        days_back=2,  # 증분 모드에서는 무시됨
        incremental=True  # ✅ 증분 모드 (마스터 파일 이어서 수집)
    )
    
    print("\n" + "=" * 70)
    print("🏀 NBA Alternate Spreads Collector v2")
    print("=" * 70)
    print(f"📅 Mode: {'Incremental (new dates only)' if collector.incremental else f'Full scan (last {collector.days_back} days)'}")
    print(f"🎯 Target: FanDuel alternate spreads (±2.5 to ±12.5, 22 options)")
    print(f"📂 Master file: data/alternate_spreads/nba_alternate_spreads_fanduel_master.json")
    print("=" * 70)
    
    # Spreads 수집 실행
    output_file = collector.collect_alternate_spreads()
    
    if output_file:
        print("\n✅ Success! Data saved to:")
        print(f"   {output_file}")
        print("\n💡 Next steps:")
        print("   1. Set incremental=True for daily updates")
        print("   2. Run merge and analysis pipeline")
    else:
        print("\n❌ Collection failed. Please check the logs above.")


if __name__ == "__main__":
    main()

