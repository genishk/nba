#!/usr/bin/env python3
"""
NBA Historical Odds Collector
- 과거 N일간의 NBA 경기 FanDuel 머니라인 배당률 수집
- The-Odds-API Historical API 사용
- ROI 분석을 위한 데이터 수집
"""

import requests
import json
import logging
import pytz
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time


class NBAHistoricalOddsCollector:
    """NBA 과거 배당률 수집기"""
    
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
        self.odds_dir = self.project_root / "data" / "historical_odds"
        self.odds_dir.mkdir(parents=True, exist_ok=True)
        
        # 마스터 파일 경로
        self.master_file = self.odds_dir / "nba_historical_odds_fanduel_master.json"
        
        # 로깅 설정
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("NBAHistoricalOddsCollector")
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
            self.logger.info(f"📂 Loaded {len(data)} existing odds from master file")
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
        
        Returns:
            날짜 문자열 리스트 (오래된 날짜부터) ['2025-11-11', '2025-11-12']
        """
        # ✅ 동부시간 기준으로 오늘 날짜 계산
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
        dates = []
        for i in range(self.days_back, 0, -1):
            target_date = today_et - timedelta(days=i)
            dates.append(target_date.strftime('%Y-%m-%d'))
        
        return dates
    
    def fetch_historical_odds(self, date_str: str) -> Optional[List[Dict]]:
        """
        특정 날짜의 과거 NBA 배당률 데이터 가져오기
        
        Args:
            date_str: 날짜 문자열 (YYYY-MM-DD)
            
        Returns:
            경기 배당률 데이터 리스트 또는 None
        """
        url = f"{self.base_url}/historical/sports/{self.sport}/odds"
        
        # Historical API는 특정 시점의 스냅샷을 요청
        # 정오(UTC 12:00)로 설정하여 대부분의 경기 포함
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',  # Head to head (moneyline)
            'oddsFormat': 'american',
            'date': f"{date_str}T12:00:00Z",  # ISO 8601 format
            'bookmakers': 'fanduel'
        }
        
        try:
            self.logger.info(f"📡 Fetching odds for {date_str}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Historical API 응답 구조 처리
            # 응답은 {'data': [...]} 또는 직접 리스트일 수 있음
            if isinstance(data, dict) and 'data' in data:
                games = data['data']
            elif isinstance(data, list):
                games = data
            else:
                self.logger.warning(f"⚠️  Unexpected response structure for {date_str}")
                return []
            
            self.logger.info(f"✅ Found {len(games)} games for {date_str}")
            
            # API 사용량 정보 (헤더에서 가져올 수 있으면)
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                self.logger.info(f"📊 API requests remaining: {remaining}")
            
            return games
            
        except requests.exceptions.Timeout:
            self.logger.error(f"❌ Timeout fetching odds for {date_str}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Error fetching odds for {date_str}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"   Response status: {e.response.status_code}")
                try:
                    error_data = e.response.json()
                    self.logger.error(f"   Error message: {error_data}")
                except:
                    pass
            return None
    
    def process_odds_data(self, raw_odds: List[Dict], target_date: str) -> List[Dict]:
        """
        배당률 데이터 처리 및 구조화
        
        Args:
            raw_odds: API에서 받은 원본 데이터
            target_date: 요청한 날짜
            
        Returns:
            처리된 배당률 데이터 리스트
        """
        from datetime import datetime
        
        processed_odds = []
        
        # 오늘 날짜 (ET 기준)
        today_et = datetime.now(self.eastern_tz).date()
        
        for game in raw_odds:
            try:
                # 경기 기본 정보
                game_id = game.get('id')
                commence_time_utc = game.get('commence_time')
                home_team_full = game.get('home_team', '')
                away_team_full = game.get('away_team', '')
                
                # 팀 이름을 약어로 변환
                home_team_abbrev = self.NBA_TEAM_ABBREV.get(home_team_full, home_team_full)
                away_team_abbrev = self.NBA_TEAM_ABBREV.get(away_team_full, away_team_full)
                
                # FanDuel 배당률 추출
                bookmakers = game.get('bookmakers', [])
                fanduel_odds = None
                
                for bookmaker in bookmakers:
                    if bookmaker.get('key') == 'fanduel':
                        markets = bookmaker.get('markets', [])
                        
                        for market in markets:
                            if market.get('key') == 'h2h':
                                outcomes = market.get('outcomes', [])
                                
                                home_odds = None
                                away_odds = None
                                
                                # outcomes에서 홈/어웨이 배당률 추출
                                for outcome in outcomes:
                                    team_name = outcome.get('name', '')
                                    odds_value = outcome.get('price')
                                    
                                    if team_name == home_team_full:
                                        home_odds = odds_value
                                    elif team_name == away_team_full:
                                        away_odds = odds_value
                                
                                # 양쪽 배당률이 모두 있는 경우만
                                if home_odds is not None and away_odds is not None:
                                    fanduel_odds = {
                                        'home_odds': home_odds,
                                        'away_odds': away_odds
                                    }
                                    break
                        
                        if fanduel_odds:
                            break
                
                # FanDuel 배당률이 있는 경우에만 저장
                if fanduel_odds:
                    # UTC 시간을 동부시간(ET)으로 변환하여 실제 경기 날짜 추출
                    if commence_time_utc:
                        # UTC 시간 파싱
                        utc_dt = datetime.fromisoformat(commence_time_utc.replace('Z', '+00:00'))
                        # 동부시간으로 변환
                        et_dt = utc_dt.astimezone(self.eastern_tz)
                        # 동부시간 기준 날짜
                        game_date = et_dt.strftime('%Y-%m-%d')
                        game_date_obj = et_dt.date()
                        # 동부시간 문자열
                        commence_time_et = et_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                    else:
                        game_date = target_date
                        game_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                        commence_time_et = None
                    
                    # 미래 경기 필터링 (오늘 또는 그 이후 경기는 제외 - 어제까지만)
                    if game_date_obj >= today_et:
                        self.logger.debug(f"  ⏭️  Skipping today/future game: {home_team_abbrev} vs {away_team_abbrev} on {game_date}")
                        continue
                    
                    odds_record = {
                        'game_id': game_id,
                        'date': game_date,
                        'commence_time_utc': commence_time_utc,
                        'commence_time_et': commence_time_et,
                        'home_team': home_team_abbrev,
                        'away_team': away_team_abbrev,
                        'home_team_full': home_team_full,
                        'away_team_full': away_team_full,
                        'home_odds': fanduel_odds['home_odds'],
                        'away_odds': fanduel_odds['away_odds'],
                        'bookmaker': 'fanduel'
                    }
                    
                    processed_odds.append(odds_record)
                    self.logger.debug(
                        f"  📝 {home_team_abbrev} vs {away_team_abbrev}: "
                        f"{fanduel_odds['home_odds']:+d} / {fanduel_odds['away_odds']:+d}"
                    )
                else:
                    self.logger.debug(f"  ⚠️  No FanDuel odds for {home_team_full} vs {away_team_full}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️  Error processing game data: {e}")
                continue
        
        return processed_odds
    
    def collect_historical_odds(self) -> str:
        """
        과거 N일간의 배당률 수집 실행
        
        Returns:
            저장된 파일 경로 (실패 시 빈 문자열)
        """
        # 수집할 날짜 생성
        target_dates = self.get_target_dates()
        
        if not target_dates:
            self.logger.info("✅ No dates to collect. Master file is up to date.")
            return str(self.master_file) if self.master_file.exists() else ""
        
        self.logger.info("=" * 60)
        self.logger.info("🏀 NBA Historical Odds Collection Started")
        self.logger.info("=" * 60)
        self.logger.info(f"📅 Target dates: {', '.join(target_dates)}")
        self.logger.info(f"🎯 Bookmaker: FanDuel")
        self.logger.info(f"📊 Market: Moneyline (h2h)")
        self.logger.info("=" * 60)
        
        all_odds = []
        successful_dates = 0
        failed_dates = 0
        
        for i, date_str in enumerate(target_dates, 1):
            self.logger.info(f"\n[{i}/{len(target_dates)}] Processing {date_str}...")
            
            # 배당률 데이터 가져오기
            raw_odds = self.fetch_historical_odds(date_str)
            
            if raw_odds is not None:
                # 데이터 처리
                processed_odds = self.process_odds_data(raw_odds, date_str)
                all_odds.extend(processed_odds)
                successful_dates += 1
                
                self.logger.info(f"✅ {len(processed_odds)} FanDuel odds collected for {date_str}")
            else:
                failed_dates += 1
                self.logger.warning(f"❌ Failed to collect data for {date_str}")
            
            # API rate limiting 준수 (Historical API는 약간 더 여유)
            if i < len(target_dates):
                self.logger.info("⏳ Waiting 1.5 seconds...")
                time.sleep(1.5)
        
        # 결과 저장
        if all_odds or self.incremental:
            if self.incremental:
                # 증분 모드: 기존 데이터와 병합
                existing_data = self.load_master_file()
                
                if all_odds:
                    # 새 데이터 추가
                    combined_data = existing_data + all_odds
                    # 중복 제거 (game_id + date 기준)
                    seen = set()
                    unique_data = []
                    for item in combined_data:
                        key = (item.get('game_id'), item.get('date'), item.get('home_team'), item.get('away_team'))
                        if key not in seen:
                            seen.add(key)
                            unique_data.append(item)
                    
                    # 날짜와 경기 시작 시간으로 정렬
                    unique_data.sort(key=lambda x: (x['date'], x.get('commence_time_utc', '')))
                    
                    # 마스터 파일 저장
                    with open(self.master_file, 'w', encoding='utf-8') as f:
                        json.dump(unique_data, f, indent=2, ensure_ascii=False)
                    
                    self.logger.info("\n" + "=" * 60)
                    self.logger.info("🎉 Incremental Update Completed!")
                    self.logger.info("=" * 60)
                    self.logger.info(f"📊 New odds collected: {len(all_odds)}")
                    self.logger.info(f"📚 Total odds in master: {len(unique_data)}")
                    self.logger.info(f"✅ Successful dates: {successful_dates}/{len(target_dates)}")
                    self.logger.info(f"❌ Failed dates: {failed_dates}/{len(target_dates)}")
                    self.logger.info(f"💾 Master file updated: {self.master_file}")
                    self.logger.info("=" * 60)
                    
                    return str(self.master_file)
                else:
                    self.logger.info("\n✅ No new data to add. Master file unchanged.")
                    return str(self.master_file)
            else:
                # 전체 수집 모드: 타임스탬프 파일 + 마스터 파일 업데이트
                all_odds.sort(key=lambda x: (x['date'], x['commence_time_utc']))
                
                # 1. 타임스탬프 파일 저장 (백업용)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = self.odds_dir / f"nba_historical_odds_fanduel_{timestamp}.json"
                
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(all_odds, f, indent=2, ensure_ascii=False)
                
                # 2. 마스터 파일 업데이트
                with open(self.master_file, 'w', encoding='utf-8') as f:
                    json.dump(all_odds, f, indent=2, ensure_ascii=False)
                
                # 요약 출력
                self.logger.info("\n" + "=" * 60)
                self.logger.info("🎉 Full Collection Completed!")
                self.logger.info("=" * 60)
                self.logger.info(f"📊 Total odds collected: {len(all_odds)}")
                self.logger.info(f"✅ Successful dates: {successful_dates}/{len(target_dates)}")
                self.logger.info(f"❌ Failed dates: {failed_dates}/{len(target_dates)}")
                self.logger.info(f"💾 Backup saved to: {backup_file}")
                self.logger.info(f"💾 Master file updated: {self.master_file}")
                self.logger.info("=" * 60)
                
                # 팀별 통계
                teams = set()
                for odds in all_odds:
                    teams.add(odds['home_team'])
                    teams.add(odds['away_team'])
                self.logger.info(f"🏀 Teams involved: {len(teams)} teams")
                
                return str(self.master_file)
        else:
            self.logger.error("\n" + "=" * 60)
            self.logger.error("❌ No odds data collected")
            self.logger.error("=" * 60)
            return ""
    
    def test_api_connection(self) -> bool:
        """
        API 연결 테스트 (현재 경기 데이터로 확인)
        
        Returns:
            연결 성공 여부
        """
        url = f"{self.base_url}/sports/{self.sport}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american',
            'bookmakers': 'fanduel'
        }
        
        try:
            self.logger.info("🔍 Testing API connection...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games_count = len(data) if isinstance(data, list) else 0
            
            self.logger.info(f"✅ API connection successful")
            self.logger.info(f"📊 Current NBA games available: {games_count}")
            
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                self.logger.info(f"📈 API requests remaining: {remaining}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ API connection failed: {e}")
            return False


def main():
    """메인 실행 함수"""
    # API 키 설정
    API_KEY = "c284c82e218e82d4dd976a07e0a7b403"
    
    # 수집기 초기화 (과거 2일치 테스트)
    collector = NBAHistoricalOddsCollector(
        api_key=API_KEY,
        days_back=25,
        incremental=True
    )
    
    print("\n" + "=" * 60)
    print("🏀 NBA Historical Odds Collector")
    print("=" * 60)
    
    # 1. API 연결 테스트
    if not collector.test_api_connection():
        print("\n❌ API connection test failed. Please check your API key.")
        return
    
    # 2. 과거 배당률 수집 실행
    output_file = collector.collect_historical_odds()
    
    if output_file:
        print("\n✅ Success! Data saved to:")
        print(f"   {output_file}")
        print("\n💡 Next steps:")
        print("   1. Collect historical game results from ESPN API")
        print("   2. Match odds with results")
        print("   3. Calculate ROI by team")
    else:
        print("\n❌ Collection failed. Please check the logs above.")


if __name__ == "__main__":
    main()

