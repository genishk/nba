#!/usr/bin/env python3
"""
NBA Alternate Spreads Collector
- 과거 경기의 Alternate Spreads 수집 (-2.5 ~ -12.5)
- The-Odds-API의 /events/{eventId}/odds 엔드포인트 사용
- 경기별로 개별 API 호출 필요
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
    """NBA Alternate Spreads 수집기"""
    
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
    
    # 수집할 Spread 구간
    TARGET_SPREADS = [-2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5, -10.5, -11.5, -12.5]
    
    def __init__(self, api_key: str, days_back: int = 1, max_games_per_date: int = 3):
        """
        Args:
            api_key: The-Odds-API 키
            days_back: 과거 며칠치 데이터 수집 (기본 1일 - 테스트용)
            max_games_per_date: 날짜당 최대 수집 경기 수 (테스트용)
        """
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_nba"
        self.days_back = days_back
        self.max_games_per_date = max_games_per_date
        
        # 시간대 설정 (동부시간)
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        # 디렉토리 설정
        self.project_root = Path(__file__).parent
        self.odds_dir = self.project_root / "data" / "alternate_spreads"
        self.odds_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def get_target_dates(self) -> List[str]:
        """수집할 날짜 리스트 생성"""
        today_et = datetime.now(self.eastern_tz).date()
        dates = []
        
        for i in range(self.days_back, 0, -1):
            target_date = today_et - timedelta(days=i)
            dates.append(target_date.strftime('%Y-%m-%d'))
        
        return dates
    
    def fetch_game_list(self, date_str: str) -> Optional[List[Dict]]:
        """
        특정 날짜의 경기 목록 가져오기
        
        Args:
            date_str: 날짜 문자열 (YYYY-MM-DD)
            
        Returns:
            경기 목록 또는 None
        """
        url = f"{self.base_url}/historical/sports/{self.sport}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',  # 경기 목록만 가져오기
            'oddsFormat': 'american',
            'date': f"{date_str}T12:00:00Z",
            'bookmakers': 'fanduel'
        }
        
        try:
            self.logger.info(f"📡 Fetching game list for {date_str}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, dict) and 'data' in data:
                games = data['data']
            elif isinstance(data, list):
                games = data
            else:
                self.logger.warning(f"⚠️  Unexpected response structure")
                return []
            
            # 최대 경기 수 제한
            if self.max_games_per_date and len(games) > self.max_games_per_date:
                self.logger.info(f"⚠️  Limiting to {self.max_games_per_date} games (test mode)")
                games = games[:self.max_games_per_date]
            
            self.logger.info(f"✅ Found {len(games)} games")
            
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                self.logger.info(f"📊 API requests remaining: {remaining}")
            
            return games
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching game list: {e}")
            return None
    
    def fetch_alternate_spreads(self, game_id: str, date_str: str) -> Optional[Dict]:
        """
        특정 경기의 alternate spreads 가져오기
        
        Args:
            game_id: 경기 ID
            date_str: 날짜 문자열 (YYYY-MM-DD)
            
        Returns:
            경기 데이터 또는 None
        """
        url = f"{self.base_url}/historical/sports/{self.sport}/events/{game_id}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'alternate_spreads',
            'oddsFormat': 'american',
            'date': f"{date_str}T12:00:00Z",
            'bookmakers': 'fanduel'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching alternate spreads for {game_id}: {e}")
            return None
    
    def process_alternate_spreads(self, game_data: Dict, date_str: str) -> List[Dict]:
        """
        Alternate spreads 데이터 처리
        
        Args:
            game_data: API 응답 데이터
            date_str: 날짜 문자열
            
        Returns:
            처리된 데이터 리스트
        """
        processed = []
        
        # 오늘 날짜 (ET 기준)
        today_et = datetime.now(self.eastern_tz).date()
        
        try:
            # Historical API 응답 구조: {'data': {...}}
            if 'data' in game_data:
                game_data = game_data['data']
            
            # 경기 기본 정보
            game_id = game_data.get('id')
            commence_time_utc = game_data.get('commence_time')
            home_team_full = game_data.get('home_team', '')
            away_team_full = game_data.get('away_team', '')
            
            # 팀 약어 변환
            home_team = self.NBA_TEAM_ABBREV.get(home_team_full, home_team_full)
            away_team = self.NBA_TEAM_ABBREV.get(away_team_full, away_team_full)
            
            # 시간 변환
            if commence_time_utc:
                utc_dt = datetime.fromisoformat(commence_time_utc.replace('Z', '+00:00'))
                et_dt = utc_dt.astimezone(self.eastern_tz)
                game_date = et_dt.strftime('%Y-%m-%d')
                game_date_obj = et_dt.date()
                commence_time_et = et_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                game_date = date_str
                game_date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                commence_time_et = None
            
            # 미래 경기 필터링
            if game_date_obj >= today_et:
                return []
            
            # FanDuel alternate spreads 추출
            for bookmaker in game_data.get('bookmakers', []):
                if bookmaker.get('key') == 'fanduel':
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'alternate_spreads':
                            outcomes = market.get('outcomes', [])
                            
                            # 홈팀과 원정팀 스프레드 분리
                            home_spreads = [o for o in outcomes if o['name'] == home_team_full]
                            away_spreads = [o for o in outcomes if o['name'] == away_team_full]
                            
                            # 우리가 원하는 구간만 필터링
                            for spread_data in home_spreads + away_spreads:
                                spread_point = spread_data.get('point')
                                
                                # TARGET_SPREADS 구간에 있는지 확인
                                if spread_point in self.TARGET_SPREADS:
                                    team_name = spread_data.get('name')
                                    is_home = (team_name == home_team_full)
                                    
                                    record = {
                                        'game_id': game_id,
                                        'date': game_date,
                                        'commence_time_utc': commence_time_utc,
                                        'commence_time_et': commence_time_et,
                                        'home_team': home_team,
                                        'away_team': away_team,
                                        'home_team_full': home_team_full,
                                        'away_team_full': away_team_full,
                                        'team': home_team if is_home else away_team,
                                        'is_home': is_home,
                                        'spread': spread_point,
                                        'odds': spread_data.get('price'),
                                        'bookmaker': 'fanduel'
                                    }
                                    
                                    processed.append(record)
            
        except Exception as e:
            self.logger.warning(f"⚠️  Error processing game data: {e}")
        
        return processed
    
    def collect_alternate_spreads(self) -> str:
        """
        Alternate spreads 수집 실행
        
        Returns:
            저장된 파일 경로
        """
        target_dates = self.get_target_dates()
        
        if not target_dates:
            self.logger.info("✅ No dates to collect")
            return ""
        
        self.logger.info("=" * 70)
        self.logger.info("🏀 NBA Alternate Spreads Collection Started")
        self.logger.info("=" * 70)
        self.logger.info(f"📅 Target dates: {', '.join(target_dates)}")
        self.logger.info(f"🎯 Bookmaker: FanDuel")
        self.logger.info(f"📊 Market: Alternate Spreads")
        self.logger.info(f"📏 Target spreads: {', '.join([f'{s:+.1f}' for s in self.TARGET_SPREADS])}")
        self.logger.info(f"🎮 Max games per date: {self.max_games_per_date} (TEST MODE)")
        self.logger.info("=" * 70)
        
        all_data = []
        total_api_calls = 0
        
        for date_str in target_dates:
            self.logger.info(f"\n📅 Processing {date_str}...")
            
            # Step 1: 경기 목록 가져오기
            games = self.fetch_game_list(date_str)
            total_api_calls += 1
            
            if not games:
                self.logger.warning(f"⚠️  No games found for {date_str}")
                continue
            
            # Step 2: 각 경기의 alternate spreads 가져오기
            for i, game in enumerate(games, 1):
                game_id = game['id']
                home_team = game['home_team']
                away_team = game['away_team']
                
                self.logger.info(f"\n  [{i}/{len(games)}] {home_team} vs {away_team}")
                self.logger.info(f"      Game ID: {game_id}")
                
                # Alternate spreads 요청
                game_data = self.fetch_alternate_spreads(game_id, date_str)
                total_api_calls += 1
                
                if game_data:
                    # 데이터 처리
                    processed = self.process_alternate_spreads(game_data, date_str)
                    all_data.extend(processed)
                    
                    self.logger.info(f"      ✅ Collected {len(processed)} spread options")
                else:
                    self.logger.warning(f"      ❌ Failed to get alternate spreads")
                
                # Rate limiting
                if i < len(games):
                    time.sleep(1.5)
        
        # 결과 저장
        if all_data:
            # 정렬
            all_data.sort(key=lambda x: (x['date'], x['game_id'], x['spread']))
            
            # 파일 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.odds_dir / f"nba_alternate_spreads_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info("🎉 Collection Completed!")
            self.logger.info("=" * 70)
            self.logger.info(f"📊 Total spread options collected: {len(all_data)}")
            self.logger.info(f"🔢 Total API calls: {total_api_calls}")
            self.logger.info(f"💾 Saved to: {output_file}")
            
            # 통계
            games_count = len(set(d['game_id'] for d in all_data))
            spreads_per_game = len(all_data) / games_count if games_count > 0 else 0
            
            self.logger.info(f"\n📈 Statistics:")
            self.logger.info(f"   Games processed: {games_count}")
            self.logger.info(f"   Avg spreads per game: {spreads_per_game:.1f}")
            
            # Spread 분포
            spread_counts = {}
            for item in all_data:
                spread = item['spread']
                spread_counts[spread] = spread_counts.get(spread, 0) + 1
            
            self.logger.info(f"\n📊 Spread distribution:")
            for spread in sorted(spread_counts.keys()):
                self.logger.info(f"   {spread:+6.1f}: {spread_counts[spread]} options")
            
            self.logger.info("=" * 70)
            
            return str(output_file)
        else:
            self.logger.error("\n❌ No data collected")
            return ""


def main():
    """메인 실행 함수"""
    API_KEY = "c284c82e218e82d4dd976a07e0a7b403"
    
    # 전체 시즌 수집: 10/21 ~ 어제까지
    collector = NBAAlternateSpreadsCollector(
        api_key=API_KEY,
        days_back=33,  # 약 33일 (10/21부터)
        max_games_per_date=None  # 제한 없음 (전체 수집)
    )
    
    print("\n" + "=" * 70)
    print("🏀 NBA Alternate Spreads Collector (FULL SEASON)")
    print("=" * 70)
    print(f"📅 Collection period: Last 33 days (10/21 ~ yesterday)")
    print(f"🎮 Max games: No limit (all games)")
    print(f"📏 Target spreads: -2.5 to -12.5")
    print(f"⚠️  This will use ~330+ API calls")
    print("=" * 70)
    
    output_file = collector.collect_alternate_spreads()
    
    if output_file:
        print(f"\n✅ Success! Data saved to:")
        print(f"   {output_file}")
        print("\n💡 Next steps:")
        print("   1. Review the collected data")
        print("   2. If looks good, increase days_back and max_games_per_date")
        print("   3. Match with game results")
    else:
        print("\n❌ Collection failed. Please check the logs above.")


if __name__ == "__main__":
    main()

