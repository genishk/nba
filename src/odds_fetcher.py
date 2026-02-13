import requests
import pandas as pd
from pathlib import Path
import json
import os
from datetime import datetime
import time

class OddsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.sport = "basketball_nba"
        
    def fetch_moneyline_odds(self):
        """NBA 경기 머니라인(승패) 배당률 가져오기"""
        # 1. 먼저 경기 목록 가져오기
        events_url = f"{self.base_url}/{self.sport}/events"
        events_params = {
            'apiKey': self.api_key,
        }
        
        try:
            # 경기 목록 가져오기
            events_response = requests.get(events_url, params=events_params)
            events_response.raise_for_status()
            events_data = events_response.json()
            
            print(f"\nFound {len(events_data)} games")
            
            all_odds = []
            # 2. 각 경기별로 배당률 가져오기
            for event in events_data:
                event_id = event['id']
                odds_url = f"{self.base_url}/{self.sport}/events/{event_id}/odds"
                odds_params = {
                    'apiKey': self.api_key,
                    'regions': 'us',
                    'markets': 'h2h',
                    'oddsFormat': 'american',
                    'bookmakers': 'draftkings'
                }
                
                print(f"\nFetching odds for {event['home_team']} vs {event['away_team']}")
                odds_response = requests.get(odds_url, params=odds_params)
                odds_response.raise_for_status()
                
                odds_data = odds_response.json()
                all_odds.append(odds_data)
                
                # API 호출 간격 조절 (rate limiting 방지)
                time.sleep(1)
            
            # 응답 저장
            self._save_raw_response(all_odds)
            
            return self._process_odds(all_odds)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching odds: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
                print(f"URL: {e.response.url}")
            return None
    
    def _process_odds(self, raw_data):
        """API 응답을 처리하여 DataFrame으로 변환"""
        # NBA 팀명 매핑 딕셔너리 (전체 이름 -> 약자)
        TEAM_ABBREV = {
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

        processed_odds = []
        
        try:
            for game in raw_data:
                # 경기 관련 추가 정보
                game_id = game['id']
                home_team = TEAM_ABBREV.get(game['home_team'], game['home_team'])
                away_team = TEAM_ABBREV.get(game['away_team'], game['away_team'])
                commence_time = game['commence_time']
                
                # 추가 가능한 경기 정보들
                sport_title = game.get('sport_title')
                sport_key = game.get('sport_key')
                commence_datetime = game.get('commence_time')
                completed = game.get('completed', False)
                home_score = game.get('scores', {}).get('home')
                away_score = game.get('scores', {}).get('away')
                last_update = game.get('last_update')
                
                for bookmaker in game.get('bookmakers', []):
                    bookmaker_key = bookmaker['key']
                    bookmaker_title = bookmaker['title']
                    bookmaker_last_update = bookmaker.get('last_update')
                    
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            market_key = market['key']
                            market_last_update = market.get('last_update')
                            
                            # Process home team odds
                            home_outcome = next((outcome for outcome in market['outcomes'] 
                                               if outcome['name'] == game['home_team']), None)
                            # Process away team odds
                            away_outcome = next((outcome for outcome in market['outcomes']
                                               if outcome['name'] == game['away_team']), None)
                            
                            if home_outcome and away_outcome:
                                # Add home team odds
                                processed_odds.append({
                                    'game_id': game_id,
                                    'date': datetime.fromisoformat(commence_time).strftime('%Y-%m-%d'),
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'team': home_team,
                                    'is_home': True,
                                    'market_key': market_key,
                                    'odds': home_outcome['price'],
                                    'bookmaker': bookmaker_key,
                                    'probability': self._convert_odds_to_probability(home_outcome['price']),
                                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                                    'sport_title': sport_title,
                                    'commence_datetime': commence_datetime,
                                    'bookmaker_title': bookmaker_title,
                                    'market_last_update': market_last_update,
                                    'completed': completed,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'game_last_update': last_update,
                                    'bookmaker_last_update': bookmaker_last_update
                                })
                                
                                # Add away team odds
                                processed_odds.append({
                                    'game_id': game_id,
                                    'date': datetime.fromisoformat(commence_time).strftime('%Y-%m-%d'),
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'team': away_team,
                                    'is_home': False,
                                    'market_key': market_key,
                                    'odds': away_outcome['price'],
                                    'bookmaker': bookmaker_key,
                                    'probability': self._convert_odds_to_probability(away_outcome['price']),
                                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                                    'sport_title': sport_title,
                                    'commence_datetime': commence_datetime,
                                    'bookmaker_title': bookmaker_title,
                                    'market_last_update': market_last_update,
                                    'completed': completed,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'game_last_update': last_update,
                                    'bookmaker_last_update': bookmaker_last_update
                                })
                            
            df = pd.DataFrame(processed_odds)
            if not df.empty:
                # 시간 정보 처리
                df['commence_datetime'] = pd.to_datetime(df['commence_datetime'])
                df['market_last_update'] = pd.to_datetime(df['market_last_update'])
                df['game_last_update'] = pd.to_datetime(df['game_last_update'])
                df['bookmaker_last_update'] = pd.to_datetime(df['bookmaker_last_update'])
                
                print("\nComplete data structure:")
                print("\nColumns:", df.columns.tolist())
                print("\nSample row:")
                print(df.iloc[0].to_dict())
                
            return df
            
        except Exception as e:
            print(f"Error processing odds: {e}")
            print("Raw data sample:", json.dumps(raw_data[:1], indent=2))
            return pd.DataFrame()
        
    def _convert_odds_to_probability(self, american_odds):
        """미국식 배당률을 확률로 변환"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return (-american_odds) / (-american_odds + 100)
    
    def _save_raw_response(self, data):
        """API 응답 원본 저장"""
        odds_dir = Path(__file__).parent.parent / 'data' / 'odds'
        os.makedirs(odds_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = odds_dir / f'nba_odds_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get_best_odds(self):
        """각 팀의 최고 배당률 가져오기"""
        odds_df = self.fetch_moneyline_odds()
        if odds_df is None:
            return None
            
        # 팀별로 최고 배당률 선택
        best_odds = odds_df.loc[odds_df.groupby(['game_id', 'team'])['probability'].idxmax()]
        return best_odds.sort_values('probability', ascending=False)

def main():
    """테스트 실행"""
    api_key = "96d2a1ba46ec7b941044f395d532f8cd"
        
    fetcher = OddsFetcher(api_key)
    odds = fetcher.get_best_odds()
    
    if odds is not None:
        print("\nBest Available Odds:")
        print(odds[['team', 'odds', 'probability', 'bookmaker']].head(10))
        
        # 파일로 저장
        odds_dir = Path(__file__).parent.parent / 'data' / 'odds'
        os.makedirs(odds_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        odds_file = odds_dir / f'processed_nba_odds_{timestamp}.json'
        odds.to_json(odds_file, orient='records', indent=2)
        print(f"\nOdds saved to: {odds_file}")

if __name__ == "__main__":
    main() 