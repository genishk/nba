# test_espn_api_one_day.py
import requests
from datetime import datetime
from pprint import pprint
import json

class ESPNDataCollector:
    def __init__(self):
        self.base_url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba"
    
    def fetch_one_day_data(self):
        """하루치 데이터 수집"""
        all_data = {
            'games': [],
            'team_stats': {},
            'rosters': {},
            'game_details': []
        }
        
        # 1. 오늘 경기 일정 및 기본 정보
        games_data = self.fetch_games()
        if games_data and 'events' in games_data:
            for game in games_data['events']:
                all_data['games'].append(game)
                
                # 2. 각 경기의 상세 정보
                game_id = game['id']
                details = self.fetch_game_details(game_id)
                if details:
                    all_data['game_details'].append(details)
                
                # 3. 참가 팀 정보 수집
                for team in game['competitions'][0]['competitors']:
                    team_id = team['team']['id']
                    if team_id not in all_data['team_stats']:
                        stats = self.fetch_team_stats(team_id)
                        roster = self.fetch_team_roster(team_id)
                        all_data['team_stats'][team_id] = stats
                        all_data['rosters'][team_id] = roster
        
        return all_data
    
    def fetch_games(self, date=None):
        """경기 일정 데이터"""
        date_str = date.strftime("%Y%m%d") if date else datetime.now().strftime("%Y%m%d")
        url = f"{self.base_url}/scoreboard?dates={date_str}"
        return self._make_request(url)
    
    def fetch_game_details(self, game_id):
        """경기 상세 정보"""
        url = f"{self.base_url}/summary?event={game_id}"
        return self._make_request(url)
    
    def fetch_team_stats(self, team_id):
        """팀 통계 정보"""
        url = f"{self.base_url}/teams/{team_id}/statistics"
        return self._make_request(url)
    
    def fetch_team_roster(self, team_id):
        """팀 로스터 정보"""
        url = f"{self.base_url}/teams/{team_id}/roster"
        return self._make_request(url)
    
    def _make_request(self, url):
        print(f"Requesting: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

def save_data(data, filename):
    """데이터를 JSON 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def main():
    collector = ESPNDataCollector()
    print("오늘 하루 데이터 수집 시작...")
    
    # 하루치 데이터 수집
    one_day_data = collector.fetch_one_day_data()
    
    # 데이터 저장
    save_data(one_day_data, 'nba_data_one_day.json')
    print("\n데이터 수집 완료! 'nba_data_one_day.json' 파일을 확인해주세요.")
    
    # 수집된 데이터 기본 통계
    print("\n=== 수집된 데이터 통계 ===")
    print(f"수집된 경기 수: {len(one_day_data['games'])}")
    print(f"수집된 팀 수: {len(one_day_data['team_stats'])}")
    print(f"상세 정보가 있는 경기 수: {len(one_day_data['game_details'])}")

if __name__ == "__main__":
    main()