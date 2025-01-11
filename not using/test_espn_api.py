# test_espn_api.py
import requests
from datetime import datetime
from pprint import pprint  # 보기 좋게 출력하기 위해

class ESPNDataCollector:
    def __init__(self):
        self.base_url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba"
    
    def fetch_games(self, date: datetime = None) -> dict:
        """NBA 경기 데이터 수집"""
        date_str = date.strftime("%Y%m%d") if date else datetime.now().strftime("%Y%m%d")
        url = f"{self.base_url}/scoreboard?dates={date_str}"
        
        print(f"Requesting URL: {url}")  # API 요청 URL 확인
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {str(e)}")
            return None

def test_espn_api():
    collector = ESPNDataCollector()
    data = collector.fetch_games()
    
    if data:
        print("\n=== ESPN API 테스트 결과 ===")
        print(f"받은 데이터 타입: {type(data)}")
        
        # 경기 정보 출력
        if 'events' in data:
            games = data['events']
            print(f"\n총 {len(games)}개의 경기 데이터 수신\n")
            
            for game in games:
                print(f"경기: {game['name']}")
                print(f"상태: {game['status']['type']['name']}")
                print(f"시간: {game['date']}")
                print("팀:")
                for team in game['competitions'][0]['competitors']:
                    print(f"- {team['team']['name']}: {team['score']}")
                print("-" * 50)
        else:
            print("경기 데이터가 없습니다.")
    
if __name__ == "__main__":
    test_espn_api()