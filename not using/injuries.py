from datetime import datetime, timedelta
import requests
from typing import Dict, Any, List
from pathlib import Path
import json

class InjuryDataCollector:
    def __init__(self):
        self.base_url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba"
        self.session = requests.Session()
        
        # 데이터 저장 경로
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "injuries"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_injuries_data(self, team_id: str, days_back: int = 10) -> Dict[str, Any]:
        """특 로스터와 스코어보드에서 부상 정보 수집"""
        print(f"\n=== {team_id} 팀의 부상 정보 수집 시작 ===")
        
        all_data = {
            'collection_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'team_id': team_id,
            'roster_injuries': self._get_roster_injuries(team_id),
            'scoreboard_injuries': self._get_scoreboard_injuries(team_id, days_back)
        }
        
        # 데이터 저장
        self._save_data(all_data, team_id)
        return all_data
    
    def _get_roster_injuries(self, team_id: str) -> List[Dict]:
        """팀 로스터에서 부상 정보 수집"""
        url = f"{self.base_url}/teams/{team_id}/roster"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            injuries = []
            if 'athletes' in data:
                for player in data['athletes']:
                    if 'injuries' in player:
                        injuries.append({
                            'player_name': player.get('fullName'),
                            'injuries': player['injuries']
                        })
            
            return injuries
            
        except Exception as e:
            print(f"Error fetching roster data: {str(e)}")
            return []
    
    def _get_scoreboard_injuries(self, team_id: str, days_back: int) -> List[Dict]:
        """스코어보드에서 부상 정보 수집"""
        injuries = []
        
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            
            url = f"{self.base_url}/scoreboard"
            params = {'dates': date_str}
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'events' in data:
                    for game in data['events']:
                        for team in game['competitions'][0]['competitors']:
                            if team['team']['id'] == team_id and 'injuries' in team:
                                injuries.append({
                                    'date': date_str,
                                    'injuries': team['injuries']
                                })
                
            except Exception as e:
                print(f"Error fetching scoreboard data for {date_str}: {str(e)}")
        
        return injuries

    def _save_data(self, data: Dict[str, Any], team_id: str):
        """수집된 부상 데이터를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"injuries_{team_id}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n부상 데이터가 저장되었습니다: {filename}")

# 테스트 코드
if __name__ == "__main__":
    collector = InjuryDataCollector()
    
    test_teams = [
        ("7", "Nuggets"),
        ("2", "Celtics"),
        ("17", "Lakers")
    ]
    
    for team_id, team_name in test_teams:
        print(f"\n=== {team_name} 부상 정보 수집 중 ===")
        injury_data = collector.collect_injuries_data(team_id)
        if injury_data:
            print(f"데이터 구조:")
            print(json.dumps(injury_data, indent=2)) 