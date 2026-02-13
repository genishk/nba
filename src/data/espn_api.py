# src/data/espn_api.py
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

class ESPNNBADataCollector:
    def __init__(self):
        self.base_url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba"
        self.session = requests.Session()
        
        # 데이터 저장 경로 수정
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.historical_dir = self.data_dir / "raw" / "historical"
        self.historical_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_historical_data(self, days_back: int = 10) -> Dict[str, Any]:
        """과거 경기 데이터 수집"""
        print(f"\n최근 {days_back}일간의 과거 데이터 수집 시작...")
        
        all_data = {
            'collection_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'days_collected': days_back,
            'games': [],
            'team_stats': {}
        }

        # 과거 날짜별 데이터 수집
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            print(f"\n{date.strftime('%Y-%m-%d')} 데이터 수집 중...")
            
            games_data = self._fetch_daily_games(date_str)
            if games_data and 'events' in games_data:
                for game in games_data['events']:
                    game_data = self._process_game_data(game)
                    all_data['games'].append(game_data)
                    
                    # 참가 팀 데이터 수집
                    for team in game['competitions'][0]['competitors']:
                        team_id = team['team']['id']
                        if team_id not in all_data['team_stats']:
                            print(f"팀 데이터 수집 중: {team['team']['name']}")
                            all_data['team_stats'][team_id] = self._fetch_team_stats(team_id)

        # 데이터를 JSON 파일로 저장 (이 부분이 누락되었습니다)
        self._save_data(all_data)
        return all_data

    def collect_upcoming_data(self, days_ahead: int = 7) -> Dict[str, Any]:
        """향후 예정된 경기 데이터 수집"""
        print(f"\n향후 {days_ahead}일간의 예정 경기 데이터 수집 시작...")
        
        all_data = {
            'collection_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'days_ahead': days_ahead,
            'games': [],
            'team_stats': {}
        }

        # 미래 날짜별 데이터 수집
        for i in range(days_ahead):
            date = datetime.now() + timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            print(f"\n{date.strftime('%Y-%m-%d')} 예정 경기 수집 중...")
            
            games_data = self._fetch_daily_games(date_str)
            if games_data and 'events' in games_data:
                for game in games_data['events']:
                    game_data = self._process_game_data(game)
                    all_data['games'].append(game_data)
                    
                    # 참가 팀 데이터 수집
                    for team in game['competitions'][0]['competitors']:
                        team_id = team['team']['id']
                        if team_id not in all_data['team_stats']:
                            print(f"팀 데이터 수집 중: {team['team']['name']}")
                            all_data['team_stats'][team_id] = self._fetch_team_stats(team_id)

        # 데이터를 JSON 파일로 저장
        self._save_upcoming_data(all_data)
        return all_data
    
    def _fetch_daily_games(self, date: str) -> Dict[str, Any]:
        """특정 날짜의 경기 데이터 조회"""
        url = f"{self.base_url}/scoreboard"
        params = {'dates': date, 'limit': 100}  # 더 많은 경기 데이터 요청
        return self._make_request(url, params)
    
    def _fetch_team_stats(self, team_id: str) -> Dict[str, Any]:
        """팀 통계 데이터 조회"""
        urls = [
            f"{self.base_url}/teams/{team_id}/statistics",
            f"{self.base_url}/teams/{team_id}/schedule",
            f"{self.base_url}/teams/{team_id}/roster",
            f"{self.base_url}/teams/{team_id}/leaders",
            f"{self.base_url}/teams/{team_id}/injuries",
            f"{self.base_url}/teams/{team_id}/splits"
        ]
        
        team_data = {}
        for url in urls:
            data = self._make_request(url)
            if data:
                # API 응답 구조에 따라 데이터 처리
                if 'statistics' in url:
                    team_data['stats'] = data.get('stats', {})
                elif 'roster' in url:
                    team_data['roster'] = data.get('roster', [])
                elif 'leaders' in url:
                    team_data['leaders'] = data.get('leaders', [])
                elif 'injuries' in url:
                    team_data['injuries'] = data.get('injuries', [])
                elif 'splits' in url:
                    team_data['splits'] = data.get('splits', {})
                else:
                    team_data.update(data)
        
        return team_data
    
    def _fetch_team_roster(self, team_id: str) -> Dict[str, Any]:
        """팀 로스터 데이터 조회"""
        url = f"{self.base_url}/teams/{team_id}/roster"
        return self._make_request(url)
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """API 요청 처리"""
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching data from {url}: {str(e)}")
            return {}
    
    def _save_data(self, data: Dict[str, Any]):
        """수집된 데이터를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.historical_dir / f"nba_data_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n데이터가 저장되었습니다: {filename}")
    
    def _process_game_data(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """경기 데이터 처리 및 정리"""
        return {
            # 기본 경기 정보
            'game_id': game['id'],
            'date': game['date'],
            'status': game['status']['type']['name'],
            
            # 홈팀 정보
            'home_team': self._process_team_data(game['competitions'][0]['competitors'][0]),
            
            # 원정팀 정보
            'away_team': self._process_team_data(game['competitions'][0]['competitors'][1]),
            
            # 배당률 정보
            'odds': game['competitions'][0].get('odds', {}),
            
            # 관중 수
            'attendance': game['competitions'][0].get('attendance'),
            
            # 시즌 정보
            'season': {
                'year': game.get('season', {}).get('year'),
                'type': game.get('season', {}).get('type')
            }
        }

    def _process_team_data(self, team_data: Dict[str, Any]) -> Dict[str, Any]:
        """팀 데이터 처리"""
        team_info = {
            'id': team_data['team']['id'],
            'name': team_data['team']['name'],
            'score': team_data.get('score'),
            'statistics': team_data.get('statistics', []),
            'records': team_data.get('records', []),
            'leaders': team_data.get('leaders', []),
            'injuries': team_data['team'].get('injuries', []),  # 부상자 정보
            'form': team_data['team'].get('form', [])  # 최근 경기 결과
        }
        return team_info

    def _save_upcoming_data(self, data: Dict[str, Any]):
        """수집된 예정 경기 데이터를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upcoming_dir = self.data_dir / "upcoming" / "games"
        upcoming_dir.mkdir(parents=True, exist_ok=True)
        
        filename = upcoming_dir / f"upcoming_games_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n예정된 경기 데이터가 저장되었습니다: {filename}")

# 테스트 코드
if __name__ == "__main__":
    collector = ESPNNBADataCollector()
    data = collector.collect_historical_data(days_back=155)  # collect_game_data -> collect_historical_data
    
    print("\n=== 수집된 데이터 통계 ===")
    print(f"수집된 경기 수: {len(data['games'])}")
    print(f"수집된 팀 수: {len(data['team_stats'])}")