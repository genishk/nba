import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

def load_latest_data(data_type: str = 'historical') -> Dict:
    """최신 데이터 파일 로드"""
    data_dir = Path(__file__).parent.parent.parent / "data"
    if data_type == 'historical':
        data_dir = data_dir / "raw" / "historical"
    else:
        data_dir = data_dir / "upcoming" / "games"
    
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_dir}")
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"\n데이터 파일 로드: {latest_file.name}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_team_name(games: List[Dict], team_id: str) -> str:
    """팀 ID에 해당하는 정확한 팀 이름 찾기"""
    for game in games:
        if game['home_team']['id'] == team_id:
            return game['home_team']['name']
        if game['away_team']['id'] == team_id:
            return game['away_team']['name']
    return f"Unknown Team ({team_id})"

def analyze_recent_stats(games: List[Dict], n_games: int = 5) -> Dict:
    """최근 N경기 통계 분석"""
    # 완료된 경기만 필터링
    completed_games = [g for g in games if g['score'] != 'Not played'][-n_games:]
    
    if not completed_games:
        return {
            'recent_win_rate': 0.0,
            'recent_avg_score': 0.0,
            'recent_home_win_rate': 0.0,
            'recent_away_win_rate': 0.0
        }
    
    wins = 0
    total_score = 0
    home_games = 0
    home_wins = 0
    away_games = 0
    away_wins = 0
    
    for game in completed_games:
        # 점수 파싱
        my_score, opp_score = map(int, game['score'].split('-'))
        
        # 승/패 계산
        if my_score > opp_score:
            wins += 1
            if game['is_home']:
                home_wins += 1
            else:
                away_wins += 1
        
        # 점수 누적
        total_score += my_score
        
        # 홈/원정 경기 수 계산
        if game['is_home']:
            home_games += 1
        else:
            away_games += 1
    
    return {
        'recent_win_rate': wins / len(completed_games),
        'recent_avg_score': total_score / len(completed_games),
        'recent_home_win_rate': home_wins / home_games if home_games > 0 else 0.0,
        'recent_away_win_rate': away_wins / away_games if away_games > 0 else 0.0
    }

def analyze_team_games(data: Dict) -> None:
    """각 팀별 경기 데이터 분석"""
    team_games = defaultdict(list)
    
    # 각 팀별 경기 수집
    for game in data['games']:
        game_date = datetime.fromisoformat(game['date'].replace('Z', '+00:00'))
        
        # 홈팀 경기 추가
        home_team = game['home_team']
        team_games[home_team['id']].append({
            'date': game_date,
            'opponent_id': game['away_team']['id'],
            'is_home': True,
            'score': f"{home_team['score']}-{game['away_team']['score']}" if game['status'] == 'STATUS_FINAL' else 'Not played'
        })
        
        # 원정팀 경기 추가
        away_team = game['away_team']
        team_games[away_team['id']].append({
            'date': game_date,
            'opponent_id': game['home_team']['id'],
            'is_home': False,
            'score': f"{away_team['score']}-{home_team['score']}" if game['status'] == 'STATUS_FINAL' else 'Not played'
        })
    
    # 각 팀별 분석 결과 출력
    print("\n=== 팀별 경기 데이터 분석 ===")
    for team_id, games in team_games.items():
        # 날짜순 정렬
        games.sort(key=lambda x: x['date'])
        
        # 팀 이름 찾기 (수정된 부분)
        team_name = get_team_name(data['games'], team_id)
        
        print(f"\n{team_name} (ID: {team_id})")
        print(f"총 경기 수: {len(games)}")
        if games:
            print(f"첫 경기: {games[0]['date'].strftime('%Y-%m-%d')}")
            print(f"마지막 경기: {games[-1]['date'].strftime('%Y-%m-%d')}")
            
            # 최근 5경기 통계 추가
            recent_stats = analyze_recent_stats(games)
            print(f"\n최근 5경기 통계:")
            print(f"- 승률: {recent_stats['recent_win_rate']:.3f}")
            print(f"- 평균 득점: {recent_stats['recent_avg_score']:.1f}")
            print(f"- 홈 승률: {recent_stats['recent_home_win_rate']:.3f}")
            print(f"- 원정 승률: {recent_stats['recent_away_win_rate']:.3f}")
            
            # 기존의 최근 5경기 정보 출력 유지
            print("\n최근 5경기:")
            for game in games[-5:]:
                home_away = "홈" if game['is_home'] else "원정"
                print(f"- {game['date'].strftime('%Y-%m-%d')} ({home_away}): {game['score']}")

if __name__ == "__main__":
    data = load_latest_data()
    analyze_team_games(data)
