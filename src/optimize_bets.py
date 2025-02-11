from models.betting_optimizer import BettingOptimizer

# NBA 전체 팀 리스트
NBA_TEAMS = [
    "76ers", "Bucks", "Bulls", "Cavaliers", "Celtics",
    "Clippers", "Grizzlies", "Hawks", "Heat", "Hornets",
    "Jazz", "Kings", "Knicks", "Lakers", "Magic",
    "Mavericks", "Nets", "Nuggets", "Pacers", "Pelicans",
    "Pistons", "Raptors", "Rockets", "Spurs", "Suns",
    "Thunder", "Timberwolves", "Trail Blazers", "Warriors", "Wizards"
]

def input_odds():
    """배당률 입력 도우미"""
    odds_data = {}
    
    # 예측 데이터에서 오늘 경기 팀 목록 가져오기
    optimizer = BettingOptimizer()
    predictions = optimizer.load_latest_predictions()
    today_teams = set()
    
    for _, game in predictions.iterrows():
        today_teams.add(game['home_team_name'])
        today_teams.add(game['away_team_name'])
    
    print("\n=== 오늘의 경기 팀 목록 ===")
    for team in sorted(today_teams):
        print(f"- {team}")
    
    print("\n배당률을 입력하세요.")
    print("입력 방법:")
    print("1. 팀 이름을 정확히 입력")
    print("2. 해당 팀의 배당률 입력 (-164 또는 +138 형식)")
    print("3. 모든 팀 입력 완료 시 'q' 입력하여 종료\n")
    
    remaining_teams = list(sorted(today_teams))
    
    while remaining_teams:
        print("\n남은 팀 목록:")
        for i, team in enumerate(remaining_teams, 1):
            print(f"{i}. {team}")
            
        team = input("\n팀 이름 (또는 'q'로 종료): ").strip()
        if team.lower() == 'q':
            break
            
        if team not in today_teams:
            print("❌ 잘못된 팀 이름입니다. 위 목록에서 정확한 팀 이름을 입력해주세요.")
            continue
            
        if team in odds_data:
            print("❌ 이미 입력된 팀입니다.")
            continue
            
        odds = input(f"✓ {team}의 배당률 입력 (-164 또는 +138): ").strip()
        if not (odds.startswith('+') or odds.startswith('-')):
            print("❌ 배당률은 +나 -로 시작해야 합니다.")
            continue
            
        odds_data[team] = odds
        remaining_teams.remove(team)
        print(f"✓ {team}: {odds} 입력 완료!")
    
    print("\n=== 입력된 배당률 확인 ===")
    for team, odds in odds_data.items():
        print(f"{team}: {odds}")
    
    return odds_data

def main():
    # 배당률 입력 받기
    odds_data = input_odds()
    
    # 초기 자본금 설정
    bankroll = 1000  # $1000
    
    try:
        optimizer = BettingOptimizer()
        optimizer.analyze_and_save(odds_data, bankroll)
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 