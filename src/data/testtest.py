import json

# JSON 파일 읽기
with open('../../src/data/processed_20250123_151221.json', 'r') as file:
    data = json.load(file)

# home_team_name과 away_team_name에서 모든 팀 이름 수집
team_names = set()
for game in data:
    team_names.add(game['home_team_name'])
    team_names.add(game['away_team_name'])

# 정렬된 리스트로 변환
sorted_team_names = sorted(list(team_names))

# 결과 출력
print("Unique team names:")
for team in sorted_team_names:
    print(team)