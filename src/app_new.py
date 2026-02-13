import streamlit as st
from models.betting_optimizer import BettingOptimizer
from datetime import datetime
from pathlib import Path
import os
import json

def shorten_team_name(name: str) -> str:
    """팀 이름 축약"""
    shortcuts = {
        # 전체 이름 -> 약자
        'Trail Blazers': 'POR',
        'Timberwolves': 'MIN',
        'Mavericks': 'DAL',
        'Warriors': 'GSW',
        'Nuggets': 'DEN',
        'Clippers': 'LAC',
        'Lakers': 'LAL',
        'Raptors': 'TOR',
        'Celtics': 'BOS',
        'Knicks': 'NYK',
        'Nets': 'BKN',
        'Hawks': 'ATL',
        'Heat': 'MIA',
        'Bulls': 'CHI',
        'Bucks': 'MIL',
        'Magic': 'ORL',
        'Wizards': 'WAS',
        'Pistons': 'DET',
        'Pacers': 'IND',
        'Cavaliers': 'CLE',
        'Hornets': 'CHA',
        '76ers': 'PHI',
        'Thunder': 'OKC',
        'Spurs': 'SAS',
        'Rockets': 'HOU',
        'Grizzlies': 'MEM',
        'Pelicans': 'NOP',
        'Kings': 'SAC',
        'Suns': 'PHX',
        'Jazz': 'UTA',
        # 이미 약자인 경우는 그대로 반환
        'POR': 'POR',
        'MIN': 'MIN',
        'DAL': 'DAL',
        'GSW': 'GSW',
        'DEN': 'DEN',
        'LAC': 'LAC',
        'LAL': 'LAL',
        'TOR': 'TOR',
        'BOS': 'BOS',
        'NYK': 'NYK',
        'BKN': 'BKN',
        'ATL': 'ATL',
        'MIA': 'MIA',
        'CHI': 'CHI',
        'MIL': 'MIL',
        'ORL': 'ORL',
        'WAS': 'WAS',
        'DET': 'DET',
        'IND': 'IND',
        'CLE': 'CLE',
        'CHA': 'CHA',
        'PHI': 'PHI',
        'OKC': 'OKC',
        'SAS': 'SAS',
        'HOU': 'HOU',
        'MEM': 'MEM',
        'NOP': 'NOP',
        'SAC': 'SAC',
        'PHX': 'PHX',
        'UTA': 'UTA'
    }
    return shortcuts.get(name, name)

def main():
    st.set_page_config(layout="wide")
    
    # 전체 스타일 설정
    st.markdown("""
        <style>
            .main { background-color: #FFFFFF; }
            .stApp { 
                background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
                font-family: 'Inter', sans-serif;
            }
            .stButton > button {
                background: linear-gradient(135deg, #1D428A 0%, #000000 100%);
                border: none;
                padding: 0.75rem 2.5rem;
                border-radius: 12px;
                font-weight: 600;
                color: white;
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px rgba(29, 66, 138, 0.1);
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #000000 0%, #1D428A 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(29, 66, 138, 0.2);
            }
            .stTab > button {
                font-weight: 500;
                color: #1A1A1A;
                border-radius: 8px;
                padding: 0.5rem 1rem;
            }
            .stTab > button:hover {
                background-color: rgba(29, 66, 138, 0.05);
            }
            .stTab > button[data-baseweb="tab"][aria-selected="true"] {
                background-color: rgba(29, 66, 138, 0.1);
                border-bottom-color: #1D428A;
            }
            .stMetric {
                background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
                padding: 1.5rem;
                border-radius: 16px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                border: 1px solid rgba(29, 66, 138, 0.1);
                transition: all 0.3s ease;
            }
            .stMetric:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            }
            div[data-testid="stMetricValue"] {
                color: #1A1A1A;
                font-weight: 500;
                font-family: 'SF Mono', 'Roboto Mono', monospace;
            }
            div[data-testid="stMetricLabel"] {
                color: #64748B;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # 헤더
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0;'>
            <h1 style='
                font-weight: 800;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #17264D 0%, #17264D 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.8rem;
                font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
                letter-spacing: -0.02em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.08);
                position: relative;
                display: inline-block;
            '>
                NBA Betting Optimizer
            </h1>
            <div style='
                width: 160px;
                height: 3px;
                background: #17264D;
                border-radius: 2px;
                margin: 0.5rem auto;
                opacity: 0.9;
            '></div>
        </div>
    """, unsafe_allow_html=True)
    
    # 데이터 로드
    optimizer = BettingOptimizer()
    predictions = optimizer.load_latest_predictions()
    
    # API로 받아온 odds 데이터 로드
    odds_dir = Path(__file__).parent.parent / 'data' / 'odds'
    latest_odds_file = max(odds_dir.glob('processed_nba_odds_*.json'), key=os.path.getctime)
    with open(latest_odds_file) as f:
        odds_list = json.load(f)
    
    # matches 데이터 먼저 생성
    matches = [
        {
            'home_team': game['home_team_name'],
            'away_team': game['away_team_name'],
            'date': game['date']
        }
        for _, game in predictions.iterrows()
    ]
    
    # 디버깅을 위한 출력
    print("\nAvailable matches:")
    for match in matches:
        print(f"Date: {match['date']}, {match['away_team']} @ {match['home_team']}")
    
    print("\nAvailable odds data:")
    for item in odds_list:
        print(f"Date: {item['date']}, {item['team']}: {item['odds']}")
    
    # odds 데이터를 날짜와 팀별로 정리
    target_date = matches[0]['date']  # 오늘 날짜
    print(f"\nTarget date from matches: {target_date}")
    
    # odds 데이터의 모든 날짜 확인
    available_dates = sorted(list(set(item['date'] for item in odds_list)))
    print(f"Available dates in odds data: {available_dates}")
    
    # odds 데이터는 약자를 키로 사용
    odds_data = {}
    # 날짜별로 정렬하여 처리
    sorted_odds = sorted(odds_list, key=lambda x: x['date'])
    
    for item in sorted_odds:
        team_abbrev = item['team']
        # 이미 해당 팀의 배당이 있다면 (더 이른 날짜의 것) 건너뛰기
        if team_abbrev in odds_data:
            continue
            
        odds = item['odds']
        # odds를 문자열로 변환하고 +/- 기호 추가
        odds_str = str(int(odds))
        if not odds_str.startswith('-') and not odds_str.startswith('+'):
            odds_str = '+' + odds_str
        odds_data[team_abbrev] = odds_str
        print(f"Added odds for {team_abbrev} (date: {item['date']}): {odds_str}")  # 디버깅용
    
    print("\nFinal odds_data:", odds_data)  # 디버깅용
    
    st.markdown(f"""
        <div style='margin: 2rem 0 1.5rem;'>
            <div style='
                font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 1.1rem;
                color: #64748B;
                font-weight: 500;
                letter-spacing: -0.01em;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            '>
                <span>📅</span>
                <span style='
                    color: #17264D;
                    font-weight: 600;
                '>{target_date}</span>
            </div>
            <h2 style='
                font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 1.4rem;
                font-weight: 600;
                color: #17264D;
                letter-spacing: -0.01em;
                margin: 1rem 0;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            '>
                <span style='color: #64748B;'>📊</span>
                <span>Today's Games & Odds</span>
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    # 배당률 표시 섹션 (입력 대신 표시만)
    cols = st.columns(4)
    
    for idx, match in enumerate(matches):
        col = cols[idx % 4]
        with col:
            away_team_abbrev = shorten_team_name(match['away_team'])
            home_team_abbrev = shorten_team_name(match['home_team'])
            
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #FFFFFF 0%, #FAFBFF 100%);
                    padding: 1.25rem;
                    border-radius: 16px;
                    margin: 0.75rem 0;
                    box-shadow: 0 4px 12px rgba(23, 38, 77, 0.05);
                    border: 1px solid rgba(23, 38, 77, 0.08);
                    transition: all 0.3s ease;
                    max-width: 100%;
                    overflow: visible;
                    position: relative;
                    &:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 8px 16px rgba(23, 38, 77, 0.08);
                    }}
                '>
                    <div style='
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        gap: 0.5rem;
                        flex-wrap: nowrap;
                    '>
                        <div style='
                            display: flex;
                            align-items: center;
                            gap: 0.5rem;
                            flex: 1;
                            min-width: 0;
                        '>
                            <div style='
                                display: flex;
                                flex-direction: column;
                                align-items: flex-start;
                                min-width: 0;
                            '>
                                <span style='
                                    font-size: 0.95rem;
                                    font-weight: 600;
                                    font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
                                    color: #1E3A8A;
                                    white-space: nowrap;
                                    overflow: hidden;
                                    text-overflow: ellipsis;
                                    letter-spacing: -0.01em;
                                '>{away_team_abbrev}</span>
                                <div style='
                                    font-family: "IBM Plex Mono", "SF Mono", monospace;
                                    font-size: 0.8rem;
                                    font-weight: 500;
                                    padding: 0.15rem 0.5rem;
                                    background: rgba(30, 58, 138, 0.04);
                                    border: 1px solid rgba(30, 58, 138, 0.1);
                                    border-radius: 6px;
                                    color: #1E3A8A;
                                    margin-top: 0.2rem;
                                    letter-spacing: 0.02em;
                                '>
                                    {odds_data.get(away_team_abbrev, 'N/A')}
                                </div>
                            </div>
                        </div>
                        <div style='
                            font-size: 0.8rem;
                            color: #64748B;
                            font-weight: 500;
                            padding: 0.25rem;
                            background: rgba(23, 38, 77, 0.03);
                            border: 1px solid rgba(23, 38, 77, 0.06);
                            border-radius: 50%;
                            width: 1.5rem;
                            height: 1.5rem;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin: 0 0.3rem;
                        '>@</div>
                        <div style='
                            display: flex;
                            align-items: center;
                            gap: 0.5rem;
                            flex: 1;
                            min-width: 0;
                            justify-content: flex-end;
                        '>
                            <div style='
                                display: flex;
                                flex-direction: column;
                                align-items: flex-end;
                                min-width: 0;
                            '>
                                <span style='
                                    font-size: 0.95rem;
                                    font-weight: 600;
                                    font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
                                    color: #991B1B;
                                    white-space: nowrap;
                                    overflow: hidden;
                                    text-overflow: ellipsis;
                                    letter-spacing: -0.01em;
                                '>{home_team_abbrev}</span>
                                <div style='
                                    font-family: "IBM Plex Mono", "SF Mono", monospace;
                                    font-size: 0.8rem;
                                    font-weight: 500;
                                    padding: 0.15rem 0.5rem;
                                    background: rgba(153, 27, 27, 0.04);
                                    border: 1px solid rgba(153, 27, 27, 0.1);
                                    border-radius: 6px;
                                    color: #991B1B;
                                    margin-top: 0.2rem;
                                    letter-spacing: 0.02em;
                                '>
                                    {odds_data.get(home_team_abbrev, 'N/A')}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # 분석 버튼
    _, col2, _ = st.columns([1,1,1])
    with col2:
        analyze_button = st.button(
            "🎯 Run Analysis",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button:
        try:
            # betting_optimizer에 전달하기 전에 약자를 전체 이름으로 변환
            ABBREV_TO_FULL = {
                'ATL': 'Hawks',
                'BOS': 'Celtics',
                'BKN': 'Nets',
                'CHA': 'Hornets',
                'CHI': 'Bulls',
                'CLE': 'Cavaliers',
                'DAL': 'Mavericks',
                'DEN': 'Nuggets',
                'DET': 'Pistons',
                'GSW': 'Warriors',
                'HOU': 'Rockets',
                'IND': 'Pacers',
                'LAC': 'Clippers',
                'LAL': 'Lakers',
                'MEM': 'Grizzlies',
                'MIA': 'Heat',
                'MIL': 'Bucks',
                'MIN': 'Timberwolves',
                'NOP': 'Pelicans',
                'NYK': 'Knicks',
                'OKC': 'Thunder',
                'ORL': 'Magic',
                'PHI': '76ers',
                'PHX': 'Suns',
                'POR': 'Trail Blazers',
                'SAC': 'Kings',
                'SAS': 'Spurs',
                'TOR': 'Raptors',
                'UTA': 'Jazz',
                'WAS': 'Wizards'
            }
            
            optimizer_odds = {ABBREV_TO_FULL[k]: v for k, v in odds_data.items()}
            portfolio = optimizer.analyze_and_save(optimizer_odds, bankroll=300)
            
            st.markdown("---")
            st.markdown("""
                <h2 style='text-align: center; color: #2E7D32;'>
                    📈 Recommended Betting Portfolio
                </h2>
            """, unsafe_allow_html=True)
            
            # 포트폴리오 요약
            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Investment", f"${portfolio['total_investment']:.2f}")
            with cols[1]:
                st.metric("Maximum Loss", f"-${portfolio['max_loss']:.2f}")
            with cols[2]:
                roi = portfolio['expected_profit']/portfolio['total_investment']*100
                st.metric("Expected ROI", f"{roi:.1f}%")
            
            # 베팅 탭
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Single Bets", 
                "🎯 2-Team Parlays", 
                "🎲 3-Team Parlays",
                "💫 High Odds 2-Team Parlays",
                "🌟 High Odds 3-Team Parlays"
            ])
            
            # 베팅 카드 스타일 정의
            card_style = """
                background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFF 100%);
                padding: 15px;
                border-radius: 12px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border: 1px solid rgba(74, 144, 226, 0.1);
            """
            
            # 단일 베팅
            with tab1:
                if portfolio['singles']:
                    for bet in portfolio['singles']:
                        st.markdown(f"""
                            <div style='{card_style}'>
                                <div style='margin-bottom: 8px;'>
                                    <h4 style='color: #1D428A; font-size: 1.2em; margin-bottom: 12px;'>
                                        {bet['match']}
                                    </h4>
                                    <div style='
                                        display: grid;
                                        grid-template-columns: repeat(2, 1fr);
                                        gap: 12px;
                                        margin-bottom: 8px;
                                    '>
                                        <div>
                                            <span style='color: #666666;'>Pick:</span>
                                            <span style='color: #C8102E; font-weight: 600;'> {bet['team']}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Odds:</span>
                                            <span style='font-weight: 600;'> {bet['odds']:.2f}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Win Prob:</span>
                                            <span style='font-weight: 600;'> {bet['probability']:.1%}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Stake:</span>
                                            <span style='font-weight: 600;'> ${bet['amount']:.2f}</span>
                                        </div>
                                    </div>
                                    <div style='
                                        background-color: #F8F9FA;
                                        padding: 8px 12px;
                                        border-radius: 6px;
                                        margin-top: 8px;
                                    '>
                                        <span style='color: #666666;'>Expected Profit:</span>
                                        <span style='color: #1D428A; font-weight: 600;'> ${bet['expected_profit']:.2f}</span>
                                        <span style='color: #C8102E;'> ({bet['expected_profit']/bet['amount']*100:.1f}%)</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
            
            # 2팀 파라레이
            with tab2:
                two_team_parlays = [p for p in portfolio['parlays'] if p['type'] == '2_team_parlay']
                if two_team_parlays:
                    for parlay in two_team_parlays:
                        matches_str = " | ".join([f"{match}: {team}" for match, team in zip(parlay['matches'], parlay['teams'])])
                        st.markdown(f"""
                            <div style='{card_style}'>
                                <div style='margin-bottom: 8px;'>
                                    <h4 style='color: #1D428A; font-size: 1.2em; margin-bottom: 12px;'>
                                        {parlay['type'].upper()}
                                    </h4>
                                    <div style='color: #C8102E; font-weight: 600; margin-bottom: 12px;'>
                                        {matches_str}
                                    </div>
                                    <div style='
                                        display: grid;
                                        grid-template-columns: repeat(2, 1fr);
                                        gap: 12px;
                                        margin-bottom: 8px;
                                    '>
                                        <div>
                                            <span style='color: #666666;'>Base Odds:</span>
                                            <span style='font-weight: 600;'> {parlay['odds']:.2f}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Boosted:</span>
                                            <span style='color: #C8102E; font-weight: 600;'> {parlay['boosted_odds']:.2f}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Win Prob:</span>
                                            <span style='font-weight: 600;'> {parlay['probability']:.1%}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Stake:</span>
                                            <span style='font-weight: 600;'> ${parlay['amount']:.2f}</span>
                                        </div>
                                    </div>
                                    <div style='
                                        background-color: #F8F9FA;
                                        padding: 8px 12px;
                                        border-radius: 6px;
                                        margin-top: 8px;
                                    '>
                                        <span style='color: #666666;'>Expected Profit:</span>
                                        <span style='color: #1D428A; font-weight: 600;'> ${parlay['expected_profit']:.2f}</span>
                                        <span style='color: #C8102E;'> ({parlay['expected_profit']/parlay['amount']*100:.1f}%)</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
            
            # 3팀 파라레이
            with tab3:
                three_team_parlays = [p for p in portfolio['parlays'] if p['type'] == '3_team_parlay']
                if three_team_parlays:
                    for parlay in three_team_parlays:
                        matches_str = " | ".join([f"{match}: {team}" for match, team in zip(parlay['matches'], parlay['teams'])])
                        st.markdown(f"""
                            <div style='{card_style}'>
                                <div style='margin-bottom: 8px;'>
                                    <h4 style='color: #1D428A; font-size: 1.2em; margin-bottom: 12px;'>
                                        {parlay['type'].upper()}
                                    </h4>
                                    <div style='color: #C8102E; font-weight: 600; margin-bottom: 12px;'>
                                        {matches_str}
                                    </div>
                                    <div style='
                                        display: grid;
                                        grid-template-columns: repeat(2, 1fr);
                                        gap: 12px;
                                        margin-bottom: 8px;
                                    '>
                                        <div>
                                            <span style='color: #666666;'>Base Odds:</span>
                                            <span style='font-weight: 600;'> {parlay['odds']:.2f}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Boosted:</span>
                                            <span style='color: #C8102E; font-weight: 600;'> {parlay['boosted_odds']:.2f}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Win Prob:</span>
                                            <span style='font-weight: 600;'> {parlay['probability']:.1%}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Stake:</span>
                                            <span style='font-weight: 600;'> ${parlay['amount']:.2f}</span>
                                        </div>
                                    </div>
                                    <div style='
                                        background-color: #F8F9FA;
                                        padding: 8px 12px;
                                        border-radius: 6px;
                                        margin-top: 8px;
                                    '>
                                        <span style='color: #666666;'>Expected Profit:</span>
                                        <span style='color: #1D428A; font-weight: 600;'> ${parlay['expected_profit']:.2f}</span>
                                        <span style='color: #C8102E;'> ({parlay['expected_profit']/parlay['amount']*100:.1f}%)</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            # High Odds 2팀 파라레이
            with tab4:
                high_odds_2team = [p for p in portfolio['parlays'] 
                                 if p['type'] == '2_team_parlay' and 
                                 all(float(odds) >= -500 if isinstance(odds, str) else odds >= -500 
                                     for odds in p.get('individual_american_odds', []))]
                if high_odds_2team:
                    for parlay in high_odds_2team:
                        matches_str = " | ".join([f"{match}: {team}" for match, team in zip(parlay['matches'], parlay['teams'])])
                        st.markdown(f"""
                            <div style='{card_style}'>
                                <div style='margin-bottom: 8px;'>
                                    <h4 style='color: #1D428A; font-size: 1.2em; margin-bottom: 12px;'>
                                        {parlay['type'].upper()}
                                    </h4>
                                    <div style='color: #C8102E; font-weight: 600; margin-bottom: 12px;'>
                                        {matches_str}
                                    </div>
                                    <div style='
                                        display: grid;
                                        grid-template-columns: repeat(2, 1fr);
                                        gap: 12px;
                                        margin-bottom: 8px;
                                    '>
                                        <div>
                                            <span style='color: #666666;'>Base Odds:</span>
                                            <span style='font-weight: 600;'> {parlay['odds']:.2f}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Boosted:</span>
                                            <span style='color: #C8102E; font-weight: 600;'> {parlay['boosted_odds']:.2f}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Win Prob:</span>
                                            <span style='font-weight: 600;'> {parlay['probability']:.1%}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Stake:</span>
                                            <span style='font-weight: 600;'> ${parlay['amount']:.2f}</span>
                                        </div>
                                    </div>
                                    <div style='
                                        background-color: #F8F9FA;
                                        padding: 8px 12px;
                                        border-radius: 6px;
                                        margin-top: 8px;
                                    '>
                                        <span style='color: #666666;'>Expected Profit:</span>
                                        <span style='color: #1D428A; font-weight: 600;'> ${parlay['expected_profit']:.2f}</span>
                                        <span style='color: #C8102E;'> ({parlay['expected_profit']/parlay['amount']*100:.1f}%)</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            # High Odds 3팀 파라레이
            with tab5:
                high_odds_3team = [p for p in portfolio['parlays'] 
                                 if p['type'] == '3_team_parlay' and 
                                 all(float(odds) >= -500 if isinstance(odds, str) else odds >= -500 
                                     for odds in p.get('individual_american_odds', []))]
                if high_odds_3team:
                    for parlay in high_odds_3team:
                        matches_str = " | ".join([f"{match}: {team}" for match, team in zip(parlay['matches'], parlay['teams'])])
                        st.markdown(f"""
                            <div style='{card_style}'>
                                <div style='margin-bottom: 8px;'>
                                    <h4 style='color: #1D428A; font-size: 1.2em; margin-bottom: 12px;'>
                                        {parlay['type'].upper()}
                                    </h4>
                                    <div style='color: #C8102E; font-weight: 600; margin-bottom: 12px;'>
                                        {matches_str}
                                    </div>
                                    <div style='
                                        display: grid;
                                        grid-template-columns: repeat(2, 1fr);
                                        gap: 12px;
                                        margin-bottom: 8px;
                                    '>
                                        <div>
                                            <span style='color: #666666;'>Base Odds:</span>
                                            <span style='font-weight: 600;'> {parlay['odds']:.2f}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Boosted:</span>
                                            <span style='color: #C8102E; font-weight: 600;'> {parlay['boosted_odds']:.2f}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Win Prob:</span>
                                            <span style='font-weight: 600;'> {parlay['probability']:.1%}</span>
                                        </div>
                                        <div>
                                            <span style='color: #666666;'>Stake:</span>
                                            <span style='font-weight: 600;'> ${parlay['amount']:.2f}</span>
                                        </div>
                                    </div>
                                    <div style='
                                        background-color: #F8F9FA;
                                        padding: 8px 12px;
                                        border-radius: 6px;
                                        margin-top: 8px;
                                    '>
                                        <span style='color: #666666;'>Expected Profit:</span>
                                        <span style='color: #1D428A; font-weight: 600;'> ${parlay['expected_profit']:.2f}</span>
                                        <span style='color: #C8102E;'> ({parlay['expected_profit']/parlay['amount']*100:.1f}%)</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 