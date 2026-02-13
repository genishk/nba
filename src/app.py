import streamlit as st
from models.betting_optimizer import BettingOptimizer
from datetime import datetime

def shorten_team_name(name: str) -> str:
    """팀 이름 축약"""
    shortcuts = {
        'Trail Blazers': 'Blazers',
        'Timberwolves': 'Wolves',
        'Mavericks': 'Mavs',
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
        'Jazz': 'UTA'
    }
    return shortcuts.get(name, name)

def main():
    st.set_page_config(layout="wide")
    
    # 전체 스타일 설정
    st.markdown("""
        <style>
            .main { background-color: #FAFBFF; }
            .stApp { background: linear-gradient(135deg, #FAFBFF 0%, #F5F7FF 100%); }
            .stTextInput > label { font-size: 0.95em; font-weight: 500; color: #5C6BC0; }
            .stTextInput > div > div > input {
                font-size: 1em;
                padding: 0.6rem;
                border-radius: 10px;
                border: 1px solid rgba(74, 144, 226, 0.2);
                background-color: #FFFFFF;
                transition: all 0.2s ease;
            }
            .stTextInput > div > div > input:focus {
                border-color: #4A90E2;
                box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.15);
                transform: translateY(-1px);
            }
            .stButton > button {
                background: linear-gradient(120deg, #4A90E2 0%, #5C6BC0 100%);
                border: none;
                padding: 0.6rem 2rem;
                border-radius: 10px;
                font-weight: 600;
                transition: all 0.2s ease;
            }
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(74, 144, 226, 0.2);
            }
            .stTab > button {
                font-weight: 500;
                color: #5C6BC0;
            }
            .stMetric {
                background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFF 100%);
                padding: 1rem;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border: 1px solid rgba(74, 144, 226, 0.1);
            }
        </style>
    """, unsafe_allow_html=True)
    
    # 헤더
    st.markdown("""
        <h1 style='
            text-align: center;
            color: #1D428A;  # NBA 공식 블루
            font-weight: 800;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        '>
            <span style='color: #C8102E;'>NBA</span> Betting Optimizer
        </h1>
    """, unsafe_allow_html=True)
    
    # 데이터 로드
    optimizer = BettingOptimizer()
    predictions = optimizer.load_latest_predictions()
    matches = [
        {
            'home_team': game['home_team_name'],
            'away_team': game['away_team_name'],
            'date': game['date']
        }
        for _, game in predictions.iterrows()
    ]
    
    st.markdown(f"### 📅 Games on {matches[0]['date']}")
    st.markdown("### 📊 Enter Odds for Each Game")
    
    # 배당률 입력 섹션
    odds_data = {}
    cols = st.columns(4)
    
    for idx, match in enumerate(matches):
        col = cols[idx % 4]
        with col:
            # 매치 카드 순서도 변경
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFF 100%);
                    padding: 15px;
                    border-radius: 12px;
                    margin: 8px 0;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    border: 1px solid rgba(74, 144, 226, 0.1);
                    transition: transform 0.2s ease;
                '>
                    <div style='
                        font-size: 1.2em;
                        font-weight: 600;
                        color: #C8102E;  # NBA 공식 레드
                    '>
                        ✈️ {shorten_team_name(match['away_team'])}
                    </div>
                    <div style='
                        font-size: 0.9em;
                        color: #666666;
                        margin: 4px 0;
                        font-weight: 500;
                    '>
                        VS
                    </div>
                    <div style='
                        font-size: 1.2em;
                        font-weight: 600;
                        color: #1D428A;  # NBA 공식 블루
                        margin-bottom: 6px;
                    '>
                        {shorten_team_name(match['home_team'])} 🏠
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # 배당률 입력 순서 변경 - Away를 먼저, Home을 나중에 표시
            away_odds = st.text_input(
                "✈️ Away",
                key=f"away_{match['away_team']}",
                placeholder="+138",
                help="Enter American odds (e.g., +138)"
            )
            if away_odds and (away_odds.startswith('+') or away_odds.startswith('-')):
                odds_data[match['away_team']] = away_odds
            
            home_odds = st.text_input(
                "🏠 Home",
                key=f"home_{match['home_team']}",
                placeholder="-164",
                help="Enter American odds (e.g., -164)"
            )
            if home_odds and (home_odds.startswith('+') or home_odds.startswith('-')):
                odds_data[match['home_team']] = home_odds
    
    # 분석 버튼
    _, col2, _ = st.columns([1,1,1])
    with col2:
        analyze_button = st.button(
            "🎯 Run Analysis",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button:
        if len(odds_data) < len(matches) * 2:
            st.error("❌ Please enter odds for all teams!")
            return
            
        try:
            portfolio = optimizer.analyze_and_save(odds_data, bankroll=300)
            
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
            tab1, tab2, tab3 = st.tabs(["📊 Single Bets", "🎯 2-Team Parlays", "🎲 3-Team Parlays"])
            
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
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 