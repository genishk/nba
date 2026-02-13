import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from models.betting_optimizer import BettingOptimizer
from datetime import datetime
from pathlib import Path
import os

# Set up page configuration
st.set_page_config(
    page_title="NBA Betting Optimization System",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color theme settings (NBA colors)
NBA_COLORS = {
    'primary': '#17264D',    # NBA logo dark blue
    'secondary': '#C8102E',  # NBA logo red
    'accent': '#1D428A',     # NBA blue
    'background': '#F8FAFC', # Light background
    'text': '#333333'        # Text color
}

# Apply styles
st.markdown(f"""
<style>
    .reportview-container .main .block-container{{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {NBA_COLORS['primary']};
        font-weight: bold;
    }}
    .stButton>button {{
        background-color: {NBA_COLORS['primary']};
        color: white;
    }}
    .stProgress .st-bo {{
        background-color: {NBA_COLORS['accent']};
    }}
    .highlight {{
        background-color: {NBA_COLORS['secondary']};
        color: white;
        padding: 0.3rem;
        border-radius: 0.3rem;
    }}
    .good-value {{
        color: green;
        font-weight: bold;
    }}
    .bad-value {{
        color: red;
        font-weight: bold;
    }}
    .info-box {{
        background-color: {NBA_COLORS['accent']};
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #F0F0F0;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {NBA_COLORS['primary']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

# Main title and description
st.title("NBA Betting Optimization System 🏀")
st.markdown("""
This system analyzes NBA game predictions and current odds to identify the most valuable betting opportunities.
It optimizes capital allocation through the Kelly Criterion and portfolio optimization techniques.
""")

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

# Sidebar settings
st.sidebar.header("Settings")

# Bankroll settings
bankroll = st.sidebar.slider("Betting Bankroll ($)", 100, 1000, 300, 50)

# Minimum bet amount
min_bet = st.sidebar.slider("Minimum Bet Amount ($)", 1, 20, 5, 1)

# Initialize betting optimizer
optimizer = BettingOptimizer()

# Load predictions data - using original approach from the NBA dashboard
try:
    predictions = optimizer.load_latest_predictions()
except Exception as e:
    st.error(f"Error loading predictions: {str(e)}")
    predictions = None

# Create matches data from predictions
matches = []
if predictions is not None:
    matches = [
        {
            'home_team': game['home_team_name'],
            'away_team': game['away_team_name'],
            'date': game['date']
        }
        for _, game in predictions.iterrows()
    ]
    
# Load odds data
odds_list = []
try:
    odds_dir = Path(__file__).parent.parent / 'data' / 'odds'
    latest_odds_file = max(odds_dir.glob('processed_nba_odds_*.json'), key=os.path.getctime)
    with open(latest_odds_file) as f:
        odds_list = json.load(f)
except (FileNotFoundError, ValueError) as e:
    st.warning("Odds data file not found. You'll need to enter odds manually.")

# Debug information (only if debug mode is enabled)
debug_mode = False  # Set to True for debugging
if debug_mode:
    st.write("Matches:", matches)
    st.write("Odds data:", odds_list)

# Process odds data
target_date = matches[0]['date'] if matches else datetime.now().strftime("%Y-%m-%d")
odds_data = {}

# Sort odds by date
sorted_odds = sorted(odds_list, key=lambda x: x['date']) if odds_list else []

for item in sorted_odds:
    team_abbrev = item['team']
    # Skip if already have odds for this team (use earliest date)
    if team_abbrev in odds_data:
        continue
        
    odds = item['odds']
    # Format odds string with +/- sign
    odds_str = str(int(odds))
    if not odds_str.startswith('-') and not odds_str.startswith('+'):
        odds_str = '+' + odds_str
    odds_data[team_abbrev] = odds_str

# Manual odds input fields for each match
st.sidebar.subheader("Game Odds")
st.sidebar.info("Enter odds in American format. Example: +120, -110")

updated_odds = {}

# Create odds input UI for each game
for idx, match in enumerate(matches):
    home_team = match['home_team']
    away_team = match['away_team']
    home_abbr = shorten_team_name(home_team)
    away_abbr = shorten_team_name(away_team)
    
    st.sidebar.markdown(f"**{away_team} @ {home_team}**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        updated_odds[away_team] = st.text_input(
            f"{away_abbr} (Away)", 
            odds_data.get(away_abbr, "0"),
            key=f"away_{idx}"
        )
    with col2:
        updated_odds[home_team] = st.text_input(
            f"{home_abbr} (Home)", 
            odds_data.get(home_abbr, "0"),
            key=f"home_{idx}"
        )
    
    # Update odds_data with user input
    odds_data[away_abbr] = updated_odds[away_team]
    odds_data[home_abbr] = updated_odds[home_team]
    
    # Add divider
    st.sidebar.markdown("---")

# Run analysis button
if st.sidebar.button("🎯 Run Analysis", type="primary", use_container_width=True):
    with st.spinner("Analyzing betting opportunities..."):
        try:
            # Configure optimizer with min bet amount
            optimizer.min_bet_amount = min_bet
            
            # Run analysis with updated odds
            portfolio = optimizer.analyze_and_save(updated_odds, bankroll=bankroll)
            
            # Save portfolio to session state for use in tabs
            st.session_state['portfolio'] = portfolio
            st.session_state['bankroll'] = bankroll
            
            st.success("Analysis completed successfully!")
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["Game Predictions", "Betting Recommendations", "Portfolio Analysis"])

# Tab 1: Game Predictions
with tab1:
    st.header("NBA Game Predictions")
    
    # Display date
    if matches:
        st.markdown(f"""
                    <div style='margin: 1rem 0;'>
                <div style='
                    font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
                    font-size: 1.1rem;
                    color: #64748B;
                    font-weight: 500;
                    margin-bottom: 0.5rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                '>
                    <span>📅</span>
                        <span style='color: {NBA_COLORS['primary']}; font-weight: 600;'>{target_date}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
        # Display game predictions
        for idx, match in enumerate(matches):
            col1, col2, col3 = st.columns([2, 1, 2])
            
            away_team = match['away_team']
            home_team = match['home_team']
            away_abbr = shorten_team_name(away_team)
            home_abbr = shorten_team_name(home_team)
            
            # Find this game in predictions
            game_data = None
            for _, game in predictions.iterrows():
                if game['home_team_name'] == home_team and game['away_team_name'] == away_team:
                    game_data = game
                    break
            
            if game_data is not None:
                home_win_prob = game_data['home_win_probability']
                away_win_prob = 1 - home_win_prob
                
                # Away team column
                with col1:
                    st.markdown(f"### {away_team}")
                    st.progress(float(away_win_prob))
                    st.markdown(f"<h4 style='text-align: center;'>{away_win_prob:.1%}</h4>", unsafe_allow_html=True)
                
                # Center column with VS
                with col2:
                    st.markdown(f"<h4 style='text-align: center;'>VS</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center;'>Odds: {odds_data.get(away_abbr, 'N/A')} / {odds_data.get(home_abbr, 'N/A')}</p>", unsafe_allow_html=True)
                
                # Home team column
                with col3:
                    st.markdown(f"### {home_team}")
                    st.progress(float(home_win_prob))
                    st.markdown(f"<h4 style='text-align: center;'>{home_win_prob:.1%}</h4>", unsafe_allow_html=True)
                
                # Predicted winner
                winner = home_team if home_win_prob >= 0.5 else away_team
                winner_prob = home_win_prob if home_win_prob >= 0.5 else away_win_prob
                st.markdown(f"<div style='text-align: center; margin-bottom: 2rem;'><span class='highlight'>Predicted Winner: {winner} ({winner_prob:.1%})</span></div>", unsafe_allow_html=True)
            
            # Add divider between games
            if idx < len(matches) - 1:
                st.markdown("---")
    else:
        st.info("No games found for analysis. Please check your prediction data.")

# Tab 2: Betting Recommendations
with tab2:
    st.header("Value Betting Recommendations")
    
    if 'portfolio' in st.session_state:
        portfolio = st.session_state['portfolio']
        bankroll = st.session_state['bankroll']
        
        # Portfolio summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Investment", f"${portfolio['total_investment']:.2f}", f"{portfolio['total_investment']/bankroll*100:.1f}% of bankroll")
        with col2:
            st.metric("Maximum Possible Loss", f"-${portfolio['max_loss']:.2f}")
        with col3:
            expected_profit = portfolio['expected_profit']
            roi = (expected_profit / portfolio['total_investment']) * 100 if portfolio['total_investment'] > 0 else 0
            st.metric("Expected Profit", f"${expected_profit:.2f}", f"{roi:.1f}% ROI")
        with col4:
            st.metric("Capital Preservation", f"{(bankroll - portfolio['max_loss'])/bankroll*100:.1f}%")
        
        # Create betting recommendation tabs - 8개 탭으로 확장
        bet_tab1, bet_tab2, bet_tab3, bet_tab4, bet_tab5, bet_tab6, bet_tab7, bet_tab8 = st.tabs([
            "Single Bets", 
            "2-Team Parlays", 
            "3-Team Parlays",
            "High Odds 2-Team Parlays",
            "High Odds 3-Team Parlays",
            "Value Underdog Bets",
            "Random 3-Team Parlays",
            "Random 4-Team Parlays"
        ])
        
        # Display single bets
        with bet_tab1:
            if portfolio['singles']:
                # Create dataframe for singles
                singles_df = pd.DataFrame(portfolio['singles'])
                singles_df['roi'] = singles_df['expected_profit'] / singles_df['amount'] * 100
                
                # Display table
                st.dataframe(
                    singles_df[['match', 'team', 'probability', 'odds', 'amount', 'potential_profit', 'roi']].rename(columns={
                        'match': 'Game', 
                        'team': 'Team',
                        'probability': 'Win Probability',
                        'odds': 'Odds',
                        'amount': 'Bet Amount($)',
                        'potential_profit': 'Potential Profit($)',
                        'roi': 'Expected ROI(%)'
                    }).style.format({
                        'Win Probability': '{:.1%}',
                        'Odds': '{:.2f}',
                        'Bet Amount($)': '${:.2f}',
                        'Potential Profit($)': '${:.2f}',
                        'Expected ROI(%)': '{:.1f}%'
                    }),
                    height=150
                )
                
                # Chart - Win Probability vs Odds
                fig = px.scatter(
                    singles_df, 
                    x='probability', 
                    y='odds', 
                    size='amount',
                    color='roi',
                    hover_name='team',
                    hover_data=['match', 'amount', 'potential_profit', 'roi'],
                    title='Single Bet Value Analysis',
                    color_continuous_scale='RdYlGn',
                    labels={
                        'probability': 'Win Probability', 
                        'odds': 'Odds',
                        'amount': 'Bet Amount',
                        'roi': 'Expected ROI(%)'
                    }
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recommended single bets found in the analysis.")
        
        # Display 2-team parlays
        with bet_tab2:
            parlays_2team = [p for p in portfolio['parlays'] if p['type'] == '2_team_parlay']
            if parlays_2team:
                # Create dataframe
                parlays_2team_df = pd.DataFrame([
                    {
                        'teams': ' & '.join(p['teams']),
                        'probability': p['probability'],
                        'odds': p['boosted_odds'],
                        'amount': p['amount'],
                        'potential_profit': p['potential_profit'],
                        'roi': p['expected_profit'] / p['amount'] * 100 if p['amount'] > 0 else 0
                    } for p in parlays_2team
                ])
                
                # Display table
                st.dataframe(
                    parlays_2team_df.rename(columns={
                        'teams': 'Teams',
                        'probability': 'Win Probability',
                        'odds': 'Odds (Boosted)',
                        'amount': 'Bet Amount($)',
                        'potential_profit': 'Potential Profit($)',
                        'roi': 'Expected ROI(%)'
                    }).style.format({
                        'Win Probability': '{:.1%}',
                        'Odds (Boosted)': '{:.2f}',
                        'Bet Amount($)': '${:.2f}',
                        'Potential Profit($)': '${:.2f}',
                        'Expected ROI(%)': '{:.1f}%'
                    }),
                    height=150
                )
            else:
                st.info("No recommended 2-team parlays found in the analysis.")
        
        # Display 3-team parlays
        with bet_tab3:
            parlays_3team = [p for p in portfolio['parlays'] if p['type'] == '3_team_parlay']
            if parlays_3team:
                # Create dataframe
                parlays_3team_df = pd.DataFrame([
                    {
                        'teams': ' & '.join(p['teams']),
                        'probability': p['probability'],
                        'odds': p['boosted_odds'],
                        'amount': p['amount'],
                        'potential_profit': p['potential_profit'],
                        'roi': p['expected_profit'] / p['amount'] * 100 if p['amount'] > 0 else 0
                    } for p in parlays_3team
                ])
                
                # Display table
                st.dataframe(
                    parlays_3team_df.rename(columns={
                        'teams': 'Teams',
                        'probability': 'Win Probability',
                        'odds': 'Odds (Boosted)',
                        'amount': 'Bet Amount($)',
                        'potential_profit': 'Potential Profit($)',
                        'roi': 'Expected ROI(%)'
                    }).style.format({
                        'Win Probability': '{:.1%}',
                        'Odds (Boosted)': '{:.2f}',
                        'Bet Amount($)': '${:.2f}',
                        'Potential Profit($)': '${:.2f}',
                        'Expected ROI(%)': '{:.1f}%'
                    }),
                    height=150
                )
            else:
                st.info("No recommended 3-team parlays found in the analysis.")
                
        # Display High Odds 2-team parlays
        with bet_tab4:
            # Filter high odds 2-team parlays
            high_odds_2team = [p for p in portfolio['parlays'] 
                             if p['type'] == '2_team_parlay' and 
                             all(float(odds) >= -500 if isinstance(odds, str) else odds >= -500 
                                 for odds in p.get('individual_american_odds', []))]
            
            if high_odds_2team:
                # Create dataframe
                high_odds_2team_df = pd.DataFrame([
                    {
                        'teams': ' & '.join(p['teams']),
                        'probability': p['probability'],
                        'odds': p['boosted_odds'],
                        'amount': p['amount'],
                        'potential_profit': p['potential_profit'],
                        'roi': p['expected_profit'] / p['amount'] * 100 if p['amount'] > 0 else 0
                    } for p in high_odds_2team
                ])
                
                # Display table
                st.dataframe(
                    high_odds_2team_df.rename(columns={
                        'teams': 'Teams',
                        'probability': 'Win Probability',
                        'odds': 'Odds (Boosted)',
                        'amount': 'Bet Amount($)',
                        'potential_profit': 'Potential Profit($)',
                        'roi': 'Expected ROI(%)'
                    }).style.format({
                        'Win Probability': '{:.1%}',
                        'Odds (Boosted)': '{:.2f}',
                        'Bet Amount($)': '${:.2f}',
                        'Potential Profit($)': '${:.2f}',
                        'Expected ROI(%)': '{:.1f}%'
                    }),
                    height=150
                )
            else:
                st.info("No recommended High Odds 2-team parlays found in the analysis.")
        
        # Display High Odds 3-team parlays
        with bet_tab5:
            # Filter high odds 3-team parlays
            high_odds_3team = [p for p in portfolio['parlays'] 
                             if p['type'] == '3_team_parlay' and 
                             all(float(odds) >= -500 if isinstance(odds, str) else odds >= -500 
                                 for odds in p.get('individual_american_odds', []))]
            
            if high_odds_3team:
                # Create dataframe
                high_odds_3team_df = pd.DataFrame([
                    {
                        'teams': ' & '.join(p['teams']),
                        'probability': p['probability'],
                        'odds': p['boosted_odds'],
                        'amount': p['amount'],
                        'potential_profit': p['potential_profit'],
                        'roi': p['expected_profit'] / p['amount'] * 100 if p['amount'] > 0 else 0
                    } for p in high_odds_3team
                ])
                
                # Display table
                st.dataframe(
                    high_odds_3team_df.rename(columns={
                        'teams': 'Teams',
                        'probability': 'Win Probability',
                        'odds': 'Odds (Boosted)',
                        'amount': 'Bet Amount($)',
                        'potential_profit': 'Potential Profit($)',
                        'roi': 'Expected ROI(%)'
                    }).style.format({
                        'Win Probability': '{:.1%}',
                        'Odds (Boosted)': '{:.2f}',
                        'Bet Amount($)': '${:.2f}',
                        'Potential Profit($)': '${:.2f}',
                        'Expected ROI(%)': '{:.1f}%'
                    }),
                    height=150
                )
            else:
                st.info("No recommended High Odds 3-team parlays found in the analysis.")
        
        # Display Value Underdog Bets
        with bet_tab6:
            # Value Underdog Bets - two cases
            value_underdogs = []
            
            # Case 1: Good win probability (60%+) but high odds (2.0+)
            high_prob_underdogs = [bet for bet in portfolio['singles'] 
                                  if bet['probability'] >= 0.6 and bet['odds'] >= 2.0]
            
            # Case 2: Low win probability (<50%) but positive expected value
            low_prob_positive_ev = [bet for bet in portfolio['singles'] 
                                   if bet['probability'] < 0.5 and bet['expected_profit'] > 0]
            
            # Combine both cases
            value_underdogs = high_prob_underdogs + low_prob_positive_ev
            
            if value_underdogs:
                # Create dataframe
                underdogs_df = pd.DataFrame(value_underdogs)
                underdogs_df['roi'] = underdogs_df['expected_profit'] / underdogs_df['amount'] * 100
                
                # Display table
                st.dataframe(
                    underdogs_df[['match', 'team', 'probability', 'odds', 'amount', 'potential_profit', 'roi']].rename(columns={
                        'match': 'Game', 
                        'team': 'Team',
                        'probability': 'Win Probability',
                        'odds': 'Decimal Odds',
                        'amount': 'Bet Amount($)',
                        'potential_profit': 'Potential Profit($)',
                        'roi': 'Expected ROI(%)'
                    }).style.format({
                        'Win Probability': '{:.1%}',
                        'Decimal Odds': '{:.2f}',
                        'Bet Amount($)': '${:.2f}',
                        'Potential Profit($)': '${:.2f}',
                        'Expected ROI(%)': '{:.1f}%'
                    }),
                    height=150
                )
                
                # Chart - Win Probability vs Decimal Odds
                fig = px.scatter(
                    underdogs_df, 
                    x='probability', 
                    y='odds', 
                    size='amount',
                    color='roi',
                    hover_name='team',
                    hover_data=['match', 'potential_profit', 'roi'],
                    title='Value Underdog Bet Analysis',
                    color_continuous_scale='RdYlGn',
                    labels={
                        'probability': 'Win Probability', 
                        'odds': 'Decimal Odds',
                        'amount': 'Bet Amount',
                        'roi': 'Expected ROI(%)'
                    }
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recommended Value Underdog bets found in the analysis.")
                
        # Display Random 3-Team Parlays - 새로 추가하는 탭
        with bet_tab7:
            if portfolio['singles']:
                import random
                import itertools
                import numpy as np
                
                st.markdown("""
                **Random 3-Team Parlays:**  
                These parlays are randomly created from single bets with odds of 1.2 or higher, with each pick appearing in up to 3 different combinations.
                """)
                
                # 1. 단일 베팅 데이터 얻기 (odds >= 1.2인 것으로 필터링)
                single_bets = [bet for bet in portfolio['singles'] if bet['odds'] >= 1.2]
                
                # 단일 베팅 정보 표시
                st.write(f"Number of single bets available (odds >= 1.2): {len(single_bets)}")
                st.write("Available picks:")
                for i, bet in enumerate(single_bets):
                    st.write(f"{i+1}. {bet['team']} in {bet['match']} (odds: {bet['odds']:.2f})")
                
                # 2. 모든 픽 정보 저장
                picks = []
                for i, bet in enumerate(single_bets):
                    picks.append({
                        'id': i,
                        'team': bet['team'],
                        'match': bet['match'],
                        'odds': bet['odds'],
                        'probability': bet['probability']
                    })
                
                # 3. 모든 가능한 3-팀 조합 생성 (nC3)
                all_combos = list(itertools.combinations(range(len(picks)), 3))
                st.write(f"Total possible 3-team combinations: {len(all_combos)}")
                
                # 4. 완전히 랜덤하게 섞기
                shuffled_combos = all_combos.copy()
                random.shuffle(shuffled_combos)
                
                # 5. 각 픽의 등장 횟수 추적 (초기화)
                pick_usage = {i: 0 for i in range(len(picks))}
                
                # 6. 결과 저장용 리스트
                selected_parlays = []
                
                # 7. 셔플된 조합을 순서대로 처리하며 각 픽이 최대 3번만 등장하도록 제한
                for combo in shuffled_combos:
                    can_add = True
                    
                    # 조합 내 각 픽의 현재 등장 횟수 확인
                    for pick_id in combo:
                        if pick_usage[pick_id] >= 3:  # 이미 3번 사용됐으면 이 조합 스킵
                            can_add = False
                            break
                    
                    if can_add:
                        # 파라레이 정보 생성
                        parlay_picks = [picks[i] for i in combo]
                        
                        # 배당률 및 확률 계산
                        odds = np.prod([pick['odds'] for pick in parlay_picks])
                        probability = np.prod([pick['probability'] for pick in parlay_picks])
                        boosted_odds = 1 + ((odds - 1) * 1.20)  # 20% 부스트
                        
                        # 결과에 추가
                        selected_parlays.append({
                            'picks': parlay_picks,
                            'odds': odds,
                            'boosted_odds': boosted_odds,
                            'probability': probability
                        })
                        
                        # 픽 사용 횟수 업데이트
                        for pick_id in combo:
                            pick_usage[pick_id] += 1
                
                # 결과 출력
                st.write(f"Number of parlays created: {len(selected_parlays)}")
                
                # 각 픽의 사용 횟수 표시
                st.write("### Pick Usage Counts")
                for i, pick in enumerate(picks):
                    st.write(f"{pick['team']}: {pick_usage[i]} times")
                
                if selected_parlays:
                    # 결과 테이블 생성
                    parlay_data = []
                    for i, parlay in enumerate(selected_parlays):
                        teams = [p['team'] for p in parlay['picks']]
                        # $5 베팅 가정
                        potential_profit = 5 * (parlay['boosted_odds'] - 1)
                        # ROI 계산 추가 (투자 대비 수익률)
                        roi = (parlay['probability'] * (parlay['boosted_odds'] - 1) - (1 - parlay['probability'])) * 100
                        
                        parlay_data.append({
                            'teams': ' & '.join(teams),
                            'probability': parlay['probability'],
                            'odds': parlay['boosted_odds'],
                            'potential_profit': potential_profit,
                            'roi': roi
                        })
                    
                    # 데이터프레임으로 변환
                    parlay_df = pd.DataFrame(parlay_data)
                    
                    # 테이블 표시 (ROI 열 추가)
                    st.dataframe(
                        parlay_df.rename(columns={
                            'teams': 'Teams',
                            'probability': 'Win Probability',
                            'odds': 'Odds (Boosted)',
                            'potential_profit': 'Potential Profit on $5 Bet',
                            'roi': 'Expected ROI(%)'
                        }).style.format({
                            'Win Probability': '{:.1%}',
                            'Odds (Boosted)': '{:.2f}',
                            'Potential Profit on $5 Bet': '${:.2f}',
                            'Expected ROI(%)': '{:.1f}%'
                        }),
                        height=400
                    )
                else:
                    st.warning("No parlays could be created with the given constraints.")
            else:
                st.info("No single bets available to create random parlays.")
                
        # Display Random 4-Team Parlays - 새로 추가하는 탭
        with bet_tab8:
            if portfolio['singles']:
                import random
                import itertools
                import numpy as np
                
                st.markdown("""
                **Random 4-Team Parlays:**  
                These parlays are randomly created from single bets with odds of 1.2 or higher, with each pick appearing in up to 3 different combinations.
                """)
                
                # 1. 단일 베팅 데이터 얻기 (odds >= 1.2인 것으로 필터링)
                single_bets = [bet for bet in portfolio['singles'] if bet['odds'] >= 1.2]
                
                # 단일 베팅 정보 표시
                st.write(f"Number of single bets available (odds >= 1.2): {len(single_bets)}")
                st.write("Available picks:")
                for i, bet in enumerate(single_bets):
                    st.write(f"{i+1}. {bet['team']} in {bet['match']} (odds: {bet['odds']:.2f})")
                
                # 2. 모든 픽 정보 저장
                picks = []
                for i, bet in enumerate(single_bets):
                    picks.append({
                        'id': i,
                        'team': bet['team'],
                        'match': bet['match'],
                        'odds': bet['odds'],
                        'probability': bet['probability']
                    })
                
                # 3. 모든 가능한 4-팀 조합 생성 (nC4)
                all_combos = list(itertools.combinations(range(len(picks)), 4))
                st.write(f"Total possible 4-team combinations: {len(all_combos)}")
                
                # 4. 완전히 랜덤하게 섞기
                shuffled_combos = all_combos.copy()
                random.shuffle(shuffled_combos)
                
                # 5. 각 픽의 등장 횟수 추적 (초기화)
                pick_usage = {i: 0 for i in range(len(picks))}
                
                # 6. 결과 저장용 리스트
                selected_parlays = []
                
                # 7. 셔플된 조합을 순서대로 처리하며 각 픽이 최대 3번만 등장하도록 제한
                for combo in shuffled_combos:
                    can_add = True
                    
                    # 조합 내 각 픽의 현재 등장 횟수 확인
                    for pick_id in combo:
                        if pick_usage[pick_id] >= 3:  # 이미 3번 사용됐으면 이 조합 스킵
                            can_add = False
                            break
                    
                    if can_add:
                        # 파라레이 정보 생성
                        parlay_picks = [picks[i] for i in combo]
                        
                        # 배당률 및 확률 계산
                        odds = np.prod([pick['odds'] for pick in parlay_picks])
                        probability = np.prod([pick['probability'] for pick in parlay_picks])
                        boosted_odds = 1 + ((odds - 1) * 1.30)  # 30% 부스트 (4팀은 더 큰 부스트)
                        
                        # 결과에 추가
                        selected_parlays.append({
                            'picks': parlay_picks,
                            'odds': odds,
                            'boosted_odds': boosted_odds,
                            'probability': probability
                        })
                        
                        # 픽 사용 횟수 업데이트
                        for pick_id in combo:
                            pick_usage[pick_id] += 1
                
                # 결과 출력
                st.write(f"Number of parlays created: {len(selected_parlays)}")
                
                # 각 픽의 사용 횟수 표시
                st.write("### Pick Usage Counts")
                for i, pick in enumerate(picks):
                    st.write(f"{pick['team']}: {pick_usage[i]} times")
                
                if selected_parlays:
                    # 결과 테이블 생성
                    parlay_data = []
                    for i, parlay in enumerate(selected_parlays):
                        teams = [p['team'] for p in parlay['picks']]
                        # $5 베팅 가정
                        potential_profit = 5 * (parlay['boosted_odds'] - 1)
                        # ROI 계산 추가 (투자 대비 수익률)
                        roi = (parlay['probability'] * (parlay['boosted_odds'] - 1) - (1 - parlay['probability'])) * 100
                        
                        parlay_data.append({
                            'teams': ' & '.join(teams),
                            'probability': parlay['probability'],
                            'odds': parlay['boosted_odds'],
                            'potential_profit': potential_profit,
                            'roi': roi
                        })
                    
                    # 데이터프레임으로 변환
                    parlay_df = pd.DataFrame(parlay_data)
                    
                    # 테이블 표시 (ROI 열 추가)
                    st.dataframe(
                        parlay_df.rename(columns={
                            'teams': 'Teams',
                            'probability': 'Win Probability',
                            'odds': 'Odds (Boosted)',
                            'potential_profit': 'Potential Profit on $5 Bet',
                            'roi': 'Expected ROI(%)'
                        }).style.format({
                            'Win Probability': '{:.1%}',
                            'Odds (Boosted)': '{:.2f}',
                            'Potential Profit on $5 Bet': '${:.2f}',
                            'Expected ROI(%)': '{:.1f}%'
                        }),
                        height=400
                    )
                else:
                    st.warning("No parlays could be created with the given constraints.")
            else:
                st.info("No single bets available to create random parlays.")
                
        # Parlay analysis chart - combine all parlays
        all_parlays = portfolio['parlays']
        if all_parlays:
            st.subheader("Parlay Visual Analysis")
            
            all_parlays_df = pd.DataFrame([
                {
                    'teams': ' & '.join(p['teams']),
                    'type': p['type'],
                    'probability': p['probability'],
                    'odds': p['boosted_odds'],
                    'amount': p['amount'],
                    'potential_profit': p['potential_profit'],
                    'roi': p['expected_profit'] / p['amount'] * 100 if p['amount'] > 0 else 0
                } for p in all_parlays
            ])
            
            # Chart - Parlay Odds vs Win Probability
            fig = px.scatter(
                all_parlays_df, 
                x='probability', 
                y='odds', 
                size='amount',
                color='type',
                hover_name='teams',
                hover_data=['amount', 'potential_profit', 'roi'],
                title='Parlay Bet Value Analysis',
                color_discrete_map={
                    '2_team_parlay': NBA_COLORS['primary'],
                    '3_team_parlay': NBA_COLORS['secondary']
                },
                labels={
                    'probability': 'Win Probability', 
                    'odds': 'Odds (Boosted)',
                    'amount': 'Bet Amount',
                    'type': 'Parlay Type',
                    'roi': 'Expected ROI(%)'
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click the 'Run Analysis' button in the sidebar to get betting recommendations.")

# Tab 3: Portfolio Analysis
with tab3:
    st.header("Portfolio Analysis")
    
    if 'portfolio' in st.session_state:
        portfolio = st.session_state['portfolio']
        bankroll = st.session_state['bankroll']
        
        # Capital allocation pie chart
        st.subheader("Capital Allocation")
        
        # Calculate total amounts for different bet types
        singles_total = sum(bet['amount'] for bet in portfolio['singles'])
        parlays_2team_total = sum(bet['amount'] for bet in portfolio['parlays'] if bet['type'] == '2_team_parlay')
        parlays_3team_total = sum(bet['amount'] for bet in portfolio['parlays'] if bet['type'] == '3_team_parlay')
        reserved_amount = bankroll - portfolio['total_investment']
        
        # Pie chart data
        allocation_data = pd.DataFrame([
            {'Category': 'Single Bets', 'Amount': singles_total},
            {'Category': '2-Team Parlays', 'Amount': parlays_2team_total},
            {'Category': '3-Team Parlays', 'Amount': parlays_3team_total},
            {'Category': 'Reserved Capital', 'Amount': reserved_amount}
        ])
        
        fig = px.pie(
            allocation_data,
            values='Amount',
            names='Category',
            title=f'Capital Allocation (Total ${bankroll:.2f})',
            color='Category',
            color_discrete_map={
                'Single Bets': NBA_COLORS['primary'],
                '2-Team Parlays': NBA_COLORS['secondary'],
                '3-Team Parlays': NBA_COLORS['accent'],
                'Reserved Capital': '#A9A9A9'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk-Reward Analysis
        st.subheader("Risk-Reward Analysis")
        
        # Combine all bets
        all_bets = []
        for bet in portfolio['singles']:
            all_bets.append({
                'type': 'Single Bet',
                'team': bet['team'],
                'probability': bet['probability'],
                'amount': bet['amount'],
                'potential_profit': bet['potential_profit'],
                'risk_amount': bet['risk_amount'],
                'expected_profit': bet['expected_profit'],
                'roi': bet['expected_profit'] / bet['amount'] * 100 if bet['amount'] > 0 else 0
            })
        
        for bet in portfolio['parlays']:
            all_bets.append({
                'type': '2-Team Parlay' if bet['type'] == '2_team_parlay' else '3-Team Parlay',
                'team': ' & '.join(bet['teams']),
                'probability': bet['probability'],
                'amount': bet['amount'],
                'potential_profit': bet['potential_profit'],
                'risk_amount': bet['risk_amount'],
                'expected_profit': bet['expected_profit'],
                'roi': bet['expected_profit'] / bet['amount'] * 100 if bet['amount'] > 0 else 0
            })
        
        all_bets_df = pd.DataFrame(all_bets)
        
        if not all_bets_df.empty:
            # Risk/Reward Analysis Chart
            fig = px.scatter(
                all_bets_df,
                x='risk_amount',
                y='potential_profit',
                size='amount',
                color='type',
                hover_name='team',
                hover_data=['probability', 'expected_profit', 'roi'],
                title='Risk-Reward Analysis',
                color_discrete_map={
                    'Single Bet': NBA_COLORS['primary'],
                    '2-Team Parlay': NBA_COLORS['secondary'],
                    '3-Team Parlay': NBA_COLORS['accent']
                },
                labels={
                    'risk_amount': 'Risk Amount ($)',
                    'potential_profit': 'Potential Profit ($)',
                    'amount': 'Bet Amount',
                    'type': 'Bet Type',
                    'roi': 'Expected ROI(%)'
                }
            )
            # Add 45-degree reference line (risk:reward = 1:1)
            max_val = max(all_bets_df['risk_amount'].max(), all_bets_df['potential_profit'].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='1:1 Risk-Reward'
                )
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # ROI by Bet Type (boxplot) - Changed from histogram to boxplot
            fig2 = px.box(
                all_bets_df,
                x='type',
                y='roi',
                color='type',
                title='Expected ROI by Bet Type',
                color_discrete_map={
                    'Single Bet': NBA_COLORS['primary'],
                    '2-Team Parlay': NBA_COLORS['secondary'],
                    '3-Team Parlay': NBA_COLORS['accent']
                },
                points="all"
            )

            fig2.update_layout(
                height=400,
                xaxis_title="Bet Type",
                yaxis_title="Expected ROI (%)",
                showlegend=False
            )

            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Click the 'Run Analysis' button in the sidebar to view portfolio analysis.")

# Footer
st.markdown("""
---
### Notes
- The Kelly Criterion is used for bet sizing but is applied conservatively to prevent high volatility.
- Betting results improve as the model's predicted probabilities align more closely with actual outcomes.
- Always bet responsibly. This tool is intended to assist in decision-making only.
""")

# Key terms explanation
with st.expander("Key Terms Explained"):
    st.markdown("""
    - **Odds**: The ratio of payout received upon a successful bet. Example: Odds of 2.0 return twice your stake (including the original stake).
    - **American Odds**: Odds format that shows how much you would win on a $100 bet (e.g. +150) or how much you need to bet to win $100 (e.g. -110).
    - **Parlay**: A bet that combines multiple selections into one wager. All selections must win for the bet to succeed. Offers higher payouts but carries higher risk.
    - **Kelly Criterion**: A mathematical formula for determining optimal bet size to maximize long-term capital growth.
    - **Expected Value (EV)**: The mathematical expectation of a bet. A positive EV indicates long-term profitability.
    - **ROI (Return on Investment)**: The percentage return on invested capital, calculated as expected profit divided by bet amount.
    """)

def main():
    """Legacy main function - code is now in the main script body"""
    pass  # All functionality has been moved to the main script body

if __name__ == "__main__":
    # Everything is now in the main script body
    pass 