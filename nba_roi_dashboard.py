#!/usr/bin/env python3
"""
NBA ROI Dashboard (Streamlit)
- 팀별 ROI 분석 시각화
- 기간별 비교
- 인터랙티브 차트 및 테이블
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nba_roi_analyzer import NBAROIAnalyzer
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="NBA ROI Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐시된 데이터 로드
@st.cache_data
def load_analyzer():
    """Analyzer 로드 (캐싱)"""
    return NBAROIAnalyzer()

@st.cache_data
def get_period_analysis(period):
    """기간별 분석 결과 (캐싱)"""
    analyzer = load_analyzer()
    return analyzer.get_all_teams_analysis(period)

@st.cache_data
def get_composite_rankings():
    """통합 순위 (캐싱)"""
    analyzer = load_analyzer()
    return analyzer.get_composite_rankings()

# Analyzer 초기화
try:
    analyzer = load_analyzer()
    data_summary = analyzer.get_data_summary()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ============================================================
# Header
# ============================================================
st.title("🏀 NBA Moneyline ROI Dashboard")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 데이터 요약
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Total Games", f"{data_summary['total_games']:,}")
with col2:
    st.metric("🏀 Total Teams", data_summary['total_teams'])
with col3:
    st.metric("📅 Date Range", f"{data_summary['date_range']['start']} ~ {data_summary['date_range']['end']}")

st.divider()

# ============================================================
# Sidebar - Period Selection
# ============================================================
st.sidebar.header("⚙️ Settings")

period = st.sidebar.selectbox(
    "📅 Analysis Period",
    options=['season', '30days', '14days', '7days'],
    format_func=lambda x: {
        'season': '🏆 Full Season',
        '30days': '📊 Last 30 Days',
        '14days': '📈 Last 14 Days',
        '7days': '⚡ Last 7 Days'
    }[x],
    index=0
)

# 기간별 데이터 로드
df_analysis = get_period_analysis(period)

# ============================================================
# Composite Rankings (NEW!)
# ============================================================
st.header("🏆 Composite Rankings")
st.markdown("**Weighted Average:** 7 Days (50%) + 14 Days (30%) + 30 Days (20%)")

composite_df = get_composite_rankings()

# Top 20 Composite Rankings
st.subheader("🔥 Top 20 Teams (Composite Score)")

top_20_composite = composite_df.head(20).copy()

# 스타일링 함수
def highlight_trend(val):
    if val == '🔥':
        return 'background-color: #ffcccc; font-weight: bold'
    elif val == '↗️':
        return 'background-color: #ccffcc'
    elif val == '↘️':
        return 'background-color: #ffffcc'
    return ''

def highlight_rank(val):
    if isinstance(val, (int, float)):
        if val <= 5:
            return 'background-color: #90EE90; font-weight: bold'
        elif val <= 10:
            return 'background-color: #FFFFE0'
        elif val >= 25:
            return 'background-color: #FFB6C1'
    return ''

def highlight_roi(val):
    """ROI 컬럼 색상 (positive=초록, negative=빨강)"""
    if isinstance(val, (int, float)):
        if val > 0:
            return 'background-color: #ccffcc'
        elif val < 0:
            return 'background-color: #ffcccc'
    return ''

# 테이블 표시
st.dataframe(
    top_20_composite[[
        'composite_rank', 'team', 'composite_score', 'trend',
        '7days_rank', '7days_roi',
        '14days_rank', '14days_roi',
        '30days_rank', '30days_roi'
    ]].style.format({
        'composite_score': '{:.2f}',
        '7days_roi': '{:+.2f}%',
        '14days_roi': '{:+.2f}%',
        '30days_roi': '{:+.2f}%'
    }).applymap(highlight_trend, subset=['trend'])
    .applymap(highlight_rank, subset=['7days_rank', '14days_rank', '30days_rank'])
    .applymap(highlight_roi, subset=['7days_roi', '14days_roi', '30days_roi']),
    hide_index=True,
    use_container_width=True,
    height=750
)

# Composite Score 시각화
st.subheader("📊 Composite Score Distribution (Lower is Better)")

fig_composite = px.bar(
    composite_df.head(20),
    x='team',
    y='composite_score',
    title='Top 20 Teams - Composite Score',
    labels={'composite_score': 'Composite Score (Lower = Better)', 'team': 'Team'},
    color='composite_score',
    color_continuous_scale='RdYlGn_r',  # 낮을수록 초록색
    height=450
)
fig_composite.update_layout(showlegend=False)
st.plotly_chart(fig_composite, use_container_width=True)

# 기간별 순위 비교 히트맵
st.subheader("🗺️ Period Rankings Heatmap (Top 20)")

heatmap_data = composite_df.head(20)[['team', '7days_rank', '14days_rank', '30days_rank']].set_index('team')
heatmap_data.columns = ['7 Days', '14 Days', '30 Days']

fig_heatmap = px.imshow(
    heatmap_data.T,
    labels=dict(x="Team", y="Period", color="Rank"),
    x=heatmap_data.index,
    y=heatmap_data.columns,
    color_continuous_scale='RdYlGn_r',
    aspect="auto",
    height=300
)
fig_heatmap.update_xaxes(side="bottom")
st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# ============================================================
# Top/Bottom Performers (Period-Specific)
# ============================================================
st.header(f"📊 ROI Rankings - {period.replace('days', ' Days').title()}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Top 10 Teams")
    top_10 = df_analysis.head(10).copy()
    top_10['rank'] = range(1, len(top_10) + 1)
    
    # 스타일링
    def highlight_positive(val):
        if isinstance(val, (int, float)):
            color = 'lightgreen' if val > 0 else 'lightcoral' if val < 0 else 'white'
            return f'background-color: {color}'
        return ''
    
    st.dataframe(
        top_10[['rank', 'team', 'games', 'wins', 'win_rate', 'avg_roi', 'total_roi']].style.format({
            'win_rate': '{:.1f}%',
            'avg_roi': '{:+.2f}%',
            'total_roi': '{:+.1f}%'
        }).applymap(highlight_positive, subset=['avg_roi', 'total_roi']),
        hide_index=True,
        use_container_width=True
    )

with col2:
    st.subheader("📉 Bottom 10 Teams")
    bottom_10 = df_analysis.tail(10).copy()
    bottom_10['rank'] = range(len(df_analysis) - 9, len(df_analysis) + 1)
    
    st.dataframe(
        bottom_10[['rank', 'team', 'games', 'wins', 'win_rate', 'avg_roi', 'total_roi']].style.format({
            'win_rate': '{:.1f}%',
            'avg_roi': '{:+.2f}%',
            'total_roi': '{:+.1f}%'
        }).applymap(highlight_positive, subset=['avg_roi', 'total_roi']),
        hide_index=True,
        use_container_width=True
    )

st.divider()

# ============================================================
# ROI Visualization
# ============================================================
st.header("📈 ROI Visualization")

# Average ROI Bar Chart
fig_roi = px.bar(
    df_analysis,
    x='team',
    y='avg_roi',
    title='Average ROI by Team',
    labels={'avg_roi': 'Average ROI (%)', 'team': 'Team'},
    color='avg_roi',
    color_continuous_scale=['red', 'yellow', 'green'],
    height=500
)
fig_roi.update_layout(showlegend=False)
st.plotly_chart(fig_roi, use_container_width=True)

st.divider()

# ============================================================
# Home vs Away Performance
# ============================================================
st.header("🏠 Home vs Away Performance")

# Home/Away 비교 데이터 준비
home_away_data = df_analysis[df_analysis['games'] >= 5].copy()  # 최소 5경기 이상

fig_home_away = go.Figure()

fig_home_away.add_trace(go.Bar(
    name='Home ROI',
    x=home_away_data['team'],
    y=home_away_data['home_roi'],
    marker_color='lightblue'
))

fig_home_away.add_trace(go.Bar(
    name='Away ROI',
    x=home_away_data['team'],
    y=home_away_data['away_roi'],
    marker_color='lightcoral'
))

fig_home_away.update_layout(
    title='Home vs Away Average ROI',
    xaxis_title='Team',
    yaxis_title='Average ROI (%)',
    barmode='group',
    height=500
)

st.plotly_chart(fig_home_away, use_container_width=True)

st.divider()

# ============================================================
# Team Detail Analysis
# ============================================================
st.header("🔍 Team Detail Analysis")

selected_team = st.selectbox(
    "Select Team",
    options=sorted(df_analysis['team'].unique()),
    index=0
)

if selected_team:
    team_detail = analyzer.get_team_detail(selected_team, period)
    
    # 팀 요약
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Games Played",
            team_detail['overall']['games'],
            delta=f"{team_detail['overall']['wins']} wins"
        )
    
    with col2:
        st.metric(
            "Win Rate",
            f"{team_detail['overall']['win_rate']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Avg ROI",
            f"{team_detail['overall']['avg_roi']:+.2f}%",
            delta=f"Total: {team_detail['overall']['total_roi']:+.1f}%"
        )
    
    with col4:
        st.metric(
            "Avg Odds",
            f"{team_detail['overall']['avg_odds']:+.0f}"
        )
    
    # Home vs Away 상세
    st.subheader(f"📊 {selected_team} Home/Away Split")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Home Performance")
        st.metric("Games", team_detail['home']['games'])
        st.metric("Wins", team_detail['home']['wins'])
        st.metric("Win Rate", f"{team_detail['home']['win_rate']:.1f}%")
        st.metric("Avg ROI", f"{team_detail['home']['avg_roi']:+.2f}%")
    
    with col2:
        st.markdown("### ✈️ Away Performance")
        st.metric("Games", team_detail['away']['games'])
        st.metric("Wins", team_detail['away']['wins'])
        st.metric("Win Rate", f"{team_detail['away']['win_rate']:.1f}%")
        st.metric("Avg ROI", f"{team_detail['away']['avg_roi']:+.2f}%")
    
    # ROI Trend
    st.subheader(f"📈 {selected_team} Cumulative ROI Trend")
    
    trend_data = analyzer.get_roi_trend(selected_team, period)
    
    if len(trend_data) > 0:
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=trend_data['date'],
            y=trend_data['cumulative_roi'],
            mode='lines+markers',
            name='Cumulative ROI',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))
        
        fig_trend.update_layout(
            title=f'{selected_team} Cumulative ROI Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative ROI (%)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Recent Games
    st.subheader(f"📋 {selected_team} Recent Games (Last 10)")
    
    recent_games_df = pd.DataFrame(team_detail['recent_games'])
    if len(recent_games_df) > 0:
        recent_games_df['date'] = pd.to_datetime(recent_games_df['date']).dt.strftime('%Y-%m-%d')
        recent_games_df['result'] = recent_games_df['team_won'].apply(lambda x: '✅ Win' if x else '❌ Loss')
        
        st.dataframe(
            recent_games_df[['date', 'opponent', 'location', 'result', 'team_roi']].style.format({
                'team_roi': '{:+.2f}%'
            }).applymap(
                lambda x: 'background-color: lightgreen' if '✅' in str(x) else 'background-color: lightcoral' if '❌' in str(x) else '',
                subset=['result']
            ),
            hide_index=True,
            use_container_width=True
        )

st.divider()

# ============================================================
# Full Data Table
# ============================================================
st.header("📋 Complete Team Statistics")

# 컬럼 선택
show_columns = st.multiselect(
    "Select columns to display",
    options=['team', 'games', 'wins', 'win_rate', 'avg_roi', 'total_roi', 'avg_odds', 
             'home_games', 'home_roi', 'away_games', 'away_roi', 'best_roi', 'worst_roi'],
    default=['team', 'games', 'wins', 'win_rate', 'avg_roi', 'total_roi']
)

if show_columns:
    display_df = df_analysis[show_columns].copy()
    
    # 포맷팅
    format_dict = {}
    for col in show_columns:
        if 'rate' in col or 'roi' in col:
            if col == 'win_rate':
                format_dict[col] = '{:.1f}%'
            else:
                format_dict[col] = '{:+.2f}%'
        elif col == 'avg_odds':
            format_dict[col] = '{:+.0f}'
    
    st.dataframe(
        display_df.style.format(format_dict).applymap(
            highlight_positive,
            subset=[col for col in show_columns if 'roi' in col]
        ),
        hide_index=True,
        use_container_width=True,
        height=600
    )

# Footer
st.divider()
st.markdown("---")
st.caption(f"Data source: {data_summary['data_file']}")
st.caption("💡 Tip: Use the sidebar to change the analysis period")

