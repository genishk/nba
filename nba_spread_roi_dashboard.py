#!/usr/bin/env python3
"""
NBA Spread ROI Dashboard
- Streamlit 기반 인터랙티브 대시보드
- Moneyline vs Spread ROI 비교 시각화
- 팀별 최적 전략 제시
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class NBASpreadROIDashboard:
    """NBA Spread ROI 대시보드"""
    
    # Odds 구간 순서 정의 (Favorite → Underdog)
    ODDS_RANGE_ORDER = [
        "Overwhelming Favorite (-501+)",
        "Heavy Favorite (-500 to -301)",
        "Strong Favorite (-300 to -201)",
        "Moderate Favorite (-200 to -151)",
        "Slight Favorite (-150 to -100)",
        "Slight Underdog (+100 to +150)",
        "Moderate Underdog (+151 to +200)",
        "Strong Underdog (+201 to +300)",
        "Heavy Underdog (+301 to +500)",
        "Overwhelming Underdog (+501+)"
    ]
    
    def __init__(self):
        """초기화"""
        self.project_root = Path(__file__).parent
        self.analysis_dir = self.project_root / "data" / "roi_analysis"
        
        # 페이지 설정
        st.set_page_config(
            page_title="NBA Spread ROI Analysis",
            page_icon="🏀",
            layout="wide"
        )
    
    def find_latest_file(self, directory: Path, pattern: str):
        """디렉토리에서 가장 최신 파일 찾기"""
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)
    
    def load_matched_data(self) -> List[Dict]:
        """매칭된 원본 데이터 로드 (날짜 필터링용)"""
        matched_dir = self.project_root / "data" / "spread_matched"
        matched_file = self.find_latest_file(matched_dir, "nba_spread_matched_*.json")
        
        if not matched_file or not matched_file.exists():
            st.error("❌ Matched data not found.")
            return []
        
        with open(matched_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def filter_data_by_date(self, matched_data: List[Dict], start_date: str, end_date: str) -> List[Dict]:
        """날짜 범위로 데이터 필터링"""
        filtered = [game for game in matched_data if start_date <= game['date'] <= end_date]
        return filtered
    
    def recalculate_analysis(self, matched_data: List[Dict]) -> Dict:
        """필터링된 데이터로 ROI 재계산"""
        from nba_spread_roi_analyzer import NBASpreadROIAnalyzer
        
        analyzer = NBASpreadROIAnalyzer()
        
        # Odds Range별 분석
        range_analysis = analyzer.analyze_by_odds_range(matched_data)
        
        # 팀별 분석
        team_analysis = analyzer.analyze_by_team(matched_data)
        
        return {
            'by_odds_range': range_analysis,
            'by_team': team_analysis
        }
    
    def create_odds_range_comparison(self, range_analysis: Dict):
        """Odds Range별 Moneyline vs Best Spread 비교 차트"""
        st.header("📊 ROI by Moneyline Odds Range")
        
        # 데이터 준비 (순서대로)
        ranges = []
        ml_rois = []
        best_spread_rois = []
        best_spreads = []
        improvements = []
        
        for range_label in self.ODDS_RANGE_ORDER:
            if range_label not in range_analysis:
                continue
            
            data = range_analysis[range_label]
            ml_roi = data['moneyline']
            
            if ml_roi['total_bets'] == 0:
                continue
            
            # 최고 ROI spread 찾기
            best_spread = None
            best_roi = ml_roi['roi']
            
            for spread_point, spread_roi in data['spreads'].items():
                if spread_roi['total_bets'] >= 5 and spread_roi['roi'] > best_roi:
                    best_spread = spread_point
                    best_roi = spread_roi['roi']
            
            ranges.append(range_label)
            ml_rois.append(ml_roi['roi'])
            best_spread_rois.append(best_roi)
            best_spreads.append(best_spread if best_spread else 'ML')
            improvements.append(best_roi - ml_roi['roi'])
        
        # 차트 생성
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Moneyline',
            x=ranges,
            y=ml_rois,
            text=[f"{roi:.1f}%" for roi in ml_rois],
            textposition='auto',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Best Strategy',
            x=ranges,
            y=best_spread_rois,
            text=[f"{roi:.1f}%<br>({spread})" for roi, spread in zip(best_spread_rois, best_spreads)],
            textposition='auto',
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title="Moneyline vs Best Strategy ROI Comparison",
            xaxis_title="Odds Range",
            yaxis_title="ROI (%)",
            barmode='group',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 개선율 차트
        st.subheader("💰 ROI Improvement Over Moneyline")
        
        fig_improvement = go.Figure()
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        fig_improvement.add_trace(go.Bar(
            x=ranges,
            y=improvements,
            text=[f"{imp:+.1f}%" for imp in improvements],
            textposition='auto',
            marker_color=colors
        ))
        
        fig_improvement.update_layout(
            title="ROI Improvement by Using Best Strategy",
            xaxis_title="Odds Range",
            yaxis_title="Improvement (%)",
            height=400,
            hovermode='x'
        )
        
        st.plotly_chart(fig_improvement, use_container_width=True)
        
        # 📋 요약 테이블 추가
        st.subheader("📋 Summary Table")
        
        table_data = []
        for i, range_label in enumerate(ranges):
            table_data.append({
                'Odds Range': range_label,
                'ML Bets': range_analysis[range_label]['moneyline']['total_bets'],
                'ML Win%': f"{range_analysis[range_label]['moneyline']['win_rate']:.1f}%",
                'ML ROI': f"{ml_rois[i]:.2f}%",
                'Best Strategy': best_spreads[i],
                'Best ROI': f"{best_spread_rois[i]:.2f}%",
                'Improvement': f"{improvements[i]:+.2f}%"
            })
        
        df_table = pd.DataFrame(table_data)
        
        # 스타일링
        def highlight_positive(val):
            """양수는 초록색, 음수는 빨간색"""
            if isinstance(val, str) and '%' in val:
                num = float(val.replace('%', '').replace('+', ''))
                if num > 0:
                    return 'background-color: #90EE90'
                elif num < 0:
                    return 'background-color: #FFB6C6'
            return ''
        
        styled_df = df_table.style.applymap(highlight_positive, subset=['ML ROI', 'Best ROI', 'Improvement'])
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    def create_odds_range_details(self, range_analysis: Dict):
        """Odds Range별 상세 정보 테이블"""
        st.header("📋 Detailed Statistics by Odds Range")
        
        for range_label in self.ODDS_RANGE_ORDER:
            if range_label not in range_analysis:
                continue
            
            data = range_analysis[range_label]
            ml_roi = data['moneyline']
            
            if ml_roi['total_bets'] == 0:
                continue
            
            with st.expander(f"📊 {range_label}"):
                # Moneyline 정보
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ML Bets", ml_roi['total_bets'])
                with col2:
                    st.metric("ML Win Rate", f"{ml_roi['win_rate']:.1f}%")
                with col3:
                    st.metric("ML ROI", f"{ml_roi['roi']:.2f}%")
                with col4:
                    st.metric("ML Profit", f"${ml_roi['profit']:.2f}")
                
                st.markdown("---")
                
                # Spread 상세 테이블
                st.subheader("All Spreads Performance")
                
                # Spread ROI 정렬
                spread_items = [(sp, roi) for sp, roi in data['spreads'].items() if roi['total_bets'] >= 3]
                spread_items.sort(key=lambda x: x[1]['roi'], reverse=True)
                
                if spread_items:
                    spread_table = []
                    for spread_point, spread_roi in spread_items:
                        spread_table.append({
                            'Spread': spread_point,
                            'Bets': spread_roi['total_bets'],
                            'Wins': spread_roi['wins'],
                            'Losses': spread_roi['losses'],
                            'Win Rate': f"{spread_roi['win_rate']:.1f}%",
                            'ROI': f"{spread_roi['roi']:.2f}%",
                            'Profit': f"${spread_roi['profit']:.2f}",
                            'vs ML': f"{spread_roi['roi'] - ml_roi['roi']:+.2f}%"
                        })
                    
                    df_spreads = pd.DataFrame(spread_table)
                    
                    # 스타일링
                    def color_roi(val):
                        if isinstance(val, str) and '%' in val:
                            num = float(val.replace('%', '').replace('+', ''))
                            if num > 0:
                                return 'background-color: #90EE90'
                            elif num < 0:
                                return 'background-color: #FFB6C6'
                        return ''
                    
                    styled_spreads = df_spreads.style.applymap(color_roi, subset=['ROI', 'vs ML'])
                    st.dataframe(styled_spreads, use_container_width=True)
                else:
                    st.info("No spread data available (minimum 3 bets required)")
    
    def create_team_analysis(self, team_analysis: Dict):
        """팀별 분석"""
        st.header("🏀 Team Analysis")
        
        # 전체 팀 테이블
        st.subheader("📊 All Teams Summary")
        
        team_table = []
        for team, data in team_analysis.items():
            ml_data = data['moneyline']
            best_strat = data['best_strategy']
            
            team_table.append({
                'Team': team,
                'Total Bets': ml_data['total_bets'],
                'ML Win%': f"{ml_data['win_rate']:.1f}%",
                'ML ROI': f"{ml_data['roi']:.2f}%",
                'Best Strategy': f"{best_strat['type'].upper()}" + (f" {best_strat['spread']}" if best_strat['spread'] else ""),
                'Best ROI': f"{best_strat['roi']:.2f}%",
                'Improvement': f"{best_strat['roi'] - ml_data['roi']:+.2f}%"
            })
        
        df_teams = pd.DataFrame(team_table)
        df_teams = df_teams.sort_values('Best ROI', ascending=False, key=lambda x: x.str.replace('%', '').astype(float))
        
        # 스타일링
        def color_metric(val):
            if isinstance(val, str) and '%' in val:
                num = float(val.replace('%', '').replace('+', ''))
                if num > 0:
                    return 'background-color: #90EE90'
                elif num < 0:
                    return 'background-color: #FFB6C6'
            return ''
        
        styled_teams = df_teams.style.applymap(color_metric, subset=['ML ROI', 'Best ROI', 'Improvement'])
        st.dataframe(styled_teams, use_container_width=True, height=600)
        
        st.markdown("---")
        
        # Top 10 팀 (ROI 기준)
        team_rois = [(team, data['best_strategy']['roi']) for team, data in team_analysis.items()]
        team_rois.sort(key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 10 Teams by ROI")
            
            top_teams = team_rois[:10]
            teams = [t[0] for t in top_teams]
            rois = [t[1] for t in top_teams]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=teams,
                y=rois,
                text=[f"{roi:.1f}%" for roi in rois],
                textposition='auto',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title="Top 10 Teams by Best Strategy ROI",
                xaxis_title="Team",
                yaxis_title="ROI (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📉 Bottom 10 Teams by ROI")
            
            bottom_teams = team_rois[-10:]
            teams = [t[0] for t in bottom_teams]
            rois = [t[1] for t in bottom_teams]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=teams,
                y=rois,
                text=[f"{roi:.1f}%" for roi in rois],
                textposition='auto',
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="Bottom 10 Teams by Best Strategy ROI",
                xaxis_title="Team",
                yaxis_title="ROI (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 팀별 상세 정보
        st.subheader("📊 Team Details")
        
        # 팀 선택
        selected_team = st.selectbox(
            "Select a team:",
            options=sorted(team_analysis.keys())
        )
        
        if selected_team:
            team_data = team_analysis[selected_team]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Strategy", 
                         f"{team_data['best_strategy']['type'].upper()}" + 
                         (f" {team_data['best_strategy']['spread']}" if team_data['best_strategy']['spread'] else ""))
                st.metric("Best ROI", f"{team_data['best_strategy']['roi']:.2f}%")
            
            with col2:
                ml_data = team_data['moneyline']
                st.metric("ML Total Bets", ml_data['total_bets'])
                st.metric("ML Win Rate", f"{ml_data['win_rate']:.1f}%")
                st.metric("ML ROI", f"{ml_data['roi']:.2f}%")
            
            with col3:
                st.metric("ML Profit", f"${ml_data['profit']:.2f}")
            
            # Spread별 ROI 차트
            st.subheader(f"📈 {selected_team} - Spread ROI Comparison")
            
            spread_points = []
            spread_rois = []
            
            for spread_point, spread_roi in team_data['spreads'].items():
                if spread_roi['total_bets'] >= 3:  # 최소 3번 이상
                    spread_points.append(float(spread_point))
                    spread_rois.append(spread_roi['roi'])
            
            if spread_points:
                df = pd.DataFrame({
                    'Spread': spread_points,
                    'ROI': spread_rois
                })
                df = df.sort_values('Spread')
                
                fig = go.Figure()
                
                # Moneyline ROI 기준선
                fig.add_hline(y=ml_data['roi'], line_dash="dash", 
                            line_color="blue", annotation_text="Moneyline ROI")
                
                # Spread ROI
                fig.add_trace(go.Scatter(
                    x=df['Spread'],
                    y=df['ROI'],
                    mode='lines+markers',
                    name='Spread ROI',
                    line=dict(color='green', width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f"{selected_team} - ROI by Spread Point",
                    xaxis_title="Spread Point",
                    yaxis_title="ROI (%)",
                    height=400,
                    hovermode='x'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def create_summary_metrics(self, range_analysis: Dict, team_analysis: Dict):
        """전체 요약 메트릭"""
        st.header("📈 Overall Summary")
        
        # 전체 통계 계산
        total_bets = 0
        total_ml_profit = 0
        total_best_profit = 0
        
        for range_label, data in range_analysis.items():
            ml_roi = data['moneyline']
            total_bets += ml_roi['total_bets']
            total_ml_profit += ml_roi['profit']
            
            # 최고 spread 찾기
            best_profit = ml_roi['profit']
            for spread_point, spread_roi in data['spreads'].items():
                if spread_roi['total_bets'] >= 5 and spread_roi['profit'] > best_profit:
                    best_profit = spread_roi['profit']
            
            total_best_profit += best_profit
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Bets Analyzed", total_bets)
        
        with col2:
            st.metric("Moneyline Total Profit", f"${total_ml_profit:.2f}")
        
        with col3:
            st.metric("Best Strategy Total Profit", f"${total_best_profit:.2f}")
        
        with col4:
            improvement = total_best_profit - total_ml_profit
            st.metric("Profit Improvement", f"${improvement:.2f}", 
                     delta=f"{(improvement/abs(total_ml_profit)*100) if total_ml_profit != 0 else 0:.1f}%")
    
    def run(self):
        """대시보드 실행"""
        st.title("🏀 NBA Spread ROI Analysis Dashboard")
        st.markdown("---")
        
        # 원본 데이터 로드
        matched_data = self.load_matched_data()
        
        if not matched_data:
            return
        
        # 날짜 범위 추출
        all_dates = sorted(set(game['date'] for game in matched_data))
        min_date = all_dates[0]
        max_date = all_dates[-1]
        
        # 사이드바: 날짜 필터
        st.sidebar.header("📅 Date Range Filter")
        st.sidebar.info(f"Available data: {min_date} to {max_date}")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime(min_date),
                min_value=pd.to_datetime(min_date),
                max_value=pd.to_datetime(max_date)
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime(max_date),
                min_value=pd.to_datetime(min_date),
                max_value=pd.to_datetime(max_date)
            )
        
        # 날짜를 문자열로 변환
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # 데이터 필터링
        if start_date_str > end_date_str:
            st.error("❌ Start date must be before end date!")
            return
        
        with st.spinner('🔄 Filtering and recalculating...'):
            filtered_data = self.filter_data_by_date(matched_data, start_date_str, end_date_str)
            
            if not filtered_data:
                st.error(f"❌ No data found for the selected date range ({start_date_str} to {end_date_str})")
                return
            
            # ROI 재계산
            analysis_data = self.recalculate_analysis(filtered_data)
        
        range_analysis = analysis_data['by_odds_range']
        team_analysis = analysis_data['by_team']
        
        # 필터 정보 표시
        st.success(f"✅ Showing data from **{start_date_str}** to **{end_date_str}** ({len(filtered_data)} games)")
        st.markdown("---")
        
        # 요약 메트릭
        self.create_summary_metrics(range_analysis, team_analysis)
        
        st.markdown("---")
        
        # Odds Range 분석
        self.create_odds_range_comparison(range_analysis)
        
        st.markdown("---")
        
        # Odds Range 상세
        self.create_odds_range_details(range_analysis)
        
        st.markdown("---")
        
        # 팀별 분석
        self.create_team_analysis(team_analysis)
        
        st.markdown("---")
        
        # Footer
        st.info("""
        💡 **Key Insights:**
        - Compare Moneyline vs Spread betting strategies across different odds ranges
        - Identify which spread points offer the best ROI for each odds range
        - Discover team-specific betting opportunities
        - Optimize your betting strategy based on data-driven analysis
        """)


def main():
    """메인 실행 함수"""
    dashboard = NBASpreadROIDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

