import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import logging

class ModelPerformanceAnalyzer:
    """모델 성과 분석 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.merged_dir = self.project_root / 'data' / 'merged'
        self.records_dir = self.project_root / 'data' / 'records'
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ModelPerformanceAnalyzer')
        
        # 모델 리스트
        self.models = [f'model{i}' for i in range(1, 8)] + ['ensemble']
        
        # 팀명 매핑 (다양한 형식 처리)
        self.team_abbrev_map = {
            'Anaheim Ducks': 'ANA', 'Arizona Coyotes': 'ARI', 'Boston Bruins': 'BOS',
            'Buffalo Sabres': 'BUF', 'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR',
            'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ',
            'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM',
            'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN',
            'Montreal Canadiens': 'MTL', 'Nashville Predators': 'NSH', 'New Jersey Devils': 'NJD',
            'New York Islanders': 'NYI', 'New York Rangers': 'NYR', 'Ottawa Senators': 'OTT',
            'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJS',
            'Seattle Kraken': 'SEA', 'St. Louis Blues': 'STL', 'Tampa Bay Lightning': 'TBL',
            'Toronto Maple Leafs': 'TOR', 'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK',
            'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG', 'Utah Mammoth': 'UTA',
            'Montréal Canadiens': 'MTL'
        }
    
    def load_merged_predictions(self, exclude_today: bool = True) -> pd.DataFrame:
        """병합된 예측 파일들 로드 (오늘 제외 가능)"""
        merged_files = sorted(self.merged_dir.glob('nhl_merged_predictions_odds_*.json'))
        
        if not merged_files:
            self.logger.error("병합 파일을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        all_predictions = []
        today = datetime.now().strftime('%Y%m%d')
        
        for file in merged_files:
            # 파일명에서 날짜 추출
            file_date = file.stem.split('_')[-2]
            
            # 오늘 파일 제외 옵션
            if exclude_today and file_date == today:
                self.logger.info(f"오늘 파일 제외: {file.name}")
                continue
            
            self.logger.info(f"로드 중: {file.name}")
            
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_predictions.extend(data)
        
        df = pd.DataFrame(all_predictions)
        
        if df.empty:
            return df
        
        # 날짜 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 팀명 약어 추가
        df['home_team_abbrev'] = df['home_team_name'].map(self.team_abbrev_map)
        df['away_team_abbrev'] = df['away_team_name'].map(self.team_abbrev_map)
        
        self.logger.info(f"총 {len(df)}개 예측 로드 완료")
        return df
    
    def load_historical_records(self) -> pd.DataFrame:
        """과거 경기 결과 로드"""
        records_file = self.records_dir / 'nhl_historical_records_20251117_112445.json'
        
        if not records_file.exists():
            # 가장 최신 파일 찾기
            records_files = sorted(self.records_dir.glob('nhl_historical_records_*.json'))
            if not records_files:
                self.logger.error("경기 결과 파일을 찾을 수 없습니다.")
                return pd.DataFrame()
            records_file = records_files[-1]
        
        self.logger.info(f"경기 결과 로드 중: {records_file.name}")
        
        with open(records_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        df = pd.DataFrame(records)
        
        if df.empty:
            return df
        
        # 날짜 변환
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        self.logger.info(f"총 {len(df)}개 경기 결과 로드 완료")
        return df
    
    def match_predictions_with_results(self, predictions_df: pd.DataFrame, 
                                      records_df: pd.DataFrame) -> pd.DataFrame:
        """예측과 실제 결과 매칭"""
        
        if predictions_df.empty or records_df.empty:
            self.logger.error("데이터가 비어있습니다.")
            return pd.DataFrame()
        
        matched_data = []
        unmatched_count = 0
        
        for idx, pred in predictions_df.iterrows():
            game_date = pred['date'].strftime('%Y-%m-%d')
            home_abbrev = pred['home_team_abbrev']
            away_abbrev = pred['away_team_abbrev']
            
            # 날짜 + 팀으로 매칭
            mask = (
                (records_df['date'].dt.strftime('%Y-%m-%d') == game_date) &
                (records_df['home_team_abbrev'] == home_abbrev) &
                (records_df['away_team_abbrev'] == away_abbrev)
            )
            
            matched_records = records_df[mask]
            
            if len(matched_records) > 0:
                result = matched_records.iloc[0]
                
                # 예측 데이터에 실제 결과 추가
                matched_game = pred.to_dict()
                matched_game['actual_home_win'] = int(result['home_win'])
                matched_game['actual_home_score'] = result['home_score']
                matched_game['actual_away_score'] = result['away_score']
                
                matched_data.append(matched_game)
            else:
                unmatched_count += 1
                self.logger.debug(f"매칭 실패: {game_date} - {home_abbrev} vs {away_abbrev}")
        
        matched_df = pd.DataFrame(matched_data)
        
        self.logger.info(f"매칭 성공: {len(matched_df)}경기, 실패: {unmatched_count}경기")
        
        return matched_df
    
    def calculate_betting_roi(self, matched_df: pd.DataFrame) -> pd.DataFrame:
        """베팅 ROI 계산"""
        
        results = []
        
        for model in self.models:
            model_data = matched_df.copy()
            
            if model == 'ensemble':
                prob_col = 'home_win_probability'
            else:
                prob_col = f'{model}_prob'
            
            if prob_col not in model_data.columns:
                continue
            
            # 각 경기별 베팅 결과 계산
            for idx, row in model_data.iterrows():
                home_prob = row[prob_col]
                home_odds = row['home_odds']
                away_odds = row['away_odds']
                actual_home_win = row['actual_home_win']
                
                # 예측: 홈팀이 더 높은 확률
                bet_on_home = home_prob > 0.5
                
                if bet_on_home:
                    # 홈팀에 베팅
                    if actual_home_win == 1:
                        # 승리
                        payout = self._calculate_payout(100, home_odds)
                        profit = payout - 100
                    else:
                        # 패배
                        profit = -100
                    
                    bet_team = 'home'
                    bet_odds = home_odds
                    bet_prob = home_prob
                    
                else:
                    # 원정팀에 베팅
                    if actual_home_win == 0:
                        # 승리
                        payout = self._calculate_payout(100, away_odds)
                        profit = payout - 100
                    else:
                        # 패배
                        profit = -100
                    
                    bet_team = 'away'
                    bet_odds = away_odds
                    bet_prob = 1 - home_prob
                
                # 배당률 기반 내재 확률
                if bet_odds > 0:
                    implied_prob = 100 / (bet_odds + 100)
                else:
                    implied_prob = (-bet_odds) / (-bet_odds + 100)
                
                # 예측 ROI (확률 - 내재확률)
                predicted_roi = (bet_prob - implied_prob) * 100
                
                # 실제 ROI
                actual_roi = profit
                
                results.append({
                    'model': model,
                    'game_id': row['game_id'],
                    'date': row['date'],
                    'home_team': row['home_team_name'],
                    'away_team': row['away_team_name'],
                    'bet_team': bet_team,
                    'bet_odds': bet_odds,
                    'bet_probability': bet_prob,
                    'implied_probability': implied_prob,
                    'predicted_roi_pct': predicted_roi,
                    'actual_profit': profit,
                    'actual_roi_pct': actual_roi,
                    'won': profit > 0,
                    'confidence_level': self._get_confidence_level(bet_prob),
                    'predicted_roi_bucket': self._get_roi_bucket(predicted_roi)
                })
        
        return pd.DataFrame(results)
    
    def _calculate_payout(self, stake: float, american_odds: float) -> float:
        """아메리칸 배당률로 배당금 계산"""
        if american_odds > 0:
            return stake + (stake * american_odds / 100)
        else:
            return stake + (stake * 100 / (-american_odds))
    
    def _get_confidence_level(self, probability: float) -> str:
        """신뢰도 구간 분류"""
        if probability >= 0.8:
            return '80%+'
        elif probability >= 0.7:
            return '70-80%'
        elif probability >= 0.6:
            return '60-70%'
        else:
            return '50-60%'
    
    def _get_roi_bucket(self, predicted_roi: float) -> str:
        """예측 ROI 구간 분류"""
        if predicted_roi >= 20:
            return '20%+'
        elif predicted_roi >= 10:
            return '10-20%'
        elif predicted_roi >= 0:
            return '0-10%'
        else:
            return 'Negative'
    
    def analyze_model_performance(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        """모델별 전체 성과 분석"""
        
        summary = []
        
        for model in self.models:
            model_bets = betting_results[betting_results['model'] == model]
            
            if len(model_bets) == 0:
                continue
            
            total_bets = len(model_bets)
            wins = model_bets['won'].sum()
            losses = total_bets - wins
            win_rate = wins / total_bets * 100
            
            total_profit = model_bets['actual_profit'].sum()
            total_staked = total_bets * 100
            roi = (total_profit / total_staked) * 100
            
            avg_odds = model_bets['bet_odds'].mean()
            avg_probability = model_bets['bet_probability'].mean()
            
            summary.append({
                'Model': model.upper(),
                'Total Bets': total_bets,
                'Wins': wins,
                'Losses': losses,
                'Win Rate (%)': round(win_rate, 2),
                'Total Profit ($)': round(total_profit, 2),
                'ROI (%)': round(roi, 2),
                'Avg Odds': round(avg_odds, 0),
                'Avg Confidence': round(avg_probability * 100, 2)
            })
        
        return pd.DataFrame(summary).sort_values('ROI (%)', ascending=False)
    
    def analyze_by_confidence(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        """신뢰도 구간별 성과 분석"""
        
        confidence_levels = ['50-60%', '60-70%', '70-80%', '80%+']
        summary = []
        
        for model in self.models:
            for conf_level in confidence_levels:
                mask = (betting_results['model'] == model) & \
                       (betting_results['confidence_level'] == conf_level)
                model_bets = betting_results[mask]
                
                if len(model_bets) == 0:
                    continue
                
                total_bets = len(model_bets)
                wins = model_bets['won'].sum()
                win_rate = wins / total_bets * 100
                
                total_profit = model_bets['actual_profit'].sum()
                total_staked = total_bets * 100
                roi = (total_profit / total_staked) * 100
                
                summary.append({
                    'Model': model.upper(),
                    'Confidence': conf_level,
                    'Bets': total_bets,
                    'Wins': wins,
                    'Win Rate (%)': round(win_rate, 2),
                    'ROI (%)': round(roi, 2)
                })
        
        return pd.DataFrame(summary)
    
    def analyze_by_predicted_roi(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        """예측 ROI 구간별 실제 성과 분석"""
        
        roi_buckets = ['Negative', '0-10%', '10-20%', '20%+']
        summary = []
        
        for model in self.models:
            for roi_bucket in roi_buckets:
                mask = (betting_results['model'] == model) & \
                       (betting_results['predicted_roi_bucket'] == roi_bucket)
                model_bets = betting_results[mask]
                
                if len(model_bets) == 0:
                    continue
                
                total_bets = len(model_bets)
                wins = model_bets['won'].sum()
                win_rate = wins / total_bets * 100
                
                total_profit = model_bets['actual_profit'].sum()
                total_staked = total_bets * 100
                actual_roi = (total_profit / total_staked) * 100
                
                avg_predicted_roi = model_bets['predicted_roi_pct'].mean()
                
                summary.append({
                    'Model': model.upper(),
                    'Predicted ROI': roi_bucket,
                    'Bets': total_bets,
                    'Wins': wins,
                    'Win Rate (%)': round(win_rate, 2),
                    'Avg Pred ROI (%)': round(avg_predicted_roi, 2),
                    'Actual ROI (%)': round(actual_roi, 2)
                })
        
        return pd.DataFrame(summary)


def main():
    """Streamlit 대시보드 메인"""
    
    st.set_page_config(
        page_title="NHL Model Performance Dashboard",
        page_icon="🏒",
        layout="wide"
    )
    
    st.title("🏒 NHL Model Performance Analysis Dashboard")
    st.markdown("---")
    
    # 분석기 초기화
    analyzer = ModelPerformanceAnalyzer()
    
    # 데이터 로드
    with st.spinner("데이터 로드 중..."):
        predictions_df = analyzer.load_merged_predictions(exclude_today=True)
        records_df = analyzer.load_historical_records()
        
        if predictions_df.empty or records_df.empty:
            st.error("데이터를 로드할 수 없습니다.")
            return
        
        # 매칭
        matched_df = analyzer.match_predictions_with_results(predictions_df, records_df)
        
        if matched_df.empty:
            st.error("예측과 결과를 매칭할 수 없습니다.")
            return
        
        # ROI 계산
        betting_results = analyzer.calculate_betting_roi(matched_df)
    
    st.success(f"✅ 총 {len(matched_df)}경기 분석 완료!")
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overall Performance",
        "🎯 Confidence Analysis", 
        "💰 Predicted vs Actual ROI",
        "📋 Detailed Results"
    ])
    
    # Tab 1: 전체 성과
    with tab1:
        st.header("Overall Model Performance")
        
        overall_perf = analyzer.analyze_model_performance(betting_results)
        
        # 메트릭 카드
        col1, col2, col3, col4 = st.columns(4)
        
        best_roi_model = overall_perf.iloc[0]
        with col1:
            st.metric("Best ROI Model", best_roi_model['Model'])
        with col2:
            st.metric("Best ROI", f"{best_roi_model['ROI (%)']}%")
        with col3:
            st.metric("Win Rate", f"{best_roi_model['Win Rate (%)']}%")
        with col4:
            st.metric("Total Profit", f"${best_roi_model['Total Profit ($)']}")
        
        st.markdown("---")
        
        # 성과 테이블
        st.subheader("📈 Model Performance Summary")
        
        # 소숫점 자릿수 통일 및 포맷팅
        display_perf = overall_perf.copy()
        
        # 원본 숫자 값 저장 (색상 판단용)
        display_perf['_roi_num'] = display_perf['ROI (%)']
        display_perf['_profit_num'] = display_perf['Total Profit ($)']
        
        # 포맷팅
        display_perf['Win Rate (%)'] = display_perf['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
        display_perf['Total Profit ($)'] = display_perf['Total Profit ($)'].apply(lambda x: f"{x:.2f}")
        display_perf['ROI (%)'] = display_perf['ROI (%)'].apply(lambda x: f"{x:.2f}")
        display_perf['Avg Odds'] = display_perf['Avg Odds'].round(0).astype(int)
        display_perf['Avg Confidence'] = display_perf['Avg Confidence'].apply(lambda x: f"{x:.2f}")
        
        # 색상 스타일링 함수
        def style_performance(row):
            styles = [''] * len(row)
            
            # 인덱스 찾기
            roi_idx = display_perf.columns.get_loc('ROI (%)')
            profit_idx = display_perf.columns.get_loc('Total Profit ($)')
            
            # 원본 숫자 값으로 색상 결정
            roi_val = row['_roi_num']
            profit_val = row['_profit_num']
            
            if roi_val > 0:
                styles[roi_idx] = 'color: green; font-weight: bold'
            else:
                styles[roi_idx] = 'color: red; font-weight: bold'
            
            if profit_val > 0:
                styles[profit_idx] = 'color: green; font-weight: bold'
            else:
                styles[profit_idx] = 'color: red; font-weight: bold'
            
            return styles
        
        # 스타일 적용 및 숨김 컬럼 제거
        styled_perf = display_perf.style.apply(style_performance, axis=1)
        
        # 숨김 컬럼을 제외한 컬럼만 선택해서 표시
        visible_cols = [col for col in display_perf.columns if not col.startswith('_')]
        
        st.dataframe(styled_perf, use_container_width=True, height=400, column_config={
            '_roi_num': None,
            '_profit_num': None
        })
        
        # ROI 비교 차트
        st.subheader("📊 ROI Comparison")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=overall_perf['Model'],
            y=overall_perf['ROI (%)'],
            marker_color=['green' if x > 0 else 'red' for x in overall_perf['ROI (%)']],
            text=overall_perf['ROI (%)'].round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Model ROI Comparison (%)",
            xaxis_title="Model",
            yaxis_title="ROI (%)",
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 승률 비교 차트
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                overall_perf,
                x='Model',
                y='Win Rate (%)',
                title='Win Rate by Model',
                color='Win Rate (%)',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                overall_perf,
                x='Win Rate (%)',
                y='ROI (%)',
                size='Total Bets',
                color='Model',
                title='Win Rate vs ROI',
                hover_data=['Total Bets']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: 신뢰도 구간별 분석
    with tab2:
        st.header("Performance by Confidence Level")
        
        confidence_perf = analyzer.analyze_by_confidence(betting_results)
        
        # 모델 선택 (단일 선택)
        selected_model = st.selectbox(
            "Select Model to Analyze",
            options=[m.upper() for m in analyzer.models],
            key='confidence_model_select'
        )
        
        if selected_model:
            model_conf_data = confidence_perf[confidence_perf['Model'] == selected_model].copy()
            
            if not model_conf_data.empty:
                # 구간 순서 정렬
                confidence_order = ['50-60%', '60-70%', '70-80%', '80%+']
                model_conf_data['Confidence'] = pd.Categorical(
                    model_conf_data['Confidence'], 
                    categories=confidence_order, 
                    ordered=True
                )
                model_conf_data = model_conf_data.sort_values('Confidence')
                
                # 메트릭 카드
                st.subheader(f"📊 {selected_model} Performance by Confidence")
                
                cols = st.columns(len(model_conf_data))
                for idx, (_, row) in enumerate(model_conf_data.iterrows()):
                    with cols[idx]:
                        st.metric(
                            label=row['Confidence'],
                            value=f"{row['ROI (%)']:.2f}%",
                            delta=f"{row['Win Rate (%)']:.1f}% Win Rate"
                        )
                        st.caption(f"{int(row['Bets'])} bets, {int(row['Wins'])} wins")
                
                st.markdown("---")
                
                # 차트
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=model_conf_data['Confidence'],
                        y=model_conf_data['ROI (%)'],
                        marker_color=['green' if x > 0 else 'red' for x in model_conf_data['ROI (%)']],
                        text=model_conf_data['ROI (%)'].round(2),
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title=f"{selected_model} - ROI by Confidence Level",
                        xaxis_title="Confidence Level",
                        yaxis_title="ROI (%)",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=model_conf_data['Confidence'],
                        y=model_conf_data['Win Rate (%)'],
                        marker_color='lightblue',
                        text=model_conf_data['Win Rate (%)'].round(1),
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title=f"{selected_model} - Win Rate by Confidence Level",
                        xaxis_title="Confidence Level",
                        yaxis_title="Win Rate (%)",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 상세 테이블
                st.subheader("📋 Detailed Statistics")
                
                # 컬럼명 정리 및 포맷팅
                display_conf = model_conf_data.copy()
                
                # 원본 숫자 값 저장
                display_conf['_roi_num'] = display_conf['ROI (%)']
                
                display_conf['Bets'] = display_conf['Bets'].astype(int)
                display_conf['Wins'] = display_conf['Wins'].astype(int)
                display_conf['Win Rate (%)'] = display_conf['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
                display_conf['ROI (%)'] = display_conf['ROI (%)'].apply(lambda x: f"{x:.2f}")
                
                # ROI 색상 스타일링
                def style_conf_roi(row):
                    styles = [''] * len(row)
                    roi_idx = display_conf.columns.get_loc('ROI (%)')
                    roi_val = row['_roi_num']
                    
                    if roi_val > 0:
                        styles[roi_idx] = 'color: green; font-weight: bold'
                    else:
                        styles[roi_idx] = 'color: red; font-weight: bold'
                    
                    return styles
                
                # 스타일 적용
                styled_conf = display_conf.style.apply(style_conf_roi, axis=1)
                
                st.dataframe(styled_conf, use_container_width=True, column_config={
                    '_roi_num': None
                })
        
        # 전체 모델 비교 섹션
        st.markdown("---")
        st.subheader("🔄 Compare All Models")
        
        # 히트맵 데이터 준비
        pivot_roi = confidence_perf.pivot(
            index='Model',
            columns='Confidence',
            values='ROI (%)'
        )
        
        # 구간 순서 정렬
        confidence_order = ['50-60%', '60-70%', '70-80%', '80%+']
        pivot_roi = pivot_roi.reindex(columns=confidence_order)
        
        fig = px.imshow(
            pivot_roi,
            labels=dict(x="Confidence Level", y="Model", color="ROI (%)"),
            color_continuous_scale='RdYlGn',
            aspect="auto",
            title="ROI (%) Heatmap - All Models by Confidence Level"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: 예측 ROI vs 실제 ROI
    with tab3:
        st.header("Predicted ROI vs Actual ROI Analysis")
        
        roi_analysis = analyzer.analyze_by_predicted_roi(betting_results)
        
        # 모델 선택
        selected_model_roi = st.selectbox(
            "Select Model",
            options=[m.upper() for m in analyzer.models],
            key='roi_model_select'
        )
        
        if selected_model_roi:
            model_roi_data = roi_analysis[roi_analysis['Model'] == selected_model_roi].copy()
            
            if not model_roi_data.empty:
                # ROI 구간 순서 정렬
                roi_order = ['Negative', '0-10%', '10-20%', '20%+']
                model_roi_data['Predicted ROI'] = pd.Categorical(
                    model_roi_data['Predicted ROI'], 
                    categories=roi_order, 
                    ordered=True
                )
                model_roi_data = model_roi_data.sort_values('Predicted ROI')
                
                # 메트릭 카드
                st.subheader(f"📊 {selected_model_roi} Performance by Predicted ROI Bucket")
                
                cols = st.columns(len(model_roi_data))
                for idx, (_, row) in enumerate(model_roi_data.iterrows()):
                    with cols[idx]:
                        st.metric(
                            label=row['Predicted ROI'],
                            value=f"{row['Actual ROI (%)']:.2f}%",
                            delta=f"Pred: {row['Avg Pred ROI (%)']:.2f}%"
                        )
                        st.caption(f"{int(row['Bets'])} bets, {int(row['Wins'])} wins")
                
                st.markdown("---")
                
                # 예측 vs 실제 ROI 비교 차트
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Predicted ROI (Avg)',
                    x=model_roi_data['Predicted ROI'],
                    y=model_roi_data['Avg Pred ROI (%)'],
                    marker_color='lightblue',
                    text=model_roi_data['Avg Pred ROI (%)'].round(2),
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    name='Actual ROI',
                    x=model_roi_data['Predicted ROI'],
                    y=model_roi_data['Actual ROI (%)'],
                    marker_color=['green' if x > 0 else 'red' for x in model_roi_data['Actual ROI (%)']],
                    text=model_roi_data['Actual ROI (%)'].round(2),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f"{selected_model_roi} - Predicted vs Actual ROI by Bucket",
                    xaxis_title="Predicted ROI Bucket",
                    yaxis_title="ROI (%)",
                    barmode='group',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 테이블
                st.subheader("📋 Detailed ROI Bucket Statistics")
                
                # 포맷팅
                display_roi = model_roi_data.copy()
                
                # 원본 숫자 값 저장
                display_roi['_actual_roi_num'] = display_roi['Actual ROI (%)']
                display_roi['_pred_roi_num'] = display_roi['Avg Pred ROI (%)']
                
                display_roi['Bets'] = display_roi['Bets'].astype(int)
                display_roi['Wins'] = display_roi['Wins'].astype(int)
                display_roi['Win Rate (%)'] = display_roi['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
                display_roi['Avg Pred ROI (%)'] = display_roi['Avg Pred ROI (%)'].apply(lambda x: f"{x:.2f}")
                display_roi['Actual ROI (%)'] = display_roi['Actual ROI (%)'].apply(lambda x: f"{x:.2f}")
                
                # ROI 색상 스타일링
                def style_roi_table(row):
                    styles = [''] * len(row)
                    actual_roi_idx = display_roi.columns.get_loc('Actual ROI (%)')
                    pred_roi_idx = display_roi.columns.get_loc('Avg Pred ROI (%)')
                    
                    actual_roi_val = row['_actual_roi_num']
                    pred_roi_val = row['_pred_roi_num']
                    
                    if actual_roi_val > 0:
                        styles[actual_roi_idx] = 'color: green; font-weight: bold'
                    else:
                        styles[actual_roi_idx] = 'color: red; font-weight: bold'
                    
                    if pred_roi_val > 0:
                        styles[pred_roi_idx] = 'color: blue; font-weight: bold'
                    elif pred_roi_val < 0:
                        styles[pred_roi_idx] = 'color: orange; font-weight: bold'
                    
                    return styles
                
                # 스타일 적용
                styled_roi = display_roi.style.apply(style_roi_table, axis=1)
                
                st.dataframe(styled_roi, use_container_width=True, column_config={
                    '_actual_roi_num': None,
                    '_pred_roi_num': None
                })
        
        # 전체 모델 비교
        st.markdown("---")
        st.subheader("🔄 Compare All Models")
        
        # 히트맵 - ROI 구간 순서 정렬
        roi_order = ['Negative', '0-10%', '10-20%', '20%+']
        pivot_actual_roi = roi_analysis.pivot(
            index='Model',
            columns='Predicted ROI',
            values='Actual ROI (%)'
        )
        pivot_actual_roi = pivot_actual_roi.reindex(columns=roi_order)
        
        fig = px.imshow(
            pivot_actual_roi,
            labels=dict(x="Predicted ROI Bucket", y="Model", color="Actual ROI (%)"),
            color_continuous_scale='RdYlGn',
            aspect="auto",
            title="Actual ROI Heatmap by Predicted ROI Bucket - All Models"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: 상세 결과
    with tab4:
        st.header("Detailed Betting Results")
        
        # 필터
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_model = st.selectbox(
                "Filter by Model",
                options=['All'] + [m.upper() for m in analyzer.models],
                key='detail_model_filter'
            )
        
        with col2:
            filter_result = st.selectbox(
                "Filter by Result",
                options=['All', 'Won', 'Lost']
            )
        
        with col3:
            filter_confidence = st.selectbox(
                "Filter by Confidence",
                options=['All', '50-60%', '60-70%', '70-80%', '80%+']
            )
        
        # 필터 적용
        filtered_results = betting_results.copy()
        
        if filter_model != 'All':
            filtered_results = filtered_results[filtered_results['model'] == filter_model.lower()]
        
        if filter_result == 'Won':
            filtered_results = filtered_results[filtered_results['won'] == True]
        elif filter_result == 'Lost':
            filtered_results = filtered_results[filtered_results['won'] == False]
        
        if filter_confidence != 'All':
            filtered_results = filtered_results[filtered_results['confidence_level'] == filter_confidence]
        
        # 결과 표시
        st.subheader(f"📋 Showing {len(filtered_results)} bets")
        
        # 컬럼 선택 및 포맷팅
        display_cols = [
            'date', 'model', 'home_team', 'away_team', 'bet_team',
            'bet_probability', 'bet_odds', 'predicted_roi_pct',
            'actual_profit', 'actual_roi_pct', 'won', 'confidence_level'
        ]
        
        display_df = filtered_results[display_cols].copy()
        
        # 원본 숫자 값 저장 (색상 판단용)
        display_df['_won'] = display_df['won']
        display_df['_profit_num'] = display_df['actual_profit']
        display_df['_roi_num'] = display_df['actual_roi_pct']
        
        # 포맷팅
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['model'] = display_df['model'].str.upper()
        display_df['bet_probability'] = (display_df['bet_probability'] * 100).apply(lambda x: f"{x:.2f}")
        display_df['bet_odds'] = display_df['bet_odds'].round(0).astype(int)
        display_df['predicted_roi_pct'] = display_df['predicted_roi_pct'].apply(lambda x: f"{x:.2f}")
        display_df['actual_profit'] = display_df['actual_profit'].apply(lambda x: f"{x:.2f}")
        display_df['actual_roi_pct'] = display_df['actual_roi_pct'].apply(lambda x: f"{x:.2f}")
        
        # 컬럼명 변경 (가독성)
        display_df.columns = [
            'Date', 'Model', 'Home Team', 'Away Team', 'Bet On',
            'Confidence (%)', 'Odds', 'Pred ROI (%)',
            'Profit ($)', 'Actual ROI (%)', 'Won', 'Confidence Level',
            '_won', '_profit_num', '_roi_num'
        ]
        
        # 스타일링
        def style_results(row):
            won = row['_won']
            profit_val = row['_profit_num']
            roi_val = row['_roi_num']
            
            # 배경색
            if won:
                bg_color = 'background-color: #d4edda'
            else:
                bg_color = 'background-color: #f8d7da'
            
            styles = [bg_color] * len(row)
            
            # 인덱스 찾기
            profit_idx = display_df.columns.get_loc('Profit ($)')
            roi_idx = display_df.columns.get_loc('Actual ROI (%)')
            
            if profit_val > 0:
                styles[profit_idx] = f'{bg_color}; color: green; font-weight: bold'
            else:
                styles[profit_idx] = f'{bg_color}; color: red; font-weight: bold'
            
            if roi_val > 0:
                styles[roi_idx] = f'{bg_color}; color: green; font-weight: bold'
            else:
                styles[roi_idx] = f'{bg_color}; color: red; font-weight: bold'
            
            return styles
        
        # 스타일 적용
        styled_results = display_df.style.apply(style_results, axis=1)
        
        st.dataframe(styled_results, use_container_width=True, height=600, column_config={
            '_won': None,
            '_profit_num': None,
            '_roi_num': None
        })
        
        # 다운로드 버튼
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"model_performance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()

