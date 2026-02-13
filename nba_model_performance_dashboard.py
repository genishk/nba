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


class NBAModelPerformanceAnalyzer:
    """NBA 모델 성과 분석 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.analysis_dir = self.project_root / 'src' / 'analysis'
        self.data_dir = self.project_root / 'src' / 'data'
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('NBAModelPerformanceAnalyzer')
        
        # 팀명 매핑 (전체 이름 → 약어)
        self.team_name_to_abbrev = {
            'Hawks': 'ATL',
            'Celtics': 'BOS',
            'Nets': 'BKN',
            'Hornets': 'CHA',
            'Bulls': 'CHI',
            'Cavaliers': 'CLE',
            'Mavericks': 'DAL',
            'Nuggets': 'DEN',
            'Pistons': 'DET',
            'Warriors': 'GSW',
            'Rockets': 'HOU',
            'Pacers': 'IND',
            'Clippers': 'LAC',
            'Lakers': 'LAL',
            'Grizzlies': 'MEM',
            'Heat': 'MIA',
            'Bucks': 'MIL',
            'Timberwolves': 'MIN',
            'Pelicans': 'NOP',
            'Knicks': 'NYK',
            'Thunder': 'OKC',
            'Magic': 'ORL',
            '76ers': 'PHI',
            'Suns': 'PHX',
            'Trail Blazers': 'POR',
            'Kings': 'SAC',
            'Spurs': 'SAS',
            'Raptors': 'TOR',
            'Jazz': 'UTA',
            'Wizards': 'WAS'
        }
    
    def load_merged_predictions(self, exclude_today: bool = True, model_tag: str = 'active') -> pd.DataFrame:
        """병합된 예측 파일들 로드 (오늘 제외 가능)
        
        Args:
            exclude_today: 오늘 파일 제외 여부
            model_tag: 모델 태그 ('active', 'shadow')
                      - 'active': _active.json + 태그 없는 기존 파일 (Active로 취급)
                      - 'shadow': _shadow.json만
        """
        merged_files = []
        
        if model_tag == 'active':
            # Active: _active.json 파일들
            active_files = list(self.analysis_dir.glob('merged_predictions_odds_*_active.json'))
            merged_files.extend(active_files)
            
            # + 태그 없는 기존 파일들 (Active로 취급)
            all_files = list(self.analysis_dir.glob('merged_predictions_odds_*.json'))
            for f in all_files:
                # _active, _shadow 태그가 없는 파일만 추가
                if not f.stem.endswith('_active') and not f.stem.endswith('_shadow'):
                    merged_files.append(f)
            
            merged_files = sorted(set(merged_files))
            
        elif model_tag == 'shadow':
            # Shadow: _shadow.json만
            merged_files = sorted(self.analysis_dir.glob('merged_predictions_odds_*_shadow.json'))
        else:
            # 기타: 모든 파일
            merged_files = sorted(self.analysis_dir.glob('merged_predictions_odds_*.json'))
        
        if not merged_files:
            self.logger.error(f"병합 파일을 찾을 수 없습니다. (tag: {model_tag})")
            return pd.DataFrame()
        
        all_predictions = []
        today = datetime.now().strftime('%Y%m%d')
        
        for file in merged_files:
            # 파일명에서 날짜 추출
            # 새 형식: merged_predictions_odds_20251118_112710_active.json
            # 기존 형식: merged_predictions_odds_20251118_112710.json
            try:
                parts = file.stem.split('_')
                # 태그가 있으면 4번째가 날짜, 없으면 3번째가 날짜
                if len(parts) >= 5 and parts[-1] in ['active', 'shadow']:
                    file_date = parts[-3]
                else:
                    file_date = parts[-2]
            except:
                self.logger.warning(f"날짜 추출 실패: {file.name}")
                continue
            
            # 오늘 파일 제외 옵션
            if exclude_today and file_date == today:
                self.logger.info(f"오늘 파일 제외: {file.name}")
                continue
            
            self.logger.info(f"로드 중: {file.name}")
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_predictions.extend(data)
            except Exception as e:
                self.logger.error(f"파일 로드 실패 {file.name}: {e}")
                continue
        
        if not all_predictions:
            self.logger.error("예측 데이터가 없습니다.")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_predictions)
        
        # 날짜 변환
        df['date'] = pd.to_datetime(df['date'])
        
        self.logger.info(f"총 {len(df)}개 예측 로드 완료")
        return df
    
    def load_game_results(self) -> pd.DataFrame:
        """실제 경기 결과 로드 (processed_*.json)"""
        # 가장 최신 processed 파일 찾기 (with_odds, spread, prediction 제외)
        all_processed = list(self.data_dir.glob('processed_*.json'))
        processed_files = sorted([
            f for f in all_processed 
            if '_with_odds' not in f.name 
            and '_spread' not in f.name 
            and '_prediction' not in f.name
        ])
        
        if not processed_files:
            self.logger.error("경기 결과 파일을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        results_file = processed_files[-1]
        self.logger.info(f"경기 결과 로드 중: {results_file.name}")
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception as e:
            self.logger.error(f"파일 로드 실패: {e}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        if df.empty:
            return df
        
        # 완료된 경기만 필터링
        df = df[df['status'] == 'STATUS_FINAL'].copy()
        
        # 날짜 변환 (UTC → date만)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # 팀명 → 약어 변환
        df['home_team_abbrev'] = df['home_team_name'].map(self.team_name_to_abbrev)
        df['away_team_abbrev'] = df['away_team_name'].map(self.team_name_to_abbrev)
        
        # 매핑 실패 확인
        missing_home = df[df['home_team_abbrev'].isna()]['home_team_name'].unique()
        missing_away = df[df['away_team_abbrev'].isna()]['away_team_name'].unique()
        
        if len(missing_home) > 0:
            self.logger.warning(f"매핑 실패 (홈팀): {missing_home}")
        if len(missing_away) > 0:
            self.logger.warning(f"매핑 실패 (원정팀): {missing_away}")
        
        self.logger.info(f"총 {len(df)}개 완료된 경기 로드")
        return df
    
    def match_predictions_with_results(self, predictions_df: pd.DataFrame, 
                                      results_df: pd.DataFrame) -> pd.DataFrame:
        """예측과 실제 결과 매칭"""
        
        if predictions_df.empty or results_df.empty:
            self.logger.error("데이터가 비어있습니다.")
            return pd.DataFrame()
        
        matched_data = []
        unmatched_count = 0
        
        for idx, pred in predictions_df.iterrows():
            # 날짜를 date 객체로 변환 (시간 제거)
            pred_date = pred['date'].date()
            home_abbrev = pred['home_team_abbrev']
            away_abbrev = pred['away_team_abbrev']
            
            # 날짜 + 팀으로 매칭
            mask = (
                (results_df['date'] == pred_date) &
                (results_df['home_team_abbrev'] == home_abbrev) &
                (results_df['away_team_abbrev'] == away_abbrev)
            )
            
            matched_results = results_df[mask]
            
            if len(matched_results) > 0:
                result = matched_results.iloc[0]
                
                # 예측 데이터에 실제 결과 추가
                matched_game = pred.to_dict()
                matched_game['actual_home_win'] = 1 if result['home_team_score'] > result['away_team_score'] else 0
                matched_game['actual_home_score'] = int(result['home_team_score'])
                matched_game['actual_away_score'] = int(result['away_team_score'])
                
                matched_data.append(matched_game)
            else:
                unmatched_count += 1
                self.logger.debug(f"매칭 실패: {pred_date} - {home_abbrev} vs {away_abbrev}")
        
        matched_df = pd.DataFrame(matched_data)
        
        self.logger.info(f"매칭 성공: {len(matched_df)}경기, 실패: {unmatched_count}경기")
        
        return matched_df
    
    def detect_models(self, matched_df: pd.DataFrame) -> List[str]:
        """데이터프레임에서 모델 자동 감지"""
        # model1_home_win_prob, model2_home_win_prob 형식 찾기
        model_cols = [col for col in matched_df.columns 
                     if col.startswith('model') and col.endswith('_home_win_prob')]
        
        # model1, model2, ... 추출
        models = sorted([col.replace('_home_win_prob', '') for col in model_cols])
        
        # ensemble 추가 (home_win_probability 컬럼이 있으면)
        if 'home_win_probability' in matched_df.columns:
            models.append('ensemble')
        
        self.logger.info(f"감지된 모델: {models}")
        return models
    
    def calculate_betting_roi(self, matched_df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
        """베팅 ROI 계산"""
        
        results = []
        
        for model in models:
            model_data = matched_df.copy()
            
            # 확률 컬럼 선택
            if model == 'ensemble':
                prob_col = 'home_win_probability'
            else:
                prob_col = f'{model}_home_win_prob'
            
            if prob_col not in model_data.columns:
                self.logger.warning(f"{model}의 확률 컬럼이 없습니다: {prob_col}")
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
                    'predicted_roi_bucket': self._get_roi_bucket(predicted_roi),
                    'odds_bucket': self._get_odds_bucket(bet_odds)
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
    
    def _get_odds_bucket(self, american_odds: float) -> str:
        """배당률 구간 분류 (아메리칸 오즈)"""
        # 언더독 (플러스 배당)
        if american_odds >= 300:
            return '+300 이상 (Heavy Underdog)'
        elif american_odds >= 200:
            return '+200 ~ +299 (Underdog)'
        elif american_odds >= 150:
            return '+150 ~ +199'
        elif american_odds >= 100:
            return '+100 ~ +149'
        # 픽엠 근처
        elif american_odds >= -110:
            return '-110 ~ +99 (Pick\'em)'
        # 페이보릿 (마이너스 배당)
        elif american_odds >= -150:
            return '-150 ~ -111'
        elif american_odds >= -200:
            return '-200 ~ -151 (Favorite)'
        elif american_odds >= -300:
            return '-300 ~ -201'
        else:
            return '-300 이하 (Heavy Favorite)'
    
    def analyze_model_performance(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        """모델별 전체 성과 분석"""
        
        summary = []
        
        models = betting_results['model'].unique()
        
        for model in models:
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
        
        models = betting_results['model'].unique()
        
        for model in models:
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
        
        models = betting_results['model'].unique()
        
        for model in models:
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
    
    def analyze_by_odds(self, betting_results: pd.DataFrame) -> pd.DataFrame:
        """배당률 구간별 실제 성과 분석"""
        
        # 배당률 구간 순서 정의
        odds_buckets = [
            '-300 이하 (Heavy Favorite)',
            '-300 ~ -201',
            '-200 ~ -151 (Favorite)',
            '-150 ~ -111',
            '-110 ~ +99 (Pick\'em)',
            '+100 ~ +149',
            '+150 ~ +199',
            '+200 ~ +299 (Underdog)',
            '+300 이상 (Heavy Underdog)'
        ]
        
        summary = []
        
        models = betting_results['model'].unique()
        
        for model in models:
            for odds_bucket in odds_buckets:
                mask = (betting_results['model'] == model) & \
                       (betting_results['odds_bucket'] == odds_bucket)
                model_bets = betting_results[mask]
                
                if len(model_bets) == 0:
                    continue
                
                total_bets = len(model_bets)
                wins = model_bets['won'].sum()
                win_rate = wins / total_bets * 100
                
                total_profit = model_bets['actual_profit'].sum()
                total_staked = total_bets * 100
                actual_roi = (total_profit / total_staked) * 100
                
                avg_odds = model_bets['bet_odds'].mean()
                avg_confidence = model_bets['bet_probability'].mean() * 100
                
                summary.append({
                    'Model': model.upper(),
                    'Odds Range': odds_bucket,
                    'Bets': total_bets,
                    'Wins': wins,
                    'Win Rate (%)': round(win_rate, 2),
                    'Avg Odds': round(avg_odds, 0),
                    'Avg Confidence (%)': round(avg_confidence, 2),
                    'Actual ROI (%)': round(actual_roi, 2)
                })
        
        return pd.DataFrame(summary)


def main():
    """Streamlit 대시보드 메인"""
    
    st.set_page_config(
        page_title="NBA Model Performance Dashboard",
        page_icon="🏀",
        layout="wide"
    )
    
    st.title("🏀 NBA Model Performance Analysis Dashboard")
    st.markdown("---")
    
    # 분석기 초기화
    analyzer = NBAModelPerformanceAnalyzer()
    
    # 데이터 로드
    with st.spinner("데이터 로드 중..."):
        # 1. 예측 데이터 로드 (오늘 제외)
        predictions_df = analyzer.load_merged_predictions(exclude_today=True)
        
        if predictions_df.empty:
            st.error("예측 데이터를 로드할 수 없습니다.")
            return
        
        # 2. 실제 경기 결과 로드
        results_df = analyzer.load_game_results()
        
        if results_df.empty:
            st.error("경기 결과 데이터를 로드할 수 없습니다.")
            return
        
        # 3. 매칭
        matched_df = analyzer.match_predictions_with_results(predictions_df, results_df)
        
        if matched_df.empty:
            st.error("예측과 결과를 매칭할 수 없습니다.")
            return
        
        # 4. 모델 자동 감지
        models = analyzer.detect_models(matched_df)
        
        if not models:
            st.error("모델을 감지할 수 없습니다.")
            return
        
        st.info(f"✅ 감지된 모델: {', '.join([m.upper() for m in models])}")
        
        # 5. ROI 계산
        betting_results = analyzer.calculate_betting_roi(matched_df, models)
    
    st.success(f"✅ 총 {len(matched_df)}경기 분석 완료!")
    
    # 날짜 필터 추가
    st.markdown("---")
    st.subheader("📅 Date Range Filter")
    
    # 날짜 범위 확인
    min_date = betting_results['date'].min().date()
    max_date = betting_results['date'].max().date()
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key='start_date'
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key='end_date'
        )
    
    # 날짜 필터 적용
    date_mask = (betting_results['date'].dt.date >= start_date) & \
                (betting_results['date'].dt.date <= end_date)
    betting_results_filtered = betting_results[date_mask].copy()
    
    # 필터링된 결과 표시
    total_games = len(betting_results_filtered)
    date_range_display = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
    
    st.info(f"📊 **Filtered Period:** {date_range_display} | **Total Games:** {total_games}")
    
    if total_games == 0:
        st.warning("⚠️ 선택한 날짜 범위에 데이터가 없습니다. 날짜 범위를 조정해주세요.")
        return
    
    st.markdown("---")
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overall Performance",
        "🎯 Confidence Analysis", 
        "💰 Predicted vs Actual ROI",
        "🎲 Odds Range (Active)",
        "🌙 Odds Range (Shadow)",
        "📋 Detailed Results",
        "🔄 Active vs Shadow",
        "🎰 With Odds Models"
    ])
    
    # Tab 1: 전체 성과
    with tab1:
        st.header("Overall Model Performance")
        
        overall_perf = analyzer.analyze_model_performance(betting_results_filtered)
        
        if overall_perf.empty:
            st.warning("성과 데이터가 없습니다.")
        else:
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
        
        confidence_perf = analyzer.analyze_by_confidence(betting_results_filtered)
        
        if confidence_perf.empty:
            st.warning("신뢰도 분석 데이터가 없습니다.")
        else:
            # 모델 선택 (단일 선택)
            selected_model = st.selectbox(
                "Select Model to Analyze",
                options=[m.upper() for m in models],
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
                else:
                    st.info(f"{selected_model}에 대한 신뢰도 데이터가 없습니다.")
        
        # 전체 모델 비교 섹션
        if not confidence_perf.empty:
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
        
        roi_analysis = analyzer.analyze_by_predicted_roi(betting_results_filtered)
        
        if roi_analysis.empty:
            st.warning("ROI 분석 데이터가 없습니다.")
        else:
            # 모델 선택
            selected_model_roi = st.selectbox(
                "Select Model",
                options=[m.upper() for m in models],
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
                else:
                    st.info(f"{selected_model_roi}에 대한 ROI 데이터가 없습니다.")
        
        # 전체 모델 비교
        if not roi_analysis.empty:
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
    
    # Tab 4: 배당률 구간별 분석 (Active)
    with tab4:
        st.header("🏀 Performance by Odds Range (Active)")
        
        odds_analysis = analyzer.analyze_by_odds(betting_results_filtered)
        
        if odds_analysis.empty:
            st.warning("배당률 분석 데이터가 없습니다.")
        else:
            # 모델 선택
            selected_model_odds = st.selectbox(
                "Select Model",
                options=[m.upper() for m in models],
                key='odds_model_select'
            )
            
            if selected_model_odds:
                model_odds_data = odds_analysis[odds_analysis['Model'] == selected_model_odds].copy()
                
                if not model_odds_data.empty:
                    # 배당률 구간 순서 정의 (페이보릿 → 언더독)
                    odds_order = [
                        '-300 이하 (Heavy Favorite)',
                        '-300 ~ -201',
                        '-200 ~ -151 (Favorite)',
                        '-150 ~ -111',
                        '-110 ~ +99 (Pick\'em)',
                        '+100 ~ +149',
                        '+150 ~ +199',
                        '+200 ~ +299 (Underdog)',
                        '+300 이상 (Heavy Underdog)'
                    ]
                    
                    model_odds_data['Odds Range'] = pd.Categorical(
                        model_odds_data['Odds Range'], 
                        categories=odds_order, 
                        ordered=True
                    )
                    model_odds_data = model_odds_data.sort_values('Odds Range')
                    
                    # 요약 메트릭
                    st.subheader(f"📊 {selected_model_odds} Performance by Odds Range")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        best_roi_odds = model_odds_data.loc[model_odds_data['Actual ROI (%)'].idxmax()]
                        st.metric(
                            "Best ROI Range",
                            best_roi_odds['Odds Range'],
                            f"{best_roi_odds['Actual ROI (%)']:.2f}%"
                        )
                    
                    with col2:
                        total_bets = model_odds_data['Bets'].sum()
                        total_wins = model_odds_data['Wins'].sum()
                        overall_win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
                        st.metric("Overall Win Rate", f"{overall_win_rate:.2f}%")
                    
                    with col3:
                        # 가장 많이 베팅한 구간
                        most_bets_odds = model_odds_data.loc[model_odds_data['Bets'].idxmax()]
                        st.metric("Most Bets Range", most_bets_odds['Odds Range'], f"{int(most_bets_odds['Bets'])} bets")
                    
                    with col4:
                        # 전체 ROI
                        weighted_roi = (model_odds_data['Actual ROI (%)'] * model_odds_data['Bets']).sum() / total_bets if total_bets > 0 else 0
                        st.metric("Weighted Avg ROI", f"{weighted_roi:.2f}%")
                    
                    st.markdown("---")
                    
                    # ROI by Odds Range 차트
                    st.subheader("📈 ROI by Odds Range")
                    
                    fig = go.Figure()
                    
                    # 간단한 레이블 (차트용)
                    simple_labels = []
                    for odds_range in model_odds_data['Odds Range']:
                        if 'Heavy Favorite' in odds_range:
                            simple_labels.append('Heavy Fav')
                        elif 'Heavy Underdog' in odds_range:
                            simple_labels.append('Heavy Dog')
                        elif 'Favorite' in odds_range:
                            simple_labels.append('Favorite')
                        elif 'Underdog' in odds_range:
                            simple_labels.append('Underdog')
                        elif 'Pick\'em' in odds_range:
                            simple_labels.append('Pick\'em')
                        else:
                            # -300 ~ -201 형식
                            simple_labels.append(odds_range)
                    
                    fig.add_trace(go.Bar(
                        x=simple_labels,
                        y=model_odds_data['Actual ROI (%)'],
                        marker_color=['green' if x > 0 else 'red' for x in model_odds_data['Actual ROI (%)']],
                        text=model_odds_data['Actual ROI (%)'].round(2),
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>ROI: %{y:.2f}%<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_model_odds} - ROI by Odds Range",
                        xaxis_title="Odds Range (Favorite ← → Underdog)",
                        yaxis_title="ROI (%)",
                        showlegend=False,
                        height=500,
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Win Rate & Bet Count 차트
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=simple_labels,
                            y=model_odds_data['Win Rate (%)'],
                            marker_color='lightblue',
                            text=model_odds_data['Win Rate (%)'].round(1),
                            textposition='outside'
                        ))
                        fig.update_layout(
                            title=f"{selected_model_odds} - Win Rate by Odds Range",
                            xaxis_title="Odds Range",
                            yaxis_title="Win Rate (%)",
                            showlegend=False,
                            height=400,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=simple_labels,
                            y=model_odds_data['Bets'],
                            marker_color='lightcoral',
                            text=model_odds_data['Bets'],
                            textposition='outside'
                        ))
                        fig.update_layout(
                            title=f"{selected_model_odds} - Bet Distribution by Odds Range",
                            xaxis_title="Odds Range",
                            yaxis_title="Number of Bets",
                            showlegend=False,
                            height=400,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 상세 테이블
                    st.subheader("📋 Detailed Odds Range Statistics")
                    
                    # 포맷팅
                    display_odds = model_odds_data.copy()
                    
                    # 원본 숫자 값 저장
                    display_odds['_roi_num'] = display_odds['Actual ROI (%)']
                    
                    display_odds['Bets'] = display_odds['Bets'].astype(int)
                    display_odds['Wins'] = display_odds['Wins'].astype(int)
                    display_odds['Win Rate (%)'] = display_odds['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
                    display_odds['Avg Odds'] = display_odds['Avg Odds'].astype(int)
                    display_odds['Avg Confidence (%)'] = display_odds['Avg Confidence (%)'].apply(lambda x: f"{x:.2f}")
                    display_odds['Actual ROI (%)'] = display_odds['Actual ROI (%)'].apply(lambda x: f"{x:.2f}")
                    
                    # ROI 색상 스타일링
                    def style_odds_table(row):
                        styles = [''] * len(row)
                        roi_idx = display_odds.columns.get_loc('Actual ROI (%)')
                        roi_val = row['_roi_num']
                        
                        if roi_val > 0:
                            styles[roi_idx] = 'color: green; font-weight: bold'
                        else:
                            styles[roi_idx] = 'color: red; font-weight: bold'
                        
                        return styles
                    
                    # 스타일 적용
                    styled_odds = display_odds.style.apply(style_odds_table, axis=1)
                    
                    st.dataframe(styled_odds, use_container_width=True, column_config={
                        '_roi_num': None
                    })
                else:
                    st.info(f"{selected_model_odds}에 대한 배당률 데이터가 없습니다.")
        
        # 전체 모델 비교
        if not odds_analysis.empty:
            st.markdown("---")
            st.subheader("🔄 Compare All Models")
            
            # 배당률 구간 순서
            odds_order = [
                '-300 이하 (Heavy Favorite)',
                '-300 ~ -201',
                '-200 ~ -151 (Favorite)',
                '-150 ~ -111',
                '-110 ~ +99 (Pick\'em)',
                '+100 ~ +149',
                '+150 ~ +199',
                '+200 ~ +299 (Underdog)',
                '+300 이상 (Heavy Underdog)'
            ]
            
            # 히트맵 데이터 준비
            pivot_odds_roi = odds_analysis.pivot(
                index='Model',
                columns='Odds Range',
                values='Actual ROI (%)'
            )
            
            # 구간 순서 정렬
            pivot_odds_roi = pivot_odds_roi.reindex(columns=odds_order)
            
            # 간단한 컬럼명으로 변경 (히트맵 가독성)
            simple_col_names = []
            for col in pivot_odds_roi.columns:
                if pd.isna(col):
                    simple_col_names.append(col)
                elif 'Heavy Favorite' in col:
                    simple_col_names.append('Heavy Fav')
                elif 'Heavy Underdog' in col:
                    simple_col_names.append('Heavy Dog')
                elif 'Favorite' in col:
                    simple_col_names.append('Favorite')
                elif 'Underdog' in col:
                    simple_col_names.append('Underdog')
                elif 'Pick\'em' in col:
                    simple_col_names.append('Pick\'em')
                else:
                    simple_col_names.append(col)
            
            pivot_odds_roi.columns = simple_col_names
            
            fig = px.imshow(
                pivot_odds_roi,
                labels=dict(x="Odds Range", y="Model", color="ROI (%)"),
                color_continuous_scale='RdYlGn',
                aspect="auto",
                title="ROI (%) Heatmap - All Models by Odds Range"
            )
            
            fig.update_xaxes(tickangle=-45)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: 배당률 구간별 분석 (Shadow)
    with tab5:
        st.header("🌙 Performance by Odds Range (Shadow)")
        
        # Shadow 데이터 로드
        shadow_predictions_df = analyzer.load_merged_predictions(exclude_today=True, model_tag='shadow')
        
        if shadow_predictions_df.empty:
            st.warning("⚠️ Shadow 병합 데이터가 없습니다. 데이터가 쌓이면 표시됩니다.")
            st.info("Shadow 데이터는 `main_new.py` 실행 시 자동 생성되며, 경기 결과가 있어야 분석 가능합니다.")
        else:
            # Shadow 매칭 및 분석
            shadow_matched = analyzer.match_predictions_with_results(shadow_predictions_df, results_df)
            
            if shadow_matched.empty:
                st.warning("⚠️ Shadow 예측과 매칭되는 경기 결과가 없습니다.")
            else:
                shadow_models = analyzer.detect_models(shadow_matched)
                shadow_betting = analyzer.calculate_betting_roi(shadow_matched, shadow_models)
                shadow_odds_analysis = analyzer.analyze_by_odds(shadow_betting)
                
                if shadow_odds_analysis.empty:
                    st.warning("Shadow 배당률 분석 데이터가 없습니다.")
                else:
                    # 모델 선택
                    selected_shadow_model = st.selectbox(
                        "Select Model",
                        options=[m.upper() for m in shadow_models],
                        key='shadow_odds_model_select'
                    )
                    
                    if selected_shadow_model:
                        shadow_model_data = shadow_odds_analysis[shadow_odds_analysis['Model'] == selected_shadow_model].copy()
                        
                        if not shadow_model_data.empty:
                            # 배당률 구간 순서 정의
                            odds_order = [
                                '-300 이하 (Heavy Favorite)',
                                '-300 ~ -201',
                                '-200 ~ -151 (Favorite)',
                                '-150 ~ -111',
                                '-110 ~ +99 (Pick\'em)',
                                '+100 ~ +149',
                                '+150 ~ +199',
                                '+200 ~ +299 (Underdog)',
                                '+300 이상 (Heavy Underdog)'
                            ]
                            
                            shadow_model_data['Odds Range'] = pd.Categorical(
                                shadow_model_data['Odds Range'], 
                                categories=odds_order, 
                                ordered=True
                            )
                            shadow_model_data = shadow_model_data.sort_values('Odds Range')
                            
                            # 요약 메트릭
                            st.subheader(f"📊 {selected_shadow_model} Performance by Odds Range (Shadow)")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                best_roi = shadow_model_data.loc[shadow_model_data['Actual ROI (%)'].idxmax()]
                                st.metric("Best ROI Range", best_roi['Odds Range'], f"{best_roi['Actual ROI (%)']:.2f}%")
                            
                            with col2:
                                total_bets = shadow_model_data['Bets'].sum()
                                total_wins = shadow_model_data['Wins'].sum()
                                win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
                                st.metric("Overall Win Rate", f"{win_rate:.2f}%")
                            
                            with col3:
                                most_bets = shadow_model_data.loc[shadow_model_data['Bets'].idxmax()]
                                st.metric("Most Bets Range", most_bets['Odds Range'], f"{int(most_bets['Bets'])} bets")
                            
                            with col4:
                                weighted_roi = (shadow_model_data['Actual ROI (%)'] * shadow_model_data['Bets']).sum() / total_bets if total_bets > 0 else 0
                                st.metric("Weighted Avg ROI", f"{weighted_roi:.2f}%")
                            
                            st.markdown("---")
                            
                            # ROI by Odds Range 차트
                            st.subheader("📈 ROI by Odds Range (Shadow)")
                            
                            fig = go.Figure()
                            
                            simple_labels = []
                            for odds_range in shadow_model_data['Odds Range']:
                                if 'Heavy Favorite' in str(odds_range):
                                    simple_labels.append('Heavy Fav')
                                elif 'Heavy Underdog' in str(odds_range):
                                    simple_labels.append('Heavy Dog')
                                elif 'Favorite' in str(odds_range):
                                    simple_labels.append('Favorite')
                                elif 'Underdog' in str(odds_range):
                                    simple_labels.append('Underdog')
                                elif 'Pick\'em' in str(odds_range):
                                    simple_labels.append('Pick\'em')
                                else:
                                    simple_labels.append(str(odds_range))
                            
                            fig.add_trace(go.Bar(
                                x=simple_labels,
                                y=shadow_model_data['Actual ROI (%)'],
                                marker_color=['green' if x > 0 else 'red' for x in shadow_model_data['Actual ROI (%)']],
                                text=shadow_model_data['Actual ROI (%)'].round(2),
                                textposition='outside'
                            ))
                            
                            fig.update_layout(
                                title=f"{selected_shadow_model} (Shadow) - ROI by Odds Range",
                                xaxis_title="Odds Range (Favorite ← → Underdog)",
                                yaxis_title="ROI (%)",
                                showlegend=False,
                                height=500,
                                xaxis_tickangle=-45
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Win Rate & Bet Count 차트
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=simple_labels,
                                    y=shadow_model_data['Win Rate (%)'],
                                    marker_color='darkorange',
                                    text=shadow_model_data['Win Rate (%)'].round(1),
                                    textposition='outside'
                                ))
                                fig.update_layout(
                                    title=f"{selected_shadow_model} (Shadow) - Win Rate",
                                    xaxis_title="Odds Range",
                                    yaxis_title="Win Rate (%)",
                                    showlegend=False,
                                    height=400,
                                    xaxis_tickangle=-45
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=simple_labels,
                                    y=shadow_model_data['Bets'],
                                    marker_color='mediumpurple',
                                    text=shadow_model_data['Bets'],
                                    textposition='outside'
                                ))
                                fig.update_layout(
                                    title=f"{selected_shadow_model} (Shadow) - Bet Distribution",
                                    xaxis_title="Odds Range",
                                    yaxis_title="Number of Bets",
                                    showlegend=False,
                                    height=400,
                                    xaxis_tickangle=-45
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # 상세 테이블
                            st.subheader("📋 Detailed Odds Range Statistics (Shadow)")
                            
                            display_shadow = shadow_model_data.copy()
                            display_shadow['_roi_num'] = display_shadow['Actual ROI (%)']
                            display_shadow['Bets'] = display_shadow['Bets'].astype(int)
                            display_shadow['Wins'] = display_shadow['Wins'].astype(int)
                            display_shadow['Win Rate (%)'] = display_shadow['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
                            display_shadow['Avg Odds'] = display_shadow['Avg Odds'].astype(int)
                            display_shadow['Avg Confidence (%)'] = display_shadow['Avg Confidence (%)'].apply(lambda x: f"{x:.2f}")
                            display_shadow['Actual ROI (%)'] = display_shadow['Actual ROI (%)'].apply(lambda x: f"{x:.2f}")
                            
                            def style_shadow_table(row):
                                styles = [''] * len(row)
                                roi_idx = display_shadow.columns.get_loc('Actual ROI (%)')
                                roi_val = row['_roi_num']
                                if roi_val > 0:
                                    styles[roi_idx] = 'color: green; font-weight: bold'
                                else:
                                    styles[roi_idx] = 'color: red; font-weight: bold'
                                return styles
                            
                            styled_shadow = display_shadow.style.apply(style_shadow_table, axis=1)
                            st.dataframe(styled_shadow, use_container_width=True, column_config={'_roi_num': None})
                        else:
                            st.info(f"{selected_shadow_model}에 대한 Shadow 데이터가 없습니다.")
                
                # 전체 모델 비교 히트맵
                if not shadow_odds_analysis.empty:
                    st.markdown("---")
                    st.subheader("🔄 Compare All Shadow Models")
                    
                    odds_order = [
                        '-300 이하 (Heavy Favorite)', '-300 ~ -201', '-200 ~ -151 (Favorite)',
                        '-150 ~ -111', '-110 ~ +99 (Pick\'em)', '+100 ~ +149',
                        '+150 ~ +199', '+200 ~ +299 (Underdog)', '+300 이상 (Heavy Underdog)'
                    ]
                    
                    pivot_shadow = shadow_odds_analysis.pivot(index='Model', columns='Odds Range', values='Actual ROI (%)')
                    pivot_shadow = pivot_shadow.reindex(columns=odds_order)
                    
                    simple_cols = []
                    for col in pivot_shadow.columns:
                        if pd.isna(col):
                            simple_cols.append(col)
                        elif 'Heavy Favorite' in str(col):
                            simple_cols.append('Heavy Fav')
                        elif 'Heavy Underdog' in str(col):
                            simple_cols.append('Heavy Dog')
                        elif 'Favorite' in str(col):
                            simple_cols.append('Favorite')
                        elif 'Underdog' in str(col):
                            simple_cols.append('Underdog')
                        elif 'Pick\'em' in str(col):
                            simple_cols.append('Pick\'em')
                        else:
                            simple_cols.append(str(col))
                    
                    pivot_shadow.columns = simple_cols
                    
                    fig = px.imshow(
                        pivot_shadow,
                        labels=dict(x="Odds Range", y="Model", color="ROI (%)"),
                        color_continuous_scale='RdYlGn',
                        aspect="auto",
                        title="ROI (%) Heatmap - Shadow Models by Odds Range"
                    )
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: 상세 결과
    with tab6:
        st.header("Detailed Betting Results")
        
        # 필터
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_model = st.selectbox(
                "Filter by Model",
                options=['All'] + [m.upper() for m in models],
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
        
        # 추가 필터 (두 번째 줄)
        col4, col5 = st.columns(2)
        
        with col4:
            filter_odds = st.selectbox(
                "Filter by Odds Range",
                options=['All', '-300 이하 (Heavy Favorite)', '-300 ~ -201', 
                        '-200 ~ -151 (Favorite)', '-150 ~ -111', '-110 ~ +99 (Pick\'em)',
                        '+100 ~ +149', '+150 ~ +199', '+200 ~ +299 (Underdog)', 
                        '+300 이상 (Heavy Underdog)']
            )
        
        # 필터 적용 (날짜 필터가 이미 적용된 데이터 사용)
        filtered_results = betting_results_filtered.copy()
        
        if filter_model != 'All':
            filtered_results = filtered_results[filtered_results['model'] == filter_model.lower()]
        
        if filter_result == 'Won':
            filtered_results = filtered_results[filtered_results['won'] == True]
        elif filter_result == 'Lost':
            filtered_results = filtered_results[filtered_results['won'] == False]
        
        if filter_confidence != 'All':
            filtered_results = filtered_results[filtered_results['confidence_level'] == filter_confidence]
        
        if filter_odds != 'All':
            filtered_results = filtered_results[filtered_results['odds_bucket'] == filter_odds]
        
        # 결과 표시
        st.subheader(f"📋 Showing {len(filtered_results)} bets")
        
        if not filtered_results.empty:
            # 컬럼 선택 및 포맷팅
            display_cols = [
                'date', 'model', 'home_team', 'away_team', 'bet_team',
                'bet_probability', 'bet_odds', 'odds_bucket', 'predicted_roi_pct',
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
                'Confidence (%)', 'Odds', 'Odds Range', 'Pred ROI (%)',
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
                file_name=f"nba_model_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("필터 조건에 맞는 데이터가 없습니다.")
    
    # Tab 7: Active vs Shadow 비교
    with tab7:
        st.header("🔄 Active vs Shadow Model Comparison")
        st.markdown("""
        **Active 모델**: 현재 운영 중인 모델 (실제 베팅에 사용)  
        **Shadow 모델**: 테스트 중인 새로운 모델 (주간 전환 대기)
        """)
        
        # Active 데이터 로드
        active_predictions_df = analyzer.load_merged_predictions(exclude_today=True, model_tag='active')
        
        # Shadow 데이터 로드  
        shadow_predictions_df = analyzer.load_merged_predictions(exclude_today=True, model_tag='shadow')
        
        has_active = not active_predictions_df.empty
        has_shadow = not shadow_predictions_df.empty
        
        if not has_active and not has_shadow:
            st.warning("⚠️ Active/Shadow 태그가 붙은 병합 데이터가 없습니다.")
            st.info("""
            **Active/Shadow 시스템 설정 방법:**
            1. `python src/predict_ensemble.py --model-tag active` 또는 `--model-tag shadow` 실행
            2. `python src/analysis/merge_predictions_odds.py --model-tag active` 또는 `--model-tag shadow` 실행
            3. 또는 `--model-tag both`로 둘 다 실행
            """)
        else:
            # 상태 표시
            col1, col2 = st.columns(2)
            with col1:
                if has_active:
                    st.success(f"✅ Active 데이터: {len(active_predictions_df)}개 예측")
                else:
                    st.warning("⚠️ Active 데이터 없음")
            with col2:
                if has_shadow:
                    st.success(f"✅ Shadow 데이터: {len(shadow_predictions_df)}개 예측")
                else:
                    st.warning("⚠️ Shadow 데이터 없음")
            
            # 비교 분석 (둘 다 있는 경우)
            if has_active and has_shadow:
                st.markdown("---")
                st.subheader("📊 Performance Comparison")
                
                # Active ROI 계산
                active_matched = analyzer.match_predictions_with_results(active_predictions_df, results_df)
                if not active_matched.empty:
                    active_models = analyzer.detect_models(active_matched)
                    active_betting = analyzer.calculate_betting_roi(active_matched, active_models)
                    active_perf = analyzer.analyze_model_performance(active_betting)
                    active_perf['Type'] = 'Active'
                else:
                    active_perf = pd.DataFrame()
                
                # Shadow ROI 계산
                shadow_matched = analyzer.match_predictions_with_results(shadow_predictions_df, results_df)
                if not shadow_matched.empty:
                    shadow_models = analyzer.detect_models(shadow_matched)
                    shadow_betting = analyzer.calculate_betting_roi(shadow_matched, shadow_models)
                    shadow_perf = analyzer.analyze_model_performance(shadow_betting)
                    shadow_perf['Type'] = 'Shadow'
                else:
                    shadow_perf = pd.DataFrame()
                
                # 비교 테이블
                if not active_perf.empty and not shadow_perf.empty:
                    comparison_df = pd.concat([active_perf, shadow_perf], ignore_index=True)
                    
                    # 모델별 비교 피벗
                    st.subheader("📈 ROI Comparison by Model")
                    
                    pivot_comparison = comparison_df.pivot(
                        index='Model',
                        columns='Type',
                        values='ROI (%)'
                    ).reset_index()
                    
                    if 'Active' in pivot_comparison.columns and 'Shadow' in pivot_comparison.columns:
                        pivot_comparison['Difference'] = pivot_comparison['Shadow'] - pivot_comparison['Active']
                        pivot_comparison['Better'] = pivot_comparison['Difference'].apply(
                            lambda x: '🟢 Shadow' if x > 0 else ('🔴 Active' if x < 0 else '🟡 Same')
                        )
                        
                        st.dataframe(pivot_comparison, use_container_width=True)
                        
                        # 비교 차트
                        fig = go.Figure()
                        
                        models = pivot_comparison['Model'].tolist()
                        active_roi = pivot_comparison['Active'].tolist() if 'Active' in pivot_comparison.columns else []
                        shadow_roi = pivot_comparison['Shadow'].tolist() if 'Shadow' in pivot_comparison.columns else []
                        
                        fig.add_trace(go.Bar(
                            name='Active',
                            x=models,
                            y=active_roi,
                            marker_color='royalblue',
                            text=[f"{x:.2f}%" for x in active_roi],
                            textposition='outside'
                        ))
                        
                        fig.add_trace(go.Bar(
                            name='Shadow',
                            x=models,
                            y=shadow_roi,
                            marker_color='darkorange',
                            text=[f"{x:.2f}%" for x in shadow_roi],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title="Active vs Shadow ROI Comparison",
                            xaxis_title="Model",
                            yaxis_title="ROI (%)",
                            barmode='group',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 승률 비교
                        st.subheader("📊 Win Rate Comparison")
                        
                        win_pivot = comparison_df.pivot(
                            index='Model',
                            columns='Type',
                            values='Win Rate (%)'
                        ).reset_index()
                        
                        if 'Active' in win_pivot.columns and 'Shadow' in win_pivot.columns:
                            fig_win = go.Figure()
                            
                            fig_win.add_trace(go.Bar(
                                name='Active',
                                x=win_pivot['Model'].tolist(),
                                y=win_pivot['Active'].tolist(),
                                marker_color='royalblue'
                            ))
                            
                            fig_win.add_trace(go.Bar(
                                name='Shadow',
                                x=win_pivot['Model'].tolist(),
                                y=win_pivot['Shadow'].tolist(),
                                marker_color='darkorange'
                            ))
                            
                            fig_win.update_layout(
                                title="Active vs Shadow Win Rate Comparison",
                                xaxis_title="Model",
                                yaxis_title="Win Rate (%)",
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig_win, use_container_width=True)
                        
                        # 추천 결정
                        st.markdown("---")
                        st.subheader("💡 Recommendation")
                        
                        # 전체 평균 ROI 비교
                        active_avg_roi = active_perf['ROI (%)'].mean() if not active_perf.empty else 0
                        shadow_avg_roi = shadow_perf['ROI (%)'].mean() if not shadow_perf.empty else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Active Avg ROI", f"{active_avg_roi:.2f}%")
                        with col2:
                            st.metric("Shadow Avg ROI", f"{shadow_avg_roi:.2f}%")
                        with col3:
                            diff = shadow_avg_roi - active_avg_roi
                            if diff > 2:
                                st.success(f"🟢 Shadow가 {diff:.2f}% 더 우수\n\n**전환 권장!**")
                            elif diff < -2:
                                st.error(f"🔴 Active가 {-diff:.2f}% 더 우수\n\n**전환 보류**")
                            else:
                                st.warning(f"🟡 차이 {abs(diff):.2f}%\n\n**더 많은 데이터 필요**")
                    else:
                        st.info("비교할 데이터가 부족합니다.")
                else:
                    if active_perf.empty:
                        st.warning("Active 매칭 데이터가 없습니다.")
                    if shadow_perf.empty:
                        st.warning("Shadow 매칭 데이터가 없습니다.")
            
            # 단일 데이터만 있는 경우
            elif has_active:
                st.info("Active 데이터만 있습니다. Shadow 모델 예측을 실행하면 비교 분석이 가능합니다.")
            elif has_shadow:
                st.info("Shadow 데이터만 있습니다. Active 모델 예측을 실행하면 비교 분석이 가능합니다.")
    
    # Tab 8: With Odds 모델 성과
    with tab8:
        st.header("🎰 With Odds Models Performance")
        st.markdown("""
        **배당 변수 포함 모델 (With Odds)**
        - 기존 모델에 `home_odds_bucket`, `away_odds_bucket` 변수 추가
        - 8구간 배당 버킷화로 과적합 방지
        - 배당 정보를 활용한 예측 성능 분석
        """)
        
        # With Odds 데이터 로드
        with_odds_files = sorted(analyzer.analysis_dir.glob('merged_predictions_odds_*_with_odds.json'))
        
        if not with_odds_files:
            st.warning("⚠️ With Odds 병합 데이터가 없습니다.")
            st.info("""
            **With Odds 모델 실행 방법:**
            1. `python src/main_with_odds.py` 실행
            2. 또는 개별 실행:
               - `python src/data/processor_modelinput_with_odds.py`
               - `python src/predict_ensemble_with_odds.py`
               - `python src/analysis/merge_predictions_odds_with_odds.py`
            """)
        else:
            # 모든 with_odds 예측 로드
            all_with_odds_predictions = []
            today = datetime.now().strftime('%Y%m%d')
            
            for file in with_odds_files:
                try:
                    # 파일명에서 날짜 추출
                    parts = file.stem.split('_')
                    file_date = parts[-3] if len(parts) >= 5 else parts[-2]
                    
                    # 오늘 파일 제외
                    if file_date == today:
                        continue
                    
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_with_odds_predictions.extend(data)
                except Exception as e:
                    st.warning(f"파일 로드 실패: {file.name}")
                    continue
            
            if not all_with_odds_predictions:
                st.info("📅 오늘 이전의 With Odds 예측 데이터가 없습니다. 내일 경기 결과가 나오면 분석 가능합니다.")
            else:
                with_odds_df = pd.DataFrame(all_with_odds_predictions)
                with_odds_df['date'] = pd.to_datetime(with_odds_df['date'])
                
                st.success(f"✅ With Odds 예측 로드: {len(with_odds_df)}개")
                
                # 결과 매칭
                with_odds_matched = analyzer.match_predictions_with_results(with_odds_df, results_df)
                
                if with_odds_matched.empty:
                    st.warning("⚠️ With Odds 예측과 매칭되는 경기 결과가 없습니다.")
                    st.info("경기가 완료되면 결과가 표시됩니다.")
                else:
                    st.success(f"✅ 매칭 완료: {len(with_odds_matched)}경기")
                    
                    # 모델 감지
                    with_odds_models = analyzer.detect_models(with_odds_matched)
                    
                    # ROI 계산
                    with_odds_betting = analyzer.calculate_betting_roi(with_odds_matched, with_odds_models)
                    
                    # 전체 성과 분석
                    with_odds_perf = analyzer.analyze_model_performance(with_odds_betting)
                    
                    if with_odds_perf.empty:
                        st.warning("성과 데이터가 없습니다.")
                    else:
                        # 메트릭 카드
                        st.markdown("---")
                        st.subheader("📊 Overall Performance (With Odds)")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        best_roi_model = with_odds_perf.iloc[0]
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
                        st.subheader("📈 Model Performance Summary (With Odds)")
                        
                        display_perf = with_odds_perf.copy()
                        display_perf['_roi_num'] = display_perf['ROI (%)']
                        display_perf['_profit_num'] = display_perf['Total Profit ($)']
                        
                        display_perf['Win Rate (%)'] = display_perf['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
                        display_perf['Total Profit ($)'] = display_perf['Total Profit ($)'].apply(lambda x: f"{x:.2f}")
                        display_perf['ROI (%)'] = display_perf['ROI (%)'].apply(lambda x: f"{x:.2f}")
                        display_perf['Avg Odds'] = display_perf['Avg Odds'].round(0).astype(int)
                        display_perf['Avg Confidence'] = display_perf['Avg Confidence'].apply(lambda x: f"{x:.2f}")
                        
                        def style_with_odds_perf(row):
                            styles = [''] * len(row)
                            roi_idx = display_perf.columns.get_loc('ROI (%)')
                            profit_idx = display_perf.columns.get_loc('Total Profit ($)')
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
                        
                        styled_perf = display_perf.style.apply(style_with_odds_perf, axis=1)
                        st.dataframe(styled_perf, use_container_width=True, height=400, column_config={
                            '_roi_num': None,
                            '_profit_num': None
                        })
                        
                        # ROI 비교 차트
                        st.subheader("📊 ROI Comparison (With Odds)")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=with_odds_perf['Model'],
                            y=with_odds_perf['ROI (%)'],
                            marker_color=['green' if x > 0 else 'red' for x in with_odds_perf['ROI (%)']],
                            text=with_odds_perf['ROI (%)'].round(2),
                            textposition='outside'
                        ))
                        fig.update_layout(
                            title="With Odds Model ROI Comparison (%)",
                            xaxis_title="Model",
                            yaxis_title="ROI (%)",
                            showlegend=False,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 배당 버킷 분포 표시 (있는 경우)
                        if 'home_odds_bucket' in with_odds_matched.columns:
                            st.markdown("---")
                            st.subheader("🎰 Odds Bucket Distribution")
                            
                            bucket_labels = {
                                0: '압도적 페이버릿 (<-400)',
                                1: '강한 페이버릿 (-400~-250)',
                                2: '페이버릿 (-250~-150)',
                                3: '약한 페이버릿 (-150~-100)',
                                4: '약한 언더독 (-100~+150)',
                                5: '언더독 (+150~+250)',
                                6: '강한 언더독 (+250~+400)',
                                7: '압도적 언더독 (>+400)'
                            }
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                home_bucket_counts = with_odds_matched['home_odds_bucket'].value_counts().sort_index()
                                fig_home = go.Figure()
                                fig_home.add_trace(go.Bar(
                                    x=[bucket_labels.get(int(b), str(b)) for b in home_bucket_counts.index],
                                    y=home_bucket_counts.values,
                                    marker_color='royalblue'
                                ))
                                fig_home.update_layout(
                                    title="Home Team Odds Bucket Distribution",
                                    xaxis_title="Bucket",
                                    yaxis_title="Count",
                                    height=400,
                                    xaxis_tickangle=-45
                                )
                                st.plotly_chart(fig_home, use_container_width=True)
                            
                            with col2:
                                away_bucket_counts = with_odds_matched['away_odds_bucket'].value_counts().sort_index()
                                fig_away = go.Figure()
                                fig_away.add_trace(go.Bar(
                                    x=[bucket_labels.get(int(b), str(b)) for b in away_bucket_counts.index],
                                    y=away_bucket_counts.values,
                                    marker_color='darkorange'
                                ))
                                fig_away.update_layout(
                                    title="Away Team Odds Bucket Distribution",
                                    xaxis_title="Bucket",
                                    yaxis_title="Count",
                                    height=400,
                                    xaxis_tickangle=-45
                                )
                                st.plotly_chart(fig_away, use_container_width=True)
                        
                        # 신뢰도 구간별 분석
                        st.markdown("---")
                        st.subheader("🎯 Performance by Confidence (With Odds)")
                        
                        with_odds_conf = analyzer.analyze_by_confidence(with_odds_betting)
                        
                        if not with_odds_conf.empty:
                            # 모델 선택
                            selected_model_wo = st.selectbox(
                                "Select Model",
                                options=[m.upper() for m in with_odds_models],
                                key='with_odds_conf_model'
                            )
                            
                            if selected_model_wo:
                                model_conf = with_odds_conf[with_odds_conf['Model'] == selected_model_wo].copy()
                                
                                if not model_conf.empty:
                                    confidence_order = ['50-60%', '60-70%', '70-80%', '80%+']
                                    model_conf['Confidence'] = pd.Categorical(
                                        model_conf['Confidence'],
                                        categories=confidence_order,
                                        ordered=True
                                    )
                                    model_conf = model_conf.sort_values('Confidence')
                                    
                                    cols = st.columns(len(model_conf))
                                    for idx, (_, row) in enumerate(model_conf.iterrows()):
                                        with cols[idx]:
                                            st.metric(
                                                label=row['Confidence'],
                                                value=f"{row['ROI (%)']:.2f}%",
                                                delta=f"{row['Win Rate (%)']:.1f}% Win"
                                            )
                                            st.caption(f"{int(row['Bets'])} bets")


if __name__ == "__main__":
    main()

