# src/models/model1_totals.py
"""
NBA 총점(Over/Under) 예측 모델 - LightGBM Regressor
기존 model1.py 기반, 타겟만 total_score로 변경
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor


class TotalsModel1:
    """총점 예측 모델 (LightGBM Regressor)"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent / "saved_models"
        self.model_dir.mkdir(exist_ok=True)
        self.dates = None
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 레이블 추출"""
        df = pd.DataFrame(data)
        
        # 날짜 기준으로 정렬
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        self.dates = df['date']
        
        # ★ 타겟: 총점 (홈팀 점수 + 원정팀 점수)
        y = df['home_team_score'].astype(float) + df['away_team_score'].astype(float)
        
        # 기본 특성 선택 (model1.py와 동일)
        base_features = [
            # 기본 경기력 지표
            'home_rebounds', 'away_rebounds',
            'home_assists', 'away_assists',
            'home_fieldGoalsAttempted', 'away_fieldGoalsAttempted',
            'home_fieldGoalsMade', 'away_fieldGoalsMade',
            'home_fieldGoalPct', 'away_fieldGoalPct',
            'home_freeThrowsAttempted', 'away_freeThrowsAttempted',
            'home_freeThrowsMade', 'away_freeThrowsMade',
            'home_freeThrowPct', 'away_freeThrowPct',
            'home_threePointFieldGoalsAttempted', 'away_threePointFieldGoalsAttempted',
            'home_threePointFieldGoalsMade', 'away_threePointFieldGoalsMade',
            'home_threePointPct', 'away_threePointPct',
            
            # 리더 통계
            'home_leader_points', 'away_leader_points',
            'home_leader_rebounds', 'away_leader_rebounds',
            'home_leader_assists', 'away_leader_assists',
            
            # 팀 기록
            'home_overall_record_win_rate', 'away_overall_record_win_rate',
            'home_home_record_win_rate', 'away_home_record_win_rate',
            'home_road_record_win_rate', 'away_road_record_win_rate',
            'home_vs_away_win_rate',
            
            # 최근 트렌드
            'home_recent_win_rate', 'away_recent_win_rate',
            'home_recent_avg_score', 'away_recent_avg_score',
            'home_recent_home_win_rate', 'away_recent_home_win_rate',
            'home_recent_away_win_rate', 'away_recent_away_win_rate',
            
            # 컨디션
            'home_rest_days', 'away_rest_days'
        ]
        
        # 기본 특성으로 DataFrame 생성
        X = df[base_features].copy()
        
        # 최근 트렌드 특성 복제 (model1과 동일)
        recent_features = [
            'recent_win_rate',
            'recent_avg_score',
            'recent_home_win_rate',
            'recent_away_win_rate'
        ]
        
        for col in recent_features:
            for team in ['home', 'away']:
                orig_col = f'{team}_{col}'
                new_col = f'{orig_col}_2'
                X[new_col] = X[orig_col]
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """LightGBM Regressor 모델 학습"""
        n_samples = len(X)
        # 지수적 증가 가중치 (최근 데이터에 더 급격한 가중치)
        sample_weights = np.exp(np.linspace(0, 1, n_samples))
        
        # 파라미터 (model1.py와 유사, objective만 regression으로 변경)
        best_params = {
            'objective': 'regression',  # ★ 회귀로 변경
            'metric': 'rmse',            # ★ 평가 지표 변경
            'colsample_bytree': 0.7,
            'learning_rate': 0.05,
            'max_depth': 4,
            'min_child_samples': 70,
            'n_estimators': 300,
            'num_leaves': 24,
            'reg_alpha': 1.0,
            'reg_lambda': 10.0,
            'subsample': 0.65,
            'random_state': 42,
            'verbose': -1,
            'boosting_type': 'gbdt',
            'importance_type': 'gain',
            'feature_fraction_seed': 42,
            'bagging_seed': 42
        }
        
        # ★ Regressor로 변경
        self.model = LGBMRegressor(**best_params)
        
        print("\n=== 총점 예측 모델 학습 시작 ===")
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # 특성 중요도 계산
        importances = self.model.booster_.feature_importance(importance_type='gain')
        importances = 100.0 * (importances / importances.sum())
        
        metrics = {
            'feature_importance': dict(zip(
                self.feature_names,
                importances
            ))
        }
        
        # 상위 10개 중요 특성 출력
        print("\n=== 상위 10개 중요 특성 (%) ===")
        sorted_features = sorted(
            metrics['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.2f}%")
        
        return metrics
    
    def evaluate_recent_games(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        """학습된 모델로 최근 N경기 예측 성능 평가"""
        
        # 최근 n_games 선택
        X_recent = X[-n_games:]
        y_recent = y[-n_games:]
        dates_recent = self.dates[-n_games:]
        
        # 예측 수행 (회귀이므로 연속값 반환)
        y_pred = self.model.predict(X_recent)
        
        # ★ 회귀 평가 지표
        mae = mean_absolute_error(y_recent, y_pred)
        rmse = np.sqrt(mean_squared_error(y_recent, y_pred))
        r2 = r2_score(y_recent, y_pred)
        
        # 결과 저장
        results = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': list(zip(
                dates_recent.dt.strftime('%Y-%m-%d').tolist(),
                y_recent.tolist(),
                y_pred.tolist()
            ))
        }
        
        # 결과 출력
        print(f"\n=== 최근 {n_games}경기 총점 예측 성능 ===")
        print(f"MAE (평균 절대 오차): {mae:.2f}점")
        print(f"RMSE (평균 제곱근 오차): {rmse:.2f}점")
        print(f"R² (결정 계수): {r2:.3f}")
        
        print(f"\n=== 최근 10경기 예측 상세 ===")
        print(f"{'날짜':<12} {'실제 총점':>10} {'예측 총점':>10} {'오차':>8}")
        print("-" * 45)
        for date, actual, pred in results['predictions'][-10:]:
            error = pred - actual
            print(f"{date:<12} {actual:>10.0f} {pred:>10.1f} {error:>+8.1f}")
        
        # 추가 통계
        errors = y_pred - y_recent.values
        print(f"\n=== 오차 분포 ===")
        print(f"평균 오차: {errors.mean():+.2f}점")
        print(f"오차 표준편차: {errors.std():.2f}점")
        print(f"최대 과대예측: {errors.max():+.1f}점")
        print(f"최대 과소예측: {errors.min():+.1f}점")
        
        # 오차 범위별 정확도
        within_5 = np.sum(np.abs(errors) <= 5) / len(errors) * 100
        within_10 = np.sum(np.abs(errors) <= 10) / len(errors) * 100
        within_15 = np.sum(np.abs(errors) <= 15) / len(errors) * 100
        
        print(f"\n=== 오차 범위별 적중률 ===")
        print(f"±5점 이내: {within_5:.1f}%")
        print(f"±10점 이내: {within_10:.1f}%")
        print(f"±15점 이내: {within_15:.1f}%")
        
        return results
    
    def save_model(self, timestamp: str = None) -> None:
        """학습된 모델과 특성 이름을 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. 먼저 모델을 학습해주세요.")
        
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 파일 경로 (totals 구분)
        model_path = self.model_dir / f"totals_model1_{timestamp}.joblib"
        feature_path = self.model_dir / f"totals_features1_{timestamp}.json"
        
        # 모델 저장
        joblib.dump(self.model, model_path)
        
        # 특성 이름 저장
        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {
                    'type': 'lightgbm_regressor',
                    'target': 'total_score',
                    'params': self.model.get_params()
                }
            }, f, indent=2)
        
        print(f"\n=== 모델 저장 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"특성 정보 저장 경로: {feature_path}")


def get_latest_processed_data() -> List[Dict]:
    """src/data 폴더에서 가장 최신의 processed json 파일 로드"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # processed_ 파일만 찾기 (with_odds, prediction, spread 제외)
    json_files = list(data_dir.glob("processed_*.json"))
    json_files = [f for f in json_files 
                  if 'prediction' not in f.name 
                  and 'spread' not in f.name 
                  and 'with_odds' not in f.name]
    
    if not json_files:
        raise FileNotFoundError("처리된 데이터 파일을 찾을 수 없습니다.")
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"데이터 파일 로드: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # 최신 데이터 로드
    data = get_latest_processed_data()
    
    # 완료된 경기만 필터링
    data = [g for g in data if g.get('status') == 'STATUS_FINAL']
    print(f"완료된 경기 수: {len(data)}")
    
    # 모델 초기화 및 특성 준비
    model = TotalsModel1()
    X, y = model.prepare_features(data)
    
    print("\n=== 데이터 준비 완료 ===")
    print(f"특성 수: {len(model.feature_names)}")
    print(f"샘플 수: {len(X)}")
    print(f"타겟 (총점) 평균: {y.mean():.1f}")
    print(f"타겟 (총점) 범위: {y.min():.0f} ~ {y.max():.0f}")
    
    # 전체 데이터로 모델 학습
    metrics = model.train_model(X, y)
    
    # 최근 70경기 성능 평가
    eval_results = model.evaluate_recent_games(X, y, n_games=70)
    
    # 모델 저장
    model.save_model()

