# src/models/model2.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier





class BettingModel2:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent / "saved_models"
        self.model_dir.mkdir(exist_ok=True)
        self.dates = None  # 날짜 정보 저장을 위한 변수 추가
    
    # def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
    #     """데이터에서 특성과 레이블 추출"""
    #     df = pd.DataFrame(data)
        
    #     # 날짜 기준으로 정렬
    #     df['date'] = pd.to_datetime(df['date'])
    #     df = df.sort_values('date')
    #     self.dates = df['date']  # 날짜 정보 저장
        
    #     # 승패 레이블 생성 (홈팀 기준)
    #     y = (df['home_team_score'] > df['away_team_score']).astype(int)
        
    #     # 특성 선택
    #     features = [
    #         # 기본 경기력 지표
    #         'home_rebounds', 'away_rebounds',
    #         'home_assists', 'away_assists',
    #         'home_fieldGoalsAttempted', 'away_fieldGoalsAttempted',
    #         'home_fieldGoalsMade', 'away_fieldGoalsMade',
    #         'home_fieldGoalPct', 'away_fieldGoalPct',
    #         'home_freeThrowsAttempted', 'away_freeThrowsAttempted',
    #         'home_freeThrowsMade', 'away_freeThrowsMade',
    #         'home_freeThrowPct', 'away_freeThrowPct',
    #         'home_threePointFieldGoalsAttempted', 'away_threePointFieldGoalsAttempted',
    #         'home_threePointFieldGoalsMade', 'away_threePointFieldGoalsMade',
    #         'home_threePointPct', 'away_threePointPct',
            
    #         # 리더 통계
    #         'home_leader_points', 'away_leader_points',
    #         'home_leader_rebounds', 'away_leader_rebounds',
    #         'home_leader_assists', 'away_leader_assists',
            
    #         # 팀 기록
    #         'home_overall_record_win_rate', 'away_overall_record_win_rate',
    #         'home_home_record_win_rate', 'away_home_record_win_rate',
    #         'home_road_record_win_rate', 'away_road_record_win_rate',
    #         'home_vs_away_win_rate',
            
    #         # 최근 트렌드
    #         'home_recent_win_rate', 'away_recent_win_rate',
    #         'home_recent_avg_score', 'away_recent_avg_score',
    #         'home_recent_home_win_rate', 'away_recent_home_win_rate',
    #         'home_recent_away_win_rate', 'away_recent_away_win_rate',
            
    #         # 컨디션
    #         'home_rest_days', 'away_rest_days'
    #     ]
        
    #     X = df[features]
    #     self.feature_names = X.columns.tolist()
        
    #     return X, y
    
    # def optimize_params(self, X: pd.DataFrame, y: pd.Series) -> Dict:
    #     """베이지안 최적화로 CatBoost 최적 파라미터 탐색"""
    #     def objective(
    #         learning_rate,
    #         depth,
    #         l2_leaf_reg,
    #         subsample,
    #         random_strength
    #     ):
    #         params = {
    #             'iterations': 1000,  # 고정 파라미터
    #             'learning_rate': learning_rate,
    #             'depth': int(depth),
    #             'l2_leaf_reg': l2_leaf_reg,
    #             'bootstrap_type': 'Bernoulli',
    #             'subsample': subsample,
    #             'random_strength': random_strength,
    #             'verbose': False,
    #             'random_state': 42
    #         }
            
    #         # TimeSeriesSplit으로 시계열 교차 검증
    #         tscv = TimeSeriesSplit(n_splits=5)
    #         scores = []
            
    #         for train_idx, val_idx in tscv.split(X):
    #             X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    #             y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
    #             model = CatBoostClassifier(**params)
    #             model.fit(X_train, y_train, eval_set=(X_val, y_val),
    #                     early_stopping_rounds=50, verbose=False)
                
    #             pred = model.predict_proba(X_val)[:, 1]
    #             score = roc_auc_score(y_val, pred)
    #             scores.append(score)
            
    #         return np.mean(scores)

    #     # 파라미터 범위 설정
    #     param_bounds = {
    #         'learning_rate': (0.01, 0.3),
    #         'depth': (3, 10),
    #         'l2_leaf_reg': (1, 10),
    #         'subsample': (0.5, 1.0),
    #         'random_strength': (0, 10)
    #     }

    #     optimizer = BayesianOptimization(
    #         f=objective,
    #         pbounds=param_bounds,
    #         random_state=42
    #     )

    #     print("\n=== 베이지안 최적화 시작 ===")
    #     optimizer.maximize(
    #         init_points=5,
    #         n_iter=25
    #     )

    #     # 최적 파라미터 추출
    #     best_params = {
    #         'iterations': 1000,
    #         'learning_rate': optimizer.max['params']['learning_rate'],
    #         'depth': int(optimizer.max['params']['depth']),
    #         'l2_leaf_reg': optimizer.max['params']['l2_leaf_reg'],
    #         'bootstrap_type': 'Bernoulli',
    #         'subsample': optimizer.max['params']['subsample'],
    #         'random_strength': optimizer.max['params']['random_strength'],
    #         'verbose': False,
    #         'random_state': 42
    #     }

    #     print("\n=== 최적 파라미터 ===")
    #     for param, value in best_params.items():
    #         print(f"{param}: {value}")

    #     return best_params
    
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 레이블 추출"""
        df = pd.DataFrame(data)
        
        # 날짜 기준으로 정렬
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        self.dates = df['date']  # 날짜 정보 저장
        
        # 승패 레이블 생성 (홈팀 기준)
        y = (df['home_team_score'] > df['away_team_score']).astype(int)
        
        # 기본 특성 선택
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
        
        # 최근 트렌드 특성 복제
        recent_features = [
            'recent_win_rate',
            'recent_avg_score',
            'recent_home_win_rate',
            'recent_away_win_rate'
        ]
        
        # 복제된 특성 추가
        for col in recent_features:
            for team in ['home', 'away']:
                orig_col = f'{team}_{col}'
                new_col = f'{orig_col}_2'
                X[new_col] = X[orig_col]  # 동일한 값으로 복제
        
        self.feature_names = X.columns.tolist()
        
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """최적 파라미터로 CatBoost 모델 학습"""
        
        # # 최적 파라미터 설정
        # best_params = {
        #     'iterations': 1000,
        #     'learning_rate': 0.29700612045944963,
        #     'depth': 3,
        #     'l2_leaf_reg': 7.718932298487503,
        #     'bootstrap_type': 'Bernoulli',
        #     'subsample': 0.8095576540733551,
        #     'random_strength': 2.054604742936732,
        #     'verbose': False,
        #     'random_state': 42
        # }
        # best_params = {
        #     'iterations': 3000,              # 더 많은 트리로 복잡한 패턴 학습
        #     'learning_rate': 0.03,          # 더 작은 학습률로 안정적인 학습
        #     'depth': 6,                     # 적당한 깊이로 과적합과 과소적합 사이 균형
        #     'l2_leaf_reg': 3.0,            # 적절한 규제로 과적합 방지
        #     'bootstrap_type': 'Bernoulli',
        #     'subsample': 0.85,             # 약간 더 많은 데이터 사용
        #     'random_strength': 1.0,        # 기본값으로 설정
        #     'verbose': False,
        #     'random_state': 42,
        #     'early_stopping_rounds': 100,   # 조기 종료로 과적합 방지
        #     'task_type': 'CPU',            # 명시적 CPU 사용
        #     'loss_function': 'Logloss',    # 이진 분류에 적합
        #     'eval_metric': 'AUC',          # ROC-AUC로 평가
        #     'leaf_estimation_method': 'Newton'  # 더 정확한 리프 값 추정
        # }
        
        n_samples = len(X)
        # 지수적 증가 가중치 (최근 데이터에 더 급격한 가중치)
        sample_weights = np.exp(np.linspace(0, 1, n_samples))
        
        best_params = {
            'iterations': 1500,              # 3000 -> 1500으로 줄여서 과적합 방지
            'learning_rate': 0.01,          # 0.03 -> 0.01로 더 낮춰서 안정성 확보
            'depth': 3,                     # 6 -> 4로 줄여서 모델 복잡도 감소
            'l2_leaf_reg': 6.0,            # 3.0 -> 5.0으로 높여서 규제 강화
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.7,              # 0.85 -> 0.7로 줄여서 과적합 방지
            'random_strength': 2.5,        # 랜덤성 약간 증가
            'early_stopping_rounds': 50,    # 100 -> 50으로 줄여서 더 일찍 멈추게
            'task_type': 'CPU',
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'leaf_estimation_method': 'Newton',
            'class_weights': {0: 1.0, 1: 1.0},  # 클래스 가중치 균등하게 설정
            'verbose': False
        }
        
        # 모델 초기화 및 학습
        self.model = CatBoostClassifier(**best_params)
        
        print("\n=== 모델 학습 시작 ===")
        self.model.fit(X, y, sample_weight=sample_weights)
        
        
        # 특성 중요도 계산
        importances = self.model.feature_importances_
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
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
        
        # 최근 n_games 선택
        X_recent = X[-n_games:]
        y_recent = y[-n_games:]
        dates_recent = self.dates[-n_games:]
        
        # 예측 수행
        y_pred = self.model.predict(X_recent)
        y_pred_proba = self.model.predict_proba(X_recent)[:, 1]
        
        # 혼동 행렬 계산
        conf_matrix = confusion_matrix(y_recent, y_pred)
        
        # 결과 저장
        results = {
            'accuracy': accuracy_score(y_recent, y_pred),
            'roc_auc': roc_auc_score(y_recent, y_pred_proba),
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': list(zip(
                dates_recent.dt.strftime('%Y-%m-%d').tolist(),
                y_recent.tolist(),
                y_pred.tolist(),
                y_pred_proba.tolist()
            ))
        }
        
        # 결과 출력
        print(f"\n=== 최근 {n_games}경기 예측 성능 ===")  # 동적 메시지로 수정
        print(f"정확도: {results['accuracy']:.3f}")
        print(f"ROC-AUC: {results['roc_auc']:.3f}")
        print("\n혼동 행렬:")
        print(f"TN: {conf_matrix[0,0]}, FP: {conf_matrix[0,1]}")
        print(f"FN: {conf_matrix[1,0]}, TP: {conf_matrix[1,1]}")
        
        print("\n=== 최근 10경기 예측 상세 ===")
        for date, true, pred, prob in results['predictions'][-10:]:
            print(f"날짜: {date}, 실제: {true}, 예측: {pred}, 승리확률: {prob:.3f}")
        
        return results
    
    def save_model(self, timestamp: str = None) -> None:
        """학습된 모델과 특성 이름을 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. 먼저 모델을 학습해주세요.")
        
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 파일 경로
        model_path = self.model_dir / f"betting_model2_{timestamp}.joblib"
        feature_path = self.model_dir / f"features2_{timestamp}.json"
        
        # 모델 저장
        joblib.dump(self.model, model_path)
        
        # 특성 이름 저장
        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {
                    'type': 'catboost',
                    'params': self.model.get_params()
                }
            }, f, indent=2)
        
        print(f"\n=== 모델 저장 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"특성 정보 저장 경로: {feature_path}")


def get_latest_processed_data() -> List[Dict]:
    """src/data 폴더에서 가장 최신의 processed json 파일 로드 (prediction 제외)"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # prediction, spread 제외한 processed_ 파일만 찾기 (모델 학습용 40일 데이터)
    json_files = list(data_dir.glob("processed_*.json"))
    json_files = [f for f in json_files if 'prediction' not in f.name and 'spread' not in f.name]
    
    if not json_files:
        raise FileNotFoundError("처리된 데이터 파일을 찾을 수 없습니다.")
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"데이터 파일 로드: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)



if __name__ == "__main__":
    # 최신 데이터 로드
    data = get_latest_processed_data()
    
    # 모델 초기화 및 특성 준비
    model = BettingModel2()
    X, y = model.prepare_features(data)
    
    print("\n=== 데이터 준비 완료 ===")
    print(f"특성 수: {len(model.feature_names)}")
    print(f"샘플 수: {len(X)}")
    
    # 전체 데이터로 모델 학습
    metrics = model.train_model(X, y)
    
    # 최근 600경기 성능 평가
    eval_results = model.evaluate_recent_games(X, y, n_games=70)
    
    # 모델 저장
    model.save_model()
