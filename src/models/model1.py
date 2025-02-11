# src/models/model1.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization






class BettingModel1:
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
        """최적 파라미터로 LightGBM 모델 학습"""
        from lightgbm import LGBMClassifier
        import numpy as np

        n_samples = len(X)
        # 지수적 증가 가중치 (최근 데이터에 더 급격한 가중치)
        sample_weights = np.exp(np.linspace(0, 1, n_samples))

        # # 최적 파라미터 설정
        # best_params = {
        #     'colsample_bytree': 0.688,
        #     'learning_rate': 0.194,
        #     'max_depth': 8,
        #     'min_child_samples': 29,
        #     'n_estimators': 151,
        #     'num_leaves': 50,
        #     'reg_alpha': 0.525,
        #     'reg_lambda': 6.402,
        #     'subsample': 0.827,
        #     'random_state': 42,
        #     'verbose': -1
        # }
        
        best_params = {
            'colsample_bytree': 0.8,        # 더 많은 특성 사용
            'learning_rate': 0.05,          # 낮은 학습률로 안정적 학습
            'max_depth': 5,                 # 적당한 깊이로 과적합 방지
            'min_child_samples': 50,        # 안정적인 리프 노드
            'n_estimators': 300,            # 충분한 트리 개수
            'num_leaves': 32,               # 적은 리프 노드로 과적합 방지
            'reg_alpha': 1.0,               # L1 규제
            'reg_lambda': 10.0,             # L2 규제
            'subsample': 0.7,               # 데이터 샘플링으로 과적합 방지
            'random_state': 42,
            'verbose': -1,
            'boosting_type': 'gbdt',
            'importance_type': 'gain',
            'feature_fraction_seed': 42,
            'bagging_seed': 42
        }
        
        # 모델 초기화 및 학습
        self.model = LGBMClassifier(**best_params)
        
        print("\n=== 모델 학습 시작 ===")
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # 특성 중요도 계산 (gain 기준으로 변경하고 정규화)
        importances = self.model.booster_.feature_importance(importance_type='gain')
        importances = 100.0 * (importances / importances.sum())  # 퍼센트로 변환
        
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
        model_path = self.model_dir / f"betting_model1_{timestamp}.joblib"
        feature_path = self.model_dir / f"features1_{timestamp}.json"
        
        # 모델 저장
        joblib.dump(self.model, model_path)
        
        # 특성 이름 저장
        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {
                    'type': 'lightgbm',
                    'params': self.model.get_params()
                }
            }, f, indent=2)
        
        print(f"\n=== 모델 저장 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"특성 정보 저장 경로: {feature_path}")


def get_latest_processed_data() -> List[Dict]:
    """src/data 폴더에서 가장 최신의 processed json 파일 로드 (prediction 제외)"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # prediction이 포함되지 않은 processed_ 파일만 찾기
    json_files = list(data_dir.glob("processed_*.json"))
    json_files = [f for f in json_files if 'prediction' not in f.name]
    
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
    model = BettingModel1()
    X, y = model.prepare_features(data)
    
    print("\n=== 데이터 준비 완료 ===")
    print(f"특성 수: {len(model.feature_names)}")
    print(f"샘플 수: {len(X)}")
    
    # 전체 데이터로 모델 학습
    metrics = model.train_model(X, y)
    
    # 최근 600경기 성능 평가
    eval_results = model.evaluate_recent_games(X, y, n_games=50)
    
    # 모델 저장
    model.save_model()
