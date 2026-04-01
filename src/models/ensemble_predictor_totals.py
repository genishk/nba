# src/models/ensemble_predictor_totals.py
# 총점 예측용 앙상블 예측기

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import joblib
from datetime import datetime


class EnsemblePredictorTotals:
    """총점 예측용 앙상블 예측기 (회귀 모델)"""
    
    def __init__(self):
        """앙상블 예측기 초기화"""
        self.models = []
        self.feature_names = None
        self.model_dir = Path(__file__).parent / "saved_models"
        
    def load_latest_models(self) -> List[Dict]:
        """각 totals 모델의 최신 버전 로드 (1~8번 모델)"""
        loaded_models = []
        
        # 모델 타입 정보
        model_types = {
            1: 'lightgbm_regressor',
            2: 'catboost_regressor', 
            3: 'xgboost_regressor',
            4: 'lightgbm_gbdt_regressor',
            5: 'catboost_ordered_regressor',
            6: 'xgboost_hist_regressor',
            7: 'random_forest_regressor',
            8: 'extra_trees_regressor'
        }
        
        print(f"\n=== [TOTALS] 모델 로드 시작 ===")
        
        # 각 모델 타입별로 최신 모델 로드 (1~8)
        for model_num in range(1, 9):
            # totals 모델 파일 찾기
            model_files = list(self.model_dir.glob(f"totals_model{model_num}_*.joblib"))
            
            if not model_files:
                print(f"  ⚠️ totals_model{model_num} 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue
                
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # 특성 파일 찾기
            feature_files = list(self.model_dir.glob(f"totals_features{model_num}_*.json"))
            if not feature_files:
                print(f"  ⚠️ totals_features{model_num} 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue
            feature_file = max(feature_files, key=lambda x: x.stat().st_mtime)
            
            # 모델 로드
            model = joblib.load(latest_model)
            
            # 특성 정보 로드
            with open(feature_file, 'r') as f:
                feature_info = json.load(f)
            
            loaded_models.append({
                'model': model,
                'features': feature_info['feature_names'],
                'type': f'model{model_num}',
                'algorithm': model_types.get(model_num, 'unknown')
            })
            
            print(f"  ✅ Model{model_num} ({model_types.get(model_num, 'unknown')}) 로드 완료")
            print(f"      - 파일: {latest_model.name}")
        
        if not loaded_models:
            raise FileNotFoundError("로드할 수 있는 totals 모델이 없습니다.")
        
        self.models = loaded_models
        print(f"\n=== [TOTALS] 총 {len(loaded_models)}개 모델 로드 완료 ===")
        return loaded_models
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터에서 특성 추출 (분류 모델과 동일한 피처 사용)"""
        # 날짜 기준으로 정렬
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
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
        X = data[base_features].copy()
        
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
                X[new_col] = X[orig_col]
        
        return X
    
    def predict_games(self, df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
        """앙상블 총점 예측 수행 (회귀)"""
        X = self.prepare_features(df)
        
        if weights is None:
            # 기본 가중치 설정 (로드된 모델에 대해 동일 가중치)
            weights = {model_info['type']: 1.0 for model_info in self.models}
        
        # 각 모델의 예측값 저장
        model_predictions = {}
        
        for model_info in self.models:
            model = model_info['model']
            model_type = model_info['type']
            
            # 회귀 예측 (predict_proba 대신 predict 사용)
            pred = model.predict(X)
            model_predictions[model_type] = pred
        
        # 가중 평균 계산
        total_weight = sum(weights.get(model_info['type'], 0) for model_info in self.models)
        if total_weight == 0:
            total_weight = 1
            
        weighted_sum = np.zeros(len(df))
        for model_info in self.models:
            model_type = model_info['type']
            weight = weights.get(model_type, 0)
            weighted_sum += model_predictions[model_type] * weight
        
        ensemble_predictions = weighted_sum / total_weight
        
        # 결과 DataFrame 생성
        results_df = df[['date', 'home_team_name', 'away_team_name']].copy()
        results_df['predicted_total'] = ensemble_predictions
        
        # 각 모델의 개별 예측값 추가 (1~8번 모델)
        for i in range(1, 9):
            col_name = f'model{i}_predicted_total'
            results_df[col_name] = model_predictions.get(f'model{i}', np.zeros(len(df)))
        
        # 날짜 형식 변환
        results_df['date'] = pd.to_datetime(results_df['date']).dt.strftime('%Y-%m-%d')
        
        # 모델 이름 매핑
        model_names = {
            'model1': 'LightGBM',
            'model2': 'CatBoost',
            'model3': 'XGBoost',
            'model4': 'LightGBM-GBDT',
            'model5': 'CatBoost-Ordered',
            'model6': 'XGBoost-Hist',
            'model7': 'RandomForest',
            'model8': 'ExtraTrees'
        }
        
        # 결과 출력
        print("\n=== 🏀 총점 예측 결과 ===")
        for _, row in results_df.iterrows():
            print(f"\n📅 {row['date']} 경기:")
            print(f"   {row['home_team_name']} vs {row['away_team_name']}")
            print(f"   🎯 예측 총점: {row['predicted_total']:.1f}점")
            
            # 로드된 모델들의 예측값 출력
            print("   개별 모델 예측:")
            for model_info in self.models:
                model_type = model_info['type']
                model_name = model_names.get(model_type, model_type)
                col_name = f'{model_type}_predicted_total'
                if col_name in row:
                    print(f"     - {model_name}: {row[col_name]:.1f}점")
        
        return results_df
    
    def save_predictions(self, predictions: pd.DataFrame) -> Path:
        """예측 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "predictions"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"totals_predictions_{timestamp}.json"
        
        predictions.to_json(output_path, orient='records', indent=2)
        
        print(f"\n=== 총점 예측 결과 저장 완료 ===")
        print(f"저장 경로: {output_path}")
        
        return output_path
    
    def load_prediction_data(self) -> pd.DataFrame:
        """최신 예측용 데이터 로드 (기존과 동일한 파일 사용)"""
        data_dir = Path(__file__).parent.parent / "data"
        
        # model_input_features로 시작하는 가장 최신 파일 찾기
        # with_odds 및 totals 제외
        pred_files = list(data_dir.glob("model_input_features_*.json"))
        pred_files = [f for f in pred_files if 'with_odds' not in f.name and 'totals' not in f.name]
        
        if not pred_files:
            raise FileNotFoundError("예측할 데이터 파일을 찾을 수 없습니다.")
        
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        print(f"\n예측 데이터 파일 로드: {latest_file.name}")
        
        # JSON 파일 로드 및 DataFrame 변환
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

