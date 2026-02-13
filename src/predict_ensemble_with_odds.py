# src/predict_ensemble_with_odds.py
"""
배당 변수 포함 앙상블 예측
- model*_with_odds 모델들을 사용한 예측
- home_odds_bucket, away_odds_bucket 변수 포함
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import joblib
from datetime import datetime
import argparse


class EnsemblePredictorWithOdds:
    """배당 변수를 포함한 앙상블 예측기"""
    
    def __init__(self):
        self.models = []
        self.feature_names = None
        self.model_dir = Path(__file__).parent / "models" / "saved_models"
        
    def load_latest_models(self) -> List[Dict]:
        """with_odds 모델들 로드 (1~8번)"""
        loaded_models = []
        
        model_types = {
            1: 'lightgbm',
            2: 'catboost', 
            3: 'xgboost',
            4: 'lightgbm_dart',
            5: 'catboost_ordered',
            6: 'xgboost_hist',
            7: 'random_forest',
            8: 'extra_trees'
        }
        
        print("\n=== [WITH_ODDS] 배당 포함 모델 로드 ===")
        
        for model_num in range(1, 9):
            # with_odds 모델 파일 찾기
            model_files = list(self.model_dir.glob(f"betting_model{model_num}_with_odds_*.joblib"))
            
            if not model_files:
                print(f"  모델{model_num} with_odds 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # 특성 파일 찾기
            feature_files = list(self.model_dir.glob(f"features{model_num}_with_odds_*.json"))
            if not feature_files:
                print(f"  모델{model_num} with_odds 특성 파일을 찾을 수 없습니다. 건너뜁니다.")
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
            
            print(f"  ✅ 모델{model_num} ({model_types.get(model_num, 'unknown')}) 로드 완료")
            print(f"     - 모델: {latest_model.name}")
            print(f"     - 특성 수: {len(feature_info['feature_names'])}")
        
        if not loaded_models:
            raise FileNotFoundError("로드할 수 있는 with_odds 모델이 없습니다.")
        
        self.models = loaded_models
        print(f"\n=== 총 {len(loaded_models)}개 [WITH_ODDS] 모델 로드 완료 ===")
        return loaded_models
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터에서 특성 추출 (배당 변수 포함!)"""
        # 날짜 기준으로 정렬
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # 기본 특성 선택 (기존과 동일)
        base_features = [
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
            'home_leader_points', 'away_leader_points',
            'home_leader_rebounds', 'away_leader_rebounds',
            'home_leader_assists', 'away_leader_assists',
            'home_overall_record_win_rate', 'away_overall_record_win_rate',
            'home_home_record_win_rate', 'away_home_record_win_rate',
            'home_road_record_win_rate', 'away_road_record_win_rate',
            'home_vs_away_win_rate',
            'home_recent_win_rate', 'away_recent_win_rate',
            'home_recent_avg_score', 'away_recent_avg_score',
            'home_recent_home_win_rate', 'away_recent_home_win_rate',
            'home_recent_away_win_rate', 'away_recent_away_win_rate',
            'home_rest_days', 'away_rest_days'
        ]
        
        X = data[base_features].copy()
        
        # 최근 트렌드 특성 복제
        recent_features = ['recent_win_rate', 'recent_avg_score', 'recent_home_win_rate', 'recent_away_win_rate']
        for col in recent_features:
            for team in ['home', 'away']:
                orig_col = f'{team}_{col}'
                new_col = f'{orig_col}_2'
                X[new_col] = X[orig_col]
        
        # ★★★ 배당 변수 추가 ★★★
        if 'home_odds_bucket' in data.columns and 'away_odds_bucket' in data.columns:
            X['home_odds_bucket'] = data['home_odds_bucket']
            X['away_odds_bucket'] = data['away_odds_bucket']
            print(f"  ✅ 배당 변수 포함: home_odds_bucket, away_odds_bucket")
        else:
            print(f"  ⚠️ 배당 변수가 데이터에 없습니다! 기본값(4) 사용")
            X['home_odds_bucket'] = 4
            X['away_odds_bucket'] = 4
        
        return X
    
    def predict_games(self, df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
        """앙상블 예측 수행"""
        X = self.prepare_features(df)
        
        if weights is None:
            weights = {model_info['type']: 1/len(self.models) for model_info in self.models}
        
        model_predictions = {}
        weighted_predictions = []
        
        total_weight = sum(weights.get(model_info['type'], 0) for model_info in self.models)
        if total_weight == 0:
            total_weight = 1
        
        for model_info in self.models:
            model = model_info['model']
            model_type = model_info['type']
            prob = model.predict_proba(X)[:, 1]
            
            model_predictions[model_type] = prob
            
            weight = weights.get(model_type, 0)
            if total_weight > 0:
                weighted_predictions.append(prob * (weight / total_weight))
        
        ensemble_probabilities = np.sum(weighted_predictions, axis=0)
        
        # 결과 DataFrame 생성
        results_df = df[['date', 'home_team_name', 'away_team_name']].copy()
        results_df['home_win_probability'] = ensemble_probabilities
        
        # 배당 정보 추가 (있으면)
        if 'home_odds_raw' in df.columns:
            results_df['home_odds'] = df['home_odds_raw'].values
            results_df['away_odds'] = df['away_odds_raw'].values
        
        # ★★★ 배당 버킷 정보 추가 ★★★
        if 'home_odds_bucket' in df.columns:
            results_df['home_odds_bucket'] = df['home_odds_bucket'].values
        if 'away_odds_bucket' in df.columns:
            results_df['away_odds_bucket'] = df['away_odds_bucket'].values
        
        # 각 모델의 개별 확률 추가
        for i in range(1, 9):
            col_name = f'model{i}_home_win_prob'
            results_df[col_name] = model_predictions.get(f'model{i}', np.zeros(len(df)))
        
        results_df['predicted_winner'] = np.where(
            ensemble_probabilities > 0.5,
            results_df['home_team_name'],
            results_df['away_team_name']
        )
        results_df['win_probability'] = np.where(
            ensemble_probabilities > 0.5,
            ensemble_probabilities,
            1 - ensemble_probabilities
        )
        
        # 날짜 형식 변환
        results_df['date'] = pd.to_datetime(results_df['date']).dt.strftime('%Y-%m-%d')
        
        # 결과 출력
        model_names = {
            'model1': 'LightGBM', 'model2': 'CatBoost', 'model3': 'XGBoost',
            'model4': 'LightGBM-DART', 'model5': 'CatBoost-Ordered',
            'model6': 'XGBoost-Hist', 'model7': 'RandomForest', 'model8': 'ExtraTrees'
        }
        
        print("\n=== [WITH_ODDS] 앙상블 예측 결과 ===")
        for _, row in results_df.iterrows():
            print(f"\n{row['date']} 경기:")
            print(f"  {row['home_team_name']} vs {row['away_team_name']}")
            if 'home_odds' in row and pd.notna(row['home_odds']):
                print(f"  배당: 홈 {row['home_odds']:+.0f} / 어웨이 {row['away_odds']:+.0f}")
            print(f"  예상 승자: {row['predicted_winner']}")
            print(f"  승리 확률: {row['win_probability']:.1%}")
            
            for model_info in self.models:
                model_type = model_info['type']
                model_name = model_names.get(model_type, model_type)
                col_name = f'{model_type}_home_win_prob'
                if col_name in row:
                    print(f"    - {model_name}: {row[col_name]:.1%}")
        
        return results_df
    
    def save_predictions(self, predictions: pd.DataFrame) -> Path:
        """예측 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "predictions"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"ensemble_predictions_{timestamp}_with_odds.json"
        
        predictions.to_json(output_path, orient='records', indent=2)
        
        print(f"\n=== [WITH_ODDS] 예측 결과 저장 완료 ===")
        print(f"저장 경로: {output_path}")
        
        return output_path
    
    def load_prediction_data(self) -> pd.DataFrame:
        """최신 배당 포함 예측용 데이터 로드"""
        data_dir = Path(__file__).parent / "data"
        
        # with_odds 데이터 우선, 없으면 일반 데이터
        pred_files = list(data_dir.glob("model_input_features_with_odds_*.json"))
        
        if not pred_files:
            print("⚠️ with_odds 예측 데이터가 없습니다. 일반 데이터 사용...")
            pred_files = list(data_dir.glob("model_input_features_*.json"))
        
        if not pred_files:
            raise FileNotFoundError("예측할 데이터 파일을 찾을 수 없습니다.")
        
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        print(f"\n예측 데이터 파일 로드: {latest_file.name}")
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)


def main():
    print("=" * 70)
    print("🏀 [WITH_ODDS] 배당 포함 앙상블 예측")
    print("=" * 70)
    
    # 예측기 초기화
    predictor = EnsemblePredictorWithOdds()
    
    # 모델 로드
    loaded_models = predictor.load_latest_models()
    print(f"\n총 {len(loaded_models)}개 [WITH_ODDS] 모델 로드 완료")
    
    # 예측 데이터 로드
    data = predictor.load_prediction_data()
    print(f"예측할 경기 수: {len(data)}")
    
    # 모델별 가중치 설정
    weights = {
        'model1': 1,    # LightGBM
        'model2': 1,    # CatBoost
        'model3': 1,    # XGBoost
        'model4': 1,    # LightGBM-DART
        'model5': 1,    # CatBoost-Ordered
        'model6': 1,    # XGBoost-Hist
        'model7': 1,    # RandomForest
        'model8': 1     # ExtraTrees
    }
    
    # 앙상블 예측 수행
    predictions = predictor.predict_games(data, weights=weights)
    
    # 예측 결과 저장
    output_path = predictor.save_predictions(predictions)
    
    # 예측 신뢰도 통계
    high_confidence = (predictions['win_probability'] >= 0.7).sum()
    medium_confidence = ((predictions['win_probability'] >= 0.6) & 
                       (predictions['win_probability'] < 0.7)).sum()
    low_confidence = (predictions['win_probability'] < 0.6).sum()
    
    print(f"\n=== 예측 신뢰도 분석 ===")
    print(f"높은 신뢰도 (70% 이상): {high_confidence}경기")
    print(f"중간 신뢰도 (60-70%): {medium_confidence}경기")
    print(f"낮은 신뢰도 (60% 미만): {low_confidence}경기")
    
    print("\n" + "=" * 70)
    print("✅ [WITH_ODDS] 배당 포함 앙상블 예측 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()

