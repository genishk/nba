# src/predict_ensemble_totals.py
# NBA 경기 총점 예측 실행 스크립트

from models.ensemble_predictor_totals import EnsemblePredictorTotals
from pathlib import Path
import json


def main():
    """총점 예측 실행"""
    print("\n" + "="*60)
    print("🏀 NBA 총점(Total Score) 예측 시작")
    print("="*60)
    
    try:
        # 예측기 초기화
        predictor = EnsemblePredictorTotals()
        
        # 모델 로드
        loaded_models = predictor.load_latest_models()
        print(f"\n총 {len(loaded_models)}개 Totals 모델 로드 완료")
        
        # 예측 데이터 로드
        data = predictor.load_prediction_data()
        print(f"예측할 경기 수: {len(data)}")
        
        # 모델별 가중치 설정 (총합이 1이 되도록 자동 정규화됨)
        # 성능 좋은 모델에 더 높은 가중치 부여 가능
        weights = {
            'model1': 0,    # LightGBM (MAE: 3.16)
            'model2': 0,    # CatBoost (MAE: 8.03)
            'model3': 1,    # XGBoost (MAE: 3.04) - 최고
            'model4': 1,    # LightGBM-GBDT (MAE: 5.92)
            'model5': 0,    # CatBoost-Ordered (MAE: 0.09 - 과적합 가능)
            'model6': 0,    # XGBoost-Hist (MAE: 0.10 - 과적합 가능)
            'model7': 0,    # RandomForest (MAE: 11.75)
            'model8': 0     # ExtraTrees (MAE: 9.65)
        }
        
        # 앙상블 예측 수행
        predictions = predictor.predict_games(data, weights=weights)
        
        # 예측 결과 저장
        output_path = predictor.save_predictions(predictions)
        
        # 예측 통계 출력
        avg_total = predictions['predicted_total'].mean()
        min_total = predictions['predicted_total'].min()
        max_total = predictions['predicted_total'].max()
        
        print("\n" + "="*60)
        print("📊 예측 통계")
        print("="*60)
        print(f"평균 예측 총점: {avg_total:.1f}점")
        print(f"최저 예측 총점: {min_total:.1f}점")
        print(f"최고 예측 총점: {max_total:.1f}점")
        
        # 모델간 예측 편차 분석
        print("\n📈 모델간 예측 편차:")
        for _, row in predictions.iterrows():
            model_preds = []
            for i in range(1, 9):
                col = f'model{i}_predicted_total'
                if col in row and row[col] > 0:
                    model_preds.append(row[col])
            
            if model_preds:
                std_dev = np.std(model_preds)
                print(f"  {row['home_team_name']} vs {row['away_team_name']}: "
                      f"예측 {row['predicted_total']:.1f}점 (표준편차: {std_dev:.1f}점)")
        
        print("\n" + "="*60)
        print("✅ 총점 예측 완료!")
        print(f"결과 저장: {output_path}")
        print("="*60)
        
        return predictor, predictions, output_path
            
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# numpy import 추가 (통계 계산용)
import numpy as np


if __name__ == "__main__":
    main()

