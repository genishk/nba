from models.ensemble_predictor import EnsemblePredictor
from pathlib import Path
import json

def main():
    try:
        # 예측기 초기화
        predictor = EnsemblePredictor()
        
        # 모든 모델 로드
        loaded_models = predictor.load_latest_models()
        print(f"\n총 {len(loaded_models)}개 모델 로드 완료")
        
        # 예측 데이터 로드
        data = predictor.load_prediction_data()
        print(f"예측할 경기 수: {len(data)}")
        
        # 특성 준비 (이제 prepare_features는 predict_games 내부에서 사용됨)
        
        
        # 모델별 가중치 설정
        
        # weights = None
        weights = {
            'model1': 0.3,  # LightGBM (더 안정적인 예측을 보이는 모델)
            'model2': 0.3,  # CatBoost
            'model3': 0.4   # XGBoost
        }
        
        # 앙상블 예측 수행
        try:
            predictions = predictor.predict_games(data, weights=weights)
            
            # 예측 결과 저장
            output_path = predictor.save_predictions(predictions)
            
            # 예측 신뢰도 통계
            high_confidence = (predictions['win_probability'] >= 0.7).sum()
            medium_confidence = ((predictions['win_probability'] >= 0.6) & 
                               (predictions['win_probability'] < 0.7)).sum()
            low_confidence = (predictions['win_probability'] < 0.6).sum()
            
            print("\n=== 예측 신뢰도 분석 ===")
            print(f"높은 신뢰도 (70% 이상): {high_confidence}경기")
            print(f"중간 신뢰도 (60-70%): {medium_confidence}경기")
            print(f"낮은 신뢰도 (60% 미만): {low_confidence}경기")
            
        except Exception as e:
            print(f"\n예측 중 오류 발생: {str(e)}")
            raise
            
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 