# src/analysis/merge_predictions_totals.py
# 총점 예측과 배당 병합 - Over/Under 추천 생성

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class TotalsPredictionsMerger:
    """총점 예측과 배당 데이터 병합"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.predictions_dir = self.project_root / "src" / "predictions"
        self.odds_dir = self.project_root / "data" / "odds"
        self.output_dir = self.project_root / "src" / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # 팀명 매핑 (전체명 → 약칭)
        self.full_to_short = {
            'Atlanta Hawks': 'Hawks',
            'Boston Celtics': 'Celtics',
            'Brooklyn Nets': 'Nets',
            'Charlotte Hornets': 'Hornets',
            'Chicago Bulls': 'Bulls',
            'Cleveland Cavaliers': 'Cavaliers',
            'Dallas Mavericks': 'Mavericks',
            'Denver Nuggets': 'Nuggets',
            'Detroit Pistons': 'Pistons',
            'Golden State Warriors': 'Warriors',
            'Houston Rockets': 'Rockets',
            'Indiana Pacers': 'Pacers',
            'Los Angeles Clippers': 'Clippers',
            'Los Angeles Lakers': 'Lakers',
            'Memphis Grizzlies': 'Grizzlies',
            'Miami Heat': 'Heat',
            'Milwaukee Bucks': 'Bucks',
            'Minnesota Timberwolves': 'Timberwolves',
            'New Orleans Pelicans': 'Pelicans',
            'New York Knicks': 'Knicks',
            'Oklahoma City Thunder': 'Thunder',
            'Orlando Magic': 'Magic',
            'Philadelphia 76ers': '76ers',
            'Phoenix Suns': 'Suns',
            'Portland Trail Blazers': 'Trail Blazers',
            'Sacramento Kings': 'Kings',
            'San Antonio Spurs': 'Spurs',
            'Toronto Raptors': 'Raptors',
            'Utah Jazz': 'Jazz',
            'Washington Wizards': 'Wizards'
        }
    
    def load_latest_predictions(self) -> List[Dict]:
        """최신 totals 예측 파일 로드"""
        pred_files = list(self.predictions_dir.glob("totals_predictions_*.json"))
        
        if not pred_files:
            raise FileNotFoundError(f"Totals 예측 파일이 없습니다: {self.predictions_dir}")
        
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        print(f"📊 예측 파일 로드: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_latest_odds(self) -> List[Dict]:
        """최신 totals 배당 파일 로드"""
        odds_files = list(self.odds_dir.glob("processed_nba_totals_*.json"))
        
        if not odds_files:
            raise FileNotFoundError(f"Totals 배당 파일이 없습니다: {self.odds_dir}")
        
        latest_file = max(odds_files, key=lambda x: x.stat().st_mtime)
        print(f"💰 배당 파일 로드: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_short_name(self, full_name: str) -> str:
        """전체 팀명 → 약칭"""
        return self.full_to_short.get(full_name, full_name.split()[-1])
    
    def merge(self) -> pd.DataFrame:
        """예측과 배당 병합"""
        predictions = self.load_latest_predictions()
        odds = self.load_latest_odds()
        
        print(f"\n📋 예측: {len(predictions)}경기, 배당: {len(odds)}경기")
        
        # 배당 데이터에 약칭 추가
        for o in odds:
            o['home_short'] = self._get_short_name(o['home_team'])
            o['away_short'] = self._get_short_name(o['away_team'])
        
        # 병합
        merged = []
        matched = 0
        
        for pred in predictions:
            home = pred['home_team_name']
            away = pred['away_team_name']
            
            # 매칭되는 배당 찾기
            matching_odds = None
            for o in odds:
                if o['home_short'] == home and o['away_short'] == away:
                    matching_odds = o
                    break
            
            if matching_odds:
                matched += 1
                predicted_total = pred['predicted_total']
                total_line = matching_odds['total_line']
                diff = predicted_total - total_line
                
                # Over/Under 추천
                if diff > 0:
                    recommendation = 'OVER'
                    bet_odds = matching_odds['over_odds']
                else:
                    recommendation = 'UNDER'
                    bet_odds = matching_odds['under_odds']
                
                # 신뢰도 (예측과 라인의 차이 기반)
                confidence = min(abs(diff) / 5.0, 1.0)  # 5점 차이면 100% 신뢰
                
                merged.append({
                    'date': pred['date'],
                    'home_team': home,
                    'away_team': away,
                    'predicted_total': round(predicted_total, 1),
                    'total_line': total_line,
                    'difference': round(diff, 1),
                    'recommendation': recommendation,
                    'bet_odds': bet_odds,
                    'confidence': round(confidence, 2),
                    'over_odds': matching_odds['over_odds'],
                    'under_odds': matching_odds['under_odds'],
                    # 개별 모델 예측값
                    'model1_pred': round(pred.get('model1_predicted_total', 0), 1),
                    'model2_pred': round(pred.get('model2_predicted_total', 0), 1),
                    'model3_pred': round(pred.get('model3_predicted_total', 0), 1),
                    'model4_pred': round(pred.get('model4_predicted_total', 0), 1),
                    'model5_pred': round(pred.get('model5_predicted_total', 0), 1),
                    'model6_pred': round(pred.get('model6_predicted_total', 0), 1),
                    'model7_pred': round(pred.get('model7_predicted_total', 0), 1),
                    'model8_pred': round(pred.get('model8_predicted_total', 0), 1),
                })
            else:
                print(f"  ⚠️ 배당 없음: {home} vs {away}")
        
        print(f"\n✅ {matched}/{len(predictions)} 경기 매칭됨")
        
        if not merged:
            print("❌ 매칭된 경기가 없습니다.")
            return pd.DataFrame()
        
        df = pd.DataFrame(merged)
        return df
    
    def save_merged(self, df: pd.DataFrame) -> Path:
        """병합 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"merged_totals_{timestamp}.json"
        
        df.to_json(output_path, orient='records', indent=2)
        print(f"\n💾 저장: {output_path}")
        
        return output_path
    
    def print_recommendations(self, df: pd.DataFrame):
        """추천 결과 출력"""
        print("\n" + "="*70)
        print("🎯 TOTALS BETTING RECOMMENDATIONS")
        print("="*70)
        
        for _, row in df.iterrows():
            emoji = "🔼" if row['recommendation'] == 'OVER' else "🔽"
            conf_bar = "█" * int(row['confidence'] * 10) + "░" * (10 - int(row['confidence'] * 10))
            
            print(f"\n📅 {row['date']} | {row['home_team']} vs {row['away_team']}")
            print(f"   예측: {row['predicted_total']}점 | Line: {row['total_line']}")
            print(f"   {emoji} 추천: {row['recommendation']} ({row['bet_odds']:+d})")
            print(f"   차이: {row['difference']:+.1f}점 | 신뢰도: [{conf_bar}] {row['confidence']*100:.0f}%")


def main():
    """실행"""
    print("\n" + "="*60)
    print("🏀 Totals 예측 + 배당 병합")
    print("="*60)
    
    merger = TotalsPredictionsMerger()
    
    try:
        df = merger.merge()
        
        if not df.empty:
            merger.print_recommendations(df)
            merger.save_merged(df)
            
            # 요약 통계
            print("\n" + "="*60)
            print("📊 요약")
            print("="*60)
            over_count = (df['recommendation'] == 'OVER').sum()
            under_count = (df['recommendation'] == 'UNDER').sum()
            avg_conf = df['confidence'].mean()
            
            print(f"OVER 추천: {over_count}경기")
            print(f"UNDER 추천: {under_count}경기")
            print(f"평균 신뢰도: {avg_conf*100:.1f}%")
            
    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {e}")
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

