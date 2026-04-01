"""
NBA Active/Shadow 모델 전환 및 학습 스크립트

주간 실행 권장:
1. Shadow 모델을 Active로 승격
2. 새로운 Shadow 모델 학습
3. 이전 Active 모델 백업/정리

사용법:
    python nba_switch_and_train.py --action switch      # Shadow → Active 전환
    python nba_switch_and_train.py --action train       # 새 Shadow 모델 학습
    python nba_switch_and_train.py --action full        # 전환 + 학습 (전체 사이클)
    python nba_switch_and_train.py --action status      # 현재 모델 상태 확인
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json
import logging


class NBAModelManager:
    """NBA Active/Shadow 모델 관리자"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.model_dir = self.project_root / 'src' / 'models' / 'saved_models'
        self.backup_dir = self.project_root / 'src' / 'models' / 'backup_models'
        self.src_dir = self.project_root / 'src'
        
        # 예측/분석 파일 디렉토리
        self.predictions_dir = self.project_root / 'src' / 'predictions'
        self.analysis_dir = self.project_root / 'src' / 'analysis'
        self.data_backup_dir = self.project_root / 'src' / 'analysis' / 'backup_data'
        
        # 디렉토리 생성
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.data_backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        self.setup_logging()
        
        # 모델 타입 정보
        self.model_types = {
            1: 'LightGBM',
            2: 'CatBoost', 
            3: 'XGBoost',
            4: 'LightGBM-DART',
            5: 'CatBoost-Ordered',
            6: 'XGBoost-Hist',
            7: 'RandomForest',
            8: 'ExtraTrees'
        }
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'model_switch_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('NBAModelManager')
    
    def get_model_status(self) -> dict:
        """현재 모델 상태 확인"""
        status = {
            'active': {},
            'shadow': {},
            'fixed': {},
            'backup': []
        }
        
        # Active 모델 확인
        for num in range(1, 9):
            model_file = self.model_dir / f'betting_model{num}_active.joblib'
            feature_file = self.model_dir / f'features{num}_active.json'
            
            if model_file.exists():
                mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
                status['active'][num] = {
                    'model': model_file.name,
                    'features': feature_file.name if feature_file.exists() else None,
                    'last_modified': mtime.strftime('%Y-%m-%d %H:%M:%S'),
                    'algorithm': self.model_types.get(num, 'Unknown')
                }
        
        # Shadow 모델 확인
        for num in range(1, 9):
            model_file = self.model_dir / f'betting_model{num}_shadow.joblib'
            feature_file = self.model_dir / f'features{num}_shadow.json'
            
            if model_file.exists():
                mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
                status['shadow'][num] = {
                    'model': model_file.name,
                    'features': feature_file.name if feature_file.exists() else None,
                    'last_modified': mtime.strftime('%Y-%m-%d %H:%M:%S'),
                    'algorithm': self.model_types.get(num, 'Unknown')
                }
        
        # Fixed 모델 확인 (하위 호환)
        for num in range(1, 9):
            model_file = self.model_dir / f'betting_model{num}_fixed.joblib'
            feature_file = self.model_dir / f'features{num}_fixed.json'
            
            if model_file.exists():
                mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
                status['fixed'][num] = {
                    'model': model_file.name,
                    'features': feature_file.name if feature_file.exists() else None,
                    'last_modified': mtime.strftime('%Y-%m-%d %H:%M:%S'),
                    'algorithm': self.model_types.get(num, 'Unknown')
                }
        
        # 백업 폴더 확인
        if self.backup_dir.exists():
            for backup_folder in sorted(self.backup_dir.iterdir(), reverse=True):
                if backup_folder.is_dir():
                    status['backup'].append(backup_folder.name)
        
        return status
    
    def display_status(self):
        """현재 모델 상태 출력"""
        status = self.get_model_status()
        
        print("\n" + "="*70)
        print("🏀 NBA MODEL STATUS")
        print("="*70)
        
        # Active 모델
        print("\n📍 ACTIVE MODELS (현재 운영 중)")
        print("-"*50)
        if status['active']:
            for num, info in sorted(status['active'].items()):
                print(f"  Model {num} ({info['algorithm']})")
                print(f"    파일: {info['model']}")
                print(f"    수정: {info['last_modified']}")
        else:
            print("  ⚠️  Active 모델이 없습니다.")
        
        # Shadow 모델
        print("\n🌙 SHADOW MODELS (테스트 대기 중)")
        print("-"*50)
        if status['shadow']:
            for num, info in sorted(status['shadow'].items()):
                print(f"  Model {num} ({info['algorithm']})")
                print(f"    파일: {info['model']}")
                print(f"    수정: {info['last_modified']}")
        else:
            print("  ℹ️  Shadow 모델이 없습니다.")
        
        # Fixed 모델 (하위 호환)
        if status['fixed']:
            print("\n📌 FIXED MODELS (고정/레거시)")
            print("-"*50)
            for num, info in sorted(status['fixed'].items()):
                print(f"  Model {num} ({info['algorithm']})")
                print(f"    파일: {info['model']}")
                print(f"    수정: {info['last_modified']}")
        
        # 백업
        print("\n📦 BACKUPS")
        print("-"*50)
        if status['backup']:
            for backup in status['backup'][:5]:  # 최근 5개만
                print(f"  {backup}")
            if len(status['backup']) > 5:
                print(f"  ... 외 {len(status['backup'])-5}개")
        else:
            print("  ℹ️  백업이 없습니다.")
        
        print("\n" + "="*70)
    
    def backup_active_models(self) -> Path:
        """Active 모델 백업"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_folder = self.backup_dir / f'active_backup_{timestamp}'
        backup_folder.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Active 모델 백업 시작: {backup_folder}")
        
        backed_up = 0
        for num in range(1, 9):
            model_file = self.model_dir / f'betting_model{num}_active.joblib'
            feature_file = self.model_dir / f'features{num}_active.json'
            
            if model_file.exists():
                shutil.copy2(model_file, backup_folder / model_file.name)
                backed_up += 1
                self.logger.info(f"  백업: {model_file.name}")
            
            if feature_file.exists():
                shutil.copy2(feature_file, backup_folder / feature_file.name)
        
        self.logger.info(f"백업 완료: {backed_up}개 모델")
        return backup_folder
    
    def switch_shadow_to_active(self) -> bool:
        """Shadow 모델을 Active로 전환"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🔄 Shadow → Active 전환 시작")
        self.logger.info("="*50)
        
        # Shadow 모델 존재 확인
        shadow_count = sum(1 for num in range(1, 9) 
                         if (self.model_dir / f'betting_model{num}_shadow.joblib').exists())
        
        if shadow_count == 0:
            self.logger.error("❌ Shadow 모델이 없습니다. 먼저 학습을 실행하세요.")
            return False
        
        # 1. 기존 Active 모델 백업
        status = self.get_model_status()
        if status['active']:
            self.logger.info("\n1️⃣ 기존 Active 모델 백업...")
            self.backup_active_models()
        
        # 2. 기존 Active 모델 삭제
        self.logger.info("\n2️⃣ 기존 Active 모델 제거...")
        for num in range(1, 9):
            model_file = self.model_dir / f'betting_model{num}_active.joblib'
            feature_file = self.model_dir / f'features{num}_active.json'
            
            if model_file.exists():
                model_file.unlink()
                self.logger.info(f"  삭제: {model_file.name}")
            if feature_file.exists():
                feature_file.unlink()
        
        # 3. Shadow → Active 이름 변경
        self.logger.info("\n3️⃣ Shadow → Active 전환...")
        switched = 0
        for num in range(1, 9):
            shadow_model = self.model_dir / f'betting_model{num}_shadow.joblib'
            shadow_feature = self.model_dir / f'features{num}_shadow.json'
            active_model = self.model_dir / f'betting_model{num}_active.joblib'
            active_feature = self.model_dir / f'features{num}_active.json'
            
            if shadow_model.exists():
                shadow_model.rename(active_model)
                self.logger.info(f"  전환: Model {num} ({self.model_types.get(num, 'Unknown')})")
                switched += 1
            
            if shadow_feature.exists():
                shadow_feature.rename(active_feature)
        
        self.logger.info(f"\n✅ 모델 전환 완료: {switched}개 모델이 Active로 승격됨")
        
        # 4. 예측/분석 파일 전환
        self.logger.info("\n4️⃣ 예측/분석 파일 전환...")
        self.switch_prediction_files()
        
        return True
    
    def switch_prediction_files(self):
        """예측/분석 파일 전환 (Shadow → Active, 기존 Active → 백업)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 백업 폴더 생성
        backup_folder = self.data_backup_dir / f'active_data_backup_{timestamp}'
        backup_folder.mkdir(parents=True, exist_ok=True)
        
        # === predictions 폴더 처리 ===
        self.logger.info("\n  [predictions 폴더]")
        
        # 1. 기존 _active.json + 태그없음 → 백업
        active_pred_files = list(self.predictions_dir.glob('ensemble_predictions_*_active.json'))
        legacy_pred_files = [f for f in self.predictions_dir.glob('ensemble_predictions_*.json')
                            if not f.stem.endswith('_active') and not f.stem.endswith('_shadow')
                            and 'with_odds' not in f.name]
        
        backed_up_pred = 0
        for f in active_pred_files + legacy_pred_files:
            shutil.move(str(f), str(backup_folder / f.name))
            backed_up_pred += 1
        self.logger.info(f"    백업: {backed_up_pred}개 파일 → {backup_folder.name}")
        
        # 2. _shadow.json → _active.json으로 이름 변경
        shadow_pred_files = list(self.predictions_dir.glob('ensemble_predictions_*_shadow.json'))
        renamed_pred = 0
        for f in shadow_pred_files:
            new_name = f.name.replace('_shadow.json', '_active.json')
            f.rename(self.predictions_dir / new_name)
            renamed_pred += 1
        self.logger.info(f"    전환: {renamed_pred}개 파일 (_shadow → _active)")
        
        # === analysis 폴더 처리 ===
        self.logger.info("\n  [analysis 폴더]")
        
        # 1. 기존 _active.json + 태그없음 → 백업
        active_analysis_files = list(self.analysis_dir.glob('merged_predictions_odds_*_active.json'))
        legacy_analysis_files = [f for f in self.analysis_dir.glob('merged_predictions_odds_*.json')
                                if not f.stem.endswith('_active') and not f.stem.endswith('_shadow')
                                and 'with_odds' not in f.name]
        
        backed_up_analysis = 0
        for f in active_analysis_files + legacy_analysis_files:
            shutil.move(str(f), str(backup_folder / f.name))
            backed_up_analysis += 1
        self.logger.info(f"    백업: {backed_up_analysis}개 파일 → {backup_folder.name}")
        
        # 2. _shadow.json → _active.json으로 이름 변경
        shadow_analysis_files = list(self.analysis_dir.glob('merged_predictions_odds_*_shadow.json'))
        renamed_analysis = 0
        for f in shadow_analysis_files:
            new_name = f.name.replace('_shadow.json', '_active.json')
            f.rename(self.analysis_dir / new_name)
            renamed_analysis += 1
        self.logger.info(f"    전환: {renamed_analysis}개 파일 (_shadow → _active)")
        
        self.logger.info(f"\n✅ 예측/분석 파일 전환 완료!")
        self.logger.info(f"   - 백업된 파일: {backed_up_pred + backed_up_analysis}개")
        self.logger.info(f"   - 전환된 파일: {renamed_pred + renamed_analysis}개")
        self.logger.info(f"   - 백업 위치: {backup_folder}")
    
    def train_shadow_models(self) -> bool:
        """새로운 Shadow 모델 학습"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🎓 새 Shadow 모델 학습 시작")
        self.logger.info("="*50)
        
        # 환경 설정
        env = os.environ.copy()
        env['PROJECT_ROOT'] = str(self.project_root)
        env['PYTHONPATH'] = str(self.project_root)
        
        trained = 0
        failed = []
        
        for num in range(1, 9):
            model_script = self.src_dir / 'models' / f'model{num}.py'
            
            if not model_script.exists():
                self.logger.warning(f"⚠️  Model {num} 스크립트 없음: {model_script}")
                continue
            
            self.logger.info(f"\n📊 Model {num} ({self.model_types.get(num, 'Unknown')}) 학습 중...")
            
            try:
                result = subprocess.run(
                    [sys.executable, str(model_script)],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,
                    cwd=str(self.project_root),
                    timeout=600  # 10분 타임아웃
                )
                
                # 학습된 모델을 shadow로 이름 변경
                # 최신 모델 파일 찾기 (타임스탬프 버전)
                model_files = list(self.model_dir.glob(f"betting_model{num}_2*.joblib"))
                if model_files:
                    latest = max(model_files, key=lambda x: x.stat().st_mtime)
                    shadow_model = self.model_dir / f'betting_model{num}_shadow.joblib'
                    
                    # 기존 shadow 있으면 삭제
                    if shadow_model.exists():
                        shadow_model.unlink()
                    
                    latest.rename(shadow_model)
                    self.logger.info(f"  모델 저장: {shadow_model.name}")
                
                # feature 파일도 처리
                feature_files = list(self.model_dir.glob(f"features{num}_2*.json"))
                if feature_files:
                    latest = max(feature_files, key=lambda x: x.stat().st_mtime)
                    shadow_feature = self.model_dir / f'features{num}_shadow.json'
                    
                    if shadow_feature.exists():
                        shadow_feature.unlink()
                    
                    latest.rename(shadow_feature)
                    self.logger.info(f"  피처 저장: {shadow_feature.name}")
                
                trained += 1
                self.logger.info(f"  ✅ Model {num} 학습 완료")
                
            except subprocess.TimeoutExpired:
                self.logger.error(f"  ❌ Model {num} 타임아웃 (10분 초과)")
                failed.append(num)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"  ❌ Model {num} 학습 실패: {e.stderr[:200] if e.stderr else 'Unknown error'}")
                failed.append(num)
            except Exception as e:
                self.logger.error(f"  ❌ Model {num} 오류: {str(e)}")
                failed.append(num)
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"🎓 학습 결과: 성공 {trained}개, 실패 {len(failed)}개")
        if failed:
            self.logger.info(f"   실패한 모델: {failed}")
        self.logger.info("="*50)
        
        return trained > 0
    
    def migrate_fixed_to_active(self) -> bool:
        """Fixed 모델을 Active로 마이그레이션 (최초 전환용)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🔄 Fixed → Active 마이그레이션")
        self.logger.info("="*50)
        
        migrated = 0
        for num in range(1, 9):
            fixed_model = self.model_dir / f'betting_model{num}_fixed.joblib'
            fixed_feature = self.model_dir / f'features{num}_fixed.json'
            active_model = self.model_dir / f'betting_model{num}_active.joblib'
            active_feature = self.model_dir / f'features{num}_active.json'
            
            # Fixed 존재하고 Active 없으면 복사
            if fixed_model.exists() and not active_model.exists():
                shutil.copy2(fixed_model, active_model)
                self.logger.info(f"  복사: {fixed_model.name} → {active_model.name}")
                migrated += 1
            
            if fixed_feature.exists() and not active_feature.exists():
                shutil.copy2(fixed_feature, active_feature)
        
        self.logger.info(f"\n✅ 마이그레이션 완료: {migrated}개 모델")
        return migrated > 0
    
    def cleanup_old_backups(self, keep_count: int = 5):
        """오래된 백업 정리"""
        if not self.backup_dir.exists():
            return
        
        backups = sorted(
            [d for d in self.backup_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if len(backups) > keep_count:
            for old_backup in backups[keep_count:]:
                self.logger.info(f"오래된 백업 삭제: {old_backup.name}")
                shutil.rmtree(old_backup)


def is_jupyter():
    """Jupyter 환경인지 확인"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


def parse_args():
    """커맨드라인 인자 파싱"""
    # Jupyter 환경에서는 기본값 사용 (전환 + 학습)
    if is_jupyter():
        class Args:
            action = 'full'
            keep_backups = 5
        return Args()
    
    parser = argparse.ArgumentParser(
        description='NBA Active/Shadow 모델 관리',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python nba_switch_and_train.py                       # 전환 + 학습 (기본값)
  python nba_switch_and_train.py --action full         # 전환 + 학습 (전체 사이클)
  python nba_switch_and_train.py --action status       # 현재 상태 확인
  python nba_switch_and_train.py --action migrate      # Fixed → Active 마이그레이션
  python nba_switch_and_train.py --action train        # 새 Shadow 모델 학습만
  python nba_switch_and_train.py --action switch       # Shadow → Active 전환만
        """
    )
    parser.add_argument(
        '--action', '-a',
        type=str,
        choices=['status', 'migrate', 'train', 'switch', 'full'],
        default='full',
        help='실행할 작업 (기본: full = 전환 + 학습)'
    )
    parser.add_argument(
        '--keep-backups', '-k',
        type=int,
        default=5,
        help='유지할 백업 개수 (기본: 5)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manager = NBAModelManager()
    
    try:
        if args.action == 'status':
            manager.display_status()
            
        elif args.action == 'migrate':
            # Fixed → Active 마이그레이션 (최초 한 번)
            manager.migrate_fixed_to_active()
            manager.display_status()
            
        elif args.action == 'train':
            # 새 Shadow 모델 학습
            manager.train_shadow_models()
            manager.display_status()
            
        elif args.action == 'switch':
            # Shadow → Active 전환
            if manager.switch_shadow_to_active():
                manager.cleanup_old_backups(args.keep_backups)
            manager.display_status()
            
        elif args.action == 'full':
            # 전체 사이클: 전환 + 학습
            print("\n🔄 전체 사이클 시작: Shadow → Active 전환 후 새 Shadow 학습")
            
            # 1. 전환
            if manager.switch_shadow_to_active():
                manager.cleanup_old_backups(args.keep_backups)
                
                # 2. 새 Shadow 학습
                manager.train_shadow_models()
            else:
                print("\n⚠️  전환 실패. Shadow 모델이 없으면 먼저 train을 실행하세요.")
            
            manager.display_status()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()

