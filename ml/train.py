"""
ML model training pipeline for flood risk prediction.
Supports RandomForest and GradientBoosting classifiers with optional calibration.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib
import logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, brier_score_loss
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    'water_mean_5', 'water_slope_5', 'water_max_10',
    'rain_sum_20', 'rain_mean_10', 'soil_mean_10',
    'consensus', 'health'
]

TARGET_COLUMN = 'flood_in_next_T'


class ModelTrainer:
    """
    Trains and evaluates ML models for flood risk prediction.
    """
    
    def __init__(self, model_type: str = 'rf', calibrate: bool = True,
                 random_state: int = 42):
        self.model_type = model_type
        self.calibrate = calibrate
        self.random_state = random_state
        self.model = None
        self.feature_importance = {}
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data into train/test sets (episode-based to prevent leakage)."""
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Missing target column: {TARGET_COLUMN}")
        
        # Episode-based split: same episode must stay in same set (no leakage)
        if 'episode_id' in df.columns:
            episodes = df['episode_id'].unique()
            np.random.seed(self.random_state)
            np.random.shuffle(episodes)
            
            n_test = int(len(episodes) * 0.2)
            test_episodes = episodes[:n_test]
            train_episodes = episodes[n_test:]
            
            train_df = df[df['episode_id'].isin(train_episodes)]
            test_df = df[df['episode_id'].isin(test_episodes)]
            
            logger.info(f"Split by episodes: {len(train_episodes)} train episodes, {len(test_episodes)} test episodes")
        else:
            # Fallback: stratified split (no episode info)
            logger.warning("No episode_id column found, using standard stratified split")
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=self.random_state, stratify=df[TARGET_COLUMN]
            )
        
        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[TARGET_COLUMN]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"Positive rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return (X_train, y_train), (X_test, y_test)
    
    def create_model(self):
        """Create base classifier."""
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        logger.info(f"Training {self.model_type.upper()} model...")
        
        base_model = self.create_model()
        
        if self.calibrate:
            logger.info("Applying probability calibration (isotonic)...")
            self.model = CalibratedClassifierCV(
                base_model, method='isotonic', cv=5
            )
        else:
            self.model = base_model
        
        self.model.fit(X_train, y_train)
        
        if hasattr(self.model, 'calibrated_classifiers_'):
            importances = np.mean([
                est.estimator.feature_importances_ 
                for est in self.model.calibrated_classifiers_
            ], axis=0)
        elif hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            importances = np.zeros(len(FEATURE_COLUMNS))
        
        self.feature_importance = dict(zip(FEATURE_COLUMNS, importances))
        
        logger.info("Training complete.")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model on test set."""
        y_pred = self.model.predict(X_test)
        
        # Handle edge case: only one class in training data
        y_prob_all = self.model.predict_proba(X_test)
        if y_prob_all.shape[1] == 1:
            # Only one class - all predictions are same class
            logger.warning("Model only learned one class (insufficient data variation)")
            y_prob = np.zeros(len(y_pred)) if y_test.sum() == 0 else np.ones(len(y_pred))
            brier = 1.0  # Worst possible score
        else:
            y_prob = y_prob_all[:, 1]
            # Brier score measures calibration quality (lower is better)
            brier = brier_score_loss(y_test, y_prob)
        
        metrics = {
            'auc_roc': float(roc_auc_score(y_test, y_prob)),
            'f1': float(f1_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'accuracy': float((y_pred == y_test).mean()),
            'brier_score': float(brier),  # Calibration quality metric
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'calibrated': self.calibrate
        }
        
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Brier Score: {metrics['brier_score']:.4f} (lower is better)")
        logger.info("\nFeature Importance:")
        for feat, imp in sorted(self.feature_importance.items(), 
                                key=lambda x: -x[1]):
            logger.info(f"  {feat}: {imp:.4f}")
        logger.info("="*50)
        
        return metrics
    
    def save(self, model_path: str, report_path: Optional[str] = None,
             metrics: Optional[Dict] = None) -> None:
        """Save model and training report."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        if report_path and metrics:
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved report to {report_path}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """Perform cross-validation."""
        base_model = self.create_model()
        
        scores = cross_val_score(base_model, X, y, cv=cv, scoring='roc_auc')
        
        result = {
            'cv_auc_mean': float(scores.mean()),
            'cv_auc_std': float(scores.std()),
            'cv_scores': scores.tolist()
        }
        
        logger.info(f"Cross-validation AUC: {result['cv_auc_mean']:.4f} "
                   f"(+/- {result['cv_auc_std']:.4f})")
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Train flood risk ML model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data (parquet)')
    parser.add_argument('--model', type=str, default='rf',
                        choices=['rf', 'gb'],
                        help='Model type: rf (RandomForest) or gb (GradientBoosting)')
    parser.add_argument('--out', type=str, default='outputs/models/risk_model.pkl',
                        help='Output path for trained model')
    parser.add_argument('--report', type=str, default='outputs/models/train_report.json',
                        help='Output path for training report')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='Disable probability calibration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(
        model_type=args.model,
        calibrate=not args.no_calibrate,
        random_state=args.seed
    )
    
    (X_train, y_train), (X_test, y_test) = trainer.load_data(args.data)
    
    trainer.train(X_train, y_train)
    
    metrics = trainer.evaluate(X_test, y_test)
    
    cv_results = trainer.cross_validate(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test])
    )
    metrics.update(cv_results)
    
    trainer.save(args.out, args.report, metrics)


if __name__ == '__main__':
    main()
