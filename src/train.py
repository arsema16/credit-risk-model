"""
Model Training Module for Credit Risk Model
Task 5: Model training, hyperparameter tuning, and MLflow tracking
"""

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import logging
import os
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, hyperparameter tuning, and MLflow tracking"""

    def __init__(self, random_state=42, experiment_name="credit_risk_model"):
        self.random_state = random_state
        self.experiment_name = experiment_name

        # Define models and their hyperparameter grids
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }

        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set: {experiment_name}")

    def evaluate_model(self, model, X_test, y_test):
        """Calculate comprehensive evaluation metrics"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Confusion matrix values
        cm = confusion_matrix(y_test, y_pred)
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]

        return metrics

    def train_with_tuning(self, X_train, X_test, y_train, y_test, model_name, cv=5, n_iter=10):
        """Train a model with hyperparameter tuning"""
        logger.info(f"Training {model_name} with RandomizedSearchCV...")

        model_config = self.models[model_name]
        base_model = model_config['model']
        param_grid = model_config['param_grid']

        # Randomized Search for efficiency
        searcher = RandomizedSearchCV(
            base_model, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, n_iter=n_iter, random_state=self.random_state,
            verbose=0
        )

        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_tuned"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("tuning_method", "RandomizedSearchCV")
            mlflow.log_param("cv_folds", cv)
            mlflow.log_param("n_iterations", n_iter)

            # Fit the model
            searcher.fit(X_train, y_train)

            # Get best model
            best_model = searcher.best_estimator_
            best_params = searcher.best_params_

            # Log best parameters
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)

            # Evaluate
            metrics = self.evaluate_model(best_model, X_test, y_test)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log feature importance for tree-based models
            if hasattr(best_model, 'feature_importances_'):
                feature_imps = best_model.feature_importances_[:10]
                for i, imp in enumerate(feature_imps):
                    mlflow.log_metric(f"feature_importance_{i}", float(imp))

            # Log model
            y_pred_train = best_model.predict(X_train)
            signature = infer_signature(X_train, y_pred_train)
            mlflow.sklearn.log_model(
                best_model,
                model_name,
                signature=signature,
                registered_model_name=f"credit_risk_{model_name}"
            )

            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Metrics: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}")

            return best_model, metrics, best_params

    def train_all_models(self, X, y, test_size=0.2):
        """Train and compare all models"""

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Training set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")

        results = {}
        best_model_name = None
        best_score = 0

        for model_name in self.models.keys():
            separator = "=" * 50
            logger.info(f"\n{separator}")
            logger.info(f"Training {model_name}")
            logger.info(separator)

            try:
                model, metrics, best_params = self.train_with_tuning(
                    X_train, X_test, y_train, y_test, model_name
                )

                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'best_params': best_params
                }

                if metrics['roc_auc'] > best_score:
                    best_score = metrics['roc_auc']
                    best_model_name = model_name

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

        # Log comparison summary
        with mlflow.start_run(run_name="model_comparison"):
            for model_name, result in results.items():
                for metric, value in result['metrics'].items():
                    mlflow.log_metric(f"{model_name}_{metric}", value)

        # Display comparison
        separator = "=" * 50
        logger.info(f"\n{separator}")
        logger.info("MODEL COMPARISON RESULTS")
        logger.info(separator)

        comparison_df = pd.DataFrame({
            model_name: result['metrics']
            for model_name, result in results.items()
        }).T.round(4)

        logger.info(f"\n{comparison_df}")
        logger.info(f"\nBest model: {best_model_name} (ROC-AUC: {best_score:.4f})")

        return results, best_model_name, (X_train, X_test, y_train, y_test)


def save_model(model, model_path='models/best_model.pkl'):
    """Save the best model to disk"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


def save_feature_columns(X, feature_path='models/feature_columns.json'):
    """Save feature column names for inference"""
    import json
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    with open(feature_path, 'w') as f:
        json.dump(list(X.columns), f)
    logger.info(f"Feature columns saved to {feature_path}")


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load processed data
    logger.info("\n[Step 1] Loading processed data...")
    processed_df = pd.read_csv('data/processed/processed_data.csv')
    logger.info(f"Loaded data shape: {processed_df.shape}")

    # Prepare features and target
    logger.info("\n[Step 2] Preparing features and target...")
    exclude_cols = ['CustomerId', 'is_high_risk', 'Recency', 'Frequency', 'Monetary', 'Cluster']
    feature_cols = [col for col in processed_df.columns if col not in exclude_cols]
    X = processed_df[feature_cols]
    y = processed_df['is_high_risk']

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    logger.info(f"Features: {feature_cols}")

    # Train models
    logger.info("\n[Step 3] Training models with MLflow tracking...")
    trainer = ModelTrainer(random_state=42, experiment_name="credit_risk_bati_bank")
    results, best_model_name, data_splits = trainer.train_all_models(X, y)

    # Save best model
    logger.info("\n[Step 4] Saving best model...")
    best_model = results[best_model_name]['model']
    save_model(best_model, 'models/best_model.pkl')
    save_feature_columns(X, 'models/feature_columns.json')

    # Print final summary
    final_separator = "=" * 60
    print(f"\n{final_separator}")
    print("TRAINING COMPLETE")
    print(final_separator)
    print(f"Best Model: {best_model_name}")
    print(f"Best ROC-AUC: {results[best_model_name]['metrics']['roc_auc']:.4f}")
    print(f"Best F1 Score: {results[best_model_name]['metrics']['f1_score']:.4f}")
    print("\nModel saved to: models/best_model.pkl")
    print("Feature columns saved to: models/feature_columns.json")
    print("\nTo view MLflow UI, run: mlflow ui")
    print("Then navigate to: http://localhost:5000")


if __name__ == "__main__":
    main()
