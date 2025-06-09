import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import jieba
import jieba.analyse
from snownlp import SnowNLP
import re
import os
import json
from datetime import datetime, timedelta
from collections import Counter
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import xgboost as xgb

warnings.filterwarnings('ignore')

# Log Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Encoder for JSON serialization of datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


# Advanced User Behavior Analysis Class
class AdvancedUserAnalysis:
    """Advanced User Behavior Analytics Module"""

    def __init__(self, db_config: Dict[str, str]):
        """
        Initializes the advanced user analyzer

        Args:
            db_config: Database configuration dictionary, including host, user, password, database, etc.
        """
        self.db_config = db_config
        self.connection = None
        self.cursor = None

        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), 'Advanced user analysis results')
        os.makedirs(self.output_dir, exist_ok=True)

        # Model storage
        self.cluster_model = None
        self.prediction_models = {}
        self.scalers = {}

    def connect_to_database(self) -> bool:
        """Connects to the MySQL database"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to the MySQL database")
            return True
        except Error as e:
            logger.error(f"Failed to connect to the database: {e}")
            return False

    def load_user_data(self) -> pd.DataFrame:
        """Loads user behavior data from the database"""
        try:
            # Check if user behavior analysis table exists
            self.cursor.execute("SHOW TABLES LIKE 'user_behavior_analysis'")
            if not self.cursor.fetchone():
                logger.error("The user behavior analysis table does not exist.")
                return pd.DataFrame()

            # Load user behavior data
            query = "SELECT * FROM user_behavior_analysis"
            self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()

            df = pd.DataFrame(data, columns=columns)
            logger.info(f"Loaded {len(df)} user behavior data")

            return df

        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
            return pd.DataFrame()

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering, building advanced user features"""
        try:
            # Ensure the dataframe is not empty
            if df.empty:
                logger.warning("The input data is empty, so feature engineering cannot be performed.")
                return df

            # 1. Standardize basic features
            numeric_cols = ['content_count', 'total_interactions', 'avg_interactions', 'max_interactions']
            for col in numeric_cols:
                if col in df.columns:
                    norm_col = f"{col}_norm"
                    df[norm_col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-6)

            # 2. Time-related features
            if 'first_post_date' in df.columns and 'last_post_date' in df.columns:
                df['first_post_date'] = pd.to_datetime(df['first_post_date'])
                df['last_post_date'] = pd.to_datetime(df['last_post_date'])
                df['account_age_days'] = (df['last_post_date'] - df['first_post_date']).dt.days + 1
                df['posting_consistency'] = df['content_count'] / df['account_age_days'].replace(0, 1)
            else:
                df['account_age_days'] = 1
                df['posting_consistency'] = df['content_count']

            # 3. Interaction quality features
            df['interaction_stability'] = 1 / (1 + df.get('interaction_std', 0) / (df.get('avg_interactions', 0) + 1))
            df['growth_potential'] = df.get('max_interactions', 0) / (df.get('avg_interactions', 0) + 1)

            # 4. Content quality features
            df['title_optimization'] = df.get('avg_title_length', 20) / 20  # Assuming 20 characters is optimal length
            df['special_char_usage'] = np.log1p(df.get('avg_special_chars', 0))

            # 5. Posting time preference features
            time_features = ['morning_ratio', 'afternoon_ratio', 'evening_ratio', 'night_ratio']
            for feature in time_features:
                if feature not in df.columns:
                    df[feature] = 0

            df['prime_time_ratio'] = df['afternoon_ratio'] + df['evening_ratio']  # Prime time

            # 6. Viral propagation capability
            df['viral_capability'] = df.get('viral_content_count', 0) / (df.get('content_count', 1))

            # 7. Comprehensive user activity score
            df['activity_score'] = (
                    df['content_count_norm'] * 0.3 +
                    df['interaction_efficiency_norm'] * 0.4 +
                    df['posting_consistency'] * 0.3
            )

            # 8. Influence score
            df['influence_score'] = (
                    np.log1p(df.get('total_interactions', 0)) * 0.4 +
                    df.get('avg_interactions', 0) * 0.3 +
                    df['viral_capability'] * 100 * 0.3
            )

            logger.info("Feature engineering completed")
            return df

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return df

    def advanced_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced clustering analysis"""
        try:
            # Select clustering features
            cluster_features = [
                'content_count', 'avg_interactions', 'interaction_efficiency',
                'daily_avg_posts', 'avg_sentiment', 'avg_title_length',
                'interaction_stability', 'growth_potential', 'prime_time_ratio',
                'viral_capability', 'activity_score', 'influence_score'
            ]

            # Ensure all features exist
            available_features = [f for f in cluster_features if f in df.columns]
            X = df[available_features].fillna(0)

            # Data standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['clustering'] = scaler

            # 1. Determine optimal number of clusters
            inertias = []
            silhouette_scores = []
            k_range = range(2, min(11, len(df) // 2))  # Ensure enough data points

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

            # Select optimal K value (highest silhouette score)
            optimal_k = k_range[np.argmax(silhouette_scores)]

            # 2. K-means clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
            self.cluster_model = kmeans

            # 3. DBSCAN clustering (identify anomalous users)
            dbscan = DBSCAN(eps=0.5, min_samples=max(2, len(df) // 20))
            df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

            # 4. Anomaly detection
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            df['anomaly_score'] = isolation_forest.decision_function(X_scaled)
            df['is_anomaly'] = isolation_forest.predict(X_scaled) == -1

            # 5. Cluster label interpretation
            cluster_labels = self.interpret_clusters(df, available_features)
            df['cluster_label'] = df['kmeans_cluster'].map(cluster_labels)

            # 6. Visualize clustering results
            self.visualize_clustering_results(df, X_scaled, available_features)

            logger.info(f"Clustering analysis completed, optimal number of clusters: {optimal_k}")
            return df

        except Exception as e:
            logger.error(f"Clustering analysis failed: {e}")
            return df

    def interpret_clusters(self, df: pd.DataFrame, features: List[str]) -> Dict[int, str]:
        """Interprets clustering results"""
        cluster_interpretations = {}

        for cluster_id in df['kmeans_cluster'].unique():
            cluster_data = df[df['kmeans_cluster'] == cluster_id]

            # Calculate the average value and relative ranking of each feature
            feature_scores = {}
            for feature in features:
                if feature in df.columns:
                    cluster_mean = cluster_data[feature].mean()
                    overall_mean = df[feature].mean()
                    feature_scores[feature] = cluster_mean / (overall_mean + 1e-6)

            # Determine cluster labels based on feature scores
            if feature_scores.get('influence_score', 0) > 1.5:
                label = "Influential User"
            elif feature_scores.get('activity_score', 0) > 1.3:
                label = "Active User"
            elif feature_scores.get('viral_capability', 0) > 1.2:
                label = "Viral Content Creator"
            elif feature_scores.get('interaction_efficiency', 0) > 1.2:
                label = "High-Quality User"
            elif feature_scores.get('content_count', 0) > 1.5:
                label = "High-Volume User"
            else:
                label = "Normal User"

            cluster_interpretations[cluster_id] = label

        return cluster_interpretations

    def visualize_clustering_results(self, df: pd.DataFrame, X_scaled: np.ndarray, features: List[str]):
        """Visualizes clustering results"""
        try:
            # 1. PCA dimensionality reduction visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            plt.figure(figsize=(15, 5))

            # PCA clustering plot
            plt.subplot(1, 3, 1)
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['kmeans_cluster'], cmap='viridis', alpha=0.7)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('K-means Clustering Results (PCA)')
            plt.colorbar(scatter)

            # t-SNE visualization
            plt.subplot(1, 3, 2)
            if len(df) > 4:  # Ensure enough data points for t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) // 4))
                X_tsne = tsne.fit_transform(X_scaled)
                scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['kmeans_cluster'], cmap='viridis', alpha=0.7)
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.title('K-means Clustering Results (t-SNE)')
                plt.colorbar(scatter)
            else:
                plt.text(0.5, 0.5, 'Not enough data points\nfor t-SNE', ha='center', va='center',
                         transform=plt.gca().transAxes)
                plt.title('t-SNE (Insufficient Data)')

            # Anomaly detection results
            plt.subplot(1, 3, 3)
            colors = ['red' if x else 'blue' for x in df['is_anomaly']]
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('Anomaly User Detection')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'clustering_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Cluster feature radar chart
            self.create_cluster_radar_chart(df, features)

            # 3. Cluster distribution statistics
            self.create_cluster_distribution_chart(df)

        except Exception as e:
            logger.error(f"Clustering visualization failed: {e}")

    def create_cluster_radar_chart(self, df: pd.DataFrame, features: List[str]):
        """Creates a radar chart of cluster features"""
        try:
            # Calculate mean of features for each cluster
            cluster_means = df.groupby('kmeans_cluster')[features].mean()

            # Data standardization
            for col in cluster_means.columns:
                cluster_means[col] = (cluster_means[col] - cluster_means[col].min()) / (
                            cluster_means[col].max() - cluster_means[col].min() + 1e-6)

            # Create radar chart
            n_clusters = len(cluster_means)
            if n_clusters > 0:
                n_features = len(features)
                angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
                angles += angles[:1]  # Close the loop

                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

                for i, (idx, row) in enumerate(cluster_means.iterrows()):
                    values = row.values.flatten().tolist()
                    values += values[:1]  # Close the loop
                    ax.plot(angles, values, linewidth=2, label=f'Cluster {idx}')
                    ax.fill(angles, values, alpha=0.1)

                # Add feature labels
                plt.xticks(angles[:-1], features, size=12)
                plt.yticks([])
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.title('User Cluster Feature Radar Chart', size=15)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'cluster_radar_chart.png'), dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.error(f"Failed to create radar chart: {e}")

    def create_cluster_distribution_chart(self, df: pd.DataFrame):
        """Creates a cluster distribution statistical chart"""
        try:
            # Cluster distribution
            plt.figure(figsize=(12, 5))

            # Cluster size distribution
            plt.subplot(1, 2, 1)
            cluster_counts = df['cluster_label'].value_counts()
            cluster_counts.plot(kind='bar', color='skyblue')
            plt.title('User Cluster Distribution')
            plt.xlabel('Cluster Type')
            plt.ylabel('Number of Users')
            plt.xticks(rotation=45)

            # Cluster interaction efficiency distribution
            plt.subplot(1, 2, 2)
            if 'interaction_efficiency' in df.columns and 'cluster_label' in df.columns:
                df.boxplot(column='interaction_efficiency', by='cluster_label', grid=False)
                plt.title('Interaction Efficiency Distribution by Cluster')
                plt.suptitle('')  # Remove auto-generated title
                plt.xlabel('Cluster Type')
                plt.ylabel('Interaction Efficiency')
                plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Failed to create distribution chart: {e}")

    def build_prediction_models(self, df: pd.DataFrame) -> Dict:
        """Builds prediction models"""
        try:
            # Ensure the dataframe is not empty and has enough samples
            if df.empty or len(df) < 10:
                logger.warning("Insufficient data to build prediction models")
                return {}

            # Select features and target variable
            features = [
                'content_count', 'avg_sentiment', 'avg_title_length', 'avg_special_chars',
                'interaction_stability', 'growth_potential', 'activity_score',
                'title_optimization', 'special_char_usage', 'prime_time_ratio'
            ]

            # Ensure all features exist
            available_features = [f for f in features if f in df.columns]

            if 'avg_interactions' not in df.columns or len(available_features) < 3:
                logger.warning("Missing necessary features, cannot build prediction models")
                return {}

            target = 'avg_interactions'

            # Prepare data
            X = df[available_features].fillna(0)
            y = df[target].fillna(0)

            # Data standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['prediction'] = scaler

            # 1. Random Forest Regression
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_scaled, y)
            rf_pred = rf_model.predict(X_scaled)
            rf_score = rf_model.score(X_scaled, y)
            self.prediction_models['random_forest'] = rf_model

            # 2. XGBoost Regression
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            xgb_model.fit(X_scaled, y)
            xgb_pred = xgb_model.predict(X_scaled)
            xgb_score = xgb_model.score(X_scaled, y)
            self.prediction_models['xgboost'] = xgb_model

            # 3. Feature importance analysis
            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance_rf': rf_model.feature_importances_,
                'importance_xgb': xgb_model.feature_importances_
            })
            feature_importance['avg_importance'] = (feature_importance['importance_rf'] + feature_importance[
                'importance_xgb']) / 2
            feature_importance = feature_importance.sort_values('avg_importance', ascending=False)

            # 4. Visualize feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'], feature_importance['avg_importance'], color='skyblue')
            plt.xlabel('Average Feature Importance')
            plt.ylabel('Feature')
            plt.title('User Interaction Prediction Model - Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 5. Visualize model performance comparison
            plt.figure(figsize=(8, 6))
            models = ['Random Forest', 'XGBoost']
            scores = [rf_score, xgb_score]
            plt.bar(models, scores, color=['lightblue', 'lightgreen'])
            plt.ylim(0, 1)
            plt.xlabel('Model')
            plt.ylabel('R² Score')
            plt.title('Prediction Model Performance Comparison')
            for i, v in enumerate(scores):
                plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'model_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Prediction models built successfully")

            return {
                'random_forest': {
                    'model': rf_model,
                    'score': rf_score,
                    'predictions': rf_pred.tolist()
                },
                'xgboost': {
                    'model': xgb_model,
                    'score': xgb_score,
                    'predictions': xgb_pred.tolist()
                },
                'feature_importance': feature_importance.to_dict('records')
            }

        except Exception as e:
            logger.error(f"Failed to build prediction models: {e}")
            return {}

    def generate_user_insights(self, df: pd.DataFrame) -> Dict:
        """Generates user insights report"""
        insights = {
            'overview': {},
            'cluster_insights': {},
            'recommendations': []
        }

        try:
            # 1. Overall insights
            insights['overview'] = {
                'total_users': len(df),
                'avg_content_count': df['content_count'].mean() if 'content_count' in df.columns else 0,
                'avg_interaction_efficiency': df[
                    'interaction_efficiency'].mean() if 'interaction_efficiency' in df.columns else 0,
                'anomaly_user_ratio': df['is_anomaly'].mean() if 'is_anomaly' in df.columns else 0
            }

            # 2. Cluster insights
            if 'cluster_label' in df.columns:
                for cluster_label in df['cluster_label'].unique():
                    cluster_df = df[df['cluster_label'] == cluster_label]
                    insights['cluster_insights'][cluster_label] = {
                        'user_count': len(cluster_df),
                        'percentage': len(cluster_df) / len(df) * 100,
                        'avg_interactions': cluster_df[
                            'avg_interactions'].mean() if 'avg_interactions' in cluster_df.columns else 0,
                        'avg_sentiment': cluster_df[
                            'avg_sentiment'].mean() if 'avg_sentiment' in cluster_df.columns else 0,
                        'key_characteristics': self.get_cluster_characteristics(cluster_df, df)
                    }

            # 3. Recommendations
            insights['recommendations'] = self.generate_recommendations(df)

            # 4. Save insights report
            with open(os.path.join(self.output_dir, 'user_insights.json'), 'w', encoding='utf-8') as f:
                json.dump(insights, f, ensure_ascii=False, indent=4, cls=DateTimeEncoder)

            logger.info("User insights report generated successfully")
            return insights

        except Exception as e:
            logger.error(f"Failed to generate user insights: {e}")
            return insights

    def get_cluster_characteristics(self, cluster_df: pd.DataFrame, all_df: pd.DataFrame) -> List[str]:
        """Gets key characteristics of the cluster"""
        characteristics = []

        try:
            # Calculate relative differences of features
            for col in cluster_df.columns:
                if col in ['id', 'author', 'kmeans_cluster', 'dbscan_cluster', 'cluster_label', 'is_anomaly']:
                    continue

                if pd.api.types.is_numeric_dtype(cluster_df[col]):
                    cluster_mean = cluster_df[col].mean()
                    all_mean = all_df[col].mean()

                    if all_mean != 0:
                        diff_pct = (cluster_mean - all_mean) / all_mean * 100

                        if abs(diff_pct) >= 20:  # Only record if difference exceeds 20%
                            direction = "higher than" if diff_pct > 0 else "lower than"
                            characteristics.append(f"{col} is {direction} average by {abs(diff_pct):.1f}%")

        except Exception as e:
            logger.error(f"Failed to get cluster characteristics: {e}")

        return characteristics[:5]  # Return the 5 most significant characteristics

    def generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generates recommendations"""
        recommendations = []

        try:
            # Recommendations based on data analysis
            if 'avg_sentiment' in df.columns and df['avg_sentiment'].mean() < 0.5:
                recommendations.append(
                    "Overall content sentiment is negative, it is recommended to guide users to post more positive content.")

            if 'viral_capability' in df.columns and df['viral_capability'].mean() < 0.1:
                recommendations.append(
                    "Users' viral content creation ability is weak, it is recommended to provide content creation guidance and hot topic recommendations.")

            if 'interaction_efficiency' in df.columns and df['interaction_efficiency'].std() > df[
                'interaction_efficiency'].mean():
                recommendations.append(
                    "User interaction efficiency varies greatly, it is recommended to provide personalized content optimization suggestions for low-efficiency users.")

            # Recommendations based on clustering
            if 'cluster_label' in df.columns:
                cluster_counts = df['cluster_label'].value_counts()
                if 'Normal User' in cluster_counts and cluster_counts['Normal User'] > len(df) * 0.5:
                    recommendations.append(
                        "The proportion of normal users is too high, it is recommended to implement user incentive programs to enhance user activity.")

                if 'Influential User' in cluster_counts and cluster_counts['Influential User'] < len(df) * 0.1:
                    recommendations.append(
                        "Influential users are scarce, it is recommended to focus on cultivating and supporting potential users.")

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")

        return recommendations

    def update_database_with_analysis(self, df: pd.DataFrame):
        """Updates the database with analysis results"""
        try:
            # Check if the table has necessary columns
            self.cursor.execute("DESCRIBE user_behavior_analysis")
            columns = [row[0] for row in self.cursor.fetchall()]

            # Build update SQL, only update existing columns
            update_fields = []
            if 'user_cluster' in columns:
                update_fields.append("user_cluster = %s")
            if 'cluster_label' in columns:
                update_fields.append("cluster_label = %s")

            if not update_fields:
                logger.warning("No analysis result columns available for update in the database table")
                return

            # Update user behavior analysis table
            for _, row in df.iterrows():
                update_values = []
                if 'user_cluster' in columns:
                    update_values.append(int(row.get('kmeans_cluster', 0)))
                if 'cluster_label' in columns:
                    update_values.append(row.get('cluster_label', 'Unknown'))

                update_values.append(row['author'])

                update_sql = f"""
                    UPDATE user_behavior_analysis 
                    SET {', '.join(update_fields)}
                    WHERE author = %s
                """

                self.cursor.execute(update_sql, update_values)

            self.connection.commit()
            logger.info("Analysis results updated to the database")

        except Exception as e:
            logger.error(f"Failed to update database: {e}")

    def run_complete_analysis(self) -> Dict:
        """Runs the complete user behavior analysis"""
        try:
            logger.info("Starting advanced user behavior analysis")

            # 1. Load data
            df = self.load_user_data()
            if df.empty:
                logger.error("Could not load user data")
                return {}

            # 2. Feature Engineering
            df = self.feature_engineering(df)

            # 3. Clustering Analysis
            df = self.advanced_clustering(df)

            # 4. Build Prediction Models
            model_results = self.build_prediction_models(df)

            # 5. Generate Insights Report
            insights = self.generate_user_insights(df)

            # 6. Update Database
            self.update_database_with_analysis(df)

            # 7. Save Analysis Results
            df.to_csv(os.path.join(self.output_dir, 'advanced_user_analysis_results.csv'),
                      index=False, encoding='utf-8-sig')

            logger.info("Advanced user behavior analysis completed")

            return {
                'user_data': df,
                'model_results': model_results,
                'insights': insights,
                'output_dir': self.output_dir
            }

        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            return {}

    def close_connection(self):
        """Closes the database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")

    def get_data_quality_report(self, batch_id: Optional[str] = None) -> Dict:
        """Gets data quality report"""
        try:
            if batch_id:
                sql = "SELECT * FROM data_quality_metrics WHERE batch_id = %s"
                self.cursor.execute(sql, (batch_id,))
            else:
                sql = "SELECT * FROM data_quality_metrics ORDER BY created_at DESC LIMIT 10"
                self.cursor.execute(sql)

            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]

            reports = []
            for result in results:
                report = dict(zip(columns, result))

                # Convert datetime objects to string
                for key, value in report.items():
                    if isinstance(value, datetime):
                        report[key] = value.isoformat()

                if report['error_details']:
                    try:
                        report['error_details'] = json.loads(report['error_details'])
                    except:
                        pass
                reports.append(report)

            return {'reports': reports}

        except Exception as e:
            logger.error(f"Error while getting data quality report: {e}")
            return {'reports': []}


# Usage example
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': 'root',
        'database': 'rednote',
        'charset': 'utf8mb4'
    }

    # Create advanced analyzer
    analyzer = AdvancedUserAnalysis(db_config)

    try:
        # Connect to database
        if analyzer.connect_to_database():
            # Run complete analysis
            results = analyzer.run_complete_analysis()

            if results:
                print("Advanced user behavior analysis completed!")
                print(f"Analysis results saved in: {results['output_dir']}")
                print(
                    f"User insights: {json.dumps(results['insights']['overview'], indent=2, ensure_ascii=False, cls=DateTimeEncoder)}")
            else:
                print("Analysis failed")

            # Get data quality report
            report = analyzer.get_data_quality_report()
            print("Data Quality Report:", json.dumps(report, indent=2, ensure_ascii=False, cls=DateTimeEncoder))

    finally:
        analyzer.close_connection()
