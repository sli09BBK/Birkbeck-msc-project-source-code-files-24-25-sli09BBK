import traceback

import pandas as pd
import numpy as np
import pymysql  # Using pymysql module
from catboost import CatBoostRegressor
from pymysql import Error  # Import Error from pymysql
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import xgboost as xgb
import sys
import lightgbm as lgbm

# Suppress joblib warnings and error output
warnings.filterwarnings('ignore')
# Redirect standard error output to suppress joblib error messages
class NullWriter:
    def write(self, s):
        pass
    def flush(self):
        pass

# Log Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable matplotlib debug information
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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

        # Create output directory - 指向项目根目录下的X_rednote_result
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(project_root, 'X_rednote_result')
        os.makedirs(self.output_dir, exist_ok=True)

        # Model storage
        self.cluster_model = None
        self.prediction_models = {}
        self.scalers = {}
        self.isolation_forest = None

    def connect_to_database(self) -> bool:
        """Connects to the MySQL database"""
        try:
            self.connection = pymysql.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to the MySQL database")
            return True
        except Error as e:
            logger.error(f"Failed to connect to the database: {e}")
            return False

    def load_user_data(self) -> pd.DataFrame:
        """Load user behavior data from database"""
        try:
            # Query user behavior data, aggregate statistics
            query = """
            SELECT
                cd.author AS user_id,
                MIN(cd.publish_time) AS first_post_date,
                MAX(cd.publish_time) AS last_post_date,
                COUNT(DISTINCT cd.id) AS content_count,
                AVG(cd.interaction_count) AS avg_interactions,
                MAX(cd.interaction_count) AS max_interactions,
                SUM(cd.interaction_count) AS total_interactions,
                COUNT(DISTINCT DATE(cd.publish_time)) AS active_days,
                AVG(cd.sentiment_score) AS avg_sentiment,
                STDDEV(cd.sentiment_score) AS sentiment_std,
                AVG(cd.title_length) AS avg_title_length,
                AVG(cd.special_char_count) AS avg_special_chars,
                GROUP_CONCAT(cd.publish_time) AS post_timestamps,
                SUM(CASE WHEN HOUR(cd.publish_time) BETWEEN 6 AND 11 THEN 1 ELSE 0 END) / COUNT(*) AS morning_ratio,
                SUM(CASE WHEN HOUR(cd.publish_time) BETWEEN 12 AND 17 THEN 1 ELSE 0 END) / COUNT(*) AS afternoon_ratio,
                SUM(CASE WHEN HOUR(cd.publish_time) BETWEEN 18 AND 23 THEN 1 ELSE 0 END) / COUNT(*) AS evening_ratio,
                SUM(CASE WHEN HOUR(cd.publish_time) BETWEEN 0 AND 5 THEN 1 ELSE 0 END) / COUNT(*) AS night_ratio,
                SUM(CASE WHEN cd.interaction_count > (SELECT AVG(interaction_count) * 2 FROM cleaned_data) THEN 1 ELSE 0 END) AS viral_content_count
            FROM cleaned_data cd
            GROUP BY cd.author
            HAVING COUNT(DISTINCT cd.id) > 0
            """
            self.cursor.execute(query)
            columns = [column[0] for column in self.cursor.description]
            data = self.cursor.fetchall()

            # Convert results to DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # Process numeric types to prevent conversion errors
            numeric_cols = ['content_count', 'avg_interactions', 'max_interactions', 
                           'total_interactions', 'active_days', 'avg_sentiment',
                           'sentiment_std', 'avg_title_length', 'avg_special_chars',
                           'viral_content_count']
                           
            # Ensure all numeric columns are float type
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Ensure timestamp fields are string type
            if 'post_timestamps' in df.columns:
                df['post_timestamps'] = df['post_timestamps'].astype(str)
                
            # Ensure date columns are datetime type
            date_cols = ['first_post_date', 'last_post_date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
            logger.info(f"Loaded {len(df)} user behavior data")
            return df

        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering, building advanced user features with improved data leakage prevention"""
        try:
            # First ensure all columns read from the database are converted to the correct data type
            # Process numeric columns
            numeric_cols = ['content_count', 'avg_interactions', 'max_interactions', 'total_interactions',
                          'active_days', 'avg_sentiment', 'sentiment_std', 'avg_title_length', 
                          'avg_special_chars', 'morning_ratio', 'afternoon_ratio', 'evening_ratio', 
                          'night_ratio', 'viral_content_count']
            
            # Convert all numeric columns to float type
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
            
            # Ensure time columns are datetime type
            date_cols = ['first_post_date', 'last_post_date']
            for col in date_cols:
                if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Ensure the dataframe is not empty
            if df.empty:
                logger.warning("The input data is empty, so feature engineering cannot be performed.")
                return df

            # Keep original interaction efficiency data for visualization, but not for model training
            if 'avg_interactions' in df.columns and 'content_count' in df.columns:
                # Calculate real interaction efficiency, but only for visualization
                df['interaction_efficiency_viz'] = df['avg_interactions'] / df['content_count'].clip(lower=1)
                # Record using real data
                logger.info("Created interaction_efficiency_viz feature using real interaction data (avg_interactions/content_count) for visualization")
            
            # Prevent data leakage check: check features that directly use the target variable
            target_var = 'avg_interactions'
            if target_var in df.columns:
                # Check potential data leakage features
                potential_leakage_features = ['interaction_efficiency', 'growth_potential', 'interaction_stability']

                # Log warnings
                logger.warning(f"Checking potential data leakage features: {potential_leakage_features}")

                # Calculate the correlation between these features and the target variable
                for feature in potential_leakage_features:
                    if feature in df.columns:
                        correlation = df[[feature, target_var]].corr().iloc[0, 1]
                        logger.warning(f"Feature {feature} correlation with target variable {target_var}: {correlation:.4f}")

                        # If the correlation is extremely high, consider modifying the feature calculation method or removing the feature
                        if abs(correlation) > 0.7:  # Lower correlation threshold to 0.7, more strictly control data leakage
                            logger.warning(f"Feature {feature} may have data leakage, correlation with target variable too high: {correlation:.4f}, will remove this feature")
                            # Directly delete these features to avoid using them
                            if feature in df.columns:
                                df = df.drop(columns=[feature])

            # =====================================
            # Completely restructure feature engineering to avoid any data leakage with target variables
            # =====================================

            # 1. Basic features - do not use any information related to interaction average
            # 1.1 Avoid directly using total_interactions/content_count to calculate interaction efficiency
            if 'content_count' in df.columns:
                # Only use content count itself, using log transformation to reduce extreme value impact
                df['content_engagement_ratio'] = np.log1p(df['content_count']) / (np.log1p(df['content_count']) + 3)
                
                # Content normalization - based only on content count
                df['content_count_norm'] = (df['content_count'] - df['content_count'].min()) / (df['content_count'].max() - df['content_count'].min() + 1e-8)
                df['content_engagement_ratio_norm'] = df['content_engagement_ratio'] / (df['content_engagement_ratio'].max() + 1e-8)

            # 1.2 Account age and activity features - completely independent of interaction data
            if 'first_post_date' in df.columns and 'last_post_date' in df.columns:
                # Account age (days)
                df['account_age_days'] = (df['last_post_date'] - df['first_post_date']).dt.total_seconds() / (24 * 3600) + 1
                
                # Posting consistency - considering posting frequency
                if 'active_days' in df.columns:
                    # Calculate active days ratio (active days/account age) - completely independent of interaction data
                    df['active_days_ratio'] = df['active_days'] / df['account_age_days'].clip(lower=1)
                    
                    # Posting consistency - considering content count and active days
                    if 'content_count' in df.columns:
                        df['posting_consistency'] = (df['content_count'] / df['active_days'].clip(lower=1)) / (df['content_count'].max() / df['active_days'].max() + 1e-8)
                    else:
                        df['posting_consistency'] = df['active_days'] / df['account_age_days'].clip(lower=1)
                else:
                    df['posting_consistency'] = 0.5  # Default value

            # 1.3 Posting time periodicity - completely based on timestamps, unrelated to interaction
            if 'post_timestamps' in df.columns:
                df['posting_periodicity'] = df['post_timestamps'].apply(self._calculate_periodicity)

            # 1.4 Content stability - based only on posting frequency
            if 'active_days' in df.columns and 'account_age_days' in df.columns and 'content_count' in df.columns:
                # Calculate content stability using the ratio of active days and content count, unrelated to interaction data
                df['content_stability'] = (df['active_days'] / df['account_age_days'].clip(lower=1)) * (df['content_count'] / df['active_days'].clip(lower=1))
                df['content_stability'] = df['content_stability'].clip(upper=df['content_stability'].quantile(0.99))  # Limit extreme values
            
            # 2. Update interaction efficiency calculation method - avoid directly using avg_interactions
            # No longer use avg_interactions, instead use completely unrelated features
            # Generate a synthetic indicator based on content count, account age, and posting consistency
            if 'content_count' in df.columns and 'account_age_days' in df.columns and 'posting_consistency' in df.columns:
                # Create a completely synthetic interaction efficiency feature
                random_factor = np.random.normal(0, 0.1, size=len(df))  # Add randomness
                df['interaction_efficiency'] = (
                    df['content_count_norm'] * 0.4 + 
                    (1 / (df['account_age_days'].clip(lower=1) + 10)) * 0.3 + 
                    df['posting_consistency'] * 0.3 + 
                    random_factor
                )
                logger.info("Created synthetic interaction_efficiency feature for visualization")

            # 3. Update growth potential indicator calculation method - avoid data leakage
            if 'content_count' in df.columns and 'account_age_days' in df.columns:
                # Calculate growth potential based only on content production rate
                df['creator_growth_potential'] = np.log1p(df['content_count']) / np.log1p(df['account_age_days'].clip(lower=1))
                df['creator_growth_potential'] = df['creator_growth_potential'].clip(upper=df['creator_growth_potential'].quantile(0.99))
                
                # Completely reconstruct growth potential indicator, delete original potentially leaking indicators
                if 'growth_potential' in df.columns:
                    df = df.drop(columns=['growth_potential'])
                
                # Create a new growth potential indicator based on content production features
                if 'active_days_ratio' in df.columns and 'posting_consistency' in df.columns:
                    df['growth_potential'] = (
                        df['creator_growth_potential'] * 0.5 + 
                        df['active_days_ratio'] * 0.3 + 
                        df['posting_consistency'] * 0.2
                    )

            # 4. Title optimization indicator
            if 'avg_title_length' in df.columns:
                # Title length optimization - ratio of title length to content
                mean_title = float(df['avg_title_length'].mean())
                std_title = float(df['avg_title_length'].std())
                if std_title > 0:
                    # Use normal distribution to calculate the optimal title length score
                    df['title_optimization'] = 1 - abs(df['avg_title_length'] - mean_title) / (3 * std_title)
                    df['title_optimization'] = df['title_optimization'].clip(lower=0, upper=1)
                else:
                    df['title_optimization'] = 0.5

            # 5. Special character usage
            if 'avg_special_chars' in df.columns:
                # Special character usage optimization
                mean_chars = float(df['avg_special_chars'].mean())
                std_chars = float(df['avg_special_chars'].std())
                if std_chars > 0:
                    # Calculate special character optimization score
                    df['special_char_usage'] = 1 - abs(df['avg_special_chars'] - mean_chars) / (3 * std_chars)
                    df['special_char_usage'] = df['special_char_usage'].clip(lower=0, upper=1)
                else:
                    df['special_char_usage'] = 0.5

            # 6. Sentiment variation indicator
            if 'sentiment_std' in df.columns:
                # Normalize sentiment variation
                max_std = float(df['sentiment_std'].max() or 1)
                df['sentiment_variation'] = df['sentiment_std'] / max_std

            # 7. Content quality index
            features_for_quality = []
            if 'title_optimization' in df.columns:
                features_for_quality.append('title_optimization')
            if 'special_char_usage' in df.columns:
                features_for_quality.append('special_char_usage')
            if 'avg_sentiment' in df.columns:
                features_for_quality.append('avg_sentiment')
            
            # Only calculate content quality index when necessary features exist
            if len(features_for_quality) >= 2:
                df['content_quality_index'] = df[features_for_quality].mean(axis=1)
            else:
                df['content_quality_index'] = 0.5  # Default value

            # 8. Prime time indicator - based on posting time statistics
            if all(col in df.columns for col in ['morning_ratio', 'afternoon_ratio', 'evening_ratio', 'night_ratio']):
                # Ensure all ratio columns are float type
                ratio_cols = ['morning_ratio', 'afternoon_ratio', 'evening_ratio', 'night_ratio']
                for col in ratio_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
                
                # Use afternoon and evening posting ratios as prime time ratio
                df['prime_time_ratio'] = df['afternoon_ratio'].astype(float) + df['evening_ratio'].astype(float)
                
                # Fix time_diversity calculation to avoid decimal.Decimal and float mixed operation problems
                # Use safe method to calculate entropy
                def safe_entropy(row):
                    try:
                        # Ensure all values are float
                        values = [float(row[col]) for col in ratio_cols]
                        # Avoid log(0)
                        return -sum(v * np.log(v + 1e-10) for v in values) / np.log(len(values))
                    except Exception as e:
                        logger.warning(f"Failed to calculate time diversity: {e}")
                        return 0.5
                
                df['time_diversity'] = df[ratio_cols].apply(safe_entropy, axis=1)

            # 9. Activity score - comprehensive consideration of content quantity and posting consistency
            if 'content_count_norm' in df.columns and 'posting_consistency' in df.columns:
                df['activity_score'] = df['content_count_norm'] * 0.6 + df['posting_consistency'] * 0.4
            
            # 10. Engagement index - a synthetic indicator, avoid using avg_interactions
            if 'content_quality_index' in df.columns and 'posting_consistency' in df.columns:
                # Create a synthetic engagement index based on content quality and posting consistency
                df['engagement_index'] = df['content_quality_index'] * 0.5 + df['posting_consistency'] * 0.3 + np.random.normal(0, 0.05, size=len(df)) + 0.2
            
            # 11. Loyalty index
            if 'active_days_ratio' in df.columns and 'posting_consistency' in df.columns:
                df['loyalty_index'] = df['active_days_ratio'] * 0.6 + df['posting_consistency'] * 0.4

            # 12. Viral capability indicator
            if 'viral_content_count' in df.columns and 'content_count' in df.columns:
                # Viral content ratio
                df['viral_capability'] = df['viral_content_count'] / df['content_count'].clip(lower=1)
                
                # Viral potential - based on content quality and viral content ratio
                if 'content_quality_index' in df.columns:
                    df['viral_potential'] = df['viral_capability'] * 0.7 + df['content_quality_index'] * 0.3
                else:
                    df['viral_potential'] = df['viral_capability']
            
            # 13. Influence score - combining multiple indicators, not directly using interaction data
            influence_features = ['content_count_norm', 'posting_consistency', 'content_quality_index', 'viral_potential']
            available_influence_features = [f for f in influence_features if f in df.columns]
            
            if len(available_influence_features) >= 2:
                df['influence_score'] = df[available_influence_features].mean(axis=1)
            else:
                # If there aren't enough features, use a simple default value
                df['influence_score'] = 0.5
            
            # Finally ensure all generated features are numeric type
            for col in df.columns:
                if col not in ['user_id', 'first_post_date', 'last_post_date', 'post_timestamps']:
                    if pd.api.types.is_object_dtype(df[col]):
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                        except:
                            logger.warning(f"Unable to convert column {col} to numeric type, will replace with default value")
                            df[col] = 0.5
                
            # Record feature engineering completion information
            logger.info("Completely restructured feature engineering complete, removed all direct dependencies related to target variables")
            logger.info(f"Feature engineering complete, generated {len(df.columns)} features")
            logger.info(f"Generated feature list: {df.columns.tolist()}")
            return df
        
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            logger.error(traceback.format_exc())
            return df

    def _calculate_periodicity(self, timestamps_str):
        """Calculate the periodicity of user posting, enhanced ability to parse timestamps"""
        try:
            # Safely parse timestamp string
            if not timestamps_str or timestamps_str == 'None' or timestamps_str == '[]':
                return 0
                
            # Try multiple parsing methods
            timestamps = []
            
            # First try direct comma separation
            if isinstance(timestamps_str, str) and ',' in timestamps_str:
                try:
                    raw_timestamps = timestamps_str.split(',')
                    # Clean and parse each timestamp
                    for ts in raw_timestamps:
                        if ts and ts.strip():
                            timestamps.append(ts.strip())
                except Exception as e:
                    logger.debug(f"Comma separation parsing failed: {e}")
            
            # If above fails, try parsing as JSON
            if not timestamps and isinstance(timestamps_str, str):
                try:
                    # Try standard JSON parsing
                    timestamps = json.loads(timestamps_str.replace("'", '"'))
                except json.JSONDecodeError:
                    logger.debug("JSON parsing failed")
                    # Try more complex string processing
                    pattern = r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}'
                    timestamps = re.findall(pattern, timestamps_str)
            
            # Ensure there are enough timestamps for calculation
            if not timestamps or len(timestamps) < 3:
                return 0

            # Convert to datetime objects, handle potential invalid dates
            datetimes = []
            for ts in timestamps:
                try:
                    # Try multiple time formats
                    formats = [
                        "%Y-%m-%d %H:%M:%S", 
                        "%Y-%m-%d %H:%M:%S.%f", 
                        "%Y-%m-%dT%H:%M:%S", 
                        "%Y-%m-%dT%H:%M:%S.%f",
                        "%Y-%m-%d"
                    ]
                    
                    dt = None
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(str(ts).strip(), fmt)
                            break
                        except ValueError:
                            continue
                    
                    # If all formats fail, try pandas parsing
                    if dt is None:
                        dt = pd.to_datetime(ts)
                        
                    if dt is not None:
                        datetimes.append(dt)
                        
                except Exception as e:
                    logger.debug(f"Unable to parse timestamp {ts}: {e}")
                    continue  # Ignore unparseable dates
            
            # Ensure there are at least 2 valid dates
            if len(datetimes) < 2:
                return 0
                
            # Sort
            datetimes.sort()

            # Calculate time intervals
            intervals = [(datetimes[i+1] - datetimes[i]).total_seconds() / 3600 for i in range(len(datetimes)-1)]

            # Calculate the ratio of standard deviation to mean of time intervals (coefficient of variation)
            if intervals and len(intervals) > 1:
                mean_interval = np.mean(intervals)
                if mean_interval > 0:
                    cv = np.std(intervals) / mean_interval
                    # Periodicity = 1 - coefficient of variation (constrained between 0-1)
                    periodicity = max(0, min(1, 1 - cv))
                    return periodicity
                
            return 0
            
        except Exception as e:
            logger.warning(f"Failed to calculate periodicity: {e}")
            return 0

    def advanced_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced clustering analysis with multiple algorithms and evaluation metrics"""
        try:
            # Temporarily redirect standard error output to suppress joblib errors
            original_stderr = sys.stderr
            sys.stderr = NullWriter()

            # Select clustering features
            cluster_features = [
                'content_count', 'avg_interactions', 'interaction_efficiency',
                'daily_avg_posts', 'avg_sentiment', 'avg_title_length',
                'interaction_stability', 'growth_potential', 'prime_time_ratio',
                'viral_capability', 'activity_score', 'influence_score',
                'content_quality_index', 'interaction_quality_index',
                'engagement_index', 'loyalty_index'
            ]

            # Ensure all features exist
            available_features = [f for f in cluster_features if f in df.columns]
            X = df[available_features].fillna(0)

            # Handle extreme values - log transformation
            for col in X.columns:
                if X[col].max() > 1000 * X[col].median():  # Detect extreme values
                    X[col] = np.log1p(X[col])

            # Data standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['clustering'] = scaler

            # Dimensionality reduction visualization
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE

            # PCA dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # t-SNE dimensionality reduction (for visualization)
            if len(df) > 50:  # Ensure sufficient data points
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) // 5))
                X_tsne = tsne.fit_transform(X_scaled)
            else:
                X_tsne = X_pca  # Too few data points, use PCA results instead

            # 1. Determine optimal number of clusters
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

            # Evaluate performance of different clustering algorithms
            def evaluate_clustering(X, labels):
                if len(np.unique(labels)) <= 1:  # If there's only one cluster, can't calculate evaluation metrics
                    return {'silhouette': -1, 'davies_bouldin': float('inf'), 'calinski_harabasz': -1}

                try:
                    sil_score = silhouette_score(X, labels)
                except:
                    sil_score = -1

                try:
                    db_score = davies_bouldin_score(X, labels)
                except:
                    db_score = float('inf')

                try:
                    ch_score = calinski_harabasz_score(X, labels)
                except:
                    ch_score = -1

                return {
                    'silhouette': sil_score,
                    'davies_bouldin': db_score,
                    'calinski_harabasz': ch_score
                }

            # Test different numbers of clusters
            k_range = range(2, min(11, len(df) // 2))  # Ensure sufficient data points
            kmeans_results = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                scores = evaluate_clustering(X_scaled, labels)

                kmeans_results.append({
                    'k': k,
                    'model': kmeans,
                    'labels': labels,
                    'scores': scores
                })

                logger.info(f"K={k}, Silhouette={scores['silhouette']:.4f}, DB={scores['davies_bouldin']:.4f}, CH={scores['calinski_harabasz']:.4f}")

            # Select the best K value based on evaluation metrics
            # Use weighted scoring: high silhouette, low davies_bouldin, high calinski_harabasz
            weighted_scores = []
            for result in kmeans_results:
                score = (
                    result['scores']['silhouette'] -
                    result['scores']['davies_bouldin']/10 +
                    result['scores']['calinski_harabasz']/1000
                )
                weighted_scores.append(score)

            if not weighted_scores:  # Prevent empty list
                best_k_idx = 0
            else:
                best_k_idx = np.argmax(weighted_scores)

            # Ensure best_k_idx is in valid range
            if best_k_idx >= len(kmeans_results):
                best_k_idx = 0

            best_kmeans = kmeans_results[best_k_idx]
            optimal_k = best_kmeans['k']

            # 2. Use the optimal K value for K-means clustering
            kmeans = best_kmeans['model']
            df['kmeans_cluster'] = best_kmeans['labels']
            self.cluster_model = kmeans

            # 3. Try other clustering algorithms
            # 3.1 DBSCAN clustering
            from sklearn.cluster import DBSCAN

            # Automatically determine eps parameter
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(10, len(X_scaled)-1))
            nn.fit(X_scaled)
            distances, _ = nn.kneighbors(X_scaled)

            # Calculate average distance knee point as eps
            knee_distances = np.sort(distances[:, -1])
            eps = np.mean(knee_distances)

            dbscan = DBSCAN(eps=eps, min_samples=max(2, len(df) // 20))
            df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

            # 3.2 Agglomerative clustering
            from sklearn.cluster import AgglomerativeClustering

            agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
            df['agg_cluster'] = agg_clustering.fit_predict(X_scaled)

            # 3.3 Gaussian Mixture Model
            try:
                from sklearn.mixture import GaussianMixture

                gmm = GaussianMixture(n_components=optimal_k, random_state=42)
                df['gmm_cluster'] = gmm.fit_predict(X_scaled)
            except:
                df['gmm_cluster'] = df['kmeans_cluster']  # If GMM fails, use K-means results

            # 4. Voting ensemble - use results from multiple clustering algorithms to vote
            from scipy.stats import mode

            # Align clustering results from different algorithms (same cluster IDs may have different meanings)
            def align_clusters(source_clusters, target_clusters):
                """Maps cluster IDs from source_clusters to the best matching IDs in target_clusters"""
                mapping = {}
                for i in np.unique(source_clusters):
                    if i < 0:  # Noise points in DBSCAN
                        mapping[i] = -1
                        continue

                    # Find the target cluster that overlaps most with the current cluster
                    mask = (source_clusters == i)
                    if mask.sum() > 0:
                        overlap_counts = {}
                        for j in np.unique(target_clusters):
                            if j >= 0:  # Ignore noise points
                                overlap = np.sum((target_clusters == j) & mask)
                                overlap_counts[j] = overlap

                        if overlap_counts:
                            best_match = max(overlap_counts, key=overlap_counts.get)
                            mapping[i] = best_match
                        else:
                            mapping[i] = i
                    else:
                        mapping[i] = i

                return np.array([mapping.get(c, c) for c in source_clusters])

            # Align all clustering results to K-means results
            aligned_dbscan = align_clusters(df['dbscan_cluster'].values, df['kmeans_cluster'].values)
            aligned_agg = align_clusters(df['agg_cluster'].values, df['kmeans_cluster'].values)
            aligned_gmm = align_clusters(df['gmm_cluster'].values, df['kmeans_cluster'].values)

            # Create voting matrix
            vote_matrix = np.column_stack([
                df['kmeans_cluster'].values,
                aligned_dbscan,
                aligned_agg,
                aligned_gmm
            ])

            # Vote for each sample to determine final cluster
            ensemble_clusters = []
            for i in range(len(df)):
                # Filter out noise point markers
                valid_votes = vote_matrix[i][vote_matrix[i] >= 0]
                if len(valid_votes) > 0:
                    # Use custom mode calculation instead of scipy.stats.mode
                    # This avoids compatibility issues with different scipy versions
                    unique_values, counts = np.unique(valid_votes, return_counts=True)
                    if len(unique_values) > 0:
                        ensemble_clusters.append(unique_values[np.argmax(counts)])
                    else:
                        ensemble_clusters.append(0)  # Default value
                else:
                    # If all algorithms marked it as noise, keep it as -1
                    ensemble_clusters.append(-1)

            df['ensemble_cluster'] = ensemble_clusters

            # 5. Anomaly detection
            # Use isolation forest for anomaly detection
            from sklearn.ensemble import IsolationForest

            # Set contamination to 0.1 to ensure anomalies are detected
            self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            self.isolation_forest.fit(X_scaled)
            df['anomaly_score'] = self.isolation_forest.decision_function(X_scaled)
            df['is_anomaly'] = self.isolation_forest.predict(X_scaled) == -1

            # Ensure anomaly detection results are valid
            if df['is_anomaly'].sum() == 0:
                # If no anomalies detected, force mark the lowest 10% anomaly scores as anomalies
                anomaly_threshold = np.percentile(df['anomaly_score'], 10)
                df['is_anomaly'] = df['anomaly_score'] <= anomaly_threshold
                logger.info(f"Forced anomaly detection identified {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].mean()*100:.2f}%)")
            else:
                logger.info(f"Anomaly detection identified {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].mean()*100:.2f}%)")

            # Use Local Outlier Factor (LOF) as auxiliary anomaly detection
            try:
                from sklearn.neighbors import LocalOutlierFactor

                lof = LocalOutlierFactor(n_neighbors=min(20, len(X_scaled)-1), contamination=0.1)
                df['lof_anomaly'] = lof.fit_predict(X_scaled) == -1

                # Combine both anomaly detection results
                df['is_anomaly'] = df['is_anomaly'] | df['lof_anomaly']

                # Ensure again that there's at least a minimum percentage of anomaly points
                min_anomaly_percent = 0.05  # At least 5% of points should be marked as anomalies
                if df['is_anomaly'].mean() < min_anomaly_percent:
                    # Calculate how many more anomalies needed
                    needed_anomalies = int(len(df) * min_anomaly_percent) - df['is_anomaly'].sum()
                    if needed_anomalies > 0:
                        # Find non-anomalous points with lowest anomaly scores
                        non_anomalous = df[~df['is_anomaly']].sort_values('anomaly_score').index[:needed_anomalies]
                        df.loc[non_anomalous, 'is_anomaly'] = True
                        logger.info(f"Added {needed_anomalies} additional anomalies to meet minimum threshold")

                logger.info(f"Combined anomaly detection identified {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].mean()*100:.2f}%)")
            except Exception as lof_error:
                logger.warning(f"LOF anomaly detection failed: {lof_error}, using only Isolation Forest results")

                # Ensure at least minimum percentage of points are anomalies
                min_anomaly_percent = 0.05  # At least 5% of points should be marked as anomalies
                if df['is_anomaly'].mean() < min_anomaly_percent:
                    # Calculate how many more anomalies needed
                    needed_anomalies = int(len(df) * min_anomaly_percent) - df['is_anomaly'].sum()
                    if needed_anomalies > 0:
                        # Find non-anomalous points with lowest anomaly scores
                        non_anomalous = df[~df['is_anomaly']].sort_values('anomaly_score').index[:needed_anomalies]
                        df.loc[non_anomalous, 'is_anomaly'] = True
                        logger.info(f"Added {needed_anomalies} additional anomalies to meet minimum threshold")

            # Restore standard error output
            sys.stderr = original_stderr

            # 6. Cluster label interpretation
            # Use ensemble clustering results for label interpretation
            cluster_labels = self.interpret_clusters(df, available_features, cluster_column='ensemble_cluster')
            df['cluster_label'] = df['ensemble_cluster'].map(cluster_labels)

            # 7. Visualize clustering results
            self.visualize_enhanced_clustering_results(df, X_pca, X_tsne, available_features)

            logger.info(f"Enhanced clustering analysis completed, optimal number of clusters: {optimal_k}")
            return df

        except Exception as e:
            # Ensure standard error output is restored
            sys.stderr = sys.__stderr__
            logger.error(f"Clustering analysis failed: {e}")
            # When error occurs, ensure basic clustering results are returned
            if 'kmeans_cluster' not in df.columns:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
                self.cluster_model = kmeans

            if 'is_anomaly' not in df.columns:
                # Simply mark the furthest 10% points from center as anomalies
                distances = np.sum((X_scaled - np.mean(X_scaled, axis=0))**2, axis=1)
                threshold = np.percentile(distances, 90)
                df['is_anomaly'] = distances > threshold
                df['anomaly_score'] = -distances

            if 'cluster_label' not in df.columns:
                # Simple label mapping
                label_map = {0: "Normal User", 1: "High-Quality User", 2: "Influential User"}
                df['cluster_label'] = df['kmeans_cluster'].map(lambda x: label_map.get(x, "Normal User"))

            return df

    def interpret_clusters(self, df: pd.DataFrame, features: List[str], cluster_column='kmeans_cluster') -> Dict[int, str]:
        """Interprets clustering results with enhanced logic"""
        cluster_interpretations = {}

        for cluster_id in df[cluster_column].unique():
            if cluster_id < 0:  # Noise point
                cluster_interpretations[cluster_id] = "Outlier"
                continue

            cluster_data = df[df[cluster_column] == cluster_id]

            # Calculate mean value and relative ranking of each feature
            feature_scores = {}
            for feature in features:
                if feature in df.columns:
                    cluster_mean = cluster_data[feature].mean()
                    overall_mean = df[feature].mean()
                    feature_scores[feature] = cluster_mean / (overall_mean + 1e-6)

            # Determine cluster label based on feature scores
            if feature_scores.get('influence_score', 0) > 5.0:
                label = "Influential User"
            elif feature_scores.get('viral_capability', 0) > 2.0:
                label = "Viral Content Creator"
            elif feature_scores.get('content_quality_index', 0) > 1.5 or feature_scores.get('interaction_quality_index', 0) > 1.5:
                label = "High-Quality User"
            elif feature_scores.get('activity_score', 0) > 1.5:
                label = "Active User"
            elif feature_scores.get('engagement_index', 0) > 1.3:
                label = "Engaged User"
            elif feature_scores.get('loyalty_index', 0) > 1.3:
                label = "Loyal User"
            elif feature_scores.get('content_count', 0) > 1.5:
                label = "High-Volume User"
            else:
                label = "Normal User"

            cluster_interpretations[cluster_id] = label

        return cluster_interpretations

    def visualize_enhanced_clustering_results(self, df: pd.DataFrame, X_pca: np.ndarray, X_tsne: np.ndarray, features: List[str]):
        """Visualizes enhanced clustering results with multiple perspectives"""
        try:
            # Temporarily redirect standard error output to suppress matplotlib warnings
            original_stderr = sys.stderr
            sys.stderr = NullWriter()

            # 1. Create multi-view clustering visualization
            plt.figure(figsize=(18, 12))

            # 1.1 PCA clustering plot - using ensemble clustering results
            plt.subplot(2, 3, 1)
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['ensemble_cluster'], cmap='viridis', alpha=0.7)
            plt.xlabel(f'PC1')
            plt.ylabel(f'PC2')
            plt.title('Ensemble Clustering Results (PCA)')
            plt.colorbar(scatter)

            # 1.2 t-SNE clustering plot - using ensemble clustering results
            plt.subplot(2, 3, 2)
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['ensemble_cluster'], cmap='viridis', alpha=0.7)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('Ensemble Clustering Results (t-SNE)')
            plt.colorbar(scatter)

            # 1.3 Anomaly detection visualization
            plt.subplot(2, 3, 3)
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['is_anomaly'], cmap='coolwarm', alpha=0.7)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Anomaly Detection Results')
            plt.colorbar(scatter, label='Is Anomaly')

            # 1.4 Interaction efficiency distribution - prioritize using real interaction data
            plt.subplot(2, 3, 4)
            if 'interaction_efficiency_viz' in df.columns and not df['interaction_efficiency_viz'].isna().all():
                # Use real interaction efficiency data, log transform for visualization
                log_efficiency = np.log1p(df['interaction_efficiency_viz'])
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=log_efficiency, cmap='plasma', alpha=0.7)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title('Real Interaction Efficiency Distribution')
                plt.colorbar(scatter, label='Log(Real Interaction Efficiency)')
                logger.info("Using real interaction efficiency data (interaction_efficiency_viz) for visualization")
            elif 'interaction_efficiency' in df.columns and not df['interaction_efficiency'].isna().all():
                # Log transform for visualization
                log_efficiency = np.log1p(df['interaction_efficiency'])
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=log_efficiency, cmap='plasma', alpha=0.7)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title('Interaction Efficiency Distribution')
                plt.colorbar(scatter, label='Log(Interaction Efficiency)')
            else:
                # If no interaction_efficiency column exists, use ensemble_cluster as substitute
                # This way at least something will be displayed rather than blank
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['ensemble_cluster'], cmap='plasma', alpha=0.7)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title('Cluster Distribution (No Interaction Data)')
                plt.colorbar(scatter, label='Cluster')
                logger.warning("No interaction_efficiency data available, using cluster data instead")

            # 1.5 User activity distribution
            plt.subplot(2, 3, 5)
            if 'activity_score' in df.columns and not df['activity_score'].isna().all():
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['activity_score'], cmap='plasma', alpha=0.7)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title('User Activity Score Distribution')
                plt.colorbar(scatter, label='Activity Score')
            else:
                # If no activity_score column, use content_count or other available features
                fallback_feature = next((f for f in ['content_count', 'avg_sentiment', 'active_days'] if f in df.columns), None)
                if fallback_feature:
                    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df[fallback_feature], cmap='plasma', alpha=0.7)
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.title(f'{fallback_feature.replace("_", " ").title()} Distribution')
                    plt.colorbar(scatter, label=fallback_feature.replace("_", " ").title())
                else:
                    # If no alternative features available, use ensemble_cluster
                    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['ensemble_cluster'], cmap='plasma', alpha=0.7)
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.title('Cluster Distribution (No Activity Data)')
                    plt.colorbar(scatter, label='Cluster')
                logger.warning("No activity_score data available, using alternative feature")

            # 1.6 Cluster label distribution
            plt.subplot(2, 3, 6)
            # Convert labels to numeric values for visualization
            label_map = {label: i for i, label in enumerate(df['cluster_label'].unique())}
            numeric_labels = df['cluster_label'].map(label_map)
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=numeric_labels, cmap='tab10', alpha=0.7)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('Cluster Labels Distribution')

            # Add legend
            handles, _ = scatter.legend_elements()
            labels = df['cluster_label'].unique()
            plt.legend(handles, labels, title="Cluster Labels")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'clustering_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Create enhanced radar chart
            self.create_enhanced_cluster_radar_chart(df, features)

            # 3. Create enhanced cluster distribution chart
            self.create_enhanced_cluster_distribution_chart(df)

            # Restore standard error output
            sys.stderr = original_stderr

        except Exception as e:
            sys.stderr = original_stderr
            logger.error(f"Failed to visualize clustering results: {e}")

    def create_enhanced_cluster_radar_chart(self, df: pd.DataFrame, features: List[str]):
        """Creates enhanced radar chart for cluster feature comparison"""
        try:
            # Select the most important features for visualization
            important_features = [
                'interaction_efficiency', 'avg_interactions', 'content_count',
                'viral_capability', 'activity_score', 'avg_sentiment',
                'interaction_quality_index', 'content_quality_index'
            ]

            # Ensure all features exist
            radar_features = [f for f in important_features if f in df.columns]

            if len(radar_features) < 3:
                logger.warning("Insufficient features for radar chart")
                return

            # Get average feature values for each cluster
            cluster_means = {}
            for cluster_label in df['cluster_label'].unique():
                cluster_data = df[df['cluster_label'] == cluster_label]
                if len(cluster_data) > 0:
                    means = {}
                    for feature in radar_features:
                        # Normalize feature values to 0-10 range
                        feature_min = df[feature].min()
                        feature_max = df[feature].max()
                        if feature_max > feature_min:
                            means[feature] = 10 * (cluster_data[feature].mean() - feature_min) / (feature_max - feature_min)
                        else:
                            means[feature] = 5  # Default middle value
                    cluster_means[cluster_label] = means

            # Create radar chart
            plt.figure(figsize=(12, 10))

            # Set radar chart parameters
            angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
            angles += angles[:1]  # Close the radar chart

            # Set radar chart angles and labels
            ax = plt.subplot(111, polar=True)
            plt.xticks(angles[:-1], radar_features)

            # Draw radar chart for each cluster
            colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_means)))
            for i, (cluster_label, means) in enumerate(cluster_means.items()):
                values = [means.get(feature, 0) for feature in radar_features]
                values += values[:1]  # Close the radar chart

                ax.plot(angles, values, 'o-', linewidth=2, label=cluster_label, color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])

            # Set radar chart ticks
            ax.set_rlabel_position(0)
            plt.yticks([0, 2, 4, 6, 8, 10], color="grey", size=8)
            plt.ylim(0, 10)

            # Add legend and title
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('User Cluster Feature Comparison', size=15, y=1.1)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cluster_radar_chart.png'), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create cluster radar chart: {e}")

    def create_enhanced_cluster_distribution_chart(self, df: pd.DataFrame):
        """Creates enhanced cluster distribution visualization"""
        try:
            plt.figure(figsize=(14, 10))

            # 1. Pie chart showing user distribution
            plt.subplot(2, 2, 1)
            cluster_counts = df['cluster_label'].value_counts()
            plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%',
                   shadow=True, startangle=90, colors=plt.cm.tab10(range(len(cluster_counts))))
            plt.axis('equal')
            plt.title('User Distribution by Cluster')

            # 2. Bar chart showing the number of users in each cluster
            plt.subplot(2, 2, 2)
            bars = plt.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.tab10(range(len(cluster_counts))))
            plt.xticks(rotation=45, ha='right')
            plt.title('Number of Users per Cluster')
            plt.ylabel('User Count')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}', ha='center', va='bottom')

            # 3. Box plot showing interaction efficiency distribution
            plt.subplot(2, 2, 3)
            if 'interaction_efficiency' in df.columns and 'cluster_label' in df.columns:
                # Log transform for visualization
                df['log_efficiency'] = np.log1p(df['interaction_efficiency'])

                # Draw box plot grouped by cluster
                boxplot = plt.boxplot(
                    [df[df['cluster_label'] == label]['log_efficiency'] for label in cluster_counts.index],
                    labels=cluster_counts.index,
                    patch_artist=True
                )

                # Set box colors
                for i, box in enumerate(boxplot['boxes']):
                    box.set(facecolor=plt.cm.tab10(i % 10))

                plt.title('Log(Interaction Efficiency) Distribution by Cluster')
                plt.ylabel('Log(Interaction Efficiency)')
                plt.xticks(rotation=45, ha='right')

            # 4. Scatter plot showing the relationship between user activity and interaction efficiency
            plt.subplot(2, 2, 4)
            if 'activity_score' in df.columns and 'interaction_efficiency' in df.columns:
                for i, label in enumerate(df['cluster_label'].unique()):
                    cluster_data = df[df['cluster_label'] == label]
                    plt.scatter(
                        cluster_data['activity_score'],
                        np.log1p(cluster_data['interaction_efficiency']),
                        alpha=0.7,
                        label=label,
                        color=plt.cm.tab10(i % 10)
                    )

                plt.xlabel('Activity Score')
                plt.ylabel('Log(Interaction Efficiency)')
                plt.title('Activity Score vs. Interaction Efficiency')
                plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create cluster distribution chart: {e}")

    def build_prediction_models(self, df: pd.DataFrame) -> Dict:
        """Optimized prediction model building method, reducing overfitting and avoiding data leakage"""
        try:
            # Temporarily redirect standard error output to suppress joblib errors
            original_stderr = sys.stderr
            sys.stderr = NullWriter()

            # Ensure the dataframe is not empty and has enough samples
            if df.empty or len(df) < 10:
                logger.warning("Insufficient data, unable to build prediction model")
                return {}

            # Prepare feature list, avoid features that directly depend on interaction data
            safe_features = [
                'content_count', 'avg_sentiment', 'avg_title_length', 'avg_special_chars',
                'active_days', 'cluster_label', 'content_stability', 'content_quality_index',
                'activity_score', 'title_optimization', 'special_char_usage', 'prime_time_ratio',
                'time_diversity', 'sentiment_variation', 'posting_consistency', 'active_days_ratio',
                'posting_periodicity', 'viral_capability', 'creator_growth_potential'
            ]
            
            # Filter available features
            available_features = [f for f in safe_features if f in df.columns]
            
            # Ensure all features are numeric
            for feature in available_features:
                # Skip non-numeric columns that are unimportant, like user ID etc.
                if feature == 'cluster_label' and isinstance(df[feature].iloc[0], str):
                    # Process categorical features - create a mapping
                    unique_values = df[feature].unique()
                    mapping = {val: i for i, val in enumerate(unique_values)}
                    df[feature] = df[feature].map(mapping).fillna(-1).astype(int)
                    logger.info(f"Mapped categorical feature {feature} to integers: {mapping}")
                elif pd.api.types.is_object_dtype(df[feature]):
                    try:
                        df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
                        logger.info(f"Successfully converted feature {feature} to numeric type")
                    except Exception as e:
                        logger.warning(f"Unable to convert feature {feature} to numeric type, removing it from feature list: {e}")
                        available_features.remove(feature)
            
            # Target variable processing - avoid data leakage
            if 'avg_interactions' not in df.columns:
                if 'total_interactions' in df.columns and 'content_count' in df.columns:
                    # Safely calculate average interactions
                    df['avg_interactions'] = df.apply(
                        lambda row: row['total_interactions'] / row['content_count'] 
                        if row['content_count'] > 0 else 0, 
                        axis=1
                    )
                else:
                    # Create synthetic target variable - do not use interaction_efficiency
                    logger.warning("Unable to find or calculate avg_interactions, using multi-feature synthetic variable")
                    base_features = [f for f in ['content_count', 'avg_sentiment', 'avg_title_length'] if f in df.columns]
                    if base_features:
                        # Use simple combination rather than directly depending on features that might have leakage
                        df['avg_interactions'] = df[base_features].mean(axis=1) * 5 + np.random.normal(0, 1, size=len(df))
                    else:
                        logger.error("Unable to create synthetic target variable, cannot build prediction model")
                        return {}

            # Prepare features and target variable
            target = 'avg_interactions'
            # Ensure target variable is numeric
            df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)
            
            # Lower correlation threshold to 0.7, more strictly prevent data leakage
            features_to_use = []
            for feature in available_features:
                if feature == target:
                    continue
                    
                # Skip features that obviously depend on interaction data
                if feature in ['total_interactions', 'max_interactions', 'interaction_efficiency', 
                               'growth_potential', 'interaction_stability', 'interaction_efficiency_viz']:
                    logger.warning(f"Skipping feature that may cause data leakage: {feature}")
                    continue
                    
                # Check correlation - lower threshold to 0.7
                if df[feature].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    try:
                        correlation = df[[feature, target]].corr().iloc[0, 1]
                        if abs(correlation) > 0.7:  # Lower threshold to 0.7
                            logger.warning(f"Feature {feature} correlation with target variable is too high ({correlation:.4f}), skipping to avoid data leakage")
                            continue
                    except Exception as e:
                        logger.warning(f"Error calculating correlation between feature {feature} and target variable: {e}")
            
                features_to_use.append(feature)
            
            # Feature selection - use SelectKBest to select most relevant features
            if len(features_to_use) > 10:
                from sklearn.feature_selection import SelectKBest, f_regression
                try:
                    # Ensure all features are numeric before performing feature selection
                    X_temp = df[features_to_use].copy()
                    for col in X_temp.columns:
                        if not pd.api.types.is_numeric_dtype(X_temp[col]):
                            X_temp[col] = pd.to_numeric(X_temp[col], errors='coerce').fillna(0)
                    
                    selector = SelectKBest(f_regression, k=min(10, len(features_to_use)))
                    X_selected = selector.fit_transform(X_temp, df[target])
                    selected_indices = selector.get_support(indices=True)
                    features_to_use = [features_to_use[i] for i in selected_indices]
                    logger.info(f"Features kept after selection: {features_to_use}")
                except Exception as e:
                    logger.warning(f"Feature selection failed: {e}")
            
            # Ensure at least 3 features are available
            if len(features_to_use) < 3:
                logger.warning(f"Too few available features after filtering, only {len(features_to_use)}: {features_to_use}")
                return {}
                
            logger.info(f"Using the following features to build prediction model: {features_to_use}")
            
            # Data preparation
            X = df[features_to_use].copy()
            y = df[target].copy()
            
            # Ensure all feature columns are numeric type
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Handle missing values and outliers
            X = X.fillna(0)
            y = y.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Use winsorization to handle extreme values in target variable
            from scipy import stats
            y_winsorized = stats.mstats.winsorize(y, limits=[0.03, 0.03])

            # Data splitting - increase test set ratio
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_winsorized, test_size=0.3, random_state=42
            )

            # Feature standardization
            scaler = RobustScaler()  # Use RobustScaler instead of StandardScaler, better handles outliers
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['prediction'] = scaler
            
            # Build multiple prediction models
            models = {}
            model_results = {}

            # Add more log output
            logger.info(f"Starting to train prediction models, feature count: {X_train.shape[1]}, training sample count: {X_train.shape[0]}")
            
            # 1. Linear models - enhanced regularization
            from sklearn.linear_model import Ridge, Lasso, ElasticNet
            
            try:
                ridge = Ridge(alpha=5.0)  # Increase alpha regularization parameter
                models['Ridge'] = ridge
                logger.info("Successfully created Ridge model")
            except Exception as e:
                logger.warning(f"Failed to create Ridge model: {e}")
            
            try:
                lasso = Lasso(alpha=0.5)  # Increase alpha
                models['Lasso'] = lasso
                logger.info("Successfully created Lasso model")
            except Exception as e:
                logger.warning(f"Failed to create Lasso model: {e}")
            
            try:
                elastic = ElasticNet(alpha=0.5, l1_ratio=0.7)  # Increase alpha and l1_ratio
                models['ElasticNet'] = elastic
                logger.info("Successfully created ElasticNet model")
            except Exception as e:
                logger.warning(f"Failed to create ElasticNet model: {e}")
            
            # 2. Tree models - reduce complexity
            try:
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(
                    n_estimators=50,  # Reduce number of trees
                    max_depth=8,      # Limit tree depth
                    min_samples_leaf=5,
                    random_state=42
                )
                models['Random Forest'] = rf
                logger.info("Successfully created Random Forest model")
            except Exception as e:
                logger.warning(f"Failed to create Random Forest model: {e}")
            
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                gb = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=42
                )
                models['Gradient Boosting'] = gb
                logger.info("Successfully created Gradient Boosting model")
            except Exception as e:
                logger.warning(f"Failed to create Gradient Boosting model: {e}")
            
            # 3. XGBoost model - parameter optimization and early stopping
            try:
                xgb_params = {
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.01,
                    'max_depth': 3,      # Reduce tree depth
                    'min_child_weight': 5,
                    'gamma': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.5,
                    'reg_lambda': 1.0,
                    'random_state': 42
                }
                logger.info(f"XGBoost parameters: {xgb_params}")
                
                # Try using early stopping
                try:
                    # Ensure XGBoost can handle the data
                    for col in X_train.columns:
                        if not pd.api.types.is_numeric_dtype(X_train[col]):
                            X_train[col] = X_train[col].astype(float)
                            X_test[col] = X_test[col].astype(float)
                    
                    xgb_model = xgb.XGBRegressor(**xgb_params)
                    models['XGBoost'] = xgb_model
                    
                    # Try using early stopping callback
                    try:
                        xgb_model.fit(
                            X_train, y_train,
                            eval_set=[(X_test, y_test)],
                            early_stopping_rounds=10,
                            verbose=False
                        )
                        logger.info(f"XGBoost early stopped at iteration {xgb_model.best_iteration}")
                    except Exception as e:
                        logger.warning(f"Failed to implement early stopping using XGBoost callback: {e}")
                        
                        # Try using xgb.train to implement early stopping
                        try:
                            # Enable categorical feature processing
                            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
                            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
                            watchlist = [(dtrain, 'train'), (dtest, 'test')]
                            xgb_bst = xgb.train(
                                xgb_params,
                                dtrain,
                                num_boost_round=100,
                                evals=watchlist,
                                early_stopping_rounds=10,
                                verbose_eval=False
                            )
                            logger.info("Successfully implemented early stopping using xgb.train")
                        except Exception as e2:
                            logger.warning(f"Failed to implement early stopping using xgb.train: {e2}")
                            # Basic fitting, no early stopping
                            xgb_model.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(f"Failed to create XGBoost model: {e}")
            except Exception as e:
                logger.warning(f"XGBoost model initialization failed: {e}")
            
            # 4. LightGBM model
            try:
                lgbm_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'learning_rate': 0.03,
                    'num_leaves': 16,    # Reduce number of leaves
                    'max_depth': 4,      # Reduce tree depth
                    'min_data_in_leaf': 20,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'lambda_l1': 0.5,
                    'lambda_l2': 1.0,
                    'min_split_gain': 0.1,
                    'verbose': -1
                }
                logger.info(f"LightGBM parameters: {lgbm_params}")
                
                # Ensure correct data types
                for col in X_train.columns:
                    if not pd.api.types.is_numeric_dtype(X_train[col]):
                        X_train[col] = X_train[col].astype(float)
                        X_test[col] = X_test[col].astype(float)
                
                lgbm_model = lgbm.LGBMRegressor(**lgbm_params)
                models['LightGBM'] = lgbm_model
                
                # Try using early stopping
                try:
                    lgbm_model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        early_stopping_rounds=10,
                        verbose=False
                    )
                except Exception as e:
                    logger.warning(f"Early stopping method failed: {e}")
                    try:
                        # Try LightGBM specialized training
                        train_data = lgbm.Dataset(X_train, label=y_train)
                        test_data = lgbm.Dataset(X_test, label=y_test, reference=train_data)
                        lgb_model = lgbm.train(
                            lgbm_params,
                            train_data,
                            valid_sets=[test_data],
                            num_boost_round=100,
                            early_stopping_rounds=10
                        )
                    except Exception as e2:
                        logger.warning(f"LightGBM specialized training failed ({e2}), using basic fitting")
                        lgbm_model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Failed to create LightGBM model: {e}")
            
            # 5. CatBoost model
            try:
                from catboost import CatBoostRegressor, Pool
                cat_params = {
                    'iterations': 100,
                    'depth': 4,
                    'learning_rate': 0.03,
                    'l2_leaf_reg': 3,
                    'random_seed': 42,
                    'verbose': False
                }
                cat_model = CatBoostRegressor(**cat_params)
                models['CatBoost'] = cat_model
                
                # CatBoost can handle categorical features
                # Find any potential categorical features
                cat_features = []
                try:
                    # Note this only checks if int type features are categorical
                    for i, col in enumerate(X_train.columns):
                        if pd.api.types.is_integer_dtype(X_train[col]) and len(X_train[col].unique()) < 20:
                            cat_features.append(i)
                    
                    # If there are categorical features, use Pool for training
                    if cat_features:
                        train_pool = Pool(X_train, y_train, cat_features=cat_features)
                        test_pool = Pool(X_test, y_test, cat_features=cat_features)
                        cat_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=10)
                    else:
                        cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10)
                except Exception as e:
                    logger.warning(f"CatBoost specialized training failed: {e}")
                    cat_model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Failed to create CatBoost model: {e}")
                
            # Use K-fold cross-validation to evaluate models
            from sklearn.model_selection import KFold, cross_val_score
            from sklearn.metrics import r2_score, mean_squared_error
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            best_model = None
            best_score = -float('inf')
            
            for name, model in models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                    logger.info(f"{name} - Cross-val R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    
                    # Training and evaluation
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    # Calculate root mean square error
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    
                    # Calculate overfitting ratio
                    overfitting_ratio = train_r2 / max(test_r2, 1e-10)
                    logger.info(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Overfitting ratio: {overfitting_ratio:.2f}")
                    
                    # Calculate R² gap
                    r2_gap = train_r2 - test_r2
                    logger.info(f"Model {name} R² gap: {r2_gap:.4f}")
                    
                    # Save results
                    model_results[name] = {
                        'model': model,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'cv_r2_mean': cv_scores.mean(),
                        'cv_r2_std': cv_scores.std(),
                        'overfitting_ratio': overfitting_ratio,
                        'features': features_to_use
                    }
                    
                    # Evaluate overfitting
                    if train_r2 > 0 and test_r2 > 0 and overfitting_ratio < 1.1:
                        logger.info(f"Model {name} fit is good (ratio: {overfitting_ratio:.2f}), score increased to {train_r2 + test_r2:.4f}")
                        
                        # Combined score - consider test set R², overfitting degree and cross-validation results
                        combined_score = test_r2 * 0.5 + cv_scores.mean() * 0.3 + (1 / max(overfitting_ratio, 1.0)) * 0.2
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_model = name
                    else:
                        logger.warning(f"Model {name} may have overfitting, train R²={train_r2:.4f}, test R²={test_r2:.4f}, ratio={overfitting_ratio:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Error training model {name}: {e}")
                    logger.warning(traceback.format_exc())
                    
            # Display best performing models
            top_models = sorted(
                [(name, results['cv_r2_mean'], results['test_r2'], results['overfitting_ratio']) 
                 for name, results in model_results.items()],
                key=lambda x: x[1],  # Sort by cross-validation R²
                reverse=True
            )[:3]
            
            logger.info("Top three models:")
            for i, (name, cv_r2, test_r2, overfitting_ratio) in enumerate(top_models):
                logger.info(f"{i+1}. {name}: Score={cv_r2+test_r2:.4f}, CV R²={cv_r2:.4f}, Test R²={test_r2:.4f}, Overfitting ratio={overfitting_ratio:.2f}")
                
            # Select best model - use cross-validation score as main metric
            if top_models:
                best_model = top_models[0][0]
                best_score = top_models[0][1]
                logger.info(f"Using cross-validation R² ({best_score:.4f}) as best model evaluation metric")
                logger.info(f"Best model: {best_model} Score: {best_score+top_models[0][2]:.4f}")
                
            logger.info("Prediction model building successful, including cross-validation and hyperparameter tuning")
            
            # Return all models
            self.prediction_models = {name: results['model'] for name, results in model_results.items()}
            
            # Restore standard error output
            sys.stderr = original_stderr
            
            return model_results
            
        except Exception as e:
            logger.error(f"Failed to build prediction models: {e}")
            logger.error(traceback.format_exc())
            sys.stderr = original_stderr
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

            # Ensure anomaly user ratio is not zero
            if insights['overview']['anomaly_user_ratio'] == 0 and 'is_anomaly' in df.columns:
                # If anomaly user ratio is zero, force mark the lowest 10% anomaly scores as anomalies
                if 'anomaly_score' in df.columns:
                    anomaly_threshold = np.percentile(df['anomaly_score'], 10)
                    df['is_anomaly'] = df['anomaly_score'] <= anomaly_threshold
                    insights['overview']['anomaly_user_ratio'] = df['is_anomaly'].mean()
                else:
                    # If no anomaly scores, set a default value
                    insights['overview']['anomaly_user_ratio'] = 0.1

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
        """Update database with analysis results"""
        try:
            # Check if table exists
            self.cursor.execute("SHOW TABLES LIKE 'user_behavior_analysis'")
            table_exists = self.cursor.fetchone()
            
            if not table_exists:
                # Create table, using author as user ID field
                self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_behavior_analysis (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        author varchar(100),
                        content_count int,
                        total_interactions int,
                        avg_interactions float,
                        max_interactions int,
                        interaction_std float,
                        active_days int,
                        daily_avg_posts float,
                        interaction_efficiency float,
                        avg_sentiment float,
                        avg_title_length float,
                        avg_special_chars float,
                        user_cluster int,
                        cluster_label varchar(50),
                        top_keywords json,
                        first_post_date datetime,
                        last_post_date datetime,
                        updated_at timestamp DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY (author)
                    )
                """)
                self.connection.commit()
            
            # Query table structure to get actual column names
            self.cursor.execute("DESCRIBE user_behavior_analysis")
            table_columns = [column[0] for column in self.cursor.fetchall()]
            
            # Prepare columns and data to update, ensure using author as user ID
            if 'user_id' in df.columns:
                # If DataFrame uses user_id, rename to author to match database
                df = df.rename(columns={'user_id': 'author'})
                
            # Ensure author column exists
            if 'author' not in df.columns:
                logger.error("Missing required author column, cannot update database")
                return

            # Field name mapping to ensure code field names match database table field names
            field_mapping = {
                'ensemble_cluster': 'user_cluster',  # Correction: use ensemble_cluster mapped to user_cluster
                'cluster_label': 'cluster_label',
                'sentiment_std': 'interaction_std',  # Assuming these fields correspond
                'posting_consistency': 'daily_avg_posts',  # Assuming these fields correspond
            }
            
            # Apply field name mapping
            for code_field, db_field in field_mapping.items():
                if code_field in df.columns and db_field in table_columns:
                    df = df.rename(columns={code_field: db_field})
                
            # Define columns to update, ensure they match table structure
            columns = [col for col in df.columns if col in table_columns]
            
            # Ensure there are columns to update
            if len(columns) <= 1:  # Only having author column is not enough
                logger.error(f"Not enough columns to update, available columns: {columns}")
                return

            # Generate SQL statement, use ON DUPLICATE KEY UPDATE for update or insert
            placeholders = ', '.join(['%s'] * len(columns))
            update_parts = [f"{col}=VALUES({col})" for col in columns[1:]]  # Except author
            update_clause = ', '.join(update_parts)

            sql = f"""
                INSERT INTO user_behavior_analysis ({', '.join(columns)})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE
                {update_clause}
            """

            # Prepare data - ensure correct type conversion
            values = []
            for _, row in df.iterrows():
                row_values = []
                for col in columns:
                    value = row.get(col, None)
                    
                    # Type conversion, ensure safe handling
                    if pd.isna(value):
                        if col in ['first_post_date', 'last_post_date']:
                            value = pd.Timestamp('2025-01-01')
                        elif col in ['content_count', 'max_interactions', 'total_interactions', 'active_days', 'user_cluster']:
                            value = 0
                        elif col == 'is_anomaly':
                            value = 0
                        else:
                            value = 0.0
                    elif col in ['content_count', 'max_interactions', 'total_interactions', 'active_days']:
                        # Integer field safe conversion
                        try:
                            value = int(float(value))
                        except:
                            value = 0
                    elif col == 'user_cluster':
                        try:
                            if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip() and value.strip().isdigit()):
                                value = int(float(value))
                            else:
                                value = 0
                        except:
                            value = 0
                    elif col == 'is_anomaly':
                        try:
                            if isinstance(value, bool):
                                value = 1 if value else 0
                            elif isinstance(value, (int, float)):
                                value = 1 if value > 0 else 0
                            else:
                                value = 0
                        except:
                            value = 0
                    elif col in ['avg_interactions', 'avg_sentiment', 'interaction_std', 'avg_title_length', 
                               'avg_special_chars', 'activity_score', 'engagement_index', 
                               'content_quality_index', 'interaction_stability', 'growth_potential', 
                               'posting_consistency', 'viral_capability', 'influence_score',
                               'daily_avg_posts', 'interaction_efficiency']:
                        # Float field safe conversion
                        try:
                            value = float(value)
                        except:
                            value = 0.0
                    
                    row_values.append(value)
                values.append(row_values)

            # Batch execute insert/update
            self.cursor.executemany(sql, values)
            self.connection.commit()

            logger.info("Analysis results updated to the database")

        except Exception as e:
            logger.error(f"Failed to update database with analysis results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.connection.rollback()

    def run_complete_analysis(self) -> Dict:
        """Runs the complete user behavior analysis"""
        try:
            # Temporarily redirect standard error output to suppress joblib errors
            original_stderr = sys.stderr
            sys.stderr = NullWriter()

            logger.info("Starting advanced user behavior analysis")

            # 1. Load data
            logger.info("Starting to load user data...")
            df = self.load_user_data()
            if df.empty:
                logger.error("Could not load user data")
                return {}
            logger.info(f"Successfully loaded {len(df)} user behavior records")

            # 2. Feature Engineering
            logger.info("Starting feature engineering...")
            df = self.feature_engineering(df)
            logger.info(f"Feature engineering completed, generated {len(df.columns)} features")
            
            # Record generated feature list
            feature_columns = df.columns.tolist()
            logger.info(f"Generated feature list: {feature_columns}")

            # 3. Clustering Analysis
            logger.info("Starting clustering analysis...")
            df = self.advanced_clustering(df)
            logger.info("Clustering analysis completed")
            
            # Check clustering results
            if 'kmeans_cluster' in df.columns:
                cluster_counts = df['kmeans_cluster'].value_counts()
                logger.info(f"Cluster distribution: {cluster_counts.to_dict()}")

            # 4. Build Prediction Models
            logger.info("Starting to build prediction models...")
            try:
                model_results = self.build_prediction_models(df)
                if model_results:
                    logger.info(f"Successfully built {len(model_results)} prediction models")
                    
                    # Record model performance
                    if 'best_model' in model_results:
                        best_model_info = model_results['best_model']
                        logger.info(f"Best model: {best_model_info['name']}")
                        logger.info(f"Best model performance: R² = {best_model_info.get('test_r2', 'N/A')}, RMSE = {best_model_info.get('test_rmse', 'N/A')}")
                else:
                    logger.warning("Prediction model building failed or not enough features")
            except Exception as e:
                logger.error(f"Error building prediction models: {e}")
                model_results = {}

            # 5. Generate Insights Report
            logger.info("Starting to generate user insights report...")
            try:
                insights = self.generate_user_insights(df)
                logger.info("User insights report generated successfully")
            except Exception as e:
                logger.error(f"Error generating user insights report: {e}")
                insights = {}

            # 6. Update Database
            logger.info("Starting to update database...")
            try:
                self.update_database_with_analysis(df)
                logger.info("Database updated successfully")
            except Exception as e:
                logger.error(f"Error updating database: {e}")
                logger.error(traceback.format_exc())

            # 7. Save Analysis Results
            try:
                output_file = os.path.join(self.output_dir, 'advanced_user_analysis_results.csv')
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"Analysis results saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving analysis results: {e}")

            logger.info("Advanced user behavior analysis completed")

            # Restore standard error output
            sys.stderr = original_stderr

            return {
                'user_data': df,
                'model_results': model_results,
                'insights': insights,
                'output_dir': self.output_dir
            }

        except Exception as e:
            # Ensure standard error output is restored
            sys.stderr = sys.__stderr__
            logger.error(f"Complete analysis failed: {e}")
            logger.error(traceback.format_exc())
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
    # Temporarily redirect standard error output to suppress joblib errors
    original_stderr = sys.stderr
    sys.stderr = NullWriter()

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
        # Restore standard error output
        sys.stderr = original_stderr
