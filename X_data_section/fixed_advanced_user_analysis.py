import pandas as pd
import numpy as np
import pymysql
from catboost import CatBoostRegressor
from pymysql import Error
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

# 抑制joblib的警告和错误输出
warnings.filterwarnings('ignore')
# 重定向标准错误输出，抑制joblib的错误信息
class NullWriter:
    def write(self, s):
        pass
    def flush(self):
        pass

# Log Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用matplotlib的调试信息
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

        # Create output directory
        # 设置输出目录为项目根目录下的X_rednote_result
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
        """从数据库加载用户行为数据"""
        try:
            # 查询用户行为数据，聚合统计
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
            
            # 将结果转换为DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # 处理数值类型，防止异常转换
            numeric_cols = ['content_count', 'avg_interactions', 'max_interactions', 
                           'total_interactions', 'active_days', 'avg_sentiment',
                           'sentiment_std', 'avg_title_length', 'avg_special_chars',
                           'viral_content_count']
                           
            # 确保所有数值列都是float类型
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # 确保时间戳字段是字符串类型
            if 'post_timestamps' in df.columns:
                df['post_timestamps'] = df['post_timestamps'].astype(str)
                
            # 确保日期列是日期类型
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
        """Feature engineering, building advanced user features"""
        try:
            # Ensure the dataframe is not empty
            if df.empty:
                logger.warning("The input data is empty, so feature engineering cannot be performed.")
                return df

            # 防止数据泄露检查：检查直接使用目标变量的特征
            target_var = 'avg_interactions'
            if target_var in df.columns:
                # 检查潜在数据泄露的特征
                potential_leakage_features = ['interaction_efficiency', 'growth_potential', 'interaction_stability']

                # 记录警告
                logger.warning(f"检查潜在的数据泄露特征: {potential_leakage_features}")

                # 计算这些特征与目标变量的相关性
                for feature in potential_leakage_features:
                    if feature in df.columns:
                        correlation = df[[feature, target_var]].corr().iloc[0, 1]
                        logger.warning(f"特征 {feature} 与目标变量 {target_var} 的相关性: {correlation:.4f}")

                        # 如果相关性极高，考虑修改特征计算方法
                        if abs(correlation) > 0.95:
                            logger.warning(f"特征 {feature} 可能存在数据泄露，与目标变量相关性过高: {correlation:.4f}")
                            # 直接删除这些特征，避免使用它们
                            if feature in df.columns:
                                df = df.drop(columns=[feature])

            # =====================================
            # 完全重构特征工程，避免任何与目标变量的数据泄露
            # =====================================

            # 1. 基本特征 - 不使用任何与互动平均值相关的信息
            # 1.1 避免直接使用total_interactions/content_count计算互动效率
            if 'content_count' in df.columns:
                # 只使用内容数量本身
                df['content_engagement_ratio'] = np.log1p(df['content_count']) / (np.log1p(df['content_count']) + 2)
            else:
                df['content_engagement_ratio'] = 0.5  # 默认值

            # 2. 标准化特征时，只标准化不依赖于目标变量的特征
            safe_numeric_cols = ['content_count', 'content_engagement_ratio']
            
            # 明确移除可能导致数据泄露的特征
            leaked_features = ['total_interactions', 'max_interactions', 'interaction_efficiency', 'max_interactions_norm']
            for feature in leaked_features:
                if feature in df.columns:
                    df = df.drop(columns=[feature])

            for col in safe_numeric_cols:
                if col in df.columns:
                    norm_col = f"{col}_norm"
                    df[norm_col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-6)

            # 3. 时间相关特征 - 这些与目标变量无关，可以保留
            if 'first_post_date' in df.columns and 'last_post_date' in df.columns:
                df['first_post_date'] = pd.to_datetime(df['first_post_date'])
                df['last_post_date'] = pd.to_datetime(df['last_post_date'])
                df['account_age_days'] = (df['last_post_date'] - df['first_post_date']).dt.days + 1
                # 使用.clip(lower=1)代替replace(0,1)，更稳健地处理零值
                df['posting_consistency'] = df['content_count'] / df['account_age_days'].clip(lower=1)

                # 活跃天数比例
                if 'active_days' in df.columns:
                    df['active_days_ratio'] = df['active_days'] / df['account_age_days'].clip(lower=1)
                else:
                    df['active_days_ratio'] = 0.5
                    df['active_days'] = df['account_age_days'] // 2  # 估计值

                # 活跃周期性特征
                if 'post_timestamps' in df.columns:
                    try:
                        df['posting_periodicity'] = df['post_timestamps'].apply(
                            lambda x: self._calculate_periodicity(x) if isinstance(x, str) else 0
                        )
                    except Exception as e:
                        logger.warning(f"计算posting_periodicity失败: {e}")
                        df['posting_periodicity'] = 0
            else:
                df['account_age_days'] = 1
                df['posting_consistency'] = df['content_count']
                df['active_days_ratio'] = 0.5
                df['posting_periodicity'] = 0

            # 4. 完全重构互动相关特征，避免数据泄露
            # 4.1 稳定性指标 - 使用内容发布频率的稳定性
            df['content_stability'] = df['posting_periodicity']
            
            # 不再使用interaction_stability，使用content_stability替代
            df['interaction_stability'] = df['content_stability']  # 保持兼容性，但完全不同的计算方式

            # 4.2 成长潜力指标 - 完全重新设计，使用内容指标而非互动指标
            if 'active_days' in df.columns and 'account_age_days' in df.columns:
                # 使用活跃度和内容密度作为成长潜力指标
                recent_activity = df['active_days'] / (df['account_age_days'] + 10)
                content_density = df['content_count'] / (df['active_days'].clip(lower=1))
                df['creator_growth_potential'] = recent_activity * content_density
            else:
                df['creator_growth_potential'] = 0.5

            # 更新growth_potential，使用全新的计算方法
            df['growth_potential'] = df['creator_growth_potential']

            # 4.3 内容质量特征 - 基于非互动指标
            if 'avg_title_length' in df.columns:
                # 假设标题长度在10-30之间最优
                title_length = df['avg_title_length'].clip(lower=0)
                df['title_optimization'] = 1 - np.abs((title_length - 20) / 20).clip(upper=1)
            else:
                df['title_optimization'] = 0.5
                
            if 'avg_special_chars' in df.columns:
                # 特殊字符使用，避免过多
                df['special_char_usage'] = 1 - np.clip(df['avg_special_chars'] / 10, 0, 1)
            else:
                df['special_char_usage'] = 0.5

            # 5. 情感分析特征
            sentiment_cols = ['avg_sentiment', 'sentiment_std']
            for col in sentiment_cols:
                if col not in df.columns:
                    df[col] = 0.5  # 默认中性情感

            # 情感多样性
            df['sentiment_variation'] = df['sentiment_std']

            # 内容质量综合指数 - 完全基于非互动指标
            # 确保所有数值都是float类型，避免decimal.Decimal类型问题
            avg_sentiment = pd.to_numeric(df['avg_sentiment'], errors='coerce').fillna(0.5).astype(float)
            title_optimization = pd.to_numeric(df['title_optimization'], errors='coerce').fillna(0.5).astype(float)
            special_char_usage = pd.to_numeric(df['special_char_usage'], errors='coerce').fillna(0.5).astype(float)
            
            df['content_quality_index'] = (
                avg_sentiment * 0.3 +
                title_optimization * 0.4 +
                special_char_usage * 0.3
            )
            
            # 完全替代interaction_quality_index
            df['interaction_quality_index'] = df['content_quality_index']

            # 6. 发布时间偏好特征 - 这些与目标变量无关
            time_features = ['morning_ratio', 'afternoon_ratio', 'evening_ratio', 'night_ratio']
            for feature in time_features:
                if feature not in df.columns:
                    df[feature] = 0.25  # 默认均匀分布
                else:
                    # 确保时间特征是float类型
                    df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.25).astype(float)

            # 黄金时段比例
            df['prime_time_ratio'] = df['afternoon_ratio'] + df['evening_ratio']

            # 时间多样性指数 - 使用熵
            # 重构这部分代码，避免类型错误
            time_diversity = np.zeros(len(df))
            for feature in time_features:
                if feature in df.columns:
                    # 确保是numpy数组并进行向量化计算
                    feature_values = np.array(df[feature].values, dtype=np.float64)
                    # 安全计算熵，避免log(0)问题
                    safe_values = np.clip(feature_values, 1e-10, 1.0)
                    # 使用numpy的向量化操作
                    entropy_values = -safe_values * np.log2(safe_values)
                    # 替换任何可能的NaN值
                    entropy_values = np.nan_to_num(entropy_values, nan=0.0)
                    time_diversity += entropy_values
            
            # 标准化熵值
            df['time_diversity'] = time_diversity / len(time_features)

            # 7. 用户活跃度特征 - 无数据泄露
            # 确保数值类型一致
            content_count = pd.to_numeric(df['content_count'], errors='coerce').fillna(0).astype(float)
            account_age_days = pd.to_numeric(df['account_age_days'], errors='coerce').fillna(1).astype(float)
            
            # 活跃度指标
            df['activity_score'] = (
                (content_count / (account_age_days.clip(lower=1) + 5)) * 50
            ).clip(upper=1)

            # 参与度指标 - 基于内容发布频率和活跃天数比例
            # 确保所有数值都是float类型
            content_stability = pd.to_numeric(df['content_stability'], errors='coerce').fillna(0).astype(float)
            posting_consistency = pd.to_numeric(df['posting_consistency'], errors='coerce').fillna(0).astype(float)
            active_days_ratio = pd.to_numeric(df['active_days_ratio'], errors='coerce').fillna(0).astype(float)
            
            df['engagement_index'] = (
                content_stability * 0.4 +
                posting_consistency.clip(upper=1) * 0.3 +
                active_days_ratio.clip(upper=1) * 0.3
            )

            # 忠诚度指标
            time_diversity = pd.to_numeric(df['time_diversity'], errors='coerce').fillna(0).astype(float)
            
            df['loyalty_index'] = (
                posting_consistency.clip(upper=1) * 0.4 +
                active_days_ratio.clip(upper=1) * 0.4 +
                time_diversity.clip(upper=1) * 0.2
            )

            # 8. 病毒传播潜力 - 重构，避免使用互动数据
            # 创建合成的病毒内容比率
            content_quality_index = pd.to_numeric(df['content_quality_index'], errors='coerce').fillna(0).astype(float)
            prime_time_ratio = pd.to_numeric(df['prime_time_ratio'], errors='coerce').fillna(0).astype(float)
            
            df['viral_capability'] = (
                content_quality_index * 0.5 +
                prime_time_ratio * 0.3 +
                time_diversity * 0.2
            )
            
            # 病毒传播潜力
            df['viral_potential'] = pd.to_numeric(df['viral_capability'], errors='coerce').fillna(0).astype(float) * content_quality_index

            # 9. 内容影响力指标 - 不使用互动数据
            df['influence_score'] = (
                np.log1p(content_count) * 0.4 +
                content_quality_index * 0.3 +
                pd.to_numeric(df['viral_capability'], errors='coerce').fillna(0).astype(float) * 0.3
            )

            # 最后检查：移除与目标变量高度相关的特征（如果目标变量存在）
            if target_var in df.columns:
                # 计算所有特征与目标变量的相关性
                correlation_threshold = 0.8  # 进一步降低阈值，更积极地移除可能的数据泄露
                feature_correlations = {}
                for col in df.columns:
                    if col != target_var and pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            corr = df[[col, target_var]].corr().iloc[0, 1]
                            feature_correlations[col] = corr

                            if abs(corr) > correlation_threshold:
                                logger.warning(f"移除高相关性特征: {col}, 与目标变量相关性: {corr:.4f}")
                                # 直接从数据集中移除该特征
                                df = df.drop(columns=[col])
                        except Exception as e:
                            logger.warning(f"计算特征'{col}'与目标变量相关性时出错: {e}")

                # 记录相关性排序
                sorted_correlations = sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                logger.info(f"特征与目标变量相关性排序: {sorted_correlations[:5]}")

            logger.info("完全重构的特征工程完成，移除了所有与目标变量相关的直接依赖")
            return df

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df

    def _calculate_periodicity(self, timestamps_str):
        """计算用户发帖的周期性"""
        try:
            # 安全解析时间戳字符串
            if not timestamps_str or timestamps_str == 'None' or timestamps_str == '[]':
                return 0
                
            try:
                # 尝试标准JSON解析
                timestamps = json.loads(timestamps_str.replace("'", '"'))
            except json.JSONDecodeError:
                # 备选方案：按逗号分隔
                timestamps = timestamps_str.split(',')
            
            # 确保有足够的时间戳进行计算
            if not timestamps or len(timestamps) < 3:
                return 0

            # 转换为datetime对象，处理可能的无效日期
            datetimes = []
            for ts in timestamps:
                try:
                    dt = pd.to_datetime(ts.strip())
                    datetimes.append(dt)
                except:
                    continue  # 忽略无法解析的日期
            
            # 确保至少有2个有效日期
            if len(datetimes) < 2:
                return 0
                
            # 排序
            datetimes.sort()

            # 计算时间间隔
            intervals = [(datetimes[i+1] - datetimes[i]).total_seconds() / 3600 for i in range(len(datetimes)-1)]

            # 计算时间间隔的标准差与均值的比值（变异系数）
            if intervals and len(intervals) > 1:
                mean_interval = np.mean(intervals)
                if mean_interval > 0:
                    cv = np.std(intervals) / mean_interval
                    # 周期性 = 1 - 变异系数（受限在0-1之间）
                    periodicity = max(0, min(1, 1 - cv))
                    return periodicity
            return 0
        except Exception as e:
            logger.warning(f"计算周期性失败: {e}")
            return 0 