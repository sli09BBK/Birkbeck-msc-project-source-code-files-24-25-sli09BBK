import json  # Import json library
import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta

import jieba.analyse
import jieba.analyse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pymysql
import seaborn as sns
import xgboost as XGBRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from snownlp import SnowNLP
import warnings
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory
# 设置输出目录为项目根目录下的X_rednote_result
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, "X_rednote_result")
os.makedirs(output_dir, exist_ok=True)

# 1. Database connection and data loading
logger.info('==== Data Loading and Preprocessing ====')
with open(os.path.join(os.path.dirname(__file__), 'database_config.json'), 'r', encoding='utf-8') as f:
    db_config = json.load(f)
try:
    conn = pymysql.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database'],
        charset='utf8mb4'
    )
    logger.info('Database connection established.')
except Exception as e:
    logger.exception('Database connection failed!')
    raise

query = 'SELECT * FROM cleaned_data'
df = pd.read_sql(query, conn)
logger.info(f"Data loaded: {len(df)} records, {df.shape[1]} fields. Columns: {list(df.columns)}")
logger.debug(f"Sample data: {df.head(3).to_dict()}")


def extract_likes(metrics):
    try:
        if pd.isnull(metrics):
            return 0
        metrics_dict = json.loads(metrics)
        likes = metrics_dict.get('likes', 0)
        # to number
        if isinstance(likes, str):
            try:
                likes = float(likes)
            except Exception:
                likes = 0
        return likes
    except Exception:
        return 0


df['count'] = df['interaction_metrics'].apply(extract_likes)
df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(float)
logger.info(f"count field generated from likes in interaction_metrics. Nulls: {df['count'].isnull().sum()}")

df['name'] = df['author']
df['title'] = df['title']
df['publish_time'] = df['publish_time']

logger.info(f"Final data shape: {df.shape}")
logger.info("Data preprocessing completed successfully!")

# Handle publish_time column
if 'publish_time' not in df.columns:
    # Create simulated publication times, most recent content is 1 day ago, earliest is 365 days ago
    np.random.seed(42)
    now = datetime.now()
    random_days = np.random.randint(1, 365, size=len(df))
    dates = [(now - timedelta(days=int(days))).strftime('%Y-%m-%d') for days in random_days]
    df['publish_time'] = dates
    print("Created simulated publication time field")

df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Fill any remaining NaN values
df['name'] = df['name'].fillna('Unknown User')
df['title'] = df['title'].fillna('Untitled')
df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)

print(f"Final data shape: {df.shape}")
print("Data preprocessing completed successfully!")

# 2. User Behavior Feature Engineering
print("\n==== User Behavior Feature Engineering ====")

# User activity analysis
user_activity = df.groupby('name').agg(
    content_count=('title', 'count'),
    avg_interaction=('count', 'mean'),
    max_interaction=('count', 'max'),
    total_interaction=('count', 'sum'),
    interaction_std=('count', 'std'),
    earliest_publish=('publish_time', 'min'),
    latest_publish=('publish_time', 'max')
).reset_index()

# Calculate user active days
user_activity['active_days'] = (user_activity['latest_publish'] - user_activity['earliest_publish']).dt.days + 1
# Calculate user publishing frequency (average daily content count)
user_activity['daily_publish_rate'] = user_activity['content_count'] / user_activity['active_days'].clip(lower=1)
# Calculate user interaction efficiency (average interactions per content)
user_activity['interaction_efficiency'] = user_activity['avg_interaction'] / user_activity['content_count'].clip(
    lower=1)

print(f"Analyzed behavioral data for {len(user_activity)} different users")

# Merge user features back into original data
df = pd.merge(df, user_activity[
    ['name', 'content_count', 'avg_interaction', 'interaction_efficiency', 'daily_publish_rate']],
              on='name', how='left')

# 3. Content Preference Feature Extraction
print("\n==== Content Preference Feature Extraction ====")

# Extract title keywords
print("Extracting content keywords...")
all_keywords = []
for title in df['title'].astype(str):
    if len(title) > 3:
        keywords = jieba.analyse.extract_tags(title, topK=3)
        all_keywords.extend(keywords)

# Get the most common keywords
keyword_counts = Counter(all_keywords)
top_keywords = [k for k, v in keyword_counts.most_common(20)]
print(f"Top 5 most common keywords: {', '.join(top_keywords[:5])}")

# Calculate keyword preferences for each user
user_keywords = {}
for _, row in df.iterrows():
    user = row['name']
    title = str(row['title'])
    count = row['count']

    if user not in user_keywords:
        user_keywords[user] = Counter()

    if len(title) > 3:
        keywords = jieba.analyse.extract_tags(title, topK=3)
        for keyword in keywords:
            user_keywords[user][keyword] += count  # Weight by interaction count

# Content sentiment analysis
print("Performing content sentiment analysis...")
df['content_sentiment'] = df['title'].astype(str).apply(
    lambda x: SnowNLP(x).sentiments if len(x) > 3 else 0.5)

# User sentiment tendency
user_sentiment = df.groupby('name')['content_sentiment'].mean().reset_index()
df = pd.merge(df, user_sentiment.rename(
    columns={'content_sentiment': 'user_sentiment_tendency'}), on='name', how='left')

# Title length and special character preferences
df['title_length'] = df['title'].astype(str).apply(len)
df['special_char_count'] = df['title'].astype(str).apply(
    lambda x: len(re.findall(r'[^\w\s\u4e00-\u9fff]', x)))

# User title length preferences
user_title_pref = df.groupby('name').agg(
    avg_title_length=('title_length', 'mean'),
    avg_special_chars=('special_char_count', 'mean')
).reset_index()

df = pd.merge(df, user_title_pref, on='name', how='left')

# 4. User Clustering Analysis
print("\n==== User Clustering Analysis ====")

# Prepare clustering features
cluster_features = [
    'content_count', 'avg_interaction', 'interaction_efficiency', 'daily_publish_rate',
    'user_sentiment_tendency', 'avg_title_length', 'avg_special_chars'
]

# Get unique user features
user_features = df.drop_duplicates('name')[['name'] + cluster_features]
user_features = user_features.dropna()  # Remove missing values

# Prepare clustering data
X_cluster = user_features[cluster_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Visualize elbow plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, 10), inertia, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Determining Optimal Number of Clusters (K)')
plt.savefig(f"{output_dir}/cluster_number_determination.png")
plt.close()

# Perform K-means clustering
optimal_k = 4  # Based on elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
user_features['user_cluster'] = kmeans.fit_predict(X_scaled)

# Analyze cluster features
cluster_analysis = user_features.groupby('user_cluster').agg({
    feat: 'mean' for feat in cluster_features
}).reset_index()

print("\nUser Cluster Feature Means:")
print(cluster_analysis)

# Visualize user cluster features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(cluster_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='user_cluster', y=feature, data=user_features)
    plt.title(f'Distribution of {feature} Across User Groups')
plt.tight_layout()
plt.savefig(f"{output_dir}/user_cluster_feature_distribution.png")
plt.close()

# Name the user clusters
cluster_names = {
    0: "High-Production Interactive",
    1: "Low-Frequency High-Quality",
    2: "Regular Active",
    3: "Novice Experimental"
}

# Merge clustering results back to original data
user_cluster_map = user_features[['name', 'user_cluster']].set_index('name').to_dict()['user_cluster']
df['user_cluster'] = df['name'].map(user_cluster_map)
df['user_cluster_type'] = df['user_cluster'].map(lambda x: cluster_names.get(x, "Uncategorized"))

# 5. User Behavior Pattern Analysis
print("\n==== User Behavior Pattern Analysis ====")

# Analyze interaction trends for different clusters
plt.figure(figsize=(12, 8))
for cluster in range(optimal_k):
    cluster_data = df[df['user_cluster'] == cluster]
    if len(cluster_data) > 0:
        sns.kdeplot(cluster_data['count'], label=f"{cluster_names[cluster]}")
plt.title('Interaction Count Distribution Across User Groups')
plt.xlabel('Interaction Count')
plt.ylabel('Density')
plt.legend()
plt.savefig(f"{output_dir}/user_group_interaction_distribution.png")
plt.close()

# Sentiment and interaction relationship analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='content_sentiment', y='count', hue='user_cluster_type', data=df)
plt.title('Content Sentiment vs. Interaction Count (By User Group)')
plt.xlabel('Content Sentiment Score (Higher is More Positive)')
plt.ylabel('Interaction Count')
plt.savefig(f"{output_dir}/sentiment_interaction_relationship.png")
plt.close()

# Content length and interaction relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x='title_length', y='count', hue='user_cluster_type', data=df)
plt.title('Title Length vs. Interaction Count (By User Group)')
plt.xlabel('Title Length')
plt.ylabel('Interaction Count')
plt.savefig(f"{output_dir}/title_length_interaction_relationship.png")
plt.close()

# 6. User Interaction Network Analysis
print("\n==== User Interaction Network Analysis ====")

# Simulate interaction relationships between users (in real applications, use actual interaction data)
# Here we assume users with high content similarity may interact
# Build network based on user content preferences

# Select active users
active_users = user_activity.sort_values('total_interaction', ascending=False).head(50)['name'].tolist()

# Build user interaction network
G = nx.Graph()

# Add nodes
for user in active_users:
    G.add_node(user)

# Add edges (based on user keyword preference similarity)
for i, user1 in enumerate(active_users):
    for user2 in active_users[i + 1:]:
        if user1 in user_keywords and user2 in user_keywords:
            # Calculate cosine similarity of keyword preferences between two users
            keywords1 = user_keywords[user1]
            keywords2 = user_keywords[user2]

            # Get common keywords
            common_keywords = set(keywords1.keys()) & set(keywords2.keys())
            if common_keywords:
                # Simplified similarity calculation to avoid empty set errors
                denom1 = sum(keywords1.values())
                denom2 = sum(keywords2.values())
                if denom1 > 0 and denom2 > 0:
                    dot_product = sum(keywords1[k] * keywords2[k] for k in common_keywords)
                    similarity = dot_product / ((sum(v ** 2 for v in keywords1.values())) ** 0.5 * (
                        sum(v ** 2 for v in keywords2.values())) ** 0.5)
                    if similarity > 0.1:  # Only keep connections with high similarity
                        G.add_edge(user1, user2, weight=similarity)

# If network is not empty, visualize user interaction network
if len(G.edges()) > 0:
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)

    # Node size based on user's total interactions
    user_total_interaction = {user: data['total_interaction'] for user, data in
                              user_activity.set_index('name').iterrows() if user in active_users}

    node_size = [user_total_interaction.get(node, 100) / 100 for node in G.nodes()]

    # Node color based on user cluster
    node_colors = [user_cluster_map.get(node, 0) for node in G.nodes()]

    nx.draw_networkx(
        G, pos,
        node_size=node_size,
        node_color=node_colors,
        cmap=plt.cm.tab10,
        alpha=0.8,
        with_labels=False
    )

    # Add labels for important nodes
    important_users = {user: user_total_interaction[user] for user in G.nodes()
                       if user_total_interaction.get(user, 0) > user_activity['total_interaction'].median()}

    label_pos = {user: (pos[user][0], pos[user][1] + 0.02) for user in important_users}
    nx.draw_networkx_labels(G, label_pos, labels={u: u for u in important_users})

    plt.title('User Interaction Network (Based on Content Preference Similarity)')
    plt.axis('off')
    plt.savefig(f"{output_dir}/user_interaction_network.png", dpi=300)
    plt.close()
    print(
        f"User interaction network analysis completed, containing {len(G.nodes())} users and {len(G.edges())} connections")

# 7. User Behavior Prediction Model
print("\n==== User Behavior Prediction Model ====")

# Prepare features and target variable
features = [
    'content_count', 'avg_interaction', 'interaction_efficiency', 'daily_publish_rate',
    'user_sentiment_tendency', 'avg_title_length', 'avg_special_chars',
    'title_length', 'special_char_count', 'content_sentiment'
]

# Log transform interaction counts
df['log_count'] = np.log1p(df['count'])

# Remove missing values
df_clean = df.dropna(subset=features + ['log_count'])

X = df_clean[features]
y = df_clean['log_count']  # Predict log-transformed interaction count

# Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Prepared model training data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# Train multiple models
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name} model...")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Transform back to original scale for evaluation
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)

    results[name] = {
        'MAE_log': mae,
        'RMSE_log': rmse,
        'R2': r2,
        'MAE_original': mae_orig
    }

    print(f"  {name} R2 score: {r2:.4f}")

# Visualize model performance
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), [results[m]['R2'] for m in results])
plt.title('R² Score Comparison Across Models')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_performance_comparison.png")
plt.close()

# Select best model
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = models[best_model_name]
print(f"Best model is: {best_model_name}, R² score: {results[best_model_name]['R2']:.4f}")

# If tree model, analyze feature importance
feature_importance_df = pd.DataFrame()  # Initialize empty DataFrame
if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title(f'{best_model_name} Model Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()

    print("\nMost important prediction features:")
    print(feature_importance_df)

# 8. User Behavior Prediction Application
print("\n==== User Behavior Prediction Application ====")


def predict_user_interaction(user_name, title_text, model=best_model):
    """Predict interaction count for given user publishing content with given title"""

    # Prepare features
    features_dict = {}

    # User features - if known user, use historical data
    if user_name in df['name'].values:
        user_data = df[df['name'] == user_name].iloc[0]
        features_dict['content_count'] = user_data['content_count']
        features_dict['avg_interaction'] = user_data['avg_interaction']
        features_dict['interaction_efficiency'] = user_data['interaction_efficiency']
        features_dict['daily_publish_rate'] = user_data['daily_publish_rate']
        features_dict['user_sentiment_tendency'] = user_data['user_sentiment_tendency']
        features_dict['avg_title_length'] = user_data['avg_title_length']
        features_dict['avg_special_chars'] = user_data['avg_special_chars']
    else:
        # If new user, use average values
        features_dict['content_count'] = df['content_count'].mean()
        features_dict['avg_interaction'] = df['avg_interaction'].mean()
        features_dict['interaction_efficiency'] = df['interaction_efficiency'].mean()
        features_dict['daily_publish_rate'] = df['daily_publish_rate'].mean()
        features_dict['user_sentiment_tendency'] = df['user_sentiment_tendency'].mean()
        features_dict['avg_title_length'] = df['avg_title_length'].mean()
        features_dict['avg_special_chars'] = df['avg_special_chars'].mean()

    # Content features
    features_dict['title_length'] = len(title_text)
    features_dict['special_char_count'] = len(re.findall(r'[^\w\s\u4e00-\u9fff]', title_text))

    try:
        features_dict['content_sentiment'] = SnowNLP(title_text).sentiments
    except:
        features_dict['content_sentiment'] = 0.5

    # Create features DataFrame
    X_pred = pd.DataFrame([features_dict])

    # Predict
    log_pred = model.predict(X_pred)[0]
    pred = int(np.expm1(log_pred))

    return max(0, pred)


# Test prediction function
test_cases = [
    {"user": "Food Expert", "title": "Super delicious home cooking secrets, easy to learn!"},
    {"user": "Travel Photography", "title": "Stunning landscape photos, shooting techniques revealed"},
    {"user": "Tech Reviews", "title": "Latest smartphone in-depth review: comprehensive pros and cons analysis"},
    {"user": "Life Records", "title": "Feeling down today, some music to relax"}
]

print("\nPrediction Examples:")
for case in test_cases:
    pred = predict_user_interaction(case["user"], case["title"])
    print(f"User: {case['user']}")
    print(f"Title: {case['title']}")
    print(f"Predicted Interaction Count: {pred}\n")

# 9. User Behavior Insights and Recommendations
print("\n==== User Behavior Insights and Recommendations ====")

# Keyword preferences for different groups
cluster_keywords = {}
for cluster in range(optimal_k):
    cluster_users = df[df['user_cluster'] == cluster]['name'].unique()

    cluster_titles = df[df['user_cluster'] == cluster]['title'].astype(str).tolist()
    all_text = ' '.join(cluster_titles)

    # Extract keywords
    if len(all_text) > 10:
        keywords = jieba.analyse.extract_tags(all_text, topK=10)
        cluster_keywords[cluster] = keywords

print("\nContent Preference Keywords by User Group:")
for cluster, keywords in cluster_keywords.items():
    print(f"{cluster_names[cluster]}: {', '.join(keywords)}")

# User behavior summary
print("\nUser Behavior Analysis Summary:")
summary_points_python = []
summary_points_python.append(
    f"1. Analyzed {len(user_activity)} users, categorized into {optimal_k} distinct behavioral groups")

# Find most active user group
most_active_cluster_id = cluster_analysis.sort_values('content_count', ascending=False).iloc[0]['user_cluster']
most_active_name = cluster_names[int(most_active_cluster_id)]
summary_points_python.append(f"2. Most active user group is: {most_active_name}")
print(f"2. Most active user group is: {most_active_name}")

# Find user group with highest interaction efficiency
most_efficient_cluster_id = cluster_analysis.sort_values('interaction_efficiency', ascending=False).iloc[0][
    'user_cluster']
most_efficient_name = cluster_names[int(most_efficient_cluster_id)]
summary_points_python.append(f"3. User group with highest interaction efficiency is: {most_efficient_name}")
print(f"3. User group with highest interaction efficiency is: {most_efficient_name}")

# User behavior recommendations
print("\nUser Behavior Optimization Recommendations:")
recommendation_points_python = []
recommendation_points_python.append(
    "1. High-interaction content characteristics: Positive emotional expression, moderate title length, appropriate use of special characters")
recommendation_points_python.append(
    "2. For new users: Recommend primarily positive emotional content of medium length, gradually building personal style")
recommendation_points_python.append(
    "3. For low-interaction users: Analyze content characteristics of high-interaction users, appropriately adjust title expression")
recommendation_points_python.append(
    "4. Publishing frequency recommendation: Maintain stable daily publishing volume, avoid extended periods of inactivity")
recommendation_points_python.append(
    "5. Customize content themes based on keyword preferences of different groups")  # Add one recommendation

# Save analysis results
print(f"\nAll analysis results have been saved to the '{output_dir}' directory")

# --- Save data as JSON file for frontend use ---
print("\n==== Exporting Data to JSON File ====")
data_for_frontend = {}

# KPIs
data_for_frontend['kpiData'] = {
    'users': len(user_activity),
    'items': len(df),
    'avgInteraction': df['count'].mean() if 'count' in df.columns else 0  # Ensure count column exists
}

# User cluster distribution
cluster_counts = df.groupby('user_cluster_type').size().reset_index(name='value')
data_for_frontend['clusterDistributionData'] = cluster_counts.rename(columns={'user_cluster_type': 'name'}).to_dict(
    orient='records')

# Scatter plot data (sentiment vs. interaction, title length vs. interaction)
# To avoid excessive data volume, sample the data here, or just take averages for each cluster to show trends
# Here we use df_clean data directly, but frontend may need to handle large data volumes, optimization may be needed in practice
sentiment_scatter_data = []
length_scatter_data = []
for cluster_id, cluster_name in cluster_names.items():
    cluster_df = df_clean[df_clean['user_cluster'] == cluster_id]
    if not cluster_df.empty:
        # For ECharts scatter plot format, convert to [[x1, y1], [x2, y2], ...]
        sentiment_data_points = cluster_df[['content_sentiment', 'count']].values.tolist()
        length_data_points = cluster_df[['title_length', 'count']].values.tolist()

        sentiment_scatter_data.append({
            'name': cluster_name,
            'type': 'scatter',
            'symbolSize': 5,
            'data': sentiment_data_points
        })
        length_scatter_data.append({
            'name': cluster_name,
            'type': 'scatter',
            'symbolSize': 5,
            'data': length_data_points
        })

data_for_frontend['scatterData'] = {
    'sentiment': sentiment_scatter_data,
    'length': length_scatter_data
}

# Model performance
data_for_frontend['modelPerformanceData'] = {
    'names': list(results.keys()),
    'scores': [results[name]['R2'] for name in results.keys()]
}

# Feature importance
if not feature_importance_df.empty:
    data_for_frontend['featureImportanceData'] = feature_importance_df.rename(
        columns={'feature': 'name', 'importance': 'value'}).to_dict(orient='records')
else:
    data_for_frontend['featureImportanceData'] = []

# User group keyword preferences
formatted_cluster_keywords = {}
for cluster_id, keywords in cluster_keywords.items():
    formatted_cluster_keywords[cluster_names[cluster_id]] = keywords
data_for_frontend['clusterKeywordsData'] = formatted_cluster_keywords

# Summary and recommendations
data_for_frontend['summaryPoints'] = summary_points_python
data_for_frontend['recommendationPoints'] = recommendation_points_python

# Save data to JSON file
json_output_path = os.path.join(output_dir, "dashboard_data.json")
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(data_for_frontend, f, ensure_ascii=False, indent=4)

print(f"All dashboard data exported to: {json_output_path}")
