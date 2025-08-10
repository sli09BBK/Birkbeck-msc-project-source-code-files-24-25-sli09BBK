from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class RawDataModel(BaseModel):
    """Raw Data Model"""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    title: str
    content: Optional[str] = None
    author: str
    publish_time: Optional[datetime] = None
    like_count: Optional[str] = None
    comment_count: Optional[str] = None
    share_count: Optional[str] = None
    view_count: Optional[str] = None
    url: Optional[str] = None
    platform: str = "小红书"
    batch_id: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CleanedDataModel(BaseModel):
    """Cleaned Data Model"""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    raw_data_id: Optional[int] = None
    title: str
    author: str
    interaction_count: int = 0
    publish_time: Optional[datetime] = None
    content_url: Optional[str] = None
    title_length: Optional[int] = None
    special_char_count: Optional[int] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    keywords: Optional[List[str]] = None
    # Fixed type annotation - use Tuple for a list of (string, float) pairs
    emotion_scores: Optional[List[Tuple[str, float]]] = None  # Detailed emotion scores
    processed_at: Optional[datetime] = None


class UserBehaviorModel(BaseModel):
    """User Behavior Analysis Model"""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    author: str
    content_count: int = 0
    total_interactions: int = 0
    avg_interactions: float = 0
    max_interactions: int = 0
    interaction_std: float = 0
    active_days: int = 0
    daily_avg_posts: float = 0
    interaction_efficiency: float = 0
    avg_sentiment: float = 0.5
    avg_title_length: float = 0
    avg_special_chars: float = 0
    user_cluster: Optional[int] = None
    cluster_label: Optional[str] = None
    top_keywords: Optional[List[str]] = None
    first_post_date: Optional[datetime] = None
    last_post_date: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class KeywordAnalysisModel(BaseModel):
    """Keyword Analysis Model"""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    keyword: str
    frequency: int = 0
    total_interactions: int = 0
    avg_interactions: float = 0
    associated_authors: Optional[List[str]] = None
    sentiment_distribution: Optional[Dict[str, int]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DataQualityModel(BaseModel):
    """Data Quality Metrics Model"""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    batch_id: str
    total_records: Optional[int] = None
    valid_records: Optional[int] = None
    duplicate_records: Optional[int] = None
    invalid_records: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class PredictionModel(BaseModel):
    """Prediction Model Metadata"""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    model_name: str
    model_type: str
    target_variable: str
    features: Optional[List[str]] = None
    model_params: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    model_file_path: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SystemConfigModel(BaseModel):
    """System Configuration Model"""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    config_key: str
    config_value: Optional[str] = None
    config_type: str = "string"
    description: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TaskScheduleModel(BaseModel):
    """Task Scheduling Model"""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    task_name: str
    task_type: str
    schedule_expression: Optional[str] = None
    task_params: Optional[Dict[str, Any]] = None
    last_run_time: Optional[datetime] = None
    next_run_time: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Data Validation and Conversion Utility Class
class ModelUtils:
    """Model utility class, providing common data processing methods"""

    @staticmethod
    def to_dict(model: BaseModel, exclude_none: bool = True) -> Dict[str, Any]:
        """Converts model to dictionary"""
        return model.model_dump(exclude_none=exclude_none)

    @staticmethod
    def to_json(model: BaseModel, exclude_none: bool = True) -> str:
        """Converts model to JSON string"""
        return model.model_dump_json(exclude_none=exclude_none)

    @staticmethod
    def from_dict(model_class: type, data: Dict[str, Any]) -> BaseModel:
        """Creates a model instance from a dictionary"""
        return model_class.model_validate(data)

    @staticmethod
    def from_json(model_class: type, json_str: str) -> BaseModel:
        """Creates a model instance from a JSON string"""
        return model_class.model_validate_json(json_str)

    @staticmethod
    def update_model(model: BaseModel, updates: Dict[str, Any]) -> BaseModel:
        """Updates model fields"""
        model_data = model.model_dump()
        model_data.update(updates)
        return model.__class__.model_validate(model_data)


# If running this file directly, perform simple tests
if __name__ == "__main__":
    print("🧪 Testing Pydantic V2 Data Models...")

    # Create test instances
    raw_data = RawDataModel(
        title="Test Title",
        author="Test Author",
        platform="小红书"
    )

    cleaned_data = CleanedDataModel(
        title="Cleaned Title",
        author="Test Author",
        interaction_count=100,
        sentiment_score=0.8,
        sentiment_label="positive",
        keywords=["test", "example"],
        emotion_scores=[("happy", 0.8), ("sad", 0.1)]  # Correct format
    )

    user_behavior = UserBehaviorModel(
        author="Test Author",
        content_count=10,
        total_interactions=500
    )

    # Use new Pydantic V2 methods
    print("\n📊 Model Data Output:")
    print("Raw Data Model:", raw_data.model_dump())
    print("Cleaned Data Model:", cleaned_data.model_dump())
    print("User Behavior Model:", user_behavior.model_dump())

    # Test utility class methods
    print("\n🔧 Testing Utility Class Methods:")

    # Convert to dictionary (excluding None values)
    raw_dict = ModelUtils.to_dict(raw_data, exclude_none=True)
    print("Raw data (excluding None):", raw_dict)

    # Convert to JSON
    cleaned_json = ModelUtils.to_json(cleaned_data)
    print("Cleaned data JSON:", cleaned_json)

    # Create model from dictionary
    new_user = ModelUtils.from_dict(UserBehaviorModel, {
        "author": "New User",
        "content_count": 5,
        "total_interactions": 200
    })
    print("User model created from dictionary:", new_user.model_dump())

    # Update model
    updated_user = ModelUtils.update_model(new_user, {
        "content_count": 15,
        "avg_interactions": 40.0
    })
    print("Updated user model:", updated_user.model_dump())

    # Test data validation
    print("\n✅ Testing Data Validation:")
    try:
        # Test required field validation
        invalid_data = CleanedDataModel(
            # title="",  # Missing required field
            author="Test Author"
        )
    except Exception as e:
        print("Validation error (expected):", str(e))

    # Test field type validation
    try:
        invalid_type = UserBehaviorModel(
            author="Test",
            content_count="not a number"  # Incorrect type
        )
    except Exception as e:
        print("Type validation error (expected):", str(e))

    print("\n🎉 All tests completed!")
