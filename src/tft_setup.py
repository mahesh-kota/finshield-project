"""
FinShield - Temporal Fusion Transformer Setup
=============================================
Tests TFT library installation and prepares
the fused social+market feature data for
sequential anomaly modeling.

Usage:
    python src/tft_setup.py
"""
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import warnings
warnings.filterwarnings("ignore")

print("PyTorch Forecasting imported successfully")
print(f"PyTorch version: {torch.__version__}")

# Load cross-modal anomaly data
anomalies_df = pd.read_csv("data/cross_modal_anomalies.csv")
anomalies_df["date"] = pd.to_datetime(anomalies_df["date"])
anomalies_df = anomalies_df.sort_values("date").reset_index(drop=True)

print(f"\nLoaded cross-modal data: {len(anomalies_df)} days")
print(f"Date range: {anomalies_df['date'].min().date()} to {anomalies_df['date'].max().date()}")
print(f"Anomaly days: {anomalies_df['is_cross_modal_anomaly'].sum()}")

# Prepare TFT-compatible format
# TFT needs: time_idx (integer), group_id, target, known/unknown covariates
anomalies_df["time_idx"] = range(len(anomalies_df))
anomalies_df["group"] = "finshield"  # single group (all data combined)

# Fill NaN values
feature_cols = ["suspicious_ratio", "avg_sentiment", "avg_urgency",
                "muril_confidence", "avg_abnormal_return",
                "avg_volume_zscore", "avg_range_zscore", "anomaly_ratio"]
anomalies_df[feature_cols] = anomalies_df[feature_cols].fillna(0)

# Target: anomaly_ratio (continuous) — TFT will learn to forecast this
# Using isolation score as the main target
anomalies_df["target"] = -anomalies_df["iso_score"]  # flip so higher = more anomalous

print(f"\nFeature columns: {feature_cols}")
print(f"Target (anomaly score) stats:")
print(anomalies_df["target"].describe().round(4))

# Check if we have enough data for TFT
min_required = 20
if len(anomalies_df) < min_required:
    print(f"\nWARNING: Only {len(anomalies_df)} days — TFT needs at least {min_required}")
    print("For now, verifying TimeSeriesDataSet creation only")

# Create TimeSeriesDataSet
max_encoder_length = 10  # look back 10 days
max_prediction_length = 3  # predict 3 days ahead

# Need enough data for at least one sample
if len(anomalies_df) > max_encoder_length + max_prediction_length:
    training_cutoff = len(anomalies_df) - max_prediction_length

    training = TimeSeriesDataSet(
        anomalies_df[anomalies_df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=feature_cols + ["target"],
        target_normalizer=GroupNormalizer(groups=["group"]),
    )

    print(f"\nTimeSeriesDataSet created successfully")
    print(f"Training samples: {len(training)}")

    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training,
        anomalies_df,
        predict=True,
        stop_randomization=True
    )
    print(f"Validation samples: {len(validation)}")

    # Create dataloaders
    train_dataloader = training.to_dataloader(train=True, batch_size=8, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=8, num_workers=0)
    print(f"Dataloaders created successfully")

    # Build TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=None,
    )
    print(f"\nTFT model created successfully")
    print(f"Number of parameters: {sum(p.numel() for p in tft.parameters()):,}")
    print("\nTFT setup complete — ready for training")

else:
    print(f"\nNot enough data for TFT training ({len(anomalies_df)} days)")
    print("TFT library verified working — will train when more overlap data available")
    print("TFT setup verification complete")