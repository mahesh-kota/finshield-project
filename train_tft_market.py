"""
FinShield - TFT market-stress forecaster (market-only, 47 NSE stocks).

Matched to data/market_features.csv:
  ticker, Date, Close, Volume, daily_return, abnormal_return,
  volume_zscore, range_zscore, is_anomaly

Target   : |abnormal_return| on the next trading day (market-stress proxy)
Models   : naive (yesterday), 5-day rolling mean, TFT
Metrics  : MAE, RMSE on a chronological hold-out (last 20% of days)
Social   : deliberately NOT included here. Only 75 days have Telegram posts
           (2022-06 to 2026-06), too sparse for a daily TFT covariate.
           Social signals are evaluated separately (event study).

Run from the repo root:  python train_tft_market.py
CPU-friendly (small model); no checkpoints written.
"""
import os
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

# ----------------------------- CONFIG ------------------------------------
MARKET_CSV = "data/market_features.csv"
ENC_LEN = 20
TEST_FRAC = 0.2
HIDDEN = 8
EPOCHS = 30
BATCH = 128
SEED = 42
RESULTS_CSV = "results/tft_market_results.csv"
# --------------------------------------------------------------------------

pl.seed_everything(SEED)


def load_data():
    df = pd.read_csv(MARKET_CSV, parse_dates=["Date"])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    cols = ["abnormal_return", "volume_zscore", "range_zscore", "daily_return"]
    df = df.dropna(subset=cols).reset_index(drop=True)   # early rolling-window NaNs
    df["abs_ar"] = df["abnormal_return"].abs()
    df["time_idx"] = df.groupby("ticker").cumcount()
    df["ticker"] = df["ticker"].astype(str)
    return df


def make_datasets(df, reals):
    max_t = df["time_idx"].max()
    test_start = int(max_t * (1 - TEST_FRAC))
    val_start = int(max_t * (1 - 2 * TEST_FRAC))

    training = TimeSeriesDataSet(
        df[df.time_idx < val_start],
        time_idx="time_idx",
        target="abs_ar",
        group_ids=["ticker"],
        max_encoder_length=ENC_LEN,
        max_prediction_length=1,
        static_categoricals=["ticker"],
        time_varying_unknown_reals=reals,
        target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )
    val_df = df[(df.time_idx >= val_start - ENC_LEN) & (df.time_idx < test_start)]
    test_df = df[df.time_idx >= test_start - ENC_LEN]
    val = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, test_df, predict=False, stop_randomization=True)
    return training, val, test, test_start


def run_tft(df, reals):
    training, val, test, test_start = make_datasets(df, reals)
    train_dl = training.to_dataloader(train=True, batch_size=BATCH, num_workers=0)
    val_dl = val.to_dataloader(train=False, batch_size=BATCH, num_workers=0)
    test_dl = test.to_dataloader(train=False, batch_size=BATCH, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=HIDDEN,
        attention_head_size=1,
        dropout=0.2,
        hidden_continuous_size=HIDDEN // 2,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=3,
    )
    print(f"TFT parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"train windows: {len(training):,}  val: {len(val):,}  test: {len(test):,}")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dl, val_dl)

    out = model.predict(test_dl, mode="prediction", return_y=True, return_index=True)
    pred = out.output.numpy().ravel()
    actual = out.y[0].numpy().ravel()
    idx = out.index.reset_index(drop=True)
    keep = (idx["time_idx"] >= test_start).values
    return idx[keep].reset_index(drop=True), pred[keep], actual[keep]


def baselines(df, idx):
    look = df.set_index(["ticker", "time_idx"])["abs_ar"]
    naive, roll5 = [], []
    for t, k in zip(idx["ticker"], idx["time_idx"]):
        hist = [look.get((t, k - j), np.nan) for j in range(1, 6)]
        naive.append(hist[0])
        roll5.append(np.nanmean(hist))
    return np.array(naive), np.array(roll5)


def metrics(actual, pred):
    m = ~(np.isnan(actual) | np.isnan(pred))
    err = actual[m] - pred[m]
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "n": int(m.sum())}


def main():
    df = load_data()
    print(f"{df['ticker'].nunique()} tickers, {len(df):,} rows")
    reals = ["abs_ar", "volume_zscore", "range_zscore", "daily_return"]

    idx, pred, actual = run_tft(df, reals)
    naive, roll5 = baselines(df, idx)

    res = pd.DataFrame({
        "Naive (yesterday)": metrics(actual, naive),
        "Rolling mean (5d)": metrics(actual, roll5),
        "TFT (market-only)": metrics(actual, pred),
    }).T
    print("\n=== Test results (same rows for every model) ===")
    print(res.round(5))

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    res.to_csv(RESULTS_CSV)
    print(f"\nSaved -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
