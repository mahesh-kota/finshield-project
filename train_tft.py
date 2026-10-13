"""
FinShield - TFT training, baselines, and social-signal ablation.

Question answered: does adding MuRIL-derived social features improve
next-day market-stress forecasting over price/volume alone?

Target  : abs_ret  = |log return| on the next trading day (stress proxy)
Models  : naive (yesterday), 5-day rolling mean, TFT (market only),
          TFT (market + social)
Metrics : MAE, RMSE on a chronological hold-out (last TEST_FRAC of days)

!! Edit the CONFIG block to match your real CSV paths / column names.
Written for pytorch-forecasting 1.x + lightning (you have 1.8.0).
CPU-friendly: ~tiny model, runs on an i5 / 8GB laptop.
"""
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

# ----------------------------- CONFIG ------------------------------------
PRICE_CSV = "data/prices.csv"            # cols: date, ticker, close, volume
SOCIAL_CSV = "data/daily_social.csv"     # cols: date, ticker, n_posts, susp_prob, urgency, sentiment
SOCIAL_COLS = ["n_posts", "susp_prob", "urgency", "sentiment"]
ENC_LEN = 20          # trading days of history fed to the encoder
TEST_FRAC = 0.2       # last 20% of days = test, the 20% before that = val
HIDDEN = 8            # keep small: 224 labelled posts, CPU training
EPOCHS = 30
BATCH = 64
SEED = 42
RESULTS_CSV = "results/tft_results.csv"
# --------------------------------------------------------------------------

pl.seed_everything(SEED)


def load_data():
    px = pd.read_csv(PRICE_CSV, parse_dates=["date"])
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = px.groupby("ticker")
    px["abs_ret"] = np.abs(np.log(px["close"] / g["close"].shift(1)))
    vol_mean = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    vol_std = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).std())
    px["volume_z"] = ((px["volume"] - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan)

    soc = pd.read_csv(SOCIAL_CSV, parse_dates=["date"])
    df = px.merge(soc, on=["date", "ticker"], how="left")
    df[SOCIAL_COLS] = df[SOCIAL_COLS].fillna(0.0)   # no posts that day -> 0

    df = df.dropna(subset=["abs_ret", "volume_z"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("ticker").cumcount()   # trading-day index
    df["ticker"] = df["ticker"].astype(str)
    return df


def make_datasets(df, reals):
    max_t = df["time_idx"].max()
    test_start = int(max_t * (1 - TEST_FRAC))
    val_start = int(max_t * (1 - 2 * TEST_FRAC))

    train_df = df[df.time_idx < val_start]
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="abs_ret",
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
    # include ENC_LEN days of history before each split so windows are full
    val_df = df[df.time_idx >= val_start - ENC_LEN]
    val_df = val_df[val_df.time_idx < test_start]
    test_df = df[df.time_idx >= test_start - ENC_LEN]

    val = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, test_df, predict=False, stop_randomization=True)
    # only score decoder steps that fall inside the test window
    return training, val, test, test_start


def run_tft(df, reals, name):
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
    print(f"[{name}] parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,   # don't write big weight files into the repo
    )
    trainer.fit(model, train_dl, val_dl)

    out = model.predict(test_dl, mode="prediction", return_y=True, return_index=True)
    pred = out.output.numpy().ravel()
    actual = out.y[0].numpy().ravel()
    idx = out.index.reset_index(drop=True)
    keep = (idx["time_idx"] >= test_start).values
    return idx[keep], pred[keep], actual[keep]


def baselines(df, idx):
    look = df.set_index(["ticker", "time_idx"])["abs_ret"]
    naive, roll5 = [], []
    for t, k in zip(idx["ticker"], idx["time_idx"]):
        hist = [look.get((t, k - j), np.nan) for j in range(1, 6)]
        naive.append(hist[0])
        roll5.append(np.nanmean(hist))
    return np.array(naive), np.array(roll5)


def metrics(actual, pred):
    m = ~(np.isnan(actual) | np.isnan(pred))
    err = actual[m] - pred[m]
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))), "n": int(m.sum())}


def main():
    df = load_data()
    market = ["abs_ret", "volume_z"]

    idx_m, pred_m, act_m = run_tft(df, market, "TFT market-only")
    idx_s, pred_s, act_s = run_tft(df, market + SOCIAL_COLS, "TFT market+social")

    # Score every model on the SAME rows (intersection of both TFT test indexes)
    key = lambda i: set(zip(i["ticker"], i["time_idx"]))
    common = key(idx_m) & key(idx_s)

    def subset(idx, pred, act):
        mask = np.array([(t, k) in common for t, k in zip(idx["ticker"], idx["time_idx"])])
        return idx[mask].reset_index(drop=True), pred[mask], act[mask]

    idx_m, pred_m, act_m = subset(idx_m, pred_m, act_m)
    idx_s, pred_s, act_s = subset(idx_s, pred_s, act_s)
    naive, roll5 = baselines(df, idx_m)

    rows = {
        "Naive (yesterday)": metrics(act_m, naive),
        "Rolling mean (5d)": metrics(act_m, roll5),
        "TFT market-only": metrics(act_m, pred_m),
        "TFT market+social": metrics(act_s, pred_s),
    }
    res = pd.DataFrame(rows).T
    print("\n=== Test results (identical rows for every model) ===")
    print(res.round(5))

    import os
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    res.to_csv(RESULTS_CSV)
    print(f"\nSaved -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
