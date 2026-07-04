# LSTM Time Series — Reliance Stock Forecasting

**Marvellous Infosystems — OJT 2026**
**Trainee : Shubhada A. Palwe**

---

## About This Folder

This is the most advanced project in the deep learning portfolio. We use LSTM (Long Short-Term Memory) to forecast Reliance stock prices based on historical close price data.

LSTM is a special kind of RNN that can remember important information for a long time and forget irrelevant information. That makes it great for time series data like stock prices, weather, or sales.

File 48: `48_Reliance_LSTM_Time_Series.py` — 18 detailed steps from data loading to next-day prediction.

---

## Why LSTM and Not SimpleRNN?

In the rnn_fundamentals folder we used `SimpleRNN`. The problem with SimpleRNN is something called the **vanishing gradient problem** — when the sequence is long (like 10 or 50 days of stock prices), the gradient shrinks as it travels back through time and the early timesteps stop contributing to learning.

LSTM fixes this with a **cell state** — a separate memory channel that can carry information across many timesteps without it being lost.

---

## LSTM Internal Structure

LSTM has two states at each timestep:
- `h_t` — hidden state (short-term output)
- `C_t` — cell state (long-term memory)

Four gates control what gets remembered or forgotten:

```
Forget Gate   : f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)
                → what old memory to throw away

Input Gate    : i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)
                → what new info to store

Candidate     : C̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)
                → candidate new memory values

Cell State    : C_t = f_t × C_{t-1} + i_t × C̃_t
                → updated long-term memory

Output Gate   : o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)
                → what part of cell state to output

Hidden State  : h_t = o_t × tanh(C_t)
                → final output for this timestep
```

The file manually calculates all 6 of these for a single timestep (Step 8) so we can see the actual numbers.

---

## Dataset

```
marvellous_reliance_stock_sample.csv
```

Columns: Date, Open, High, Low, Close, Volume. We only use the **Close** price for this beginner-level project. In advanced projects, all columns + technical indicators can be used.

---

## Step-by-Step Walkthrough

### Step 1-2 — Load and Sort

```python
data = pd.read_csv(DATASET_PATH)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)
```

Converting date to datetime ensures proper chronological sorting. `reset_index(drop=True)` resets row numbers after sorting.

---

### Step 3 — Extract Close Price

```python
close_prices = data[["Close"]].values   # shape: (n_rows, 1)
```

Using double brackets `[["Close"]]` keeps it as a 2D array (n, 1) instead of a 1D Series. This matters for the scaler which expects 2D input.

---

### Step 4 — Min-Max Scaling

The file first shows the manual formula:

```
Scaled = (value - min) / (max - min)
```

Then uses sklearn:

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_prices)
```

We scale to [0, 1] so the LSTM doesn't struggle with large raw rupee values (like 2400, 2600). Neural networks learn much better when inputs are small numbers.

Note: we save `scaler` because we'll need it later to convert predictions back to rupees (inverse transform).

---

### Step 5 — Create Sequences (Sliding Window)

```python
TIME_STEPS = 10

for i in range(TIME_STEPS, len(dataset)):
    X.append(dataset[i-TIME_STEPS:i, 0])   # previous 10 days
    y.append(dataset[i, 0])                # next day
```

We use 10 days of history to predict day 11. This sliding window shifts one day at a time:
- X[0] = days 1–10, y[0] = day 11
- X[1] = days 2–11, y[1] = day 12
- ...and so on

---

### Step 6 — Reshape for LSTM

```python
X = X.reshape(X.shape[0], X.shape[1], 1)
# shape: (samples, 10 timesteps, 1 feature)
```

LSTM always needs 3D input: `(samples, timesteps, features)`. We have 1 feature (Close price), so the last dimension is 1.

---

### Step 9 — Train/Test Split

```python
train_size = int(len(X) * 0.80)
X_train = X[:train_size]
X_test  = X[train_size:]
```

We split chronologically — first 80% for training, last 20% for testing. We do NOT shuffle here. Shuffling would mix future data into training which would be unrealistic (and cheating).

---

### Step 10 — Model Architecture

```python
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(TIME_STEPS, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25, activation="relu"))
model.add(Dense(units=1))
```

**`return_sequences=True` on first LSTM:** because there's a second LSTM after it. The first LSTM must return its output at every timestep (shape: samples × timesteps × 50) so the second LSTM has a full sequence to work with.

**`return_sequences=False` on second LSTM:** it's the last LSTM. It returns only the final hidden state (shape: samples × 50). The Dense layers then process this.

**`Dropout(0.2)`:** randomly drops 20% of neurons during training. This prevents the model from memorizing the training data (overfitting). At prediction time, Dropout is automatically disabled.

**`Dense(25, relu)`:** learns a non-linear combination from the LSTM output.

**`Dense(1)`:** regression output — one number (next day's scaled close price). No activation because we want any real value, not squashed to (0,1).

Loss: `mean_squared_error` — same MSE we learned in loss_and_backprop folder.

---

### Step 11 — Training with EarlyStopping

```python
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=60, batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)
```

`EarlyStopping` watches the validation loss. If it doesn't improve for 10 epochs in a row, training stops automatically. `restore_best_weights=True` means the final model is the one that had the best validation loss, not the one from the last epoch.

`validation_split=0.2` takes 20% of training data for validation (to monitor overfitting). This is separate from the test set.

---

### Step 12-13 — Predict and Inverse Scale

```python
predicted_scaled = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_scaled)
actual_prices    = scaler.inverse_transform(y_test.reshape(-1, 1))
```

Predictions come out scaled (between 0 and 1). We use the same scaler from Step 4 to convert back to rupees:

```
Original = Scaled × (max - min) + min
```

`y_test.reshape(-1, 1)` converts the 1D array to 2D for the scaler.

---

### Step 14 — Error Metrics

```python
mae  = mean_absolute_error(actual_prices, predicted_prices)
mse  = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
```

- **MAE** — average rupee difference between predicted and actual. Easy to interpret.
- **MSE** — like MAE but squares errors, penalizes big mistakes more.
- **RMSE** — square root of MSE, brings units back to rupees. Most commonly reported.

---

### Step 17 — Next-Day Prediction

```python
last_10_days = scaled_close[-TIME_STEPS:]
last_10_days = last_10_days.reshape(1, TIME_STEPS, 1)
next_day_scaled = model.predict(last_10_days)
next_day_price = scaler.inverse_transform(next_day_scaled)
```

We take the last 10 days of actual data, reshape to match the LSTM input format, predict, then inverse-scale to get the actual rupee price for the next day.

---

### Step 18 — Save Model and Output

```python
model.save("marvellous_reliance_lstm_detailed_model.h5")
output_df.to_csv("marvellous_reliance_prediction_output.csv", index=False)
```

`.h5` is the Keras format for saving LSTM/Keras models. It saves the architecture + trained weights. joblib is used for sklearn models; `.h5` is used for Keras/TensorFlow models.

---

## Files Generated When Running

| File | Contents |
|------|----------|
| `marvellous_actual_vs_predicted_detailed.png` | Graph: actual vs predicted prices |
| `marvellous_training_loss_detailed.png` | Graph: training and validation loss per epoch |
| `marvellous_reliance_lstm_detailed_model.h5` | Saved Keras model |
| `marvellous_reliance_prediction_output.csv` | CSV with actual, predicted, and difference |

---

## How to Run

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn

# Put the dataset in the same folder:
# marvellous_reliance_stock_sample.csv

python 48_Reliance_LSTM_Time_Series.py
```

---

## SimpleRNN vs LSTM — Quick Comparison

| | SimpleRNN | LSTM |
|---|---|---|
| Memory | Short-term only | Short-term (h_t) + Long-term (C_t) |
| Vanishing gradient | Yes — problem on long sequences | Solved by cell state |
| Gates | None | 4 gates (Forget, Input, Candidate, Output) |
| Parameters | Fewer | More (heavier to train) |
| Best for | Short sequences | Long sequences, time series |

---

*Marvellous Infosystems — From raw stock prices to a trained LSTM that predicts tomorrow.*
