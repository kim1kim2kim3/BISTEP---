import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# 1. 설정 (Configuration)
# ==========================================
FILE_NAME = 'ETTm1_LULL.csv'
TARGET_COLUMN = -1  # -1이면 마지막 컬럼을 타겟으로 사용. 이름이 있다면 문자열 지정
SEQ_LENGTH = 24     # 과거 24시점을 보고 다음 1시점을 예측
PREDICT_HORIZON = 1 # 1시점 뒤 예측
HIDDEN_DIM = 64
LAYER_DIM = 2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, FILE_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, 'lstm_runs', 'vanilla_lstm')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")


def set_seed(seed: int = SEED) -> None:
    """재현성을 위해 모든 주요 난수를 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ==========================================
# 2. 데이터 로드 (단일 채널)
# ==========================================
def load_target_series(filename: str):
    """CSV를 읽어 단일 채널 시계열(타겟)만 반환."""
    df = pd.read_csv(filename)

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    if isinstance(TARGET_COLUMN, int):
        series = df.iloc[:, TARGET_COLUMN].values.astype(np.float32)
        col_name = df.columns[TARGET_COLUMN]
    else:
        series = df[TARGET_COLUMN].values.astype(np.float32)
        col_name = TARGET_COLUMN

    print(f"Target Column: {col_name}, Data Shape: {series.shape}")
    return series.reshape(-1, 1)


target_series = load_target_series(DATA_PATH)


# ==========================================
# 3. 데이터셋 생성 (Sliding Window)
# ==========================================
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(target_series)


def create_sequences(data, seq_length, horizon):
    """Sliding window로 (입력 시퀀스, 예측 타겟)을 생성."""
    xs, ys = [], []
    max_index = len(data) - seq_length - horizon + 1
    for i in range(max_index):
        x_window = data[i:(i + seq_length)]
        y_window = data[i + seq_length + horizon - 1]
        xs.append(x_window)
        ys.append(y_window)
    return np.array(xs), np.array(ys)


X_seq, y_seq = create_sequences(scaled_series, SEQ_LENGTH, PREDICT_HORIZON)

train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.FloatTensor(y_train),
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.FloatTensor(y_test),
)

torch_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch_generator,
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 4. 모델 정의: Vanilla LSTM
# ==========================================
class VanillaLSTM(nn.Module):
    """STL/어텐션 없이 단일 LSTM 스택만 사용하는 베이스라인 모델."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # 마지막 타임스텝만 사용
        out = self.fc(last_hidden)
        return out


model = VanillaLSTM(
    input_dim=1,
    hidden_dim=HIDDEN_DIM,
    layer_dim=LAYER_DIM,
    output_dim=PREDICT_HORIZON,
).to(DEVICE)


# ==========================================
# 5. 학습 (Training)
# ==========================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStart Training...")
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss', color='tab:blue')
plt.title('Training Loss Curve (Vanilla LSTM)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
loss_curve_path = os.path.join(OUTPUT_DIR, 'training_loss_curve.png')
plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# ==========================================
# 6. 평가 (Evaluation)
# ==========================================
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch)
        predictions.append(preds.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

predictions = scaler.inverse_transform(predictions).flatten()
actuals = scaler.inverse_transform(actuals).flatten()

mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
print(f"\nTest MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")


# ==========================================
# 7. 시각화
# ==========================================
plt.figure(figsize=(15, 6))
subset_len = min(200, len(actuals))
subset_start = max(0, len(actuals) - subset_len)
time_axis = np.arange(subset_start, subset_start + subset_len)
plt.plot(
    time_axis,
    actuals[subset_start:subset_start + subset_len],
    label='Actual',
    color='black',
    alpha=0.8,
)
plt.plot(
    time_axis,
    predictions[subset_start:subset_start + subset_len],
    label='Predicted (Vanilla LSTM)',
    color='red',
    linestyle='--',
    linewidth=1.5,
)
plt.title(f'Prediction Result (Last {subset_len} samples of Test Set)')
plt.xlabel('Test Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
prediction_plot_path = os.path.join(OUTPUT_DIR, 'test_prediction_example.png')
plt.savefig(prediction_plot_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"\n시각화 및 로그 파일은 '{OUTPUT_DIR}' 경로에 저장되었습니다.")

