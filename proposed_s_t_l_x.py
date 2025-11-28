import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


FILE_NAME = 'ETTm1_LULL.csv'
TARGET_COLUMN = -1  # -1이면 마지막 컬럼을 타겟으로 사용. 이름이 있다면 'OT' 처럼 문자열로 지정
SEQ_LENGTH = 24     # 과거 24시점을 보고 다음 1시점을 예측
PREDICT_HORIZON = 1 # 1시점 뒤 예측
HIDDEN_DIM = 64
LAYER_DIM = 2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
STL_PERIOD = 24     # 데이터 주기에 맞춰 설정 (시간 단위면 24, 10분 단위면 144 등)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, FILE_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, 'lstm_runs', 'stl_attention_lstm')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device 설정
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# 데이터 로드 및 STL 분해 함수------------------------------------------------------
def load_and_process_data(filename: str):
    
    df = pd.read_csv(filename)

    # 날짜 컬럼 처리(첫번째 열이 시간 인덱스)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    # 타겟 데이터 추출
    if isinstance(TARGET_COLUMN, int):
        # 타겟이 정수(인덱스 번호)로 주어졌을 때
        data = df.iloc[:, TARGET_COLUMN].values
        col_name = df.columns[TARGET_COLUMN]
    else:
        # 타겟이 문자열(컬럼 이름)로 주어졌을 때
        data = df[TARGET_COLUMN].values
        col_name = TARGET_COLUMN

    print(f"Target Column: {col_name}, Data Shape: {data.shape}")

    # STL 분해
    series = pd.Series(data, index=df.index)
    stl = STL(series, period=STL_PERIOD, robust=True)
    result = stl.fit()

    trend = result.trend.values
    seasonal = result.seasonal.values
    resid = result.resid.values

    # STL 분해 결과 시각화
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(series.index, series.values, color='black', linewidth=1.0)
    axes[0].set_ylabel('Original', fontsize=10)
    axes[0].set_title(f'STL Decomposition of {col_name}')

    axes[1].plot(series.index, trend, color='tab:blue', linewidth=1.0)
    axes[1].set_ylabel('Trend', fontsize=10)

    axes[2].plot(series.index, seasonal, color='tab:green', linewidth=1.0)
    axes[2].set_ylabel('Seasonal', fontsize=10)

    axes[3].plot(series.index, resid, color='tab:orange', linewidth=1.0)
    axes[3].set_ylabel('Residual', fontsize=10)
    axes[3].set_xlabel('Timestamp', fontsize=10)

    fig.tight_layout()
    stl_plot_path = os.path.join(OUTPUT_DIR, 'stl_decomposition.png')
    fig.savefig(stl_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # Feature 결합: [Trend, Seasonal, Resid] -> LSTM의 Input Feature는 3개
    features = np.stack([data, trend, seasonal, resid], axis=1) 
    return features, data
#---------------------------------------------------------------------------

# 데이터 실행
features, original_data = load_and_process_data(DATA_PATH)


# 데이터셋 생성 (Sliding Window)------------------------------------------------------

# 입력 데이터 스케일링
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 타겟(정답) 스케일링
target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(original_data.reshape(-1, 1))

# Sliding window로 (입력 시퀀스, 예측 타겟)을 생성하는 함수
def create_sequences(X, y, seq_length, horizon):
    xs, ys = [], []
    max_index = len(X) - seq_length - horizon + 1
    for i in range(max_index):
        x_window = X[i:(i + seq_length)]
        y_window = y[i + seq_length + horizon - 1]
        xs.append(x_window)
        ys.append(y_window)
    return np.array(xs), np.array(ys)

#---------------------------------------------------------------------------

X_seq, y_seq = create_sequences(features_scaled, target_scaled, SEQ_LENGTH, PREDICT_HORIZON)

# Train/Val/Test Split
total_samples = len(X_seq)
train_size = int(total_samples * TRAIN_RATIO)
val_size = int(total_samples * VAL_RATIO)
test_size = total_samples - train_size - val_size

if val_size == 0:
    raise ValueError("VAL_RATIO 설정으로 Validation 세트가 비었습니다. 비율을 조정하세요.")
if test_size <= 0:
    raise ValueError("Train/Val 비율이 너무 높습니다. TEST 세트를 위한 비율을 조정하세요.")

train_end = train_size
val_end = train_end + val_size

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

# Tensor 변환
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

torch_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch_generator,
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 정의 ------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):

        # lstm_output shape: [batch_size, seq_len, hidden_dim]
        attn_weights = torch.tanh(self.attention(lstm_output)) # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)      # Softmax over seq_len
        
        # Context vector (Weighted Sum)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1) # [batch, hidden_dim]
        return context_vector, attn_weights

class STL_Attention_LSTM(nn.Module):
    """STL 3채널 입력 → LSTM → Attention → Dense로 구성된 예측 모델."""
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(STL_Attention_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Attention Layer
        self.attention = Attention(hidden_dim)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim] (Input dim is 3 from STL)
        
        # LSTM Forward
        out, (hn, cn) = self.lstm(x) # out: [batch, seq_len, hidden_dim]
        
        # Attention Mechanism
        context_vector, attn_weights = self.attention(out)
        
        # Final Prediction
        out = self.fc(context_vector)
        return out, attn_weights
#---------------------------------------------------------------------------

# 모델 초기화
model = STL_Attention_LSTM(input_dim=4, hidden_dim=HIDDEN_DIM, layer_dim=LAYER_DIM, output_dim=PREDICT_HORIZON)
model = model.to(DEVICE)

# 학습 ------------------------------------------------------

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStart Training...")
train_loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_loss_history.append(avg_loss)

    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            val_outputs, _ = model(X_batch)
            val_loss = criterion(val_outputs, y_batch)
            val_epoch_loss += val_loss.item()
    avg_val_loss = val_epoch_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Training Loss', color='tab:red')
plt.plot(val_loss_history, label='Validation Loss', color='tab:blue')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
loss_curve_path = os.path.join(OUTPUT_DIR, 'training_loss_curve.png')
plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#---------------------------------------------------------------------------

# 평가 및 Attention 시각화 ------------------------------------------------------

model.eval()
predictions = []
actuals = []
attention_maps = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        preds, attn_weights = model(X_batch)
        predictions.append(preds.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())
        attention_maps.append(attn_weights.cpu().numpy())  # 나중에 시각화용

# 역스케일링 (원래 값으로 변환)
predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)
attention_matrix = np.concatenate(attention_maps, axis=0).squeeze(-1)

predictions = target_scaler.inverse_transform(predictions).flatten()
actuals = target_scaler.inverse_transform(actuals).flatten()

# 성능 지표 계산
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
print(f"\nTest MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

# -------------------------------------------------------
# [논문용 그림 1] 예측 결과 비교 (전체 구간 중 일부만 확대)
# -------------------------------------------------------
plt.figure(figsize=(15, 6))
subset_len = min(200, len(actuals))
subset_start = max(0, len(actuals) - subset_len)
time_axis = np.arange(subset_start, subset_start + subset_len)
plt.plot(time_axis, actuals[subset_start:subset_start + subset_len], label='Actual Wave Height', color='black', alpha=0.8)
plt.plot(time_axis, predictions[subset_start:subset_start + subset_len], label='Predicted (STL-Attn-LSTM)', color='red', linestyle='--', linewidth=1.5)
plt.title(f'Prediction Result (Last {subset_len} samples of Test Set)')
plt.xlabel('Test Sample Index')
plt.ylabel('Wave Height')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
prediction_plot_path = os.path.join(OUTPUT_DIR, 'test_prediction_example.png')
plt.savefig(prediction_plot_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# -------------------------------------------------------
# [논문용 그림 2] Attention Map 시각화 (모델의 해석)
# -------------------------------------------------------
attn_samples = min(200, attention_matrix.shape[0])
attn_to_plot = attention_matrix[:attn_samples]
plt.figure(figsize=(12, 6))
im = plt.imshow(attn_to_plot, aspect='auto', cmap='magma')
plt.colorbar(im, label='Attention Weight')
plt.xticks(ticks=np.arange(SEQ_LENGTH), labels=np.arange(SEQ_LENGTH))
plt.yticks(ticks=np.arange(attn_samples), labels=np.arange(attn_samples))
plt.title('Temporal Attention Weights Across Test Samples')
plt.xlabel('Time Step within Input Window (0 = Oldest)')
plt.ylabel('Test Sample Index')
attention_heatmap_path = os.path.join(OUTPUT_DIR, 'attention_weight_heatmap.png')
plt.savefig(attention_heatmap_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("\n[Analysis Tip] 밝은 영역일수록 해당 시점의 정보가 최종 예측에 강하게 반영되었음을 뜻합니다.")
print(f"시각화 및 로그 파일은 '{OUTPUT_DIR}' 경로에 저장되었습니다.")