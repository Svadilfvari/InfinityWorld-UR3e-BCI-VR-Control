import argparse
import pickle
import socket

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def bandpass_filter(data, lowcut=8.0, highcut=12.0, fs=250, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data, axis=0)


class ModelTheta(nn.Module):
    def __init__(self, input_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(128)
        self.bottleneck = nn.Linear(512 * 128, 128)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        x = x.view(x.size(0), -1)
        return self.bottleneck(x)


class ModelLambda(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.drop(x)
        return self.fc2(x)


def recv_pickled(sock: socket.socket):
    data = sock.recv(4096)
    if not data:
        return None
    return pickle.loads(data)


def transform(chunks: np.ndarray, fs: int = 250, seq_len: int = 248):
    # chunks expected shape (T, C)
    filtered = bandpass_filter(chunks, 8.0, 12.0, fs, order=5)

    if filtered.shape[0] >= seq_len:
        filtered = filtered[-seq_len:, :]
    else:
        pad = np.zeros((seq_len - filtered.shape[0], filtered.shape[1]), dtype=filtered.dtype)
        filtered = np.vstack([pad, filtered])

    # TODO: load the scaler used during training instead of fitting here
    scaler = StandardScaler()
    filtered = scaler.fit_transform(filtered)

    x = torch.tensor(filtered.T, dtype=torch.float32).unsqueeze(0)  # (1, C, T)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--listen-host", default="0.0.0.0")
    p.add_argument("--listen-port", type=int, default=12346)
    p.add_argument("--theta", default="python/models/theta.pth")
    p.add_argument("--lambda", dest="lambda_path", default="python/models/lambda.pth")
    p.add_argument("--fs", type=int, default=250)
    args = p.parse_args()

    device = "cpu"
    model_theta = ModelTheta().to(device)
    model_lambda = ModelLambda().to(device)
    model_theta.load_state_dict(torch.load(args.theta, map_location=device))
    model_lambda.load_state_dict(torch.load(args.lambda_path, map_location=device))
    model_theta.eval()
    model_lambda.eval()
    print("Models loaded.")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((args.listen_host, args.listen_port))
    server.listen(1)
    print("Listening for EEG sender on", (args.listen_host, args.listen_port))
    client, addr = server.accept()
    print("Connected:", addr)

    try:
        while True:
            channels_data = recv_pickled(client)
            if channels_data is None:
                print("Connection closed.")
                break

            arr = np.asarray(channels_data)
            if arr.ndim == 3:
                arr = arr[0]

            x = transform(arr, fs=args.fs)
            with torch.no_grad():
                latent = model_theta(x)
                logits = model_lambda(latent)
                pred = int(logits.argmax(dim=1).item())

            print("Predicted class:", pred)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        server.close()


if __name__ == "__main__":
    main()
