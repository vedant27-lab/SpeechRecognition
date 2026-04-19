import torchaudio
import os

folder = 'audio'
audio_extensions = ('.mp3', '.wav', '.flac')

data = []

text_file = [
    "hello world",
    "how are you",
    "i am fine",
    "open chrome",
    "play music",
    "stop music",
    "what time is it",
    "close the window",
    "turn on wifi",
    "turn off wifi"
]

for i, filename in enumerate(os.listdir(folder)):
    if filename.lower().endswith(audio_extensions):
        path = os.path.join(folder, filename)
        waveform, sampleRate = torchaudio.load(path)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
        )(waveform)

        log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        log_mel = log_mel.squeeze(0)
        log_mel = log_mel.transpose(0, 1)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
        text = text_file[i]
        data.append((log_mel, text))


#print(waveform.shape[0])

##print(log_mel.shape)   #This outputs 3 dimensional tuple with dimension 0 --> no. of audio channels (mono, sterio)
                       # dimension 1 --> n_mels = 80
                       # dimension 2 --> time frames


import string

chars = list(string.ascii_lowercase + " '")
char2idx = {c: i+1 for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

def encode(text):
    return [char2idx[c] for c in text.lower() if c in char2idx]

import torch
import torch.nn as nn

class SimpleASR(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, vocab_size=30):
        super().__init__()

        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=2
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        x = x.permute(1, 0, 2)

        x = self.transformer(x)

        x = self.fc(x)
        return x

model = SimpleASR()

features = log_mel.unsqueeze(0)

output = model(features)

def decode(output):
    pred = torch.argmax(output, dim=-1)
    pred = pred.squeeze(1).cpu().numpy()

    text = ""
    prev = -1
    for p in pred:
        if p != prev and p !=  0:
            text += idx2char.get(p, "")
        prev = p

    return text

predicted_text = decode(output)
#print(predicted_text)

text = "hello world"
target = torch.tensor(encode(text))


#input_length = torch.tensor([output.size(0)])
#target_length = torch.tensor([len(target)])
#
#target = target.unsqueeze(0)
#
#ctc_loss = nn.CTCLoss(blank=0)
#
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



ctc_loss = nn.CTCLoss(blank=0)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(400):
    model.train()
    total_loss = 0
    for features, text in data:
        features = features.unsqueeze(0)
        target = torch.tensor(encode(text))

        input_length = torch.tensor([features.size(1)])
        target_length = torch.tensor([len(target)])

        target = target.unsqueeze(0)
        optimizer.zero_grad()
        output = model(features)

        loss = ctc_loss(
            output.log_softmax(2),
            target,
            input_length,
            target_length
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss}")

    model.eval()

test_features, _ = data[3]
test_features = test_features.unsqueeze(0)

output = model(test_features)

print("Prediction:", decode(output))