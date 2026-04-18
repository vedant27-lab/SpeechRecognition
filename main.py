import torchaudio

waveform, sampleRate = torchaudio.load("sample.wav")

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=80
)(waveform)

log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

print(log_mel.shape)

