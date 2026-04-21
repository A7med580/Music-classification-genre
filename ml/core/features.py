"""
Feature extraction module for Music Genre Classification.

Provides functions to extract audio features from raw audio files using librosa.
Also contains utilities for working with pre-extracted CSV features.
"""

import numpy as np

# Only import librosa if available (not needed for CSV-only workflows)
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# Feature group definitions for analysis
FEATURE_GROUPS = {
    'MFCC': {
        'columns': [f'mfcc{i}_{stat}' for i in range(1, 21) for stat in ['mean', 'var']],
        'description': 'Mel-Frequency Cepstral Coefficients (timbral texture)',
        'reason_use': (
            'MFCCs capture the spectral envelope of audio signals, encoding timbral '
            'characteristics that distinguish genres. Rock/metal have harsh timbres '
            '(high-energy upper MFCCs), while jazz/classical have smoother profiles.'
        ),
    },
    'Spectral Centroid': {
        'columns': ['spectral_centroid_mean', 'spectral_centroid_var'],
        'description': 'Center of mass of the spectrum (brightness)',
        'reason_use': (
            'Spectral centroid measures perceived "brightness." Metal and rock have '
            'high centroids due to distorted guitars, while blues and jazz have lower '
            'centroids from warm, bass-heavy instruments.'
        ),
    },
    'Zero Crossing Rate': {
        'columns': ['zero_crossing_rate_mean', 'zero_crossing_rate_var'],
        'description': 'Rate of sign changes in audio signal (noisiness)',
        'reason_use': (
            'ZCR distinguishes percussive, noisy signals (high ZCR in metal/rock) from '
            'smoother, tonal signals (low ZCR in classical/jazz). It is a simple but '
            'effective proxy for signal noisiness.'
        ),
    },
    'Chroma STFT': {
        'columns': ['chroma_stft_mean', 'chroma_stft_var'],
        'description': 'Pitch class energy distribution',
        'reason_use': 'Captures harmonic content and key signatures typical of genres.',
    },
    'Spectral Bandwidth': {
        'columns': ['spectral_bandwidth_mean', 'spectral_bandwidth_var'],
        'description': 'Width of the spectral band',
        'reason_use': 'Indicates spectral spread; genres with complex instrumentation have wider bandwidth.',
    },
    'Spectral Rolloff': {
        'columns': ['rolloff_mean', 'rolloff_var'],
        'description': 'Frequency below which 85% of energy lies',
        'reason_use': 'Distinguishes bright vs dark timbres across genres.',
    },
    'Tempo': {
        'columns': ['tempo'],
        'description': 'Estimated beats per minute',
        'reason_exclude': (
            'Tempo overlaps heavily across genres. A jazz piece can be 60 BPM or 200 BPM. '
            'Disco and metal can share similar tempos. The feature has low discriminative '
            'power and high intra-class variance.'
        ),
    },
    'Harmony/Perceptr': {
        'columns': ['harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var'],
        'description': 'Harmonic and percussive decomposition',
        'reason_exclude': (
            'These are redundant with MFCCs, which already capture harmonic content '
            'through the cepstral representation. Including both adds multicollinearity '
            'without improving classification accuracy.'
        ),
    },
    'RMS Energy': {
        'columns': ['rms_mean', 'rms_var'],
        'description': 'Root Mean Square energy (loudness)',
        'reason_exclude': (
            'RMS energy is highly dependent on recording conditions, mastering levels, '
            'and playback normalization. A quietly recorded metal track may have lower '
            'RMS than a loudly mastered pop track, making this feature unreliable.'
        ),
    },
}

# Features to USE (top 3 groups)
FEATURES_TO_USE = ['MFCC', 'Spectral Centroid', 'Zero Crossing Rate']

# Features to EXCLUDE (top 3 groups)
FEATURES_TO_EXCLUDE = ['Tempo', 'Harmony/Perceptr', 'RMS Energy']


def extract_features_from_audio(file_path, sr=22050, duration=3, offset=0.5):
    """Extract a feature vector from a raw audio file.

    Extracts the same 57 features used in features_3_sec.csv:
    chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff,
    zero_crossing_rate, harmony, perceptr, tempo, and 20 MFCCs.

    Args:
        file_path: Path to the audio file (.wav, .mp3, etc.).
        sr: Sampling rate for loading audio.
        duration: Duration in seconds to load.
        offset: Offset in seconds from start of file.

    Returns:
        numpy array of shape (57,) containing extracted features.

    Raises:
        ImportError: If librosa is not installed.
        FileNotFoundError: If the audio file doesn't exist.
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for audio feature extraction. "
                          "Install it with: pip install librosa")

    y, sr = librosa.load(file_path, duration=duration, offset=offset, sr=sr)

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(cent)
    cent_var = np.var(cent)

    # Spectral Bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bw_mean = np.mean(bw)
    bw_var = np.var(bw)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_var = np.var(zcr)

    # Harmonic and Percussive components
    harmony, perceptr = librosa.effects.hpss(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    perceptr_mean = np.mean(perceptr)
    perceptr_var = np.var(perceptr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0] if len(tempo) > 0 else 0

    # MFCCs (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_feat = []
    for e in mfcc:
        mfcc_feat.append(np.mean(e))
        mfcc_feat.append(np.var(e))

    features = [
        chroma_stft_mean, chroma_stft_var,
        rms_mean, rms_var,
        cent_mean, cent_var,
        bw_mean, bw_var,
        rolloff_mean, rolloff_var,
        zcr_mean, zcr_var,
        harmony_mean, harmony_var,
        perceptr_mean, perceptr_var,
        tempo
    ] + mfcc_feat

    return np.array(features)


def get_feature_names():
    """Get the ordered list of feature column names matching the CSV.

    Returns:
        List of 57 feature name strings.
    """
    names = [
        'chroma_stft_mean', 'chroma_stft_var',
        'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'rolloff_mean', 'rolloff_var',
        'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var',
        'perceptr_mean', 'perceptr_var',
        'tempo'
    ]
    for i in range(1, 21):
        names.append(f'mfcc{i}_mean')
        names.append(f'mfcc{i}_var')
    return names
