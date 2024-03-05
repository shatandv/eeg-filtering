DEFAULT_ACTION_ARTIFACT_FREQS: dict[str, dict[str, list]] = {
    "HEAD_LEFT": {
        "Fp1": [(0.1, 8)],
        "Fp2": [(0.1, 8)],
    },
    "STAND": {
        "T3": [(0.1, 8)],
        "T4": [(0.1, 8)],
        "Fp1": [(0.1, 8)],
        "Fp2": [(0.1, 8)],
        "O1": [(0.1, 8)],
        "O2": [(0.1, 8)],
    },
    "BLINK": {
        "O1": [(0.1, 8)],
        "O2": [(0.1, 8)],
    },
}