"""Record the software/hardware environment to results/environment.txt."""
import os, platform, sys
import torch, transformers
from importlib.metadata import version
from common import MODEL_TYPE
from bert_score.utils import model2layers

os.makedirs("results", exist_ok=True)
lines = [
    f"Python: {sys.version.split()[0]}",
    f"OS: {platform.platform()}",
    f"Processor: {platform.processor() or platform.machine()}",
    f"CPU cores: {os.cpu_count()}",
    f"torch: {torch.__version__}",
    f"transformers: {transformers.__version__}",
    f"bert-score (pip): {version('bert-score')}",
    f"CUDA available: {torch.cuda.is_available()}",
    f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU only)'}",
    f"BERTScore model: {MODEL_TYPE} (layer {model2layers.get(MODEL_TYPE, '?')}), idf=False, no baseline rescaling",
]
text = "\n".join(lines)
open("results/environment.txt", "w").write(text + "\n")
print(text)
