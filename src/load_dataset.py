import os
import soundfile as sf
from datasets import load_dataset

os.makedirs("audio", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)
max_samples = int(os.getenv("MAX_SAMPLES", "500"))

print("Carregando dataset...")
ds = load_dataset(
    "nilc-nlp/CORAA-MUPE-ASR",
    data_files={"test": "data/test-*.parquet"},
    split="test",
)

# Filtra qualidade alta e duração estilo WhatsApp
ds_filtered = ds.filter(lambda x:
    x["audio_quality"] == "high" and
    2.0 <= x["duration"] <= 30.0
)
total_available = len(ds_filtered)
print(f"Total disponível após filtros: {total_available}")

print("Salvando amostras...")
for i, sample in enumerate(ds_filtered):
    if i >= max_samples:
        break

    filename = f"sample_{i:04d}"

    sf.write(f"audio/{filename}.wav",
             sample["audio"]["array"],
             sample["audio"]["sampling_rate"])

    with open(f"transcripts/{filename}.txt", "w", encoding="utf-8") as f:
        f.write(sample["normalized_text"].strip())  # campo correto

    if i % 50 == 0:
        print(f"  [{i}] {filename} ({sample['duration']:.1f}s) — {sample['birth_state']}")

print(f"\n✅ Concluído!")