import random
from pathlib import Path

def create_manifests(mfa_output_dir, manifest_dir, val_split=0.05, seed=42):
    mfa_dir = Path(mfa_output_dir)
    out_dir = Path(manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Сканируем {mfa_dir}...")
    tg_files = list(mfa_dir.glob("*.TextGrid"))
    file_ids = [f.stem for f in tg_files]

    if not file_ids:
        raise RuntimeError("❌ Папка MFA пуста! Нет файлов для манифеста.")

    random.seed(seed)
    random.shuffle(file_ids)

    val_size = max(1, int(len(file_ids) * val_split))
    train_ids = file_ids[:-val_size]
    val_ids = file_ids[-val_size:]

    with open(out_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(train_ids)))

    with open(out_dir / "val.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(val_ids)))

    print("✅ Манифесты созданы!")
    print(f"Трэйн: {len(train_ids)} шт. -> {out_dir}/train.txt")
    print(f"Валидация: {len(val_ids)} шт. -> {out_dir}/val.txt")

if __name__ == "__main__":
    create_manifests(
        mfa_output_dir="/content/mfa_output",
        manifest_dir="/content/fs2_ruslan/artifacts/manifests"
    )
