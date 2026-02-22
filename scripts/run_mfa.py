import os
import subprocess
from pathlib import Path

def run_mfa_alignment(mfa_env_path, data_dir, out_dir):
    mamba_bin = "/usr/local/bin/micromamba"
    mfa_env = Path(mfa_env_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    align_env = os.environ.copy()
    align_env["PATH"] = f"{mfa_env}/bin:{align_env['PATH']}"
    align_env["MPLBACKEND"] = "Agg"
    align_env["OMP_NUM_THREADS"] = "1"
    align_env["LD_LIBRARY_PATH"] = f"{mfa_env}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

    print("🚀 Запускаем MFA Align...")
    cmd = [
        str(mamba_bin), "run", "-p", str(mfa_env),
        "mfa", "align", str(data_dir), "russian_mfa", "russian_mfa", str(out_dir),
        "--clean", "--single_speaker", "-j", "1", "--output_format", "long_textgrid", "--verbose"
    ]
    
    proc = subprocess.Popen(cmd, env=align_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    if proc.returncode == 0:
        print("\n✅ Выравнивание завершено. TextGrids сгенерированы.")
    else:
        print(f"\n❌ Ошибка выравнивания. Код: {proc.returncode}")

if __name__ == "__main__":
    run_mfa_alignment(
        mfa_env_path="/content/mfa_runtime",
        data_dir="/content/mfa_data_16k",
        out_dir="/content/mfa_output"
    )
