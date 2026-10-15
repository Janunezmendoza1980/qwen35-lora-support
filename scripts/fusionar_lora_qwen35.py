
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE = "Qwen/Qwen3.5-0.8B"
ADAPTER = ROOT / "modelos" / "lora-qwen35"
OUTPUT = ROOT / "modelos" / "qwen35-lora-fusionado"

if OUTPUT.exists():
    raise FileExistsError(
        f"Ya existe {OUTPUT}. No se sobrescribirá."
    )

if not (ADAPTER / "adapter_config.json").is_file():
    raise FileNotFoundError(f"No encuentro el adaptador: {ADAPTER}")

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER,
    trust_remote_code=False,
)

print("Cargando modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    trust_remote_code=False,
    torch_dtype=torch.float32,
    use_safetensors=True,
    attn_implementation="eager",
)

print("Aplicando adaptador LoRA...")
model = PeftModel.from_pretrained(base_model, str(ADAPTER))

print("Fusionando los pesos...")
model = model.merge_and_unload()
model.eval()

print(f"Guardando copia fusionada en {OUTPUT}...")
model.save_pretrained(
    OUTPUT,
    safe_serialization=True,
    max_shard_size="2GB",
)
tokenizer.save_pretrained(OUTPUT)

print("Fusión terminada.")
print(f"Modelo fusionado: {OUTPUT}")
