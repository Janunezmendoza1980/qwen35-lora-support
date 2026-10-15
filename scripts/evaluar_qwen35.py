
# Evaluación independiente de Qwen3.5-0.8B con y sin adaptador LoRA.
# Reutiliza los prompts y el validador del proyecto sin modificar los archivos originales.

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from curso.common import (
    ROOT,
    GENERATION,
    assert_frozen,
    read_jsonl,
    messages,
    validate_answer,
    write_jsonl,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def load_model(adapter_path=None):
    """Carga siempre Qwen3.5-0.8B; aplica LoRA solo si se solicita."""
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=False,
        token=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=False,
        token=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer, device


def generate_answer(model, tokenizer, device, incident):
    """Genera una respuesta sin reparar ni transformar el texto generado."""
    prompt_messages = messages(incident, fewshot=False)

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    generation_params = dict(GENERATION)
    generation_params["do_sample"] = False

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **generation_params,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency = time.perf_counter() - start
    new_tokens = output[0, inputs["input_ids"].shape[1]:]

    return {
        "respuesta_original": tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ),
        "latencia_s": latency,
        "tokens_entrada": int(inputs["input_ids"].shape[1]),
        "tokens_salida": int(len(new_tokens)),
        "coste_usd": None,
        "coste_metodo": "Inferencia local; consumo energético no medido",
        "hardware": device,
        "model": MODEL_ID,
        "revision": None,
        "parametros": generation_params,
        "enable_thinking": False,
        "fewshot": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa Qwen3.5-0.8B base o Qwen3.5-0.8B + LoRA."
    )
    parser.add_argument(
        "--condicion",
        choices=("qwen35_base", "qwen35_lora"),
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=("desarrollo", "validacion", "test"),
        default="test",
    )
    parser.add_argument(
        "--adaptador",
        default=str(ROOT / "modelos" / "lora-qwen35"),
    )
    parser.add_argument("--salida", required=True)
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Evalúa solo los primeros N casos; útil para una prueba inicial.",
    )
    args = parser.parse_args()

    if args.limite is not None and args.limite < 1:
        parser.error("--limite debe ser un entero mayor que cero.")

    output_path = Path(args.salida)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if output_path.exists():
        parser.error(
            f"El archivo {output_path} ya existe. Elige otro nombre "
            "para no sobrescribir resultados anteriores."
        )

    adapter_path = Path(args.adaptador)
    if not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path

    if args.condicion == "qwen35_lora":
        required_files = ("adapter_config.json", "adapter_model.safetensors")
        missing = [
            name for name in required_files
            if not (adapter_path / name).is_file()
        ]
        if missing:
            parser.error(
                f"El adaptador {adapter_path} no está completo. "
                f"Faltan: {', '.join(missing)}"
            )
    else:
        adapter_path = None

    # Verifica la integridad del experimento/datos originales congelados.

    # Comprueba los hashes de los datos y prompts congelados,
    # pero permite evaluar un modelo base diferente.
    frozen_manifest = {
        "modelo_base": MODEL_ID,
        "revision": None,
        "generation": dict(GENERATION),
        "enable_thinking": False,
        "files": {},
    }

    original_manifest_path = ROOT / "config" / "experimento-congelado.json"
    original_manifest = json.loads(
        original_manifest_path.read_text(encoding="utf-8")
    )

    for relative_path, expected_hash in original_manifest["files"].items():
        actual_hash = __import__("hashlib").sha256(
            (ROOT / relative_path).read_bytes()
        ).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                f"Archivo congelado modificado: {relative_path}"
            )

        frozen_manifest["files"][relative_path] = actual_hash


    data_path = ROOT / "datos" / f"{args.split}.jsonl"
    rows = read_jsonl(data_path)

    if args.limite is not None:
        rows = rows[:args.limite]

    if not rows:
        parser.error(f"No hay registros para evaluar en {data_path}")

    print(f"Modelo: {MODEL_ID}")
    print(f"Condición: {args.condicion}")
    print(f"Partición: {args.split}")
    print(f"Casos: {len(rows)}")
    print(f"Dispositivo: CPU")
    print("Cargando modelo; esta fase no se incluye en las latencias...")

    model, tokenizer, device = load_model(adapter_path)

    results = []
    total = len(rows)

    for index, row in enumerate(rows, start=1):
        try:
            generated = generate_answer(
                model,
                tokenizer,
                device,
                row["incidencia"],
            )
            error = None
        except Exception as exc:
            generated = {
                "respuesta_original": None,
                "latencia_s": None,
                "tokens_entrada": None,
                "tokens_salida": None,
                "coste_usd": None,
                "hardware": device,
                "model": MODEL_ID,
                "parametros": dict(GENERATION),
            }
            error = type(exc).__name__
            print(
                f"Advertencia: error en {row.get('id', index)}: {error}",
                flush=True,
            )

        result = {
            "id": row["id"],
            "condicion": args.condicion,
            "split": args.split,
            "incidencia": row["incidencia"],
            "objetivo": row["objetivo"],
            "fecha": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "modelo_base": MODEL_ID,
            "adaptador": str(adapter_path) if adapter_path else None,
            "manifiesto_experimento_original": frozen_manifest,
            **generated,
            "error": error,
            "valoracion": validate_answer(
                generated.get("respuesta_original"),
                row["objetivo"],
            ),
        }

        results.append(result)

        # Conserva los resultados completados incluso si el proceso se interrumpe.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_path, results)

        rating = result["valoracion"]
        print(
            f"[{index}/{total}] {row['id']} | "
            f"JSON={rating['json_valido']} | "
            f"Esquema={rating['esquema_valido']} | "
            f"Ambas etiquetas={rating['ambas_correctas']} | "
            f"Error={error}",
            flush=True,
        )

    print("\nEvaluación terminada.")
    print(f"Resultados guardados en: {output_path}")
    print(f"Casos registrados: {len(results)}")
    print(f"Errores de generación: {sum(r['error'] is not None for r in results)}")


if __name__ == "__main__":
    main()
