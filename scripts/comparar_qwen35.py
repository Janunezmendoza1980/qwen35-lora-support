
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARCHIVOS = {
    "Base": ROOT / "resultados" / "qwen35_base_test.jsonl",
    "LoRA": ROOT / "resultados" / "qwen35_lora_test.jsonl",
}


def cargar_resultados(ruta):
    with ruta.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def porcentaje(n, total):
    return 100 * n / total if total else 0.0


def analizar(nombre, ruta):
    filas = cargar_resultados(ruta)
    total = len(filas)

    json_validos = sum(
        r.get("valoracion", {}).get("json_valido") is True
        for r in filas
    )
    esquemas_validos = sum(
        r.get("valoracion", {}).get("esquema_valido") is True
        for r in filas
    )
    etiquetas_correctas = sum(
        r.get("valoracion", {}).get("ambas_correctas") is True
        for r in filas
    )

    latencias = [
        r["latencia_s"]
        for r in filas
        if isinstance(r.get("latencia_s"), (int, float))
    ]

    errores = sum(r.get("error") is not None for r in filas)

    print(f"\n=== {nombre} ===")
    print(f"Casos: {total}")
    print(f"JSON válido: {json_validos}/{total} "
          f"({porcentaje(json_validos, total):.1f}%)")
    print(f"Esquema válido: {esquemas_validos}/{total} "
          f"({porcentaje(esquemas_validos, total):.1f}%)")
    print(f"Ambas etiquetas correctas: {etiquetas_correctas}/{total} "
          f"({porcentaje(etiquetas_correctas, total):.1f}%)")
    print(f"Errores de generación: {errores}")

    if latencias:
        print(f"Latencia media: {statistics.mean(latencias):.2f} s")
        print(f"Latencia mediana: {statistics.median(latencias):.2f} s")

    return {r["id"]: r for r in filas}


resultados = {}
for nombre, ruta in ARCHIVOS.items():
    if not ruta.is_file():
        print(f"\nFalta el fichero de {nombre}: {ruta}")
        raise SystemExit(1)
    resultados[nombre] = analizar(nombre, ruta)

base = resultados["Base"]
lora = resultados["LoRA"]

ids_base = set(base)
ids_lora = set(lora)

if ids_base != ids_lora:
    print("\nAVISO: los dos resultados no contienen los mismos IDs.")
    print("Solo en base:", sorted(ids_base - ids_lora))
    print("Solo en LoRA:", sorted(ids_lora - ids_base))
else:
    cambios = []
    for caso_id in sorted(ids_base):
        b = base[caso_id].get("valoracion", {})
        l = lora[caso_id].get("valoracion", {})

        b_ok = b.get("ambas_correctas") is True
        l_ok = l.get("ambas_correctas") is True

        if b_ok != l_ok:
            cambios.append((caso_id, b_ok, l_ok))

    mejoran = sum(not b and l for _, b, l in cambios)
    empeoran = sum(b and not l for _, b, l in cambios)

    print("\n=== CAMBIOS ENTRE BASE Y LoRA ===")
    print(f"Casos que pasan de incorrectos a correctos: {mejoran}")
    print(f"Casos que pasan de correctos a incorrectos: {empeoran}")
    print(f"Casos cuyo resultado conjunto cambia: {len(cambios)}")

    for caso_id, b_ok, l_ok in cambios:
        print(
            f"{caso_id}: Base={'correcto' if b_ok else 'incorrecto'}; "
            f"LoRA={'correcto' if l_ok else 'incorrecto'}"
        )
