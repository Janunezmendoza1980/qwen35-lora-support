import sys
from pathlib import Path

sys.path.insert(0, r".\herramientas\llama.cpp\gguf-py")
from gguf import GGUFReader

folder = Path(r"C:\Users\janun\.lmstudio\models\unsloth\Qwen3.5-0.8B-GGUF")
original = GGUFReader(str(folder / "Qwen3.5-0.8B-Q8_0.gguf"))
fusionado = GGUFReader(str(folder / "qwen35-lora-fusionado-f16.gguf"))

a = {t.name: tuple(t.shape) for t in original.tensors}
b = {t.name: tuple(t.shape) for t in fusionado.tensors}

print("Tensores originales:", len(a))
print("Tensores fusionados:", len(b))
print("Faltan en el fusionado:", sorted(set(a) - set(b)))
print("Sobran en el fusionado:", sorted(set(b) - set(a)))

diferentes = [
    (nombre, a[nombre], b[nombre])
    for nombre in sorted(set(a) & set(b))
    if a[nombre] != b[nombre]
]
print("Tensores con dimensiones diferentes:", len(diferentes))
for elemento in diferentes[:20]:
    print(" ", elemento)

print("\nNúmero de metadatos originales:", len(original.fields))
print("Número de metadatos fusionados:", len(fusionado.fields))

for key in sorted(set(original.fields) | set(fusionado.fields)):
    if key.startswith(("general.", "qwen35.")):
        x = original.fields.get(key)
        y = fusionado.fields.get(key)
        xv = x.contents() if x else "<ausente>"
        yv = y.contents() if y else "<ausente>"
        if str(xv) != str(yv):
            print(f"{key}: original={xv!r} | fusionado={yv!r}")
