"""
split_dataset.py
-----------------
Regenera el blind_test tomando un 20% estratificado de los datos reales
que ya se encuentran en train_val, evitando mezclar datos sinteticos.

USO: python split_dataset.py
"""
import os
import random
import shutil

random.seed(42)

TRAIN_VAL_DIR = 'dataset/train_val'
BLIND_TEST_DIR = 'dataset/blind_test'

def clean_synthetic_blind_test():
    """Elimina las imagenes sinteticas del blind_test (las que se llaman img_N.png)."""
    removed = 0
    for cls in ['ship', 'no_ship']:
        d = os.path.join(BLIND_TEST_DIR, cls)
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            # Las sinteticas tienen el patron img_N.png
            if f.startswith('img_') and f.endswith('.png'):
                os.remove(os.path.join(d, f))
                removed += 1
    print(f"Eliminadas {removed} imagenes sinteticas del blind_test.")

def split_real_data():
    """Mueve el 20% de train_val (datos reales) a blind_test."""
    moved = {'ship': 0, 'no_ship': 0}

    for cls in ['ship', 'no_ship']:
        src_dir = os.path.join(TRAIN_VAL_DIR, cls)
        dst_dir = os.path.join(BLIND_TEST_DIR, cls)
        os.makedirs(dst_dir, exist_ok=True)

        # Solo mover archivos reales (coordenadas GPS en el nombre)
        files = [f for f in os.listdir(src_dir) if f.endswith('.png') and not f.startswith('img_')]
        random.shuffle(files)

        n_test = int(len(files) * 0.20)
        test_files = files[:n_test]

        for f in test_files:
            shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            moved[cls] += 1

    return moved

def count_images():
    result = {}
    for split in ['train_val', 'blind_test']:
        result[split] = {}
        for cls in ['ship', 'no_ship']:
            d = os.path.join('dataset', split, cls)
            n = len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.exists(d) else 0
            result[split][cls] = n
    return result

if __name__ == "__main__":
    print("=== Regenerando blind_test con datos reales ===\n")

    print("Paso 1: Limpiando blind_test sintetico...")
    clean_synthetic_blind_test()

    print("Paso 2: Moviendo 20% de train_val a blind_test...")
    moved = split_real_data()
    print(f"  Movidos -> ship: {moved['ship']}, no_ship: {moved['no_ship']}")

    print("\nDistribucion final del dataset:")
    counts = count_images()
    for split, classes in counts.items():
        print(f"  {split}:")
        for cls, n in classes.items():
            print(f"    {cls}: {n} imagenes")

    print("\nListo! Ahora ejecuta: python train.py")
