import os, zipfile, random
from pathlib import Path

zip_path = Path.home() / 'Downloads' / 'Celeb-DF-v2.zip'
out_root = Path('data/raw')
real_out = out_root / 'real'
fake_out = out_root / 'fake'
real_out.mkdir(parents=True, exist_ok=True)
fake_out.mkdir(parents=True, exist_ok=True)

# clear existing files for clean smoke test
for p in list(real_out.glob('*')) + list(fake_out.glob('*')):
    if p.is_file():
        p.unlink()

with zipfile.ZipFile(zip_path) as z:
    names = [n for n in z.namelist() if n.lower().endswith('.mp4')]
    real = [n for n in names if n.startswith('Celeb-real/') or n.startswith('YouTube-real/')]
    fake = [n for n in names if n.startswith('Celeb-synthesis/')]

    random.seed(42)
    random.shuffle(real)
    random.shuffle(fake)

    # balanced subset for quick run
    real = real[:240]
    fake = fake[:240]

    for n in real:
        target = real_out / Path(n).name
        with z.open(n) as src, open(target, 'wb') as dst:
            dst.write(src.read())

    for n in fake:
        target = fake_out / Path(n).name
        with z.open(n) as src, open(target, 'wb') as dst:
            dst.write(src.read())

print(f'extracted_real={len(real)}')
print(f'extracted_fake={len(fake)}')
