import pandas as pd
import random

# Data perawat
data_perawat = []
sertifikat_opsi = ['umum', 'icu', 'bayi', 'gigi']

for i in range(1, 276):
    nama = f"Perawat {i:03}"
    umur = random.randint(22, 60)
    lama_bekerja = random.randint(1, umur - 21)
    sertifikat = random.sample(sertifikat_opsi, k=random.randint(1, 2))
    data_perawat.append({
        "nama": nama,
        "umur": umur,
        "lama_bekerja": lama_bekerja,
        "sertifikat": ", ".join(sertifikat)
    })

df_perawat = pd.DataFrame(data_perawat)

# Sheet cuti
df_cuti = pd.DataFrame([
    {"nama": "Perawat 001", "tanggal_cuti": 3},
    {"nama": "Perawat 010", "tanggal_cuti": 12},
])

# Sheet swap
df_swap = pd.DataFrame([
    {"nama": "Perawat 001", "tanggal_swap": 5},
    {"nama": "Perawat 010", "tanggal_swap": 15},
])

# Simpan ke Excel
with pd.ExcelWriter("perawat.xlsx", engine='openpyxl') as writer:
    df_perawat.to_excel(writer, sheet_name="Sheet1", index=False)
    df_cuti.to_excel(writer, sheet_name="cuti", index=False)
    df_swap.to_excel(writer, sheet_name="swap", index=False)

print("✅ File 'perawat.xlsx' berhasil dibuat.")
