import numpy as np
import random as random
import pandas as pd
# pip install pandas

# Parameter
perawat = 270
hari = 30
shift = 3
partikel = 40
iterasi = 100

nama_shift = ["Pagi", "Sore", "Malam"]
daftar_bangsal = [
    {"nama": "Penyakit Menular", "kapasitas": 24},
    {"nama": "Penyakit Tidak Menular", "kapasitas": 6},
    {"nama": "ICU", "kapasitas": 12},
    {"nama": "Ibu Melahirkan", "kapasitas": 4},
    {"nama": "Bayi Prematur", "kapasitas": 8},
    {"nama": "Klinik Umum", "kapasitas": 4},
    {"nama": "Klinik Gigi", "kapasitas": 2},
    {"nama": "IGD", "kapasitas": 8}
]

klinik_tidak_buka_malam = {"Klinik Umum", "Klinik Gigi"}

daftar_perawat = [
    {"id": i, "nama": f"Perawat {i}", "umur": random.randint(20, 50), "lama_bekerja":random.uniform(0.0,30.0),
     "sertif_bayi": random.randint(0, 1), "sertif_ICU": random.randint(0, 1),
     "sertif_gigi": random.randint(0, 1), "day_off_left_per_minggu": 2}
    for i in range(1, perawat + 1)
]

#cek ada sertif atau gk
def has_required_certification(nurse, bangsal):
    if bangsal == "Bayi Prematur" and not nurse["sertif_bayi"]:
        return False
    if bangsal == "ICU" and not nurse["sertif_ICU"]:
        return False
    if bangsal == "Klinik Gigi" and not nurse["sertif_gigi"]:
        return False
    return True
