# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 2025
@author: Greg
Versi lengkap PSO untuk penjadwalan 275 perawat rumah sakit selama 1 bulan
"""

import random
from collections import defaultdict
import copy

# Parameter dasar
JUMLAH_PERAWAT = 275
JUMLAH_HARI = 30
JUMLAH_SHIFT = 3
SHIFT_LABEL = ['Pagi', 'Sore', 'Malam']

# Struktur bangsal dan kebutuhan per shift
struktur_bangsal = {
    "menular": {"jumlah": 6, "per_shift": 4},
    "dalam": {"jumlah": 3, "per_shift": 2},
    "icu": {"jumlah": 3, "per_shift": 4, "sertif": "icu"},
    "ibu": {"jumlah": 1, "per_shift": 4},
    "bayi": {"jumlah": 1, "per_shift": 8, "sertif": "neonatal"},
    "klinik_umum": {"jumlah": 2, "per_shift": 2, "shift": [1, 2]},
    "klinik_gigi": {"jumlah": 1, "per_shift": 2, "shift": [1, 2], "sertif": "gigi"},
    "igd": {"jumlah": 1, "per_shift": 8}
}

# Data dummy perawat
perawat_list = []
sertif_opsi = [[], ['icu'], ['neonatal'], ['gigi'], ['icu', 'neonatal']]
for i in range(JUMLAH_PERAWAT):
    perawat = {
        "nama": f"Perawat_{i+1}",
        "umur": random.randint(22, 50),
        "lama_bekerja": random.randint(0, 25),
        "sertifikat": random.choice(sertif_opsi)
    }
    perawat_list.append(perawat)

# Representasi Particle
class Particle:
    def __init__(self, perawat):
        self.perawat = perawat
        self.position = [random.randint(0, JUMLAH_SHIFT) for _ in range(JUMLAH_HARI)]
        self.velocity = [0 for _ in range(JUMLAH_HARI)]
        self.best_position = copy.deepcopy(self.position)
        self.best_fitness = float('inf')

# PSO Algorithm
class PSO:
    def __init__(self, swarm_size=275, max_iter=100):
        self.swarm = [Particle(p) for p in perawat_list]
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.max_iter = max_iter

    def fitness(self, particle):
        skor = 0
        jadwal = particle.position
        
        # Cek apakah perawat bekerja lebih dari 1 shift per hari
        for hari in range(JUMLAH_HARI):
            shift_hari_ini = jadwal[hari]
            if jadwal.count(shift_hari_ini) > 1:
                skor += 50  # Penalti tinggi jika perawat bekerja lebih dari 1 shift per hari

        # Cek perawat yang bekerja 2 shift berturut-turut
        for hari in range(1, JUMLAH_HARI):
            if jadwal[hari] > 0 and jadwal[hari-1] > 0:
                skor += 10  # Penalti jika perawat bekerja dua shift berturut-turut

        # Cek perawat yang ditugaskan ke bangsal ICU, bayi, gigi tanpa sertifikat yang sesuai
        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                for unit, config in struktur_bangsal.items():
                    if unit in ['icu', 'bayi', 'klinik_gigi'] and shift_ke in config.get("shift", [1, 2, 3]):
                        sertif_diperlukan = config.get("sertif")
                        if sertif_diperlukan:
                            for p in [p for p in pso.swarm if p.best_position[hari] == shift_ke]:
                                if sertif_diperlukan not in p.perawat["sertifikat"]:
                                    skor += 30  # Penalti jika perawat tidak memiliki sertifikat yang sesuai

        # Cek alokasi perawat sesuai kebutuhan bangsal
        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                for unit, config in struktur_bangsal.items():
                    dibutuhkan = config.get("per_shift", 0)
                    perawat_terpilih = [p for p in pso.swarm if p.best_position[hari] == shift_ke]
                    if len(perawat_terpilih) != dibutuhkan:
                        skor += 40  # Penalti jika jumlah perawat tidak sesuai kebutuhan

        # Cek perawat baru tidak berpasangan dengan perawat senior
        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                perawat_aktif = [p for p in pso.swarm if p.best_position[hari] == shift_ke]
                pasangan = pasangan_baru_senior(perawat_aktif)
                for b, s in pasangan:
                    if b.perawat["lama_bekerja"] < 5 and s.perawat["lama_bekerja"] > 20:
                        # Semua baik-baik saja, tidak ada penalti
                        pass
                    else:
                        skor += 25  # Penalti jika pasangan baru-senior tidak valid

        # Cek kepala shift di setiap bangsal setiap shift
        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                for unit, config in struktur_bangsal.items():
                    perawat_untuk_shift = [p for p in pso.swarm if p.best_position[hari] == shift_ke]
                    kepala = tunjuk_kepala_shift(perawat_untuk_shift)
                    if not kepala:
                        skor += 50  # Penalti jika tidak ada kepala shift yang ditunjuk

        return skor


    def update_velocity_position(self, particle):
        w, c1, c2 = 0.5, 1.5, 1.5
        for i in range(JUMLAH_HARI):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (particle.best_position[i] - particle.position[i])
            social = c2 * r2 * (self.global_best[i] - particle.position[i])
            particle.velocity[i] = int(w * particle.velocity[i] + cognitive + social)
            particle.velocity[i] = max(min(particle.velocity[i], 2), -2)
            particle.position[i] += particle.velocity[i]
            particle.position[i] = max(0, min(particle.position[i], 3))

    def optimize(self):
        for iterasi in range(self.max_iter):
            for p in self.swarm:
                f = self.fitness(p)
                if f < p.best_fitness:
                    p.best_fitness = f
                    p.best_position = copy.deepcopy(p.position)
                if f < self.global_best_fitness:
                    self.global_best_fitness = f
                    self.global_best = copy.deepcopy(p.position)
            for p in self.swarm:
                self.update_velocity_position(p)
            print(f"Iterasi {iterasi+1}: Global Best Fitness = {self.global_best_fitness}")

# Kepala Bangsal berdasarkan pengalaman kerja terbanyak
def tunjuk_kepala_shift(dipilih):
    if not dipilih:
        return None
    return max(dipilih, key=lambda p: p.perawat["lama_bekerja"])

# Pasangan baru (<5 tahun) dan senior (>20 tahun)
def pasangan_baru_senior(dipilih):
    baru = [p for p in dipilih if p.perawat["lama_bekerja"] < 5]
    senior = [p for p in dipilih if p.perawat["lama_bekerja"] > 20]
    pasangan = []
    for b in baru:
        if senior:
            pasangan.append((b, senior.pop(0)))
    return pasangan

# Alokasi perawat ke bangsal sesuai shift dan kriteria
def alokasikan_ke_bangsal(perawat_aktif, shift_ke, hari_ke):
    alokasi = defaultdict(list)
    for nama_bangsal, config in struktur_bangsal.items():
        if "shift" in config and shift_ke not in config["shift"]:
            continue
        for i in range(config["jumlah"]):
            nama_unit = f"{nama_bangsal}_{i+1}"
            dibutuhkan = config["per_shift"]
            kandidat = []
            for p in perawat_aktif:
                sertif_diperlukan = config.get("sertif")
                if sertif_diperlukan and sertif_diperlukan not in p.perawat["sertifikat"]:
                    continue
                kandidat.append(p)
            terpilih = kandidat[:dibutuhkan]
            alokasi[nama_unit] = terpilih
            for p in terpilih:
                if p in perawat_aktif:
                    perawat_aktif.remove(p)
    return alokasi

# Tampilkan jadwal
# def tampilkan_jadwal(pso):
#     print("\nJadwal Perawat Rumah Sakit Selama 30 Hari:")
#     for hari in range(JUMLAH_HARI):
#         print(f"Hari {hari+1}:")
#         for shift_ke in [1, 2, 3]:
#             print(f"  Shift {SHIFT_LABEL[shift_ke-1]}:")
#             perawat_untuk_shift = [p for p in pso.swarm if p.best_position[hari] == shift_ke]
#             for p in perawat_untuk_shift:
#                 print(f"    {p.perawat['nama']}")

# Main execution
if __name__ == "__main__":
    pso = PSO(swarm_size=JUMLAH_PERAWAT, max_iter=4)
    pso.optimize()
    # tampilkan_jadwal(pso)

    for hari in range(JUMLAH_HARI):
        for shift_ke in [1, 2, 3]:
            aktif = [p for p in pso.swarm if p.best_position[hari] == shift_ke]
            hasil_alokasi = alokasikan_ke_bangsal(aktif[:], shift_ke, hari)
            for unit, dipilih in hasil_alokasi.items():
                kepala = tunjuk_kepala_shift(dipilih)
                pasangan = pasangan_baru_senior(dipilih)
                
                print(f"\nHari-{hari+1} | Shift-{SHIFT_LABEL[shift_ke-1]} | Unit: {unit}")
                print(f"  Kepala Shift: {kepala.perawat['nama']} (Lama bekerja: {kepala.perawat['lama_bekerja']} tahun)" if kepala else "  Kepala Shift: Belum ada")

                print("  Daftar Perawat:")
                for p in dipilih:
                    print(f"    - {p.perawat['nama']} (Lama bekerja: {p.perawat['lama_bekerja']} tahun)")
                
                if pasangan:
                    print("  Pasangan Baru-Senior:")
                    for b, s in pasangan:
                        print(f"    - {b.perawat['nama']} (Baru, {b.perawat['lama_bekerja']} th) & {s.perawat['nama']} (Senior, {s.perawat['lama_bekerja']} th)")
                else:
                    print("  Tidak ada pasangan baru-senior.")

