import pandas as pd
import random
from collections import defaultdict
import copy

# Parameter dasar
JUMLAH_HARI = 30
JUMLAH_SHIFT = 3
SHIFT_LABEL = ['Pagi', 'Sore', 'Malam']

# Membaca file Excel
df_bangsal = pd.read_excel("struktur_bangsal.xlsx")

# Konversi ke struktur dictionary
struktur_bangsal = {}
for _, row in df_bangsal.iterrows():
    unit = row['unit']
    struktur_bangsal[unit] = {
        "jumlah": int(row['jumlah']),
        "per_shift": int(row['per_shift']),
    }
    if pd.notna(row.get('sertif')) and row['sertif']:
        struktur_bangsal[unit]["sertif"] = str(row['sertif']).strip()
    if pd.notna(row.get('shift')) and str(row['shift']).strip() != "1, 2, 3":
        struktur_bangsal[unit]["shift"] = [int(s.strip()) for s in str(row['shift']).split(',')]

# Membaca data perawat dari Excel
df = pd.read_excel("perawat.xlsx")
perawat_list = []
for _, row in df.iterrows():
    sertifikat = [s.strip() for s in str(row['sertifikat']).split(',')] if pd.notna(row['sertifikat']) else []
    perawat = {
        "nama": row['nama'],
        "umur": int(row['umur']),
        "lama_bekerja": int(row['lama_bekerja']),
        "sertifikat": sertifikat
    }
    perawat_list.append(perawat)

JUMLAH_PERAWAT = len(perawat_list)

cuti_list = []
swap_list = []

class Particle:
    def __init__(self, perawat, cuti_list, swap_list):
        self.perawat = perawat
        self.position = [random.randint(0, JUMLAH_SHIFT) for _ in range(JUMLAH_HARI)]
        self.velocity = [0 for _ in range(JUMLAH_HARI)]
        self.best_position = copy.deepcopy(self.position)
        self.best_fitness = float('inf')

        for cuti in cuti_list:
            if cuti["nama"] == perawat["nama"] and 0 <= cuti["tanggal_cuti"] < JUMLAH_HARI:
                self.position[cuti["tanggal_cuti"]] = 0

        for swap in swap_list:
            if swap["nama"] == perawat["nama"] and 0 <= swap["tanggal_swap"] < JUMLAH_HARI:
                self.position[swap["tanggal_swap"]] = random.randint(1, JUMLAH_SHIFT)



class PSO:
    def __init__(self, swarm_size=275, max_iter=100):
        self.swarm = [Particle(p, cuti_list, swap_list) for p in perawat_list]
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.max_iter = max_iter
        self.fitness_history = []


    def fitness(self, particle):
        skor = 0
        jadwal = particle.position
        nama = particle.perawat["nama"]

        for hari in range(JUMLAH_HARI):
            shift_hari_ini = jadwal[hari]
            if jadwal.count(shift_hari_ini) > 1:
                skor += 5
            if hari > 0 and jadwal[hari - 1] == 3 and jadwal[hari] == 1:
                skor += 5

        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                for unit, config in struktur_bangsal.items():
                    if unit in ['icu', 'bayi', 'klinik_gigi'] and shift_ke in config.get("shift", [1, 2, 3]):
                        sertif_diperlukan = config.get("sertif")
                        if sertif_diperlukan:
                            for p in [p for p in self.swarm if p.best_position[hari] == shift_ke]:
                                if sertif_diperlukan not in p.perawat["sertifikat"]:
                                    skor += 3

        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                for unit, config in struktur_bangsal.items():
                    dibutuhkan = config.get("per_shift", 0)
                    perawat_terpilih = [p for p in self.swarm if p.best_position[hari] == shift_ke]
                    if len(perawat_terpilih) != dibutuhkan:
                        skor += 10 * abs(dibutuhkan - len(perawat_terpilih))

        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                perawat_aktif = [p for p in self.swarm if p.best_position[hari] == shift_ke]
                pasangan = pasangan_baru_senior(perawat_aktif)
                for b, s in pasangan:
                    if not (b.perawat["lama_bekerja"] < 5 and s.perawat["lama_bekerja"] > 20):
                        skor += 7

        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                for unit, config in struktur_bangsal.items():
                    perawat_untuk_shift = [p for p in self.swarm if p.best_position[hari] == shift_ke]
                    kepala = tunjuk_kepala_shift(perawat_untuk_shift)
                    if not kepala:
                        skor += 8

        for minggu_ke in range(4):
            awal, akhir = minggu_ke * 7, (minggu_ke + 1) * 7
            hari_kerja = sum(1 for hari in range(awal, min(akhir, JUMLAH_HARI)) if jadwal[hari] > 0)
            if hari_kerja > 5:
                skor += (hari_kerja - 5)

        for cuti in cuti_list:
            if cuti["nama"] == nama and 0 <= cuti["tanggal_cuti"] < JUMLAH_HARI:
                if jadwal[cuti["tanggal_cuti"]] > 0:
                    skor += 1000
                else:
                    skor -= 100  # reward kecil jika berhasil


        for swap in swap_list:
            if swap["nama"] == nama and 0 <= swap["tanggal_swap"] < JUMLAH_HARI:
                if jadwal[swap["tanggal_swap"]] == 0:
                    skor += 1000
                else:
                    skor -= 100  # reward kecil jika berhasil

        return skor

    def update_velocity_position(self, particle):
        w, c1, c2 = 0.9, 1.5, 1.5
        for i in range(JUMLAH_HARI):
            r1, r2 = random.random(), random.random()
            v_new = int(w * particle.velocity[i] + c1 * r1 * (particle.best_position[i] - particle.position[i]) + c2 * r2 * (self.global_best[i] - particle.position[i]))
            v_new = max(min(v_new, 2), -2)
            particle.velocity[i] = v_new
            particle.position[i] += v_new
            particle.position[i] = max(0, min(particle.position[i], 3))

            # Hormati cuti dan swap
            nama = particle.perawat["nama"]
            for cuti in cuti_list:
                if cuti["nama"] == nama and cuti["tanggal_cuti"] == i:
                    particle.position[i] = 0
            for swap in swap_list:
                if swap["nama"] == nama and swap["tanggal_swap"] == i:
                    if particle.position[i] == 0:
                        particle.position[i] = random.randint(1, JUMLAH_SHIFT)

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
            self.fitness_history.append(self.global_best_fitness)


def tunjuk_kepala_shift(dipilih):
    return max(dipilih, key=lambda p: p.perawat["lama_bekerja"], default=None)

def pasangan_baru_senior(dipilih):
    baru = [p for p in dipilih if p.perawat["lama_bekerja"] < 5]
    senior = [p for p in dipilih if p.perawat["lama_bekerja"] > 20]
    pasangan = []
    for b in baru:
        if senior:
            pasangan.append((b, senior.pop(0)))
    return pasangan

def alokasikan_ke_bangsal(perawat_aktif, shift_ke, hari_ke):
    alokasi = defaultdict(list)
    random.shuffle(perawat_aktif)
    for nama_bangsal, config in struktur_bangsal.items():
        if "shift" in config and shift_ke not in config["shift"]:
            continue
        for i in range(config["jumlah"]):
            nama_unit = f"{nama_bangsal}_{i+1}"
            dibutuhkan = config["per_shift"]
            kandidat = [p for p in perawat_aktif if config.get("sertif") in p.perawat["sertifikat"] or not config.get("sertif")]
            random.shuffle(kandidat)
            terpilih = kandidat[:dibutuhkan]
            alokasi[nama_unit] = terpilih
            for p in terpilih:
                if p in perawat_aktif:
                    perawat_aktif.remove(p)
    return alokasi

if __name__ == "__main__":
    while True:
        print("1. Request Cuti")
        print("2. Request Swap Jadwal")
        print("3. Generate Jadwal")
        print("0. Keluar")

        pilihan = input("Pilih menu: ")

        if pilihan == '1':
            nama_perawat = input("Nama Perawat: ")
            tanggal_cuti = int(input("Tanggal Cuti (1-30): ")) - 1
            cuti_list.append({"nama": nama_perawat, "tanggal_cuti": tanggal_cuti})

        elif pilihan == '2':
            nama_perawat = input("Nama Perawat: ")
            tanggal_pertama = int(input("Tanggal pertama (cuti, 1-30): ")) - 1
            tanggal_kedua = int(input("Tanggal kedua (ganti kerja, 1-30): ")) - 1
            cuti_list.append({"nama": nama_perawat, "tanggal_cuti": tanggal_pertama})
            swap_list.append({"nama": nama_perawat, "tanggal_swap": tanggal_kedua})

        elif pilihan == '3':
            pso = PSO(swarm_size=JUMLAH_PERAWAT, max_iter=10)
            pso.optimize()
            for hari in range(JUMLAH_HARI):
                for shift_ke in [1, 2, 3]:
                    aktif = [p for p in pso.swarm if p.best_position[hari] == shift_ke]
                    hasil_alokasi = alokasikan_ke_bangsal(aktif[:], shift_ke, hari)
                    for unit, dipilih in hasil_alokasi.items():
                        kepala = tunjuk_kepala_shift(dipilih)
                        pasangan = pasangan_baru_senior(dipilih)
                        print(f"\nHari-{hari+1} | Shift-{SHIFT_LABEL[shift_ke-1]} | Unit: {unit}")
                        if kepala:
                            print(f"  Kepala Shift: {kepala.perawat['nama']} ({kepala.perawat['lama_bekerja']} th)")
                        else:
                            print("  Kepala Shift: Belum ada")
                        print("  Daftar Perawat:")
                        for p in dipilih:
                            print(f"    - {p.perawat['nama']} ({p.perawat['lama_bekerja']} th)")
                        if pasangan:
                            print("  Pasangan Baru-Senior:")
                            for b, s in pasangan:
                                print(f"    - {b.perawat['nama']} & {s.perawat['nama']}")
                        else:
                            print("  Tidak ada pasangan baru-senior.")

        elif pilihan == '0':
            print("Good Bye...")
            break

        else:
            print("Pilihan tidak dikenali")
