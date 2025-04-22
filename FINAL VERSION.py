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
    "bayi": {"jumlah": 1, "per_shift": 8, "sertif": "bayi"},
    "klinik_umum": {"jumlah": 2, "per_shift": 2, "shift": [1, 2]},
    "klinik_gigi": {"jumlah": 1, "per_shift": 2, "shift": [1, 2], "sertif": "gigi"},
    "igd": {"jumlah": 1, "per_shift": 8}
}

# Data dummy perawat
perawat_list = []
sertif_opsi = [[], ['icu'], ['bayi'], ['gigi'], ['icu', 'bayi'], ['icu', 'gigi'], ['bayi', 'gigi'], ['icu', 'bayi', 'gigi']]
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
        #posisi perawat setiap hari
        self.position = [random.randint(0, JUMLAH_SHIFT) for _ in range(JUMLAH_HARI)]
        #kecepatan perawat setiap hari di set 0 untuk setiap hari
        self.velocity = [0 for _ in range(JUMLAH_HARI)]
        # posisi terbaik perawat
        self.best_position = copy.deepcopy(self.position)
        # fitness terbaik perawat
        self.best_fitness = float('inf')

# PSO Algorithm
class PSO:
    def __init__(self, swarm_size=275, max_iter=100):
        # Inisialisasi swarm dengan partikel perawat
        self.swarm = [Particle(p) for p in perawat_list]   
        # Inisialisasi posisi terbaik global
        self.global_best = None
        # Inisialisasi fitness terbaik global
        self.global_best_fitness = float('inf')
        # Inisialisasi jumlah iterasi maksimum
        self.max_iter = max_iter

    def fitness(self, particle):
        skor = 0
        # Ambil jadwal perawat
        jadwal = particle.position
        
        # Cek apakah perawat bekerja lebih dari 1 shift per hari
        for hari in range(JUMLAH_HARI):
            shift_hari_ini = jadwal[hari]
            if jadwal.count(shift_hari_ini) > 1:
                skor += 5  # Penalti jika perawat bekerja lebih dari 1 shift per hari

        # Cek perawat yang bekerja 2 shift berturut-turut
        for hari in range(1, JUMLAH_HARI):
            if jadwal[hari-1] == 3 and jadwal[hari] == 1:
                skor += 5  # Penalti jika ada perawat yang bekerja 2 shift berturut-turut

        # Cek perawat yang ditugaskan ke bangsal ICU, bayi, gigi tanpa sertifikat yang sesuai
        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                for unit, config in struktur_bangsal.items():
                    if unit in ['icu', 'bayi', 'klinik_gigi'] and shift_ke in config.get("shift", [1, 2, 3]):
                        sertif_diperlukan = config.get("sertif")
                        if sertif_diperlukan:
                            for p in [p for p in pso.swarm if p.best_position[hari] == shift_ke]:
                                if sertif_diperlukan not in p.perawat["sertifikat"]:
                                    skor += 3  # Penalti jika perawat tidak memiliki sertifikat yang sesuai

        # Cek alokasi perawat sesuai kebutuhan bangsal
        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                for unit, config in struktur_bangsal.items():
                    dibutuhkan = config.get("per_shift", 0)
                    perawat_terpilih = [p for p in pso.swarm if p.best_position[hari] == shift_ke]
                    if len(perawat_terpilih) != dibutuhkan:
                        skor += 10 * (dibutuhkan - len(perawat_terpilih)) # Penalti jika jumlah perawat tidak sesuai kebutuhan

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
                        skor += 7  # Penalti jika perawat baru tidak berpasangan dengan perawat senior

        # Cek kepala shift di setiap bangsal setiap shift
        for hari in range(JUMLAH_HARI):
            for shift_ke in [1, 2, 3]:
                for unit, config in struktur_bangsal.items():
                    perawat_untuk_shift = [p for p in pso.swarm if p.best_position[hari] == shift_ke]
                    kepala = tunjuk_kepala_shift(perawat_untuk_shift)
                    if not kepala:
                        skor += 8  # Penalti jika tidak ada kepala shift yang ditunjuk untuk setiap bangsal pada setiap shift

        # Penalti jika bekerja lebih dari 5 hari per minggu 
        for minggu_ke in range(4):  # 4 minggu pertama
            awal = minggu_ke * 7
            akhir = awal + 7
            hari_kerja = sum(1 for hari in range(awal, akhir) if jadwal[hari] > 0)
            if hari_kerja > 5:
                skor += (hari_kerja - 5)  # Penalti ringan per kelebihan hari kerja dalam 1 minggu

        return skor


    def update_velocity_position(self, particle):
        w, c1, c2 = 0.9, 1.5, 1.5
        for i in range(JUMLAH_HARI):
            r1, r2 = random.random(), random.random()
            # Update kecepatan dan posisi partikel
            particle.velocity[i] = int(w * particle.velocity[i] + c1 * r1 * (particle.best_position[i] - particle.position[i]) + c2 * r2 * (self.global_best[i] - particle.position[i]))
            # Batasi kecepatan dan posisi partikel
            particle.velocity[i] = max(min(particle.velocity[i], 2), -2)
            particle.position[i] += particle.velocity[i]
            particle.position[i] = max(0, min(particle.position[i], 3))

    def optimize(self):
        for iterasi in range(self.max_iter):
            # Hitung fitness untuk setiap partikel
            for p in self.swarm:
                f = self.fitness(p)
                # Jika fitness terbaru lebih kecil dari best fitness untuk particle tersebut,fitness particle sekarang dijadikan best fitness particle tersebut
                if f < p.best_fitness:
                    p.best_fitness = f
                    p.best_position = copy.deepcopy(p.position)
                # Jika fitness terbaru lebih kecil dari global best fitness, fitness particle sekarang dijadikan global best fitness
                if f < self.global_best_fitness:
                    self.global_best_fitness = f
                    self.global_best = copy.deepcopy(p.position)
            for p in self.swarm:
                #update posisi partikel
                self.update_velocity_position(p)
            # print(f"Iterasi {iterasi+1}: Global Best Fitness = {self.global_best_fitness}")

# Kepala Bangsal berdasarkan pengalaman kerja terbanyak
def tunjuk_kepala_shift(dipilih):
    if not dipilih:
        return None
    # Mengembalikan perawat dengan lama bekerja terbanyak sebagai kepala bangsal tersebut
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
    # Mengacak urutan perawat
    random.shuffle(perawat_aktif)  
    # Mengalokasikan perawat ke bangsal sesuai shift dan kriteria
    for nama_bangsal, config in struktur_bangsal.items():
        if "shift" in config and shift_ke not in config["shift"]:
            continue
        for i in range(config["jumlah"]):
            nama_unit = f"{nama_bangsal}_{i+1}"
            dibutuhkan = config["per_shift"]
            kandidat = []
            # Loop buat ngambil kandidat perawat sesuai kebutuhan sertifikat
            for p in perawat_aktif:
                sertif_diperlukan = config.get("sertif")
                if sertif_diperlukan and sertif_diperlukan not in p.perawat["sertifikat"]:
                    continue
                kandidat.append(p)
            # Mengacak kandidat yang valid
            random.shuffle(kandidat)
            terpilih = kandidat[:dibutuhkan]
            alokasi[nama_unit] = terpilih
            # Menghapus perawat yang sudah terpilih dari daftar perawat aktif
            for p in terpilih:
                if p in perawat_aktif:
                    perawat_aktif.remove(p)
    return alokasi


# Main execution
if __name__ == "__main__":
    pso = PSO(swarm_size=JUMLAH_PERAWAT, max_iter=10)
    pso.optimize()

    for hari in range(JUMLAH_HARI):
        for shift_ke in [1, 2, 3]:
            aktif = [p for p in pso.swarm if p.best_position[hari] == shift_ke]
            #alokasi perawat ke bangsal sesuai shift dan kriteria dari daftar perawat aktif
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