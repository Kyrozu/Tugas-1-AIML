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

    # Tambahkan sertifikat jika ada
    if pd.notna(row.get('sertif')) and row['sertif']:
        struktur_bangsal[unit]["sertif"] = str(row['sertif']).strip()

    # Tambahkan shift jika ada (dan tidak semua shift)
    if pd.notna(row.get('shift')) and str(row['shift']).strip() != "1, 2, 3":
        struktur_bangsal[unit]["shift"] = [int(s.strip()) for s in str(row['shift']).split(',')]


# Membaca data perawat dari Excel
df = pd.read_excel("perawat.xlsx")  # Pastikan file ini berada di folder kerja

# Konversi DataFrame ke list of dict
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
print(f"Jumlah perawat: {JUMLAH_PERAWAT}")
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
        
        #penalti perawat bekerja di hari cuti
        for cuti in cuti_list:
            if cuti["tanggal_cuti"] < JUMLAH_HARI:
                nama_perawat = cuti["nama"]
                for hari in range(JUMLAH_HARI):
                    if jadwal[hari] > 0 and perawat_list[hari]["nama"] == nama_perawat:
                        skor += 50

        # penalti perawat tidak bekerja di hari swap
        for swap in swap_list:
            if swap["tanggal_swap"] < JUMLAH_HARI:
                nama_perawat = swap["nama"]
                for hari in range(JUMLAH_HARI):
                    if jadwal[hari] == 0 and perawat_list[hari]["nama"] == nama_perawat:
                        skor += 50

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
    cuti_list = []
    swap_list = []
    while True:
        print("1. Request Cuti")
        print("2. Request Swap Jadwal")
        print("3. Generate Jadwal")
        print("0. Keluar")

        pilihan = input("Pilih menu: ")

        match pilihan:
            case '1':
                # Request Cuti
                nama_perawat = input("Nama Perawat: ")
                tanggal_cuti = int(input("Tanggal Cuti: "))
                cuti = {
                    "nama": nama_perawat,
                    "tanggal_cuti": tanggal_cuti
                }
                cuti_list.append(cuti)
                # tambah fitness perawat ini gk boleh di tanggal itu

            case '2':
                # Request Swap Jadwal
                nama_perawat = input("Nama Perawat: ")
                tanggal_pertama = int(input("Tanggal pertama: "))
                tanggal_kedua = int(input("Tanggal kedua: "))
                cuti = {
                    "nama": nama_perawat,
                    "tanggal_cuti": tanggal_pertama
                }
                cuti_list.append(cuti)
                swap = {
                    "nama": nama_perawat,
                    "tanggal_swap": tanggal_kedua
                }
                swap_list.append(swap)
                # tambah fitness perawat ini gk boleh di tanggal pertama
                # tambah fitness perawat ini harus di tanggal kedua


            case '3':
                # Generate Jadwal
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

            case '0':
                print("Good Bye...")
                break
            
            case _:
                print("Pilihan tidak dikenali")