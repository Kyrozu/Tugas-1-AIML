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


def init_particle():
    jadwal = np.full((perawat, hari, shift), -1)
    for day in range(hari):
        if (day + 1) in {7, 14, 21, 28}:
            for n in daftar_perawat:
                n["day_off_left_per_minggu"] = 2
        assigned_nurses = set()
        for s in range(shift):
            for bangsal_idx, bangsal in enumerate(daftar_bangsal):
                if s == 2 and bangsal["nama"] in klinik_tidak_buka_malam:
                    continue
                if s == 1 and bangsal["nama"] == "Klinik Umum":
                    continue
                sorted_nurses = sorted(range(perawat), key=lambda n: daftar_perawat[n]["day_off_left_per_minggu"], reverse=True)
                available_nurses = [
                    n for n in sorted_nurses if n not in assigned_nurses and has_required_certification(daftar_perawat[n], bangsal["nama"])
                ]
                needed = bangsal["kapasitas"]
                senior_nurses = [n for n in available_nurses if daftar_perawat[n]["lama_bekerja"] >= 20]
                junior_nurses = [n for n in available_nurses if daftar_perawat[n]["lama_bekerja"] < 5]
                assigned = []
                for jn in junior_nurses:
                    if senior_nurses:
                        sn = senior_nurses.pop(0)
                        assigned.append(sn)
                    assigned.append(jn)
                remaining_needed = needed - len(assigned)
                if remaining_needed > 0:
                    extra_nurses = random.sample(available_nurses, min(len(available_nurses), remaining_needed))
                    assigned.extend(extra_nurses)
                for n in assigned:
                    jadwal[n, day, s] = bangsal_idx
                    assigned_nurses.add(n)
                
                # Menjaga perawat Klinik Umum tetap di shift sore
                if bangsal["nama"] == "Klinik Umum" and s == 0:
                    for n in assigned:
                        if jadwal[n, day, 1] == -1:
                            jadwal[n, day, 1] = bangsal_idx
                            assigned_nurses.add(n)
        for n in range(perawat):
            if n not in assigned_nurses:
                daftar_perawat[n]["day_off_left_per_minggu"] -= 1
    return jadwal

#cek nurse yang paling lama bekerjanya
def determine_shift_leader(nurses):
    if nurses:
        return max(nurses, key=lambda n: daftar_perawat[n]["lama_bekerja"])
    return None


#penalty/fitness point
V1, V2, V3, V4 = 50, 50, 40, 30

#V1 : 2 shift dalam 1 hari 
#V2 : 2 shift berturut-turut
#V3 : tidak sesuai kapasitas
#V4 : tdk dipakai (dipakai jika ada kriteria penalty baru)

# menghitung jumlah fitness setiap particle
# semua awale fitnessnya sama, terus dikurangi buat setiap aturan yg dilanggar
def calculate_fitness(jadwal):
    penalty = 0
    for n in range(perawat):
        for d in range(hari):
            if np.sum(jadwal[n, d, :] >= 0) > 1:
                penalty += V1
            for s in range(shift - 1):
                if jadwal[n, d, s] >= 0 and jadwal[n, d, s + 1] >= 0:
                    penalty += V2
            if d > 0 and jadwal[n, d-1, 2] >= 0 and jadwal[n, d, 0] >= 0:
                penalty += V2
    for d in range(hari):
        for s in range(shift):
            for b_idx, bangsal in enumerate(daftar_bangsal):
                if s == 2 and bangsal["nama"] in klinik_tidak_buka_malam:
                    continue
                total_perawat = np.sum(jadwal[:, d, s] == b_idx)
                if total_perawat < bangsal["kapasitas"]:
                    penalty += V3 * (bangsal["kapasitas"] - total_perawat)
    return penalty

# inisialisasi particle, jumlah kandidat solusi
particles = [init_particle() for _ in range(partikel)]
#inisialisasi kecepatan kecepatan awal partikel
velocities = [np.zeros((perawat, hari, shift)) for _ in range(partikel)]
#mencopy pbest particle
pbest = particles.copy()
#semua pbest dihitung fitnessnya
pbest_fitness = [calculate_fitness(p) for p in pbest]
#mengambil particle dengan nilai fitness terbaik
gbest = particles[np.argmin(pbest_fitness)]
#mengambil fitness terbaik
gbest_fitness = min(pbest_fitness)
#bobot inersia (boleh diganti sesuai selera)
#kalo bobot terlalu besar, partikelnya bergerak terus tannpa konvergen
#kalo terlalu kecil, solusinya terjebak di solusi lokal
#biasanya 0.4 - 0.9
w = 0.9
#mengontrol pengaruh pbest kalo terlalu besar, cenderung mengikuti solusi terbaik diri sendiri
c1 = 1.5
#mengontrol pengaruh gbest kalo terlalu besar, cenderung mengikuti solusi terbaik sebelumya
c2 = 1.5
#karena awal, gbest sebelumnya sama dengan gbest skrg
prev_gbest_fitness = gbest_fitness

#iterasi 
for iteration in range(iterasi):
    for i in range(partikel):
        #kecepatan particle
        velocities[i] = (
            w * velocities[i] +
            c1 * random.random() * (pbest[i] - particles[i]) +
            c2 * random.random() * (gbest - particles[i])
        )
        #update posisi partikel, np.clip buat mastiin tetep dalam range
        particles[i] = np.clip(particles[i] + velocities[i], -1, len(daftar_bangsal) - 1).astype(int)
        fitness = calculate_fitness(particles[i])
        if fitness < pbest_fitness[i]:
            pbest[i] = particles[i]
            pbest_fitness[i] = fitness
        if fitness < gbest_fitness:
            gbest = particles[i]
            gbest_fitness = fitness
    #cek apakah perubahan terlalu kecil
    #1e-3 itu seperti 0.001
    if iteration > 10 and abs(prev_gbest_fitness - gbest_fitness) < 1e-3:
        print("Konvergensi tercapai pada iterasi", iteration)
        break
    prev_gbest_fitness = gbest_fitness

#display jadwal
def display_schedule(jadwal):
    data = []
    count_per_bangsal = {bangsal["nama"]: 0 for bangsal in daftar_bangsal}
    
    for d in range(hari):
        for s in range(shift):
            for b_idx, bangsal in enumerate(daftar_bangsal):
                if s == 2 and bangsal["nama"] in klinik_tidak_buka_malam:
                    continue
                
                #cek perawat disetiap shift
                nurses_in_shift = [n for n in range(perawat) if jadwal[n, d, s] == b_idx]
                #assign ketua shift pada setiap bangsal
                ketua_shift = determine_shift_leader(nurses_in_shift)
                
                for n in nurses_in_shift:
                    data.append([
                        daftar_perawat[n]["nama"], d + 1, nama_shift[s], bangsal["nama"],
                        "Ketua" if n == ketua_shift else "Anggota", round(daftar_perawat[n]["lama_bekerja"],1)
                    ])
                    count_per_bangsal[bangsal["nama"]] += 1
    
    df = pd.DataFrame(data, columns=["Perawat", "Hari", "Shift", "Bangsal/Klinik", "Peran", "Lama Bekerja"])
    print("Jumlah perawat per bangsal:")
    for bangsal, count in count_per_bangsal.items():
        print(f"{bangsal}: {count}")
    return df

print("Jadwal terbaik ditemukan:")
display_df = display_schedule(gbest)
print(display_df.to_string(index=False))

