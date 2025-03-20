import numpy as np
import random
import pandas as pd

# Parameter
perawat = 200
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
    {"id": i, "nama": f"Perawat {i}", "umur": random.randint(20, 50),
     "sertif_bayi": random.randint(0, 1), "sertif_ICU": random.randint(0, 1),
     "sertif_gigi": random.randint(0, 1)}
    for i in range(1, perawat + 1)
]

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
        assigned_nurses = set()
        assigned_nurses_klinik_umum = set()
        
        for bangsal_idx, bangsal in enumerate(daftar_bangsal):
            for s in range(shift):
                if s == 2 and bangsal["nama"] in klinik_tidak_buka_malam:
                    continue
                
                available_nurses = [n for n in range(perawat) if n not in assigned_nurses and has_required_certification(daftar_perawat[n], bangsal["nama"])]
                needed = bangsal["kapasitas"]
                
                if bangsal["nama"] == "Klinik Umum" and s == 0:
                    assigned = random.sample(available_nurses, min(len(available_nurses), needed))
                    assigned_nurses_klinik_umum.update(assigned)
                elif bangsal["nama"] == "Klinik Umum" and s == 1:
                    assigned = list(assigned_nurses_klinik_umum)
                else:
                    assigned = random.sample(available_nurses, min(len(available_nurses), needed))
                
                for n in assigned:
                    jadwal[n, day, s] = bangsal_idx
                    assigned_nurses.add(n)
    return jadwal

V1, V2, V3, V4 = 50, 50, 40, 30

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

particles = [init_particle() for _ in range(partikel)]
velocities = [np.zeros((perawat, hari, shift)) for _ in range(partikel)]
pbest = particles.copy()
pbest_fitness = [calculate_fitness(p) for p in pbest]
gbest = particles[np.argmin(pbest_fitness)]
gbest_fitness = min(pbest_fitness)

w, c1, c2 = 0.5, 1.5, 1.5
prev_gbest_fitness = gbest_fitness

for iteration in range(iterasi):
    for i in range(partikel):
        velocities[i] = (
            0.9 * velocities[i] +
            c1 * random.random() * (pbest[i] - particles[i]) +
            c2 * random.random() * (gbest - particles[i])
        )
        particles[i] = np.clip(particles[i] + velocities[i], -1, len(daftar_bangsal) - 1).astype(int)
        fitness = calculate_fitness(particles[i])
        if fitness < pbest_fitness[i]:
            pbest[i] = particles[i]
            pbest_fitness[i] = fitness
        if fitness < gbest_fitness:
            gbest = particles[i]
            gbest_fitness = fitness
    if iteration > 10 and abs(prev_gbest_fitness - gbest_fitness) < 1e-3:
        print("Konvergensi tercapai pada iterasi", iteration)
        break
    prev_gbest_fitness = gbest_fitness

def display_schedule(jadwal):
    data = []
    count_per_bangsal = {bangsal["nama"]: 0 for bangsal in daftar_bangsal}
    for n in range(perawat):
        for d in range(hari):
            for s in range(shift):
                if jadwal[n, d, s] >= 0:
                    bangsal_name = daftar_bangsal[jadwal[n, d, s]]["nama"]
                    data.append([
                        daftar_perawat[n]["nama"], d + 1, nama_shift[s], bangsal_name
                    ])
                    count_per_bangsal[bangsal_name] += 1
    df = pd.DataFrame(data, columns=["Perawat", "Hari", "Shift", "Bangsal/Klinik"])
    print("Jumlah perawat per bangsal:")
    for bangsal, count in count_per_bangsal.items():
        print(f"{bangsal}: {count}")
    return df

print("Jadwal terbaik ditemukan:")
display_df = display_schedule(gbest)
print(display_df.to_string(index=False))
