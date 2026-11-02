import numpy as np

# =========================================================
# 1. DEFINICIÓN DEL PROBLEMA (La partícula en el círculo)
# =========================================================
N = 5
# La primera fila de la matriz circulante (estado 0)
# [P(quedarse), P(horario), P(2 pasos), P(3 pasos), P(antihorario)]
primera_fila = np.array([0.0, 0.5, 0.0, 0.0, 0.5])

# Construimos la matriz circulante P
# np.roll desplaza los elementos cíclicamente para crear cada fila
P = np.zeros((N, N))
for i in range(N):
    P[i] = np.roll(primera_fila, i)

print("--- Matriz de Transición P (un paso) ---")
print(np.round(P, 2))
print("-" * 50)

# =========================================================
# 2. CONEXIÓN CON FOURIER Y RAÍCES DE LA UNIDAD
# =========================================================
# Teóricamente, los eigenvalores de una matriz circulante 
# son exactamente la Transformada Discreta de Fourier (DFT) de su primera fila.
# Usamos np.fft.fft para calcularlos directamente.
eigenvalores = np.fft.fft(primera_fila)

print("\n--- Eigenvalores (Espectro de Fourier) ---")
for k, val in enumerate(eigenvalores):
    # La parte imaginaria es prácticamente cero por la simetría (P es simétrica)
    print(f"λ_{k} = {np.real(val):.5f} + {np.imag(val):.5f}i")
print("-" * 50)

# Construimos la matriz de Fourier (F) explícitamente usando las raíces de la unidad
# F_{j,k} = (1/sqrt(N)) * exp(2*pi*i*j*k / N)
omega = np.exp(2j * np.pi / N)
F = np.zeros((N, N), dtype=complex)
for j in range(N):
    for k in range(N):
        F[j, k] = (1 / np.sqrt(N)) * (omega ** (j * k))

# Para matrices circulantes simétricas, P = F^-1 * Lambda * F
F_inv = F.conj().T # Como F es unitaria, su inversa es su transpuesta conjugada

# =========================================================
# 3. CÁLCULO DE P^n USANDO DIAGONALIZACIÓN (EIGENVALORES)
# =========================================================
print("\n--- Matrices de Transición P^n (usando F Λ^n F^-1) ---")
n_pasos = [5, 10, 20, 40, 80]

for n in n_pasos:
    # Elevamos los eigenvalores a la n
    Lambda_n = np.diag(eigenvalores ** n)
    
    # Reconstruimos P^n
    P_n = F @ Lambda_n @ F_inv
    
    # Tomamos la parte real (la parte imaginaria es ~0 por errores de redondeo)
    P_n = np.real(P_n)
    
    print(f"\nP^{n}:")
    print(np.round(P_n, 5))

# =========================================================
# 4. PROBABILIDADES DE ESTADO ESTABLE
# =========================================================
# El estado estable π es el eigenvector izquierdo asociado a λ_0 = 1
# Resolvemos πP = π (o P^T π^T = π^T)
valores, vectores = np.linalg.eig(P.T)
# Buscamos el índice del eigenvalor que sea 1 (o el más cercano a 1)
idx = np.argmin(np.abs(valores - 1))
estado_estable = np.real(vectores[:, idx])
# Normalizamos para que sume 1
estado_estable = estado_estable / np.sum(estado_estable)

print("\n" + "="*50)
print("--- PROBABILIDADES DE ESTADO ESTABLE (π) ---")
print("La distribución estacionaria es:")
print(np.round(estado_estable, 5))
print("(Como la matriz es doblemente estocástica, es la distribución uniforme 1/5)")

# Verificación: π * P = π
verificacion = estado_estable @ P
print(f"\nVerificación (π * P): {np.round(verificacion, 5)}")