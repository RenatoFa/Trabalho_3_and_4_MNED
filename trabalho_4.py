import numpy as np
import matplotlib.pyplot as plt

# Parâmetros físicos
Lx = 1.0       # Comprimento do domínio
alfa = 0.01    # Coeficiente de difusão
u = 1.0        # Velocidade de advecção
CE = 1.0       # Condição de contorno em x = 0
nx = 50                    # Número de pontos espaciais
dx = Lx / (nx - 1)         # Tamanho do passo espacial
dt_estabilidade = 1.0 / (2 * alfa / dx**2 + u / dx)
dt = 0.9 * dt_estabilidade  # Um pouco menor que o máximo permitido
print(f"Passo de tempo dt = {dt:.5f}")

T_final = 0.5                    # Tempo final da simulação
nt = int(np.ceil(T_final / dt))  # Número de passos no tempo
dt = T_final / nt                # Recalcular dt para ajustar exatamente em T_final

# Malhas espacial e temporal
x = np.linspace(0, Lx, nx)
t = np.linspace(0, T_final, nt+1)

# Condição inicial: C(x, t=0) = 0
C = np.zeros(nx)

# Para armazenar os resultados em cada passo de tempo (se necessário)
C_todos = np.zeros((nt+1, nx))
C_todos[0, :] = C.copy()

# Condição de contorno em x = Lx (condição de Neumann)


def aplicar_condicao_neumann(C):
    # ∂C/∂x em x = Lx é aproximado por (C_N - C_{N-1}) / dx = 0
    # Portanto, C_N = C_{N-1}
    C[-1] = C[-2]
    return C


# Loop no tempo
for n in range(nt):
    # Aplicar condições de contorno
    C[0] = CE  # Condição de Dirichlet em x = 0
    C = aplicar_condicao_neumann(C)

    # Criar uma cópia de C para armazenar os novos valores
    C_novo = C.copy()

    # Atualizar os pontos interiores
    for i in range(1, nx-1):
        # Calcular as diferenças finitas
        dCdx = (C[i] - C[i-1]) / dx                   # Diferença regressiva
        d2Cdx2 = (C[i+1] - 2*C[i] + C[i-1]) / dx**2   # Diferença centrada

        # Atualizar usando o esquema explícito
        C_novo[i] = C[i] - dt * (u * dCdx - alfa * d2Cdx2)

    # Atualizar no último ponto (x = Lx - dx), já que C[-1] = C[-2]
    i = nx - 1
    dCdx = (C[i] - C[i-1]) / dx                        # Diferença regressiva
    # Ajustado para a condição de Neumann
    d2Cdx2 = (C[i-1] - 2*C[i] + C[i-1]) / dx**2
    C_novo[i] = C[i] - dt * (u * dCdx - alfa * d2Cdx2)

    # Atualizar a solução
    C = C_novo.copy()
    C_todos[n+1, :] = C.copy()

# Plotar os resultados em diferentes instantes de tempo
plt.figure(figsize=(10, 6))
for i in range(0, nt+1, nt//5):
    plt.plot(x, C_todos[i, :], label=f"t = {t[i]:.2f} s")
plt.xlabel('Posição x')
plt.ylabel('Concentração C')
plt.title('Perfil de Concentração ao Longo do Tempo')
plt.legend()
plt.grid(True)
plt.show()
