import numpy as np

X = np.array([
    [80, 18],
    [70, 22],
    [65, 28],
    [55, 25],
    [50, 32],
    [40, 30],
    [35, 25],
    [30, 32],
    [20, 35],
    [10, 38]
], dtype=float)

y = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1]
], dtype=float)

escala = np.array([100, 50])

X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

pesos = np.zeros((2, 1))
sesgo = 0.0

tasa_aprendizaje = 0.5
epocas = 10000


for epoca in range(epocas):

    # Suma ponderada
    z = X_normalizado @ pesos + sesgo

    # Probabilidad
    probabilidad = sigmoide(z)

    # Error
    error = probabilidad - y

    # Gradiente de z
    gradiente_z = error * probabilidad * (1 - probabilidad)

    # Gradiente de los pesos
    gradiente_pesos = X_normalizado.T @ gradiente_z

    # Gradiente del sesgo
    gradiente_sesgo = np.sum(gradiente_z)

    # Actualizar pesos y sesgo
    pesos -= tasa_aprendizaje * gradiente_pesos
    sesgo -= tasa_aprendizaje * gradiente_sesgo

    mse = np.mean((probabilidad - y) ** 2)

    if (epoca + 1) % 1000 == 0:
        print(f"Época {epoca + 1}: MSE = {mse:.6f}")

# Calcular nuevamente las probabilidades con los pesos finales
z = X_normalizado @ pesos + sesgo
probabilidad = sigmoide(z)

# Calcular el error final
mse = np.mean((probabilidad - y) ** 2)


print("\nResultados finales:")
print("Peso humedad:", pesos[0, 0])
print("Peso temperatura:", pesos[1, 0])
print("Sesgo:", sesgo)
print("Error final:", mse)

respuesta = (probabilidad >= 0.5).astype(int)

print("\nResultados de los datos de entrenamiento:")

for i in range(len(X)):
    print(
        f"Caso {i + 1}: "
        f"Humedad = {X[i, 0]:.0f}%, "
        f"Temperatura = {X[i, 1]:.0f}°C, "
        f"Probabilidad = {probabilidad[i, 0]:.4f}, "
        f"Esperado = {int(y[i, 0])}, "
        f"Predicción = {respuesta[i, 0]}"
    )


# Pruebas con condiciones nuevas

nuevos_datos = np.array([
    [75, 30],
    [45, 34],
    [25, 22],
    [50, 25],
    [30, 40]
], dtype=float)

# Normalizar usando la misma escala del entrenamiento
nuevos_normalizados = nuevos_datos / escala

# Calcular probabilidades
z_nuevos = nuevos_normalizados @ pesos + sesgo
probabilidades_nuevas = sigmoide(z_nuevos)

# Convertir las probabilidades en 0 o 1
decisiones = (probabilidades_nuevas >= 0.5).astype(int)

print("\nPredicciones con condiciones nuevas:")

for i in range(len(nuevos_datos)):
    print(
        f"Caso nuevo {i + 1}: "
        f"Humedad = {nuevos_datos[i, 0]:.0f}%, "
        f"Temperatura = {nuevos_datos[i, 1]:.0f}°C, "
        f"Probabilidad = {probabilidades_nuevas[i, 0]:.4f}, "
        f"Decisión = {decisiones[i, 0]}"
    )


def entrenar(tasa_aprendizaje, epocas):
    pesos = np.zeros((2, 1))
    sesgo = 0.0

    for epoca in range(epocas):

        z = X_normalizado @ pesos + sesgo

        probabilidad = sigmoide(z)

        error = probabilidad - y

        gradiente_z = error * probabilidad * (1 - probabilidad)

        gradiente_pesos = X_normalizado.T @ gradiente_z

        gradiente_sesgo = np.sum(gradiente_z)

        pesos -= tasa_aprendizaje * gradiente_pesos

        sesgo -= tasa_aprendizaje * gradiente_sesgo

    z = X_normalizado @ pesos + sesgo

    probabilidad = sigmoide(z)

    mse = np.mean((probabilidad - y) ** 2)

    respuestas = (probabilidad >= 0.5).astype(int)

    correctas = np.sum(respuestas == y)

    dato_prueba = np.array([[45, 34]], dtype=float)

    dato_prueba_normalizado = dato_prueba / escala

    z_prueba = dato_prueba_normalizado @ pesos + sesgo

    probabilidad_prueba = sigmoide(z_prueba)

    return mse, correctas, probabilidad_prueba[0, 0]


experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Más épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0)
]

print("\n" + "=" * 70)
print("EXPERIMENTOS")
print("=" * 70)

for nombre, epocas_experimento, tasa_experimento in experimentos:

    mse, correctas, probabilidad_45_34 = entrenar(
        tasa_experimento,
        epocas_experimento
    )

    print(f"\n{nombre}")
    print(f"Épocas: {epocas_experimento}")
    print(f"Tasa de aprendizaje: {tasa_experimento}")
    print(f"Error final: {mse:.6f}")
    print(f"Respuestas correctas: {int(correctas)}/10")
    print(f"Probabilidad para 45% y 34°C: {probabilidad_45_34:.4f}")