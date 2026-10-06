\## 1. Descripción del problema



El proyecto consiste en desarrollar una neurona artificial capaz de determinar si una planta necesita riego a partir de dos variables de entrada:



\- Humedad del suelo (%).

\- Temperatura (°C).



La neurona recibe estos valores, los normaliza y utiliza una función sigmoide para calcular una probabilidad de riego.



La decisión final se obtiene utilizando un umbral de 0.5:



\- Si la probabilidad es mayor o igual a 0.5, se decide regar (1).

\- Si la probabilidad es menor a 0.5, se decide no regar (0).



El modelo fue entrenado utilizando 10 casos de ejemplo, donde cada caso tiene una humedad, una temperatura y una respuesta esperada de 0 o 1.





\## 2. Tecnologías utilizadas



\- Python

\- NumPy

\- uv

\- CMD (Command Prompt)







\## 3. Estructura del proyecto



```text

neurona-riego-plantas/

│

├── main.py

├── README.md

├── pyproject.toml

├── uv.lock

└── src/

&#x20;   └── neurona\_riego\_plantas/

&#x20;       └── \_\_init\_\_.py





\## 4. Estructura del proyecto

Paso 1. Abrir CMD



Ingresar a la carpeta del proyecto:



cd C:\\Taller\\neurona-riego-plantas





Paso 2. Ejecutar el programa



Utilizar el siguiente comando:



uv run main.py



El programa mostrará:



Los datos normalizados.

El error durante el entrenamiento.

Los pesos finales.

El sesgo.

El error final.

Las predicciones de los datos de entrenamiento.

Las predicciones para nuevas condiciones.

Los resultados de los 8 experimentos.





\##5. Configuración inicial



Para el entrenamiento se utilizaron los siguientes parámetros:



Pesos iniciales: 0.

Sesgo inicial: 0.

Tasa de aprendizaje: 0.5.

Número de épocas: 10.000.

Umbral de decisión: 0.5.



Los datos fueron normalizados utilizando la escala:



\[100, 50]



La humedad se dividió entre 100 y la temperatura entre 50.



TABLA CON LOS RESULTADOS DE LOS 8 EXPERIMENTOS.

\#	**Experimento     Épocas	T aprzj	Error fnl	Corrcts	Prob. 45%, 34 °C**

1	Prueba base	10.000	0.5	0.005378	10/10	0.5023

2	Pocas épocas	100	0.5	0.089365	10/10	0.5597

3	Cantidad intrmd	1.000	0.5	0.030556	10/10	0.5688

4	Más épocas	20.000	0.5	0.002800	10/10	0.4877

5	Tasa pequeña	10.000	0.01	0.067082	10/10	0.5751

6	Tasa moderada	10.000	0.1	0.019562	10/10	0.5470

7	Tasa alta	10.000	1.0	0.002800	10/10	0.4877

8	Tasa muy alta	10.000	2.0	0.001397	10/10	0.4731









| Caso | Humedad | Temperatura | Probabilidad | Decisión         |

| ---- | ------: | ----------: | -----------: | ---------------- |

| 1    |    75 % |       30 °C |       0.0000 | \*\*0 – No regar\*\* |

| 2    |    45 % |       34 °C |       0.5023 | \*\*1 – Regar\*\*    |

| 3    |    25 % |       22 °C |       0.9986 | \*\*1 – Regar\*\*    |

| 4    |    50 % |       25 °C |       0.1523 | \*\*0 – No regar\*\* |

| 5    |    30 % |       40 °C |       0.9936 | \*\*1 – Regar\*\*    |





Análisis

1\. ¿Por qué fue necesario normalizar la humedad y la temperatura?



Fue necesario normalizar los datos para llevar la humedad y la temperatura a una escala similar antes de entrenar la neurona. la humedad estaba expresada entre 0 y 100, mientras que la temperatura utilizaba valores aproximadamente entre 18 y 38 °C, se utilizó la escala \[100, 50] para convertir los valores a un rango cercano entre 0 y 1,esto permite que las variables puedan participar de manera adecuada en los cálculos de la neurona y facilita el aprendizaje.



2\. ¿En qué operaciones se utilizó X\_normalizado y para qué se conservó X?



X\_normalizado se utilizó durante el entrenamiento de la neurona, principalmente en el cálculo de la suma ponderada y en el cálculo del gradiente de los pesos.



Se conservó X con los valores originales para poder mostrar e interpretar los resultados utilizando las unidades reales como porcentaje de humedad y grados Celsius.



3\. ¿Qué ocurrió al utilizar solamente 100 épocas?



Con 100 épocas la neurona logró clasificar correctamente los 10 casos de entrenamiento pero presentó un error mayor de 0.089365. Esto indica que aunque consiguió las respuestas correctas el modelo todavía no había ajustado sus pesos tan bien como con un mayor número de épocas.



4\. ¿Más épocas siempre produjeron una mejora importante?



No, en las pruebas aumentar de 100 a 1.000, 10.000 y 20.000 épocas sí redujo el error:



100 épocas → 0.089365

1.000 épocas → 0.030556

10.000 épocas → 0.005378

20.000 épocas → 0.002800



&#x20;la mejora se fue haciendo menor a medida que aumentaban las épocas. 





5\. ¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña?



La tasa de aprendizaje de 0.01 produjo un aprendizaje más lento, con 10.000 épocas obtuvo un error de 0.067082 mayor que el obtenido con tasas como 0.5, 1.0 y 2.0.





6\. ¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta?



las tasas altas produjeron un menor error final:



Tasa 0.5 → 0.005378

Tasa 1.0 → 0.002800

Tasa 2.0 → 0.001397



en este conjunto de datos las tasas altas permitieron que la neurona ajustara sus pesos más rapido y alcanzara un menor error, pero una tasa demasiado alta puede provocar inestabilidad en otros problemas, por lo que no significa que siempre sea mejor utilizar una tasa grande.



7\. ¿Qué representa el signo del peso correspondiente a la humedad?



El peso de la humedad fue negativo, aproximadamente:



\-33.2773



El signo negativo significa que manteniendo las demás condiciones al aumentar la humedad disminuye la tendencia de la neurona a activar el riego.



8\. ¿Qué representa el signo del peso correspondiente a la temperatura?



El peso de la temperatura fue positivo, aproximadamente:



0.3467



El signo positivo significa que manteniendo las demás condiciones un aumento de la temperatura aumenta la tendencia de la neurona a activar el riego. 



9\. ¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?



La función sigmoide produce un valor entre 0 y 1 que podemos interpretar como una probabilidad o nivel de confianza pero el sistema necesita decidir: regar (1) o no regar (0).



Por eso se utiliza un umbral. Por ejemplo, con un umbral de 0.5:



Probabilidad ≥ 0.5 → 1, regar

Probabilidad < 0.5 → 0, no regar



El umbral permite convertir la probabilidad en una decisión binaria.



10\. ¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?



La principal limitación es que solo tiene en cuenta la humedad y la temperatura, mientras que en una planta real también influyen otros factores, solo fue entrenada con pocos datos por lo que no puede representar todas las situaciones de riego que pueden presentarse.











