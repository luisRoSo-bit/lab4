[metricas_filtrado_251173003.csv](https://github.com/user-attachments/files/23563266/metricas_filtrado_251173003.csv)# Análisis de Sistema sin Documentación – Identificación de Tipo de Filtro
Laura Torres y Luis Rocha,
Este repositorio contiene el análisis completo de un sistema desconocido a partir de su
**respuesta al impulso**. Se realiza un estudio para identificar el tipo de filtro y su
frecuencia de corte aplicando señales sinusoidales, convolución, y estimando la
respuesta en frecuencia.

---

# 1. Actividad

**Actividad:**  
Una empresa ha contratado a su grupo de ingenieros para analizar un sistema del cual no se tiene documentación técnica.  
Su tarea es identificar el tipo de filtro que la empresa posee y caracterizar su comportamiento.

Para ello, se les ha proporcionado la respuesta al impulso del sistema (“RespuestaImpulso_Filtro.txt”), y deberán analizar su efecto sobre distintas señales sinusoidales para determinar su frecuencia de corte y su respuesta en frecuencia.

### **Puntos a resolver:**
1. **Señales de entrada:** Crear varias señales sinusoidales con frecuencias distintas, pero con la misma amplitud. Seleccione un conjunto de frecuencias que permita observar el comportamiento del filtro.  

2. **Convolución:** Aplicar la convolución de cada señal sinusoidal del punto anterior con la respuesta impulso dada.  

3. **Respuesta en frecuencia:**  
   - Medir amplitud de la señal de salida  
   - Calcular ganancia en dB  
   - Identificar la frecuencia de corte  
   - Graficar magnitud vs frecuencia  

4. **Identificar:** ¿Qué tipo de filtro corresponde a la respuesta al impulso proporcionada?

---

#  1. Código Implementado

A continuación se muestra el código empleado para resolver toda la actividad:

```python
[import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import warnings
import io # Necesario para leer el string

# Ignorar advertencias si log10(0) ocurre
warnings.filterwarnings('ignore', 'divide by zero encountered in log10')

print("Iniciando script de análisis de Respuesta en Frecuencia (Parte 1)...")

# -------------------------------------------------------------------
# --- 1. Cargar la Respuesta al Impulso (h[n]) --
# -------------------------------------------------------------------


datos_impulso_string = """
0.00671345623592002	0.00349326998467959	-0.000861846356611661	-0.005978111650822	-0.0113107765451497
-0.0161842496330829	-0.0198527044348853	-0.0215757265779651	-0.0207012744333016	-0.0167467646668343
-0.00946868835845063	0.00108808121175401	0.0145682512228113	0.0303154915503548	0.0474139477256069
0.0647589335931474	0.0811504076506396	0.0954000354898361	0.106440838965397	0.113427870350358
0.115819117353197	0.113427870350358	0.106440838965397	0.0954000354898361	0.0811504076506396
0.0647589335931474	0.0474139477256069	0.0303154915503548	0.0145682512228113	0.00108808121175401
-0.00946868835845063	-0.0167467646668343	-0.0207012744333016	-0.0215757265779651	-0.0198527044348853
-0.0161842496330829	-0.0113107765451497	-0.005978111650822	-0.000861846356611661	0.00349326998467959
0.00671345623592002
"""

try:
    # Usamos np.fromstring para leer los números del string
    # 'sep' puede ser un espacio ' ' o un tabulador '\t'
    h_n = np.fromstring(datos_impulso_string, sep='\t')
    
    # Si están separados por espacios en lugar de tabs, intentamos con espacios
    if len(h_n) <= 1:
        h_n = np.fromstring(datos_impulso_string, sep=' ')
        
    print(f"Respuesta al impulso h[n] cargada desde el texto (Longitud={len(h_n)}).")
    
    if len(h_n) != 41:
        print(f"Advertencia: Se esperaban 41 muestras, pero se cargaron {len(h_n)}.")
        if len(h_n) == 0:
            raise ValueError("No se pudieron cargar datos desde el string.")
            
except Exception as e:
    print(f"Error: No se pudieron procesar los números que pegaste.")
    print(f"Detalle: {e}")
    exit()

# -------------------------------------------------------------------
# --- LUGAR PARA AJUSTAR PARÁMETROS DEL BARRIDO ---
#
# f_min_norm: La frecuencia más baja a probar (ej. 0.005)
# f_max_norm: La frecuencia más alta a probar (siempre 0.5)
# num_puntos: Cuántos puntos calcular (más puntos = gráfica más suave)
#
# -------------------------------------------------------------------
f_min_norm = 0.005
f_max_norm = 0.5
num_puntos = 50
# -------------------------------------------------------------------

# Genera el vector de frecuencias espaciadas logarítmicamente
freqs_to_test = np.logspace(np.log10(f_min_norm), np.log10(f_max_norm), num=num_puntos)

# --- 2. Parámetros de simulación ---
A_in = 1.0
n_samples = 2000
n = np.arange(n_samples)

results_gain_db = []

print(f"Iniciando simulación con barrido logarítmico de {num_puntos} puntos.")
print(f"Probando desde f_norm={freqs_to_test[0]:.4f} hasta f_norm={freqs_to_test[-1]:.4f}")

# --- 3. Bucle de simulación ---
for f_norm in freqs_to_test:
    
    x_n = A_in * np.cos(2 * np.pi * f_norm * n)
    
    #
    # aqui se plica el filtro 
    #
    y_n = signal.convolve(x_n, h_n, mode='same')
    
    stable_part_start = n_samples // 2
    A_out = np.max(np.abs(y_n[stable_part_start:]))
    
    if A_out == 0:
        gain_db = -np.inf
    else:
        gain = A_out / A_in
        gain_db = 20 * np.log10(gain)
        
    results_gain_db.append(gain_db)

print("Simulación completada. Generando gráfica...")

# grafica los resultados
plt.figure(figsize=(12, 7))
plt.plot(freqs_to_test, results_gain_db, 'b-', label='Ganancia Medida', linewidth=2)
plt.xscale('log') # Eje X en escala logarítmica

# calcula la frecuencia de corte
max_gain = np.max(results_gain_db[0:5]) 
cutoff_db = max_gain - 3.0

plt.axhline(y=max_gain, color='r', linestyle='--', label=f'Ganancia Máx (Banda de Paso): {max_gain:.2f} dB')
plt.axhline(y=cutoff_db, color='g', linestyle='--', label=f'Nivel de Corte (-3 dB): {cutoff_db:.2f} dB')

try:
    cutoff_index = np.where(np.array(results_gain_db) < cutoff_db)[0][0]
    cutoff_freq = freqs_to_test[cutoff_index]
    
    plt.axvline(x=cutoff_freq, color='g', linestyle=':', label=f'Frecuencia de Corte (aprox): {cutoff_freq:.4f}', linewidth=2)
    print(f"\nFrecuencia de corte (-3dB) estimada en: f_norm = {cutoff_freq:.4f}")
except IndexError:
    print("\nNo se pudo determinar la frecuencia de corte.")

plt.title('Respuesta en Frecuencia (Eje Logarítmico)', fontsize=16)
plt.xlabel('Frecuencia Normalizada (f / fs) - Escala Logarítmica', fontsize=12)
plt.ylabel('Ganancia (dB)', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig('filtro_respuesta_frecuencia_log.png')
print("Gráfica 'filtro_respuesta_frecuencia_log.png' generada.")

plt.show()

print("Script finalizado.")]
---
```
#Resultados

<img width="1200" height="700" alt="respuesta a la freuencia" src="https://github.com/user-attachments/assets/308dfbe0-3694-43b7-8738-150ff69cdb6f" />


![licensed-image](https://github.com/user-attachments/assets/98a02d86-b25e-47c6-92b9-b1a4170ed7b5)


## Actividad 2

El filtro pasa-banda utilizado es de **orden 16**, implementado mediante **8 secciones de segundo orden**.  
Los coeficientes necesarios para su implementación están almacenados en el archivo  
**`Coef_PasaBanda.txt`**, donde:

- Cada **fila** representa una sección del filtro.  
- Las **columnas 1 a 3** contienen los coeficientes \( b_k^i \).  
- Las **columnas 4 a 6** contienen los coeficientes \( a_k^i \).

Además, el archivo **`Scale_PasaBanda.txt`** proporciona los valores de **escalamiento** correspondientes a cada sección \( i \), los cuales deben aplicarse adecuadamente para garantizar la **estabilidad** y **precisión** del filtro.

---

##  1. Implementación del filtro

Desarrollar el código para implementar el filtro en **secciones de segundo orden (SOS)**  
en **forma directa II**, asegurando la correcta conexión **en cascada** de las secciones a partir de la ecuación de diferencias.

---

##  2. Respuesta al impulso

Para analizar el comportamiento del filtro implementado:

1. Genere un vector de **200 muestras en cero**.  
2. Asigne un valor de **1 en la primera posición** (impulso unitario).  
3. Aplique este vector al filtro para obtener la **respuesta impulso**, la cual debería verse similar a la siguiente imagen:


<img width="720" height="438" alt="Screenshot 2025-11-15 123751" src="https://github.com/user-attachments/assets/b3e1ab9d-0859-4ba0-9437-c1b4f3682384" />


#  2. Código Implementado


```python
[import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import io  # Para leer texto como si fuera un archivo

# --------------------------------------------------------
# arhcivo
# --------------------------------------------------------

coef_data_string = """
1   0   -1  1   -1.386664837687 0.783814855173768
1   0   -1  1   -1.99820224672999   0.998223522033566
1   0   -1  1   -1.99489296425268   0.994914352666748
1   0   -1  1   -1.15764091141019   0.486570671241462
1   0   -1  1   -1.99230047362927   0.992322051018406
1   0   -1  1   -1.02898785194599   0.318012644862377
1   0   -1  1   -1.99086192626005   0.990883646684505
1   0   -1  1   -0.971107936511139  0.241605953257694
"""

scale_data_string = """
0.313058526158952
0.313058526158952
0.285661186645456
0.285661186645456
0.268953596832098
0.268953596832098
0.261051718810645
0.261051718810645
1
"""

# Leer matrices desde texto
sos_matrix = np.loadtxt(io.StringIO(coef_data_string))
scale_values = np.loadtxt(io.StringIO(scale_data_string))

print(f"Coeficientes SOS cargados: {sos_matrix.shape[0]} secciones")
print(f"Factores de escala cargados: {len(scale_values)} valores\n")

# --------------------------------------------------------
# Literal 2: Generar la señal de impulso unitario
# --------------------------------------------------------
impulse_signal = np.zeros(200)
impulse_signal[0] = 1.0

# --------------------------------------------------------
# Literal 1: Implementación del filtro en cascada (Forma Directa II)
# --------------------------------------------------------
current_signal = impulse_signal.copy()
num_sections = sos_matrix.shape[0]

# Aplicar la "pre-ganancia" (g[0]) ANTES del bucle
current_signal = current_signal * scale_values[0]
print(f"Aplicando ganancia inicial (escala={scale_values[0]:.6f})\n")

print("Aplicando las 8 secciones SOS en cascada...\n")

for i in range(num_sections):
    b = sos_matrix[i, 0:3]
    a = sos_matrix[i, 3:6]
    
    # Filtrado de la sección (Forma Directa II)
    section_output = signal.lfilter(b, a, current_signal)
    
    # Aplicar la escala [i+1] (g[1] a g[8]) DESPUÉS
    current_signal = section_output * scale_values[i+1]
    
    print(f"   → Sección {i+1} aplicada (b={b}, a={a}, escala={scale_values[i+1]:.6f})")

# El resultado final ya está listo
final_impulse_response = current_signal

print("\nFiltrado completado correctamente.")

# --------------------------------------------------------
# Graficar la respuesta al impulso (Literal 2)
# --------------------------------------------------------
plt.figure(figsize=(10, 6))

# --- CORRECCIÓN ---
# Quitado 'use_line_collection=True' para ser compatible con tu versión
markerline, stemlines, baseline = plt.stem(final_impulse_response)
# --- FIN CORRECCIÓN ---

plt.setp(markerline, 'markersize', 5)
plt.setp(baseline, 'color', 'r', 'linewidth', 2)

plt.title('Respuesta al Impulso — Filtro Pasa-Banda (Orden 16)', fontsize=15)
plt.xlabel('Número de muestras', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.xlim(-1, 21)
plt.ylim(-0.12, 0.22)
plt.grid(True)

plt.savefig('filtro_pasabanda_impulso_corregido.png')
print("\nGráfica 'filtro_pasabanda_impulso_corregido.png' generada (¡esta SÍ debe coincidir!).")

plt.show()

# --------------------------------------------------------
# (Opcional) Mostrar los primeros valores de la respuesta
# --------------------------------------------------------
print("\nPrimeras 20 muestras de la respuesta al impulso:")
print(final_impulse_response[:20])]
---
```
#resultados
<img width="1000" height="600" alt="respuesta al inpulso" src="https://github.com/user-attachments/assets/57722f84-bea6-4f91-a862-1fef74ab197f" />

#  3. Filtrar las señales ECG de la base de datos trabajada, aplicando el filtro implementado del punto anterior.

#  3. Código Implementado 
```python
[import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import io
import wfdb
import os

# ------------------------------------------------------------
# Código de diagnóstico a buscar 
# ------------------------------------------------------------
CODE_TO_FIND = "251173003"

# ------------------------------------------------------------
# Ruta base a tu base de datos de PhysioNet (local)
# ------------------------------------------------------------
ECG_DATABASE_PATH = r"C:\Users\luis.rocha-s\Desktop\trabajos\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords"

# ------------------------------------------------------------
# Función para aplicar el filtro pasa-banda
# ------------------------------------------------------------
def apply_my_filter(signal_in, sos_matrix, scale_values):
    current_signal = signal_in.copy()
    num_sections = sos_matrix.shape[0]

    # --- LÓGICA CORREGIDA ---
    current_signal = current_signal * scale_values[0]

    for i in range(num_sections):
        b = sos_matrix[i, 0:3]
        a = sos_matrix[i, 3:6]
        section_output = signal.lfilter(b, a, current_signal)
        current_signal = section_output * scale_values[i+1]

    return current_signal
    # --- FIN DE LA CORRECCIÓN ---

# ------------------------------------------------------------
# Función para buscar el primer ECG con el Dx deseado
# ------------------------------------------------------------
def find_first_by_dx_code(database_path, diagnosis_code):
    print(f"Buscando el primer ECG con Dx que contenga el código: {diagnosis_code} ...")

    for root, _, files in os.walk(database_path):
        for file in files:
            if file.endswith(".hea"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.startswith("#Dx:") and diagnosis_code in line:
                                record_name = file.replace(".hea", "")
                                print(f"Encontrado: {record_name} en {root}")
                                print(f"→ Línea Dx: {line.strip()}\n")
                                return root, record_name
                except Exception:
                    pass

    print(f"No se encontró ningún registro con el código {diagnosis_code}.")
    return None, None

# ------------------------------------------------------------
# Cargar coeficientes del filtro
# ------------------------------------------------------------
coef_data_string = """
1   0   -1  1   -1.386664837687 0.783814855173768
1   0   -1  1   -1.99820224672999   0.998223522033566
1   0   -1  1   -1.99489296425268   0.994914352666748
1   0   -1  1   -1.15764091141019   0.486570671241462
1   0   -1  1   -1.99230047362927   0.992322051018406
1   0   -1  1   -1.02898785194599   0.318012644862377
1   0   -1  1   -1.99086192626005   0.990883646684505
1   0   -1  1   -0.971107936511139  0.241605953257694
"""
scale_data_string = """
0.313058526158952
0.313058526158952
0.285661186645456
0.285661186645456
0.268953596832098
0.268953596832098
0.261051718810645
0.261051718810645
1
"""

sos_matrix = np.loadtxt(io.StringIO(coef_data_string))
scale_values = np.loadtxt(io.StringIO(scale_data_string))
print("Coeficientes y escalas del filtro cargados correctamente.\n")

# ------------------------------------------------------------
#  Buscar el archivo y aplicar el filtro
# ------------------------------------------------------------
found_dir, record_name = find_first_by_dx_code(ECG_DATABASE_PATH, CODE_TO_FIND)

if record_name:
    try:
        print(f"Cargando registro '{record_name}' desde {found_dir} ...")
        record_path = os.path.join(found_dir, record_name)
        record = wfdb.rdrecord(record_path)

        fs = record.fs
        lead_index = 1
        lead_name = record.sig_name[lead_index]
        signal_original = record.p_signal[:, lead_index]
        time_samples = np.arange(len(signal_original)) / fs

        print(f"Aplicando filtro pasa-banda (VERSIÓN CORRECTA) a '{lead_name}' ...")
        filtered_signal = apply_my_filter(signal_original, sos_matrix, scale_values)
        print("Filtrado completado correctamente.")

        # --- Gráfica: Original vs. Filtrada ---
        
        samples_to_plot = int(10 * fs)
        plt.figure(figsize=(14, 8)) 

        plt.subplot(2, 1, 1) 
        plt.plot(time_samples[:samples_to_plot], signal_original[:samples_to_plot], 'b-')
        plt.title(f"Señal Original (Derivación {lead_name}) - Archivo: {record_name}", fontsize=14)
        plt.ylabel("Amplitud [mV]")
        plt.grid(True)

        plt.subplot(2, 1, 2) 
        plt.plot(time_samples[:samples_to_plot], filtered_signal[:samples_to_plot], 'r-')
        plt.title(f"ECG Filtrado (Derivación {lead_name}) - Dx: {CODE_TO_FIND}", fontsize=14)
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud [mV]")
        plt.grid(True)
        
        plt.tight_layout() 
        plt.savefig("ecg_comparacion_filtrado.png")
        plt.show()

        print("\nGráfica guardada como 'ecg_comparacion_filtrado.png'")

    except Exception as e:
        print(f"Error al cargar o procesar el registro: {e}")
else:
    print("No se generó gráfica: no se encontró registro con ese Dx.")

print("\nScript finalizado correctamente.")]
---
```
#resultados

<img width="1214" height="585" alt="filtrado y post" src="https://github.com/user-attachments/assets/4fc7aecc-36c8-4e4d-8db9-c6ff8d237a8b" />


##  4.Evaluación del Impacto del Filtrado en las Señales ECG

Para analizar cómo afecta el filtrado a las señales ECG, se calcularon las siguientes métricas **para cada derivación** y **para cada archivo procesado**:

###  Métricas calculadas
- **Valor promedio de la señal sin filtrar**
- **Valor promedio de la señal filtrada**
- **Coeficiente de correlación** entre la señal filtrada y la señal original

#  4. Código Implementado 
```python
[import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import io
import wfdb
import os
import pandas as pd

# ------------------------------------------------------------
#  Código de diagnóstico a buscar 
# ------------------------------------------------------------
CODE_TO_FIND = "251173003"   # Cambia este valor por el código que quieras buscar

# ------------------------------------------------------------
#  Ruta base a tu base de datos de PhysioNet (local)
# ------------------------------------------------------------
ECG_DATABASE_PATH = r"C:\Users\luis.rocha-s\Desktop\trabajos\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords"

# ------------------------------------------------------------
#  Función para aplicar el filtro pasa-banda (VERSIÓN CORREGIDA)
# ------------------------------------------------------------
def apply_my_filter(signal_in, sos_matrix, scale_values):
    """
    Aplica el filtro en cascada usando la implementación estándar
    (g[0] al inicio, g[1] a g[8] en el bucle).
    Esta SÍ es la implementación del "punto anterior".
    """
    current_signal = signal_in.copy()
    num_sections = sos_matrix.shape[0]

    

    current_signal = current_signal * scale_values[0]

    for i in range(num_sections):
        b = sos_matrix[i, 0:3]
        a = sos_matrix[i, 3:6]
        section_output = signal.lfilter(b, a, current_signal)
        
        # 2. Aplicar escalas (g[1] a g[8])
        current_signal = section_output * scale_values[i+1]

    # 3. El resultado ya está listo
    return current_signal
    # --- FIN DE LA CORRECCIÓN ---

# ------------------------------------------------------------
#  Función para buscar el primer ECG 
# ------------------------------------------------------------
def find_first_by_dx_code(database_path, diagnosis_code):
    """Busca el primer archivo .hea cuyo campo #Dx contenga diagnosis_code."""
    print(f"Buscando el primer ECG con Dx que contenga el código: {diagnosis_code} ...")
    for root, _, files in os.walk(database_path):
        for file in files:
            if file.endswith(".hea"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.startswith("#Dx:") and diagnosis_code in line:
                                record_name = file.replace(".hea", "")
                                print(f"Encontrado: {record_name} en {root}")
                                print(f"→ Línea Dx: {line.strip()}\n")
                                return root, record_name
                except Exception:
                    pass
    print(f"No se encontró ningún registro con el código {diagnosis_code}.")
    return None, None


coef_data_string = """
1   0   -1  1   -1.386664837687 0.783814855173768
1   0   -1  1   -1.99820224672999   0.998223522033566
1   0   -1  1   -1.99489296425268   0.994914352666748
1   0   -1  1   -1.15764091141019   0.486570671241462
1   0   -1  1   -1.99230047362927   0.992322051018406
1   0   -1  1   -1.02898785194599   0.318012644862377
1   0   -1  1   -1.99086192626005   0.990883646684505
1   0   -1  1   -0.971107936511139  0.241605953257694
"""
scale_data_string = """
0.313058526158952
0.313058526158952
0.285661186645456
0.285661186645456
0.268953596832098
0.268953596832098
0.261051718810645
0.261051718810645
1
"""
sos_matrix = np.loadtxt(io.StringIO(coef_data_string))
scale_values = np.loadtxt(io.StringIO(scale_data_string))
print("Coeficientes y escalas del filtro cargados correctamente.\n")

# ------------------------------------------------------------
#  Buscar el archivo, aplicar el filtro y calcular métricas
# ------------------------------------------------------------
found_dir, record_name = find_first_by_dx_code(ECG_DATABASE_PATH, CODE_TO_FIND)

if record_name:
    try:
        print(f"Cargando registro '{record_name}' desde {found_dir} ...")
        record_path = os.path.join(found_dir, record_name)
        record = wfdb.rdrecord(record_path)

        fs = record.fs
        num_leads = record.n_sig
        lead_names = record.sig_name

        results_list = []

        print(f"Analizando las {num_leads} derivaciones...")

        for i in range(num_leads):
            lead_name = lead_names[i]
            signal_original = record.p_signal[:, i]
            
            # Aplicando el filtro CORRECTO
            filtered_signal = apply_my_filter(signal_original, sos_matrix, scale_values)

            # Cálculo de métricas
            mean_original = np.mean(signal_original)
            mean_filtered = np.mean(filtered_signal)
            correlation = np.corrcoef(signal_original, filtered_signal)[0, 1]

            results_list.append({
                "Archivo": record_name,
                "Derivación": lead_name,
                "Promedio Original": mean_original,
                "Promedio Filtrado": mean_filtered,
                "Correlación": correlation
            })

        df_results = pd.DataFrame(results_list)

        # ------------------------------------------------------------
        # Guardar resultados directamente en el ESCRITORIO
        # ------------------------------------------------------------
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_path = os.path.join(desktop_path, f"metricas_filtrado_{CODE_TO_FIND}.csv")
        df_results.to_csv(output_path, index=False, float_format="%.6f")

        print("\nAnálisis completado correctamente.")
        print(f"Archivo CSV guardado en: {output_path}\n")

        # ------------------------------------------------------------
        #  Graficar derivación II (índice 1)
        # ------------------------------------------------------------
        lead_index_plot = 1
        lead_name_plot = record.sig_name[lead_index_plot]
        
        # Aplicamos el filtro (correcto) de nuevo solo para la gráfica
        filtered_signal_plot = apply_my_filter(record.p_signal[:, lead_index_plot], sos_matrix, scale_values)
        time_samples_plot = np.arange(len(filtered_signal_plot)) / fs

        samples_to_plot = int(10 * fs)
        plt.figure(figsize=(14, 5))
        plt.plot(time_samples_plot[:samples_to_plot], filtered_signal_plot[:samples_to_plot], 'r')
        plt.title(f"ECG Filtrado (Derivación {lead_name_plot}) - Dx: {CODE_TO_FIND}", fontsize=14)
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud [mV]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(desktop_path, f"ecg_filtrado_{CODE_TO_FIND}.png"))
        plt.show()

        print(f"Gráfica guardada en el escritorio como 'ecg_filtrado_{CODE_TO_FIND}.png'")

    except Exception as e:
        print(f"Error al procesar el registro: {e}")
else:
    print("No se generó tabla ni gráfica: no se encontró registro con ese Dx.")

print("\nScript finalizado correctamente.")]
---
```

#resultados

<img width="1400" height="500" alt="ecg_filtrado_251173003" src="https://github.com/user-attachments/assets/1010d2d6-cb41-4fc9-9d9b-24b5659e2ff5" />

<img width="562" height="342" alt="Screenshot 2025-11-15 131032" src="https://github.com/user-attachments/assets/80f809f0-60e9-4c7d-9f01-fe891e79112b" />












