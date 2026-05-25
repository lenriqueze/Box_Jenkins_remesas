# %% Importación de paquetes ============================

# Trabajar con rutas relativas en python 
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


# %% Configuración de directorios y rutas =========================

# Obtener la ruta del directorio raíz (BOX_JENKINS_REMESAS)
BASE_DIR = Path(__file__).resolve().parents[1]  

# Obtener la ruta del directorio con los datos
DATA_DIR = BASE_DIR / "Datos"

# Obtener la ruta del directorio de resultados y crearla si no existe
RESULTS_DIR = BASE_DIR / "Resultados"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Rutas de las bases de datos 
ruta_exp = DATA_DIR / "datos.xlsx" # Base de datos de remesas


# %% Cargar y limpiar base de datos =========================

# Base de datos con la serie importada a python
expo_base = pd.read_excel(
    ruta_exp,
    header=None,
    names=["Ingresos de Remesas de trabajadores"]
)

# --- LIMPIEZA DE DATOS (Crucial para eliminar textos informativos y vacíos al final) ---
# Forzamos la conversión a numérico (los textos del final se convertirán en NaN)
expo_base["Ingresos de Remesas de trabajadores"] = pd.to_numeric(
    expo_base["Ingresos de Remesas de trabajadores"], 
    errors="coerce"
)
# Eliminamos todas las filas que contengan NaN
expo_base = expo_base.dropna()

# Ver el tipo de objeto de la base de datos (Pandas.DataFrame)
print("\n--- Información de la Base de Datos ---")
print(type(expo_base))

# Ver primeras y últimas observaciones de la base de datos ya limpia
print("\nPrimeras observaciones cargadas:")
print(expo_base.head()) 
print("\nÚltimas observaciones cargadas:")
print(expo_base.tail()) 


# %% Creación del índice temporal de las series de tiempo

# Creación del índice temporal (Inicia en 2000 según tus datos)
fechas_expo_base = pd.date_range(
    start="2000-01-01", 
    periods=len(expo_base),
    freq="MS"
)

# Agregar el índice temporal a la base de datos
expo_base.index = fechas_expo_base

# Ver primeras y últimas observaciones de la base de datos, ahora con índice temporal
print("\n--- Base de Datos con Índice Temporal ---")
print(expo_base.head()) 
print(expo_base.tail()) 


# %% Creación de la serie de tiempo de "remesas"

# La nueva serie de tiempo se va a llamar "expo_serie" (para mantener consistencia con tu código)
# pero ahora apunta a la columna correcta
expo_serie = expo_base["Ingresos de Remesas de trabajadores"].copy()

# Ver el principio y final de la serie de tiempo
print("\n--- Serie Temporal Seleccionada (expo_serie) ---")
print(expo_serie.head())
print(expo_serie.tail())
print(type(expo_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo 
print("\nEstadísticas descriptivas:")
print(expo_serie.describe())


# %% =========================
# Paso 1: Identificación
# ============================

# Gráfica de la serie de tiempo "Remesas de trabajadores"
plt.figure(figsize=(10, 5))
plt.plot(expo_serie, color="blue", linewidth=1.5)
plt.title("Ingresos de Remesas de Trabajadores")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "01_remesas_original.png", dpi=300, bbox_inches="tight")
plt.show()

# %% FAC y FACP de la serie original

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    expo_serie,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC Remesas de Trabajadores")

plot_pacf(
    expo_serie,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP de la serie original")

plt.tight_layout()
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "02_fac_facp_original.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(expo_serie) 

print("\n=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"  {nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(expo_serie, regression="c", nlags="auto")

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"  {nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: Rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: No rechazamos H0. Según el test, la serie es estacionaria.")


# %% Serie diferenciada
expo_serie_diff = expo_serie.diff().dropna()

# Gráfica de la serie de tiempo diferenciada
plt.figure(figsize=(10, 5))
plt.plot(expo_serie_diff, color="orange")
plt.title("Serie Diferenciada (Primera Diferencia)")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "03_serie_diferenciada.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Diferencia del logaritmo

expo_serie_log_diff = np.log(expo_serie).diff().dropna()

# Gráfica de la diferencia del logaritmo
plt.figure(figsize=(10, 5))
plt.plot(expo_serie_log_diff, color="green")
plt.title("Diferencia del Logaritmo de la Serie Original")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "04_diferencia_logaritmo.png", dpi=300, bbox_inches="tight")
plt.show()

# %% FAC y FACP de la diferencia del logaritmo

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    expo_serie_log_diff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC de la diferencia del logaritmo")

plot_pacf(
    expo_serie_log_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP de la diferencia del logaritmo")

plt.tight_layout()
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "05_fac_facp_diferencia_log.png", dpi=300, bbox_inches="tight")
plt.show()


# %% =========================
# Paso 2: Estimación
# ============================

# Creamos el modelo SARIMAX (0, 1, 1) sobre el logaritmo de la serie
modelo_ma1 = SARIMAX(
    np.log(expo_serie),
    order=(0, 1, 1),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Estimación del modelo MA(1)
estimacion_ma1 = modelo_ma1.fit(disp=False)

# Imprimimos los resultados de la estimación
print("\n=== Resumen de la Estimación del Modelo ===")
print(estimacion_ma1.summary())


# %% =========================
# Paso 3: Validación de supuestos
# ============================

# Residuales del modelo
residuales = estimacion_ma1.resid.dropna()

# Gráfica de los residuales
plt.figure(figsize=(10, 5))
plt.plot(residuales.iloc[1:-1], color="purple")
plt.title("Residuales del Modelo ARIMA(0, 1, 1)")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "06_residuales.png", dpi=300, bbox_inches="tight")
plt.show()

# Descripción de los residuales 
print("\nDescriptivas de los residuales:")
print(residuales.describe())

# %% FAC y FACP de los residuales

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC de los residuales")

plot_pacf(
    residuales,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP de los residuales")

plt.tight_layout()
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "07_fac_facp_residuales.png", dpi=300, bbox_inches="tight")
plt.show()


# %% Prueba Ljung-Box

ljung_box = acorr_ljungbox(
    residuales,
    lags=[6, 12, 18, 24],
    return_df=True
)

print("\nPrueba Ljung-Box:")
print(ljung_box)


# %% FAC y FACP de los residuales al cuadrado

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    residuales**2,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC de los residuales al cuadrado")

plot_pacf(
    residuales**2,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP de los residuales al cuadrado")

plt.tight_layout()
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "08_fac_facp_residuales_cuadrado.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Prueba ARCH de heterocedasticidad

arch_test = het_arch(residuales, nlags=12) 

print("\nPrueba ARCH:")
print("Estadístico LM:", arch_test[0])
print("p-valor LM:", arch_test[1])


# %% Q-Q plot de los residuos

plt.figure(figsize=(6, 6))
probplot(residuales.iloc[1:-1], dist="norm", plot=plt)
plt.title("Q-Q plot de los residuos")
plt.grid(True)
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "09_qq_plot_residuos.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Prueba de normalidad Jarque-Bera

jb_test = jarque_bera(residuales.iloc[1:-1])

print("\nPrueba Jarque-Bera:")
print("Estadístico JB:", jb_test.statistic)
print("p-valor JB:", jb_test.pvalue)


# %% =========================
# PASO 4: Pronóstico
# ============================

# Pronóstico 12 pasos adelante
pronostico_log = estimacion_ma1.get_forecast(steps=12)

# Extraer valores esperados, varianza e intervalos
pronostico_puntual_log = pronostico_log.predicted_mean
varianza_pronostico_log = pronostico_log.var_pred_mean
intervalos_log = pronostico_log.conf_int()

# Corrección por sesgo logarítmico para volver a la escala original (niveles)
pronostico_nivel = np.exp(pronostico_puntual_log + 0.5 * varianza_pronostico_log)
intervalos_nivel = np.exp(intervalos_log)

# Tabla de pronósticos
tabla_pronostico = pd.DataFrame({
    "Pronóstico": pronostico_nivel,
    "Límite Inferior": intervalos_nivel.iloc[:, 0],
    "Límite Superior": intervalos_nivel.iloc[:, 1]
})

print("\n=== Tabla de Pronósticos (Siguientes 12 meses) ===")
print(tabla_pronostico)

# Gráfica del pronóstico final
plt.figure(figsize=(12, 6))
plt.plot(expo_serie, label="Datos Históricos (Remesas)", color="blue")
plt.plot(pronostico_nivel, label="Pronóstico (12 meses)", color="orange", linestyle="--")
plt.fill_between(
    pronostico_nivel.index,
    intervalos_nivel.iloc[:, 0],
    intervalos_nivel.iloc[:, 1],
    color="orange",
    alpha=0.2,
    label="Intervalo de Confianza (95%)"
)
plt.title("Pronóstico de Ingresos de Remesas de Trabajadores")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.legend(loc="upper left")
plt.grid(True)
# Guardar gráfico en carpeta de resultados antes de mostrarlo
plt.savefig(RESULTS_DIR / "10_pronostico_remesas.png", dpi=300, bbox_inches="tight")
plt.show()