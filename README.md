# LSTM — Selección de features y búsqueda de modelos (progreso)

> Documento de seguimiento del componente LSTM del TFG. Recoge el trabajo hecho hasta
> la fecha: del CSV inicial a la búsqueda masiva de subconjuntos de columnas y su análisis.
> **Es un documento a medias**: la evaluación final en test y el cierre del conjunto de
> features todavía están pendientes (ver §7).

---

## 1. Punto de partida: los datos

Se parte de un CSV con datos diarios de mercado de Ethereum (y Bitcoin como contexto),
con unas ~25 columnas crudas: precios OHLCV de BTC y ETH, dominancias (BTC/ETH/alt),
capitalizaciones, índice Fear & Greed, e indicadores macro (inflación, tipo de la Fed).

Sobre esas columnas crudas se construyó un conjunto amplio de **features derivadas**
(~99 columnas en total), porque las crudas no sirven directamente como entrada (no son
estacionarias, tienen problemas de escala en máximos nuevos, etc.). Entre las derivadas:

- **Retornos**: `eth_close_ret`, `btc_close_ret`, retornos de volumen y de capitalización.
- **Dominancias transformadas**: diferencias diarias y cambios a 14/30/60 días.
- **Macro en variación**: `inflation_chg30`, `fed_rate_chg30` (en lugar de los niveles crudos).
- **Volatilidad**: a 7, 14 y 30 días.
- **Osciladores técnicos**: RSI, estocástico %K/%D, MFI, Bollinger %B y ancho de banda,
  persistencia de sobrecompra/sobreventa.
- **Sentimiento**: el Fear & Greed escalado, ratios a 15/30 días, y los conteos de días
  en cada estado de ánimo (`n_miedo_ext_*`, `n_miedo_*`, `n_neutral_*`, `n_codicia_*`,
  `n_codicia_ext_*` a 15/30/60/90 días), más presiones netas y `fg_presion_*`.
- **Nivel/ciclo**: drawdown de ETH/BTC, distancia a medias móviles (SMA50/200), ratio ETH/BTC.
- **Régimen de mercado**: tres columnas one-hot (`regime_0/1/2`) provenientes del modelo HMM.

El **Fear & Greed** se descompuso en variables one-hot de estado (Extreme Fear, Fear,
Neutral, Greed, Extreme Greed) y en los conteos acumulados mencionados, para capturar
no solo el sentimiento de hoy sino su evolución reciente.

---

## 2. Decisiones metodológicas (cerradas)

Antes de modelar se fijaron varias decisiones que NO se reabren:

- **Objetivo = retorno de ETH** (`eth_close_ret`), no el precio. El retorno es estacionario
  y evita la trampa del random walk; el precio se reconstruye a posteriori si hace falta.
- **El precio crudo nunca entra como feature** (no estacionario). Sí se usa como ingrediente
  para calcular indicadores, pero queda excluido del conjunto de entrada.
- **Horizonte de predicción = 3 días. Ventana de entrada (SEQ_LEN) = 30 días.**
- **Split temporal 70/15/15 cronológico** (nunca aleatorio): train para aprender, validación
  para decidir, test para el número final (se toca una sola vez, al terminar).
- **Escalado ajustado solo con train**: RobustScaler en las continuas, StandardScaler en el
  target; las columnas one-hot de régimen y de sentimiento no se escalan.
- **Receta de entrenamiento**: Huber loss (robusta a outliers de cripto), Adam con weight
  decay, ReduceLROnPlateau, gradient clipping y early stopping que se queda con el mejor
  punto de validación.
- **Métrica direccional = DirAcc** (acierto de signo). No se usa MAPE porque los retornos
  cruzan cero. Objetivo realista: 52–55 %.

---

## 3. El problema de fondo: ¿qué columnas usar, y cuántas?

El conjunto de ~83 features candidatas (las ~99 menos las excluidas y el target) es demasiado
grande para meterlas todas: demasiados parámetros para las ~1.900 secuencias de entrenamiento,
lo que lleva a sobreajuste. Hace falta **seleccionar** un subconjunto.

El reto es que la selección de columnas y la arquitectura de la red están **acopladas**: las
mejores columnas dependen de la red y viceversa. Probar todas las combinaciones es imposible
(elegir subconjuntos de 83 columnas son del orden de 10^17 posibilidades). Por tanto se optó
por una **búsqueda aleatoria masiva** en lugar de fuerza bruta.

---

## 4. La búsqueda masiva de subconjuntos

Se diseñó un experimento que, repetido miles de veces:

1. Elige un número `k` de columnas al azar (entre 10 y 55).
2. Elige `k` columnas al azar del pool de candidatas (sin repetir el mismo conjunto exacto).
3. Entrena ese subconjunto en **5 redes de complejidad creciente** (de una red mínima que
   tiende a quedarse corta, a una máxima que tiende a sobreajustar), midiendo en cada una
   el `val_loss` y el `DirAcc` de validación.

La clave del diseño es entrenar **el mismo subconjunto en las 5 redes**: así se mide la
**robustez**. Un subconjunto que rinde parecido en todas las redes lleva señal real; uno que
solo brilla en una concreta es sospechoso de azar.

El ranking principal es por **`val_loss` medio entre las 5 redes** (más fiable que el DirAcc:
usa muchos más datos y es más estable). El test NO se toca en esta fase.

**Ejecución**: ~6.850 subconjuntos probados (≈34.000 entrenamientos de red), a un ritmo de
~835 subconjuntos/hora. Los resultados se guardan al instante en CSV (resistente a cortes).

---

## 5. Hallazgos de la búsqueda

### 5.1 La dirección a 3 días no es predecible — confirmación empírica del techo
Entre los 100 mejores subconjuntos por `val_loss`, todos caben en un rango de **0.0009**
(de 0.2576 a 0.2585): cien conjuntos distintos prácticamente empatados. Y su DirAcc salta
sin orden entre 0.498 y 0.522. Es decir: **tener buen `val_loss` no implica acertar la
dirección** — entre los buenos modelos, el DirAcc es ruido.

Esto **no es un fracaso**, es un resultado: una demostración empírica, con miles de pruebas,
de que el mercado es esencialmente eficiente respecto a la dirección a 3 días con estos datos.
El cuello de botella es la señal del mercado, no el modelo ni el conjunto de features.

### 5.2 La parsimonia gana (navaja de Occam)
El número de columnas (`k`) apenas importa: en el top-100 conviven `k` de 10 a 33 mezclados
sin que los grandes dominen. Como añadir columnas no mejora el `val_loss`, lo razonable es
quedarse con pocas. No se eligió "10 es el óptimo": se concluyó que el tamaño da igual y por
tanto conviene la simplicidad.

### 5.3 Hallazgo central — dos familias de columnas con roles distintos
Al cruzar las columnas más frecuentes entre los 100 mejores por `val_loss` (magnitud) y los
100 mejores por `DirAcc` (dirección), emergen dos grupos:

- **Predicen la MAGNITUD** (dominan en `val_loss`): `inflation_chg30`, `alt_dominance_diff`,
  `eth_cum_ret_30d`, `eth_mfi14`, `eth_dist_sma50`, `eth_stoch_d`, `eth_bb_width`,
  osciladores RSI y bandas. Son niveles, volatilidad y osciladores técnicos.

- **Predicen la DIRECCIÓN** (suben mucho en `DirAcc`): todo el bloque de **sentimiento** —
  `n_neutral_90d`, `fear_greed_scaled`, `presion_ext_neta_15d`, `n_codicia_15d`,
  `fg_presion_15d/60d`, `n_codicia_ext_60d`, los conteos de miedo/codicia.

Esto encaja con la intuición previa del proyecto: **el sentimiento se asocia al "hacia dónde",
los niveles y la volatilidad al "cuánto".** Es especialmente llamativo que `n_neutral_90d`
(la "calma" acumulada) aparezca en 47 de los 100 mejores por dirección: surgió solo de la
búsqueda, sin imponerlo.

Matiz honesto: la señal de sentimiento es **real pero débil** — aparece de forma consistente,
pero no basta para superar el ruido del DirAcc (todo sigue en ~52 %). El sentimiento *inclina*
la dirección, no la *determina*. Justamente ese hueco es lo que justifica el componente RAG
en la arquitectura del TFG: la señal cuantitativa marca la magnitud y una inclinación débil;
el contexto que la red no puede ver lo aporta el RAG.

### 5.4 Segunda búsqueda (acotada) — el techo se confirma por varios caminos
Tras la búsqueda amplia se hizo una segunda corrida **restringida a un pool de ~24 columnas
consenso** (las que dominaban las frecuencias de magnitud + dirección), con `k` entre 10 y 13.
Objetivo: comprobar si combinar las dos familias aporta, y si concentrar la búsqueda en pocas
columnas baja el techo. Resultados:

- **El techo no se mueve.** El `val_loss` converge a ~0.256–0.257, el mismo valor que con las
  83 columnas. Es decir: **24 columnas bien elegidas igualan a 83.** Acotar no perdió nada
  (refuerza la parsimonia), pero tampoco bajó el suelo (refuerza el techo).
- **Las dos familias conviven (respuesta al "1+1").** Los mejores conjuntos mezclan columnas
  de magnitud y de sentimiento, no tiran de una sola familia. Combinarlas no las penaliza ni
  las dispara: el techo manda, pero la mezcla es lo que forma los conjuntos robustos.
- **Tamaño modal ≈ 11–13 columnas.** Los conjuntos pequeños (en torno a 10–13) dominan
  sistemáticamente el ranking. No se afirma que 11 sea un óptimo matemático (la diferencia con
  10 o 12 es ruido), sino que el rango pequeño es el que manda → parsimonia.
- **Las redes simples ganan.** El top del leaderboard está copado por las arquitecturas más
  pequeñas (una sola capa, pocas neuronas); las grandes apenas aparecen. Añadir capacidad no
  baja el `val_loss`. La arquitectura final será pequeña por este motivo.

Robustez del hallazgo: el techo (~0.256–0.257) aparece igual con 83 o 24 columnas, con 5 redes
o con redes nuevas más pequeñas, con 60 o 100 épocas. Esa **insensibilidad ante tantos cambios**
es, en sí misma, la prueba más fuerte de que el límite lo pone el mercado y no la configuración.
(Nota: los `val_loss` de corridas con conjuntos de redes distintos NO son comparables al
decimal — la media depende de qué redes entran en ella. Lo comparable es el *patrón*, no el
número absoluto. Para reportar el número del techo en la memoria conviene fijar UNA corrida de
referencia.)

---

## 6. Por qué NO se elige el modelo por su DirAcc (nota metodológica importante)

Tentación natural: quedarse con el modelo que dio el DirAcc más alto (algún ~0.56 puntual).
Se descartó deliberadamente, porque sería un error de método:

- Con ~390 secuencias de validación, el DirAcc tiene una banda de ruido de ±3 puntos.
- Elegir el máximo de DirAcc entre miles de modelos selecciona **al afortunado del sorteo**,
  no al mejor: el máximo de muchos intentos ruidosos siempre parece alto aunque no haya señal.
- Se comprueba en los datos: los modelos con DirAcc más alto tienen `val_loss` **peor** que
  el campeón por `val_loss`. Si el acierto fuera real, ambas métricas irían de la mano.
- Cualquier criterio de selección que mire el DirAcc de validación (también métricas
  compuestas tipo `DirAcc + (1 − val_loss)`) hereda ese ruido y se desploma en test.

**Conclusión**: el conjunto de features y la arquitectura se eligen por `val_loss`, robustez
entre redes y parsimonia — criterios que no dependen del sorteo. El DirAcc se mide y se
reporta, pero NO se usa para seleccionar. El número direccional definitivo saldrá del test,
una sola vez, con el modelo ya elegido y contra baselines.

---

## 7. Mejora del modelo de regímenes (HMM)

El HMM de regímenes se había dejado en su primera versión. Se revisó y ajustó para que los
estados coincidieran mejor con las fases reales del mercado. El cambio fue de **variables**,
manteniendo 3 estados:

- **Antes**: `vol_30d`, `cum_ret_30d`, `dist_sma50`, `drawdown`. Problema observado en la
  gráfica: una misma subida larga (p. ej. 2021) salía pintada de varios colores, y algunos
  estados mezclaban subidas y bajadas (el modelo pesaba más la volatilidad que la dirección).
- **Después**: `vol_30d`, `cum_ret_60d`, `dist_sma200`, `drawdown`, `fear_greed_scaled`. Se
  alargaron las ventanas de dirección (60d y SMA200 en vez de 30d y SMA50) para que los estados
  no parpadeen dentro de una fase, y se añadió el sentimiento (`fear_greed_scaled`) para separar
  "caída con pánico" de "bear resignado" y de "euforia".

**Resultado**: los tres estados quedan limpios e interpretables —acumulación/lateral, caídas,
y subidas/euforia— con bloques temporales coherentes (pocos cambios de etiqueta, persistencia
alta). La subida de 2021 queda mayoritariamente de un solo color, y las correcciones internas
de una subida ya no rompen el régimen (quedan integradas en la fase alcista, como se buscaba).

Tiene una componente de criterio (qué se considera "una fase de mercado") pero también
objetiva (menos cambios de etiqueta, mayor persistencia, separación clara de medias por estado).
Para la presentación se puede mostrar la evolución v1 → v2 como ejemplo de refinamiento metodológico.

**Nota de integración**: este HMM mejorado genera un `regimenes.csv` nuevo. Las LSTM NO se
reentrenan con él (el régimen v1 ya daba resultados decentes y su efecto sobre el `val_loss`
es marginal). Se documenta que el análisis de features se hizo con el régimen v1; el HMM v2 es
la versión final y mejoraría marginalmente el resultado si se reentrenara.

---

## 8. Estado actual y próximos pasos

### Hecho
- [x] Construcción del conjunto de ~99 features derivadas a partir del CSV crudo.
- [x] Integración del régimen de mercado (HMM) como one-hot.
- [x] Decisiones metodológicas (target=retorno, split 70/15/15, escalado, receta).
- [x] Búsqueda masiva amplia (~6.850 subconjuntos × 5 redes) con guardado incremental.
- [x] Análisis del top-100 por `val_loss` y por `DirAcc`; cruce de familias de columnas.
- [x] Segunda búsqueda acotada (pool de 24 columnas, redes pequeñas) — techo confirmado.
- [x] Confirmación empírica del techo (~0.256–0.257 de val_loss; DirAcc ~50–52 %) por
      múltiples configuraciones (83 vs 24 columnas, 5 redes vs redes pequeñas, 60 vs 100 épocas).
- [x] Conclusión de parsimonia: tamaño modal ~11–13 columnas, redes de una sola capa.
- [x] Mejora del HMM de regímenes (v2: más peso a dirección + sentimiento, estados limpios).

### Pendiente
- [ ] **Fijar el conjunto de features definitivo** (columnas frecuentes de la búsqueda) y la
      arquitectura (pequeña, una capa), elegidos sin mirar el DirAcc.
- [ ] **Reconstrucción de precio**: encadenar los retornos predichos para graficar precio
      reconstruido vs real (pieza que conecta la LSTM con el RAG).
- [ ] **Evaluación final en test** (una sola vez): `val_loss`, MAE, RMSE y DirAcc del modelo
      elegido, contra las baselines (predecir 0, repetir retorno de ayer, dirección mayoritaria).
- [ ] **Caracterizar los 3 regímenes del HMM** con estadísticas por estado (media de volatilidad,
      retorno, precio, etc.) — material descriptivo para el TFG.
- [ ] **Automatizar el pipeline**: unir los notebooks dispersos (datos, indicadores, régimen)
      en scripts `.py` encadenados que se ejecuten de forma automática y dejen la información
      lista para el RAG.
- [ ] (Opcional) Optuna sobre el candidato final, para rigor en la memoria.