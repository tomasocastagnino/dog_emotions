# Trabajo práctico original (crédito de origen)

Este notebook es el trabajo práctico final de *Introducción a la Inteligencia
Artificial* (UNR) que le dio origen a este proyecto: entrena InceptionV3 por transfer
learning + fine-tuning para clasificar las mismas 4 emociones caninas, con una
metodología bastante completa (barrido de capas/bloques/convoluciones descongeladas,
k-fold estratificado, matrices de confusión) — llega a 76% de accuracy en test con un
modelo de 124 MB.

Autores: Santiago Bussanich, Tomás Castagnino, Gaspar Giménez.

Se conserva acá tal cual, sin modificaciones, solo como referencia y crédito de
origen. No hace falta correrlo para nada de lo que sigue en este repo — el
entrenamiento activo ahora vive en [`../training/train_lightweight.py`](../training/train_lightweight.py),
que es independiente de este notebook. Ver el [README principal](../README.md) para
el estado y el setup del proyecto actual.
