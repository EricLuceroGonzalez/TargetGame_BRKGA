# ga-target-game

Un juego de **tiro al blanco** donde no apuntas tú: un **algoritmo genético
(BRKGA)** aprende a lanzar un proyectil por encima de un muro hasta acertar.
La física la simula [pymunk](https://www.pymunk.org) y la animación,
[pygame](https://www.pygame.org).

Es un proyecto para aprender tres cosas a la vez: metaheurísticas, física 2D y
programación de videojuegos en Python.

```
  🎯  Cada generación lanza 20 tiros. Los mejores sobreviven,
      se cruzan entre sí y, poco a poco, la trayectoria
      "aprende" a pasar sobre el muro y caer en el blanco.
```

---

## Inicio rápido

Requiere Python 3.12+ y [uv](https://docs.astral.sh/uv/).

```bash
make setup                                   # crea el entorno e instala dependencias
uv run python src/ga_target_game/main.py     # ¡a jugar!
```

## Uso

```bash
uv run python src/ga_target_game/main.py [opciones]
```

| Opción | Por defecto | Qué hace |
|---|---|---|
| `--population N` | `20` | Individuos por generación |
| `--generations N` | `30` | Generaciones máximas (se detiene antes si acierta) |
| `--seed N` | al azar | Semilla. Siempre se muestra en pantalla para repetir la corrida |
| `--sin-muro` | — | Quita el muro (el problema se vuelve casi trivial) |
| `--solo-mejor` | — | Anima solo el mejor tiro, no toda la generación |
| `--fondo {simple,cartoon}` | `simple` | Estilo del escenario |
| `--speed N` | `2` | Pasos de simulación por fotograma (más alto = más rápido) |
| `--no-gui` | — | Solo consola: evoluciona y muestra el progreso |
| `--screenshot ARCHIVO` | — | Guarda el último fotograma como imagen y sale |

Ejemplos:

```bash
uv run python src/ga_target_game/main.py --fondo cartoon --seed 3
uv run python src/ga_target_game/main.py --no-gui --seed 7 --generations 40
```

**Controles:** `ESC` o cerrar la ventana para salir.

### Qué se ve en pantalla

- **Trayectorias de toda la generación**, coloreadas por origen:
  🔵 mejor · 🟠 élite · 🟢 mutante · 🟣 cruce.
- **Rastro gris** de los mejores tiros de generaciones anteriores.
- **Marcador:** generación, semilla, ángulo, potencia, error (px al blanco),
  cronómetro real y tiempo de vuelo simulado.

---

## Cómo funciona

### El problema

Un tiro queda definido por dos números: **ángulo** (2°–85°) y **potencia**
(300–1400 px/s). Hay que encontrar una combinación que pase sobre el muro y
toque el blanco.

### El costo (fitness, a minimizar)

Se simula el vuelo con pymunk y se mide la **distancia mínima** entre la bola y
el blanco durante toda la trayectoria, descontando el radio de impacto.
**0 = acierto.** Medir el mínimo, y no solo dónde cae la bola, da una señal
suave: un tiro que "casi" pasa por el blanco puntúa mejor que uno lejano.

### El algoritmo: BRKGA

*Biased Random-Key Genetic Algorithm*. Cada individuo es un vector de
**llaves aleatorias** en `[0, 1]`, y un **decodificador** las convierte en
ángulo y potencia. El algoritmo nunca sabe nada de física, solo maneja llaves.

```
población inicial (llaves al azar)
        │
        ▼
┌─► ordenar por costo ──► ¿acierto? ──sí──► fin
│        │ no
│        ▼
│   ÉLITE (20 %)    → pasa intacta
│   MUTANTES (20 %) → individuos nuevos al azar (única fuente de diversidad)
│   CRUCE (resto)   → padre de élite × padre no-élite;
│                     cada gen viene del de élite con prob. ρ = 0.7
└────────┘
```

Lo que lo distingue de un GA clásico es la **selección sesgada** (un padre
siempre de la élite) y que **no hay mutación gen a gen**: la exploración viene
solo de los mutantes.

> **Limitación conocida:** con solo 2 genes, el cruce uniforme genera apenas 4
> combinaciones y no crea valores nuevos. Por eso a veces el progreso se
> estanca durante varias generaciones. Ver *Ideas a futuro*.

### ¿Y otros algoritmos?

Comparación con el mismo presupuesto (500 simulaciones), con muro y semillas
0–29. Los prototipos de GA y SA **no están calibrados a fondo**:

| Algoritmo | Aciertos | Simulaciones medias hasta acertar |
|---|---|---|
| BRKGA (el del juego) | 20/30 | 233 |
| GA clásico (torneo + BLX-α + mutación gaussiana) | 26/30 | 180 |
| Simulated Annealing | 28/30 | 135 |

Para un problema continuo de 2 parámetros, los operadores que **mezclan
valores** (BLX-α, vecindario gaussiano) le ganan al BRKGA. El BRKGA brilla en
problemas combinatorios, como permutaciones o asignaciones, donde su
decodificador marca la diferencia.

---

## Estructura

```
src/ga_target_game/
  main.py            juego: decodificador, simulación, BRKGA y visualización
  logging_setup.py   configuración de logging a archivo y consola
  __main__.py        punto de entrada del paquete (aún no lanza el juego)
tests/unit/          tests unitarios
tests/integration/   tests de integración
docs/diagrams/       diagramas UML generados
logs/                logs en tiempo de ejecución (no se versiona)
```

## Desarrollo

```bash
make test      # tests con cobertura
make lint      # ruff (sin modificar archivos)
make format    # ruff: formato + orden de imports
make check     # lint + tests, lo mismo que en CI
make diagrams  # UML con pyreverse (necesita graphviz)
make help      # lista todos los comandos
```

Convención de logging: cada módulo usa `logger = logging.getLogger(__name__)`
y la configuración se hace **una sola vez**, en el punto de entrada.

> pygame y pymunk son extensiones en C. `pyproject.toml` incluye
> `extension-pkg-allow-list` para que pylint no marque falsos positivos como
> `Module 'pygame' has no 'init' member`.

## Ideas a futuro

- [ ] Varias llaves por parámetro (sigue siendo BRKGA, pero el cruce ya crea valores nuevos)
- [ ] Elegir el algoritmo desde la línea de comandos: `--algoritmo {brkga,ga,sa}`
- [ ] Que `python -m ga_target_game` lance el juego
- [ ] Tests del decodificador, de la simulación y de la evolución
- [ ] Viento o blancos móviles para hacerlo más difícil
