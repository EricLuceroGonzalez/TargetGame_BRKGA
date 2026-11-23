"""BRKGA: decodificador, simulación (costo) y evolución."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass

import pymunk

from ga_target_game.escena import (
    BALL_R,
    DT,
    GRAVITY,
    GROUND_Y,
    HIT_DIST,
    LAUNCH_POS,
    MAX_STEPS,
    SUBSTEPS,
    TARGET_POS,
    WALL_H,
    WALL_W,
    WALL_X,
    WIDTH,
)

# ── Decodificador: genes en [0, 1] -> parámetros reales ─────────────────────
ANGLE_MIN, ANGLE_MAX = 2.0, 85.0  # grados
POWER_MIN, POWER_MAX = 300.0, 1400.0  # px/s


def decode(genes: list[float]) -> tuple[float, float]:
    """Devuelve (ángulo en radianes, potencia en px/s)."""
    # [BRKGA] DECODIFICADOR: es la única pieza que conoce el problema.
    # El algoritmo genético solo manipula llaves aleatorias en [0, 1]; aquí se
    # traducen a una solución real mediante un escalado lineal a cada rango.
    # Así cualquier hijo (cruce o mutante) es siempre una solución factible.
    angle = math.radians(ANGLE_MIN + genes[0] * (ANGLE_MAX - ANGLE_MIN))
    power = POWER_MIN + genes[1] * (POWER_MAX - POWER_MIN)
    return angle, power


# ── Simulación física (pymunk) ──────────────────────────────────────────────
def simulate(
    genes: list[float], wall: bool = False, record: bool = False
) -> tuple[float, list[tuple[float, float]]]:
    """Simula un tiro. Devuelve (fitness, trayectoria).

    La trayectoria solo se guarda si ``record=True`` (para dibujarla). Durante la
    evolución se llama sin grabar, que es bastante más rápido.
    """
    # [BRKGA] FUNCIÓN DE EVALUACIÓN: cromosoma -> decodificar -> simular -> costo.
    angle, power = decode(genes)

    space = pymunk.Space()
    space.gravity = (0, GRAVITY)

    body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, BALL_R))
    body.position = LAUNCH_POS
    body.velocity = (power * math.cos(angle), -power * math.sin(angle))
    shape = pymunk.Circle(body, BALL_R)
    shape.elasticity = 0.4
    space.add(body, shape)

    if wall:
        segment = pymunk.Segment(
            space.static_body,
            (WALL_X, GROUND_Y),
            (WALL_X, GROUND_Y - WALL_H),
            WALL_W / 2,
        )
        segment.elasticity = 0.4
        space.add(segment)

    best = math.inf
    path: list[tuple[float, float]] = []
    finished = False
    for _ in range(MAX_STEPS):
        for _ in range(SUBSTEPS):
            space.step(DT / SUBSTEPS)
            x, y = body.position
            # Fitness = distancia MÍNIMA al blanco en toda la trayectoria. Medirla
            # en cada subpaso da una señal suave: un tiro que "casi" pasa por el
            # blanco puntúa mejor que uno lejano, aunque ninguno acierte.
            best = min(best, math.hypot(x - TARGET_POS[0], y - TARGET_POS[1]))
            if best <= HIT_DIST:  # acierto
                finished = True
            elif y >= GROUND_Y - BALL_R - 0.5 or not -50 < x < WIDTH + 50:
                finished = True  # tocó el suelo o salió de la pantalla
            if finished:
                break
        if record:
            path.append((x, y))
        if finished:
            break

    # Se descuenta el radio de impacto: costo 0.0 = acierto (óptimo global).
    return max(0.0, best - HIT_DIST), path


# ── Algoritmo genético (estilo BRKGA) ───────────────────────────────────────
@dataclass
class Individual:
    genes: list[float]
    cost: float
    # De dónde salió en esta generación (solo para visualizarlo, el algoritmo
    # no lo usa): "inicial" | "elite" | "mutante" | "cruce".
    origin: str = "inicial"


def evolve(
    pop_size: int = 20,
    generations: int = 30,
    elite_frac: float = 0.2,
    mutant_frac: float = 0.2,
    rho: float = 0.7,
    wall: bool = False,
    seed: int | None = None,
) -> Iterator[tuple[int, Individual, list[Individual]]]:
    """Generador: en cada generación entrega (número, mejor, población ordenada).

    La población va ordenada de mejor a peor, así que ``population[0] is best``.

    Args:
        pop_size: Individuos por generación.
        elite_frac: Fracción de élite, que pasa intacta a la siguiente.
        mutant_frac: Fracción de individuos aleatorios nuevos por generación.
        rho: Probabilidad de que un gen del hijo venga del padre de la élite.
        wall: Si True, hay un muro entre el lanzador y el blanco.
        seed: Semilla para repetir la misma evolución.
    """
    rng = random.Random(seed)  # RNG propio: con la misma semilla, misma evolución
    # [BRKGA] PARTICIÓN DE LA POBLACIÓN (p = pop_size):
    #   p_e = n_elite   -> mejores individuos, se copian sin cambios (elitismo)
    #   p_m = n_mutants -> individuos aleatorios nuevos (única fuente de diversidad)
    #   p - p_e - p_m   -> hijos de cruce élite × no-élite
    n_elite = max(1, int(pop_size * elite_frac))
    n_mutants = max(1, int(pop_size * mutant_frac))
    n_genes = 2  # tamaño del cromosoma: [llave del ángulo, llave de la potencia]

    def new_individual(
        genes: list[float] | None = None, origin: str = "inicial"
    ) -> Individual:
        # Sin genes -> vector de llaves aleatorias uniformes en [0, 1).
        # Se evalúa al crearlo, así cada individuo guarda su costo.
        genes = genes or [rng.random() for _ in range(n_genes)]
        return Individual(genes, simulate(genes, wall)[0], origin)

    # [BRKGA] POBLACIÓN INICIAL: p vectores de llaves aleatorias.
    population = [new_individual() for _ in range(pop_size)]

    for gen in range(generations + 1):
        # [BRKGA] CLASIFICACIÓN: ordenar por costo (minimización).
        population.sort(key=lambda ind: ind.cost)
        best = population[0]
        yield gen, best, population
        # CRITERIO DE PARADA: acierto (costo 0) o agotar las generaciones.
        if best.cost == 0.0:
            return  # ya acierta: no hace falta seguir

        # [BRKGA] Separar élite (los p_e mejores) del resto (no-élite).
        elite, rest = population[:n_elite], population[n_elite:]
        # [BRKGA] ELITISMO: la élite pasa intacta; no se re-evalúa porque la
        # simulación es determinista. Garantiza que el mejor nunca empeora.
        next_pop = [Individual(ind.genes, ind.cost, "elite") for ind in elite]
        # [BRKGA] MUTANTES: individuos totalmente aleatorios. BRKGA no tiene
        # operador de mutación gen a gen; la exploración viene de aquí.
        next_pop += [new_individual(origin="mutante") for _ in range(n_mutants)]
        # [BRKGA] CRUCE: completa la población con hijos.
        while len(next_pop) < pop_size:
            # Selección SESGADA: un padre siempre de la élite y otro del resto
            # (en un GA clásico ambos padres saldrían de toda la población).
            parent_e = rng.choice(elite)
            parent_o = rng.choice(rest or elite)
            # Cruce uniforme parametrizado: cada gen viene del padre élite con
            # probabilidad rho (> 0.5), así el hijo hereda más del mejor padre.
            # Nota: con 2 genes solo hay 4 combinaciones posibles; el cruce no
            # crea valores nuevos, solo recombina los existentes.
            child = [
                e if rng.random() < rho else o
                for e, o in zip(parent_e.genes, parent_o.genes, strict=True)
            ]
            next_pop.append(new_individual(child, "cruce"))
        # [BRKGA] REEMPLAZO GENERACIONAL: élite + mutantes + hijos.
        population = next_pop
