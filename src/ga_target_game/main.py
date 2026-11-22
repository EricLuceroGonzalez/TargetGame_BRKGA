#!/usr/bin/env python3
"""Tiro al blanco: algoritmo genético de llaves aleatorias + pymunk + pygame.

Un proyectil se lanza con un ángulo y una fuerza. El algoritmo evoluciona esos
dos números hasta acertar al blanco.

    Cromosoma = [g0, g1], cada gen en [0, 1]  (llaves aleatorias, como en BRKGA)
    Decodificador: g0 -> ángulo, g1 -> potencia
    Fitness (a minimizar): distancia mínima entre la bola y el blanco durante
    el vuelo, descontando el radio de impacto. 0 significa que acierta.

Uso:
    pip install pymunk pygame
    python tiro_al_blanco.py                 # evolución con animación (con muro)
    python tiro_al_blanco.py --sin-muro      # sin obstáculo: casi trivial
    python tiro_al_blanco.py --no-gui        # solo evolución, por consola
    python tiro_al_blanco.py --seed 7 --generations 40
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass

import pymunk

logger = logging.getLogger(__name__)

# ── Escena (coordenadas de pygame: y crece hacia abajo) ─────────────────────
WIDTH, HEIGHT = 1280, 720
GROUND_Y = HEIGHT - 20
GRAVITY = 900.0
DT = 1 / 60
MAX_STEPS = 60 * 8  # un tiro nunca dura más de 8 s simulados
SUBSTEPS = 4  # subpasos por fotograma: evita que la bola atraviese el muro

BALL_R = 10
LAUNCH_POS = (80.0, GROUND_Y - 5.0)
TARGET_POS = (860.0, GROUND_Y - 10.0)
TARGET_R = 20
HIT_DIST = TARGET_R + BALL_R  # a esta distancia del centro, la bola toca el blanco

WALL_X, WALL_H, WALL_W = 420.0, 400.0, 65

# ── Decodificador: genes en [0, 1] -> parámetros reales ─────────────────────
ANGLE_MIN, ANGLE_MAX = 2.0, 85.0  # grados
POWER_MIN, POWER_MAX = 100.0, 1400.0  # px/s


def decode(genes: list[float]) -> tuple[float, float]:
    """Devuelve (ángulo en radianes, potencia en px/s)."""
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
            # Se actualiza el Space en este intervalo
            space.step(DT / SUBSTEPS)
            x, y = body.position
            print(f"x = {x}, y = {y}")
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

    return max(0.0, best - HIT_DIST), path


# ── Algoritmo genético (estilo BRKGA) ───────────────────────────────────────
@dataclass
class Individual:
    genes: list[float]
    cost: float


def evolve(
    pop_size: int = 20,
    generations: int = 100,
    elite_frac: float = 0.2,
    mutant_frac: float = 0.2,
    rho: float = 0.7,
    wall: bool = False,
    seed: int | None = None,
) -> Iterator[tuple[int, Individual]]:
    """Generador: en cada generación entrega (número, mejor individuo).

    Args:
        pop_size: Individuos por generación.
        elite_frac: Fracción de élite, que pasa intacta a la siguiente.
        mutant_frac: Fracción de individuos aleatorios nuevos por generación.
        rho: Probabilidad de que un gen del hijo venga del padre de la élite.
        wall: Si True, hay un muro entre el lanzador y el blanco.
        seed: Semilla para repetir la misma evolución.
    """
    rng = random.Random(seed)
    n_elite = max(1, int(pop_size * elite_frac))
    n_mutants = max(1, int(pop_size * mutant_frac))
    n_genes = 2

    def new_individual(genes: list[float] | None = None) -> Individual:
        genes = genes or [rng.random() for _ in range(n_genes)]
        return Individual(genes, simulate(genes, wall)[0])

    population = [new_individual() for _ in range(pop_size)]
    print("===" * 20)
    # print(population)
    for gen in range(generations + 1):
        population.sort(key=lambda ind: ind.cost)
        best = population[0]
        yield gen, best
        if best.cost == 0.0:
            return  # ya acierta: no hace falta seguir

        elite, rest = population[:n_elite], population[n_elite:]
        next_pop = list(elite)
        next_pop += [new_individual() for _ in range(n_mutants)]
        while len(next_pop) < pop_size:
            parent_e = rng.choice(elite)
            parent_o = rng.choice(rest or elite)
            child = [
                e if rng.random() < rho else o
                for e, o in zip(parent_e.genes, parent_o.genes, strict=True)
            ]
            next_pop.append(new_individual(child))
        population = next_pop


# ── Visualización (pygame) ──────────────────────────────────────────────────
def run_gui(args: argparse.Namespace) -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tiro al blanco - algoritmo genético")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)

    old_paths: list[list[tuple[float, float]]] = []

    def draw(frame_path, upto, gen, best, done) -> None:
        screen.fill((245, 245, 240))
        # Linea horizontal del suelo
        colorFloor = (160, 160, 60)
        pygame.draw.line(
            surface=screen,
            color=colorFloor,
            start_pos=(0, GROUND_Y),
            end_pos=(WIDTH, GROUND_Y),
            width=3,
        )
        # Linea vertical de la pared
        if args.wall:
            pygame.draw.line(
                surface=screen,
                color=colorFloor,
                start_pos=(WALL_X, GROUND_Y),
                end_pos=(WALL_X, GROUND_Y - WALL_H),
                width=8,
            )
        # Se dibuja el Target
        for ring, color in (
            (TARGET_R, (200, 50, 50)),  # anillos del target
            (TARGET_R // 2, (245, 245, 240)),
        ):
            pygame.draw.circle(screen, color, TARGET_POS, ring)
        pygame.draw.circle(screen, (200, 50, 50), TARGET_POS, TARGET_R // 4)

        for old in old_paths:  # rastro tenue de generaciones anteriores
            if len(old) > 1:
                pygame.draw.lines(
                    surface=screen,
                    color=(205, 205, 215),
                    closed=False,
                    points=old,
                    width=1,
                )

        angle, power = decode(best.genes)
        tip = (
            LAUNCH_POS[0] + 40 * math.cos(angle),
            LAUNCH_POS[1] - 40 * math.sin(angle),
        )
        pygame.draw.line(screen, (40, 40, 140), LAUNCH_POS, tip, 6)

        shown = frame_path[: upto + 1]
        if len(shown) > 1:
            pygame.draw.lines(screen, (40, 90, 200), False, shown, 2)
        if shown:
            pygame.draw.circle(screen, (30, 60, 160), shown[-1], BALL_R)

        status = "¡ACIERTO!" if best.cost == 0 else f"error {best.cost:6.1f} px"
        lines = [
            f"Generación {gen}",
            f"ángulo {math.degrees(angle):5.1f}°   potencia {power:6.0f}",
            status + ("  (final)" if done else ""),
            "ESC para salir",
        ]
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, (30, 30, 30)), (14, 12 + 24 * i))
        pygame.display.flip()

    def replay(gen, best, done=False) -> bool:
        """Anima el tiro de ``best``. Devuelve False si el usuario cierra la ventana."""
        _, path = simulate(best.genes, args.wall, record=True)
        for upto in range(0, len(path), args.speed):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return False
            draw(path, upto, gen, best, done)
            clock.tick(60)
        draw(path, len(path), gen, best, done)
        old_paths.append(path)
        return True

    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Se ejecuta el algoritmo genético
    last = None
    for gen, best in evolve(
        args.population, args.generations, wall=args.wall, seed=args.seed
    ):
        logger.info("gen %2d | error %.1f px", gen, best.cost)
        last = (gen, best)
        if not replay(gen, best):
            pygame.quit()
            return
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    gen, best = last
    if args.screenshot:
        replay(gen, best, done=True)
        pygame.image.save(screen, args.screenshot)
        pygame.quit()
        return

    running = True
    while running:  # repite el mejor tiro hasta cerrar la ventana
        running = replay(gen, best, done=True)
        pygame.time.wait(600)
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiro al blanco con algoritmo genético"
    )
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sin-muro", action="store_true", help="quita el muro intermedio"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="solo consola, sin pygame"
    )
    parser.add_argument(
        "--speed", type=int, default=2, help="pasos de simulación por fotograma"
    )
    parser.add_argument(
        "--screenshot", default=None, help="guarda el último fotograma y sale"
    )
    args = parser.parse_args()
    args.wall = not args.sin_muro

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.no_gui:
        for gen, best in evolve(
            args.population, args.generations, wall=args.wall, seed=args.seed
        ):
            angle, power = decode(best.genes)
            logger.info(
                "gen %2d | error %6.1f px | ángulo %5.1f° | potencia %6.0f | semilla %6.0f",
                gen,
                best.cost,
                math.degrees(angle),
                power,
                args.seed,
            )
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
