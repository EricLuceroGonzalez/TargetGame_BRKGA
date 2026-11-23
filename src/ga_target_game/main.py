#!/usr/bin/env python3
"""Tiro al blanco: algoritmo genético de llaves aleatorias + pymunk + pygame.

Un proyectil se lanza con un ángulo y una fuerza. El algoritmo evoluciona esos
dos números hasta acertar al blanco.

    Cromosoma = [g0, g1], cada gen en [0, 1]  (llaves aleatorias, como en BRKGA)
    Decodificador: g0 -> ángulo, g1 -> potencia
    Fitness (a minimizar): distancia mínima entre la bola y el blanco durante
    el vuelo, descontando el radio de impacto. 0 significa que acierta.

Uso:
    uv sync          # instala pygame y pymunk (declarados en pyproject.toml)
    uv run python src/ga_target_game/main.py              # con animación y muro
    uv run python src/ga_target_game/main.py --sin-muro   # sin obstáculo: casi trivial
    uv run python src/ga_target_game/main.py --no-gui     # solo evolución, por consola
    uv run python src/ga_target_game/main.py --seed 7 --generations 40
    uv run python src/ga_target_game/main.py --fondo cartoon    # fondo dibujado
    uv run python src/ga_target_game/main.py --solo-mejor       # solo el mejor tiro
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
ANGLE_MIN, ANGLE_MAX = 5.0, 85.0  # grados
POWER_MIN, POWER_MAX = 100.0, 1400.0  # px/s


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
    genes: list[float], wall: bool = False, record: bool = True
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
            # Se actualiza el Space en este intervalo
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
    generations: int = 100,
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
    print(f"seed = {seed}")
    print("===" * 20)
    # print(population)
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


# ── Visualización (pygame) ──────────────────────────────────────────────────
Point = tuple[float, float]

# Color de cada trayectoria según el origen del individuo. El mejor se pinta
# aparte, encima de todos, más grueso y en azul.
ORIGIN_COLORS = {
    "inicial": (150, 150, 165),
    "elite": (225, 165, 30),
    "mutante": (60, 170, 90),
    "cruce": (170, 120, 200),
}
BEST_COLOR = (30, 60, 160)
TEXT_COLOR = (30, 30, 30)


def wall_rect() -> tuple[float, float, float, float]:
    """Rectángulo (x, y, ancho, alto) que ocupa el muro en la física.

    El ``pymunk.Segment`` tiene radio WALL_W/2: mide WALL_W de ancho y sobresale
    WALL_W/2 por arriba del extremo superior.
    """
    top = GROUND_Y - WALL_H - WALL_W / 2
    return (WALL_X - WALL_W / 2, top, WALL_W, GROUND_Y - top)


def build_background(pygame, wall: bool, style: str):
    """Dibuja una sola vez todo lo estático y lo devuelve como Surface.

    En cada fotograma basta con ``screen.blit(fondo, (0, 0))``: copiar una
    imagen ya hecha es mucho más barato que redibujar cielo, nubes y colinas.
    """
    bg = pygame.Surface((WIDTH, HEIGHT)).convert()
    x, y, w, h = wall_rect()

    if style == "simple":
        bg.fill((245, 245, 240))
        pygame.draw.line(bg, (60, 60, 60), (0, GROUND_Y), (WIDTH, GROUND_Y), 3)
        if wall:
            pygame.draw.rect(bg, (90, 90, 90), (x, y, w, h))
        return bg

    # ── estilo "cartoon": todo con primitivas, sin imágenes externas ──
    # Cielo: degradado vertical, una línea por fila.
    top_c, bottom_c = (110, 175, 235), (205, 235, 250)
    for row in range(GROUND_Y):
        t = row / GROUND_Y
        color = [round(a + (b - a) * t) for a, b in zip(top_c, bottom_c, strict=True)]
        pygame.draw.line(bg, color, (0, row), (WIDTH, row))

    pygame.draw.circle(bg, (255, 235, 140), (660, 110), 46)  # sol

    for cx, cy, s in ((150, 140, 1.0), (430, 90, 0.8), (600, 230, 0.6)):  # nubes
        for dx, dy, r in ((0, 0, 30), (32, -12, 36), (68, 0, 28), (34, 10, 30)):
            pygame.draw.circle(bg, (255, 255, 255), (cx + dx * s, cy + dy * s), r * s)

    # Colinas: círculos grandes con el centro bajo el suelo; se ve solo su cima.
    for cx, r, color in (
        (120, 260, (120, 185, 110)),
        (520, 330, (105, 170, 95)),
        (820, 240, (125, 190, 115)),
    ):
        pygame.draw.circle(bg, color, (cx, GROUND_Y + r * 0.55), r)

    # Suelo: franja de pasto y tierra debajo.
    pygame.draw.rect(bg, (135, 95, 60), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    pygame.draw.rect(bg, (90, 170, 70), (0, GROUND_Y - 4, WIDTH, 16))

    # Resortera (decorativa): un poste y dos brazos en el punto de lanzamiento.
    lx, ly = LAUNCH_POS
    wood = (120, 75, 40)
    pygame.draw.line(bg, wood, (lx, GROUND_Y), (lx, ly + 14), 8)
    pygame.draw.line(bg, wood, (lx, ly + 14), (lx - 12, ly - 8), 6)
    pygame.draw.line(bg, wood, (lx, ly + 14), (lx + 12, ly - 8), 6)

    if wall:  # muro de tablones de madera, del tamaño exacto de la física
        pygame.draw.rect(bg, (170, 115, 60), (x, y, w, h))
        for py in range(int(y) + 40, GROUND_Y, 40):
            pygame.draw.line(bg, (125, 80, 40), (x, py), (x + w, py), 2)
        pygame.draw.rect(bg, (100, 60, 30), (x, y, w, h), 3)
    return bg


def run_gui(args: argparse.Namespace) -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tiro al blanco - algoritmo genético")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)
    small = pygame.font.SysFont(None, 20)
    background = build_background(pygame, args.wall, args.fondo)

    # Cronómetro: tiempo real desde que se abre la ventana. Se congela cuando
    # termina la evolución, así queda a la vista cuánto tardó en resolverse.
    start_ms = pygame.time.get_ticks()
    end_ms: int | None = None

    def elapsed() -> str:
        ms = (end_ms if end_ms is not None else pygame.time.get_ticks()) - start_ms
        return f"{ms // 60000:02d}:{ms % 60000 / 1000:04.1f}"

    old_paths: list[list[Point]] = []  # rastro de los mejores anteriores

    def draw_target() -> None:
        pygame.draw.circle(screen, (200, 50, 50), TARGET_POS, TARGET_R)
        pygame.draw.circle(screen, (245, 245, 240), TARGET_POS, TARGET_R // 2)
        pygame.draw.circle(screen, (200, 50, 50), TARGET_POS, TARGET_R // 4)

    def draw_legend(n: int) -> None:
        items = [("mejor", BEST_COLOR)] + [
            (name, color) for name, color in ORIGIN_COLORS.items() if name != "inicial"
        ]
        x0, y0 = WIDTH - 130, 14
        screen.blit(small.render(f"{n} individuos", True, TEXT_COLOR), (x0, y0))
        for i, (name, color) in enumerate(items):
            y = y0 + 20 * (i + 1)
            pygame.draw.circle(screen, color, (x0 + 6, y + 6), 5)
            screen.blit(small.render(name, True, TEXT_COLOR), (x0 + 18, y))

    def draw(paths, origins, upto, gen, best, done) -> None:
        screen.blit(background, (0, 0))
        draw_target()

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

        # Resto de la generación: de peor a mejor, para que los buenos queden
        # encima. paths[0] es el mejor y se dibuja al final.
        for path, origin in reversed(list(zip(paths[1:], origins[1:], strict=True))):
            shown = path[: upto + 1]
            color = ORIGIN_COLORS.get(origin, ORIGIN_COLORS["inicial"])
            if len(shown) > 1:
                pygame.draw.lines(screen, color, False, shown, 1)
            if shown:
                pygame.draw.circle(screen, color, shown[-1], BALL_R // 2 + 1)

        shown = paths[0][: upto + 1]
        if len(shown) > 1:
            pygame.draw.lines(screen, (40, 90, 200), False, shown, 3)
        if shown:
            pygame.draw.circle(screen, BEST_COLOR, shown[-1], BALL_R)

        status = "¡ACIERTO!" if best.cost == 0 else f"error {best.cost:6.1f} px"
        flight = min(upto, len(paths[0]) - 1) * DT  # cada punto = un fotograma
        lines = [
            f"Generación {gen}   semilla {args.seed}",
            f"ángulo {math.degrees(angle):5.1f}°   potencia {power:6.0f}",
            status + ("  (final)" if done else ""),
            f"tiempo {elapsed()}   vuelo {flight:4.2f} s",
            "ESC para salir",
        ]
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, TEXT_COLOR), (14, 12 + 24 * i))
        if len(paths) > 1:
            draw_legend(len(paths))
        pygame.display.flip()

    def replay(gen, best, population, done=False) -> bool:
        """Anima la generación (o solo ``best``). False si el usuario sale."""
        shots = [best] if args.solo_mejor else population  # population[0] is best
        # Volver a simular con record=True para obtener las trayectorias. Es
        # barato (~0.5 ms por tiro) y evita guardar caminos durante evolve().
        paths = [simulate(ind.genes, args.wall, record=True)[1] for ind in shots]
        origins = [ind.origin for ind in shots]
        longest = max(len(p) for p in paths)
        for upto in range(0, longest, args.speed):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return False
            draw(paths, origins, upto, gen, best, done)
            clock.tick(60)
        draw(paths, origins, longest, gen, best, done)
        if not done:
            old_paths.append(paths[0])
        return True

    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Se ejecuta el algoritmo genético
    last = None
    for gen, best, population in evolve(
        args.population, args.generations, wall=args.wall, seed=args.seed
    ):
        logger.info("gen %2d | error %.1f px", gen, best.cost)
        last = (gen, best, population)
        if not replay(gen, best, population):
            pygame.quit()
            return
    # --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    end_ms = pygame.time.get_ticks()  # congela el cronómetro
    gen, best, population = last
    if args.screenshot:
        replay(gen, best, population, done=True)
        pygame.image.save(screen, args.screenshot)
        pygame.quit()
        return

    running = True
    while running:  # repite la última generación hasta cerrar la ventana
        running = replay(gen, best, population, done=True)
        pygame.time.wait(600)
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiro al blanco con algoritmo genético"
    )
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=52)
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
    parser.add_argument(
        "--solo-mejor",
        action="store_true",
        help="anima solo el mejor tiro (no toda la generación)",
    )
    parser.add_argument(
        "--fondo",
        choices=("simple", "cartoon"),
        default="simple",
        help="estilo del fondo",
    )
    args = parser.parse_args()
    args.wall = not args.sin_muro
    # Sin --seed se sortea una, pero se muestra: así cualquier corrida que
    # salga interesante se puede repetir con --seed N.
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(1_000_000)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("semilla %d", args.seed)

    if args.no_gui:
        for gen, best, _ in evolve(
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
