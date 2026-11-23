"""Visualización con pygame."""

from __future__ import annotations

import argparse
import logging
import math

from ga_target_game.brkga import decode, evolve, simulate
from ga_target_game.escena import (
    BALL_R,
    DT,
    HEIGHT,
    LAUNCH_POS,
    TARGET_POS,
    TARGET_R,
    WIDTH,
)
from ga_target_game.fondo import build_background

logger = logging.getLogger(__name__)

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
                pygame.draw.lines(screen, (205, 205, 215), False, old, 1)

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

    last = None
    for gen, best, population in evolve(
        args.population, args.generations, wall=args.wall, seed=args.seed
    ):
        logger.info("gen %2d | error %.1f px", gen, best.cost)
        last = (gen, best, population)
        if not replay(gen, best, population):
            pygame.quit()
            return

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
