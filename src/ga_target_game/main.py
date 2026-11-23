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

from ga_target_game.brkga import decode, evolve

logger = logging.getLogger(__name__)


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
        from ga_target_game.gui import run_gui

        run_gui(args)


if __name__ == "__main__":
    main()
