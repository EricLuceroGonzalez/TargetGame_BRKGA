"""Escena: constantes compartidas por la física y el dibujo."""

# ── Escena (coordenadas de pygame: y crece hacia abajo) ─────────────────────
WIDTH, HEIGHT = 800, 800
GROUND_Y = 660
GRAVITY = 800.0
DT = 1 / 60
MAX_STEPS = 60 * 8  # un tiro nunca dura más de 8 s simulados
SUBSTEPS = 4  # subpasos por fotograma: evita que la bola atraviese el muro

BALL_R = 8
LAUNCH_POS = (80.0, GROUND_Y - 20.0)
TARGET_POS = (760.0, GROUND_Y - 10.0)
TARGET_R = 10
HIT_DIST = TARGET_R + BALL_R  # a esta distancia del centro, la bola toca el blanco

WALL_X, WALL_H, WALL_W = 420.0, 500.0, 35


def wall_rect() -> tuple[float, float, float, float]:
    """Rectángulo (x, y, ancho, alto) que ocupa el muro en la física.

    El ``pymunk.Segment`` tiene radio WALL_W/2: mide WALL_W de ancho y sobresale
    WALL_W/2 por arriba del extremo superior.
    """
    top = GROUND_Y - WALL_H - WALL_W / 2
    return (WALL_X - WALL_W / 2, top, WALL_W, GROUND_Y - top)
