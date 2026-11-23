"""Fondo estático, dibujado una vez."""

from ga_target_game.escena import (
    GROUND_Y,
    HEIGHT,
    LAUNCH_POS,
    WIDTH,
    wall_rect,
)


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
