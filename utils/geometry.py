
def line_equation(x1, y1, x2, y2):
    """
    Return the coefficients of the line equation Ax + By + C = 0
    """
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def line_intersection(A1, B1, C1, A2, B2, C2):
    """
    Return the intersection point of two lines
    """
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None, None
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return x, y


def point_distance(x1, y1, x2, y2):
    """
    Return the distance between two points
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def line_equation_3d(x1, y1, z1, x2, y2, z2):
    """
    Return the coefficients of the line equation Ax + By + Cz + D = 0
    """
    A = y1 * z2 - y2 * z1
    B = z1 * x2 - z2 * x1
    C = x1 * y2 - x2 * y1
    D = -A * x1 - B * y1 - C * z1
    return A, B, C, D