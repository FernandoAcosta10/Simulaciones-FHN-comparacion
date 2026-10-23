# ============================================================
# Comparación dinámica de sistemas FitzHugh--Nagumo
# ============================================================

# ─── Importaciones generales ────────────────────────────────
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.integrate import solve_ivp

# ─── Gráficas ────────────────────────────────────────────────
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap, BoundaryNorm

# ─── Exportación ────────────────────────────────────────────────
from pathlib import Path
CARPETA_PROYECTO = Path(__file__).resolve().parent    # Este archivo
CARPETA_IMAGENES = CARPETA_PROYECTO / "imagenes"      # Para figuras
CARPETA_IMAGENES.mkdir(parents=True, exist_ok=True)   # Si no existe

# ─── Configuración LaTeX para Matplotlib ─────────────────────
def configurar_latex(usar_latex=False):
    """
    Configura el motor de renderizado de texto matemático.

    Parámetros
    ----------
    usar_latex : bool
        False: utiliza Mathtext (motor interno de Matplotlib).
        True: utiliza una instalación externa de LaTeX.
    """
    plt.rcParams.update({
        "text.usetex": usar_latex,
        "font.family": "serif",
    })



class ComparadorFHN:
    # ──────────────────────────────────────────────────
    #         Calcular parámetros a, b y eps
    # ──────────────────────────────────────────────────
    """
    Evalúa los parámetros reducidos del modelo FHN-Barrett:
        a(C), b(C), epsilon(C)
    usando los ajustes cuadráticos en
        s(C) = C/(C+1).
    """
    def parametros_FHN_Barrett(self, GC):
        GC = np.asarray(GC, dtype=float)
        s = GC/(GC + 1.0)

        a = (
            -0.61824408
            + 0.89923359*s
            - 0.16597666*s**2
        )
        b = (
            0.51584921
            - 0.28997508*s
            + 0.05198674*s**2
        )
        epsilon = (
            0.02865784
            + 0.01519036*s
            + 0.01700884*s**2
        )

        return a, b, epsilon

    def __init__(self):
        """
        Clase para comparar sistemas tipo FitzHugh--Nagumo.

        Cada sistema se representará mediante un diccionario con la forma:
            {
                "nombre": "FHN clásico",
                "a": 0.7,
                "b": 0.8,
                "epsilon": 0.08,
                "I": 0.5
            }

        o bien:
            {
                "nombre": "FHN-Barrett GC=1",
                "a": a_GC,
                "b": b_GC,
                "epsilon": epsilon_GC,
                "I": I
            }
        """
        pass


    def nulclina_voltaje(self, u, I):
        return u - (u**3)/3 + I


    def nulclina_recuperacion(self, u, a, b):
        return (u - a)/b

    # ──────────────────────────────────────────────────
    #             Comparación de ceroclinas
    # ──────────────────────────────────────────────────
        """
        Compara las nulclinas de dos o más sistemas FHN.

        Parámetros
        ----------
        sistemas : list[dict]
            Lista de sistemas. Cada sistema debe tener las claves:

                "nombre", "a", "b", "epsilon", "I"

        u_min, u_max : float
            Intervalo de valores de u.

        num : int
            Número de puntos para graficar.

        mostrar_cubicas_individuales : bool
            Si False, grafica una sola cúbica cuando todos los sistemas
            tienen el mismo valor de I.
            Si True, grafica una cúbica por cada sistema.

        titulo : str
            Título de la figura.

        figsize : tuple
            Tamaño de la figura.

        guardar : bool
            Si True, guarda la figura.

        ruta : str o None
            Carpeta donde se guardará la figura.

        nombre_archivo : str
            Nombre del archivo PDF.
        """
    def comparar_nulclinas(
        self,
        sistemas,
        u_min=-3,
        u_max=3,
        num=1000,
        mostrar_cubicas_individuales=False,
        titulo="Comparación de nulclinas FHN",
        figsize=(8, 6),
        guardar=False,
        ruta=CARPETA_IMAGENES,
        nombre_archivo="FHNcomp_comparacion_nulclinas.pdf"
    ):
        if not isinstance(sistemas, list):
            raise TypeError("sistemas debe ser una lista de diccionarios.")

        if len(sistemas) < 2:
            raise ValueError("Debes proporcionar al menos dos sistemas.")

        for sistema in sistemas:
            for clave in ["nombre", "a", "b", "epsilon", "I"]:
                if clave not in sistema:
                    raise KeyError(
                        f"Cada sistema debe contener la clave '{clave}'."
                    )

        u = np.linspace(u_min, u_max, num)

        plt.figure(figsize=figsize)

        # ── ── ── ── ── ── ── ── ── ── ── ── ──
        #     Graficar nulclina de voltaje
        # ── ── ── ── ── ── ── ── ── ── ── ── ──
        I_vals = np.array([sistema["I"] for sistema in sistemas], dtype=float)
        todos_I_iguales = np.allclose(I_vals, I_vals[0])

        if todos_I_iguales and not mostrar_cubicas_individuales:

            I = I_vals[0]

            w_v = self.nulclina_voltaje(u, I)

            plt.plot(
                u,
                w_v,
                linewidth=2.5,
                color="black",
                label=r"$w = u-u^3/3+I$"
            )

        else:

            for sistema in sistemas:

                I = sistema["I"]
                nombre = sistema["nombre"]

                w_v = self.nulclina_voltaje(u, I)

                plt.plot(
                    u,
                    w_v,
                    linewidth=2,
                    linestyle="-",
                    label=rf"Voltaje: {nombre}"
                )

        # ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        #    Graficar nulclinas de recuperación
        # ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        estilos_linea = ["-", "--", "-.", ":"]
        for i, sistema in enumerate(sistemas):
            nombre = sistema["nombre"]
            a = sistema["a"]
            b = sistema["b"]
            w_r = self.nulclina_recuperacion(u, a, b)
            estilo_actual = estilos_linea[i % len(estilos_linea)]
            plt.plot(
                u,
                w_r,
                linewidth=2,
                linestyle=estilo_actual,
                label=rf"{nombre}"
            )

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        #               Formato
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        plt.xlabel(r"$u$")
        plt.ylabel(r"$w$")
        plt.title(titulo, fontsize=20)
        plt.grid(alpha=0.3)
        plt.legend(
            frameon=True,
            fancybox=False,
            edgecolor="black",
            fontsize=9
        )
        plt.tight_layout()
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        if guardar:
            if ruta is None:
                ruta_completa = Path(nombre_archivo)
            else:
                ruta_completa = Path(ruta) / nombre_archivo
            ruta_completa.parent.mkdir(parents=True,exist_ok=True)
            plt.savefig(ruta_completa, bbox_inches="tight", format="pdf")

        plt.show()

        return None

    # ──────────────────────────────────────────────────
    #            Comparación de equilibrios
    # ──────────────────────────────────────────────────

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #                  Jacobiano
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Calcula la matriz Jacobiana del sistema FHN:

            du/dt = u - u^3/3 - w + I
            dw/dt = epsilon(u - a - bw)

        en un punto con coordenada u.

        La matriz Jacobiana es:

            J = [[1-u^2, -1],
                [epsilon, -epsilon*b]]
        """
    def jacobiano(self, u, b, epsilon):
        """
        Calcula la matriz Jacobiana del sistema FHN:

            du/dt = u - u^3/3 - w + I
            dw/dt = epsilon*(u - a - b*w)

        respecto a las variables (u,w).

        J = [[1-u^2, -1],
            [epsilon, -epsilon*b]]
        """
        J = np.array([
            [1.0 - u**2, -1.0],
            [epsilon, -epsilon*b]
        ])

        return J

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #              Traza y determinante
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Calcula la traza y el determinante del Jacobiano.
        """
    def traza_determinante(self, u, b, epsilon):
        traza = 1.0 - u**2 - epsilon*b
        determinante = epsilon*(1.0 - b + b*u**2)
        return traza, determinante

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #             Clasificar equilibrios
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Clasifica la estabilidad local de un equilibrio usando traza,
        determinante y autovalores.
        """
    def clasificar_equilibrio(self, u, a, b, epsilon, tol=1e-8):
        J = self.jacobiano(
            u=u,
            b=b,
            epsilon=epsilon
        )

        autovalores = np.linalg.eigvals(J)
        traza, determinante = self.traza_determinante(
            u=u,
            b=b,
            epsilon=epsilon
        )

        discriminante = traza**2 - 4.0*determinante
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Caso 1: Determinante negativo
        if determinante < -tol:
            tipo = "silla"
            estabilidad = "inestable"

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Caso 2: Determinante cercano a cero
        elif abs(determinante) <= tol:
            tipo = "no hiperbólico"
            estabilidad = "indeterminada"

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Caso 3: Determinante positivo
        else:
            if traza < -tol:
                estabilidad = "estable"
                if discriminante > tol:
                    tipo = "nodo estable"
                elif discriminante < -tol:
                    tipo = "foco estable"
                else:
                    tipo = "nodo degenerado estable"

            elif traza > tol:
                estabilidad = "inestable"
                if discriminante > tol:
                    tipo = "nodo inestable"
                elif discriminante < -tol:
                    tipo = "foco inestable"
                else:
                    tipo = "nodo degenerado inestable"

            else:
                estabilidad = "no hiperbólico"
                if determinante > tol:
                    tipo = "centro lineal / candidato a Hopf"
                else: # Realmente no se ejecuta este
                    tipo = "no hiperbólico"

        resultado = {
            "J": J,
            "autovalores": autovalores,
            "traza": traza,
            "determinante": determinante,
            "discriminante": discriminante,
            "tipo": tipo,
            "estabilidad": estabilidad
        }

        return resultado

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #             Cálculo de equilibrios
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Calcula los equilibrios de un sistema FHN. Las nulclinas son:
            w = u - u^3/3 + I
        y
            w = (u-a)/b.
        Igualando:
            u - u^3/3 + I = (u-a)/b.
        Equivalentemente:
    
            -(b/3)u^3 + (b-1)u + (bI+a) = 0.
        """
    def equilibrios_sistema(self, a, b, epsilon, I, tol_imag=1e-8):
        coeficientes = [
            -b/3.0,
            0.0,
            b - 1.0,
            b*I + a
        ]
        raices = np.roots(coeficientes)
        equilibrios = []
        for raiz in raices:
            # Si la parte imaginaria es pequeña
            if abs(np.imag(raiz)) < tol_imag:
                u_star = float(np.real(raiz))
                w_star = self.nulclina_voltaje(u_star, I)
                estabilidad = self.clasificar_equilibrio(
                    u=u_star,
                    a=a,
                    b=b,
                    epsilon=epsilon
                )
                equilibrio = {
                    "u": u_star,
                    "w": w_star,
                    "a": a,
                    "b": b,
                    "epsilon": epsilon,
                    "I": I,
                    "traza": estabilidad["traza"],
                    "determinante": estabilidad["determinante"],
                    "discriminante": estabilidad["discriminante"],
                    "autovalores": estabilidad["autovalores"],
                    "tipo": estabilidad["tipo"],
                    "estabilidad": estabilidad["estabilidad"]
                }
                equilibrios.append(equilibrio)
        # Si resulta un sistema con tres equilibrios
        equilibrios = sorted(equilibrios, key=lambda eq: eq["u"])

        return equilibrios

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #             Comparación de equilibrios
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Calcula y compara los equilibrios de varios sistemas FHN.
        Cada sistema debe tener las claves:
            "nombre", "a", "b", "epsilon", "I".
        """
    def comparar_equilibrios(
        self,
        sistemas,
        imprimir=True
    ):
        resultados = {}

        for sistema in sistemas:
            nombre = sistema["nombre"]
            a = sistema["a"]
            b = sistema["b"]
            epsilon = sistema["epsilon"]
            I = sistema["I"]
            equilibrios = self.equilibrios_sistema(
                a=a,
                b=b,
                epsilon=epsilon,
                I=I
            )
            resultados[nombre] = equilibrios

            if imprimir:
                print("Sistema:", nombre)
                print("a =", a)
                print("b =", b)
                print("epsilon =", epsilon)
                print("I =", I)
                print("Número de equilibrios:", len(equilibrios))

                for j, eq in enumerate(equilibrios):
                    print(f"  Equilibrio {j+1}")
                    print(f"    u* = {eq['u']:.8f}")
                    print(f"    w* = {eq['w']:.8f}")
                    print(f"    traza = {eq['traza']:.8f}")
                    print(f"    determinante = {eq['determinante']:.8f}")
                    print(f"    discriminante = {eq['discriminante']:.8f}")
                    print(f"    autovalores = {eq['autovalores']}")
                    print(f"    tipo = {eq['tipo']}")
                    print(f"    estabilidad = {eq['estabilidad']}")
                print()

        return resultados

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #             Graficación de equilibrios
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    def graficar_equilibrios(
        self,
        sistemas,
        u_min=-3,
        u_max=3,
        num=1000,
        titulo="Comparación de equilibrios FHN",
        figsize=(8, 6),
        guardar=False,
        ruta=CARPETA_IMAGENES,
        nombre_archivo="FHNcomp_comparacion_equilibrios.pdf"
    ):
        u = np.linspace(u_min, u_max, num)
        plt.figure(figsize=figsize)
        I_vals = np.array([sistema["I"] for sistema in sistemas], dtype=float)

        todos_I_iguales = np.allclose(I_vals, I_vals[0])
        if todos_I_iguales:
            I = I_vals[0]
            plt.plot(
                u,
                self.nulclina_voltaje(u, I),
                color="black",
                linewidth=2.5,
                label=r"$w = u - u^3/3 + I$"
            )

        else:
            for sistema in sistemas:
                I = sistema["I"]
                nombre = sistema["nombre"]
                plt.plot(
                    u,
                    self.nulclina_voltaje(u, I),
                    linewidth=2,
                    label=rf"Voltaje: {nombre}"
                )

        estilos_linea = ["-", "--", "-.", ":"]
        for i, sistema in enumerate(sistemas):
            nombre = sistema["nombre"]
            a = sistema["a"]
            b = sistema["b"]
            epsilon = sistema["epsilon"]
            I = sistema["I"]
            w_rec = self.nulclina_recuperacion(u, a, b)
            estilo_actual = estilos_linea[i % len(estilos_linea)]
            plt.plot(
                u,
                w_rec,
                linestyle=estilo_actual,
                linewidth=2,
                label=rf"{nombre}"
            )
            equilibrios = self.equilibrios_sistema(
                a=a,
                b=b,
                epsilon=epsilon,
                I=I
            )

            estilos_punto = ["*", "o", "s", "D"]
            for eq in equilibrios:
                marcador = estilos_punto[i % len(estilos_punto)]
                plt.plot(
                    eq["u"],
                    eq["w"],
                    marker=marcador,
                    markersize=8,
                    linestyle="None",
                    label=rf"Equilibrio"
                )
        plt.xlabel(r"$u$")
        plt.ylabel(r"$w$")
        plt.title(titulo, fontsize=20)
        plt.grid(alpha=0.3)
        plt.legend(
            frameon=True,
            fancybox=False,
            edgecolor="black",
            fontsize=8
        )
        plt.tight_layout()

        if guardar:
            if ruta is None:
                ruta_completa = Path(nombre_archivo)
            else:
                ruta_completa = Path(ruta) / nombre_archivo
            ruta_completa.parent.mkdir(parents=True,exist_ok=True)
            plt.savefig(ruta_completa, bbox_inches="tight", format="pdf")
        plt.show()

        return None

    # ──────────────────────────────────────────────────
    #           Calcular los posibles Hopf
    # ──────────────────────────────────────────────────
        """
        Calcula los posibles puntos de Hopf de un sistema FHN
        al variar la corriente externa I.
        La condición de Hopf es:
            traza = 0
            determinante > 0
        Como:
            traza = 1 - u^2 - epsilon*b,
        se obtiene:
            u_H = +-sqrt(1 - epsilon*b).
        Luego:
            I_H = (u_H-a)/b - u_H + u_H^3/3.
        """
    def puntos_hopf_sistema(self, a, b, epsilon, tol=1e-10):
        cantidad = 1.0 - epsilon*b
        if cantidad < -tol:
            return []
        if abs(cantidad) <= tol:
            candidatos_u = [0.0]
        else:
            raiz = np.sqrt(cantidad)
            candidatos_u = [-raiz, raiz]

        puntos_hopf = []
        for u_H in candidatos_u:
            I_H = (
                (u_H - a)/b
                - u_H
                + (u_H**3)/3.0
            )
            w_H = self.nulclina_voltaje(
                u_H,
                I_H
            )
            traza, determinante = self.traza_determinante(
                u=u_H,
                b=b,
                epsilon=epsilon
            )
            # Derivada de la traza respecto a I sobre la rama de equilibrios.
            # La condición de transversalidad requiere que no sea cero.
            denominador = b*(1.0 - u_H**2) - 1.0
            if abs(denominador) > tol:
                du_dI = -b/denominador
                dtraza_dI = -2.0*u_H*du_dI
            else:
                du_dI = np.nan
                dtraza_dI = np.nan
            if determinante > tol:
                es_hopf = True
            else:
                es_hopf = False

            punto = {
                "u_H": u_H,
                "w_H": w_H,
                "I_H": I_H,
                "traza": traza,
                "determinante": determinante,
                "du_dI": du_dI,
                "dtraza_dI": dtraza_dI,
                "transversalidad": abs(dtraza_dI) > tol if np.isfinite(dtraza_dI) else False,
                "es_hopf": es_hopf
            }
            puntos_hopf.append(punto)

        return puntos_hopf

    # ──────────────────────────────────────────────────
    #             Comparar puntos Hopf
    # ──────────────────────────────────────────────────
        """
        Calcula los posibles puntos de Hopf para varios sistemas FHN.
        """
    def comparar_hopf(
        self,
        sistemas,
        imprimir=True
    ):
        resultados = {}
        for sistema in sistemas:
            nombre = sistema["nombre"]
            a = sistema["a"]
            b = sistema["b"]
            epsilon = sistema["epsilon"]
            puntos_hopf = self.puntos_hopf_sistema(
                a=a,
                b=b,
                epsilon=epsilon
            )
            resultados[nombre] = puntos_hopf

            if imprimir:
                print("Sistema:", nombre)
                print("a =", a)
                print("b =", b)
                print("epsilon =", epsilon)
                if len(puntos_hopf) == 0:
                    print("  No hay candidatos de Hopf porque 1 - epsilon*b < 0.")
                else:
                    for j, punto in enumerate(puntos_hopf):
                        print(f"  Candidato Hopf {j+1}")
                        print(f"    u_H = {punto['u_H']:.8f}")
                        print(f"    w_H = {punto['w_H']:.8f}")
                        print(f"    I_H = {punto['I_H']:.8f}")
                        print(f"    traza = {punto['traza']:.8e}")
                        print(f"    determinante = {punto['determinante']:.8f}")
                        print(f"    d(traza)/dI = {punto['dtraza_dI']:.8f}")
                        print(f"    transversalidad = {punto['transversalidad']}")
                        print(f"    es Hopf = {punto['es_hopf']}")

                print()
        return resultados

    # ──────────────────────────────────────────────────
    #        Calcular umbrales Hopf para varios GC
    # ──────────────────────────────────────────────────
        """
        Barre valores de C y calcula los umbrales de Hopf del modelo
        FHN-Barrett usando las expresiones explícitas:

            a(C), b(C), epsilon(C).

        Para cada valor de C, calcula los dos candidatos:

            u_H = ± sqrt(1 - epsilon(C)*b(C))

        y sus respectivas corrientes críticas:

            I_H = (u_H - a(C))/b(C) - u_H + u_H^3/3.

        Devuelve un diccionario con los resultados y genera una gráfica.
        """
    def barrer_hopf_Barrett_GC(
        self,
        GC_min=0.0,
        GC_max=10.0,
        num_GC=300,
        incluir_clasico=True,
        a_clasico=0.7,
        b_clasico=0.8,
        epsilon_clasico=0.08,
        titulo=r"Desplazamiento de los umbrales de Hopf",
        figsize=(8, 5),
        guardar=False,
        ruta=CARPETA_IMAGENES,
        nombre_archivo="FHNcomp_umbrales_Hopf.pdf"
    ):
        GC_vals = np.linspace(GC_min, GC_max, num_GC)

        IH_menos = []
        IH_mas = []
        uH_menos = []
        uH_mas = []
        det_menos = []
        det_mas = []
        transversalidad_menos = []
        transversalidad_mas = []

        for GC in GC_vals:

            a, b, epsilon = self.parametros_FHN_Barrett(GC)

            puntos_hopf = self.puntos_hopf_sistema(
                a=a,
                b=b,
                epsilon=epsilon
            )

            if len(puntos_hopf) == 2:

                # Ordenamos por u_H: primero rama negativa, luego positiva.
                puntos_hopf = sorted(
                    puntos_hopf,
                    key=lambda p: p["u_H"]
                )

                p_menos = puntos_hopf[0]
                p_mas = puntos_hopf[1]

                uH_menos.append(p_menos["u_H"])
                IH_menos.append(p_menos["I_H"])
                det_menos.append(p_menos["determinante"])
                transversalidad_menos.append(p_menos["transversalidad"])

                uH_mas.append(p_mas["u_H"])
                IH_mas.append(p_mas["I_H"])
                det_mas.append(p_mas["determinante"])
                transversalidad_mas.append(p_mas["transversalidad"])

            else:

                uH_menos.append(np.nan)
                IH_menos.append(np.nan)
                det_menos.append(np.nan)
                transversalidad_menos.append(False)

                uH_mas.append(np.nan)
                IH_mas.append(np.nan)
                det_mas.append(np.nan)
                transversalidad_mas.append(False)

        IH_menos = np.array(IH_menos)
        IH_mas = np.array(IH_mas)
        uH_menos = np.array(uH_menos)
        uH_mas = np.array(uH_mas)
        det_menos = np.array(det_menos)
        det_mas = np.array(det_mas)

        transversalidad_menos = np.array(transversalidad_menos)
        transversalidad_mas = np.array(transversalidad_mas)

        resultado = {
            "GC_vals": GC_vals,
            "I_H_menos": IH_menos,
            "I_H_mas": IH_mas,
            "u_H_menos": uH_menos,
            "u_H_mas": uH_mas,
            "det_menos": det_menos,
            "det_mas": det_mas,
            "transversalidad_menos": transversalidad_menos,
            "transversalidad_mas": transversalidad_mas
        }

        # Hopf del modelo FHN clásico, si se desea incluir como referencia
        if incluir_clasico:
            puntos_hopf_clasico = self.puntos_hopf_sistema(
                a=a_clasico,
                b=b_clasico,
                epsilon=epsilon_clasico
            )
            puntos_hopf_clasico = sorted(
                puntos_hopf_clasico,
                key=lambda p: p["u_H"]
            )
            if len(puntos_hopf_clasico) == 2:
                IH_clasico_menos = puntos_hopf_clasico[0]["I_H"]
                IH_clasico_mas = puntos_hopf_clasico[1]["I_H"]
            else:
                IH_clasico_menos = np.nan
                IH_clasico_mas = np.nan

            resultado["I_H_clasico_menos"] = IH_clasico_menos
            resultado["I_H_clasico_mas"] = IH_clasico_mas

        # Gráfica
        plt.figure(figsize=figsize)
        plt.plot(
            GC_vals,
            IH_menos,
            linewidth=2,
            label=r"$I_H^-(\mathrm{C})$"
        )
        plt.plot(
            GC_vals,
            IH_mas,
            linewidth=2,
            label=r"$I_H^+(\mathrm{C})$"
        )

        if incluir_clasico:
            plt.axhline(
                IH_clasico_menos,
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                label=r"$I_{H,\mathrm{Basal}}^-$"
            )
            plt.axhline(
                IH_clasico_mas,
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                label=r"$I_{H,\mathrm{Basal}}^+$"
            )
        plt.xlabel(r"$\mathrm{C}\ [\mu\mathrm{M}]$")
        plt.ylabel(r"$I_H$")
        plt.title(titulo, fontsize=20)
        plt.grid(alpha=0.3)
        plt.legend(
            frameon=True,
            fancybox=False,
            edgecolor="black"
        )
        plt.tight_layout()

        if guardar:
            if ruta is None:
                ruta_completa = Path(nombre_archivo)
            else:
                ruta_completa = Path(ruta) / nombre_archivo
            ruta_completa.parent.mkdir(parents=True,exist_ok=True)
            plt.savefig(ruta_completa, bbox_inches="tight", format="pdf")
        plt.show()

        self.resultado_barrido_Hopf_Barrett = resultado

        return resultado

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #           Clasificar equilibrio
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Clasifica el régimen local de un sistema FHN para parámetros fijos.

        El sistema es:

            du/dt = u - u^3/3 - w + I
            dw/dt = epsilon(u - a - b w)

        La clasificación se basa en los equilibrios y su estabilidad local.

        Códigos:
            0 : equilibrio estable
            1 : equilibrio inestable
            2 : silla
            3 : no hiperbólico / candidato a bifurcación
            4 : múltiples equilibrios
        """
    def clasificar_regimen_equilibrio(
        self,
        a, b, epsilon, I,
        tol=1e-8
    ):
        equilibrios = self.equilibrios_sistema(
            a=a,
            b=b,
            epsilon=epsilon,
            I=I
        )

        if len(equilibrios) == 0:

            return {
                "codigo": np.nan,
                "regimen": "sin equilibrio real",
                "equilibrios": equilibrios
            }

        if len(equilibrios) > 1:

            return {
                "codigo": 4,
                "regimen": "múltiples equilibrios",
                "equilibrios": equilibrios
            }

        eq = equilibrios[0]

        traza = eq["traza"]
        determinante = eq["determinante"]

        if abs(traza) <= tol and determinante > tol:

            codigo = 3
            regimen = "no hiperbólico / candidato a Hopf"

        elif determinante < -tol:

            codigo = 2
            regimen = "silla"

        elif determinante > tol and traza < -tol:

            codigo = 0
            regimen = "equilibrio estable"

        elif determinante > tol and traza > tol:

            codigo = 1
            regimen = "equilibrio inestable"

        else:

            codigo = 3
            regimen = "no hiperbólico"

        return {
            "codigo": codigo,
            "regimen": regimen,
            "equilibrios": equilibrios
        }
    
    # ──────────────────────────────────────────────────
    #          Diagrama de regímenes en (C,I)
    # ──────────────────────────────────────────────────
        """
        Construye un diagrama de regímenes en el plano ([GC], I).

        Para cada par ([GC], I), evalúa:

            a([GC]), b([GC]), epsilon([GC])

        y clasifica el equilibrio del sistema FHN-Barrett.

        Códigos:
            0 : equilibrio estable
            1 : equilibrio inestable
            2 : silla
            3 : no hiperbólico / candidato a bifurcación
            4 : múltiples equilibrios

        Si incluir_hopf=True, superpone las curvas de Hopf:

            I_H^-([GC]), I_H^+([GC]).
        """
    def barrer_regimenes_GC_I(
        self,
        GC_min=0.0,
        GC_max=10.0,
        num_GC=300,
        I_min=-4.0,
        I_max=4.0,
        num_I=300,
        incluir_hopf=True,
        incluir_clasico=True,
        a_clasico=0.7,
        b_clasico=0.8,
        epsilon_clasico=0.08,
        titulo=r"Regímenes dinámicos en el plano $(\mathrm{C},I)$",
        figsize=(8, 6),
        guardar=False,
        ruta=CARPETA_IMAGENES,
        nombre_archivo="FHNcomp_regimenes_dinamicos_Barrett.pdf"
    ):
        GC_vals = np.linspace(GC_min, GC_max, num_GC)
        I_vals = np.linspace(I_min, I_max, num_I)

        codigos = np.zeros((num_I, num_GC))
        u_star_mat = np.full((num_I, num_GC), np.nan)
        w_star_mat = np.full((num_I, num_GC), np.nan)
        traza_mat = np.full((num_I, num_GC), np.nan)
        det_mat = np.full((num_I, num_GC), np.nan)
        num_equilibrios_mat = np.zeros((num_I, num_GC), dtype=int)

        for j, GC in enumerate(GC_vals):
            a, b, epsilon = self.parametros_FHN_Barrett(GC)
            for i, I in enumerate(I_vals):
                resultado = self.clasificar_regimen_equilibrio(
                    a=a, b=b, epsilon=epsilon, I=I
                )
                codigos[i, j] = resultado["codigo"]
                equilibrios = resultado["equilibrios"]
                num_equilibrios_mat[i, j] = len(equilibrios)
                if len(equilibrios) == 1:
                    eq = equilibrios[0]
                    u_star_mat[i, j] = eq["u"]
                    w_star_mat[i, j] = eq["w"]
                    traza_mat[i, j] = eq["traza"]
                    det_mat[i, j] = eq["determinante"]

        resultado_global = {
            "GC_vals": GC_vals,
            "I_vals": I_vals,
            "codigos": codigos,
            "u_star": u_star_mat,
            "w_star": w_star_mat,
            "traza": traza_mat,
            "determinante": det_mat,
            "num_equilibrios": num_equilibrios_mat
        }

        # Curvas de Hopf para FHN-Barrett
        if incluir_hopf:
            IH_menos = []
            IH_mas = []
            for GC in GC_vals:
                a, b, epsilon = self.parametros_FHN_Barrett(GC)
                puntos_hopf = self.puntos_hopf_sistema(
                    a=a,
                    b=b,
                    epsilon=epsilon
                )
                if len(puntos_hopf) == 2:
                    puntos_hopf = sorted(
                        puntos_hopf,
                        key=lambda p: p["u_H"]
                    )
                    IH_menos.append(puntos_hopf[0]["I_H"])
                    IH_mas.append(puntos_hopf[1]["I_H"])
                else:
                    IH_menos.append(np.nan)
                    IH_mas.append(np.nan)
            IH_menos = np.array(IH_menos)
            IH_mas = np.array(IH_mas)

            resultado_global["I_H_menos"] = IH_menos
            resultado_global["I_H_mas"] = IH_mas

        # Hopf del FHN clásico como referencia
        if incluir_clasico:
            puntos_hopf_clasico = self.puntos_hopf_sistema(
                a=a_clasico,
                b=b_clasico,
                epsilon=epsilon_clasico
            )
            puntos_hopf_clasico = sorted(
                puntos_hopf_clasico,
                key=lambda p: p["u_H"]
            )
            if len(puntos_hopf_clasico) == 2:
                IH_clasico_menos = puntos_hopf_clasico[0]["I_H"]
                IH_clasico_mas = puntos_hopf_clasico[1]["I_H"]
            else:
                IH_clasico_menos = np.nan
                IH_clasico_mas = np.nan

            resultado_global["I_H_clasico_menos"] = IH_clasico_menos
            resultado_global["I_H_clasico_mas"] = IH_clasico_mas

        # Gráfica del diagrama de regímenes (Simplificada a 2 colores)
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        plt.figure(figsize=figsize)
        # Mantiene el verde pastel para estable (0) y rosa pastel para inestable (1)
        cmap = ListedColormap([
            "#b7e4c7",  # 0 Estable
            "#ffccd5"   # 1 Inestable
        ])
        # Límites discretos ajustados para mapear solo el rango [0, 1]
        norm = BoundaryNorm(
            [-0.5, 0.5, 1.5],
            cmap.N
        )
        GC_grid, I_grid = np.meshgrid(
            GC_vals,
            I_vals
        )
        imagen = plt.pcolormesh(
            GC_grid,
            I_grid,
            codigos,
            cmap=cmap,
            norm=norm,
            shading="auto"
        )
        # Ajuste de marcas (ticks) exclusivamente para dos categorías
        cbar = plt.colorbar(
            imagen,
            ticks=[0, 1]
        )
        # Etiquetas reducidas a los dos regímenes presentes
        cbar.ax.set_yticklabels([
            r"Estable",
            r"Inestable"
        ])
        if incluir_hopf:
            plt.plot(
                GC_vals,
                IH_menos,
                color="black",
                linestyle="-",
                linewidth=2.0,
                label=r"$I_H^-(\mathrm{C})$"
            )
            plt.plot(
                GC_vals,
                IH_mas,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label=r"$I_H^+(\mathrm{C})$"
            )
        if incluir_clasico:
            plt.axhline(
                IH_clasico_menos,
                color="black",
                linestyle="-",
                linewidth=0.75,
                label=r"$I_{H,\mathrm{Basal}}^-$"
            )
            plt.axhline(
                IH_clasico_mas,
                color="black",
                linestyle="--",
                linewidth=0.75,
                label=r"$I_{H,\mathrm{Basal}}^+$"
            )

        plt.xlabel(r"$\mathrm{C}\ [\mu\mathrm{M}]$")
        plt.ylabel(r"$I$")
        plt.title(titulo)
        plt.grid(alpha=0.25)
        plt.legend(
            frameon=True,
            fancybox=False,
            edgecolor="black",
            fontsize=9
        )
        plt.tight_layout()

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        if guardar:
            if guardar:
                if ruta is None:
                    ruta_completa = Path(nombre_archivo)
                else:
                    ruta_completa = Path(ruta) / nombre_archivo
                ruta_completa.parent.mkdir(parents=True,exist_ok=True)
                plt.savefig(ruta_completa, bbox_inches="tight", format="pdf")
        plt.show()
        self.resultado_regimenes_GC_I = resultado_global
        return resultado_global

    # ──────────────────────────────────────────────────
    #             Intervalos I por régimen
    # ──────────────────────────────────────────────────
        """
        Extrae intervalos aproximados de I para un valor de C dado
        y un régimen específico.

        Códigos:
            0 : equilibrio estable
            1 : equilibrio inestable
            2 : silla
            3 : no hiperbólico
            4 : múltiples equilibrios
        """
    def intervalos_I_por_regimen(
        self,
        resultado_regimenes,
        GC_objetivo,
        codigo_regimen=0
    ):
        GC_vals = resultado_regimenes["GC_vals"]
        I_vals = resultado_regimenes["I_vals"]
        codigos = resultado_regimenes["codigos"]

        j = np.argmin(np.abs(GC_vals - GC_objetivo))
        GC_usado = GC_vals[j]
        codigos_columna = codigos[:, j]

        mascara = codigos_columna == codigo_regimen

        intervalos = []

        dentro = False
        inicio = None

        for i in range(len(I_vals)):

            if mascara[i] and not dentro:

                dentro = True
                inicio = I_vals[i]

            if dentro and (not mascara[i] or i == len(I_vals)-1):

                dentro = False

                if mascara[i] and i == len(I_vals)-1:
                    fin = I_vals[i]
                else:
                    fin = I_vals[i-1]

                intervalos.append(
                    (inicio, fin)
                )

        resultado = {
            "GC_objetivo": GC_objetivo,
            "GC_usado": GC_usado,
            "codigo_regimen": codigo_regimen,
            "intervalos_I": intervalos
        }

        return resultado

    # ──────────────────────────────────────────────────
    #              Poincaré-Bendixson
    # ──────────────────────────────────────────────────

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #           Campo vectorial del FHN
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    def campo_FHN(self, u, w, a, b, epsilon, I):
        du = u - (u**3)/3.0 - w + I
        dw = epsilon*(u - a - b*w)
        return du, dw
    
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #              Rectángulo atrapante
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Verifica numéricamente si el rectángulo

            [-R,R] x [-W,W]

        es positivamente invariante.

        Condiciones:
            en u = R      se pide du/dt <= 0,
            en u = -R     se pide du/dt >= 0,
            en w = W      se pide dw/dt <= 0,
            en w = -W     se pide dw/dt >= 0.
        """
    def verificar_rectangulo_atrapante(
        self,
        a,
        b,
        epsilon,
        I,
        R=4.0,
        W=6.0,
        num=400,
        tol=1e-10
    ):
        w_vals = np.linspace(-W, W, num)
        u_vals = np.linspace(-R, R, num)

        # Lado derecho: u = R.
        du_der, _ = self.campo_FHN(
            R,
            w_vals,
            a,
            b,
            epsilon,
            I
        )

        cond_der = np.all(du_der <= tol)

        # Lado izquierdo: u = -R.
        du_izq, _ = self.campo_FHN(
            -R,
            w_vals,
            a,
            b,
            epsilon,
            I
        )

        cond_izq = np.all(du_izq >= -tol)

        # Lado superior: w = W.
        _, dw_sup = self.campo_FHN(
            u_vals,
            W,
            a,
            b,
            epsilon,
            I
        )

        cond_sup = np.all(dw_sup <= tol)

        # Lado inferior: w = -W.
        _, dw_inf = self.campo_FHN(
            u_vals,
            -W,
            a,
            b,
            epsilon,
            I
        )

        cond_inf = np.all(dw_inf >= -tol)

        es_atrapante = cond_der and cond_izq and cond_sup and cond_inf

        return {
            "es_atrapante": es_atrapante,
            "R": R,
            "W": W,
            "condicion_derecha": cond_der,
            "condicion_izquierda": cond_izq,
            "condicion_superior": cond_sup,
            "condicion_inferior": cond_inf
        }
    
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #           Buscar recángulo atrapante
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Busca un rectángulo atrapante entre listas de valores posibles
        de R y W.
        """
    def buscar_rectangulo_atrapante(
        self,
        a,
        b,
        epsilon,
        I,
        R_vals=None,
        W_vals=None,
        num=400
    ):
        if R_vals is None:
            R_vals = [3, 4, 5, 6, 8, 10]
        if W_vals is None:
            W_vals = [3, 4, 6, 8, 10, 15, 20]

        for R in R_vals:
            for W in W_vals:
                resultado = self.verificar_rectangulo_atrapante(
                    a=a,
                    b=b,
                    epsilon=epsilon,
                    I=I,
                    R=R,
                    W=W,
                    num=num
                )

                if resultado["es_atrapante"]:
                    return resultado

        return {
            "es_atrapante": False,
            "R": None,
            "W": None
        }


    # ───────────────────────────────────────────────────────────────
    #                Primera etapa: candidatos
    # ───────────────────────────────────────────────────────────────
        """
        Barre el plano ([GC], I) y almacena los puntos que satisfacen
        las condiciones preliminares para una posible aplicación del
        teorema de Poincaré--Bendixson.
        Condiciones verificadas:
            1. Existe un único equilibrio.
            2. El equilibrio es inestable:
                traza > 0,
                determinante > 0.
            3. Existe un rectángulo positivamente invariante.

        Los puntos que satisfacen las tres condiciones se almacenan
        en la lista 'candidatos' para una segunda etapa de análisis.
        """
    def barrer_poincare_bendixson_GC_I(
        self,
        GC_min=0.0,
        GC_max=10.0,
        num_GC=150,
        I_min=-4.0,
        I_max=4.0,
        num_I=150,
        R=5.0,
        W=20.0,
        verificar_atrapante=True,
        titulo=r"Región candidata para la aplicación de Poincaré--Bendixson",
        figsize=(8, 6),
        guardar=False,
        ruta=CARPETA_IMAGENES,
        nombre_archivo="FHNcomp_region_aplicacion_PoincareBendixson.pdf"
    ):
        # Valores del barrido
        GC_vals = np.linspace(GC_min, GC_max, num_GC)
        I_vals = np.linspace(I_min, I_max, num_I)

        # Matrices para la representación gráfica
        matriz_PB = np.zeros((num_I, num_GC), dtype=int)
        matriz_traza = np.full(
            (num_I, num_GC),
            np.nan
        )
        matriz_det = np.full(
            (num_I, num_GC),
            np.nan
        )
        matriz_num_eq = np.zeros(
            (num_I, num_GC),
            dtype=int
        )
        # Lista donde se almacenarán los candidatos
        candidatos = []

        # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        # BARRIDO DEL PLANO (GC, I)
        # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        for j, GC in enumerate(GC_vals):
            # Parámetros reducidos
            a, b, epsilon = self.parametros_FHN_Barrett(GC)
            for i, I in enumerate(I_vals):
                # Calcular equilibrios
                # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                equilibrios = self.equilibrios_sistema(
                    a=a,
                    b=b,
                    epsilon=epsilon,
                    I=I
                )
                matriz_num_eq[i, j] = len(equilibrios)
                # Debe existir exactamente un equilibrio
                if len(equilibrios) != 1:
                    matriz_PB[i, j] = 0
                    continue

                # Obtener equilibrio
                # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                eq = equilibrios[0]
                u_star = eq["u"]
                w_star = eq["w"]
                traza = eq["traza"]
                determinante = eq["determinante"]
                matriz_traza[i, j] = traza
                matriz_det[i, j] = determinante

                # Equilibrio inestable
                # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                equilibrio_inestable = (
                    traza > 0
                    and determinante > 0
                )
                if not equilibrio_inestable:
                    matriz_PB[i, j] = 0
                    continue

                # Rectángulo atrapante
                # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                if verificar_atrapante:
                    rectangulo = self.verificar_rectangulo_atrapante(
                        a=a,
                        b=b,
                        epsilon=epsilon,
                        I=I,
                        R=R,
                        W=W,
                        num=100
                    )
                    if not rectangulo["es_atrapante"]:
                        matriz_PB[i, j] = 0
                        continue
                else:
                    rectangulo = {
                        "es_atrapante": True,
                        "R": R,
                        "W": W
                    }

                # El punto pasa todas las condiciones preliminares
                # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                matriz_PB[i, j] = 1
                # Guardar toda la información necesaria
                candidato = {
                    "GC": GC,
                    "I": I,
                    "a": a,
                    "b": b,
                    "epsilon": epsilon,
                    "u_star": u_star,
                    "w_star": w_star,
                    "traza": traza,
                    "determinante": determinante,
                    "R": rectangulo["R"],
                    "W": rectangulo["W"]
                }
                candidatos.append(candidato)

        # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        # CURVAS DE HOPF
        # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        IH_menos = []
        IH_mas = []
        for GC in GC_vals:
            a, b, epsilon = self.parametros_FHN_Barrett(GC)
            puntos_hopf = self.puntos_hopf_sistema(
                a=a,
                b=b,
                epsilon=epsilon
            )
            if len(puntos_hopf) == 2:

                puntos_hopf = sorted(
                    puntos_hopf,
                    key=lambda p: p["u_H"]
                )

                IH_menos.append(
                    puntos_hopf[0]["I_H"]
                )

                IH_mas.append(
                    puntos_hopf[1]["I_H"]
                )

            else:

                IH_menos.append(np.nan)
                IH_mas.append(np.nan)

        IH_menos = np.array(IH_menos)
        IH_mas = np.array(IH_mas)

        # RESULTADO GLOBAL
        resultado_global = {
            "GC_vals": GC_vals,
            "I_vals": I_vals,
            "matriz_PB": matriz_PB,
            "traza": matriz_traza,
            "determinante": matriz_det,
            "num_equilibrios": matriz_num_eq,
            # NUEVO
            "candidatos": candidatos,
            "R": R,
            "W": W,
            "verificar_atrapante": verificar_atrapante,
            "I_H_menos": IH_menos,
            "I_H_mas": IH_mas
        }

        # GRÁFICA
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        plt.figure(figsize=figsize)
        GC_grid, I_grid = np.meshgrid(GC_vals, I_vals)
        cmap = ListedColormap([
            "white",
            "#90be6d"
        ])
        norm = BoundaryNorm(
            [-0.5, 0.5, 1.5],
            cmap.N
        )
        imagen = plt.pcolormesh(
            GC_grid,
            I_grid,
            matriz_PB,
            cmap=cmap,
            norm=norm,
            shading="auto"
        )
        cbar = plt.colorbar(
            imagen,
            ticks=[0, 1]
        )
        cbar.ax.set_yticklabels([
            r"No candidato",
            r"Candidato"
        ])
        # Curva de Hopf inferior
        plt.plot(
            GC_vals,
            IH_menos,
            color="black",
            linewidth=2,
            label=r"$I_H^-(C)$"
        )
        # Curva de Hopf superior
        plt.plot(
            GC_vals,
            IH_mas,
            color="black",
            linestyle="--",
            linewidth=2,
            label=r"$I_H^+(C)$"
        )
        plt.xlabel(r"$C\,[\mu\mathrm{M}]$")
        plt.ylabel(r"$I$")
        plt.title(titulo)
        plt.grid(alpha=0.25)
        plt.legend(
            frameon=True,
            fancybox=False,
            edgecolor="black"
        )
        plt.tight_layout()
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        if guardar:
            if ruta is None:
                ruta_completa = Path(nombre_archivo)
            else:
                ruta_completa = Path(ruta) / nombre_archivo
            ruta_completa.parent.mkdir(parents=True,exist_ok=True)
            plt.savefig(ruta_completa, bbox_inches="tight", format="pdf")
        plt.show()
        # Guardar internamente
        self.resultado_PB_GC_I = resultado_global

        return resultado_global

    

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #       Construir elipse Lyapunov
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Construye una matriz simétrica definida positiva P para
        la función cuadrática

            V(x) = x^T P x,

        asociada al equilibrio inestable.

        Se utiliza la ecuación de Lyapunov

            J^T P + P J = Q,

        con

            Q = I.

        Como el equilibrio es inestable, -J es una matriz estable.
        Por ello se resuelve

            (-J)^T P + P(-J) = -I,

        que equivale a

            J^T P + PJ = I.

        Retorna la matriz P y los datos de verificación.
        """
    def construir_matriz_lyapunov_repulsora(
        self,
        u_star,
        b,
        epsilon,
        tolerancia=1e-12
    ):
        # Jacobiana
        J = self.jacobiano(
            u=u_star, b=b, epsilon=epsilon
        )

        # Verificar autovalores
        eigenvalues_J = np.linalg.eigvals(J)
        es_inestable = np.all(
            np.real(eigenvalues_J) > tolerancia
        )
        if not es_inestable:
            return {
                "exito": False,
                "motivo": (
                    "La Jacobiana no tiene todos sus "
                    "autovalores con parte real positiva."
                ),
                "J": J,
                "autovalores": eigenvalues_J,
                "P": None
            }

        # Resolver ecuación de Lyapunov
        # (-J)^T P + P(-J) = -I
        # scipy resuelve:
        # A X + X A^T = Q
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        A = -J
        Q = -np.eye(2)
        P = solve_continuous_lyapunov(A.T, Q)

        # Simetrizar numéricamente
        P = 0.5 * (P + P.T)

        # Comprobar que P sea definida positiva
        eigenvalues_P = np.linalg.eigvalsh(P)
        P_definida_positiva = np.all(
            eigenvalues_P > tolerancia
        )

        # Comprobar ecuación de Lyapunov
        # J^T P + P J = I
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        Q_obtenida = ( J.T @ P + P @ J )
        error_lyapunov = np.linalg.norm( Q_obtenida - np.eye(2) )
        exito = (
            P_definida_positiva
            and error_lyapunov < 1e-8
        )
        return {
            "exito": exito,
            "J": J,
            "autovalores": eigenvalues_J,
            "P": P,
            "autovalores_P": eigenvalues_P,
            "P_definida_positiva": P_definida_positiva,
            "Q_obtenida": Q_obtenida,
            "error_lyapunov": error_lyapunov
        }


    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #           Verirficar elipse repulsora
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Verifica numéricamente si una elipse centrada en el equilibrio
        es repulsora. La elipse se define mediante
            V(x) = x^T P x = rho^2,
        donde
            x = [u-u_star, w-w_star]^T
        y P se obtiene mediante una ecuación de Lyapunov.
        La condición buscada es
            dV/dt > 0
        en toda la frontera de la elipse.
        La comprobación se realiza numéricamente sobre 'num' puntos.
        """
    def verificar_elipse_repulsora(self,
        u_star, w_star,
        a, b, epsilon, I,
        rho=0.05, num=400, tol=1e-10,
        verificar_rectangulo=True,
        R=5.0, W=20.0
    ):
        # Construir matriz P
        resultado_P = (
            self.construir_matriz_lyapunov_repulsora(
                u_star=u_star,
                b=b,
                epsilon=epsilon
            )
        )
        if not resultado_P["exito"]:
            return {
                "es_repulsora": False,
                "motivo": (
                    "No fue posible construir una matriz "
                    "de Lyapunov definida positiva."
                ),
                "P": None,
                "rho": rho
            }
        P = resultado_P["P"]
        # Descomposición espectral de P
        eigenvalues_P, eigenvectors_P = ( np.linalg.eigh(P) )
        # P debe ser definida positiva
        if np.any(eigenvalues_P <= 0):
            return {
                "es_repulsora": False,
                "motivo": "La matriz P no es definida positiva.",
                "P": P,
                "rho": rho
            }

        # Construcción de la frontera de la elipse
        theta_vals = np.linspace(0.0, 2.0 * np.pi, num, endpoint=False)

        # Coordenadas en el sistema diagonalizado
        z = np.vstack([
            rho * np.cos(theta_vals),
            rho * np.sin(theta_vals)
        ])

        # P^{-1/2}
        P_inv_sqrt = (
            eigenvectors_P
            @ np.diag(1.0 / np.sqrt(eigenvalues_P))
            @ eigenvectors_P.T
        )

        # x = P^{-1/2} z
        desplazamientos = (
            P_inv_sqrt @ z
        )

        # Coordenadas físicas
        u_vals = (
            u_star
            + desplazamientos[0, :]
        )

        w_vals = (
            w_star
            + desplazamientos[1, :]
        )

        # Verificar que la elipse esté dentro del rectángulo
        if verificar_rectangulo:
            dentro_rectangulo = (
                np.all(u_vals >= -R)
                and np.all(u_vals <= R)
                and np.all(w_vals >= -W)
                and np.all(w_vals <= W)
            )

            if not dentro_rectangulo:
                return {
                    "es_repulsora": False,
                    "motivo": (
                        "La elipse no está contenida "
                        "en el rectángulo atrapante."
                    ),
                    "P": P,
                    "rho": rho,
                    "u_vals": u_vals,
                    "w_vals": w_vals
                }

        # Evaluar campo vectorial
        du_vals, dw_vals = self.campo_FHN(
            u_vals, w_vals,
            a, b, epsilon, I
        )

        # Gradiente de V
        # V(x) = x^T P x
        # grad V = 2 P x
        x = np.vstack([
            u_vals - u_star,
            w_vals - w_star
        ])
        grad_V = (2.0 * P @ x )

        # Derivada temporal de V
        # dV/dt = grad V · X
        dV_dt = ( grad_V[0, :] * du_vals + grad_V[1, :] * dw_vals)

        # Verificación de repulsión
        min_dV_dt = np.min(dV_dt)
        max_dV_dt = np.max(dV_dt)
        es_repulsora = (
            min_dV_dt > tol
        )

        # Resultado
        return {
            "es_repulsora": es_repulsora,
            "rho": rho,
            "P": P,
            "autovalores_P": eigenvalues_P,
            "u_vals": u_vals,
            "w_vals": w_vals,
            "dV_dt": dV_dt,
            "min_dV_dt": min_dV_dt,
            "max_dV_dt": max_dV_dt,
            "motivo": (
                "La derivada de V es positiva en "
                "todos los puntos muestreados."
                if es_repulsora
                else
                "La derivada de V no es positiva "
                "en todos los puntos muestreados."
            )
        }

    
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #       Verificar elipses de candidatos
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    def verificar_elipses_candidatos(
        self,
        candidatos,
        rho_vals=None,
        num_elipse=400,
        tol=1e-10
    ):
        if rho_vals is None:
            rho_vals = [
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1
            ]

        resultados = []
        for c in candidatos:
            elipse_encontrada = None
            for rho in rho_vals:
                resultado = self.verificar_elipse_repulsora(
                    u_star=c["u_star"],
                    w_star=c["w_star"],
                    a=c["a"],
                    b=c["b"],
                    epsilon=c["epsilon"],
                    I=c["I"],
                    rho=rho,
                    num=num_elipse,
                    tol=tol,
                    verificar_rectangulo=True,
                    R=c["R"],
                    W=c["W"]
                )
                if resultado["es_repulsora"]:
                    elipse_encontrada = resultado
                    break

            resultados.append({
                "GC": c["GC"],
                "I": c["I"],
                "elipse_repulsora": elipse_encontrada is not None,
                "rho": (
                    elipse_encontrada["rho"]
                    if elipse_encontrada is not None
                    else None
                ),
                "P": (
                    elipse_encontrada["P"]
                    if elipse_encontrada is not None
                    else None
                ),
                "min_dV_dt": (
                    elipse_encontrada["min_dV_dt"]
                    if elipse_encontrada is not None
                    else None
                ),
                "max_dV_dt": (
                    elipse_encontrada["max_dV_dt"]
                    if elipse_encontrada is not None
                    else None
                )
            })

        return resultados

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #          Verificar Poincaré-Bendixson
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Verifica numéricamente condiciones suficientes para la aplicación
        del teorema de Poincaré--Bendixson mediante una elipse repulsora.
        Criterio utilizado:
            1. Existe un único equilibrio.
            2. El equilibrio es inestable:
                traza > 0,
                determinante > 0.
            3. Existe un rectángulo positivamente invariante.
            4. Existe una elipse centrada en el equilibrio tal que
                dV/dt > 0
            en toda la frontera muestreada.

        Bajo estas condiciones se obtiene una región anular compacta
        positivamente invariante y libre de equilibrios. La verificación
        de la condición de la elipse es numérica.
        """
    def verificar_poincare_bendixson_sistema(
        self,
        a, b, epsilon,
        I,
        R_vals=None,
        W_vals=None,
        rho_vals=None,
        num_rectangulo=400,
        num_elipse=400,
        imprimir=True
    ):
        # Valores de rho
        if rho_vals is None:
            rho_vals = [
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1
            ]

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # 1. Equilibrios
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        equilibrios = self.equilibrios_sistema(
            a=a,
            b=b,
            epsilon=epsilon,
            I=I
        )

        # Unicidad del equilibrio
        if len(equilibrios) != 1:
            resultado = {
                "garantiza_ciclo": False,
                "motivo": "No hay un único equilibrio.",
                "num_equilibrios": len(equilibrios)
            }
            if imprimir:
                print(resultado["motivo"])
            return resultado

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # 2. Equilibrio inestable
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        eq = equilibrios[0]
        u_star = eq["u"]
        w_star = eq["w"]
        traza = eq["traza"]
        determinante = eq["determinante"]
        equilibrio_fuente = (
            traza > 0
            and determinante > 0
        )
        if not equilibrio_fuente:
            resultado = {
                "garantiza_ciclo": False,
                "motivo": "El equilibrio no es una fuente inestable.",
                "equilibrio": eq
            }
            if imprimir:
                print(resultado["motivo"])
            return resultado

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # 3. Buscar rectángulo atrapante
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        rectangulo = self.buscar_rectangulo_atrapante(
            a=a,
            b=b,
            epsilon=epsilon,
            I=I,
            R_vals=R_vals,
            W_vals=W_vals,
            num=num_rectangulo
        )
        if not rectangulo["es_atrapante"]:
            resultado = {
                "garantiza_ciclo": False,
                "motivo": "No se encontró rectángulo atrapante.",
                "equilibrio": eq,
                "rectangulo": rectangulo
            }
            if imprimir:
                print(resultado["motivo"])

            return resultado

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # 4. Buscar elipse repulsora
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        elipse_encontrada = None

        for rho in rho_vals:
            elipse = self.verificar_elipse_repulsora(
                u_star=u_star,
                w_star=w_star,
                a=a,
                b=b,
                epsilon=epsilon,
                I=I,
                rho=rho,
                num=num_elipse,
                verificar_rectangulo=True,
                R=rectangulo["R"],
                W=rectangulo["W"]
            )
            if elipse["es_repulsora"]:
                elipse_encontrada = elipse
                break

        if elipse_encontrada is None:

            resultado = {
                "garantiza_ciclo": False,
                "motivo": (
                    "No se encontró una elipse repulsora "
                    "alrededor del equilibrio."
                ),
                "equilibrio": eq,
                "rectangulo": rectangulo
            }

            if imprimir:
                print(resultado["motivo"])

            return resultado

        # 5. Resultado final
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

        resultado = {
            "garantiza_ciclo": True,
            "motivo": (
                "Se verificaron numéricamente las condiciones "
                "para la aplicación de Poincaré--Bendixson."
            ),
            "equilibrio": eq,
            "rectangulo": rectangulo,
            "elipse": elipse_encontrada,
            "R": rectangulo["R"],
            "W": rectangulo["W"],
            "rho": elipse_encontrada["rho"]
        }
        if imprimir:
            print(
                "Se verificaron las condiciones numéricas "
                "para Poincaré--Bendixson."
            )
            print(
                f"Equilibrio: "
                f"u*={u_star:.6f}, "
                f"w*={w_star:.6f}"
            )
            print(
                f"traza={traza:.6f}, "
                f"determinante={determinante:.6f}"
            )
            print(
                f"Rectángulo atrapante: "
                f"R={rectangulo['R']}, "
                f"W={rectangulo['W']}"
            )
            print(
                f"Parámetro de la elipse: "
                f"rho={elipse_encontrada['rho']}"
            )
            print(
                f"min(dV/dt)="
                f"{elipse_encontrada['min_dV_dt']:.6e}"
            )

        return resultado


    # ──────────────────────────────────────────────────
    #          Comparar Poincaré-Bendixson
    # ──────────────────────────────────────────────────
        """
        Aplica el criterio de Poincaré--Bendixson a una lista de sistemas.
        """
    def comparar_poincare_bendixson(
        self, sistemas, imprimir=True
    ):
        resultados = {}
        for sistema in sistemas:
            nombre = sistema["nombre"]
            a = sistema["a"]
            b = sistema["b"]
            epsilon = sistema["epsilon"]
            I = sistema["I"]
            if imprimir:
                print("Sistema:", nombre)

            resultado = self.verificar_poincare_bendixson_sistema(
                a=a,
                b=b,
                epsilon=epsilon,
                I=I,
                imprimir=imprimir
            )
            resultados[nombre] = resultado

            if imprimir:
                print()

        return resultados

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #       Construcción de la región final
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    def construir_matriz_elipse_repulsora(
        self,
        resultados_elipse,
        GC_vals,
        I_vals
    ):

        """
        Construye una matriz binaria que identifica los puntos
        (GC, I) para los cuales se verificó numéricamente una
        elipse repulsora.
        """

        matriz_elipse = np.zeros(
            (len(I_vals), len(GC_vals)),
            dtype=int
        )

        for resultado in resultados_elipse:

            if not resultado["elipse_repulsora"]:
                continue

            GC = resultado["GC"]
            I = resultado["I"]

            j = np.argmin(
                np.abs(GC_vals - GC)
            )

            i = np.argmin(
                np.abs(I_vals - I)
            )

            matriz_elipse[i, j] = 1

        return matriz_elipse


    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #      Graficar min(dV/dt) sobre la región final
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        """
        Grafica el valor mínimo de dV/dt sobre la frontera de la
        elipse repulsora para los puntos del plano ([GC], I).
        Para cada candidato se representa:
            min_{E} dV/dt
        donde E es la elipse definida por
            V(x) = x^T P x = rho^2.
        Los puntos que no poseen una elipse repulsora se dejan
        sin valor (NaN).
        Parámetros
        ----------
        resultado_PB : dict
            Resultado obtenido mediante
            barrer_poincare_bendixson_GC_I().
        resultados_elipse : list
            Resultados obtenidos mediante
            verificar_elipses_candidatos().
        matriz_PB_elipse : ndarray or None
            Matriz binaria que identifica los puntos donde se
            verificó una elipse repulsora.
        titulo : str
            Título de la gráfica.
        figsize : tuple
            Tamaño de la figura.
        guardar : bool
            Si es True, guarda la figura.
        ruta : str or None
            Directorio donde se guardará la figura.
        nombre_archivo : str
            Nombre del archivo de salida.
        Retorna
        -------
        matriz_min_dVdt : ndarray
            Matriz con el valor de min(dV/dt) para cada punto
            del plano ([GC], I).
        """
    def graficar_min_dVdt_elipse(
        self,
        resultado_PB,
        resultados_elipse,
        matriz_PB_elipse=None,
        titulo=(
            r"$\min_{\mathcal{E}_\rho}"
            r"\,\frac{dV}{dt}$ en la región verificada"
        ),
        figsize=(8, 6),
        guardar=False,
        ruta=CARPETA_IMAGENES,
        nombre_archivo="FHNcomp_min_dVdt_elipse.pdf"
    ):
        # Recuperar valores del barrido
        GC_vals = resultado_PB["GC_vals"]
        I_vals = resultado_PB["I_vals"]

        # Crear matriz

        matriz_min_dVdt = np.full(
            (len(I_vals), len(GC_vals)),
            np.nan
        )

        # Almacenar min(dV/dt)

        for resultado in resultados_elipse:
            if not resultado["elipse_repulsora"]:
                continue
            GC = resultado["GC"]
            I = resultado["I"]
            min_dVdt = resultado["min_dV_dt"]
            # Índices correspondientes en la malla
            j = np.argmin(np.abs(GC_vals - GC))
            i = np.argmin(np.abs(I_vals - I))
            matriz_min_dVdt[i, j] = min_dVdt

        # Malla del plano
        GC_grid, I_grid = np.meshgrid(GC_vals, I_vals)

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Figura
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        plt.figure(figsize=figsize)
        imagen = plt.pcolormesh(
            GC_grid,
            I_grid,
            matriz_min_dVdt,
            shading="auto"
        )
        # Barra de color
        cbar = plt.colorbar(imagen)
        cbar.set_label(
            r"$\min_{\mathcal{E}_\rho}"
            r"\,\frac{dV}{dt}$"
        )

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Curvas de Hopf
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        plt.plot(
            GC_vals,
            resultado_PB["I_H_menos"],
            color="black",
            linewidth=2,
            label=r"$I_H^-(C)$"
        )
        plt.plot(
            GC_vals,
            resultado_PB["I_H_mas"],
            color="black",
            linestyle="--",
            linewidth=2,
            label=r"$I_H^+(C)$"
        )

        # Etiquetas
        plt.xlabel(r"$C\,[\mu\mathrm{M}]$")
        plt.ylabel(r"$I$")
        plt.title(titulo)
        plt.grid(alpha=0.25)
        plt.legend(
            frameon=True,
            fancybox=False,
            edgecolor="black"
        )
        plt.tight_layout()

        # Guardar
        if guardar:
            if ruta is None:
                ruta_completa = nombre_archivo
            else:
                ruta_completa = (
                    f"{ruta}/{nombre_archivo}"
                )
            plt.savefig(
                ruta_completa,
                bbox_inches="tight"
            )
        plt.show()
        # Guardar internamente
        self.matriz_min_dVdt_elipse = matriz_min_dVdt

        return matriz_min_dVdt

    # ────────────────────────────────────────────────────────────────
    #           Mapa de calor para FHN-Basal
    # ────────────────────────────────────────────────────────────────
        """
        Construye un mapa de calor 2D para el caso FHN-Basal (C=0),
        replicando horizontalmente los valores obtenidos para C=0.

        La región rectangular queda delimitada por los valores de
        I_H^- e I_H^+ correspondientes al modelo basal.
        """
    def graficar_min_dVdt_elipse_basal_2D(
        self,
        resultado_PB,
        resultados_elipse,
        figsize=(8, 6),
        titulo=None,
        guardar=False,
        ruta=CARPETA_IMAGENES,
        nombre_archivo="FHN_Basal_Poincare_Bendixson.pdf"
    ):
        # Valores de C e I del barrido original
        GC_vals = np.asarray(resultado_PB["GC_vals"])
        I_vals = np.asarray(resultado_PB["I_vals"])

        # Localizar C = 0
        indice_C0 = np.argmin(np.abs(GC_vals - 0.0))
        C0 = GC_vals[indice_C0]
        print(f"Valor utilizado para FHN-Basal: C = {C0:.6f} μM")

        # Vector de min(dV/dt) para C = 0
        min_dVdt_C0 = np.full(len(I_vals), np.nan)
        for resultado in resultados_elipse:
            C = resultado.get("GC", None)
            I = resultado.get("I", None)
            if C is None or I is None:
                continue

            # Solo resultados correspondientes a C = 0
            if not np.isclose(C, C0):
                continue
            if not resultado.get("elipse_repulsora", False):
                continue

            min_dVdt = resultado.get("min_dV_dt", np.nan)
            indice_I = np.argmin(np.abs(I_vals - I))
            min_dVdt_C0[indice_I] = min_dVdt

        # Convertir el vector en matriz 2D
        matriz_basal = np.tile(
            min_dVdt_C0[:, np.newaxis],
            (1, len(GC_vals))
        )

        # Crear figura
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        fig, ax = plt.subplots(figsize=figsize)
        pcm = ax.pcolormesh(
            GC_vals,
            I_vals,
            matriz_basal,
            shading="auto"
        )

        # Barra de color
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label(
            r"$\min_{\partial\mathcal{E}_{\rho}}\,dV/dt$"
        )

        # Valores de Hopf para C = 0
        I_H_menos = resultado_PB["I_H_menos"][indice_C0]
        I_H_mas = resultado_PB["I_H_mas"][indice_C0]

        # Límites horizontales
        ax.axhline(
            I_H_menos,
            linestyle="--",
            linewidth=1.5,
            label=r"$I_H^{-}$"
        )
        ax.axhline(
            I_H_mas,
            linestyle="--",
            linewidth=1.5,
            label=r"$I_H^{+}$"
        )

        # Etiquetas
        ax.set_xlabel(r"$C\,[\mu\mathrm{M}]$")
        ax.set_ylabel(r"$I$")
        if titulo is None:
            titulo = (
                r"FHN-Basal ($C=0$): "
                r"$\min_{\partial\mathcal{E}_{\rho}}\,dV/dt$"
            )
        ax.set_title(titulo)

        ax.legend()
        plt.tight_layout()

        # Guardar
        if guardar:
            if ruta is None:
                ruta_completa = Path(nombre_archivo)
            else:
                ruta_completa = Path(ruta) / nombre_archivo
            ruta_completa.parent.mkdir(parents=True,exist_ok=True)
            plt.savefig(ruta_completa, bbox_inches="tight", format="pdf")
        plt.show()

        # Guardar matriz
        self.matriz_min_dVdt_basal = matriz_basal

        return matriz_basal


    # ─────────────────────────────────────────────────────────────
    #         Analizar comportamiento alrededor de Hopf
    # ─────────────────────────────────────────────────────────────
        """
        Analiza la dinámica del sistema FHN-Barrett para una
        concentración C, considerando cuatro corrientes alrededor
        de los dos candidatos a bifurcación de Hopf.
        Corrientes estudiadas:
            I_H_menos - delta
            I_H_menos + delta
            I_H_mas   - delta
            I_H_mas   + delta
        mediante los métodos
            parametros_FHN_Barrett
            puntos_hopf_sistema
            equilibrios_sistema
            campo_FHN
        Devuelve un diccionario con parámetros, puntos de Hopf,
        corrientes, equilibrios, estabilidad y soluciones temporales.
        """
    def analizar_hopf_concentracion(
        self,
        C,
        delta=0.1,
        t_final=500.0,
        num_tiempos=10000,
        perturbacion_inicial=0.01,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.1,
        mostrar_graficas=True,
        guardar=False,
        ruta = CARPETA_IMAGENES,
        nombre_archivo_1= "FHNcomp_Hopf_concentracion_series_temporales_FHN_C_10.pdf",
        nombre_archivo_2= "FHNcomp_Hopf_concentracion_retratos_fase_FHN_C_10.pdf"
    ):
        # Validar entradas
        if not np.isscalar(C) or not np.isfinite(C) or C < 0:
            raise ValueError(
                "C debe ser un número finito no negativo."
            )
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta debe ser positivo y finito.")
        if not np.isfinite(t_final) or t_final <= 0:
            raise ValueError("t_final debe ser positivo y finito.")
        if num_tiempos < 2:
            raise ValueError("num_tiempos debe ser al menos 2.")
        if perturbacion_inicial <= 0:
            raise ValueError("La perturbación inicial debe ser positiva.")
        if max_step <= 0 or rtol <= 0 or atol <= 0:
            raise ValueError("max_step, rtol y atol deben ser positivos.")

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Obtener parámetros del modelo para C
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        a, b, epsilon = self.parametros_FHN_Barrett(C)
        a = float(a)
        b = float(b)
        epsilon = float(epsilon)
        if b == 0:
            raise ValueError(
                "El parámetro b es cero; no se puede aplicar "
                "la formulación actual de las nulclinas."
            )
        parametros = {
            "C": float(C),
            "a": a,
            "b": b,
            "epsilon": epsilon
        }

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Calcular los candidatos a Hopf
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        puntos_hopf = self.puntos_hopf_sistema(
            a=a, b=b, epsilon=epsilon
        )
        puntos_hopf = sorted(
            puntos_hopf, key=lambda punto: punto["u_H"]
        )
        if len(puntos_hopf) != 2:
            raise ValueError(
                "No se encontraron exactamente dos candidatos "
                "a Hopf. No se pueden construir las cuatro "
                "corrientes solicitadas."
            )

        hopf_menos, hopf_mas = puntos_hopf

        corrientes = [
            {
                "nombre": r"$I_\mathrm{H}^- - \delta$",
                "rama": "menos",
                "I": hopf_menos["I_H"] - delta
            },
            {
                "nombre": r"$I_\mathrm{H}^- + \delta$",
                "rama": "menos",
                "I": hopf_menos["I_H"] + delta
            },
            {
                "nombre": r"$I_\mathrm{H}^+ - \delta$",
                "rama": "mas",
                "I": hopf_mas["I_H"] - delta
            },
            {
                "nombre": r"$I_\mathrm{H}^+ + \delta$",
                "rama": "mas",
                "I": hopf_mas["I_H"] + delta
            }
        ]

        # Preparar integración temporal
        t_eval = np.linspace(0.0, t_final, num_tiempos )
        resultados = {}

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Analizar e integrar cada corriente
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        for caso in corrientes:
            nombre = caso["nombre"]
            I = caso["I"]
            # Elegir la rama de Hopf de referencia
            if caso["rama"] == "menos":
                hopf_referencia = hopf_menos
            else:
                hopf_referencia = hopf_mas
            u_H = hopf_referencia["u_H"]
            w_H = hopf_referencia["w_H"]

            # Calcular todos los equilibrios reales
            equilibrios = self.equilibrios_sistema(
                a=a, b=b, epsilon=epsilon, I=I
            )
            if not equilibrios:
                resultados[nombre] = {
                    **caso,
                    "equilibrios": [],
                    "solucion": None,
                    "motivo": "No se encontraron equilibrios reales."
                }
                continue

            # Seleccionar el equilibrio más próximo al equilibrio
            # de Hopf de la rama correspondiente.
            equilibrio = min(
                equilibrios,
                key=lambda eq: (
                    (eq["u"] - u_H)**2
                    + (eq["w"] - w_H)**2
                )
            )
            u_eq = equilibrio["u"]
            w_eq = equilibrio["w"]

            # Condición inicial
            # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
            # Se desplaza ligeramente u respecto al equilibrio.
            # Así evitamos iniciar exactamente en un punto fijo.
            y0 = [u_eq + perturbacion_inicial, w_eq]
            
            # Integrar el sistema FHN
            # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
            def sistema(t, y):
                u, w = y
                du, dw = self.campo_FHN(
                    u=u,
                    w=w,
                    a=a,
                    b=b,
                    epsilon=epsilon,
                    I=I
                )
                return [du, dw]

            solucion = solve_ivp(
                fun=sistema,
                t_span=(0.0, t_final),
                y0=y0,
                t_eval=t_eval,
                method="DOP853",
                rtol=rtol,
                atol=atol,
                max_step=max_step
            )

            # Guardar resultados
            # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
            resultados[nombre] = {
                **caso,
                "equilibrios": equilibrios,
                "equilibrio_referencia": equilibrio,
                "condicion_inicial": y0,
                "solucion": solucion,
                "integracion_exitosa": solucion.success,
                "mensaje_integracion": solucion.message
            }

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Construir resumen de resultados
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        resumen = []
        for nombre, resultado in resultados.items():
            solucion = resultado["solucion"]
            if solucion is None:
                resumen.append({
                    "caso": nombre,
                    "I": resultado["I"],
                    "numero_equilibrios": 0,
                    "tipo_equilibrio": "Sin equilibrio real",
                    "estabilidad": "No determinada",
                    "integracion_exitosa": False
                })
                continue

            eq = resultado["equilibrio_referencia"]
            resumen.append({
                "caso": nombre,
                "I": resultado["I"],
                "numero_equilibrios": len(
                    resultado["equilibrios"]
                ),
                "u_equilibrio": eq["u"],
                "w_equilibrio": eq["w"],
                "traza": eq["traza"],
                "determinante": eq["determinante"],
                "autovalores": eq["autovalores"],
                "tipo_equilibrio": eq["tipo"],
                "estabilidad": eq["estabilidad"],
                "integracion_exitosa": solucion.success
            })

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Graficar las series temporales y retratos de fase
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        if mostrar_graficas:
            # Series temporales
            # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
            fig_t, axes_t = plt.subplots(
                4, 1,
                figsize=(11, 12),
                sharex=True
            )
            for ax, (nombre, resultado) in zip(axes_t, resultados.items()):
                solucion = resultado["solucion"]
                if solucion is None:
                    ax.set_title(f"{nombre}: sin equilibrio real")
                    continue
                if not solucion.success:
                    ax.text(
                        0.02, 0.90,
                        "Integración no completada correctamente",
                        transform=ax.transAxes
                    )
                ax.plot(
                    solucion.t,
                    solucion.y[0],
                    label=r"$u(t)$"
                )
                ax.plot(
                    solucion.t,
                    solucion.y[1],
                    label=r"$w(t)$"
                )
                ax.set_title(
                    f"{nombre}" # I={resultado['I']:.3f}
                )
                ax.set_ylabel("Estado")
                ax.grid(alpha=0.3)
                ax.legend()

            axes_t[-1].set_xlabel("Tiempo")
            fig_t.suptitle(rf"Series temporales de FHN-Cortisol, C={C:g} $\mu$M")
            fig_t.tight_layout()
            # Guardar
            if guardar:
                if ruta is None:
                    ruta_completa = Path(nombre_archivo_1)
                else:
                    ruta_completa = Path(ruta) / nombre_archivo_1
                ruta_completa.parent.mkdir(parents=True,exist_ok=True)
                plt.savefig(ruta_completa, bbox_inches="tight", format="pdf")

            # Retratos de fase
            # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
            fig_f, axes_f = plt.subplots(2, 2, figsize=(11, 9))
            for ax, (nombre, resultado) in zip(axes_f.flat, resultados.items()):
                solucion = resultado["solucion"]
                if solucion is None:
                    ax.set_title(nombre)
                    continue
                ax.plot(
                    solucion.y[0],
                    solucion.y[1],
                    label="Trayectoria"
                )
                eq = resultado["equilibrio_referencia"]
                ax.scatter(
                    [eq["u"]],
                    [eq["w"]],
                    marker="o",
                    color="red",
                    s=70, #size
                    zorder=5, # Orden en capas
                    label="Equilibrio"
                )
                ax.set_title(
                    f"{nombre} = {resultado['I']:.3f}"
                )
                ax.set_xlabel(r"$u$")
                ax.set_ylabel(r"$w$")
                # Limita el número de divisiones principales a un máximo de 5 en ambos ejes
                ax.locator_params(axis="both", nbins=5)
                ax.grid(alpha=0.3)
                ax.legend()
            fig_f.suptitle(rf"Retratos fase del sistema FHN-Cortisol, C={C:g} $\mu$M")
            fig_f.tight_layout()
            if guardar:
                if ruta is None:
                    ruta_completa = Path(nombre_archivo_2)
                else:
                    ruta_completa = Path(ruta) / nombre_archivo_2
                ruta_completa.parent.mkdir(parents=True,exist_ok=True)
                plt.savefig(ruta_completa, bbox_inches="tight", format="pdf")

            plt.show()

        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # Resultado global
        # ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        return {
            "parametros": parametros,
            "delta": delta,
            "puntos_hopf": puntos_hopf,
            "corrientes": corrientes,
            "resultados": resultados,
            "resumen": resumen
        }


    # ──────────────────────────────────────────────────────────────────
    #                 Graficación 1D del mapa de calor
        """
        Grafica en 1D el valor mínimo de dV/dt sobre la frontera
        de la elipse repulsora como función de la corriente I,
        para un valor fijo de cortisol GC.

        Parámetros
        ----------
        resultados_elipse : list
            Resultados obtenidos mediante verificar_elipses_candidatos().

        GC_fijo : float
            Concentración de cortisol [μM] para la cual se realiza
            el corte unidimensional.

        I_H_menos : float, opcional
            Corriente crítica inferior de Hopf.

        I_H_mas : float, opcional
            Corriente crítica superior de Hopf.

        titulo : str, opcional
            Título de la figura.

        figsize : tuple
            Tamaño de la figura.

        guardar : bool
            Si True, guarda la figura.

        nombre_archivo : str
            Nombre del archivo de salida.

        Retorna
        -------
        I_vals_1D : ndarray
            Valores de corriente.

        min_dVdt_1D : ndarray
            Valor mínimo de dV/dt para cada corriente.
        """
    def graficar_min_dVdt_1D(
        self,
        resultados_elipse,
        GC_fijo=0.0,
        I_H_menos=None,
        I_H_mas=None,
        titulo=None,
        figsize=(8, 5),
        guardar=False,
        ruta=CARPETA_IMAGENES,
        nombre_archivo="min_dVdt_1D.png"
    ):
        # Seleccionar resultados correspondientes a GC_fijo
        datos_1D = []
        for resultado in resultados_elipse:
            GC = resultado["GC"]
            I = resultado["I"]
            if np.isclose(GC, GC_fijo):
                if resultado.get("elipse_repulsora", False):
                    min_dVdt = resultado["min_dV_dt"]
                    datos_1D.append((I, min_dVdt))

        # Verificar que existan datos
        if len(datos_1D) == 0:
            raise ValueError(
                f"No se encontraron resultados para GC = {GC_fijo}."
            )

        # Ordenar por corriente I
        datos_1D.sort(key=lambda x: x[0])
        I_vals_1D = np.array([dato[0] for dato in datos_1D])
        min_dVdt_1D = np.array([dato[1] for dato in datos_1D])

        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            I_vals_1D,
            min_dVdt_1D,
            linewidth=2,
            label=r"$\min_{\partial\mathcal{E}_{\rho}}\,dV/dt$"
        )

        # Línea horizontal dV/dt = 0
        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.5,
            label=r"$dV/dt=0$"
        )

        # Límites de Hopf
        if I_H_menos is not None:

            ax.axvline(
                I_H_menos,
                linestyle=":",
                linewidth=1.5,
                label=rf"$I_{{H,-}}={I_H_menos:.3f}$"
            )
        if I_H_mas is not None:

            ax.axvline(
                I_H_mas,
                linestyle=":",
                linewidth=1.5,
                label=rf"$I_{{H,+}}={I_H_mas:.3f}$"
            )

        # Sombrear región de Hopf
        if I_H_menos is not None and I_H_mas is not None:

            ax.axvspan(
                I_H_menos,
                I_H_mas,
                alpha=0.15
            )

        # Etiquetas
        ax.set_xlabel(r"$I$", fontsize=12)
        ax.set_ylabel(
            r"$\min_{\mathcal{E}_\rho} \frac{dV}{dt}$",
            fontsize=12
        )
        if titulo is None:
            titulo = (
                rf"$\min_{{\partial\mathcal{{E}}_\rho}}\,dV/dt$ "
                rf"para $C={GC_fijo}\,\mu\mathrm{{M}}$"
            )
        ax.set_title(titulo, fontsize=13)
        ax.grid(
            True,
            alpha=0.3
        )
        ax.legend()
        plt.tight_layout()

        # Guardar
        if guardar:
            if ruta is None:
                ruta_completa = Path(nombre_archivo)
            else:
                ruta_completa = Path(ruta) / nombre_archivo
            ruta_completa.parent.mkdir(parents=True,exist_ok=True)
            plt.savefig(ruta_completa, bbox_inches="tight", format="pdf")

        plt.show()

        # Guardar resultados en el objeto
        self.I_vals_1D = I_vals_1D
        self.min_dVdt_1D = min_dVdt_1D

        return I_vals_1D, min_dVdt_1D