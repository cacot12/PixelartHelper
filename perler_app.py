
import os
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage
from scipy.cluster.vq import kmeans2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings('ignore')


def rgb2lab(c):
    c = np.asarray(c, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (c @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)



def bosquejo_lineas(grid, oscuros, min_region=3):
    """Líneas de 1 px entre zonas. Los contornos finos oscuros se absorben; el pelo (zona oscura gruesa) queda como zona."""
    g = grid.copy()
    H, W = g.shape
    vecinos = ((-1, 0), (1, 0), (0, -1), (0, 1))
    osc = np.isin(g, list(oscuros))
    est = np.ones((3, 3), bool)
    nucleo = ndimage.binary_erosion(osc, structure=est, border_value=0)
    gruesa = ndimage.binary_dilation(nucleo, structure=est) & osc
    finos = osc & ~gruesa
    for _ in range(3):                       # absorber contornos finos en la zona vecina
        ys, xs = np.nonzero(finos)
        if len(ys) == 0:
            break
        nuevo, nf = g.copy(), finos.copy()
        for y, x in zip(ys, xs):
            v = [g[y+dy, x+dx] for dy, dx in vecinos
                 if 0 <= y+dy < H and 0 <= x+dx < W and g[y+dy, x+dx] >= 0 and not finos[y+dy, x+dx]]
            if v:
                nuevo[y, x] = np.bincount(v).argmax()
                nf[y, x] = False
        g, finos = nuevo, nf
    for c in np.unique(g[g >= 0]):           # absorber zonas diminutas
        lab, n = ndimage.label(g == c)
        for i in range(1, n + 1):
            zona = lab == i
            if zona.sum() < min_region:
                borde = ndimage.binary_dilation(zona) & ~zona
                v = g[borde]
                v = v[v >= 0]
                if len(v):
                    g[zona] = np.bincount(v).argmax()
    fg = g >= 0
    P = np.pad(g, 1, constant_values=-1)
    centro = P[1:-1, 1:-1]
    linea = np.zeros_like(fg)
    for dy, dx in ((0, 1), (1, 0)):          # borde interno (derecha/abajo)
        linea |= fg & (P[1+dy:H+1+dy, 1+dx:W+1+dx] != centro)
    for dy, dx in ((0, -1), (-1, 0)):        # silueta (izquierda/arriba)
        linea |= fg & (P[1+dy:H+1+dy, 1+dx:W+1+dx] == -1)
    return linea


def imagen_bosquejo(linea, P=12):
    """Bosquejo blanco y negro con cuadrícula (una línea más marcada cada 10 px)."""
    S = linea.shape[0]
    g = Image.new('RGB', (S * P, S * P), 'white')
    d = ImageDraw.Draw(g)
    for i in range(S + 1):
        col = (170, 170, 170) if i % 10 == 0 else (225, 225, 225)
        d.line([(i * P, 0), (i * P, S * P)], fill=col)
        d.line([(0, i * P), (S * P, i * P)], fill=col)
    for y, x in zip(*np.nonzero(linea)):
        d.rectangle([x * P + 1, y * P + 1, (x + 1) * P - 1, (y + 1) * P - 1], fill=(30, 30, 30))
    return g


def convertir(ruta, size=72, colores=10, quitar_fondo=True, contorno_fino=True,
              margen=1, tol_fondo=30, oscuro=1.05, fusion=9):
    """Devuelve un diccionario con sprite, vista de cuentas y texto de la paleta."""
    SIZE, MARGEN = int(size), int(margen)
    N = SIZE - 2 * MARGEN
    if N < 4:
        raise ValueError('El tamaño es demasiado pequeño.')
    F = 6
    rng = np.random.default_rng(0)

    # 1. cargar y quitar el fondo
    arr = np.array(Image.open(ruta).convert('RGBA'))
    if quitar_fondo:
        if arr[..., 3].min() >= 250:
            borde = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])[:, :3].astype(float)
            fondo = np.median(borde, axis=0)
            cerca = np.sqrt(((arr[..., :3].astype(float) - fondo) ** 2).sum(-1)) < tol_fondo
            lab, _ = ndimage.label(cerca)
            con = (set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])) - {0}
            arr[..., 3] = np.where(np.isin(lab, list(con)), 0, 255)
    else:
        arr[..., 3] = 255
    im = Image.fromarray(arr)
    bbox = im.getbbox()
    if bbox:
        im = im.crop(bbox)
    w, h = im.size
    L = max(w, h)
    sq = Image.new('RGBA', (L, L), (0, 0, 0, 0))
    sq.paste(im, ((L - w) // 2, (L - h) // 2))

    # 2. paleta automática (k-means en Lab, con más peso a los bordes)
    big = np.array(sq.resize((N * F, N * F), Image.LANCZOS)).astype(float)
    op = big[..., 3] > 128
    if not op.any():
        raise ValueError('No se encontró ninguna figura (¿se borró todo el fondo?). Prueba sin quitar el fondo.')
    rgb = big[..., :3]
    lab_img = rgb2lab(rgb)
    grad = np.hypot(ndimage.sobel(lab_img[..., 0], 0), ndimage.sobel(lab_img[..., 0], 1))
    es_borde = op & (grad > np.percentile(grad[op], 85))
    sel = np.concatenate([
        np.nonzero(op.ravel())[0][rng.choice(op.sum(), min(40000, op.sum()), replace=False)],
        np.repeat(np.nonzero(es_borde.ravel())[0], 2)])
    m_lab = lab_img.reshape(-1, 3)[sel]
    m_rgb = rgb.reshape(-1, 3)[sel]
    K0 = int(min(colores + 8, 24))
    _, etq = kmeans2(m_lab, K0, minit='++', seed=1)
    grupos = [np.nonzero(etq == k)[0] for k in range(K0) if (etq == k).any()]
    centro = lambda g: m_lab[g].mean(0)
    while len(grupos) > 1:
        C = np.array([centro(g) for g in grupos])
        d = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))
        np.fill_diagonal(d, 1e9)
        i, j = np.unravel_index(d.argmin(), d.shape)
        if len(grupos) > colores or d[i, j] < fusion:
            grupos[i] = np.concatenate([grupos[i], grupos[j]])
            grupos.pop(j)
        else:
            break
    cent = np.array([centro(g) for g in grupos])
    cent_rgb = np.array([m_rgb[g].mean(0) for g in grupos])

    # 3. asignar y votar por bloque
    idx = ((lab_img[..., None, :] - cent) ** 2).sum(-1).argmin(-1)
    K = len(cent)
    lum_pal = cent[:, 0]
    osc = lum_pal < max(35, lum_pal.min() + 8)
    peso = np.where(osc, oscuro, 1.0)
    cnt = np.stack([((idx == k) & op).reshape(N, F, N, F).sum((1, 3)) for k in range(K)], -1) * peso
    res = cnt.argmax(-1)
    res[op.reshape(N, F, N, F).mean((1, 3)) < 0.5] = -1

    def vec4(M, y, x):
        return [M[yy, xx] for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)) if 0 <= yy < N and 0 <= xx < N]

    # 4. quitar píxeles sueltos
    for _ in range(2):
        nuevo = res.copy()
        for y in range(N):
            for x in range(N):
                c = res[y, x]
                if c < 0:
                    continue
                v = vec4(res, y, x)
                if c not in v:
                    v = [k for k in v if k >= 0]
                    if v:
                        nuevo[y, x] = np.bincount(v).argmax()
        res = nuevo

    # 5. adelgazar contornos gruesos
    if contorno_fino and osc.any():
        es = lambda v: (v >= 0) and osc[v]
        for _ in range(2):
            nuevo = res.copy()
            for y in range(1, N - 1):
                for x in range(1, N - 1):
                    if not es(res[y, x]):
                        continue
                    if es(res[y, x-1]) and es(res[y, x+1]) and es(res[y-1, x]) and es(res[y+1, x]):
                        otros = [k for k in vec4(res, y, x) if k >= 0 and not osc[k]]
                        if otros:
                            nuevo[y, x] = np.bincount(otros).argmax()
            res = nuevo

    # 6. cerrar la silueta con una línea de 1 px
    mask = res >= 0
    if contorno_fino and osc.any() and mask.any():
        ero = ndimage.binary_erosion(mask, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]], border_value=0)
        canto = mask & ~ero
        es_osc = np.isin(res, np.nonzero(osc)[0])
        en_canto = res[canto & es_osc]
        color_linea = np.bincount(en_canto).argmax() if len(en_canto) else int(lum_pal.argmin())
        res[canto & ~es_osc] = color_linea

    # 7. lienzo final, paleta ordenada por uso
    grid = np.full((SIZE, SIZE), -1)
    grid[MARGEN:MARGEN + N, MARGEN:MARGEN + N] = res
    orden = np.argsort([-(grid == k).sum() for k in range(K)])
    remap = np.full(K, -1)
    remap[orden] = np.arange(K)
    grid = np.where(grid >= 0, remap[np.clip(grid, 0, K - 1)], -1)
    pal = np.round(cent_rgb[orden]).astype(int)
    usados = [k for k in range(K) if (grid == k).any()]

    rgba = np.zeros((SIZE, SIZE, 4), 'uint8')
    for k in usados:
        rgba[grid == k] = (*pal[k], 255)
    sprite = Image.fromarray(rgba)

    P = 14
    prev = Image.new('RGB', (SIZE * P, SIZE * P), (245, 245, 245))
    dr = ImageDraw.Draw(prev)
    for y in range(SIZE):
        for x in range(SIZE):
            c = grid[y, x]
            if c < 0:
                continue
            dr.ellipse([x*P+1, y*P+1, (x+1)*P-2, (y+1)*P-2], fill=tuple(int(v) for v in pal[c]))
            dr.ellipse([x*P+5, y*P+5, (x+1)*P-6, (y+1)*P-6], fill=(245, 245, 245))

    # 8. conteo de cuentas
    lineas = [f'{SIZE}x{SIZE} px, {len(usados)} colores']
    total = 0
    for k in usados:
        n = int((grid == k).sum())
        total += n
        lineas.append(f'#{k+1:<2d} #{pal[k][0]:02x}{pal[k][1]:02x}{pal[k][2]:02x}   {n} cuentas')
    lineas.append(f'Total: {total} cuentas')
    osc_final = {int(remap[k]) for k in np.nonzero(osc)[0]}
    linea = bosquejo_lineas(grid, osc_final)
    return dict(sprite=sprite, cuentas=prev, linea=linea, texto='\n'.join(lineas), size=SIZE)


def guardar(res, carpeta, prefijo):
    s = res['size']
    res['sprite'].save(os.path.join(carpeta, f'{prefijo}_pixel_{s}.png'))
    res['sprite'].resize((s * 10, s * 10), Image.NEAREST).save(os.path.join(carpeta, f'{prefijo}_pixel_{s}_x10.png'))
    res['cuentas'].save(os.path.join(carpeta, f'{prefijo}_perler_{s}.png'))
    imagen_bosquejo(res['linea'], 12).save(os.path.join(carpeta, f'{prefijo}_bosquejo_{s}.png'))
    with open(os.path.join(carpeta, f'{prefijo}_paleta_{s}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{prefijo}: ' + res['texto'])


# ======================================================================
#  VENTANA
# ======================================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title('Perler / Pixel Art')
        root.minsize(900, 600)
        self.archivos = []
        self.resultados = {}
        self.fotos = []

        izq = ttk.Frame(root, padding=12)
        izq.pack(side='left', fill='y')
        der = ttk.Frame(root, padding=12)
        der.pack(side='left', fill='both', expand=True)

        # --- subir imágenes ---
        ttk.Button(izq, text='Subir imágenes…', command=self.subir).pack(fill='x')
        self.lista = tk.Listbox(izq, width=34, height=8, exportselection=False)
        self.lista.pack(fill='x', pady=(6, 0))
        self.lista.bind('<<ListboxSelect>>', self.al_elegir)
        ttk.Button(izq, text='Quitar de la lista', command=self.quitar).pack(fill='x', pady=(4, 10))

        # --- opciones ---
        op = ttk.LabelFrame(izq, text='Opciones', padding=10)
        op.pack(fill='x')
        self.v_size = tk.StringVar(value='72')
        self.v_col = tk.StringVar(value='10')
        self.v_fondo = tk.BooleanVar(value=True)
        self.v_fino = tk.BooleanVar(value=True)
        ttk.Label(op, text='Tamaño (px, cuadrado)').grid(row=0, column=0, sticky='w')
        ttk.Entry(op, textvariable=self.v_size, width=7).grid(row=0, column=1, padx=6, pady=2)
        ttk.Label(op, text='Máximo de colores').grid(row=1, column=0, sticky='w')
        ttk.Entry(op, textvariable=self.v_col, width=7).grid(row=1, column=1, padx=6, pady=2)
        ttk.Checkbutton(op, text='Quitar el fondo', variable=self.v_fondo).grid(row=2, column=0, columnspan=2, sticky='w', pady=(6, 0))
        ttk.Checkbutton(op, text='Contorno fino', variable=self.v_fino).grid(row=3, column=0, columnspan=2, sticky='w')

        # --- carpeta de salida ---
        sa = ttk.LabelFrame(izq, text='Carpeta de salida', padding=10)
        sa.pack(fill='x', pady=10)
        self.v_salida = tk.StringVar(value='')
        ttk.Entry(sa, textvariable=self.v_salida, width=30).pack(fill='x')
        ttk.Button(sa, text='Elegir…', command=self.elegir_salida).pack(fill='x', pady=(4, 0))
        ttk.Label(sa, text='Vacío = junto a cada imagen', foreground='#777').pack(anchor='w')

        # --- acciones: primero ver, luego guardar ---
        self.btn_gen = ttk.Button(izq, text='1. Generar vista previa', command=self.generar)
        self.btn_gen.pack(fill='x', pady=(4, 0))
        self.btn_guardar = ttk.Button(izq, text='2. Guardar lo que veo', command=self.guardar_todo)
        self.btn_guardar.pack(fill='x', pady=(4, 0))
        self.btn_guardar.state(['disabled'])
        self.estado = ttk.Label(izq, text='Sube una o varias imágenes.', wraplength=250, foreground='#555')
        self.estado.pack(fill='x', pady=8)

        # --- vista previa ---
        self.nb = ttk.Notebook(der)
        self.nb.pack(fill='both', expand=True)
        self.l_sprite = ttk.Label(self.nb, anchor='center')
        self.l_perler = ttk.Label(self.nb, anchor='center')
        self.l_bosq = ttk.Label(self.nb, anchor='center')
        self.nb.add(self.l_sprite, text='Pixel art')
        self.nb.add(self.l_perler, text='Perler')
        self.nb.add(self.l_bosq, text='Bosquejo')

    # ---- acciones ----
    def subir(self):
        rutas = filedialog.askopenfilenames(
            title='Elige imágenes',
            filetypes=[('Imágenes', '*.png *.jpg *.jpeg *.webp *.bmp *.gif'), ('Todos', '*.*')])
        for r in rutas:
            if r not in self.archivos:
                self.archivos.append(r)
                self.lista.insert('end', os.path.basename(r))
        if rutas:
            self.estado.config(text=f'{len(self.archivos)} imagen(es) en la lista. Pulsa «Generar vista previa».')

    def quitar(self):
        for i in reversed(self.lista.curselection()):
            self.resultados.pop(self.archivos[i], None)
            del self.archivos[i]
            self.lista.delete(i)
        if not self.resultados:
            self.btn_guardar.state(['disabled'])

    def elegir_salida(self):
        d = filedialog.askdirectory(title='Carpeta de salida')
        if d:
            self.v_salida.set(d)

    def al_elegir(self, _=None):
        sel = self.lista.curselection()
        if sel and self.archivos[sel[0]] in self.resultados:
            self.mostrar(self.resultados[self.archivos[sel[0]]])

    def mostrar(self, res):
        s = res['size']
        k = max(1, min(8, 520 // s))
        f1 = ImageTk.PhotoImage(res['sprite'].resize((s * k, s * k), Image.NEAREST))
        c = res['cuentas']
        lado = min(c.width, 560)
        f2 = ImageTk.PhotoImage(c.resize((lado, lado), Image.LANCZOS))
        f3 = ImageTk.PhotoImage(imagen_bosquejo(res['linea'], max(4, 560 // s)))
        self.fotos = [f1, f2, f3]
        self.l_sprite.config(image=f1)
        self.l_perler.config(image=f2)
        self.l_bosq.config(image=f3)

    def generar(self):
        """Solo calcula y muestra; NO guarda nada."""
        if not self.archivos:
            messagebox.showinfo('Falta una imagen', 'Primero sube al menos una imagen.')
            return
        try:
            size = int(self.v_size.get())
            colores = int(self.v_col.get())
            if not (8 <= size <= 300) or not (2 <= colores <= 24):
                raise ValueError
        except ValueError:
            messagebox.showerror('Valores no válidos', 'Tamaño: entre 8 y 300.\nColores: entre 2 y 24.')
            return
        self.btn_gen.state(['disabled'])
        self.btn_guardar.state(['disabled'])
        self.resultados = {}
        errores = []
        for i, ruta in enumerate(self.archivos):
            self.estado.config(text=f'Procesando {i + 1}/{len(self.archivos)}: {os.path.basename(ruta)}…')
            self.root.update()
            try:
                self.resultados[ruta] = convertir(ruta, size, colores, self.v_fondo.get(), self.v_fino.get())
            except Exception as e:
                errores.append(f'{os.path.basename(ruta)}: {e}')
        self.btn_gen.state(['!disabled'])
        if self.resultados:
            ultimo = [r for r in self.archivos if r in self.resultados][-1]
            self.lista.selection_clear(0, 'end')
            self.lista.selection_set(self.archivos.index(ultimo))
            self.mostrar(self.resultados[ultimo])
            self.btn_guardar.state(['!disabled'])
        msg = f'Vista previa lista ({len(self.resultados)} imagen(es)). Revisa las pestañas y, si te gusta, pulsa «Guardar».'
        if errores:
            msg = f'Con error: {len(errores)}. ' + msg
            messagebox.showwarning('Algunas imágenes fallaron', '\n'.join(errores))
        self.estado.config(text=msg)

    def guardar_todo(self):
        """Guarda exactamente lo que se generó en la vista previa."""
        if not self.resultados:
            return
        hechos, errores, carpetas = 0, [], set()
        for ruta, res in self.resultados.items():
            try:
                carpeta = self.v_salida.get().strip() or os.path.dirname(ruta)
                os.makedirs(carpeta, exist_ok=True)
                guardar(res, carpeta, os.path.splitext(os.path.basename(ruta))[0])
                carpetas.add(carpeta)
                hechos += 1
            except Exception as e:
                errores.append(f'{os.path.basename(ruta)}: {e}')
        if errores:
            messagebox.showwarning('No se pudo guardar', '\n'.join(errores))
        dest = next(iter(carpetas)) if len(carpetas) == 1 else 'la carpeta de cada imagen'
        self.estado.config(text=f'Guardadas {hechos} imagen(es) en {dest}.')


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()