import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def pointsGrid(esquinas):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 46),
                        np.linspace(esquinas[0,1], esquinas[1,1], 10))

    [w2, z2] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 10),
                        np.linspace(esquinas[0,1], esquinas[1,1], 46))

    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
                         
    return wz

def proyectarPts(T, wz):
    assert(T.shape == (2,2)) # chequeo de matriz 2x2
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida   
    xy = None
    ############### Insert code here!! ######################3    
    xy = T @ wz
    ############### Insert code here!! ######################3
    return xy

def proyectarPts2(T, wz):
    assert(T.shape == (3,3))
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida
    wz_ext = np.vstack((wz, np.ones((1,wz.shape[1])))) # 3 x N
    xy_ext = T @ wz_ext  # 3 x N    
    xy = xy_ext[:2,:]
    return xy


          
def vistform(T, wz, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
               [minlim[1]-bump[1], maxlim[1]+bump[1]]]             

    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')    
    
def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)


def main():
    print('Ejecutar el programa')
    # generar el tipo de transformacion dando valores a la matriz T
    T = pd.read_csv('funcion.csv', header=None).values
    T_def = pd.read_csv('T.csv', header=None).values
    corners = np.array([[0,0],[100,100]])
    # corners = np.array([[-100,-100],[100,100]]) array con valores positivos y negativos
    wz = pointsGrid(corners)
    vistform(T_def, wz, 'Deformar coordenadas')
    vistform(T, wz, 'Encoger coordenadas')

    plt.show()  # <-- Agrega esta línea

    t = np.linspace(0, 2*np.pi, 400)
    r = 1.0
    circle = np.vstack((r*np.cos(t), r*np.sin(t)))   # 2 x N

    T = np.array([[2,0],[0,3]])
    ellipse = T @ circle

    plt.figure()
    plt.plot(circle[0,:], circle[1,:], label='Circunferencia r=1')
    plt.plot(ellipse[0,:], ellipse[1,:], label='Imagen: Elipse')
    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    plt.show()

    d = 0.1   # probá d = 0.5, 1, -1, 2 ...
    c = d
    t = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack((np.cos(t), np.sin(t)))   # 2 x N

    T = np.array([[1,c],[d,1]])
    image = T @ circle

    plt.figure()
    plt.plot(circle[0,:], circle[1,:], label='Circunf. original')
    plt.plot(image[0,:], image[1,:], label=f'Imagen (d={d})')
    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    plt.show()

    x= np.pi*2
    def R(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    puntos= pointsGrid(corners)
    T2= R(x)    
    vistform(T2, puntos, 'Rotacion')
    plt.show()

    def rotYreescalamient0(x,y):
        matriz_rotacion = R(np.pi/4)
        matriz_escalamiento = np.array([[2,0],[0,3]])
        matriz_rotacion_inv = R(-np.pi/4)
        return matriz_rotacion_inv@matriz_escalamiento @ matriz_rotacion @ np.array([[x],[y]])
    
    p =rotYreescalamient0(20,10)
    print(p)

    def rotYreescalamient0_circ(vec):
        matriz_rotacion = R(np.pi/4)
        matriz_escalamiento = np.array([[2,0],[0,3]])
        matriz_rotacion_inv = R(-np.pi/4)
        return matriz_rotacion_inv@matriz_escalamiento @ matriz_rotacion @ vec
    
    elipse = rotYreescalamient0_circ(circle)

    plt.figure()
    plt.plot(circle[0,:], circle[1,:], label='Circunferencia original')
    plt.plot(elipse[0,:], elipse[1,:], label='Transformada (elipse rotada)')
    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    plt.title('Rotación + Reescalamiento + Rotación inversa')
    plt.show()
    

    
    
if __name__ == "__main__":
    main()
