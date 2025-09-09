--ejercicio 3

    (x, y) = T{(w, z)} = (w + cz, z + dw)

    (a) Encontrar la expresion de la matriz T y T^−1.

        [T] = (1  c)        [T^-1] = 1 /1 -cd (1 -c)
              (d  1)                          (-d 1)

    (b)
        
        en este caso b "estira" el circulo en un angulo, mientras mas grande mas lo hace

    (c)

        Ahora pasa algo similar con c pero en vez de estirarlo lo "aplasta" en otro angulo

--ejercicio 4

   (X,Y) = (r*cos(t), r*sin(t))

   entonces despues de rotar un angulo p, su nuevo angulo es t+p y entonces:
   (Xp, Yp) = (r*cos(t+p),r*sin(t+p))

   y por mlas propiedades de los angulos es igual a:
   cos(t+p) = cos(t)*cos(p) -sin(t)sin(p)
   sin(t+p) = sin(p)*cos(t) + cos(p)sin(t)

   sustituyendo:

   Xp = r*cos(t)*cos(p) - r*sin(t)sin(p) = X*cos(p) - Y*sin(p)
   Yp = r*sin(p)*cos(t) + r*cos(p)sin(t) = X*sin(p) + Y cos(p)

   