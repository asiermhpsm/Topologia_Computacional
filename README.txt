EXPLICACIÓN GENERAL DEL CÓDIGO

El funcionamiento del código se basa en la herencia de clases (u objetos, como lo quieras llamar).
El concepto de hererncia es sencillo y se usa para agrupar clases que tendrán metodos o atributos en común.
En primer lugar se crea una clase padre con todos los metodos y atributos en común y luego se crean las
clases hijas, en las que se puede añadir más metodos y atributos e incluso hacer alguna modificación a los métodos
de la clase padre. Un ejemplo típico es una clase padre que representa vehículos con atributos como motor 
y métodos como arrancar, moverse... Luego de esa clase padre se crean clases hijas como por ejemplo
coche o moto con atributos más específicos como puertas o manillar.
En nuestro caso la clase padre será una clase que "entenderá" solo de simplices simbólicos, es decir
0,1,2,3... y que tendrá todo un peso 0 por defecto (aunque se pueda cambiar manualmente). Por otro lado las clases hijas
guardarán además las coordenadas de los puntos y en su constructor calculará los pesos de los simplices.

De esta manera la clase gracias a la clase padre se podrán hacer todas las operaciones que no requieran las 
coordenadas de los puntos y las clases hijas sólo deberán encargarse de, en su constructor, calcular los pesos
de cada posible simplice y añadirlo de forma simbólica al atributo de la clase padre donde se guarda toda esa información.
Las funciones representaSubnivel() y representaPuntos() se han definido en la clase padre ya que está en común
con las posibles clases hijas, pero para llamarla se necesitan los puntos que se le pasan a una clase hija por lo
que solo debería poder llamarse desde la instancia de una clase hija. Para controlarlo se ha creado una excepcion
en la que se accede al atributo 'points' (las coordenadas de los puntos, que no existe en la clase padre pero sí en las hijas) 
y si da una excepción (es decir 'points' no existe y por lo tanto estoy en una instancia de la clase padre) se 
para la ejecución indicando el error.

Por otro lado, al crear el diagrama de persistencia y el código de barras, muchas veces ocurría que aparecían puntos
tan alejados que no se podian apreciar los resultados. Por lo tanto, una vez calculado todo, se limitan los ejes
de 0 hasta el último punto de persistencia que no tenga el nacimiento y la muerte a la vez.
Además hemos hecho adicionalmente una función animaComplejo() que va representado como evoluciona el complejo a 
medida que el peso aumenta y una función analiza() que, en una misma pestaña, representa: los puntos, la animcaión del
complejo, el diagrama de persistenca y el código de barras.

Los archivos circulo.py, circulos.py, rosa.py y aleatorios.py son ejemplos del funcionamiento del código en
los que se generan puntos aleatorios con distintas formas y se llama a la funcion analiza(). Los archivos 
.ipynb contienen todos los ejemplos subidos en el moodle. El archivo Practica4.ipynb tiene varios cuadros de código 
sin ejecutar (los que generan muchas imágenes) para reducir el tamaño del archivo y así poder subirlo al moodle. Si se 
ejecuta todo (suele haber un botón) debería dar los resultados deseado. En Practica4.ipynb se compara los resultados usando
alpha-complejos vs Vietoris-Rips (no se recomienda volver a ejecutar, puede tardar bastante).