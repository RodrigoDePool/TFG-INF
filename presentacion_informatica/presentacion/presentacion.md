## Presentación del TFG de Matemáticas

####Índice:
1. Superficies: Definición, ejemplos compactas vs no compactas, introducir aquí las superficies 'estándar'. Plantear objetivo dividiendo entre las compactas y no compactas
2. Para el estudio de las compactas necesitamos de dos herramientas: la suma conexa y la triangulación. Explicar las dos dando respectivos ejemplos (en especial con toros, suma de dos toros, triangulación de unn toro).
3. Teorema de Tibor Radó y teorema de clasificación de superficies. Explicar alguna parte del teorema.
4.  Hablar de género para probar que las clases son disjuntas (característica de euler e invariancia topológica).
6. Superficies no compactas (insistir en la segundo numerabilidad que nos permite utilizar Radó!). Hablar de las nuevas dificultades que hay
7. Extremos, clases de extremos y borde ideal. Teorema de Kerekjartó. (Idea de la demostración) "como una superficie se escapa hacia infinito"
8. Propiedades de la frontera ideal, conjunto cerrado del conjunto de Cantor. (recordar referencias)
9. Teorema de I. Richards dando un ejemplo concreto para toda posible frontera ideal. (idea de la demostración)
10. Cardinalidad.



####Texto:
- 'Diapo 0.1: Título'
El objetivo de nuestro trabajo fue el estudio del teorema de clasificación de superficies topológicas.

- 'Diapo 1: Definición de superficies'
 Recordamos que las superficies se definen como espacios topológicos, localmente homeomorfos a bolas abiertas  en R^2, que cumplen ser Hausdorff, segundo numerables y conexos.  Durante la presentación expondremos únicamente superficies orientables, pero todos los resultados tienen su análogo con superficies no orientables. El objetivo será  caracterizar topológicamente una superficie. Como las funciones continuas mantienen la compacidad, la primera distinción es tratar el caso de superficies compactas y no compactas por separado.
 
 - 'Diapo 2.1: Título de clasificación de superficies compactas'
 Tratemos primero el problema más sencillo de la clasificación de superficies compactas. Para caracterizar a una superficie compacta primero necesitaremos dos herramientas: La suma conexa y la triangulación.
 
-  'Diapo 3: La suma conexa'
La suma conexa es una operador entre superficies. La operación consiste en retirar un disco abierto de cada una de las superficies e identificarlas por el borde, como resultado obtenemos otra superficie  dotada de la topología cociente. Se puede comprobar que sumar dos superficies compactas resulta siempre en otra superficie compacta. Las imagenes que tenemos en la diapositiva sugieren una forma gráfica de entender la suma, en este caso, entre dos toros. 
Recordamos aquí que un toro también se puede representar como un rectángulo plano en el que se identifican sus lados apropiadamente (los lados opuestos entre sí, en la dirección de las flechas), como pegando los lados de una tira de papel para formar un 'donut'. En esta  representación la suma conexa de dos toros se sigue como indica la imagen, resultando en un octágono con las aristas identificadas como indican los símbolos en la dirección de las flechas. 
La suma conexa ya nos sirve de herramienta para crear un sinfín de superficies nuevas.

- 'Diapo 4: La triangulación'
Por otra parte, la triangulación de una superficie es una _especie_ de partición de una superficie en conjuntos cerrados, donde cada conjunto es homeomorfo a un triángulo en R^2. Llamaremos un vértice de la superficie a todo punto que se correponda con un vértice de un triángulo en R^2 y de igual manera lo haremos con las aristas, todo conjunto que se corresponda por homeomorfismo con una arista en R^2 entonces le llamaremos arista. A los conjuntos cerrados, además de recubrir la superficie, les exigimos que o sean disjuntos o compartan un solo vértice o una sola arista.  (Los casos que vemos en la imagen no se permiten como triangulación de una superficie)

- 'Diapo 5: Ejemplo de triangulación'
La triangulación es una herramienta muy potente que esencialmente nos permite manera una superficie como si de un conjunto de triángulos en R^2 se tratáse. Esencialmente nos permite _teselar_ una superficie por triángulos. En la diapositiva vemos un ejemplo de triangulación de un toro (en su representación plana) y la triangulación de una esfera (donde a cada triángulo en la esfera podríamos deformarlo para formar uno en el plano). 
La triangulación es una herramienta muy potente pero para utilizarla primero es necesario probar que una tal triangulación existe. Este hecho fue demostrado por Tibor Radó, que asegura que toda superficie **separable** es triangulable. Recordamos aquí que en la definición de superficie que dimos exigimos la segunda numerabilidad, por lo que todas las superficies que tratamos son separables y con ello triangulables. Con este par de herramientas ya se puede abordar el teorema de clasificación

- 'Diapo 6: Teorema de clasificación de superficies compactas'
El teorema asegura que toda superficie compacta y orientable es homeomorfa a una esfera o a una suma conexa de _n_ toros. De modo que, salvo homeomorfismo, todas las superficies están contempladas en esta sucesión de superficies (imagen)

- 'Diapo 7: Demostración'
Antes vimos que un toro en su representación plana tiene **esta pinta**. Además, comprobamos que al hacer la suma conexa de dos toros tenemos **este octágono**. En general, siguiendo el mismo procedimiento para sumar toros planos, podemos probar que la suma de n toros planos es un polígono de 4n lados, donde las aristas se identifican como se indica en **esta tercera figura**. Para demostrar el teorema lo que se hace es probar que toda superficie es homeomorfa a un polígono con este aspecto.

- 'Diapo 8: Uso de Tibor Radó y procedimiento'
Primero, por el teorema de Tibor Radó, podemos representar nuestra superficie como el espacio cociente de una triangulación y, con homeomorfismos, acabar representando cualquier superficie como un polígono de n lados con sus aristas identificadas a pares. 
A partir de aquí, la demostración procede recortando e identificando partes del polígono en un proceso iterativo y finito, que, al finalizar, acaba con una de las representaciones que vistas en la diapositiva anterior.

- 'Diapo 9: Género'
Utilizando la triangulación y la característica de Euler se demuestra que cada una de las clases topológicas presentadas (imagen) son disjuntas, esto es, que entre ellas no son homeomorfas.  A partir de esto, definimos el número de toros n como el **género** de una superficie (en caso de la esfera tenemos género 0). Y con esto acabamos el estudio de las superficies compactas.

- 'Diapo 10: Superficies no compactas'
Ejemplos de superficies no compactas podemos construirlos retirando  puntos de una superficie compacta, o conjuntos abiertos o también las observamos en superficies que se extienen al infinito. 

- 'Diapo 10.1: Superficies no compactas'
Vimos que para las superficies compactas el número de toros (el género) era suficiente para caracterizar una superficie orientable. Está claro que para las superficies no compactas será necesario que se comparta el género, porque cualquier homeomorfismo entre superficies podría restringirse a una subsuperficie compacta. Sin embargo, no está tan claro que el género sea suficiente para caracterizar la superficie y, de hecho, no lo es. Se dan aquí tres ejemplos de superficies que comparten género pero no son topológicamente equivalentes: Una de género 0, otras de género 1 y, por último, un par de género infinito.

- 'Diapo 11: Borde ideal'
Necesitamos entonces de un nuevo invariante topológico que acabe de caracterizar las superficies no compactas. Para ello primero tenemos que ver como una superficie puede extender a infinito.

- 'Diapo 12: Extremos'
Un extremo en una superficie es una sucesión encajada de subconjuntos no acotados, conexo, que cumplen tener frontera compacta y que, además, cumplen que para cualquier subconjunto compacto existe un elemento de la sucesión a partir del cual  ya no se interseca al conjunto compacto (es decir, que hay un P_n cuya intersección con cualquier conjunto A es el vacío).
Esta definición nos permite definir el borde ideal.

- 'Diapo 13: El borde ideal'
El borde ideal   de una superficie (se suele denotar como B(S)) es el conjunto de estos extremos de la superficie salvo equivalencia. Este conjunto describe como una superficie se extiende a infinito. Al borde ideal se le puede dotar de una topología, se coge cualquier subconjunto U no acotado de frontera compacta y se dice que U* (dentro de B(S)) es el conjunto de extremos que están contenidos dentro de U. Estos  conjuntos componen la base de una topología que definimos para el borde ideal.
Dentro del borde ideal podemos considerar el conjunto de extremos que tienen género infinito, ese subconjunto es cerrado y se denota por B'(S). A partir de ahora llamaremos borde ideal a la tupla (B(S), B'(S)) considerada con su topología. Este invariante acabará por caracterizar completamente a las superficies orientables.

- 'Diapo 14: Teorema de Kerekjarto'
El teorema de Kerekjarto nos da la clasificación para superficies no compactas. Nos dice que dadas dos superficies orientables, no compactas, del mismo género S y S',  son homeomorfas si y solo si lo son sus bordes ideal considerados como tuplas. 
La demostración es extremadamente técnica, la diea es expresar las dos superficies como una sucesión de subsuperficies compactas encajadas, y, usando el homoemorfismo entre los bordes ideales, establece homeomorfismos entres las subsuperficies. Luego el homeomorfismo entre las superficies acabará siendo el límite de estos homeomorfismos.

- 'Diapo 15: Clases de equivalencia de superficies no compactas'
El teorema de Kerekjarto nos da una caracterización completa de una superficie no compacta. Sin embargo, no tenemos un abánico de superficies representantes, como lo teníamos para las superficies compactas con la suma de n toros. 
Una construcción de Richards completa en este sentido el resultado para superficies no compactas. 
Primero, hace el apunte de que todo borde ideal (B(S), B'(S)) es homeomorfo a (X,Y) siendo Y subconjunto cerrado de X y siendo X un subconjunto cerrado del conjunto de Cantor. Esta observación la hace partiendo de que el borde ideal es un espacio topológico totalmente inconexo, separable y compacto.
Richards prueba que, de hecho, para cualquier par (X,Y) existe una superficie con ese borde ideal. La construcción sigue la siguiente idea:

- 'Diapo 16: Construcción de Richards'
[TODO IDEA DE LA CONSTRUCCIÓN]


- 'Diapo 17.1: Representantes para superficies no compactas'
Vimos que cualquier borde ideal es homeomorfo a un subconjunto cerrado del conjunto de Cantor. Recíprocamente, para cualquier subconjunto cerrado del conjunto de Cantor tenemos una superficie con este borde ideal. Esto nos permite dar una superficie representante para todas las clases de equivalencia. 
	1. Si una superficie es de género infinito, esta información ya viene determinada por su borde ideal. De modo que, la construcción anterior ya nos da una superficie representante para cualquier borde ideal posible de género infinito.
	2. Si la superficie es de género finito n, basta con sumarle (en el sentido de la suma conexa) a la superficie anterior n toros. Como sumar una superficie compacta no altera el borde ideal, tendremos entonces una superficie representante para cualquier borde y cualquier género.
 
 Esto ya nos termina de dar todas las posibles combinaciones salvo homeomorfimos de superficies no compactas.
 
- 'Diapo 17.2: Cardinalidad'
Recapitulando en la clasificación de superficies compactas dijimos que una superficie es homeomorfa o una esfera o una suma conexa de n toros, con lo que sabemos que existe una cantidad numerable de superficies compactas salvo homeomorfismo.
 Para el caso de superficies no compactas, vimos que, fijando el género, podemos crear al menos una superficie no compacta por cada subconjunto cerrado del conjunto de Cantor. Contando el número de subconjuntos cerrados del conjunto de Cantor (no homeomorfos entre sí), concluimos que tenemos aleph\_1 superficies no compactas salvo homeomorfismo.
 Como dato curioso, si en la definción de superficie retiramos la hipótesis de **2do numerabilidad**, obtenemos que existen alef\_2 superficies salvo homeomorfismo. 