# GUÍA PARA REALIZAR EL TRABAJO FIN DE MÁSTER

**El alumno tendrá que elegir una de las opciones siguientes** *(nota: el trabajo es individual)*

---

## 1. Análisis de un dataset (orientación Data Scientist)

### a. Objetivos

- Analizar un dataset disponible públicamente (Kaggle, UCI Machine Learning Repository, Gapminder u otra fuente que el alumno considere, siempre que el conjunto no tenga ningún derecho de uso).
- Se ha de evitar centrar el TFM en datasets ampliamente conocidos y presentados en blogs, libros. Por ejemplo: Titanic, Ames Housing, etc.
- No recomendamos el uso de datos sintéticos; las conclusiones suelen ser limitadas y no favorecen el desarrollo completo y esperado de un TFM.

### b. Fases

El estudio y análisis de este dataset deberá cumplir, de forma general, las fases de un proceso de modelización analítica estándar, entre las que se encuentran:

1. Crear un análisis descriptivo del conjunto (gráfico en lo posible).
2. Realizar las transformaciones que se consideren más adecuadas o relevantes para el conjunto.
3. Crear modelos de predicción utilizando diferentes técnicas de modelización (machine learning), justificando su uso, determinando el nivel de precisión y detallando las bondades y debilidades de cada técnica utilizada.
   - Se busca que el alumno proponga un desarrollo que aporte algo más que lo que se podría conseguir con un AutoML.
   - Se espera que el TFM no siga un esquema equivalente a un trabajo de fin de módulo. El TFM es bastante más.
4. Discusión de los resultados del modelo: explicatividad/interpretabilidad.
5. Realizar un informe final de conclusiones en el que las diferentes fases queden bien delimitadas, y en particular donde las mejoras ofrecidas por el modelo queden bien explicitadas, así como las mejoras futuras que podrían plantearse sobre el trabajo realizado.
   - Este informe final tendrá una orientación tal que pueda ser entendida por un equipo de "Negocio".
   - Podrá incluir elementos técnicos, pero deberá incluir en mayor proporción detalles que expliquen y justifiquen los resultados del modelo a una persona sin muchos conocimientos técnicos.
6. Además de las fases anteriores, se valorará muy positivamente que el modelo pueda **productivizarse**.
   - Es decir, que el modelo pueda ser utilizado en un equivalente a una aplicación empresarial, donde se le puedan pasar nuevos valores y el modelo devuelva una predicción.

### c. Extensión

- La extensión total del trabajo no debe superar **20 caras** (tamaño folio), sin contar los anexos, ni el índice de contenidos, ni la portada o contraportada.
  - El tamaño de letra y el interlineado quedan a decisión del alumno, primando el sentido común y la legibilidad (tamaño preferido: 10 u 11).
  - Tipo de letra recomendado: Verdana o Arial.
- El código asociado y los estudios preliminares se aportarán como anexos. La extensión de estos anexos no cuenta para el tope de 20 caras. Tampoco cuentan la portada ni el índice de contenidos.
- El trabajo se puede realizar por entero en un notebook, exportándose a formato HTML (caso de Jupyter).
  - Cuidado con no generar listados amplios de datos que no aportan valor.
- Si el trabajo se realiza en Colab, igualmente se debe exportar el resultado a `.html` para su correcta lectura.
  - Se puede adjuntar un link dentro del informe de conclusiones con la URL utilizada de Colab.
- En estas 20 caras se incluye la sección de bibliografía, que no debiera ser muy extensa (media cara) y sin formato estándar específico requerido.

### d. Tecnologías

- Lenguaje de programación: **Python**.
  - Se valorará la legibilidad del código, el uso de comentarios y un correcto formateado.
- Se recomienda el uso de un notebook: **Jupyter**.

### e. Realización de un video

- Se tendrá que realizar un video presentando el trabajo de forma concisa, destacando: enfoque, conclusiones, lecciones aprendidas, etc.
- Duración máxima: **5 minutos**, formato **.MP4**.
- Se espera que el video no ocupe más de **50 MB**.

---

## 3. El alumno puede proponer un trabajo que no encaje en las propuestas anteriores

### a. Objetivos

- El objetivo del trabajo ha de ser comentado primero con los tutores para su discusión/aprobación.
- El trabajo ha de estar relacionado con alguno de los temas impartidos en el curso, con una orientación de corte técnico.
- Debe implicar el desarrollo de una solución software y encuadrarse en el ámbito de la analítica avanzada.

### b. Extensión

- La extensión total no debe superar 20 caras (tamaño folio), con las mismas consideraciones del punto 1.

### c. Tecnologías

- Cualquiera de las impartidas en el Máster.

### d. Realización de un video

- Mismas condiciones que en las opciones anteriores: máximo 5 minutos, formato .MP4, tamaño recomendado no mayor a 50 MB.

---

## Notas generales

- No se admitirán cambios de tema del TFM a menos de **15 días** de la fecha de entrega.
- El TFM se realizará según los detalles adjuntos (epígrafe "Realización de los trabajos").
- En el nombre del fichero se incluirá el nombre del alumno (Nombre y dos apellidos), separando nombre y apellidos con un guion bajo (`_`).
  - Ejemplo: `Maria_Garcia_Perez_Estudio_pajaros.zip`
- Los tutores a cargo de mentorizar y corregir los trabajos son **Carlos Ortega** y **Santiago Mota**.
  - Los tutores pueden ayudar a sugerir una orientación adecuada, pero se evitará enviar diferentes versiones del trabajo para confirmar el enfoque o nivel de avance.
  - No hay seguimiento periódico del TFM.

## Realización de los trabajos

- Modalidad: **individual**.

### Estructura de entregables

**A) Documento** (20 caras) que contiene:
- El detalle del trabajo expuesto de forma no muy técnica. Incluye tablas resumen, bullets, etc.
- Referencias en el texto a diferentes partes del Anexo con detalles más profundos.
- Formato: PDF, `.docx` (Word) o `.HTML` (Jupyter Notebook exportado).

**B) Video:**
- Duración máxima: 5 minutos.
- Formato: `.mp4`.
- Se puede adjuntar como fichero o como link (YouTube, Vimeo).

**C) Anexo** (opcional incluir):
- El código desarrollado.
- Estudios más detallados (por ejemplo, del EDA o de la ejecución de diferentes modelos).

### Notas

- Si el material no cabe en la plataforma, se sube un documento de texto con la URL a un repositorio (Google Drive, Dropbox, GitHub, etc.), dando permisos de acceso a los tutores.
- Además de Memoria, Video y Anexos (imprescindibles), el alumno puede subir otros materiales que considere relevantes.

---
 
## Checklist
 
- ¿Has visto los derechos de uso de los datos?
  
  R: Sí. Según la Regla 7.A de M5 Forecasting - Accuracy (Kaggle), el uso está permitido para "academic research and education", uso no comercial. No se puede redistribuir ni republicar los datos crudos a terceros ajenos a la competencia (por ejemplo, subir el CSV completo a un repo público de GitHub sin restricción).**Fuente**: *kaggle.com/c/m5-forecasting-accuracy/rules*
- ¿Tienes el código compartido en un GitHub o en un Drive/Dropbox?
  R: 
  - ¿Es accesible desde el link?
    R: 
  - ¿Santiago Mota y Carlos Ortega tienen permisos de acceso?
    R: 
- ¿La memoria ocupa 20 hojas?
  R: 
- ¿Tienes el código en los Anexos?
  R: 
- ¿El proyecto es reproducible?
  R: 
- ¿Has incluido un apartado de conclusiones?
  R: 
- ¿Has incluido el vídeo?
  R: 
  - ¿Es de 5 minutos?
    R: 
  - ¿Describe tu proyecto (no es un elevator pitch)?
    R: 
- ¿Has incluido una breve bibliografía/referencias (media cara)?
  R: 
---

## Preguntas frecuentes

**¿Dónde subo el video si no tengo espacio en la plataforma?**
Puedes subir un fichero de texto con el link a un repositorio (Google Drive o similar) donde incluyas tu trabajo, el video, etc.

**¿Si mi conjunto de datos es de 1000-2000 filas, es suficiente?**
Se valora el uso de conjuntos grandes, ya que suponen retos de procesamiento próximos a entornos empresariales. Un conjunto limitado no impide hacer el TFM, pero se valorará menos. Con 300-400 filas es muy complicado que el TFM no difiera de un trabajo de fin de módulo.

**¿Puedo hacerlo en inglés?**
Sí, tanto la memoria como el video se pueden hacer en inglés.

**¿Puedo presentarlo en PowerPoint?**
No. El TFM ha de presentarse como memoria técnica, con redacción equivalente a un informe. Se sugiere hacer un PPT únicamente como apoyo para el video.

**¿Puedo presentar el trabajo como un artículo científico (paper)?**
No, salvo acuerdo previo con los gestores del Máster. La estructura de un paper no se adecúa al enfoque de solución empresarial esperado en el TFM.

**¿Tengo que aparecer en el video?**
No es necesario, pero sí que aparezca tu voz en off explicando objetivos, conclusiones y retos del TFM.

**¿Se cuenta la portada en la extensión?**
No, ni la contraportada ni el índice de contenidos.

**¿Tengo que poner bibliografía?**
Sí, de forma escueta (no más de media página), y cuenta dentro de las 20 hojas límite.

**¿En un HTML cómo veo que sean 20 páginas?**
Puedes exportar el HTML a PDF y contar las páginas, o contar el número de pantallas consecutivas que ocupa (sobre un monitor de 13-14 pulgadas).

**Mi TFM es de un dataset de Kaggle con mucho código ya desarrollado por otros, ¿cómo se califica?**
Se sugiere cambiar de conjunto de datos, ya que Kaggle contiene datasets muy orientados a la educación/práctica, y esto no diferiría de un trabajo de fin de módulo.

**La empresa solo me pide un cuadro de mando, ¿es suficiente?**
No. Se sugiere ofrecer a la empresa llegar a presentar un modelo relacionado con el caso propuesto. No incluir aspectos de modelización u otros desarrollados durante el Máster (por ejemplo, productivización) hace que el TFM sea limitado.

**¿Se puede mantener una reunión/call con los tutores para resolver dudas?**
Se prefiere que las dudas se trasladen por escrito (foro de la plataforma o correo con copia a los gestores), ya que ayuda a precisar mejor lo que se necesita. Las dudas no pueden ser sobre aspectos técnicos particulares del enfoque, salvo bloqueo real, ni sobre errores de instalación/ejecución (parte del día a día que el alumno debe resolver de forma autónoma).

**¿Se puede disponer de TFMs pasados para ver la estructura seguida?**
No se proporcionarán TFMs de referencia; cada TFM tiene su propio enfoque y lo válido para uno puede no serlo para otro.