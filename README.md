# Phenophoto
Estimación de biomasa mediante fotos oblicuas
# Autores
Martín Durante durante.martin@inta.gob.ar
Martín Jaurena

# Modo de uso
- El script procesarFotos.py lee una foto tomada con el panel blanco, la procesa y estima algunas variables cuantitativas.
- El procesamiento consiste en 1) recortar el panel; para eso se basa en los marcadores aruco y la imagen panelAru.jpg, que se usa para corregir la perspectiva mediante homografìa y 2) separar el área ocupada por vegetación del fondo blanco mediante un umbral
