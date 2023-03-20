import cv2
from keras.models import load_model

# Cargar modelo entrenado de la CNN
# Buscar imagenes o entrenar modelo usado en IA para reconocer prendas de vestir
model = load_model('modelo.h5')

# Definir etiquetas de clases
CLASES = {0: 'Camisa', 1: 'Pantalones', 2: 'Zapatos'}

# Inicializar cámara web
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capturar imagen de la cámara web
    ret, img = cap.read()
    
    # Preprocesar imagen
    imgpre = cv2.resize(img, (100, 100))
    imgpre = imgpre.reshape(1, 100, 100, 3)
    imgpre = imgpre.astype('float32') / 255
    
    # Realizar predicción utilizando el modelo de la CNN
    pred = model.predict(imgpre)
    clase_pred = CLASES[pred.argmax()]
    
    # Mostrar imagen y etiqueta de clase predicha en ventana
    cv2.putText(img, clase_pred, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Prendas', img)
    
    # Redirigir a la página web correspondiente según la clase predicha
    if clase_pred == 'camisa':
        # Redirigir a la página de camisas
        pass
    elif clase_pred == 'pantalones':
        # Redirigir a la página de pantalones
        pass
    elif clase_pred == 'zapatos':
        # Redirigir a la página de zapatos
        pass
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()