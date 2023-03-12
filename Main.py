import cv2
from keras.models import load_model

# Cargar modelo entrenado de la CNN
# Buscar imagenes o entrenar modelo usado en IA para reconocer prendas de vestir
model = load_model('modelo.h5')

# Definir etiquetas de clases
CLASES = {0: 'camisa', 1: 'pantalones', 2: 'zapatos'}

# Inicializar cámara web
cap = cv2.VideoCapture(0)

while True:
    # Capturar imagen de la cámara web
    ret, img = cap.read()
    
    # Preprocesar imagen
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    img = img.astype('float32') / 255
    
    # Realizar predicción utilizando el modelo de la CNN
    pred = model.predict(img)
    clase_pred = CLASES[pred.argmax()]
    
    # Mostrar imagen y etiqueta de clase predicha en ventana
    cv2.imshow('Prendas', img)
    cv2.putText(img, clase_pred, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
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