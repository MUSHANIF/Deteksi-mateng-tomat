import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Preprocessing Dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# Arsitektur CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 kelas: mentah, setengah matang, matang
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Pelatihan Model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Simpan Model
model.save('model/tomato_model.h5')
print("Model saved to 'model/tomato_model.h5'")
