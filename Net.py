from tensorflow import keras
import Preprocess

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(128, 293)),
#     keras.layers.Dense(128*2, activation='relu'),
#     keras.layers.Dense(3, activation='softmax')
# ])

model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), input_shape=(128, 293, 1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
    ])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(Preprocess.train_input, Preprocess.train_labels, epochs=5)

results = model.evaluate(Preprocess.test_input, Preprocess.test_labels)
print('test loss, test acc:', results)

model.save("model.h5")
print("Saved model to disk")
