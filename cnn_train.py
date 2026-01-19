import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from cnn_model import create_cnn_model

# 경로 및 카테고리 설정
image_folder = r"C:/Users/HyejinPark/Desktop/new/heatmap1"
categories = ['losdown', 'losup', 'loswalk', 'losN']

images = []
labels = []

max_per_category = 3000
count_per_category = {category: 0 for category in categories}

# 이미지 로딩 (폴더 내 모든 이미지 사용)
for category_idx, category in enumerate(categories):
    files = [f for f in os.listdir(image_folder) if f.startswith(category) and f.endswith('.png')]
    for filename in files:
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(category_idx)
            count_per_category[category] += 1
        else:
            print(f"[경고] 이미지 읽기 실패: {img_path}")

images = np.array(images, dtype=np.float32) / 255.0
labels = np.array(labels)

print(f"총 이미지 수: {len(images)}")
print(f"총 레이블 수: {len(labels)}")
for cat in categories:
    print(f"{cat} 개수: {count_per_category[cat]}")

# 데이터 분할 (훈련:검증 = 8:2)
X_train, X_test, Y_train, Y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# 원-핫 인코딩
Y_train = to_categorical(Y_train, num_classes=len(categories))
Y_test = to_categorical(Y_test, num_classes=len(categories))

# 모델 생성
model = create_cnn_model(input_shape=(224, 224, 3), num_classes=len(categories))

# 학습
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=15,
    batch_size=8,
    verbose=1
)

# 평가
val_loss, val_acc = model.evaluate(X_test, Y_test, verbose=1)

print(f"\n최종 Training 정확도: {history.history['accuracy'][-1]*100:.2f}%")
print(f"최종 Validation 정확도: {val_acc*100:.2f}%")

# 정확도/손실 시각화
epochs_range = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('C:/Users/HyejinPark/Desktop/model1.png', dpi=300)
plt.show()

# 모델 저장
model.save('C:/Users/HyejinPark/Desktop/trained10 - 복사본.keras')
