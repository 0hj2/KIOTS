######################################
####### 전체 검증용 예측 코드 ########
######################################

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------- 설정 -----------------
model_path   = r"C:/Users/HyejinPark/Desktop/trained10.keras"
image_folder = r"C:/Users/HyejinPark/Desktop/new/Heatmaps"
class_labels = ['losdown', 'losup', 'loswalk', 'losN']

IMG_SIZE     = 224
BATCH_SIZE   = 8

# ----------------- 모델 로드 -----------------
model = load_model(model_path)
print("✅ 모델 로드 완료")

# ----------------- 배치 예측용 전처리 함수 -----------------
def load_and_prepare(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img

# ----------------- 전체 이미지 수집 (클래스별 최대 5000장) -----------------
all_images = []
all_labels = []

MAX_PER_CLASS = 5500

for class_idx, class_name in enumerate(class_labels):
    files = [f for f in os.listdir(image_folder) if f.startswith(class_name) and f.endswith('.png')]
    
    # 클래스별 최대 MAX_PER_CLASS 장 선택
    selected_files = files[:MAX_PER_CLASS]  # 순서대로 가져오기
    
    for f in selected_files:
        all_images.append(os.path.join(image_folder, f))
        all_labels.append(class_idx)


all_labels = np.array(all_labels, dtype=np.int64)
print(f"총 검증 이미지 수: {len(all_images)}")

# ----------------- 배치 단위 예측 -----------------
predicted_classes = []

for s in range(0, len(all_images), BATCH_SIZE):
    batch_paths = all_images[s:s + BATCH_SIZE]
    batch_imgs = []

    for path in batch_paths:
        img = load_and_prepare(path)
        if img is not None:
            batch_imgs.append(img)
        else:
            print(f"[경고] 이미지 로드 실패: {path}")

    if not batch_imgs:
        continue

    batch_imgs = np.stack(batch_imgs, axis=0)
    probs = model.predict(batch_imgs, verbose=0)
    batch_pred = np.argmax(probs, axis=1)
    predicted_classes.extend(batch_pred)

predicted_classes = np.array(predicted_classes)

# ----------------- 결과 출력 -----------------
for i, pred_idx in enumerate(predicted_classes):
    actual = class_labels[all_labels[i]]
    predicted = class_labels[pred_idx]
    print(f"이미지 {i+1}: 실제 = {actual}, 예측 = {predicted}")

# ----------------- 전체 정확도 -----------------
accuracy = np.sum(predicted_classes == all_labels[:len(predicted_classes)]) / len(predicted_classes)
print(f"\n총 검증 정확도: {accuracy*100:.2f}%")

# # ----------------- 클래스별 예측 개수 -----------------
# pred_counts = {label: 0 for label in class_labels}
# for idx in predicted_classes:
#     pred_counts[class_labels[idx]] += 1

# print("\n📊 클래스별 예측 개수:")
# for label in class_labels:
#     print(f"{label}: {pred_counts[label]}개")

# ----------------- 혼동 행렬 시각화 -----------------
cm = confusion_matrix(all_labels[:len(predicted_classes)], predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

#plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
