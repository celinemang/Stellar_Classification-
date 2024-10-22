from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('Skyserver_data.csv')

# 존재하는 열을 바탕으로 features와 labels를 정의합니다.
features = data[['u', 'g', 'r', 'i', 'z']]  # 실제로 사용 가능한 열
labels = data['class']  # 'class' 열은 천체의 분류에 해당하는 듯함

# 데이터 스케일링
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# 데이터셋을 훈련셋과 테스트셋으로 나누기
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# KNN 모델 초기화 및 훈련
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 예측 및 정확도 평가
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

from sklearn.model_selection import GridSearchCV

# KNN 하이퍼파라미터 튜닝
param_grid = {'n_neighbors': range(1, 10)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 최적의 파라미터 확인
print(f'Best Parameters: {grid_search.best_params_}')

