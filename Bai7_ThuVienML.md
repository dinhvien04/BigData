# Bài 7: Thư viện ML trong PySpark

## Mục lục
1. [Giới thiệu về thư viện ML](#1-giới-thiệu-về-thư-viện-ml)
2. [Các lớp Transformer](#2-các-lớp-transformer)
3. [Các lớp Estimators](#3-các-lớp-estimators)
4. [Lớp LogisticRegression](#4-lớp-logisticregression)
5. [Lớp Pipeline](#5-lớp-pipeline)
6. [Đánh giá mô hình (Evaluating Model)](#6-đánh-giá-mô-hình-evaluating-model)
7. [Lưu mô hình](#7-lưu-mô-hình)
8. [Phân lớp (Classification)](#8-phân-lớp-classification)
9. [Phân cụm (Clustering)](#9-phân-cụm-clustering)
10. [Tăng hiệu quả cho mô hình học máy](#10-tăng-hiệu-quả-cho-mô-hình-học-máy)
11. [Thuật toán cho hệ gợi ý (ALS)](#11-thuật-toán-cho-hệ-gợi-ý-als)
12. [Xử lý dữ liệu văn bản](#12-xử-lý-dữ-liệu-văn-bản)
13. [Tổng kết](#13-tổng-kết)

---

## 1. Giới thiệu về thư viện ML

### 1.1. ML là gì?
- ML là gói thư viện học máy của Spark được hỗ trợ từ phiên bản 2.0
- Khác với MLlib, ML thao tác với dữ liệu là các **DataFrame**
- ML xây dựng mô hình học máy theo kiểu **đường ống (pipeline)**

### 1.2. Ba lớp trừu tượng cơ sở trong ML

| Lớp | Mô tả |
|-----|-------|
| **Transformer** | Dùng cho chuyển đổi dữ liệu |
| **Estimator** | Dùng lựa chọn các thuật toán học máy |
| **Pipeline** | Xây dựng mô hình học máy theo kiểu đường ống |

---

## 2. Các lớp Transformer

### 2.1. Khái niệm
- Là lớp trừu tượng dùng để chuyển đổi dữ liệu
- Phương thức: `transform(dataset[, params])`
- Hai thuộc tính: `inputCols` và `outputCol`
- Các lớp con được định nghĩa trong gói: `pyspark.ml.feature`

### 2.2. Lớp StringIndexer
Dùng để chuyển dữ liệu kiểu String thành số.

```python
from pyspark.ml.feature import StringIndexer

# Tạo dữ liệu mẫu
df = spark.createDataFrame([
    (1, 'Red'), (2, 'Red'), (3, 'Yellow'),
    (4, 'Green'), (5, 'Yellow')
], ['Id', 'Color'])

# Tạo đối tượng StringIndexer
string_indexer = StringIndexer(inputCol="Color", outputCol="Color_index")

# Áp dụng StringIndexer
df_indexed = string_indexer.fit(df).transform(df)
df_indexed.show()
```

**Kết quả:**
```
+---+------+-----------+
| Id| Color|Color_index|
+---+------+-----------+
|  1|   Red|        1.0|
|  2|   Red|        1.0|
|  3|Yellow|        0.0|
|  4| Green|        2.0|
|  5|Yellow|        0.0|
+---+------+-----------+
```

**Lưu ý:** Phương thức `fit()` trả về đối tượng `StringIndexerModel` xác định cách chuyển dữ liệu String sang số.

### 2.3. Lớp OneHotEncoder
Chuyển dữ liệu phân loại (số) thành dữ liệu vector.

```python
from pyspark.ml.feature import OneHotEncoder

oneHot = OneHotEncoder(inputCol='Color_int', outputCol='Color_OHE')
df2 = oneHot.transform(df)
df2.show()

# Xem dạng RDD
df2.collect()
```

### 2.4. Lớp SparseVector
- Dùng để thao tác với vector thưa
- Chỉ có các thành phần khác 0 được lưu trữ dưới dạng danh sách các chỉ số (indices) và giá trị tương ứng

```python
from pyspark.ml.linalg import SparseVector

# Tạo SparseVector: size=5, indices=[1,3], values=[2.0, 3.0]
sparse_vector = SparseVector(5, [1, 3], [2.0, 3.0])
# Kết quả: (5, [1, 3], [2.0, 3.0]) tương đương [0, 2.0, 0, 3.0, 0]
```

### 2.5. Lớp DenseVector
- DenseVector là vector dày đặc, lưu tất cả các tọa độ của vector
- Được sử dụng rộng rãi trong các thuật toán học máy như dữ liệu đầu vào

**Các phương thức chính:**

| Phương thức | Mô tả |
|-------------|-------|
| `toArray()` | Chuyển đổi DenseVector thành mảng Numpy |
| `size` | Trả về số chiều của DenseVector |
| `values` | Trả về danh sách các giá trị |
| `dot(other)` | Tính tích vô hướng với vector khác |
| `squared_distance(other)` | Tính khoảng cách bình phương |
| `norm(p)` | Tính chuẩn p của DenseVector |

```python
from pyspark.ml.linalg import DenseVector

# Tạo DenseVector với 3 chiều
dense_vec = DenseVector([1.0, 2.0, 3.0])

# Truy cập các giá trị
print(dense_vec.values)  # [1.0, 2.0, 3.0]

# Tính tích vô hướng
dot_product = dense_vec.dot(DenseVector([4.0, 5.0, 6.0]))
print(dot_product)  # 32.0

# Tính chuẩn
norm = dense_vec.norm(2.0)
print(norm)  # 3.7416573867739413
```

### 2.6. Lớp VectorAssembler
Dùng ghép nhiều cột dữ liệu thành một dữ liệu vector.

```python
from pyspark.ml.feature import VectorAssembler

df = spark.createDataFrame([(12, 10, 3), (1, 4, 2)], ['a', 'b', 'c'])

df2 = VectorAssembler(
    inputCols=['a', 'b', 'c'],
    outputCol='features'
).transform(df)

df2.show()
df2.select('features').collect()
```

**Kết quả:**
```python
[Row(features=DenseVector([12.0, 10.0, 3.0])),
 Row(features=DenseVector([1.0, 4.0, 2.0]))]
```

### 2.7. Lớp StandardScaler
Chuẩn hóa vector mà mỗi thành phần riêng biệt có trung bình bằng 0 và độ lệch chuẩn bằng 1.

**Công thức chuẩn hóa:**
```
x'_i = (x_i - μ_i) / σ_i
```
- `x_i`: giá trị của thành phần thứ i trong vector
- `μ_i`: giá trị trung bình của thành phần i trong dữ liệu
- `σ_i`: độ lệch chuẩn của thành phần i trong dữ liệu

```python
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors

data = [
    (0, Vectors.dense([1.0, 0.1, -1.0]),),
    (1, Vectors.dense([2.0, 1.1, 1.0]),),
    (2, Vectors.dense([3.0, 10.1, 3.0]),)
]
df = spark.createDataFrame(data, ["id", "features"])

# Khởi tạo StandardScaler
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)

# Áp dụng StandardScaler
scaler_model = scaler.fit(df)
scaled_df = scaler_model.transform(df)

# Hiển thị kết quả
scaled_df.select("id", "features", "scaled_features").show(truncate=False)
```

---

## 3. Các lớp Estimators

### 3.1. Khái niệm
- Dùng để ước lượng (tính toán) cho mô hình học máy bằng các thuật toán được chọn
- Phương thức: `fit(df)` thực hiện tính toán xây dựng mô hình từ thuật toán và dữ liệu

### 3.2. Các lớp con của Estimator

#### Classification (Phân lớp)
| Lớp | Mô tả |
|-----|-------|
| LogisticRegression | Hồi quy logistic |
| DecisionTreeClassifier | Cây quyết định |
| GBTClassifier | Gradient Boosted Trees |
| RandomForestClassifier | Rừng ngẫu nhiên |
| NaiveBayes | Naive Bayes |
| MultilayerPerceptronClassifier | Mạng nơ-ron nhiều lớp |
| OneVsRest | One vs Rest |

#### Regression (Hồi quy)
| Lớp | Mô tả |
|-----|-------|
| AFTSurvivalRegression | Hồi quy sinh tồn |
| DecisionTreeRegressor | Cây quyết định hồi quy |
| GBTRegressor | Gradient Boosted Trees hồi quy |
| GeneralizedLinearRegression | Hồi quy tuyến tính tổng quát |
| IsotonicRegression | Hồi quy đơn điệu |
| LinearRegression | Hồi quy tuyến tính |
| RandomForestRegressor | Rừng ngẫu nhiên hồi quy |

#### Clustering (Phân cụm)
| Lớp | Mô tả |
|-----|-------|
| BisectingKMeans | K-Means phân cấp |
| KMeans | K-Means |
| GaussianMixture | Gaussian Mixture Model |

---

## 4. Lớp LogisticRegression

### 4.1. Hồi quy Logistic là gì?
- Là thuật toán thường dùng phân lớp nhị phân
- Dựa vào hồi quy tuyến tính, sử dụng hàm sigmoid

### 4.2. Lớp LogisticRegression trong PySpark
- Triển khai thuật toán học máy hồi quy logistic
- Được cung cấp trong module: `pyspark.ml.classification`
- Yêu cầu dữ liệu đầu vào: DataFrame có 2 cột đặc trưng (features) và nhãn (label)

### 4.3. Các tham số của LogisticRegression

| Tham số | Mô tả |
|---------|-------|
| `featuresCol` | Tên của cột chứa đặc trưng đầu vào |
| `labelCol` | Tên của cột chứa nhãn đầu ra |
| `maxIter` | Số lần lặp tối đa trong quá trình tối ưu hóa |
| `regParam` | Tham số regularization để hạn chế overfitting |
| `elasticNetParam` | Tham số elastic net (0 đến 1) |
| `family` | Loại hồi quy ("binomial" hoặc "multinomial") |
| `threshold` | Ngưỡng xác suất để dự đoán lớp |
| `weightColumn` | Tên cột trọng số (dùng khi dữ liệu mất cân bằng) |

### 4.4. Ví dụ hoàn chỉnh

```python
# Đọc dữ liệu vào DataFrame
fraud_df = spark.read.csv("ccFraud.csv.gz", header=True, inferSchema=True)

from pyspark.ml.feature import VectorAssembler

# Tạo cột đặc trưng từ các cột gender, balance và numTrans
assembler = VectorAssembler(
    inputCols=["gender", "balance", "numTrans"],
    outputCol="features"
)
data = assembler.transform(fraud_df)
data = data.select("gender", "balance", "numTrans", "fraudRisk", "features")
data.show()

from pyspark.ml.classification import LogisticRegression

# Xây dựng mô hình
lr = LogisticRegression(featuresCol="features", labelCol="fraudRisk")

# Chia dữ liệu train và test
data_train, data_test = data.randomSplit([0.7, 0.3], seed=666)

# Huấn luyện mô hình bằng dữ liệu train
model = lr.fit(data_train)

# Dự đoán cho dữ liệu test
test_model = model.transform(data_test)
test_model.show()
```

### 4.5. Kết quả dự đoán
```python
test_model.take(2)
```

**Kết quả:**
```python
[Row(gender=1, balance=0, numTrans=0, fraudRisk=0, 
     features=DenseVector([1.0, 0.0, 0.0]),
     rawPrediction=DenseVector([7.2795, -7.2795]), 
     probability=DenseVector([0.9993, 0.0007]),
     prediction=0.0),
 Row(gender=1, balance=8503, numTrans=100, fraudRisk=1, 
     features=DenseVector([1.0, 8503.0, 100.0]),
     rawPrediction=DenseVector([-0.0003, 0.0003]),
     probability=DenseVector([0.4999, 0.5001]), 
     prediction=1.0)]
```

### 4.6. Xử lý mất cân bằng dữ liệu
```python
from pyspark.sql.functions import when

# Tạo trọng số cho lớp 1 nhiều hơn lớp 0
data = data.withColumn("weight", when(data["Exited"] == 1, 1.5).otherwise(1))

# Huấn luyện mô hình với trọng số
lr = LogisticRegression(
    labelCol="Exited",
    featuresCol="features",
    weightCol="weight"
)
```

---

## 5. Lớp Pipeline

### 5.1. Khái niệm
- Pipeline là cách tổ chức các bước quá trình biến đổi dữ liệu và ước lượng mô hình học máy
- Pipeline có thể chỉ gồm các bước biến đổi dữ liệu
- Đầu ra của bước này là đầu vào của bước kế tiếp

### 5.2. Phương thức fit()
- Thực hiện các bước trong đường ống theo thứ tự đã chỉ định
- Với đối tượng Transformer: thực hiện `transform()`
- Với đối tượng Estimators: thực hiện phương thức `fit()`

### 5.3. Sơ đồ hoạt động
```
DataFrame → Transformer 1 → Transformer 2 → ... → Estimator → Model
```

### 5.4. Ví dụ sử dụng Pipeline
```python
from pyspark.ml import Pipeline

# Tạo pipeline với các stages
pipeline = Pipeline(stages=[assembler, lr])

# Chia dữ liệu
data_train2, data_test2 = fraud.randomSplit([0.7, 0.3], seed=666)

# Huấn luyện pipeline
model2 = pipeline.fit(data_train2)

# Dự đoán
test_model2 = model2.transform(data_test2)
```

---

## 6. Đánh giá mô hình (Evaluating Model)

### 6.1. Các độ đo đánh giá

#### Accuracy
- Tỷ lệ giữa số lượng dự đoán đúng và tổng số lượng mẫu

#### Precision
- Tỷ lệ số lượng dự đoán đúng Positive và tổng số lượng Positive được dự đoán

#### Recall
- Tỷ lệ số lượng dự đoán đúng Positive và tổng số lượng Positive trong tập dữ liệu

#### F1-score
```
F1 = 2 * (precision * recall) / (precision + recall)
```

#### ROC curve và AUC
- **ROC curve**: Đường cong biểu diễn tỷ lệ giữa True Positive Rate và False Positive Rate
- **AUC** (Area Under Curve): Diện tích phía dưới đường cong ROC
- **TPR** (True Positive Rate/Sensitivity/Recall): Tỷ lệ phân loại chính xác các mẫu Positive
- **FPR** (False Positive Rate/Fall-out): Tỷ lệ gắn nhãn sai các mẫu Negative thành Positive

### 6.2. Các lớp đánh giá mô hình

| Lớp | Mô tả |
|-----|-------|
| `RegressionEvaluator` | Đánh giá mô hình hồi quy (RMSE, MSE, R2) |
| `BinaryClassificationEvaluator` | Đánh giá phân loại nhị phân (AUC, Area Under PR) |
| `MulticlassClassificationEvaluator` | Đánh giá phân loại đa lớp (accuracy, precision, recall) |
| `ClusteringEvaluator` | Đánh giá phân cụm (silhouette score, SSE) |

### 6.3. Lớp BinaryClassificationEvaluator

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="label",
    metricName="areaUnderROC"  # hoặc "areaUnderPR"
)

test_model = model.transform(data_test)
auc = evaluator.evaluate(test_model)
print("Area under ROC = %g" % auc)
```

---

## 7. Lưu mô hình

### 7.1. Lưu Pipeline và các thành phần
```python
pipelinePath = '...'
pipeline.write().overwrite().save(pipelinePath)
```

### 7.2. Nạp mô hình chưa huấn luyện
```python
loadedPipeline = Pipeline.load(pipelinePath)
loadedPipeline.fit(data_train).transform(data_test).take(1)
```

### 7.3. Lớp PipelineModel - Nạp mô hình đã huấn luyện
```python
from pyspark.ml import PipelineModel

loadedPipelineModel = PipelineModel.load(modelPath)
test_reloadedModel = loadedPipelineModel.transform(data_test)
```

---

## 8. Phân lớp (Classification)

### 8.1. Các thuật toán phân lớp trong Spark

| Thuật toán | Mô tả |
|------------|-------|
| Logistic Regression | Phân loại nhị phân, sử dụng hàm sigmoid |
| Decision Tree Classifier | Phân loại bằng cây quyết định |
| Random Forest Classifier | Sử dụng nhiều cây quyết định (ensemble) |
| Gradient-Boosted Trees | Sử dụng boosting để cải thiện độ chính xác |
| Naive Bayes | Phân loại dựa trên lý thuyết xác suất Bayes |
| Multilayer Perceptron (MLP) | Sử dụng mạng nơ-ron nhiều lớp |
| Linear SVC | Phân loại nhị phân với siêu phẳng tuyến tính |

### 8.2. Lớp DecisionTreeClassifier

#### Khái niệm
- Cài đặt thuật toán Decision Tree
- Dùng để phân lớp dữ liệu dựa trên dữ liệu đã được gán nhãn
- Thực hiện phân chia các thuộc tính thành các đoạn dữ liệu (bins)

#### Các tham số của DecisionTreeClassifier

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `labelCol` | Tên cột chứa nhãn | "label" |
| `featuresCol` | Tên cột chứa vector đặc trưng | "features" |
| `maxDepth` | Độ sâu tối đa của cây | 5 |
| `maxBins` | Số lượng tối đa các bin | 32 |
| `minInstancesPerNode` | Số lượng mẫu tối thiểu trong mỗi nút | 1 |
| `minInfoGain` | Giá trị thông tin tối thiểu để phân chia nút | 0.0 |
| `maxMemoryInMB` | Bộ nhớ tối đa để xây dựng cây | 256 |
| `cacheNodeIds` | Lưu trữ ID của các nút trong cache | False |
| `checkpointInterval` | Số cấp độ cây để lưu checkpoint | 10 |

#### Ví dụ với dataset Iris
```python
from sklearn.datasets import load_iris
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# Load dữ liệu Iris
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['label'] = pd.Series(iris.target)

# Tạo Spark DataFrame
data = spark.createDataFrame(df_iris)

# Tạo vector đặc trưng
features = iris.feature_names
va = VectorAssembler(inputCols=features, outputCol='features')
va_df = va.transform(data)
va_df = va_df.select(['features', 'label'])
va_df.show(3)

# Chia dữ liệu
(train, test) = va_df.randomSplit([0.8, 0.2])

# Huấn luyện mô hình
dtc = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dtc = dtc.fit(train)

# Dự đoán
pred = dtc.transform(test)
pred.show(3)
```

### 8.3. Đánh giá phân lớp - MulticlassClassificationEvaluator

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
acc = evaluator.evaluate(pred)
print("Prediction Accuracy: ", acc)
# Kết quả: Prediction Accuracy: 0.8883803712178999
```

#### Các tham số của MulticlassClassificationEvaluator

| Tham số | Mô tả |
|---------|-------|
| `labelCol` | Tên cột chứa nhãn thực tế |
| `predictionCol` | Tên cột chứa nhãn dự đoán |
| `metricName` | Độ đo: f1, accuracy, weightedPrecision, weightedRecall, ... |

#### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix

y_pred = pred.select("prediction").collect()
y_orig = pred.select("label").collect()
cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)
```

### 8.4. Lớp RandomForestClassifier

#### Khái niệm
- Dùng phân lớp dữ liệu được học từ dữ liệu đã gán nhãn
- Sử dụng kỹ thuật cây quyết định bằng cách tạo nhiều cây (rừng)
- Kết quả dự đoán cuối cùng được tính bằng phương pháp bầu cử đa số

#### Các tham số của RandomForestClassifier

| Tham số | Mô tả |
|---------|-------|
| `numTrees` | Số lượng cây trong rừng ngẫu nhiên |
| `maxDepth` | Độ sâu tối đa của cây quyết định |
| `maxBins` | Số lượng bin tối đa để phân tách các giá trị số |
| `impurity` | Hàm đo lường sự tinh khiết: gini/entropy |
| `featureSubsetStrategy` | Chiến lược chọn tập con thuộc tính: auto/all/sqrt/log2/onethird/n |
| `seed` | Giá trị khởi tạo cho bộ sinh số ngẫu nhiên |

#### Ví dụ
```python
from pyspark.ml.classification import RandomForestClassifier

rdc = RandomForestClassifier(featuresCol="features", labelCol="label")
rdc = rdc.fit(train)
pred_rdc = rdc.transform(test)
pred_rdc.show(3)

acc2 = evaluator.evaluate(pred_rdc)
print("Prediction Accuracy: ", acc2)
# Kết quả: Prediction Accuracy: 0.9627934570726333
```

---

## 9. Phân cụm (Clustering)

### 9.1. Khái niệm
- Clustering là kỹ thuật dùng để phân cụm dữ liệu dựa vào đặc trưng của dữ liệu
- Clustering dùng để khám phá dữ liệu, còn gọi là kỹ thuật học không giám sát
- ML cung cấp các lớp phân cụm trong gói `pyspark.ml.clustering`

### 9.2. Các thuật toán phân cụm trong Spark

| Thuật toán | Mô tả |
|------------|-------|
| KMeans | Phân nhóm dữ liệu thành k cụm dựa trên khoảng cách |
| Gaussian Mixture Model (GMM) | Sử dụng các phân phối Gaussian |
| Bisecting KMeans | Biến thể của KMeans, phân cụm phân cấp |
| Latent Dirichlet Allocation (LDA) | Phân nhóm văn bản thành các chủ đề |
| Power Iteration Clustering (PIC) | Phân nhóm các điểm trong đồ thị |

### 9.3. Lớp KMeans

#### Các tham số chính

| Tham số | Mô tả |
|---------|-------|
| `k` | Số cụm mong muốn |
| `maxIter` | Số lần chạy tối đa của thuật toán |
| `tol` | Ngưỡng dung sai trong quá trình huấn luyện |
| `initMode` | Phương thức khởi tạo điểm trung tâm: "random" hoặc "k-means" |
| `initSteps` | Số lần chạy thuật toán k-means để tạo điểm trung tâm ban đầu |
| `distanceMeasure` | Phương thức tính khoảng cách: "euclidean" hoặc "cosine" |
| `seed` | Seed để tạo điểm trung tâm ban đầu |

#### Ví dụ phân cụm trên dữ liệu Iris
```python
from pyspark.ml.clustering import KMeans

km = KMeans(featuresCol='features', k=3)
km_model = km.fit(va_df)
km_clus = km_model.transform(va_df)
km_clus.show()
```

### 9.4. Đánh giá phân cụm - ClusteringEvaluator

#### Silhouette Score
- Đo lường mức độ "gắn kết" trong mỗi cụm và "phân tán" giữa các cụm khác nhau
- Công thức: `s(i) = (b(i) − a(i)) / max(a(i), b(i))`

```python
from pyspark.ml.evaluation import ClusteringEvaluator

evaluator_clus = ClusteringEvaluator(
    featuresCol='features',
    metricName='silhouette',
    distanceMeasure='squaredEuclidean'
)
evaluation_score = evaluator_clus.evaluate(km_clus)
print("Silhouette with squared euclidean distance = ", evaluation_score)
# Kết quả: Silhouette with squared euclidean distance = 0.7344130579787832
```

### 9.5. Xác định số cụm tối ưu

#### Phương pháp Elbow
- Dựa trên việc xác định số cụm sao cho tổng khoảng cách từ các điểm đến trung tâm cụm nhỏ nhất
- Tìm "điểm gấp khúc" (elbow) trên đồ thị

```python
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

cost = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(data)
    cost.append(model.summary.trainingCost)  # WSS

# Vẽ biểu đồ Elbow
plt.plot(k_values, cost, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.title('Elbow Method to determine optimal k')
plt.show()
```

#### Phương pháp Silhouette Score
- Giá trị Silhouette gần với 1 hơn cho thấy kết quả phân cụm tốt hơn

```python
from pyspark.ml.evaluation import ClusteringEvaluator

silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(k=k, seed=1)
    model = kmeans.fit(data)
    predictions = model.transform(data)
    
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette)

# Vẽ biểu đồ Silhouette
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score to determine optimal k')
plt.show()
```

---

## 10. Tăng hiệu quả cho mô hình học máy

### 10.1. Các yếu tố ảnh hưởng đến hiệu quả mô hình
- Dữ liệu vào
- Lựa chọn các đặc trưng nào?
- Giảm chiều dữ liệu
- Lựa chọn mô hình phù hợp với bài toán
- Điều chỉnh các tham số của mô hình

### 10.2. Features Engineering
Là quá trình chuyển đổi, tạo ra và lựa chọn các đặc trưng từ dữ liệu đầu vào để cải thiện hiệu suất của mô hình.

**Các kỹ thuật Features Engineering:**
- Xử lý dữ liệu bị thiếu
- Xử lý dữ liệu nhiễu (loại bỏ outliers)
- Chuyển đổi dữ liệu
- Tạo ra các đặc trưng mới
- Scaling dữ liệu

### 10.3. Lớp ChiSqSelector
Dùng để chọn các đặc trưng quan trọng sử dụng kiểm định Chi-bình phương.

#### Các tham số

| Tham số | Mô tả |
|---------|-------|
| `numTopFeatures` | Số lượng đặc trưng quan trọng tối đa được chọn |
| `featuresCol` | Tên cột chứa vector đặc trưng |
| `outputCol` | Tên cột chứa kết quả lựa chọn đặc trưng |
| `labelCol` | Tên cột chứa biến đầu ra |
| `selectorType` | Loại chọn đặc trưng: "numTopFeatures" hoặc "percentile" |

#### Ví dụ
```python
from pyspark.ml.feature import VectorAssembler, ChiSqSelector

# Đọc dữ liệu
fraud_df = spark.read.csv("ccFraud.csv.gz", header=True, inferSchema=True)

# Tạo cột đặc trưng
assembler = VectorAssembler(
    inputCols=["gender", "state", "cardholder", "numTrans", "numIntlTrans", "creditLine"],
    outputCol="features"
)
data_assembler = assembler.transform(fraud_df)

# Trích chọn đặc trưng bằng ChiSqSelector
selector = ChiSqSelector(
    numTopFeatures=3,
    featuresCol="features",
    outputCol="selectedFeatures",
    labelCol="fraudRisk"
)
model = selector.fit(data_assembler)
print(model.selectedFeatures)  # [0, 1, 2]

data = model.transform(data_assembler)
```

### 10.4. PCA (Principal Component Analysis)
Phương pháp giảm chiều dữ liệu bằng cách chuyển dữ liệu về các thành phần chính.

#### Các tham số của PCA

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `k` | Số lượng thành phần chính được chọn | - |
| `inputCol` | Tên cột đầu vào | - |
| `outputCol` | Tên cột đầu ra | - |
| `meanCenter` | Điều chỉnh trung bình thành 0 | True |
| `std` | Chuẩn hóa đơn vị | True |
| `tol` | Độ chính xác mong muốn | 1e-06 |
| `maxIter` | Số vòng lặp tối đa | 100 |

#### Ví dụ
```python
from pyspark.ml.feature import PCA

pca = PCA(k=2, inputCol="features", outputCol="pca_features")
model_pca = pca.fit(data_assembler)
print(model_pca.explainedVariance)  # [0.6431, 0.2044]

data_pca = model_pca.transform(data_assembler)
data_pca.show()
```

### 10.5. Parameter Hyper-tuning

#### Grid Search
- Duyệt qua các giá trị tham số đã xác định trước
- Tính toán mô hình với các tham số để chọn tham số tốt nhất

#### Lớp ParamGridBuilder
```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder

# Mô hình dự đoán gian lận tín dụng
logistic = LogisticRegression(featuresCol="features", labelCol="fraudRisk")

# Tạo Grid search
param_grid = ParamGridBuilder()
param_grid.addGrid(logistic.maxIter, [2, 10, 50])
param_grid.addGrid(logistic.regParam, [0.01, 0.05, 0.3])
grid = param_grid.build()
```

### 10.6. Lớp CrossValidator
Dùng để chọn bộ tham số tối ưu bằng phương pháp k-fold cross-validation.

#### Các tham số

| Tham số | Mô tả |
|---------|-------|
| `estimator` | Mô hình cần đánh giá |
| `estimatorParamMaps` | Lưới tham số cần đánh giá |
| `evaluator` | Đối tượng đánh giá mô hình |
| `numFolds` | Số fold cần chia |

#### Ví dụ
```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = BinaryClassificationEvaluator(
    rawPredictionCol='probability',
    labelCol='fraudRisk'
)

crossval = CrossValidator(
    estimator=logistic,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    numFolds=5
)

# Chia dữ liệu train và test
data_train, data_test = data_assembler.randomSplit([0.7, 0.3], seed=666)

# Chọn mô hình tối ưu
cvModel = crossval.fit(data_train)

# Các tham số mô hình tối ưu
print(cvModel.bestModel)
print(cvModel.bestModel.coefficients)
print(cvModel.bestModel.intercept)
```

#### Đánh giá mô hình
```python
test_model = cvModel.transform(data_test)
print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderPR'}))
```

#### Lấy tham số mô hình tối ưu
```python
results = [
    ([{key.name: paramValue} for key, paramValue in zip(params.keys(), params.values())], metric)
    for params, metric in zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics)
]
sorted(results, key=lambda el: el[1], reverse=True)[0]
# Kết quả: ([{'maxIter': 50}, {'regParam': 0.01}], 0.8610476834166823)
```

### 10.7. Lớp TrainValidationSplit
Thực hiện chọn mô hình tốt nhất bằng cách chia ngẫu nhiên tập dữ liệu thành 2 phần: huấn luyện và kiểm tra.

#### Các tham số

| Tham số | Mô tả |
|---------|-------|
| `estimator` | Mô hình cần chọn tham số |
| `evaluator` | Đối tượng evaluator để đánh giá |
| `estimatorParamMaps` | Các bộ tham số để lựa chọn |
| `trainRatio` | Tỷ lệ phần trăm dữ liệu để huấn luyện |

#### Ví dụ
```python
from pyspark.ml.tuning import TrainValidationSplit

tvs = TrainValidationSplit(
    estimator=logistic,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    trainRatio=0.8,
    seed=123
)

# Huấn luyện và đánh giá mô hình
model = tvs.fit(data_train)

# Đánh giá kết quả trên tập test
test_model = model.transform(data_test)
auc = evaluator.evaluate(test_model)
print("AUC on test data = %g" % auc)

# Lấy mô hình tốt nhất
bestModel = model.bestModel
regParam = bestModel.getOrDefault("regParam")
maxIter = bestModel.getOrDefault("maxIter")
print(f"regParam: {regParam}")
print(f"maxIter: {maxIter}")
```

---

## 11. Thuật toán cho hệ gợi ý (ALS)

### 11.1. Lớp ALS (Alternating Least Squares)
- ALS là thuật toán học máy dùng trong các hệ thống gợi ý (Recommendation Systems)
- Giải quyết bài toán matrix factorization cho các hệ thống gợi ý
- Phân tách ma trận đánh giá người dùng - sản phẩm (user-item matrix) thành 2 ma trận con

### 11.2. Matrix Factorization
```
User-Item Matrix ≈ User Matrix × Item Matrix
```

### 11.3. Ứng dụng của ALS
- Dự đoán các đánh giá chưa được cung cấp (ratings)
- Đưa ra gợi ý cho người dùng
- Ứng dụng trong Netflix, Amazon, và các nền tảng truyền thông khác

### 11.4. Các tham số chính của ALS

| Tham số | Mô tả |
|---------|-------|
| `maxIter` | Số vòng lặp tối đa để thuật toán tối ưu hóa |
| `regParam` | Hệ số điều chỉnh (regularization) |
| `userCol` | Cột đại diện cho người dùng (user ID) |
| `itemCol` | Cột đại diện cho sản phẩm (item ID) |
| `ratingCol` | Cột chứa các đánh giá |
| `coldStartStrategy` | Cách xử lý cold start: drop, nan, hoặc skip |

### 11.5. Ví dụ
```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Dữ liệu mẫu: userId, movieId, rating
data = [
    (0, 1, 4.0), (0, 2, 5.0), (1, 1, 4.0),
    (1, 3, 3.0), (2, 2, 4.0)
]
df = spark.createDataFrame(data, ["userId", "movieId", "rating"])

# Khởi tạo ALS
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop"
)

# Huấn luyện mô hình
model = als.fit(df)

# Dự đoán với mô hình đã huấn luyện
predictions = model.transform(df)

# Đánh giá mô hình
evaluator = RegressionEvaluator(
    predictionCol="prediction",
    labelCol="rating",
    metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")

# Hiển thị các dự đoán
predictions.show()
```

---

## 12. Xử lý dữ liệu văn bản

### 12.1. Các chức năng xử lý văn bản

| Lớp | Mô tả |
|-----|-------|
| `Tokenizer` | Tách văn bản thành các từ (tokens) |
| `StopWordsRemover` | Loại bỏ các từ dừng (stop words) |
| `HashingTF` | Chuyển đổi danh sách từ thành vector tần suất |
| `IDF` | Tính toán trọng số từ dựa trên công thức IDF |
| `CountVectorizer` | Đếm tần suất từ và chuyển thành vector |
| `Word2Vec` | Chuyển từ thành vector số học |
| `NGram` | Tạo các n-grams (2-grams, 3-grams) |

### 12.2. Lớp Tokenizer
```python
from pyspark.ml.feature import Tokenizer

data = [
    ("I love PySpark",),
    ("Tokenization is fun!",),
    ("Spark is amazing!",)
]
columns = ["text"]
df = spark.createDataFrame(data, columns)

# Khởi tạo Tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# Áp dụng Tokenizer lên DataFrame
df_words = tokenizer.transform(df)

# Hiển thị kết quả
df_words.select("text", "words").show(truncate=False)
```

**Lưu ý:** Để tách từ tiếng Việt phải dùng các thư viện: VnCoreNLP, VietTokenizer hay PyVnTokenizer

### 12.3. Lớp StopWordsRemover
```python
from pyspark.ml.feature import StopWordsRemover

# Loại bỏ stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_clean = remover.transform(df_words)

# Hiển thị kết quả
df_clean.select("text", "filtered_words").show(truncate=False)
```

#### Loại bỏ từ dừng tiếng Việt
```python
# Tạo dữ liệu tiếng Việt
data = [
    ("Tôi yêu học lập trình",),
    ("PySpark rất mạnh mẽ và nhanh chóng",),
    ("Dữ liệu lớn cần được xử lý hiệu quả",)
]
df = spark.createDataFrame(data, ["text"])

# Tách từ
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df_words = tokenizer.transform(df)

# Loại bỏ stop words tiếng Việt
custom_stopwords = ["Tôi", "và", "được", "là"]
remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words",
    stopWords=custom_stopwords
)
df_clean = remover.transform(df_words)
df_clean.select("text", "filtered_words").show(truncate=False)
```

### 12.4. Lớp HashingTF
```python
from pyspark.ml.feature import HashingTF

# Áp dụng HashingTF để chuyển các từ thành các vector tần suất
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10)
df_features = hashingTF.transform(df_words)

# Hiển thị kết quả
df_features.select("text", "words", "raw_features").show(truncate=False)
```

### 12.5. Lớp IDF
IDF đo lường mức độ hiếm của một từ trong tập tài liệu.

```python
from pyspark.ml.feature import IDF

# Tính toán IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_features)
df_idf = idf_model.transform(df_features)
df_idf.show(truncate=False)
```

### 12.6. Lớp CountVectorizer
Tạo ra các vector tần suất từ (bag-of-words) từ dữ liệu văn bản.

```python
from pyspark.ml.feature import CountVectorizer

# Khởi tạo CountVectorizer
count_vectorizer = CountVectorizer(inputCol="words", outputCol="features")

# Chạy CountVectorizer trên DataFrame
model = count_vectorizer.fit(df_words)
df_features = model.transform(df_words)

# Hiển thị kết quả
df_features.show(truncate=False)
```

### 12.7. Lớp Word2Vec
Chuyển các từ trong văn bản thành các vector có độ dài cố định.

**Hai mô hình học:**
- **CBOW (Continuous Bag of Words)**: Dự đoán từ trung tâm từ các từ xung quanh
- **Skip-gram**: Dự đoán các từ xung quanh từ trung tâm

```python
from pyspark.ml.feature import Word2Vec

# Khởi tạo Word2Vec
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="words", outputCol="result")

# Áp dụng Word2Vec
model = word2Vec.fit(df_words)
result = model.transform(df_words)

# Hiển thị kết quả
result.show(truncate=False)
```

### 12.8. Lớp NGram
Chuyển đổi chuỗi các từ thành các n-grams (chuỗi con liên tiếp gồm n từ).

```python
from pyspark.ml.feature import NGram

# Áp dụng NGram (sử dụng bigram, tức là n=2)
ngram = NGram(n=2, inputCol="words", outputCol="bigrams")

# Tạo n-gram từ DataFrame đã tách từ
result = ngram.transform(df_words)

# Hiển thị kết quả
result.show(truncate=False)
```

---

## 13. Tổng kết

### 13.1. Thư viện ML
- Đã cài đặt một số thuật toán học máy phù hợp với môi trường dữ liệu lớn

### 13.2. Các nhóm lớp chính

| Nhóm | Mô tả |
|------|-------|
| **Transforms** | Chuyển đổi dữ liệu (StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler) |
| **Estimator** | Các thuật toán học máy (LogisticRegression, DecisionTree, RandomForest, KMeans) |
| **Evaluation** | Đánh giá mô hình (BinaryClassificationEvaluator, MulticlassClassificationEvaluator) |
| **Pipeline** | Xây dựng quy trình học máy |

### 13.3. Các kỹ thuật nâng cao
- **Trích chọn đặc trưng**: ChiSqSelector, PCA
- **Lựa chọn bộ tham số tối ưu**: ParamGridBuilder, CrossValidator, TrainValidationSplit
- **Hệ gợi ý**: ALS (Alternating Least Squares)
- **Xử lý văn bản**: Tokenizer, StopWordsRemover, TF-IDF, Word2Vec, NGram
