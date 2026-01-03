# CHUẨN BỊ THI PYSPARK - CHECKLIST

## 📋 TRƯỚC KHI THI (Chuẩn bị sẵn)

### 1. ✅ Kiểm tra môi trường
- [ ] Spark đã cài đặt và chạy được
- [ ] Jupyter Notebook hoạt động bình thường
- [ ] Đã set HADOOP_HOME (nếu dùng Windows)
- [ ] Kiro IDE + Claude AI đã sẵn sàng

### 2. ✅ Chuẩn bị file tài liệu
- [ ] `Bai6_ChuanBiDuLieu.md` - Tham khảo xử lý dữ liệu
- [ ] `Bai7_ThuVienML.md` - Tham khảo ML
- [ ] `bai8.ipynb` - File mẫu code style
- [ ] `.kiro/steering/pyspark-exam.md` - Hướng dẫn AI code đúng format

### 3. ✅ Test AI trước khi thi
Thử hỏi AI câu này để test:
```
"Làm câu a theo format 3 cells: Đọc file test.csv, 
khám phá dữ liệu (show, count, describe)"
```

Xem AI có tạo đúng 3 cells không:
- Cell 1: Markdown mô tả
- Cell 2: Code với comment
- Cell 3: Markdown giải thích

---

## 📚 KIẾN THỨC CẦN NHỚ

### Bài 6: Chuẩn bị dữ liệu

#### Xử lý trùng lặp
```python
df.count()                    # Đếm tổng số dòng
df.distinct().count()         # Đếm số dòng không trùng
df.dropDuplicates()           # Xóa dòng trùng
```

#### Xử lý dữ liệu thiếu
```python
df.dropna(how='any')          # Xóa dòng có ít nhất 1 null
df.dropna(thresh=3)           # Xóa dòng có < 3 giá trị not null
df.fillna(value)              # Điền giá trị vào null
df.fillna({'col': value})     # Điền theo từng cột
```

#### Xử lý outliers
```python
quantiles = df.approxQuantile('col', [0.25, 0.75], 0.05)
IQR = quantiles[1] - quantiles[0]
lower = quantiles[0] - 1.5 * IQR
upper = quantiles[1] + 1.5 * IQR
```

#### Thống kê
```python
df.describe().show()          # Thống kê mô tả
df.agg({'col': 'skewness'})   # Độ lệch
df.agg({'col': 'kurtosis'})   # Độ nhọn
df.corr('col1', 'col2')       # Tương quan
```

### Bài 7: Machine Learning

#### Tiền xử lý
```python
# Chuyển chuỗi thành số
df = df.withColumn('Gender', (F.col('Gender') == 'Female').cast('int'))

# StringIndexer
indexer = StringIndexer(inputCol='Geography', outputCol='Geography_SI')
df = indexer.fit(df).transform(df)

# OneHotEncoder
encoder = OneHotEncoder(inputCol='Geography_SI', outputCol='Geography_OH')
df = encoder.fit(df).transform(df)

# VectorAssembler
assembler = VectorAssembler(inputCols=['col1', 'col2'], outputCol='features')
df = assembler.transform(df)
```

#### Huấn luyện mô hình
```python
# Chia train/test
train, test = df.randomSplit([0.7, 0.3], seed=42)

# LogisticRegression
lr = LogisticRegression(featuresCol='features', labelCol='label')
model = lr.fit(train)

# Dự đoán
pred = model.transform(test)
```

#### Đánh giá mô hình
```python
# AUC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
auc = evaluator.evaluate(pred)

# Accuracy, Precision, Recall, F1
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator_acc = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator_acc.evaluate(pred)

evaluator_precision = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='weightedPrecision')
precision = evaluator_precision.evaluate(pred)

evaluator_recall = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='weightedRecall')
recall = evaluator_recall.evaluate(pred)

evaluator_f1 = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='f1')
f1 = evaluator_f1.evaluate(pred)
```

#### Pipeline
```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[indexer, encoder, assembler, lr])
model = pipeline.fit(train)
pred = model.transform(test)
```

#### Lưu/Nạp mô hình
```python
# Lưu
model.write().overwrite().save('model_path')

# Nạp
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load('model_path')
```

---

## 🎯 KHI THI - QUY TRÌNH

### Bước 1: Đọc đề kỹ
- Xác định bài toán: Classification? Regression? Clustering?
- Xác định dữ liệu đầu vào
- Xác định yêu cầu đầu ra

### Bước 2: Setup môi trường
```python
# Cell đầu tiên luôn là:
import os
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PATH'] = os.environ['HADOOP_HOME'] + '\\bin;' + os.environ['PATH']

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
spark = SparkSession.builder.getOrCreate()
```

### Bước 3: Làm từng câu theo format 3 cells

#### Cell 1: Markdown - Mô tả
```markdown
## Câu [SỐ]: [TÊN CÂU]

**Dữ liệu đầu vào:**
- [Mô tả input]

**Các bước thực hiện:**
1. [Bước 1]
2. [Bước 2]
3. [Bước 3]

**Kết quả mong đợi:**
- [Mô tả output]
```

#### Cell 2: Code - Thực thi
```python
# Bước 1: [Mô tả]
code_here

# Bước 2: [Mô tả]
code_here

# Bước 3: [Mô tả]
code_here
```

#### Cell 3: Markdown - Giải thích
```markdown
**Kết quả:**
- [Giải thích kết quả cụ thể]

**Nhận xét:**
- [Phân tích, đánh giá]
```

### Bước 4: Sử dụng AI hiệu quả

#### Cách hỏi AI tốt:
```
"Làm câu [X] theo format 3 cells:
[MÔ TẢ YÊU CẦU CỤ THỂ]
Code ngắn gọn như bai8.ipynb"
```

#### Ví dụ câu hỏi hay:
```
"Làm câu a theo format 3 cells: 
Đọc file data.csv, tiền xử lý (loại bỏ cột id, 
chuyển Gender thành số, encode Geography), 
tạo features, train LogisticRegression"
```

#### Nếu AI code sai format:
```
"Viết lại theo đúng format 3 cells: 
Cell 1 Markdown mô tả, 
Cell 2 Code, 
Cell 3 Markdown giải thích"
```

---

## 🔥 MẸO THI

### 1. Quản lý thời gian
- Đọc hết đề trước (5 phút)
- Làm câu dễ trước (câu a, b, c thường dễ)
- Để câu khó cuối (câu g, h, i)

### 2. Code nhanh
- Copy code từ bai8.ipynb và sửa
- Dùng AI cho phần phức tạp
- Chạy từng cell để kiểm tra

### 3. Tránh lỗi thường gặp
- Nhớ import thư viện
- Kiểm tra tên cột (case-sensitive)
- Kiểm tra kiểu dữ liệu (int, string, double)
- Nhớ fit() trước transform()

### 4. Nếu bị lỗi
- Đọc lỗi kỹ (thường báo rõ)
- Hỏi AI: "Lỗi này sửa thế nào: [COPY LỖI]"
- Kiểm tra lại tên cột, kiểu dữ liệu

---

## 📝 TEMPLATE CODE MẪU

### Đọc và khám phá dữ liệu
```python
df = spark.read.csv('file.csv', header=True, inferSchema=True)
df.printSchema()
df.show(5)
df.count()
df.describe().show()
df.groupBy('col').count().show()
```

### Tiền xử lý đầy đủ
```python
# Loại bỏ cột
df1 = df.drop('col1', 'col2')

# Chuyển Gender thành số
df1 = df1.withColumn('Gender', (F.col('Gender') == 'Female').cast('int'))

# StringIndexer + OneHotEncoder cho Geography
indexer_geo = StringIndexer(inputCol='Geography', outputCol='Geography_SI')
df1 = indexer_geo.fit(df1).transform(df1)
encoder_geo = OneHotEncoder(inputCol='Geography_SI', outputCol='Geography_OH')
df1 = encoder_geo.fit(df1).transform(df1)

# VectorAssembler
feature_cols = ['col1', 'col2', 'Geography_OH']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
df2 = assembler.transform(df1)
```

### Train và đánh giá
```python
# Chia train/test
train, test = df2.randomSplit([0.7, 0.3], seed=42)

# Train
lr = LogisticRegression(featuresCol='features', labelCol='label')
model = lr.fit(train)

# Predict
pred = model.transform(test)

# Evaluate
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

evaluator_auc = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
auc = evaluator_auc.evaluate(pred)

evaluator_acc = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator_acc.evaluate(pred)

print(f'AUC = {auc:.4f}')
print(f'Accuracy = {accuracy:.4f}')
```

### Pipeline đầy đủ
```python
from pyspark.ml import Pipeline

# Tạo stages
indexer_geo = StringIndexer(inputCol='Geography', outputCol='Geography_SI')
encoder_geo = OneHotEncoder(inputCol='Geography_SI', outputCol='Geography_OH')
assembler = VectorAssembler(inputCols=['col1', 'Geography_OH'], outputCol='features')
lr = LogisticRegression(featuresCol='features', labelCol='label')

# Tạo Pipeline
pipeline = Pipeline(stages=[indexer_geo, encoder_geo, assembler, lr])

# Train
model = pipeline.fit(train)

# Predict
pred = model.transform(test)
```

---

## ✅ CHECKLIST CUỐI CÙNG TRƯỚC KHI NỘP

- [ ] Tất cả cells đã chạy thành công (không có lỗi)
- [ ] Mỗi câu có đủ 3 cells (Markdown → Code → Markdown)
- [ ] Code có comment giải thích
- [ ] Kết quả đã được giải thích rõ ràng
- [ ] Đã kiểm tra lại đề (làm đủ yêu cầu chưa)

---

## 🎓 CHÚC BẠN THI TỐT!

**Nhớ:**
- Bình tĩnh, đọc đề kỹ
- Làm câu dễ trước
- Dùng AI thông minh
- Kiểm tra kỹ trước khi nộp

**Good luck! 🍀**
