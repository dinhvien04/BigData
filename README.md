# Tài liệu ôn tập PySpark - Phân tích dữ liệu lớn

## Mục lục tài liệu

| File | Nội dung |
|------|----------|
| [Bai5_SparkSQL.md](Bai5_SparkSQL.md) | SparkSQL, DataFrame, các thao tác truy vấn dữ liệu |
| [Bai6_ChuanBiDuLieu.md](Bai6_ChuanBiDuLieu.md) | Xử lý dữ liệu trùng lặp, thiếu, outliers, thống kê, trực quan hóa |
| [Bai7_ThuVienML.md](Bai7_ThuVienML.md) | Thư viện ML, Transformer, Estimator, Pipeline, Classification, Clustering |

---

## Thông tin bài thi cuối kỳ

| Thông tin | Chi tiết |
|-----------|----------|
| Hình thức | Thực hành |
| Thời gian | 90 phút |
| Được phép | Sử dụng AI |
| Dữ liệu | Text, CSV |
| Nộp bài | Qua Classroom |
| Định dạng | Notebook (.ipynb) |

---

## Cấu trúc trình bày mỗi câu (3 cells)

```
┌─────────────────────────────────────────────────────────┐
│ Cell 1: Markdown - Mô tả bài toán                       │
│ - Đánh số câu                                           │
│ - Dữ liệu đầu vào là gì?                                │
│ - Các bước tính toán thế nào?                           │
│ - Kết quả trả về là gì?                                 │
├─────────────────────────────────────────────────────────┤
│ Cell 2: Code - Thực thi                                 │
│ - Code PySpark                                          │
│ - Comment giải thích từng lệnh theo các bước ở Cell 1   │
├─────────────────────────────────────────────────────────┤
│ Cell 3: Markdown - Giải thích kết quả                   │
│ - Ý nghĩa kết quả sau khi tính toán                     │
│ - Nhận xét, phân tích bằng ngôn ngữ tự nhiên            │
└─────────────────────────────────────────────────────────┘
```

---

## Thang điểm đánh giá

| Điểm | Yêu cầu |
|------|---------|
| **4 điểm** | Nạp dữ liệu RDD (text, csv), map, flatMap, tính toán đơn giản (ghép, lọc, đếm, min, max) |
| **5-6 điểm** | Phân nhóm, tính toán trên nhóm, ghi dữ liệu ra file |
| **7-8 điểm** | Kết hợp dữ liệu từ nhiều thao tác, vẽ biểu đồ minh họa |
| **9-10 điểm** | Yêu cầu khó, tối ưu tính toán, giải pháp sáng tạo, trình bày rõ ràng |

**Lưu ý:** Quan điểm xuyên suốt khi chấm bài là phải có **TÍNH NGƯỜI** - thể hiện sự hiểu biết, không chỉ copy code máy móc.

---

## Tóm tắt kiến thức cần nhớ

### 1. SparkSQL & DataFrame (Bài 5)

```python
# Khởi tạo SparkSession
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Đọc dữ liệu
df = spark.read.csv('data.csv', header=True, inferSchema=True)
df = spark.read.json('data.json')

# Các thao tác cơ bản
df.show()                    # Hiển thị dữ liệu
df.printSchema()             # Hiển thị cấu trúc
df.count()                   # Đếm số dòng
df.select('col1', 'col2')    # Chọn cột
df.filter('age > 20')        # Lọc dòng
df.groupBy('col').count()    # Phân nhóm
df.orderBy('col')            # Sắp xếp

# Thêm/sửa cột
df.withColumn('new_col', col('old_col') * 2)
df.withColumnRenamed('old', 'new')
df.drop('col')

# Join
df1.join(df2, df1.id == df2.id, 'inner')

# SQL
df.createOrReplaceTempView("table")
spark.sql("SELECT * FROM table WHERE age > 20")
```

### 2. Chuẩn bị dữ liệu (Bài 6)

```python
# Xử lý trùng lặp
df.distinct()
df.dropDuplicates()
df.dropDuplicates(subset=['col1', 'col2'])

# Xử lý dữ liệu thiếu
df.dropna()                           # Xóa dòng có null
df.fillna(0)                          # Điền giá trị
df.fillna({'col1': 0, 'col2': 'N/A'}) # Điền theo cột

# Xử lý outliers
quantiles = df.approxQuantile('col', [0.25, 0.75], 0.05)
IQR = quantiles[1] - quantiles[0]
lower = quantiles[0] - 1.5 * IQR
upper = quantiles[1] + 1.5 * IQR

# Thống kê
df.describe().show()
df.agg({'col': 'skewness'}).show()  # Độ lệch
df.agg({'col': 'kurtosis'}).show()  # Độ nhọn
df.corr('col1', 'col2')             # Tương quan

# Trực quan hóa
import matplotlib.pyplot as plt
plt.hist(data, bins=20)
plt.scatter(x, y)
plt.bar(x, y)
plt.show()
```

### 3. Thư viện ML (Bài 7)

```python
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Tạo vector đặc trưng
assembler = VectorAssembler(inputCols=['col1', 'col2'], outputCol='features')
df = assembler.transform(df)

# Phân lớp
lr = LogisticRegression(featuresCol='features', labelCol='label')
model = lr.fit(train_data)
predictions = model.transform(test_data)

# Phân cụm
km = KMeans(featuresCol='features', k=3)
model = km.fit(df)
clusters = model.transform(df)

# Pipeline
pipeline = Pipeline(stages=[assembler, lr])
model = pipeline.fit(train_data)

# Đánh giá
evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')
auc = evaluator.evaluate(predictions)
```

---

## Template mẫu cho bài thi

```python
# ============================================
# CÂU 1: [Tên yêu cầu]
# ============================================
```

**Cell 1 (Markdown):**
```markdown
## Câu 1: [Tên yêu cầu]

### Mô tả bài toán:
- **Dữ liệu đầu vào:** [Mô tả file dữ liệu, các cột cần dùng]
- **Các bước thực hiện:**
  1. Bước 1: ...
  2. Bước 2: ...
  3. Bước 3: ...
- **Kết quả mong đợi:** [Mô tả output]
```

**Cell 2 (Code):**
```python
# Bước 1: Nạp dữ liệu
df = spark.read.csv('data.csv', header=True, inferSchema=True)

# Bước 2: Xử lý dữ liệu
df_filtered = df.filter('age > 20')

# Bước 3: Tính toán kết quả
result = df_filtered.groupBy('category').count()
result.show()
```

**Cell 3 (Markdown):**
```markdown
### Giải thích kết quả:
- Kết quả cho thấy...
- Nhận xét: ...
- Ý nghĩa: ...
```

---

## Lưu ý quan trọng

1. **Luôn giải thích code** - Không chỉ viết code mà phải comment giải thích
2. **Phân tích kết quả** - Sau khi chạy code, phải giải thích ý nghĩa kết quả
3. **Trình bày rõ ràng** - Sử dụng markdown để format đẹp
4. **Tối ưu code** - Tránh các thao tác thừa, sử dụng các hàm phù hợp
5. **Kiểm tra kết quả** - Dùng `.show()` để xem kết quả trước khi kết luận

---

## Tác giả
Tài liệu ôn tập môn Trực quan hóa và Phân tích dữ liệu lớn
