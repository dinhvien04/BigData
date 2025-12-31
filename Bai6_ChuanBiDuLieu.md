# Bài 6: Chuẩn bị dữ liệu (Data Preparation)

## Mục lục
1. [Dữ liệu trùng lặp (Duplicates)](#1-dữ-liệu-trùng-lặp-duplicates)
2. [Dữ liệu thiếu (Missing Data)](#2-dữ-liệu-thiếu-missing-data)
3. [Dữ liệu bất thường (Outliers)](#3-dữ-liệu-bất-thường-outliers)
4. [Thống kê mô tả (Descriptive Statistics)](#4-thống-kê-mô-tả-descriptive-statistics)
5. [Tương quan (Correlations)](#5-tương-quan-correlations)
6. [Trực quan hóa dữ liệu (Visualization)](#6-trực-quan-hóa-dữ-liệu-visualization)
7. [Tổng kết](#7-tổng-kết)

---

## 1. Dữ liệu trùng lặp (Duplicates)

### 1.1. Nguyên nhân dữ liệu trùng lặp
- Dữ liệu thu thập từ nhiều nguồn
- Truy vấn, tính toán, thống kê lặp lại
- Một số dữ liệu trùng cần loại bỏ

### 1.2. Các trường hợp trùng lặp

| Trường hợp | Mô tả | Ví dụ |
|------------|-------|-------|
| Giống nhau hoàn toàn | Tất cả các cột đều giống nhau | id=3 xuất hiện 2 lần |
| Giống dữ liệu, khác id | Dữ liệu giống nhưng id khác | id=1 và id=4 |
| Trùng id, khác dữ liệu | Cùng id nhưng dữ liệu khác | id=5 có 2 bản ghi khác nhau |

### 1.3. Dữ liệu minh họa
```python
df = spark.createDataFrame([
    (1, 144.5, 5.9, 33, 'M'),
    (2, 167.2, 5.4, 45, 'M'),
    (3, 124.1, 5.2, 23, 'F'),
    (4, 144.5, 5.9, 33, 'M'),  # Trùng dữ liệu với id=1
    (5, 133.2, 5.7, 54, 'F'),
    (3, 124.1, 5.2, 23, 'F'),  # Trùng hoàn toàn với dòng id=3
    (5, 129.2, 5.3, 42, 'M')   # Trùng id=5 nhưng dữ liệu khác
], ['id', 'weight', 'height', 'age', 'gender'])
```

### 1.4. Các hàm xử lý dữ liệu trùng lặp

| Hàm | Mô tả |
|-----|-------|
| `df.count()` | Đếm tổng số dòng |
| `df.distinct()` | Lấy các dòng không trùng lặp |
| `df.dropDuplicates()` | Xóa các dòng trùng lặp |
| `fn.count(col)` | Đếm số giá trị trong cột |
| `fn.countDistinct(col)` | Đếm số giá trị không trùng trong cột |
| `fn.monotonically_increasing_id()` | Tạo id tự động tăng |

### 1.5. Đếm dữ liệu trùng lặp
```python
# Đếm tổng số dòng
print('Count of rows: {0}'.format(df.count()))

# Đếm số dòng không trùng lặp
print('Count of distinct rows: {0}'.format(df.distinct().count()))
```

### 1.6. Xóa dữ liệu trùng lặp hoàn toàn
```python
# Xóa các dòng trùng lặp hoàn toàn
df = df.dropDuplicates()
df.show()
```

### 1.7. Đếm dữ liệu không trùng (bỏ qua cột id)
```python
# Đếm số dòng không trùng khi bỏ qua cột id
print('Count of ids: {0}'.format(df.count()))
print('Count of distinct ids: {0}'.format(
    df.select([c for c in df.columns if c != 'id']).distinct().count()
))
```

### 1.8. Xóa dữ liệu trùng (bỏ qua cột id)
```python
# Xóa dữ liệu trùng không xét cột id
df = df.dropDuplicates(subset=[c for c in df.columns if c != 'id'])
```

### 1.9. Đếm số id và số id không trùng
```python
import pyspark.sql.functions as fn

df.agg(
    fn.count('id').alias('count'),
    fn.countDistinct('id').alias('distinct')
).show()
```

### 1.10. Tạo id mới cho dữ liệu
```python
# Tạo id tự động tăng
df.withColumn('new_id', fn.monotonically_increasing_id()).show()
```

**Lưu ý:** Phương thức `monotonically_increasing_id()` có thể gán id cho 1 tỷ phân vùng, mỗi phân vùng 8 tỷ bản ghi. Các bản ghi trong mỗi phân vùng id không trùng nhau.

---

## 2. Dữ liệu thiếu (Missing Data)

### 2.1. Nguyên nhân dữ liệu thiếu
- Hệ thống bị lỗi
- Người nhập/thu thập dữ liệu
- Lược đồ dữ liệu thay đổi
- ...

### 2.2. Cách xử lý dữ liệu thiếu
| Phương pháp | Mô tả |
|-------------|-------|
| Xóa dòng | Xóa các dòng có dữ liệu thiếu → mất dữ liệu |
| Xóa cột | Kiểm tra các thuộc tính có nhiều dữ liệu thiếu, xóa các thuộc tính này |
| Thêm loại Missing | Dữ liệu rời rạc, phân loại: thêm loại "Missing" |
| Thay bằng mean/median | Dữ liệu số thay bằng trung bình, trung vị hoặc giá trị phù hợp |

### 2.3. Dữ liệu minh họa
```python
df_miss = spark.createDataFrame([
    (1, 143.5, 5.6, 28, 'M', 100000),
    (2, 167.2, 5.4, 45, 'M', None),
    (3, None,  5.2, None, None, None),
    (4, 144.5, 5.9, 33, 'M', None),
    (5, 133.2, 5.7, 54, 'F', None),
    (6, 124.1, 5.2, None, 'F', None),
    (7, 129.2, 5.3, 42, 'M', 76000)
], ['id', 'weight', 'height', 'age', 'gender', 'income'])
```

### 2.4. Đếm số giá trị thiếu trong từng dòng
```python
df_miss.rdd.map(
    lambda row: (row['id'], sum([c == None for c in row]))
).collect()
```

### 2.5. Xem một dòng cụ thể
```python
df_miss.where('id == 3').show()
```

### 2.6. Tính tỷ lệ phần trăm dữ liệu thiếu mỗi cột
```python
import pyspark.sql.functions as fn

df_miss.agg(*[
    (1 - (fn.count(c) / fn.count('*'))).alias(c + '_missing')
    for c in df_miss.columns
]).show()
```

### 2.7. Xóa cột có nhiều dữ liệu thiếu
```python
# Xóa cột income (có nhiều giá trị None)
df_miss_no_income = df_miss.select([c for c in df_miss.columns if c != 'income'])
```

### 2.8. Xóa dòng chứa dữ liệu thiếu - dropna()
```python
df.dropna(how="any", thresh=None, subset=None)
```

| Tham số | Mô tả |
|---------|-------|
| `how="any"` | Xóa dòng có ít nhất 1 giá trị None |
| `how="all"` | Xóa dòng có tất cả giá trị None |
| `thresh=n` | Xóa dòng có từ n giá trị None trở lên |
| `subset=[cols]` | Chỉ xét các cột trong danh sách |

```python
# Xóa dòng có ít hơn 3 giá trị không null
df_miss_no_income.dropna(thresh=3).show()
```

### 2.9. Điền dữ liệu trống - fillna()
```python
df.fillna(value, subset=None)
df.fill(value, subset=None)
```

| Tham số | Mô tả |
|---------|-------|
| `value` | Giá trị thay vào dữ liệu trống |
| `subset` | Các cột cần thay dữ liệu trống |

### 2.10. Điền dữ liệu trống bằng mean
```python
# Tính mean cho các cột số và lưu vào dict
means = df_miss_no_income.agg(
    *[fn.mean(c).alias(c) for c in df_miss_no_income.columns if c != 'gender']
).toPandas().to_dict('records')[0]

# Thêm giá trị cho cột gender
means['gender'] = 'missing'

# Điền vào các giá trị trống
df_miss_no_income.fillna(means).show()
```

**Kết quả dict means:**
```
{'id': 4.0, 'weight': 140.283333, 'height': 5.471429, 'age': 40.4}
```

---

## 3. Dữ liệu bất thường (Outliers)

### 3.1. Khái niệm
- Dữ liệu bất thường là dữ liệu có độ lệch lớn so với phân bố của dữ liệu
- Thông thường dữ liệu chấp nhận được nếu nằm trong khoảng: **Q1 - 1.5×IQR** đến **Q3 + 1.5×IQR**
- **IQR** (Interquartile Range) = Q3 - Q1

### 3.2. Công thức xác định Outliers
```
Cận dưới = Q1 - 1.5 × IQR
Cận trên = Q3 + 1.5 × IQR
```

### 3.3. Dữ liệu minh họa
```python
df_outliers = spark.createDataFrame([
    (1, 143.5, 5.3, 28),
    (2, 154.2, 5.5, 45),
    (3, 342.3, 5.1, 99),  # weight và age có thể là outlier
    (4, 144.5, 5.5, 33),
    (5, 133.2, 5.4, 54),
    (6, 124.1, 5.1, 21),
    (7, 129.2, 5.3, 42)
], ['id', 'weight', 'height', 'age'])
```

### 3.4. Tính cận dưới và cận trên bằng approxQuantile
```python
cols = ['weight', 'height', 'age']
bounds = {}

for col in cols:
    # Tính Q1 (25%) và Q3 (75%)
    quantiles = df_outliers.approxQuantile(col, [0.25, 0.75], 0.05)
    IQR = quantiles[1] - quantiles[0]
    
    # Tính cận dưới và cận trên
    bounds[col] = [
        quantiles[0] - 1.5 * IQR,  # Cận dưới
        quantiles[1] + 1.5 * IQR   # Cận trên
    ]
```

**Cú pháp approxQuantile:**
```python
approxQuantile(column, [f1, f2, ...], err)
```
- `column`: Tên cột
- `[f1, f2, ...]`: Danh sách tỷ lệ phần trăm (0.25 = 25%, 0.75 = 75%)
- `err`: Sai số cho phép

### 3.5. Kiểm tra outliers của từng dữ liệu
```python
outliers = df_outliers.select(*['id'] + [
    ((df_outliers[c] < bounds[c][0]) | (df_outliers[c] > bounds[c][1])).alias(c + '_o')
    for c in cols
])
outliers.show()
```

### 3.6. Lọc dữ liệu outlier
```python
# Join với bảng outliers
df_outliers = df_outliers.join(outliers, on='id')

# Lọc các dòng có weight là outlier
df_outliers.filter('weight_o').select('id', 'weight').show()

# Lọc các dòng có age là outlier
df_outliers.filter('age_o').select('id', 'age').show()
```

---

## 4. Thống kê mô tả (Descriptive Statistics)

### 4.1. Các chức năng thống kê
- Mean (trung bình)
- Median (trung vị)
- Mode (yếu vị)
- Stdev (độ lệch chuẩn)
- Skewness (độ lệch)
- Kurtosis (độ nhọn)
- Correlations (tương quan)

### 4.2. Dữ liệu minh họa - ccFraud Dataset
**Nguồn:** http://tomdrabas.com/data/LearningPySpark/ccFraud.csv.gz

| Tên trường | Ý nghĩa |
|------------|---------|
| custID | ID khách hàng |
| gender | Giới tính (nam: 1, nữ: 2) |
| state | Khu vực của khách hàng |
| cardholder | Khách hàng có sử dụng card không (1: có, 0: không) |
| balance | Số dư trong tài khoản ngân hàng |
| numTrans | Số giao dịch trong hệ thống ngân hàng |
| numIntlTrans | Số giao dịch liên ngân hàng |
| creditLine | Hạn mức tín dụng |
| fraudRisk | Khách hàng có gian lận hay không (1: có, 0: không) |

### 4.3. Nạp dữ liệu
```python
# Nạp dữ liệu từ file CSV nén
fraud = sc.textFile('ccFraud.csv.gz')

# Bỏ dòng header
header = fraud.first()
fraud = fraud.filter(lambda row: row != header)\
    .map(lambda row: [int(elem) for elem in row.split(',')])
```

### 4.4. Tạo lược đồ dữ liệu
```python
import pyspark.sql.types as typ

fields = [
    typ.StructField(h[1:-1], typ.IntegerType(), True)
    for h in header.split(',')
]
schema = typ.StructType(fields)
```

### 4.5. Tạo DataFrame
```python
fraud_df = spark.createDataFrame(fraud, schema)
fraud_df.printSchema()
```

### 4.6. Khám phá dữ liệu

#### Thống kê số lượng theo giới tính
```python
fraud_df.groupby('gender').count().show()
```

#### Xem mô tả một số trường
```python
numerical = ['balance', 'numTrans', 'numIntlTrans']
desc = fraud_df.describe(numerical)
desc.show()
```

### 4.7. Độ lệch (Skewness)

#### Khái niệm
- Độ lệch dùng để đo sự đối xứng của phân phối dữ liệu
- **Độ lệch = 0**: Phân phối chuẩn (Mean = Median = Mode)
- **Độ lệch dương**: Đuôi phân phối lệch về bên phải
- **Độ lệch âm**: Đuôi phân phối lệch về bên trái

#### Công thức
```
Skewness = E[(X - μ)³] / σ³
```

#### Code tính độ lệch
```python
# Cách 1: Sử dụng agg
fraud_df.agg({'balance': 'skewness'}).show()

# Cách 2: Sử dụng hàm skewness
from pyspark.sql.functions import skewness
df_outliers.select(skewness("balance")).show()
```

### 4.8. Độ nhọn (Kurtosis)

#### Khái niệm
- Độ nhọn dùng để đo độ cao phần trung tâm phân phối so với phân phối chuẩn
- **Kurtosis > 0**: Phân phối nhọn hơn phân phối chuẩn
- **Kurtosis < 0**: Phân phối phẳng hơn phân phối chuẩn
- **Kurtosis = 0**: Phân phối chuẩn

#### Công thức
```
Kurtosis = E[(X - μ)⁴] / σ⁴ - 3
```

#### Code tính độ nhọn
```python
# Cách 1: Sử dụng agg
fraud_df.agg({'balance': 'kurtosis'}).show()

# Cách 2: Sử dụng hàm kurtosis
from pyspark.sql.functions import kurtosis
df_outliers.select(kurtosis("balance")).show()
```

---

## 5. Tương quan (Correlations)

### 5.1. Khái niệm
- Dùng để đo mối tương quan giữa các thuộc tính
- Hệ số tương quan nằm trong khoảng **-1 đến 1**

### 5.2. Ý nghĩa hệ số tương quan

| Giá trị | Ý nghĩa |
|---------|---------|
| r > 0 | Hai thuộc tính đồng biến (cùng tăng hoặc cùng giảm) |
| r < 0 | Hai thuộc tính nghịch biến (một tăng, một giảm) |
| r = 0 | Hai thuộc tính độc lập |
| r = 1 | Tương quan dương hoàn hảo |
| r = -1 | Tương quan âm hoàn hảo |

### 5.3. Công thức Pearson
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
```

### 5.4. Tính hệ số tương quan giữa hai thuộc tính
```python
# Tính tương quan giữa balance và numTrans
fraud_df.corr('balance', 'numTrans')
# Kết quả: 0.00044523140172659576
```

### 5.5. Tính ma trận tương quan
```python
numerical = ['balance', 'numTrans', 'numIntlTrans']
n_numerical = len(numerical)

corr = []
for i in range(0, n_numerical):
    temp = [None] * i
    for j in range(i, n_numerical):
        temp.append(fraud_df.corr(numerical[i], numerical[j]))
    corr.append(temp)
```

---

## 6. Trực quan hóa dữ liệu (Visualization)

### 6.1. Khái niệm
- Trực quan hóa là mô tả dữ liệu dưới dạng các biểu đồ
- Giúp cho việc hiểu dữ liệu dễ dàng hơn
- Thư viện vẽ biểu đồ thông dụng trong Python: **Matplotlib**

### 6.2. Các loại biểu đồ thường dùng

| Loại biểu đồ | Hàm | Mô tả |
|--------------|-----|-------|
| Đường | `plot()` | Biểu diễn xu hướng theo thời gian |
| Thanh | `bar()`, `barh()` | So sánh các giá trị |
| Histogram | `hist()` | Phân phối tần suất |
| Tròn | `pie()` | Tỷ lệ phần trăm |
| Hộp | `boxplot()` | Phân phối và outliers |
| Phân tán | `scatter()` | Tương quan giữa 2 biến |

### 6.3. Nạp thư viện Matplotlib
```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
```

### 6.4. Biểu đồ tần suất (Histograms)

#### Khái niệm
- Histograms là cách dễ dàng để biểu diễn phân phối dữ liệu các thuộc tính dựa trên thống kê tần suất

#### 3 cách sử dụng Histograms trong PySpark

| Cách | Mô tả |
|------|-------|
| 1 | Gộp dữ liệu trên các workers rồi đếm theo từng nhóm của histogram ở Driver |
| 2 | Lấy tất cả dữ liệu về Driver rồi dùng các phương thức của thư viện để vẽ biểu đồ |
| 3 | Lấy mẫu dữ liệu ở các workers rồi chuyển về Driver để vẽ biểu đồ |

#### Phương thức RDD.histogram(buckets)
```python
# Ví dụ 1: Chia thành 2 buckets
rdd = sc.parallelize(range(51))
rdd.histogram(2)
# Kết quả: ([0, 25, 50], [25, 26])

# Ví dụ 2: Chỉ định các buckets
rdd.histogram([0, 5, 25, 50])
# Kết quả: ([0, 5, 25, 50], [5, 20, 26])

# Ví dụ 3: Chia đều các buckets
rdd.histogram([0, 15, 30, 45, 60])
# Kết quả: ([0, 15, 30, 45, 60], [15, 15, 15, 6])

# Ví dụ 4: Histogram với chuỗi
rdd = sc.parallelize(["ab", "ac", "b", "bd", "ef"])
rdd.histogram(("a", "b", "c"))
# Kết quả: (('a', 'b', 'c'), [2, 2])
```

### 6.5. Cách 1: Gộp trước khi vẽ biểu đồ
```python
# Tạo histogram với 20 buckets
hists = fraud_df.select('balance').rdd.flatMap(lambda row: row).histogram(20)

# Vẽ biểu đồ bằng Matplotlib
data = {
    'bins': hists[0][:-1],
    'freq': hists[1]
}
plt.bar(data['bins'], data['freq'], width=2000)
plt.title('Histogram of \'balance\'')
```

### 6.6. Cách 2: Lấy toàn bộ dữ liệu
```python
# Lấy dữ liệu về Driver
data_driver = {
    'obs': fraud_df.select('balance').rdd.flatMap(lambda row: row).collect()
}

# Vẽ biểu đồ bằng Matplotlib
plt.hist(data_driver['obs'], bins=20)
plt.title('Histogram of \'balance\' using .hist()')
```

### 6.7. Biểu đồ phân tán (Scatter Chart)

#### Khái niệm
- Biểu đồ phân tán cho phép biểu diễn trực quan tương tác giữa tối đa 3 thuộc tính
- PySpark không hỗ trợ module trực quan hóa nên việc trực quan hóa với các dữ liệu lớn là không thể
- Việc lấy mẫu dữ liệu với một tỷ lệ nhỏ để trực quan hóa giúp biểu diễn dữ liệu lớn

#### Trực quan hóa theo mẫu dữ liệu
```python
numerical = ['balance', 'numTrans', 'numIntlTrans']

# Trích 0.02% dữ liệu theo giới tính
data_sample = fraud_df.sampleBy(
    'gender',
    {1: 0.0002, 2: 0.0002}
).select(numerical)

# Chuyển dữ liệu về Driver
data_multi = dict([
    (elem, data_sample.select(elem).rdd.flatMap(lambda row: row).collect())
    for elem in numerical
])

# Vẽ biểu đồ phân tán
plt.scatter(data_multi['balance'], data_multi['numTrans'], color='r')
```

---

## 7. Tổng kết

1. **Việc chuẩn bị dữ liệu** trước khi xây dựng mô hình là công việc rất quan trọng và tốn nhiều thời gian

2. **Để hiểu được dữ liệu** cần nhiều công cụ, quan trọng là:
   - Thống kê mô tả
   - Trực quan hóa

3. **Hiểu dữ liệu** sẽ giúp trích xuất các đặc trưng cần thiết cho việc xây dựng mô hình

4. **Các bước chuẩn bị dữ liệu:**
   - Xử lý dữ liệu trùng lặp (Duplicates)
   - Xử lý dữ liệu thiếu (Missing)
   - Xử lý dữ liệu bất thường (Outliers)
   - Thống kê và phân tích tương quan
   - Trực quan hóa để hiểu dữ liệu
