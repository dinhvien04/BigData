# Bài 5: SparkSQL - Structured Big-Data Processing in Spark

## Mục lục
1. [Giới thiệu SparkSQL](#1-giới-thiệu-sparksql)
2. [Cấu trúc DataFrame](#2-cấu-trúc-dataframe)
3. [Tạo DataFrame](#3-tạo-dataframe)
4. [Các kiểu dữ liệu](#4-các-kiểu-dữ-liệu)
5. [Thao tác với DataFrame](#5-thao-tác-với-dataframe)
6. [Lọc dữ liệu](#6-lọc-dữ-liệu)
7. [Làm việc với các cột](#7-làm-việc-với-các-cột)
8. [Hàm gộp (Aggregate Functions)](#8-hàm-gộp-aggregate-functions)
9. [Phân nhóm (GroupBy)](#9-phân-nhóm-groupby)
10. [Pivot](#10-pivot)
11. [Cửa sổ trượt (Window Functions)](#11-cửa-sổ-trượt-window-functions)
12. [Điền dữ liệu trống](#12-điền-dữ-liệu-trống)
13. [Join DataFrame](#13-join-dataframe)
14. [DataFrame và SQL](#14-dataframe-và-sql)
15. [Hàm do người dùng định nghĩa (UDF)](#15-hàm-do-người-dùng-định-nghĩa-udf)
16. [Làm việc với Array](#16-làm-việc-với-array)
17. [Ví dụ thực tế: MovieLens](#17-ví-dụ-thực-tế-movielens)

---

## 1. Giới thiệu SparkSQL

### 1.1. Tại sao cần SparkSQL?
- **Spark RDD** phù hợp cho xử lý chung các loại dữ liệu
- Đối với dữ liệu có cấu trúc, cần tự cung cấp bộ phân tích (parser) và logic xử lý
- Xử lý dữ liệu có cấu trúc vẫn còn phổ biến trong Big Data, đặc biệt trong các thuật toán học máy

### 1.2. SparkSQL là gì?
- SparkSQL là thư viện của Spark dùng thao tác với dữ liệu có cấu trúc
- **DataFrame** là kiểu dữ liệu dùng để thao tác với dữ liệu có cấu trúc dạng bảng
- DataFrame sử dụng RDD để phân tán dữ liệu trên nhiều máy tính
- DataFrame hỗ trợ nhiều thao tác truy vấn với dữ liệu có cấu trúc như SQL
- DataFrame có thể xử lý dữ liệu rất lớn (hàng chục tỉ dòng) và tính toán phân tán trên cụm máy tính

---

## 2. Cấu trúc DataFrame

### 2.1. Đặc điểm DataFrame
- **DataFrame = RDD + Schema**
- DataFrame trong SparkSQL tương đương với các bảng trong hệ quản trị CSDL quan hệ
- Bao gồm các hàng (rows) và cột (columns)
- DataFrame KHÔNG ở dạng chuẩn 1NF (First Normal Form)

### 2.2. So sánh DataFrame với RDD

| DataFrame | RDD |
|-----------|-----|
| Thực thi trì hoãn (Lazy execution) | Thực thi trì hoãn (Lazy execution) |
| Nhận biết được mô hình dữ liệu | Mô hình dữ liệu bị ẩn |
| Nhận biết được logic truy vấn | Các phép biến đổi là hộp đen |
| Có thể tối ưu hóa truy vấn | Không thể tối ưu hóa truy vấn |

### 2.3. Catalyst Optimizer
- Tối ưu hóa truy vấn trong SparkSQL
- Thao tác với DataFrame/SQL qua các bước được tối ưu:
  - Catalog (chứa thông tin về các bảng, cột, và cấu trúc dữ liệu)
  - Phân tích câu truy vấn SQL hoặc DataFrame
  - Unresolved Logical Plan (Kế hoạch logic chưa giải quyết)
  - Cost Model (Mô hình chi phí)
  - Tạo mã Java cho kế hoạch vật lý đã chọn

### 2.4. Lịch sử Spark APIs
- **RDD (2011)**: Distribute collection of JVM objects
- **DataFrame (2013)**: Distribute collection of Row objects
- **DataSet (2015)**: Internally rows, externally JVM objects
- **Dataset (2016)**: DataFrame = Dataset[Row] (Alias)

---

## 3. Tạo DataFrame

### 3.1. Tạo SparkSession
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
```

### 3.2. Tạo DataFrame từ các đối tượng Row
```python
from datetime import datetime, date
from pyspark.sql import Row

df = spark.createDataFrame([
    Row(a=1, b=4., c='GFG1', d=date(2000, 8, 1), e=datetime(2000, 8, 1, 12, 0)),
    Row(a=2, b=8., c='GFG2', d=date(2000, 6, 2), e=datetime(2000, 6, 2, 12, 0)),
    Row(a=4, b=5., c='GFG3', d=date(2000, 5, 3), e=datetime(2000, 5, 3, 12, 0))
])

# Hiển thị bảng
df.show()

# Hiển thị schema
df.printSchema()
```

### 3.3. Tạo DataFrame từ các bộ dữ liệu và lược đồ (schema)
```python
df = spark.createDataFrame([
    (1, 4., 'GFG1', date(2000, 8, 1), datetime(2000, 8, 1, 12, 0)),
    (2, 8., 'GFG2', date(2000, 6, 2), datetime(2000, 6, 2, 12, 0)),
    (3, 5., 'GFG3', date(2000, 5, 3), datetime(2000, 5, 3, 12, 0))
], schema='a long, b double, c string, d date, e timestamp')
```

### 3.4. Tạo DataFrame sử dụng Pandas
```python
import pandas as pd

pandas_df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4., 8., 5.],
    'c': ['GFG1', 'GFG2', 'GFG3'],
    'd': [date(2000, 8, 1), date(2000, 6, 2), date(2000, 5, 3)],
    'e': [datetime(2000, 8, 1, 12, 0), datetime(2000, 6, 2, 12, 0), datetime(2000, 5, 3, 12, 0)]
})

df = spark.createDataFrame(pandas_df)
```

### 3.5. Tạo DataFrame từ RDD
```python
rdd = spark.sparkContext.parallelize([
    (1, 4., 'GFG1', date(2000, 8, 1), datetime(2000, 8, 1, 12, 0)),
    (2, 8., 'GFG2', date(2000, 6, 2), datetime(2000, 6, 2, 12, 0)),
    (3, 5., 'GFG3', date(2000, 5, 3), datetime(2000, 5, 3, 12, 0))
])

df = spark.createDataFrame(rdd, schema=['a', 'b', 'c', 'd', 'e'])
```

### 3.6. Tạo DataFrame từ file CSV
```python
# Đọc CSV cơ bản
df = spark.read.csv('data.csv')

# Đọc CSV với header
df = spark.read.option('header', True).csv('data.csv')
```

### 3.7. Tạo DataFrame từ file JSON
```python
# Đọc JSON cơ bản
df = spark.read.json('data.json')

# Đọc JSON multiline
df = spark.read.option('multiline', 'true').json('data.json')
```

### 3.8. Các phương thức đọc dữ liệu khác
```python
spark.read.csv()      # Đọc từ file CSV
spark.read.json()     # Đọc từ file JSON
spark.read.parquet()  # Đọc từ file Parquet
spark.read.jdbc()     # Đọc từ cơ sở dữ liệu SQL
spark.read.text()     # Đọc từ file văn bản
```

### 3.9. Lưu dữ liệu ra tệp
```python
df.write.csv("path")      # Lưu dưới dạng CSV
df.write.parquet("path")  # Lưu dưới dạng Parquet
df.write.json("path")     # Lưu dưới dạng JSON
```

---

## 4. Các kiểu dữ liệu

### 4.1. Bảng các kiểu dữ liệu

| Data Type | Python Type | Giá trị |
|-----------|-------------|---------|
| BooleanType | bool | True/False |
| ByteType | int | -128 đến 127 |
| ShortType | int | -32728 đến 32727 |
| IntegerType | int | Số nguyên |
| LongType | long | Số nguyên lớn |
| FloatType | float | 4 byte |
| DoubleType | float | 8 byte |
| DecimalType | decimal.Decimal | Số thập phân chính xác |
| StringType | string | Chuỗi ký tự |
| DateType | datetime.date | Ngày |
| ArrayType | list, tuple, array | Mảng |
| MapType | dict | Dictionary |
| StructType | list, tuple | Cấu trúc phức tạp |

### 4.2. Định nghĩa cấu trúc cho DataFrame
```python
from pyspark.sql.types import StructType, StructField, LongType, StringType, ByteType

data = [
    (1, 'Nguyen Van Anh', 20),
    (2, 'Le Thi Binh', 19),
    (3, 'Ho Van Danh', 22)
]

schema = StructType([
    StructField('id', LongType(), False),      # False = không cho phép null
    StructField('name', StringType(), True),   # True = cho phép null
    StructField('age', ByteType(), True)
])

df = spark.createDataFrame(data, schema)
df.printSchema()
df.show()
```

### 4.3. Tự xác định kiểu (inferSchema)
```python
# Spark tự động xác định kiểu dữ liệu cho các cột
df = spark.read.option('inferSchema', 'true').csv('data.csv')
df.printSchema()
```

---

## 5. Thao tác với DataFrame

### 5.1. Các phương thức cơ bản

| Phương thức | Mô tả |
|-------------|-------|
| `show()` | Hiển thị 20 dòng đầu tiên của DataFrame |
| `count()` | Số dòng của DataFrame |
| `printSchema()` | Hiển thị cấu trúc DataFrame |
| `collect()` | Trả về danh sách các Row của DataFrame |
| `take(n)` | Trả về danh sách n dòng đầu của DataFrame |
| `head(n)` | Trả về n dòng đầu của DataFrame |
| `limit(n)` | Giới hạn số dòng của DataFrame |
| `distinct()` | Trả về danh sách các Row phân biệt của DataFrame |

### 5.2. Ví dụ
```python
df.show()           # Hiển thị 20 dòng đầu
df.show(5)          # Hiển thị 5 dòng đầu
df.count()          # Đếm số dòng
df.printSchema()    # In cấu trúc
df.collect()        # Lấy tất cả dữ liệu
df.take(10)         # Lấy 10 dòng đầu
df.head(5)          # Lấy 5 dòng đầu
df.limit(100)       # Giới hạn 100 dòng
df.distinct()       # Lấy các dòng không trùng lặp
```

---

## 6. Lọc dữ liệu

### 6.1. Lọc các cột - select()
```python
# Chọn các cột name, age
df.select('name', 'age').show()

# Chọn nhiều cột
df.select('col1', 'col2', 'col3').show()
```

### 6.2. Lọc các dòng - filter()
```python
# Lọc những người có tuổi > 20
df.filter('age > 20').show()

# Hoặc sử dụng cú pháp khác
df.filter(df['age'] > 20).show()
```

### 6.3. Các điều kiện lọc

#### So sánh giá trị cột với giá trị cụ thể
```python
df.filter(df['age'] > 30)
```

#### Kết hợp nhiều điều kiện (AND, OR)
```python
# AND - cả 2 điều kiện đều đúng
df.filter((df['age'] > 30) & (df['gender'] == 'Male'))

# OR - một trong 2 điều kiện đúng
df.filter((df['age'] > 30) | (df['gender'] == 'Female'))
```

#### Kiểm tra nằm trong danh sách - isin()
```python
df.filter(df['gender'].isin(['Male', 'Female']))
```

#### Dùng hàm like với xâu ký tự
```python
df.filter(df['name'].like('John%'))   # Bắt đầu bằng 'John'
df.filter(df['name'].like('%son'))    # Kết thúc bằng 'son'
df.filter(df['name'].like('%ohn%'))   # Chứa 'ohn'
```

#### Dùng hàm between
```python
df.filter(df['age'].between(30, 40))  # Tuổi từ 30 đến 40
```

#### Kiểm tra giá trị trống (null)
```python
df.filter(df['name'].isNull())      # Lọc các dòng có name là null
df.filter(df['name'].isNotNull())   # Lọc các dòng có name không null
```

---

## 7. Làm việc với các cột

### 7.1. Sử dụng hàm col()
```python
from pyspark.sql.functions import col

# Lọc dữ liệu
df.filter(col('rating') > 3).show()

# Chọn và tính toán
df.select('userId', col('rating') + 1).show()
```

### 7.2. Thêm cột mới - withColumn()
```python
# Thêm cột age tính từ năm sinh
df.withColumn('age', 2022 - col('year')).show()
```

### 7.3. Xóa cột - drop()
```python
df.drop(col('timeStamp')).show()
df.drop('timeStamp').show()
```

### 7.4. Đổi tên cột - withColumnRenamed()
```python
df = df.withColumnRenamed('_c0', 'userId')
df = df.withColumnRenamed('old_name', 'new_name')
```

### 7.5. Tính toán cho các cột mới

#### Sử dụng các hằng (literal) - lit()
```python
from pyspark.sql.functions import lit

# Thêm cột với giá trị hằng số
df.withColumn('col1', lit(1))
df.withColumn('country', lit('Vietnam'))
```

#### Sử dụng các phép toán số học
```python
from pyspark.sql.functions import col

df_new = df.withColumn('new_col', col('old_col') * 2)
df_new = df.withColumn('total', col('price') * col('quantity'))
```

#### Sử dụng các hàm định nghĩa sẵn
```python
from pyspark.sql.functions import col, sqrt, abs, sin, cos

df_new = df.withColumn('sqrt_col', sqrt(col('old_col')))
df_new = df.withColumn('abs_col', abs(col('value')))
```

#### Sử dụng các hàm gộp
```python
from pyspark.sql.functions import sum, count, max, min, avg

df_new = df.agg(max("column_name_1"), max("column_name_2"))
```

### 7.6. Sử dụng câu lệnh điều kiện - when().otherwise()
```python
from pyspark.sql.functions import when, col

df_new = df.withColumn('newCol', 
    when(col('col1') == 'A', 1).otherwise(0)
)

# Nhiều điều kiện
df_new = df.withColumn('category',
    when(col('age') < 18, 'Minor')
    .when(col('age') < 65, 'Adult')
    .otherwise('Senior')
)
```

### 7.7. Thống kê dữ liệu các cột - describe()
```python
# Thống kê tất cả các cột số
df.describe().show()

# Thống kê cột cụ thể
df.describe(['rating']).show()
```

---

## 8. Hàm gộp (Aggregate Functions)

### 8.1. Các hàm gộp có sẵn
Các hàm được cung cấp trong module `pyspark.sql.functions`:

| Hàm | Mô tả |
|-----|-------|
| `count(col)` | Đếm dữ liệu trên 1 cột |
| `countDistinct(col)` | Đếm số dữ liệu khác nhau trên 1 cột |
| `min(col)` | Giá trị nhỏ nhất |
| `max(col)` | Giá trị lớn nhất |
| `sum(col)` | Tổng |
| `sumDistinct(col)` | Tổng các giá trị khác nhau |
| `avg(col)` | Trung bình |
| `collect_list(col)` | Trả về danh sách các giá trị của một cột |
| `collect_set(col)` | Trả về tập hợp các giá trị của một cột (không trùng) |

### 8.2. Ví dụ sử dụng
```python
from pyspark.sql.functions import count, avg, min, max, collect_list

# Đếm số người dùng, số bộ phim, trung bình rating
df.select(count('userId'), count('movieId'), avg('rating')).show()

# Cho biết những bộ phim người dùng 1 đã rating
df.filter('userId = 1').select(collect_list('movieId')).collect()
```

---

## 9. Phân nhóm (GroupBy)

### 9.1. Cú pháp cơ bản
```python
df.groupBy(col1, col2, ...)
```
- Thực hiện phân nhóm trên các cột theo thứ tự phân cấp
- Mỗi nhóm có thể thực hiện các hàm gộp

### 9.2. Ví dụ đếm số rating của mỗi người dùng
```python
df.groupBy('userId').count().show()
```

### 9.3. Tính nhiều hàm gộp trong một nhóm - agg()
```python
from pyspark.sql.functions import count, min, max, avg

# Với mỗi bộ phim cho biết số lượt rating, tb, thấp nhất, cao nhất của rating
df.groupBy('movieId').agg(
    count('rating'),
    min('rating'),
    max('rating'),
    avg('rating')
).show()
```

---

## 10. Pivot

### 10.1. Khái niệm
- Dùng để phân nhóm và tính toán theo cột của từng nhóm groupby (nhóm theo dòng)

### 10.2. Cú pháp
```python
gr.pivot(col1).agg_fun(col2)
```

### 10.3. Ví dụ: Thống kê số lượng rating từng mức của từng bộ phim
```python
df.groupBy('movieId').pivot('rating').count().show()
```

---

## 11. Cửa sổ trượt (Window Functions)

### 11.1. Khái niệm
- Tính toán một phép tổng hợp trên một cửa sổ dữ liệu cụ thể
- Cửa sổ xác định các hàng nào sẽ được truyền vào hàm này
- Một nhóm các hàng được gọi là một khung (frame)

### 11.2. Ví dụ minh họa
**Dữ liệu gốc:**
```
+------+-------+-----------+
| id   | amount| date      |
+------+-------+-----------+
| 001  | 100   | 2020-01-01|
| 002  | 200   | 2020-01-02|
| 003  | 300   | 2020-01-03|
| 004  | 400   | 2020-01-04|
| 005  | 500   | 2020-01-05|
| 006  | 600   | 2020-01-06|
+------+-------+-----------+
```

**Kết quả sau khi áp dụng cửa sổ trượt (tổng 3 ngày liên tiếp):**
```
+------+-------+----------+-------------------+
| id   | amount| date     |total_amount_3_days|
+------+-------+----------+-------------------+
| 001  | 100   |2020-01-01| 300               |
| 002  | 200   |2020-01-02| 600               |
| 003  | 300   |2020-01-03| 900               |
| 004  | 400   |2020-01-04| 1200              |
| 005  | 500   |2020-01-05| 1500              |
| 006  | 600   |2020-01-06| 1400              |
+------+-------+----------+-------------------+
```

### 11.3. Code thực hiện
```python
from pyspark.sql.functions import sum, col
from pyspark.sql.window import Window

# Khai báo window với chiều dài cửa sổ là 3 (dòng trước, dòng hiện tại, dòng sau)
window = Window.orderBy("date").rowsBetween(-1, 1)

# Tính tổng số tiền giao dịch trong cửa sổ 3 ngày liên tiếp
df.withColumn("total_amount_3_days", sum(col("amount")).over(window))
```

### 11.4. Các tham số của Window
- `orderBy()`: Sắp xếp dữ liệu theo cột
- `partitionBy()`: Phân nhóm dữ liệu
- `rowsBetween(start, end)`: Xác định phạm vi cửa sổ
  - `-1`: dòng trước
  - `0`: dòng hiện tại
  - `1`: dòng sau

---

## 12. Điền dữ liệu trống

### 12.1. Vấn đề
- Trong DataFrame có thể có dữ liệu trống (null, None)

### 12.2. Phương thức điền dữ liệu trống
```python
df.na.fill(value, [subset=None])
df.fillna(value, [subset=None])
```

### 12.3. Ví dụ
```python
# Điền dữ liệu trống trong cột rating bằng 0
df.fillna(value=0, subset=['rating']).show()

# Điền tất cả các cột số bằng 0
df.fillna(0).show()

# Điền các cột chuỗi bằng 'Unknown'
df.fillna('Unknown', subset=['name', 'address']).show()
```

---

## 13. Join DataFrame

### 13.1. Các kiểu Join
- **inner**: Chỉ lấy các dòng có khóa khớp ở cả 2 bảng
- **left**: Lấy tất cả dòng từ bảng trái, khớp với bảng phải nếu có
- **right**: Lấy tất cả dòng từ bảng phải, khớp với bảng trái nếu có
- **full**: Lấy tất cả dòng từ cả 2 bảng

### 13.2. Cú pháp
```python
df1.join(df2, joinExpression, joinType)
```
- `df1`, `df2`: Hai DataFrame cần kết nối
- `joinExpression`: Điều kiện kết nối 2 dòng
- `joinType`: Kiểu kết nối ('inner', 'left', 'right', 'full')

### 13.3. Ví dụ tạo dữ liệu
```python
# Tạo DataFrame nhân viên
emp = [
    (1, "Smith", -1, "2018", "10", "M", 3000),
    (2, "Rose", 1, "2010", "20", "M", 4000),
    (3, "Williams", 1, "2010", "10", "M", 1000),
    (4, "Jones", 2, "2005", "10", "F", 2000),
    (5, "Brown", 2, "2010", "40", "", -1),
    (6, "Brown", 2, "2010", "50", "", -1)
]
empColumns = ["emp_id", "name", "superior_emp_id", "year_joined", 
              "emp_dept_id", "gender", "salary"]
empDF = spark.createDataFrame(data=emp, schema=empColumns)

# Tạo DataFrame phòng ban
dept = [
    ("Finance", 10),
    ("Marketing", 20),
    ("Sales", 30),
    ("IT", 40)
]
deptColumns = ["dept_name", "dept_id"]
deptDF = spark.createDataFrame(data=dept, schema=deptColumns)
```

### 13.4. Thực hiện Join
```python
# Inner Join
empDF.join(deptDF, empDF.emp_dept_id == deptDF.dept_id, "inner").show(truncate=False)

# Left Join
empDF.join(deptDF, empDF.emp_dept_id == deptDF.dept_id, "left").show(truncate=False)

# Right Join
empDF.join(deptDF, empDF.emp_dept_id == deptDF.dept_id, "right").show(truncate=False)

# Full Outer Join
empDF.join(deptDF, empDF.emp_dept_id == deptDF.dept_id, "full").show(truncate=False)
```

### 13.5. Broadcast Join
- Khi bảng nhỏ đủ nhỏ để vừa vặn trong bộ nhớ của một nút worker
- Spark sẽ broadcast bảng nhỏ này đến tất cả các Executors
- Tối ưu hiệu suất cho các phép join với bảng nhỏ

---

## 14. DataFrame và SQL

### 14.1. Khái niệm
- Spark có thể sử dụng câu lệnh SQL để truy vấn, tính toán thống kê, kết nối dữ liệu như trên các DataFrame như các bảng trong CSDL

### 14.2. Các bước sử dụng SQL trong Spark

#### Bước 1: Tạo bảng tạm từ DataFrame
```python
df.createOrReplaceTempView("TableName")
```

#### Bước 2: Sử dụng câu lệnh SQL
```python
result_df = spark.sql("SQL statement")
```
- Trả về DataFrame là kết quả câu lệnh SQL

### 14.3. Ví dụ hoàn chỉnh
```python
# Nạp dữ liệu từ u.data
df = spark.read.option('inferSchema', 'true')\
    .option("delimiter", "\t")\
    .csv("f:/spark/u.data")

df = df.withColumnRenamed('_c0', 'userId')\
    .withColumnRenamed('_c1', 'movieId')\
    .withColumnRenamed('_c2', 'rating')\
    .withColumnRenamed('_c3', 'timeStamp')

# Tạo bảng tạm
df.createOrReplaceTempView("Rating")

# Sử dụng SQL
df2 = spark.sql("SELECT userId, movieId FROM Rating WHERE rating > 2")
df2.show()
```

### 14.4. Các câu lệnh SQL phổ biến
```python
# SELECT với điều kiện
spark.sql("SELECT * FROM Rating WHERE rating >= 4")

# GROUP BY
spark.sql("SELECT userId, COUNT(*) as count FROM Rating GROUP BY userId")

# JOIN
spark.sql("""
    SELECT r.userId, r.movieId, m.title 
    FROM Rating r 
    JOIN Movie m ON r.movieId = m.movieId
""")

# ORDER BY
spark.sql("SELECT * FROM Rating ORDER BY rating DESC")

# Aggregate functions
spark.sql("SELECT AVG(rating), MAX(rating), MIN(rating) FROM Rating")
```

---

## 15. Hàm do người dùng định nghĩa (UDF)

### 15.1. Khái niệm
- Các hàm do người dùng định nghĩa muốn dùng được trong DataFrame phải được khai báo thông qua hàm `udf`

### 15.2. Cú pháp
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

# Định nghĩa hàm Python
def my_function(input):
    # xử lý
    return result

# Đăng ký UDF
udf_my_function = udf(my_function, ReturnType())

# Sử dụng UDF
df = df.withColumn("new_col", udf_my_function(col("input_col")))
```

### 15.3. Ví dụ: Tách chuỗi thành danh sách từ
```python
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType

def my_udf(s):
    return s.split()

udf_my_udf = udf(my_udf, ArrayType(StringType()))

df = df.withColumn("word_list", udf_my_udf(col("text_col")))
```

---

## 16. Làm việc với Array

### 16.1. Ghép các cột của DataFrame

#### Sử dụng concat()
```python
from pyspark.sql.functions import concat, concat_ws

# Ghép 2 cột
df = df.withColumn("full_name", concat("first_name", "last_name"))

# Ghép với dấu phân cách
df = df.withColumn("location", concat_ws(", ", "address", "city", "state"))
```

#### Sử dụng array()
```python
from pyspark.sql.functions import array

df = df.withColumn('data', array('name', 'age'))
```

**Kết quả:**
```
+-----+---+-----------+
| name|age| data      |
+-----+---+-----------+
|John | 25|[John, 25] |
|Alice| 30|[Alice, 30]|
|Bob  | 20|[Bob, 20]  |
+-----+---+-----------+
```

### 16.2. Các hàm trên Array

| Hàm | Mô tả |
|-----|-------|
| `array(*cols)` | Trả về mảng mới chứa tất cả giá trị trong các cột đầu vào |
| `array_contains(col, value)` | True nếu mảng chứa giá trị được chỉ định |
| `concat(*cols)` | Ghép các mảng lại với nhau |
| `element_at(col, index)` | Trả về giá trị tại vị trí chỉ định |
| `explode(col)` | Chuyển mảng thành nhiều hàng |
| `array_join(col, delimiter)` | Nối các phần tử thành chuỗi |
| `array_max(col)` | Giá trị lớn nhất trong mảng |
| `array_min(col)` | Giá trị nhỏ nhất trong mảng |
| `array_position(col, value)` | Vị trí đầu tiên của giá trị |
| `array_remove(col, element)` | Loại bỏ phần tử khỏi mảng |
| `array_repeat(col, count)` | Lặp lại mảng count lần |
| `array_sort(col)` | Sắp xếp mảng tăng dần |

### 16.3. Hàm tập hợp trên Array
```python
from pyspark.sql.functions import array_intersect, array_union, array_except

# Tạo dataframe ví dụ
data = [([1, 2, 3], [2, 3, 4, 5])]
df = spark.createDataFrame(data, ["col1", "col2"])

# Giao của 2 mảng
df = df.withColumn("col3", array_intersect(df.col1, df.col2))

# Hợp của 2 mảng
df = df.withColumn("col4", array_union(df.col1, df.col2))

# Hiệu của 2 mảng (col1 - col2)
df = df.withColumn("col5", array_except(df.col1, df.col2))

df.show()
```

**Kết quả:**
```
+---------+-------------+------+---------------+------+
| col1    | col2        | col3 | col4          | col5 |
+---------+-------------+------+---------------+------+
|[1, 2, 3]|[2, 3, 4, 5] |[2, 3]|[1, 2, 3, 4, 5]| [1]  |
+---------+-------------+------+---------------+------+
```

### 16.4. Chuyển dữ liệu cột thành dòng - explode()
```python
from pyspark.sql.functions import explode

df_exploded = df.select("col1", explode("col1").alias("col2"))
df_exploded.show()
```

**Dữ liệu gốc:**
```
+------------+
| col1       |
+------------+
|[1, 2, 3]   |
|[4, 5]      |
|[6, 7, 8, 9]|
+------------+
```

**Sau khi explode:**
```
+------------+----+
| col1       |col2|
+------------+----+
|[1, 2, 3]   | 1  |
|[1, 2, 3]   | 2  |
|[1, 2, 3]   | 3  |
|[4, 5]      | 4  |
|[4, 5]      | 5  |
|[6, 7, 8, 9]| 6  |
|[6, 7, 8, 9]| 7  |
|[6, 7, 8, 9]| 8  |
|[6, 7, 8, 9]| 9  |
+------------+----+
```

---

## 17. Ví dụ thực tế: MovieLens

### 17.1. Giới thiệu Dataset MovieLen100K

#### u.data
- Chứa 100000 ratings của 943 users trên 1682 movies
- Cấu trúc (các trường cách nhau một tab): `user id | item id | rating | timestamp`

#### u.item
- Thông tin về các bộ phim và phân loại trong 19 loại
- Cấu trúc: `movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western`
- Trong các cột phân loại phim (19 cột cuối): 1 = thuộc loại đó, 0 = không thuộc

#### u.genre
- Danh sách các phân loại phim

#### u.user
- Thông tin về users
- Cấu trúc: `user id | age | gender | occupation | zip code`

#### u.occupation
- Danh sách các nghề nghiệp của users

### 17.2. Bài toán: Cho biết thể loại phim nào có trung bình rating cao nhất

### 17.3. Nạp dữ liệu MovieLen
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Nạp dữ liệu rating
dfRating = spark.read.option('inferSchema', 'true')\
    .option("delimiter", "\t").csv("u.data")

dfRating = dfRating.withColumnRenamed('_c0', 'userId')\
    .withColumnRenamed('_c1', 'movieId')\
    .withColumnRenamed('_c2', 'rating')\
    .withColumnRenamed('_c3', 'timeStamp')

# Nạp dữ liệu phim
dfMovie = spark.read.options(inferSchema=True, delimiter='|').csv('u.item')

# Đổi tên cột cho dfMovie
schemaMovie = ['movieId', 'movie_title', 'release_date', 'video_release_date', 
               'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
               'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 
               'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery', 
               'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']

for i in range(len(schemaMovie)):
    dfMovie = dfMovie.withColumnRenamed('_c' + str(i), schemaMovie[i])

dfMovie.printSchema()
```

---

### 17.4. Phương án 1: Chuyển dữ liệu thể loại phim thành dòng

#### Bước 1: Chuyển từng cột thể loại thành dòng
```python
from pyspark.sql.functions import lit, col

# Chuyển thể loại phim thành dòng
genres = schemaMovie[5:]  # Lấy danh sách các thể loại

dfGenre = dfMovie.select(['movieId', genres[0]])\
    .filter(col(genres[0]) == 1)\
    .drop(genres[0])\
    .withColumn('genre', lit(genres[0]))

for i in range(1, len(genres)):
    dftmp = dfMovie.select(['movieId', genres[i]])\
        .filter(col(genres[i]) == 1)\
        .drop(genres[i])\
        .withColumn('genre', lit(genres[i]))
    dfGenre = dfGenre.union(dftmp)

dfGenre.show()
```

#### Bước 2: Join với bảng rating và tính trung bình
```python
df = dfRating.join(dfGenre, 'movieId', 'inner').select('rating', 'genre')
df.show()

# Phân nhóm theo thể loại phim và tính trung bình rating
df1 = df.groupBy('genre').avg('rating')
df1.show()
```

#### Bước 3: Sắp xếp và tìm thể loại có rating cao nhất
```python
df1 = df1.orderBy('avg(rating)', ascending=False)
df1.show()

max_avg_rating = df1.first()['avg(rating)']
max_genre = df1.first()['genre']
print(max_avg_rating, max_genre)
# Kết quả: 3.9215233698788228 Film_Noir
```

---

### 17.5. Phương án 2: Tính theo cột

#### Bước 1: Kết nối dữ liệu
```python
dfRating1 = dfRating.select(['userId', 'movieId', 'rating'])
dfMovie1 = dfMovie.select([schemaMovie[0]] + schemaMovie[5:])

df1 = dfRating1.join(dfMovie1, 'movieId', 'inner').drop('dfMovie1.movieId')
df1.select('userId', 'movieId', 'rating', 'unknown', 'Action').show()
```

#### Bước 2: Tính tổng và đếm số rating các thể loại phim
```python
from pyspark.sql.functions import sum, expr, greatest, col

cols = schemaMovie[5:]

# Tạo biểu thức tính tổng số phim mỗi thể loại
sumExp = [sum(x).alias('sum_' + x) for x in cols]

# Tạo biểu thức tính tổng rating mỗi thể loại
sumRating = [sum(expr('rating * ' + x)).alias('srating_' + x) for x in cols]

df2 = df1.agg(*(sumExp + sumRating))
df2.show()
```

#### Bước 3: Tính trung bình rating các thể loại phim
```python
avgRating = [expr('srating_' + x + '/ sum_' + x).alias('avg_' + x) for x in cols]
df3 = df2.select(avgRating)
df3.show()
```

#### Bước 4: Tìm thể loại có rating cao nhất
```python
from pyspark.sql.functions import greatest, col

max_avg_rating = df3.select(greatest(*df3.columns).alias("max_val")).first()["max_val"]
max_col_name = df3.columns[df3.first().index(max_avg_rating)]
print(max_avg_rating, max_col_name)
# Kết quả: 3.9215233698788228 avg_Film_Noir
```

---

### 17.6. Phương án 3: Sử dụng UDF và explode

#### Bước 1: Gộp các cột thể loại phim thành danh sách
```python
from pyspark.sql.functions import array

genres = schemaMovie[5:]
df = dfMovie.withColumn('genres', array(genres)).select('movieId', 'genres')
df.show()
```

#### Bước 2: Tạo UDF chuyển đổi danh sách 0/1 thành tên thể loại
```python
from pyspark.sql.functions import udf, explode
from pyspark.sql.types import ArrayType, StringType

def convertGenre(l):
    genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 
              'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
              'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
              'Sci_Fi', 'Thriller', 'War', 'Western']
    list_genres = []
    for i in range(len(l)):
        if l[i] == 1:
            list_genres.append(genres[i])
    return list_genres

convert_udf = udf(convertGenre, ArrayType(StringType()))

dfGenre = df.withColumn('genre_list', convert_udf(col('genres'))).drop('genres')
dfGenre = dfGenre.select('movieId', explode('genre_list').alias('genre'))
dfGenre.show()
```

#### Bước 3: Join và tính kết quả
```python
from pyspark.sql.functions import max

dfGenreRating = dfRating.select('userId', 'movieId', 'rating')\
    .join(dfGenre, 'movieId', 'inner')\
    .drop('movieId', 'userId')

dfGrGenre = dfGenreRating.groupby('genre').avg('rating')

maxAvgRating = dfGrGenre.agg(max('avg(rating)')).collect()[0]['max(avg(rating))']
dfMaxAvgRating = dfGrGenre.filter(col('avg(rating)') == maxAvgRating)
dfMaxAvgRating.show()
```

---

## 18. Tổng kết

1. **DataFrame** là công cụ để xử lý dữ liệu có cấu trúc của Spark
2. **DataFrame của Spark** có khả năng phân tán nên có thể xử lý lượng dữ liệu lớn
3. **DataFrame của Spark** có thể nạp dữ liệu từ nhiều nguồn khác nhau (CSV, JSON, Parquet, JDBC, ...)
4. **SparkSQL** cho phép sử dụng câu lệnh SQL quen thuộc để truy vấn dữ liệu
5. **Catalyst Optimizer** giúp tối ưu hóa các truy vấn tự động
6. **UDF** cho phép mở rộng chức năng với các hàm tùy chỉnh
