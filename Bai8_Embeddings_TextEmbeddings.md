# BÀI 8: EMBEDDINGS VÀ TEXT EMBEDDINGS

## PHẦN 1: EMBEDDINGS - BIỂU DIỄN DỮ LIỆU BẰNG VECTOR

### 1.1. Tại sao cần biểu diễn dữ liệu bằng vector?

Máy tính không thể hiểu trực tiếp các dữ liệu phi cấu trúc như:
- Văn bản: "Tôi yêu lập trình"
- Hình ảnh: một bức ảnh mèo
- Âm thanh: một đoạn nhạc

**Giải pháp**: Chuyển đổi chúng thành vector số (mảng các số thực) để máy tính có thể:
- Lưu trữ hiệu quả
- Tính toán nhanh chóng
- So sánh độ tương tự

**Ví dụ đơn giản**:
```
"Mèo" → [0.2, 0.8, 0.1, 0.5]
"Chó" → [0.3, 0.7, 0.2, 0.4]
"Ô tô" → [0.9, 0.1, 0.8, 0.2]
```

Nhận xét: Vector của "Mèo" và "Chó" gần nhau hơn so với "Ô tô" vì chúng cùng là động vật.

### 1.2. Embeddings là gì?

**Định nghĩa**: Embedding là kỹ thuật chuyển đổi một đối tượng (từ, câu, văn bản, hình ảnh, âm thanh...) thành một vector đặc trưng trong không gian nhiều chiều.

**Đặc điểm quan trọng**:
- Các đối tượng tương tự nhau sẽ có vector gần nhau trong không gian
- Giảm chiều dữ liệu (từ hàng triệu từ → vector vài trăm chiều)
- Bảo toàn ý nghĩa ngữ nghĩa

### 1.3. Độ đo khoảng cách giữa các vector

Để đo "độ tương tự" giữa hai đối tượng, ta tính khoảng cách giữa hai vector của chúng:

#### a) Khoảng cách Euclid (Euclidean Distance)

**Công thức**:

$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n}(q_i - p_i)^2}$$

Hay viết đầy đủ:

$$d(\mathbf{p}, \mathbf{q}) = \sqrt{(q_1-p_1)^2 + (q_2-p_2)^2 + \cdots + (q_n-p_n)^2}$$

**Minh họa**:
```
      Queen [0.3, 0.9]
        ↗
       /
      /  d(King, Queen)
     /
    /
   ↗ King [0.5, 0.7]
  /
 /
O────────────────→
```

**Giải thích**:
- Đo khoảng cách thẳng giữa hai điểm trong không gian n chiều
- Khoảng cách càng **nhỏ** → hai vector càng **giống nhau**
- Khoảng cách = 0 → hai vector giống hệt nhau
- Phụ thuộc vào độ lớn (magnitude) của vector

**Ví dụ tính toán**:
```
p = [1, 2, 3]
q = [4, 5, 6]

d(p,q) = √[(4-1)² + (5-2)² + (6-3)²]
       = √[9 + 9 + 9]
       = √27
       = 5.196
```

---

#### b) Tích vô hướng (Dot Product)

**Công thức**:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \times b_i$$

Hay viết đầy đủ:

$$\mathbf{a} \cdot \mathbf{b} = a_1 \times b_1 + a_2 \times b_2 + \cdots + a_n \times b_n$$

**Minh họa**:
```
      Queen
        ↗
       /
      /  ← Tích vô hướng phụ thuộc
     /      vào độ dài và góc
    /
   ↗ King
  /
 /
O────────────────→
```

**Giải thích**:
- Nhân từng cặp phần tử tương ứng rồi cộng lại
- Giá trị càng **lớn** → hai vector càng **giống nhau**
- Giá trị dương → cùng hướng
- Giá trị âm → ngược hướng
- **Hạn chế**: Phụ thuộc vào độ lớn của vector

**Ví dụ tính toán**:
```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = (1×4) + (2×5) + (3×6)
      = 4 + 10 + 18
      = 32
```

---

#### c) Độ tương tự Cosine (Cosine Similarity) ⭐

**Công thức**:

$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**Trong đó**:
- $\mathbf{A} \cdot \mathbf{B}$ : Tích vô hướng của A và B
- $\|\mathbf{A}\|$ : Độ dài (norm) của vector A = $\sqrt{A_1^2 + A_2^2 + \cdots + A_n^2}$
- $\|\mathbf{B}\|$ : Độ dài (norm) của vector B = $\sqrt{B_1^2 + B_2^2 + \cdots + B_n^2}$
- $\theta$ : Góc giữa hai vector

**Minh họa**:
```
      Queen [0.3, 0.9]
        ↗
       / θ ← Góc giữa 2 vector
      /      (không phụ thuộc độ dài)
     /
    /
   ↗ King [0.5, 0.7]
  /
 /
O────────────────→

cos(θ) = 1   → Cùng hướng (giống nhau)
cos(θ) = 0   → Vuông góc (không liên quan)
cos(θ) = -1  → Ngược hướng (trái ngược)
```

**Giải thích**:
- Đo góc giữa hai vector (không phụ thuộc độ lớn)
- Giá trị từ **-1 đến 1**:
  - **1**: Hoàn toàn giống nhau (góc 0°)
  - **0**: Không liên quan (góc 90°)
  - **-1**: Hoàn toàn trái ngược (góc 180°)
- **Phổ biến nhất** trong NLP vì không bị ảnh hưởng bởi độ dài văn bản

**Ví dụ tính toán chi tiết**:

Cho $\mathbf{A} = [1, 2, 3]$ và $\mathbf{B} = [4, 5, 6]$

**Bước 1**: Tính tích vô hướng
$$\mathbf{A} \cdot \mathbf{B} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32$$

**Bước 2**: Tính độ dài vector A
$$\|\mathbf{A}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{1 + 4 + 9} = \sqrt{14} = 3.742$$

**Bước 3**: Tính độ dài vector B
$$\|\mathbf{B}\| = \sqrt{4^2 + 5^2 + 6^2} = \sqrt{16 + 25 + 36} = \sqrt{77} = 8.775$$

**Bước 4**: Tính cosine similarity
$$\cos(\theta) = \frac{32}{3.742 \times 8.775} = \frac{32}{32.835} = 0.975$$

**Kết luận**: Hai vector rất giống nhau! (0.975 ≈ 1)

---

**Ví dụ so sánh từ**:

Cho:
- Vector "King" = $[0.5, 0.7]$
- Vector "Queen" = $[0.3, 0.9]$
- Vector "Apple" = $[0.9, 0.1]$

**So sánh "King" vs "Queen"**:
- Tử số: $(0.5 \times 0.3) + (0.7 \times 0.9) = 0.15 + 0.63 = 0.78$
- Mẫu số: $\sqrt{0.5^2+0.7^2} \times \sqrt{0.3^2+0.9^2} = 0.860 \times 0.949 = 0.816$
- Kết quả: $\cos(\theta) = 0.78 / 0.816 = 0.955$ → **Rất giống nhau!**

**So sánh "King" vs "Apple"**:
- Tử số: $(0.5 \times 0.9) + (0.7 \times 0.1) = 0.45 + 0.07 = 0.52$
- Mẫu số: $0.860 \times 0.905 = 0.778$
- Kết quả: $\cos(\theta) = 0.52 / 0.778 = 0.668$ → **Hơi liên quan**

**Kết luận**: "King" gần "Queen" hơn là "Apple"

---

**So sánh 3 độ đo**:

| Độ đo | Công thức | Giá trị | Ưu điểm | Nhược điểm | Dùng khi nào |
|-------|-----------|---------|---------|------------|--------------|
| **Euclid** | $d = \sqrt{\sum(q_i-p_i)^2}$ | 0 → ∞ (nhỏ = giống) | Trực quan, dễ hiểu | Phụ thuộc độ lớn | So sánh vector cùng scale |
| **Dot Product** | $a \cdot b = \sum a_i b_i$ | -∞ → ∞ (lớn = giống) | Tính nhanh | Phụ thuộc độ lớn | Tính toán trung gian |
| **Cosine** | $\cos\theta = \frac{A \cdot B}{\|A\|\|B\|}$ | -1 → 1 (1 = giống) | Không phụ thuộc độ lớn | Tính chậm hơn | **NLP, Text Mining** |

---

## PHẦN 2: CÁC KỸ THUẬT EMBEDDINGS

### 2.1. Tổng quan các loại Embeddings

| Loại | Kỹ thuật | Ứng dụng |
|------|----------|----------|
| **Word Embeddings** | Word2Vec, GloVe, FastText | Biểu diễn từ đơn |
| **Sentence Embeddings** | USE, BERT, InferSent | Biểu diễn câu |
| **Document Embeddings** | Doc2Vec, LDA | Biểu diễn văn bản dài |
| **Image Embeddings** | CNN, Siamese Networks | Biểu diễn hình ảnh |
| **Audio Embeddings** | MFCC, Spectrograms | Biểu diễn âm thanh |
| **Graph Embeddings** | Node2Vec, GraphSAGE | Biểu diễn đồ thị |
| **Cross-modal** | CLIP, VisualBERT | Kết hợp nhiều loại dữ liệu |
| **Model Embeddings** | Model2vec | Biểu diễn toàn bộ mô hình ML |

---

## PHẦN 3: TEXT EMBEDDINGS - BIỂU DIỄN VĂN BẢN

### 3.1. Các bài toán cơ bản của NLP (Natural Language Processing)

Trước khi làm embeddings, cần xử lý văn bản qua các bước:

#### a) Tokenization (Tách từ)
Chia văn bản thành các đơn vị nhỏ (từ, ký tự, subword)

**Ví dụ**:
```
Input: "Tôi yêu lập trình Python"
Output: ["Tôi", "yêu", "lập trình", "Python"]
```

**Thư viện**: `nltk`, `spaCy`, `underthesea` (tiếng Việt)

#### b) Part-of-Speech Tagging (Gán nhãn từ loại)
Xác định từ loại của mỗi từ (danh từ, động từ, tính từ...)

**Ví dụ**:
```
"Tôi yêu lập trình"
→ Tôi/PRON yêu/VERB lập trình/NOUN
```

#### c) Named Entity Recognition - NER (Nhận dạng thực thể)
Tìm các thực thể như tên người, địa danh, tổ chức...

**Ví dụ**:
```
"Nguyễn Văn A sống ở Hà Nội"
→ Nguyễn Văn A [PERSON], Hà Nội [LOCATION]
```

#### d) Text Normalization (Chuẩn hóa văn bản)
- Chuyển về chữ thường: "Python" → "python"
- Loại bỏ dấu câu: "Hello!" → "Hello"
- Loại bỏ số: "Python3" → "Python"
- Chuẩn hóa Unicode: "café" → "cafe"

#### e) Stopword Removal (Loại bỏ từ dừng)
Loại bỏ các từ phổ biến không mang nhiều ý nghĩa: "là", "của", "và", "the", "a", "an"...

**Ví dụ**:
```
Input: "Tôi là một lập trình viên"
Output: "lập trình viên"
```

---

### 3.2. Word Embeddings - Biểu diễn từ đơn

#### A. Word2Vec (Google, 2013)

**Ý tưởng**: "Một từ được định nghĩa bởi ngữ cảnh xung quanh nó"

**Hai kiến trúc**:

1. **CBOW (Continuous Bag of Words)**
   - Dự đoán từ trung tâm từ các từ xung quanh
   - Nhanh hơn, phù hợp với tập dữ liệu lớn
   
   ```
   Ngữ cảnh: ["Tôi", "yêu", "___", "trình", "Python"]
   → Dự đoán: "lập"
   ```

2. **Skip-gram**
   - Dự đoán các từ xung quanh từ từ trung tâm
   - Chính xác hơn với dữ liệu nhỏ
   
   ```
   Từ trung tâm: "lập"
   → Dự đoán: ["Tôi", "yêu", "trình", "Python"]
   ```

**Đặc điểm**:
- Vector thường 100-300 chiều
- Có thể thực hiện phép toán ngữ nghĩa:
  ```
  Vector("Vua") - Vector("Nam") + Vector("Nữ") ≈ Vector("Nữ hoàng")
  ```

**Code ví dụ (Python)**:
```python
from gensim.models import Word2Vec

sentences = [["tôi", "yêu", "python"], 
             ["python", "là", "ngôn ngữ", "tuyệt vời"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Lấy vector của từ
vector = model.wv['python']
# Tìm từ tương tự
similar = model.wv.most_similar('python', topn=5)
```

#### B. GloVe (Global Vectors, Stanford, 2014)

**Ý tưởng**: Kết hợp thống kê toàn cục (đếm tần suất xuất hiện cùng nhau) và ngữ cảnh cục bộ

**Đặc điểm**:
- Huấn luyện trên ma trận đồng xuất hiện (co-occurrence matrix)
- Hiệu quả với dữ liệu lớn
- Vector pre-trained phổ biến: 50d, 100d, 200d, 300d

**Ưu điểm so với Word2Vec**:
- Nhanh hơn khi huấn luyện
- Sử dụng thông tin thống kê toàn cục

**Code ví dụ**:
```python
import gensim.downloader as api

# Tải mô hình pre-trained
glove_model = api.load("glove-wiki-gigaword-100")

# Sử dụng
vector = glove_model['computer']
similar = glove_model.most_similar('computer', topn=5)
```

#### C. FastText (Facebook, 2016)

**Ý tưởng**: Biểu diễn từ dựa trên các **subword** (n-gram ký tự)

**Ví dụ**:
```
Từ "programming" với n=3:
→ Subwords: <pr, pro, rog, ogr, gra, ram, amm, mmi, min, ing, ng>
→ Vector("programming") = tổng vector của các subword
```

**Ưu điểm**:
- Xử lý được từ **out-of-vocabulary** (từ chưa gặp)
- Tốt với ngôn ngữ có hình thái phức tạp (tiếng Đức, tiếng Thổ Nhĩ Kỳ)
- Xử lý tốt lỗi chính tả

**Ví dụ**:
```
Đã học: "running", "runner"
Gặp từ mới: "runs"
→ FastText vẫn tạo được vector hợp lý vì có subword chung
```

**Code ví dụ**:
```python
from gensim.models import FastText

sentences = [["tôi", "yêu", "python"], 
             ["python", "là", "tuyệt", "vời"]]
model = FastText(sentences, vector_size=100, window=5, min_count=1)

# Xử lý từ chưa gặp
vector_new = model.wv['pythonnn']  # Vẫn có vector!
```

---

### 3.3. Sentence Embeddings - Biểu diễn câu

Word embeddings chỉ biểu diễn từ đơn. Để biểu diễn cả câu, cần các kỹ thuật khác:

#### A. Universal Sentence Encoder - USE (Google, 2018)

**Ý tưởng**: Mã hóa toàn bộ câu thành một vector duy nhất

**Đặc điểm**:
- Vector 512 chiều
- Huấn luyện trên nhiều tác vụ (dịch máy, Q&A, phân loại...)
- Nhanh và hiệu quả

**Ứng dụng**:
- Tìm kiếm câu tương tự
- Phân loại văn bản
- Hệ thống hỏi đáp

**Code ví dụ**:
```python
import tensorflow_hub as hub

# Tải mô hình
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Mã hóa câu
sentences = ["I love programming", "I enjoy coding"]
embeddings = embed(sentences)

# Tính độ tương tự
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings)
```

#### B. BERT (Bidirectional Encoder Representations from Transformers, Google, 2018)

**Ý tưởng**: Học ngữ cảnh **hai chiều** (xem cả từ trước và sau)

**Đặc điểm**:
- Kiến trúc Transformer
- Pre-trained trên corpus khổng lồ
- Fine-tune cho từng tác vụ cụ thể

**Ưu điểm**:
- Hiểu ngữ cảnh sâu sắc
- State-of-the-art cho nhiều tác vụ NLP
- Xử lý tốt từ đa nghĩa

**Ví dụ từ đa nghĩa**:
```
"Tôi đi ngân hàng" (bank = nơi gửi tiền)
"Tôi ngồi bên bờ sông" (bank = bờ sông)
→ BERT tạo vector khác nhau cho "bank" trong hai ngữ cảnh
```

**Code ví dụ**:
```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "I love programming"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# Lấy embedding của câu (CLS token)
sentence_embedding = outputs.last_hidden_state[:, 0, :]
```

#### C. InferSent (Facebook, 2017)

**Ý tưởng**: Huấn luyện trên tác vụ **Natural Language Inference** (suy luận ngôn ngữ tự nhiên)

**Đặc điểm**:
- Học từ cặp câu có quan hệ: entailment, contradiction, neutral
- Vector 4096 chiều
- Tốt cho tác vụ so sánh câu

**Ví dụ**:
```
Câu 1: "Một người đàn ông đang chơi guitar"
Câu 2: "Một nhạc sĩ đang biểu diễn"
→ Quan hệ: Entailment (câu 2 suy ra từ câu 1)
```

---

### 3.4. Document Embeddings - Biểu diễn văn bản dài

#### A. Doc2Vec (Paragraph Vector, 2014)

**Ý tưởng**: Mở rộng Word2Vec để biểu diễn cả đoạn văn/tài liệu

**Hai kiến trúc**:
1. **PV-DM** (Distributed Memory): Giống CBOW, thêm vector tài liệu
2. **PV-DBOW** (Distributed Bag of Words): Giống Skip-gram

**Ứng dụng**:
- Phân loại tài liệu
- Tìm tài liệu tương tự
- Tóm tắt văn bản

**Code ví dụ**:
```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [
    TaggedDocument(words=['tôi', 'yêu', 'python'], tags=['doc1']),
    TaggedDocument(words=['python', 'tuyệt', 'vời'], tags=['doc2'])
]

model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, epochs=20)

# Lấy vector của tài liệu
doc_vector = model.dv['doc1']

# Tìm tài liệu tương tự
similar_docs = model.dv.most_similar('doc1')
```

#### B. LDA (Latent Dirichlet Allocation)

**Ý tưởng**: Mô hình chủ đề - mỗi tài liệu là hỗn hợp của nhiều chủ đề

**Đặc điểm**:
- Unsupervised learning
- Mỗi chủ đề là phân phối xác suất trên các từ
- Mỗi tài liệu là phân phối xác suất trên các chủ đề

**Ví dụ**:
```
Chủ đề 1 (Thể thao): bóng đá (0.3), cầu thủ (0.2), trận đấu (0.15)...
Chủ đề 2 (Công nghệ): AI (0.25), máy tính (0.2), lập trình (0.18)...

Tài liệu A: 70% Chủ đề 1 + 30% Chủ đề 2
```

**Code ví dụ**:
```python
from gensim import corpora
from gensim.models import LdaModel

# Chuẩn bị dữ liệu
documents = [
    ['bóng', 'đá', 'cầu', 'thủ'],
    ['AI', 'máy', 'tính', 'lập', 'trình']
]

dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Huấn luyện LDA
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Xem chủ đề
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")
```

---

## PHẦN 4: CÁC KỸ THUẬT EMBEDDINGS KHÁC

### 4.1. Image Embeddings

**CNN (Convolutional Neural Networks)**:
- Trích xuất đặc trưng từ hình ảnh
- Các layer cuối tạo vector embedding
- Ứng dụng: tìm kiếm hình ảnh tương tự, phân loại

**Siamese Networks**:
- Học độ tương tự giữa hai hình ảnh
- Ứng dụng: nhận diện khuôn mặt, xác thực chữ ký

### 4.2. Audio Embeddings

**MFCC (Mel-Frequency Cepstral Coefficients)**:
- Biểu diễn đặc trưng âm thanh
- Ứng dụng: nhận dạng giọng nói, phân loại âm nhạc

**Audio Spectrograms**:
- Chuyển âm thanh thành hình ảnh phổ
- Dùng CNN để trích xuất embedding

### 4.3. Graph Embeddings

**Node2Vec**:
- Biểu diễn các node trong đồ thị
- Ứng dụng: mạng xã hội, hệ thống gợi ý

**GraphSAGE**:
- Học embedding từ cấu trúc đồ thị và thuộc tính node
- Có thể xử lý đồ thị lớn

### 4.4. Cross-modal Embeddings

**CLIP (Contrastive Language-Image Pre-training, OpenAI)**:
- Kết nối văn bản và hình ảnh trong cùng không gian embedding
- Ứng dụng: tìm kiếm hình ảnh bằng văn bản, tạo caption

**VisualBERT**:
- Kết hợp BERT với visual features
- Ứng dụng: VQA (Visual Question Answering)

### 4.5. Model2Vec

**Ý tưởng**: Biểu diễn toàn bộ mô hình machine learning thành vector

**Ứng dụng**:
- So sánh các mô hình
- Tìm mô hình tương tự
- Meta-learning

---

## PHẦN 5: THỰC HÀNH VÀ CÔNG CỤ

### 5.1. Thư viện Python phổ biến

| Thư viện | Mục đích | Cài đặt |
|----------|----------|---------|
| **gensim** | Word2Vec, Doc2Vec, FastText | `pip install gensim` |
| **transformers** | BERT, GPT, T5... | `pip install transformers` |
| **sentence-transformers** | Sentence embeddings | `pip install sentence-transformers` |
| **tensorflow-hub** | USE, BERT... | `pip install tensorflow-hub` |
| **spaCy** | NLP preprocessing | `pip install spacy` |
| **nltk** | NLP toolkit | `pip install nltk` |
| **underthesea** | NLP tiếng Việt | `pip install underthesea` |

### 5.2. Workflow thực hành

```
1. Thu thập dữ liệu văn bản
   ↓
2. Tiền xử lý (tokenization, normalization, stopword removal)
   ↓
3. Chọn kỹ thuật embedding phù hợp
   ↓
4. Huấn luyện hoặc load pre-trained model
   ↓
5. Tạo embeddings cho dữ liệu
   ↓
6. Áp dụng vào tác vụ cụ thể (phân loại, tìm kiếm, clustering...)
   ↓
7. Đánh giá và tối ưu
```

### 5.3. Lựa chọn kỹ thuật embedding

| Tác vụ | Kỹ thuật đề xuất |
|--------|------------------|
| Phân tích từ đơn | Word2Vec, GloVe, FastText |
| Phân loại văn bản ngắn | BERT, USE |
| Tìm kiếm câu tương tự | USE, Sentence-BERT |
| Phân tích chủ đề | LDA, Doc2Vec |
| Xử lý ngôn ngữ hình thái phức tạp | FastText |
| Từ chưa gặp (OOV) | FastText, BERT |
| Cần tốc độ cao | Word2Vec (CBOW), GloVe |
| Cần độ chính xác cao | BERT, GPT |

---

## PHẦN 6: BÀI TẬP THỰC HÀNH

### Bài 1: Word Embeddings cơ bản
1. Cài đặt gensim
2. Huấn luyện Word2Vec trên corpus tiếng Việt
3. Tìm các từ tương tự với "máy tính"
4. Thực hiện phép toán: "vua" - "nam" + "nữ"

### Bài 2: Sentence Similarity
1. Sử dụng USE hoặc Sentence-BERT
2. Tính độ tương tự giữa các câu
3. Xây dựng hệ thống tìm câu hỏi tương tự

### Bài 3: Document Classification
1. Load dataset văn bản (VD: 20newsgroups)
2. Tạo embeddings bằng Doc2Vec hoặc BERT
3. Huấn luyện classifier (SVM, Random Forest)
4. Đánh giá kết quả

### Bài 4: Topic Modeling
1. Áp dụng LDA trên tập tin tức
2. Trích xuất các chủ đề chính
3. Visualize kết quả

---

## TÀI LIỆU THAM KHẢO

1. **Word2Vec**: Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"
2. **GloVe**: Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"
3. **FastText**: Bojanowski et al. (2016) - "Enriching Word Vectors with Subword Information"
4. **BERT**: Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers"
5. **USE**: Cer et al. (2018) - "Universal Sentence Encoder"

---

## KẾT LUẬN

Embeddings là nền tảng quan trọng của NLP và AI hiện đại:
- Chuyển đổi dữ liệu phi cấu trúc thành vector số
- Bảo toàn ý nghĩa ngữ nghĩa
- Cho phép máy tính "hiểu" và xử lý ngôn ngữ tự nhiên

**Xu hướng hiện tại**:
- Mô hình ngày càng lớn (GPT-4, LLaMA, Claude...)
- Embeddings đa phương thức (text + image + audio)
- Transfer learning và fine-tuning
- Embeddings cho ngôn ngữ ít tài nguyên (low-resource languages)

**Lời khuyên**:
- Bắt đầu với Word2Vec/GloVe để hiểu cơ bản
- Chuyển sang BERT/Transformers cho tác vụ phức tạp
- Luôn thử nghiệm nhiều kỹ thuật và so sánh kết quả
- Chú ý đến preprocessing - nó ảnh hưởng lớn đến chất lượng embedding


---

## PHẦN 7: BÀI TẬP THỰC HÀNH YÊU CẦU

### Bài tập 1: Thao tác trên Vector

**Yêu cầu**: Tìm hiểu các thao tác cơ bản trên vector (cộng, trừ, nhân, tính khoảng cách, cosine similarity)

**Code demo**:
```python
import numpy as np
from scipy.spatial.distance import cosine, euclidean

# Tạo 2 vector
v1 = np.array([1, 2, 3, 4])
v2 = np.array([2, 3, 4, 5])

# Các phép toán cơ bản
print("Vector 1:", v1)
print("Vector 2:", v2)
print("Cộng:", v1 + v2)
print("Trừ:", v1 - v2)
print("Nhân vô hướng:", np.dot(v1, v2))

# Độ dài vector
print("Độ dài v1:", np.linalg.norm(v1))

# Khoảng cách Euclid
print("Khoảng cách Euclid:", euclidean(v1, v2))

# Cosine similarity
cos_sim = 1 - cosine(v1, v2)
print("Cosine Similarity:", cos_sim)

# Cách tính cosine thủ công
cos_manual = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print("Cosine (thủ công):", cos_manual)
```

---

### Bài tập 2: Thư viện NLP cơ bản

**Yêu cầu**: Tìm hiểu tách từ, gán nhãn từ loại, nhận dạng thực thể

#### A. Tiếng Anh với spaCy

```python
import spacy

# Load model tiếng Anh (cài: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

# 1. Tokenization (Tách từ)
print("=== TOKENIZATION ===")
tokens = [token.text for token in doc]
print("Tokens:", tokens)

# 2. Part-of-Speech Tagging (Từ loại)
print("\n=== POS TAGGING ===")
for token in doc:
    print(f"{token.text:15} → {token.pos_:10} ({token.tag_})")

# 3. Named Entity Recognition (Thực thể)
print("\n=== NAMED ENTITIES ===")
for ent in doc.ents:
    print(f"{ent.text:20} → {ent.label_:15} ({spacy.explain(ent.label_)})")

# 4. Lemmatization (Từ gốc)
print("\n=== LEMMATIZATION ===")
for token in doc:
    print(f"{token.text:15} → {token.lemma_}")
```

#### B. Tiếng Việt với underthesea

```python
from underthesea import word_tokenize, pos_tag, ner

text = "Ông Nguyễn Văn A làm việc tại công ty FPT ở Hà Nội"

# 1. Tách từ
print("=== TÁCH TỪ ===")
words = word_tokenize(text)
print(words)

# 2. Gán nhãn từ loại
print("\n=== TỪ LOẠI ===")
pos = pos_tag(text)
for word, tag in pos:
    print(f"{word:15} → {tag}")

# 3. Nhận dạng thực thể
print("\n=== THỰC THỂ ===")
entities = ner(text)
for word, tag in entities:
    if tag != 'O':  # Chỉ in thực thể
        print(f"{word:15} → {tag}")
```

#### C. Với NLTK

```python
import nltk
# Cài: nltk.download('punkt'), nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker'), nltk.download('words')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

text = "John works at Google in New York"

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# POS Tagging
pos = pos_tag(tokens)
print("POS:", pos)

# NER
entities = ne_chunk(pos)
print("Entities:", entities)

# Stopwords
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]
print("Sau khi loại stopwords:", filtered)
```

---

### Bài tập 3: Word Embeddings - Embedding 2 từ

**Yêu cầu**: Đưa vào 2 từ, embedding thành 2 vector, in ra và tính cosine similarity

```python
import gensim.downloader as api
from scipy.spatial.distance import cosine
import numpy as np

# Load pre-trained Word2Vec model (Google News)
print("Đang tải model... (có thể mất vài phút lần đầu)")
model = api.load('word2vec-google-news-300')

def compare_words(word1, word2):
    """So sánh 2 từ bằng word embeddings"""
    try:
        # Lấy vector của 2 từ
        vec1 = model[word1]
        vec2 = model[word2]
        
        print(f"\n{'='*60}")
        print(f"So sánh: '{word1}' vs '{word2}'")
        print(f"{'='*60}")
        
        # In 10 giá trị đầu của vector (vector có 300 chiều)
        print(f"\nVector '{word1}' (10 giá trị đầu):")
        print(vec1[:10])
        
        print(f"\nVector '{word2}' (10 giá trị đầu):")
        print(vec2[:10])
        
        # Tính cosine similarity
        cos_sim = 1 - cosine(vec1, vec2)
        
        print(f"\n📊 Cosine Similarity: {cos_sim:.4f}")
        
        # Giải thích
        if cos_sim > 0.7:
            print("   → Rất giống nhau!")
        elif cos_sim > 0.5:
            print("   → Có liên quan")
        elif cos_sim > 0.3:
            print("   → Hơi liên quan")
        else:
            print("   → Không liên quan")
            
    except KeyError as e:
        print(f"❌ Từ '{e.args[0]}' không có trong vocabulary!")

# Test với nhiều cặp từ
print("\n🔍 DEMO: SO SÁNH CÁC CẶP TỪ\n")

# Cặp 1: Từ đồng nghĩa
compare_words("king", "queen")

# Cặp 2: Từ cùng lĩnh vực
compare_words("computer", "laptop")

# Cặp 3: Từ liên quan
compare_words("doctor", "hospital")

# Cặp 4: Từ không liên quan
compare_words("apple", "car")

# Cặp 5: Từ trái nghĩa
compare_words("hot", "cold")

# Bonus: Tìm từ tương tự
print(f"\n{'='*60}")
print("🎯 BONUS: Tìm các từ tương tự với 'python'")
print(f"{'='*60}")
similar_words = model.most_similar('python', topn=5)
for word, score in similar_words:
    print(f"  {word:20} → {score:.4f}")
```

**Output mẫu**:
```
============================================================
So sánh: 'king' vs 'queen'
============================================================

Vector 'king' (10 giá trị đầu):
[ 0.125  0.234 -0.089  0.456 ...]

Vector 'queen' (10 giá trị đầu):
[ 0.134  0.221 -0.078  0.445 ...]

📊 Cosine Similarity: 0.6510
   → Có liên quan
```

---

### Bài tập 4: Sentence Embeddings - Embedding câu

**Yêu cầu**: Đưa vào 2 câu, embedding và tính độ tương tự

```python
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# Load model (lần đầu sẽ tải về)
print("Đang tải Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def compare_sentences(sent1, sent2):
    """So sánh 2 câu bằng sentence embeddings"""
    
    print(f"\n{'='*70}")
    print(f"Câu 1: {sent1}")
    print(f"Câu 2: {sent2}")
    print(f"{'='*70}")
    
    # Tạo embeddings
    vec1 = model.encode(sent1)
    vec2 = model.encode(sent2)
    
    print(f"\nKích thước vector: {len(vec1)} chiều")
    print(f"Vector câu 1 (10 giá trị đầu): {vec1[:10]}")
    print(f"Vector câu 2 (10 giá trị đầu): {vec2[:10]}")
    
    # Tính cosine similarity
    cos_sim = 1 - cosine(vec1, vec2)
    
    print(f"\n📊 Cosine Similarity: {cos_sim:.4f}")
    
    # Giải thích
    if cos_sim > 0.8:
        print("   → Rất giống nhau về ý nghĩa!")
    elif cos_sim > 0.6:
        print("   → Có liên quan")
    elif cos_sim > 0.4:
        print("   → Hơi liên quan")
    else:
        print("   → Không liên quan")

# Test với nhiều cặp câu
print("\n🔍 DEMO: SO SÁNH CÁC CẶP CÂU\n")

# Cặp 1: Câu giống nhau về nghĩa
compare_sentences(
    "I love programming in Python",
    "I enjoy coding with Python"
)

# Cặp 2: Câu cùng chủ đề
compare_sentences(
    "The cat is sleeping on the sofa",
    "A dog is running in the park"
)

# Cặp 3: Câu khác chủ đề
compare_sentences(
    "Machine learning is fascinating",
    "I like to eat pizza"
)

# Cặp 4: Câu hỏi tương tự
compare_sentences(
    "How do I learn Python?",
    "What is the best way to study Python?"
)

# Bonus: So sánh nhiều câu cùng lúc
print(f"\n{'='*70}")
print("🎯 BONUS: Tìm câu tương tự nhất")
print(f"{'='*70}")

query = "I want to learn machine learning"
sentences = [
    "Machine learning is a subset of AI",
    "I love eating pizza",
    "How to study artificial intelligence?",
    "The weather is nice today"
]

query_vec = model.encode(query)
print(f"\nCâu truy vấn: '{query}'\n")

similarities = []
for sent in sentences:
    sent_vec = model.encode(sent)
    sim = 1 - cosine(query_vec, sent_vec)
    similarities.append((sent, sim))

# Sắp xếp theo độ tương tự
similarities.sort(key=lambda x: x[1], reverse=True)

print("Kết quả (sắp xếp theo độ tương tự):")
for i, (sent, sim) in enumerate(similarities, 1):
    print(f"{i}. [{sim:.4f}] {sent}")
```

---

### Bài tập 5: Document Embeddings - Embedding văn bản

**Yêu cầu**: Đưa vào 2 đoạn văn bản, embedding và tính độ tương tự

```python
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def compare_documents(doc1, doc2, doc1_name="Document 1", doc2_name="Document 2"):
    """So sánh 2 văn bản"""
    
    print(f"\n{'='*70}")
    print(f"{doc1_name}:")
    print(f"{doc1[:100]}..." if len(doc1) > 100 else doc1)
    print(f"\n{doc2_name}:")
    print(f"{doc2[:100]}..." if len(doc2) > 100 else doc2)
    print(f"{'='*70}")
    
    # Tạo embeddings
    vec1 = model.encode(doc1)
    vec2 = model.encode(doc2)
    
    print(f"\nKích thước vector: {len(vec1)} chiều")
    
    # Tính cosine similarity
    cos_sim = 1 - cosine(vec1, vec2)
    
    print(f"\n📊 Cosine Similarity: {cos_sim:.4f}")
    
    if cos_sim > 0.7:
        print("   → Hai văn bản rất giống nhau về chủ đề!")
    elif cos_sim > 0.5:
        print("   → Hai văn bản có liên quan")
    elif cos_sim > 0.3:
        print("   → Hai văn bản hơi liên quan")
    else:
        print("   → Hai văn bản không liên quan")

# Test với các đoạn văn bản
print("\n🔍 DEMO: SO SÁNH CÁC VĂN BẢN\n")

# Văn bản 1: Về AI
doc_ai_1 = """
Artificial Intelligence (AI) is transforming the world. Machine learning 
and deep learning are subsets of AI that enable computers to learn from data. 
Neural networks, inspired by the human brain, are the foundation of deep learning.
"""

doc_ai_2 = """
Machine learning is a powerful technology. It allows systems to automatically 
learn and improve from experience. Deep neural networks have achieved remarkable 
results in image recognition and natural language processing.
"""

# Văn bản 2: Về ẩm thực
doc_food = """
Vietnamese cuisine is known for its fresh ingredients and balanced flavors. 
Pho is a traditional noodle soup that is loved worldwide. The combination of 
herbs, spices, and broth creates a unique taste.
"""

# Văn bản 3: Về du lịch
doc_travel = """
Traveling opens your mind to new cultures and experiences. Exploring different 
countries helps you understand diverse perspectives. It's important to respect 
local customs and traditions when visiting new places.
"""

# So sánh các cặp văn bản
compare_documents(doc_ai_1, doc_ai_2, "AI Document 1", "AI Document 2")
compare_documents(doc_ai_1, doc_food, "AI Document", "Food Document")
compare_documents(doc_food, doc_travel, "Food Document", "Travel Document")

# Bonus: Tìm văn bản tương tự nhất
print(f"\n{'='*70}")
print("🎯 BONUS: Tìm văn bản tương tự nhất với query")
print(f"{'='*70}")

query_doc = "I want to learn about neural networks and deep learning"
documents = {
    "AI & ML": doc_ai_1,
    "ML & DL": doc_ai_2,
    "Food": doc_food,
    "Travel": doc_travel
}

query_vec = model.encode(query_doc)
print(f"\nQuery: '{query_doc}'\n")

results = []
for name, doc in documents.items():
    doc_vec = model.encode(doc)
    sim = 1 - cosine(query_vec, doc_vec)
    results.append((name, sim))

results.sort(key=lambda x: x[1], reverse=True)

print("Kết quả:")
for i, (name, sim) in enumerate(results, 1):
    print(f"{i}. [{sim:.4f}] {name}")
```

---

### Bài tập 6: Ứng dụng thực tế - Hệ thống tìm kiếm câu hỏi tương tự

```python
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Database câu hỏi FAQ
faq_database = [
    "How do I reset my password?",
    "What is the refund policy?",
    "How can I track my order?",
    "Do you ship internationally?",
    "How do I contact customer support?",
    "What payment methods do you accept?",
    "How long does shipping take?",
    "Can I cancel my order?",
    "Do you offer student discounts?",
    "How do I return a product?"
]

# Tạo embeddings cho tất cả câu hỏi trong database
print("Đang tạo embeddings cho FAQ database...")
faq_embeddings = model.encode(faq_database)
print(f"✓ Đã tạo {len(faq_embeddings)} embeddings\n")

def find_similar_questions(user_question, top_k=3):
    """Tìm câu hỏi tương tự trong database"""
    
    print(f"{'='*70}")
    print(f"Câu hỏi của user: '{user_question}'")
    print(f"{'='*70}\n")
    
    # Tạo embedding cho câu hỏi user
    user_vec = model.encode(user_question)
    
    # Tính similarity với tất cả câu hỏi trong database
    similarities = []
    for i, faq_vec in enumerate(faq_embeddings):
        sim = 1 - cosine(user_vec, faq_vec)
        similarities.append((faq_database[i], sim))
    
    # Sắp xếp và lấy top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_k} câu hỏi tương tự:")
    for i, (question, sim) in enumerate(similarities[:top_k], 1):
        print(f"{i}. [{sim:.4f}] {question}")
    
    return similarities[:top_k]

# Test với các câu hỏi user
print("🔍 HỆ THỐNG TÌM KIẾM CÂU HỎI TƯƠNG TỰ\n")

find_similar_questions("I forgot my password, what should I do?")
print()
find_similar_questions("Where is my package?")
print()
find_similar_questions("Can I get my money back?")
print()
find_similar_questions("Do you deliver to other countries?")
```

---

### Cài đặt thư viện cần thiết

```bash
# Thư viện cơ bản
pip install numpy scipy

# NLP
pip install spacy
python -m spacy download en_core_web_sm

pip install underthesea  # Tiếng Việt
pip install nltk

# Embeddings
pip install gensim
pip install sentence-transformers

# Nếu dùng transformers
pip install transformers torch
```

---

### Ghi chú quan trọng

1. **Word Embeddings**: Tốt cho phân tích từ đơn, nhanh, nhẹ
2. **Sentence Embeddings**: Tốt cho so sánh câu, tìm kiếm semantic
3. **Document Embeddings**: Tốt cho phân loại văn bản, clustering
4. **Cosine Similarity**: Độ đo phổ biến nhất, giá trị từ -1 đến 1
5. **Pre-trained models**: Nên dùng để tiết kiệm thời gian, cho kết quả tốt

### Tips thực hành

- Bắt đầu với model nhỏ (`all-MiniLM-L6-v2`) để test nhanh
- Dùng model lớn hơn (`all-mpnet-base-v2`) khi cần độ chính xác cao
- Luôn normalize text trước khi embedding
- Cache embeddings nếu dữ liệu không thay đổi thường xuyên
- Với tiếng Việt, cần tách từ trước khi dùng Word2Vec
