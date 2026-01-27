# Kiểm Thử Phần Mềm với Mô Hình Ngôn Ngữ Lớn: Khảo Sát, Bối Cảnh và Tầm Nhìn

## Tóm tắt

Các mô hình ngôn ngữ lớn (LLMs) được huấn luyện trước gần đây đã nổi lên như một công nghệ đột phá trong xử lý ngôn ngữ tự nhiên và trí tuệ nhân tạo, với khả năng xử lý các tập dữ liệu quy mô lớn và thể hiện hiệu suất đáng chú ý trên nhiều tác vụ khác nhau. Trong khi đó, kiểm thử phần mềm là một hoạt động quan trọng đóng vai trò nền tảng để đảm bảo chất lượng và độ tin cậy của sản phẩm phần mềm. Khi phạm vi và độ phức tạp của hệ thống phần mềm tiếp tục tăng lên, nhu cầu về các kỹ thuật kiểm thử phần mềm hiệu quả hơn trở nên ngày càng cấp thiết, khiến đây trở thành lĩnh vực phù hợp cho các phương pháp tiếp cận sáng tạo như sử dụng LLMs.

Bài báo này cung cấp một đánh giá toàn diện về việc sử dụng LLMs trong kiểm thử phần mềm. Nó phân tích **102 nghiên cứu liên quan** đã sử dụng LLMs cho kiểm thử phần mềm, từ cả góc độ kiểm thử phần mềm và góc độ LLMs. Bài báo trình bày một cuộc thảo luận chi tiết về các tác vụ kiểm thử phần mềm mà LLMs thường được sử dụng, trong đó chuẩn bị test case và sửa lỗi chương trình là hai ứng dụng đại diện nhất. Nó cũng phân tích các LLMs thường được sử dụng, các loại kỹ thuật prompt engineering được áp dụng, cũng như các kỹ thuật đi kèm với những LLMs này. Bài báo cũng tóm tắt các thách thức chính và cơ hội tiềm năng trong hướng này.

**Từ khóa:** Mô hình ngôn ngữ lớn, Kiểm thử phần mềm, Tạo test case, Sửa lỗi chương trình

---

## 1. Giới Thiệu

Kiểm thử phần mềm là quá trình đánh giá và xác minh rằng một sản phẩm phần mềm hoặc ứng dụng thực hiện những gì nó được thiết kế để làm. Nó là một phần quan trọng trong chu kỳ phát triển phần mềm, giúp phát hiện lỗi, đảm bảo chất lượng và cải thiện độ tin cậy của phần mềm.

Các mô hình ngôn ngữ lớn (LLMs) như GPT-3, GPT-4, Codex, và ChatGPT đã chứng minh khả năng đáng kinh ngạc trong nhiều tác vụ liên quan đến mã nguồn, bao gồm tạo mã, hoàn thiện mã, và hiểu mã. Điều này đã mở ra cơ hội mới cho việc áp dụng LLMs vào kiểm thử phần mềm.

---

## 2. Các Tác Vụ Kiểm Thử Phần Mềm Sử Dụng LLMs

### 2.1 Tạo Unit Test Case

**Tổng quan:**
Tạo unit test case tự động là một trong những ứng dụng phổ biến nhất của LLMs trong kiểm thử phần mềm. Mục tiêu là tối đa hóa độ bao phủ (coverage) trong phần mềm đang được kiểm thử.

**Các phương pháp chính:**

1. **Pre-training hoặc Fine-tuning LLMs:**
   - Alagarsamy et al. đầu tiên pre-train LLM với focal method và assertion statements để cho phép LLM có nền tảng kiến thức mạnh mẽ hơn về assertions
   - Tufano et al. sử dụng schema tương tự bằng cách pre-train LLM trên corpus Java lớn không giám sát
   - Hashtroudi et al. tận dụng các test do developer viết sẵn cho mỗi project để tạo dataset đặc thù cho domain adaptation

2. **Thiết kế prompt hiệu quả:**
   - Xie et al. tạo unit test cases bằng cách phân tích project, trích xuất thông tin thiết yếu, và tạo adaptive focal context
   - Yuan et al. đề xuất approach tận dụng chính ChatGPT để cải thiện chất lượng các test được tạo ra với initial test generator và iterative test refiner

3. **Kết hợp LLM với phương pháp search-based:**
   - Lemieux et al. để các kỹ thuật kiểm thử dựa trên tìm kiếm truyền thống tạo unit test case cho đến khi cải thiện coverage bị đình trệ, sau đó yêu cầu LLM cung cấp các test case mẫu cho các hàm chưa được bao phủ đầy đủ

**Hiệu suất:**
- Trên 10 Java projects: 40% correctness, 89% line coverage, 90% branch coverage (ChatGPT)
- Trên HumanEval: 78% correctness, 87% line coverage, 92% branch coverage (Codex)
- Trên SF110 benchmark: chỉ 2% coverage - cho thấy vẫn còn nhiều thách thức

### 2.2 Tạo Test Oracle

**Định nghĩa:**
Test oracle là nguồn thông tin về việc liệu đầu ra của hệ thống phần mềm có đúng hay không. Hầu hết các nghiên cứu trong danh mục này nhắm đến việc tạo test assertion.

**Các nghiên cứu chính:**
- Mastropaolo et al. pre-train mô hình T5 và fine-tune nó, đạt được 57% exact match rate
- Tufano et al. cải thiện hiệu suất lên 62% exact match rate
- Nashid et al. sử dụng prompt engineering và đạt được 76% accuracy - hiệu suất state-of-the-art

### 2.3 Tạo System Test Input

**Phân loại theo loại phần mềm:**

1. **Mobile Apps (5 nghiên cứu):**
   - Liu et al. sử dụng LLM để tạo semantic input text theo GUI context
   - GPTDroid trích xuất static context của GUI page và dynamic context của quá trình testing lặp lại

2. **Deep Learning Libraries (2 nghiên cứu):**
   - Deng et al. sử dụng LLM để tạo và mutate các chương trình DL hợp lệ/đa dạng cho fuzzing DL libraries
   - Kết hợp generative LLM (CodeX) và infilling LLM (InCoder)

3. **Compilers, SMT Solvers, và các loại phần mềm khác:**
   - Yang et al. đề xuất LLM-based compiler fuzzer với dual-model framework
   - Sun et al. sử dụng LLM để tạo test formulas cho fuzzing SMT solvers

**Phân loại theo kỹ thuật testing:**
- **Fuzz Testing:** Kỹ thuật được sử dụng phổ biến nhất
- **GUI Testing:** Tạo meaningful text input và functionality-oriented exploration traces
- **Penetration Testing:** Deng et al. tận dụng LLMs để thực hiện penetration testing tasks tự động

### 2.4 Phân Tích Lỗi (Bug Analysis)

**Các tác vụ chính:**
- Mukherjee et al. tạo câu trả lời cho các câu hỏi follow-up về bug reports thiếu sót
- Su et al. chuyển đổi bug-component triaging thành multi-classification task
- Zhang et al. tận dụng LLM trong zero-shot setting để lấy thông tin thiết yếu về bug reports
- Mahbub et al. đề xuất giải thích software bugs với LLM

### 2.5 Debug

**Framework debug tổng thể:**
- Bui et al. đề xuất unified Detect-Localize-Repair framework dựa trên LLM
- Kang et al. đề xuất automated scientific debugging
- Chen et al. chứng minh self-debugging có thể dạy LLM thực hiện rubber duck debugging

**Bug Localization:**
- Wu et al. so sánh ChatGPT và GPT-4 với các kỹ thuật fault localization hiện có
- Kang et al. đề xuất AutoFL - kỹ thuật fault localization tự động chỉ yêu cầu một failing test

**Bug Reproduction:**
- Kang et al. và Plein et al. đề xuất framework để khai thác LLM tái tạo bugs
- Feng et al. đề xuất AdbGPT để tự động tái tạo bugs từ bug reports

### 2.6 Sửa Lỗi Chương Trình (Program Repair)

**Đây là tác vụ được nghiên cứu nhiều nhất với LLMs.**

**Patch single-line bugs:**
- Lajkó et al. đề xuất fine-tune LLM với JavaScript code snippets
- Zhang et al. sử dụng program slicing để trích xuất contextual information
- Zhang et al. đề xuất STEAM framework cho patching single-line bugs

**Patch multiple-lines bugs:**
- Fu et al. fine-tune LLM bằng BPE tokenization để xử lý Out-Of-Vocabulary issues
- Ribeiro et al. tận dụng LLM để thực hiện code completion trong buggy line
- Xia et al. đề xuất conversation-driven program repair approach

**Repair với static code analyzer:**
- Jin et al. đề xuất program repair framework kết hợp với static analyzer
- Wadhwa et al. sử dụng LLM như ranker để đánh giá likelihood of acceptance

**Hiệu suất:**
- QuixBugs dataset: 31/40 Python bugs, 23/40 Java bugs (ChatGPT)
- Defects4J: 39/40 Python bugs, 34/40 Java bugs (Codex)
- DL programs từ StackOverflow: chỉ 16/72 Python bugs - cho thấy độ phức tạp thực tế

---

## 3. Phân Tích Từ Góc Độ LLM

### 3.1 Các LLM Được Sử Dụng

**Top LLMs trong các nghiên cứu:**
1. **ChatGPT (36 papers - 25%):** LLM phổ biến nhất
2. **Codex (23 papers - 16%):** Được train trên massive code corpus
3. **CodeT5 (18 papers - 13%):** Open-source, dễ fine-tune
4. **GPT-4 (14 papers - 10%):** State-of-the-art, multi-modal
5. **GPT-3, CodeGen, InCoder, PLBART, T5...**

### 3.2 Các Loại Prompt Engineering

**Phân bố:**
- 38 nghiên cứu: Pre-training/Fine-tuning
- 64 nghiên cứu: Prompt engineering
  - 51 nghiên cứu: Zero-shot learning
  - 25 nghiên cứu: Few-shot learning
  - 7 nghiên cứu: Chain-of-thought
  - 1 nghiên cứu: Self-consistency
  - 1 nghiên cứu: Automatic prompt

**Zero-shot learning:**
- Đơn giản feed task text vào model và yêu cầu kết quả
- Phù hợp cho các tác vụ liên quan đến source code

**Few-shot learning:**
- Cung cấp một tập các demonstrations chất lượng cao
- Giúp model hiểu rõ hơn ý định của con người

**Chain-of-thought:**
- Tạo chuỗi các câu ngắn mô tả logic reasoning từng bước
- Ví dụ: localize bug → explain why → fix bug

**Iterative prompt design (14 nghiên cứu):**
- Liên tục tinh chỉnh prompts với thông tin running của testing task
- Ví dụ: kết hợp test failure information vào prompt tiếp theo

### 3.3 Input của LLM

**Phân bố input:**
- **Code (78 papers - 68%):** Input phổ biến nhất
- **Bug description (12 papers - 10%)**
- **Error information (7 papers - 6%)**
- **View hierarchy file of UI (6 papers - 5%)**
- **Others (12 papers - 10%)**

### 3.4 Kết Hợp LLM với Các Kỹ Thuật Khác

**Phân bố:**
- 67 nghiên cứu: Chỉ sử dụng LLM
- 35 nghiên cứu: Kết hợp với kỹ thuật khác

**Các kỹ thuật kết hợp:**
1. **Statistical analysis:** Ranking, clustering để lọc outputs của LLM
2. **Program analysis:** Sử dụng AST, code structure
3. **Mutation testing:** Tạo test inputs đa dạng hơn
4. **Syntactic checking:** Kiểm tra và sửa lỗi syntax
5. **Differential testing:** Tìm semantic/logic bugs

---

## 4. Thách Thức

### 4.1 Thách Thức Đạt Coverage Cao

- SF110 dataset: chỉ 2% line coverage, 1% branch coverage
- TensorFlow API coverage: 66% (2215/3316)
- Cần kết hợp mutation testing để tăng diversity

### 4.2 Thách Thức về Test Oracle Problem

- Oracle problem vẫn là thách thức lâu dài
- Các giải pháp hiện tại:
  - Sử dụng differential testing
  - Chỉ tập trung vào crash bugs
- Cần khám phá khả năng của LLMs trong metamorphic testing

### 4.3 Thách Thức về Đánh Giá Nghiêm Ngặt

- Thiếu benchmark datasets chuẩn
- Vấn đề data leakage: LLMs có thể đã thấy benchmarks trong pre-training data
- Cần datasets chuyên biệt hơn và đa dạng hơn

### 4.4 Thách Thức trong Ứng Dụng Thực Tế

- Vấn đề privacy: Công ty không muốn dùng commercial LLMs
- Hạn chế về computational power
- Hiệu suất trên real-world code thấp hơn nhiều so với benchmarks
- Cần high-quality organization-specific datasets

---

## 5. Cơ Hội

### 5.1 Khám Phá LLMs trong Giai Đoạn Đầu của Testing

- Chưa có nghiên cứu về test requirements, test planning
- Cần schema human-computer interaction
- Công ty nên record và cung cấp early-stage testing data

### 5.2 Khám Phá LLMs trong Các Pha Testing Khác

- **Integration testing:** Chưa có nghiên cứu nào
- **Acceptance testing:** Phù hợp cho human-in-the-loop schema

### 5.3 Khám Phá LLMs cho Nhiều Loại Phần Mềm Hơn

- Mobile apps được nghiên cứu nhiều nhất (5 studies)
- Cơ hội cho: metaverse, quantum computing, cyber-physical systems...

### 5.4 Khám Phá LLMs cho Non-functional Testing

- Chưa có nghiên cứu về performance testing, usability testing
- LLM có thể integrate với performance testing tools
- Có thể identify parameter combinations trigger performance problems

### 5.5 Khám Phá Advanced Prompt Engineering

**Các kỹ thuật chưa được sử dụng:**
- Generate knowledge prompt
- Tree-of-thoughts
- Active-prompt
- Directional stimulus prompt
- ReAct prompt
- Multimodal chain-of-thought
- Graph prompt
- Automatic reasoning and tool-use

### 5.6 Kết Hợp LLMs với Kỹ Thuật Truyền Thống

- LLMs không phải silver bullet
- Cần khám phá cách kết hợp tốt hơn với:
  - Differential testing
  - Mutation testing
  - Program analysis
  - Formal verification
- Tích hợp LLMs với mature testing tools

---

## 6. Kết Luận

Bài báo này cung cấp đánh giá toàn diện về việc sử dụng LLMs trong kiểm thử phần mềm, phân tích 102 nghiên cứu liên quan từ cả góc độ kiểm thử phần mềm và LLMs.

**Những phát hiện chính:**
- LLMs đã được áp dụng thành công trong nhiều tác vụ testing
- Unit test case generation và program repair là hai ứng dụng phổ biến nhất
- Vẫn còn nhiều thách thức về coverage, test oracle, và ứng dụng thực tế
- Nhiều cơ hội chưa được khám phá: early-stage testing, integration testing, non-functional testing

**Tầm nhìn tương lai:**
Nghiên cứu này có thể đóng vai trò như roadmap cho nghiên cứu tương lai, làm nổi bật các hướng khám phá tiềm năng và xác định khoảng trống trong hiểu biết hiện tại về việc sử dụng LLMs trong kiểm thử phần mềm.

---

## Tài Liệu Tham Khảo

**Paper gốc:** arXiv:2307.07221  
**Tác giả:** Junjie Wang và cộng sự  
**Xuất bản:** IEEE Transactions on Software Engineering (2023)  
**Link:** https://arxiv.org/abs/2307.07221

---

*Tài liệu này được dịch và tóm tắt từ paper "Software Testing with Large Language Models: Survey, Landscape, and Vision" nhằm mục đích học tập và nghiên cứu.*
