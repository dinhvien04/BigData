# VINUNI AI20K – LEVEL 2 (DAY 01–14): GIÁO TRÌNH HỌC TỪ CON SỐ 0

> **Phạm vi:** Chỉ Level 2, Day 01 đến Day 14. **Không sử dụng Day 15 hoặc tài liệu Level 3.**  
> **Dành cho:** Người mới chưa biết AI, Data Annotation, CVAT, Detection, Tracking, LiDAR.  
> **Cách học:** Đọc **Định nghĩa → Tại sao cần → Ví dụ → Công thức và từng ký hiệu → Lỗi dễ nhầm → Câu tự kiểm tra**.  
> **Tài liệu nền:** Các slide VinUni AI20K Level 2 tương ứng. Ví dụ số tự dựng được ghi là minh họa; những công thức diễn giải thêm sẽ được phân biệt với trọng tâm slide.
>
> Đây là tài liệu để **hiểu và luyện**, không phải đề thi chính thức. Trong công việc, **guideline của dự án** là thẩm quyền quyết định cách gán nhãn.

## MỤC LỤC

1. [Day 01 – Hiểu về AI và dữ liệu](#day-01--hiểu-về-ai-và-dữ-liệu)
2. [Day 02 – Object Detection và YOLO](#day-02--object-detection-và-yolo)
3. [Day 03 – Multi-Object Tracking](#day-03--multi-object-tracking)
4. [Day 04 – Keypoint và Pose](#day-04--keypoint-và-pose)
5. [Day 05 – Segmentation](#day-05--segmentation)
6. [Day 06 – Data Annotation Lifecycle và QC](#day-06--data-annotation-lifecycle-và-qc)
7. [Day 07 – Data Pipeline và Governance](#day-07--data-pipeline-và-governance)
8. [Day 08 – AI Assistance và Active Learning](#day-08--ai-assistance-và-active-learning)
9. [Day 09 – Road Elements](#day-09--road-elements)
10. [Day 10 – DMS và OMS](#day-10--dms-và-oms)
11. [Day 11 – Surround View và Fisheye](#day-11--surround-view-và-fisheye)
12. [Day 12 – LiDAR và 3D Cuboid](#day-12--lidar-và-3d-cuboid)
13. [Day 13 – 3D Object Detection](#day-13--3d-object-detection)
14. [Day 14 – 3D Tracking và Calibration](#day-14--3d-tracking-và-calibration)
15. [Sổ tay công thức và bài tập](#sổ-tay-công-thức-và-bài-tập)

---

# DAY 01 – HIỂU VỀ AI VÀ DỮ LIỆU

## 1. AI, ML, DL và LLM là gì?

**AI – Artificial Intelligence (Trí tuệ nhân tạo):** Lĩnh vực xây dựng hệ thống thực hiện các nhiệm vụ cần khả năng xử lý thông minh như nhận diện vật thể, đọc văn bản, lập kế hoạch. AI **không phải** tên riêng của ChatGPT.

**ML – Machine Learning (Học máy):** Một nhánh AI dùng dữ liệu để học quy luật. Thay vì viết từng luật thủ công, thuật toán điều chỉnh tham số dựa trên các ví dụ.

**Ví dụ:** Lập trình theo luật “có hai bánh thì là xe máy” có thể nhầm xe đạp. Nếu huấn luyện ML bằng nhiều ảnh xe máy/xe đạp có nhãn, model có thể học nhiều dấu hiệu phức tạp hơn.

**DL – Deep Learning (Học sâu):** Một nhánh ML sử dụng mạng neural nhiều tầng để học đặc trưng và dự đoán.

**LLM – Large Language Model:** Mô hình ngôn ngữ lớn, thuộc nhóm mô hình học sâu; không phải mọi AI đều là LLM.

**Quan hệ:**

$$
\boxed{\text{Deep Learning}\subset\text{Machine Learning}\subset\text{AI}}
$$

**Dễ nhầm:** AI không bắt buộc dùng neural network; DL là một phần của ML, không phải ngược lại.

## 2. Những cách học máy thường gặp

**Supervised Learning – Học có giám sát:** Mỗi mẫu có nhãn tham chiếu tương ứng. Ví dụ ảnh A có nhãn car, ảnh B có nhãn motorcycle; hoặc ảnh đường phố có box và class cho từng vật.

**Unsupervised Learning – Học không giám sát:** Học cấu trúc từ dữ liệu mà không cần nhãn đáp án cho từng mẫu theo kiểu trên. Ví dụ gom 1.000 ảnh thành những nhóm trông giống nhau. Các nhóm đó không tự nhiên có tên class do con người đặt.

**Self-supervised Learning:** Tạo tín hiệu học từ chính dữ liệu, chẳng hạn che một phần dữ liệu rồi học dự đoán phần thiếu. Không đồng nghĩa với “không cần dữ liệu”.

**Reinforcement Learning:** Học thông qua hành động và tín hiệu thưởng/phản hồi trong môi trường, khác với việc học box YOLO có nhãn thông thường.

## 3. Dataset, Annotation, Label và Metadata

**Dataset – Tập dữ liệu:** Tập hợp mẫu dùng để huấn luyện hoặc đánh giá: ảnh, video, point cloud, nhãn, thời gian chụp, thông tin sensor.

**Annotation – Gán nhãn:** Quá trình thêm thông tin có cấu trúc cho dữ liệu. Ví dụ trong CVAT, vẽ một hình chữ nhật bao xe máy và chọn class motorcycle.

**Label – Nhãn:** Thông tin được lưu sau khi gán nhãn, như class + box hoặc vị trí 17 khớp cơ thể.

**Metadata – Siêu dữ liệu:** Thông tin mô tả ảnh/file: camera_id, timestamp, thời tiết, phiên bản dataset, nguồn dữ liệu. Hai ảnh giống cảnh nhưng chụp từ camera thường và camera fisheye có thể cần quy tắc annotation khác.

**Những kiểu annotation:**

| Kiểu | Trả lời câu hỏi gì? | Ví dụ |
|---|---|---|
| Classification | Đây là loại gì? | Ảnh chứa traffic scene |
| Bounding Box | Vật gì và ở đâu? | Hộp quanh ô tô |
| Segmentation | Pixel nào thuộc vật/vùng? | Mask thân xe |
| Keypoint/Skeleton | Điểm/khớp nào ở đâu? | Vai, khuỷu, cổ tay |
| Tracking | Có phải cùng vật qua frame? | Xe mang ID 7 |
| 3D Cuboid | Vị trí, kích thước, hướng 3D? | Hộp LiDAR quanh xe |

## 4. Ground Truth khác Prediction như thế nào? [TRỌNG TÂM]

**Ground Truth (GT):** Nhãn tham chiếu được tạo và xác lập theo quy trình để dùng trong training/evaluation.

**Prediction:** Kết quả do model dự đoán.

**Ví dụ:** Ground Truth đánh dấu xe ở box (100,100,300,220); YOLO dự đoán (105,101,305,221). Ta so kết quả YOLO **với** Ground Truth bằng metric, không gọi dự đoán đó là Ground Truth chỉ vì model có confidence 0.99.

**Quan trọng:** Ground Truth **có thể sai** do annotator gán nhãn sai. Vì vậy phải QC. Model confidence cao cũng không đảm bảo đúng class, vị trí hoặc hướng.

## 5. Tại sao dữ liệu quyết định chất lượng AI?

Một model học từ nhãn. Nếu ảnh có 10 xe máy cần vẽ mà chỉ có 7 box, ba xe bị bỏ sót. Nếu một ô tô bị ghi là truck, model nhận tín hiệu sai về class. Nếu box quá rộng, model học vùng nền không đúng.

Dữ liệu tốt cần **đúng hình học, đúng class, đầy đủ theo guideline, nhất quán và có thể truy vết**.

Ba nguồn lực được nhấn mạnh trong bài học AI: **dữ liệu, thuật toán và năng lực tính toán**. Trong công việc data annotation, con người tác động trực tiếp đến chất lượng dữ liệu.

## 6. Data Lifecycle là gì?

**Data Lifecycle – Vòng đời dữ liệu:** Chuỗi công việc từ xác định mục tiêu, thu thập dữ liệu, chuẩn bị, viết guideline, annotation, QC, phát hành, theo dõi lỗi rồi cải thiện vòng sau.

~~~text
Mục tiêu → Chọn dữ liệu → Chuẩn bị → Guideline → Pilot
        → Annotation → QC → Release → Feedback → Lặp lại
~~~

Day 06 sẽ giải thích chi tiết **chín bước chính thức** của vòng đời này.

### Tự kiểm tra Day 01

1. DL là một nhánh của ML hay ngược lại?
2. Ground Truth và Prediction khác nhau ở đâu?
3. Có 10 xe nhưng chỉ gán 7 box thì loại vấn đề nào xảy ra?
4. Dataset có chỉ gồm mỗi ảnh không?

---

# DAY 02 – OBJECT DETECTION VÀ YOLO

## 1. Object Detection là gì? [TRỌNG TÂM]

**Object Detection:** Nhiệm vụ tìm **vật gì** và **nằm ở đâu** trong ảnh. Kết quả thường có **Class + Bounding Box + Confidence**.

**Ví dụ:** Một ảnh có ô tô và xe máy. Detector có thể sinh hai dự đoán: car (box A, score 0.95), motorcycle (box B, score 0.75).

**Image Classification** chỉ nói một ảnh hoặc vùng thuộc loại gì; **Detection** còn định vị từng vật.

**Confidence Score:** Điểm model gán cho dự đoán; **Confidence Threshold:** Ngưỡng để quyết định giữ dự đoán.

## 2. Ba phần của YOLO

**Backbone:** Trích xuất đặc trưng từ ảnh như cạnh, hình dạng, kết cấu.

**Neck:** Kết hợp đặc trưng ở những cấp độ khác nhau để xử lý vật lớn/nhỏ.

**Head:** Sinh đầu ra dự đoán box, class và score theo thiết kế model.

**Training:** Model điều chỉnh tham số từ dữ liệu có nhãn. **Inference:** Dùng model để dự đoán ảnh mới. **Pretrained:** Model đã học trên dữ liệu trước đó. **Fine-tuning:** Tiếp tục huấn luyện trên dữ liệu/tác vụ của mình.

**Anchor-based:** Sử dụng các khung mẫu trong một số detector. **Anchor-free:** Cách dự đoán không dựa vào bộ anchor cố định tương tự. Các đời YOLO không hoàn toàn giống nhau.

## 3. Bounding Box: XYXY, COCO XYWH, CENTER XYWH [TRỌNG TÂM]

Giả sử x tăng từ trái sang phải, y tăng từ trên xuống; ảnh rộng **W**, cao **H**.

**XYXY:** \((x_1,y_1,x_2,y_2)\), gồm hai góc trên-trái và dưới-phải.

**COCO XYWH:** \((x,y,w,h)\) gồm **góc trên-trái** và kích thước.

**CENTER XYWH:** \((x_c,y_c,w,h)\) gồm **tâm** và kích thước.

Từ XYXY sang COCO XYWH:

$$
\boxed{w=x_2-x_1,\qquad h=y_2-y_1}
$$

Giữ nguyên \((x_1,y_1)\).

Từ XYXY sang CENTER XYWH:

$$
\boxed{x_c=\frac{x_1+x_2}{2},\quad y_c=\frac{y_1+y_2}{2}}
$$

Kích thước vẫn là \(w=x_2-x_1,\ h=y_2-y_1\).

Từ CENTER về XYXY:

$$
x_1=x_c-w/2,\quad x_2=x_c+w/2,\qquad
y_1=y_c-h/2,\quad y_2=y_c+h/2.
$$

Từ COCO về XYXY: \((x,y,x+w,y+h)\).

**YOLO normalized:** Chia x và width cho W; chia y và height cho H:

$$
\boxed{(x_n,y_n,w_n,h_n)=
\left(\frac{x_c}{W},\frac{y_c}{H},\frac{w}{W},\frac{h}{H}\right)}
$$

**Ví dụ:** Box XYXY=(20,30,120,90) → COCO=(20,30,100,60) → CENTER=(70,60,100,60). Nếu ảnh 200×100 thì YOLO=(0.35,0.60,0.50,0.60).

**Dễ nhầm nhất:** Chữ XYWH không nói rõ x,y là **góc** hay **tâm**; phải biết định dạng.

## 4. IoU – Intersection over Union [TRỌNG TÂM]

**Intersection:** Diện tích giao nhau giữa hai box.

**Union:** Diện tích vùng thuộc ít nhất một box.

$$
\boxed{IoU=\frac{S_{\mathrm{giao}}}{S_{\mathrm{hợp}}}},
\qquad
\boxed{S_{\mathrm{hợp}}=S_A+S_B-S_{\mathrm{giao}}}.
$$

**Ví dụ:** Box A diện tích 24, B diện tích 18, giao 15:

$$
IoU=\frac{15}{24+18-15}=\frac{15}{27}\approx0.556.
$$

Vì sao trừ giao? Cộng diện tích hai box đã đếm giao **hai lần**. IoU bằng 0 nếu không giao, 1 nếu trùng hoàn toàn.

## 5. NMS – Non-Maximum Suppression

**NMS:** Bước giảm box trùng lặp trong kết quả model. Nếu cùng xe được vẽ 3 box chồng nhau rất mạnh, NMS có thể giữ box score cao và loại bớt box trùng theo ngưỡng.

**Confidence Threshold:** Ngưỡng theo điểm score.

**NMS IoU Threshold:** Ngưỡng để xem các box dự đoán có trùng nhau quá nhiều không.

**Evaluation IoU Threshold:** Ngưỡng để xem dự đoán khớp GT khi đánh giá. **Đừng nhầm hai ngưỡng IoU có mục đích khác nhau.**

## 6. TP, FP và FN [TRỌNG TÂM]

**TP – True Positive:** Phát hiện khớp một vật thật hợp lệ theo class/IoU/matching của bài đánh giá.

**FP – False Positive:** Dự đoán sai, thừa hoặc không khớp vật thật.

**FN – False Negative:** Vật thật cần tìm bị bỏ sót.

**Ví dụ:** GT có 10 xe, model tìm đúng 7 và vẽ thêm 2 box sai: TP=7, FN=3, FP=2.

**Lưu ý:** FN của **model evaluation** khác với **Missing Annotation** của annotator, dù đều mang nghĩa có thứ bị bỏ sót.

## 7. Precision và Recall [TRỌNG TÂM]

**Precision:** Trong tất cả các dự đoán dương tính, bao nhiêu là đúng?

$$
\boxed{Precision=\frac{TP}{TP+FP}}
$$

**Recall:** Trong tất cả các vật thể dương tính thật, model tìm được bao nhiêu?

$$
\boxed{Recall=\frac{TP}{TP+FN}}
$$

Với TP=80, FP=20, FN=40:

$$
Precision=80/100=80\%,\qquad Recall=80/120\approx66.7\%.
$$

**Mẹo:** Precision đi với **FP**; Recall đi với **FN**.

## 8. AP, mAP, mAP50, mAP50:95

**Precision–Recall Curve:** Đường biểu diễn trade-off Precision và Recall khi thay đổi ngưỡng dự đoán.

**AP – Average Precision:** Chỉ số tổng hợp đường Precision–Recall của một class, theo giao thức đánh giá.

**mAP – mean AP:** Trung bình AP của các class:

$$
\boxed{mAP=\frac{AP_1+AP_2+\cdots+AP_C}{C}}
$$

**Ví dụ:** Car=0.9; Person=0.8; Motorcycle=0.7 → mAP=0.8.

**mAP50:** Đánh giá tại ngưỡng matching IoU 0.50.

**mAP50:95:** Trung bình kết quả qua 10 ngưỡng từ IoU 0.50 đến 0.95 với bước 0.05. Điểm chung đẹp có thể che lớp hiếm có Recall kém.

## 9. Những lỗi annotation trong Day 02 [TRỌNG TÂM]

**Missing Annotation:** Bỏ sót xe/người đáng lẽ phải vẽ.

**Localization Error:** Box sai vị trí, quá rộng hoặc quá hẹp theo rule.

**Wrong Class:** Box bám đúng vật nhưng class sai.

QC thực tế còn có **Duplicate** (vật bị vẽ lặp), **Wrong Attribute** (sai occluded/truncated...), **Inconsistency** (hai annotator áp rule khác nhau).

**Ví dụ:** Một ảnh có 10 xe máy nhưng annotator vẽ 7 → Missing Annotation; không suy ra sẵn TP=7 vì TP là kết quả matching khi **đánh giá model**, không phải số box annotator đã vẽ.

## 10. Quy trình xây detector

Thu ảnh → định nghĩa taxonomy/guideline → vẽ box → chia train/validation/test hợp lý → xuất nhãn đúng format → training/fine-tuning → inference → đánh giá Precision/Recall/mAP → xem lỗi → sửa dữ liệu.

### Tự kiểm tra Day 02

1. XYXY=(10,20,110,80) → COCO và CENTER?
2. TP=30, FP=10, FN=20 → Precision/Recall?
3. NMS dùng để làm gì?
4. IoU=0.7 có tự chứng minh class đúng không?

---

# DAY 03 – MULTI-OBJECT TRACKING

## 1. Detection khác Tracking như thế nào? [TRỌNG TÂM]

**Detection:** Ở frame này có vật gì, box ở đâu?

**Tracking:** Qua nhiều frame, những box này có thuộc **cùng một cá thể** không?

**Track ID:** Mã định danh nhất quán của vật thể qua video. Một chiếc xe ở 10 frame là **1 xe** nhưng có thể tạo **10 box theo thời gian**.

| Frame | Đối tượng | ID |
|---|---|---|
| 1 | Xe đỏ | 7 |
| 2 | Xe đỏ | 7 |
| 3 | Xe đỏ | 7 |

## 2. Bốn bước của tracker

1. **Detect:** Tìm object hiện tại.
2. **Predict:** Dự đoán vị trí của những track có sẵn.
3. **Data Association:** Ghép box mới với track cũ.
4. **Track Management:** Giữ/tạm mất/tạo mới/xóa track.

**Ví dụ:** ID 7 ở x=100 rồi x=110; tracker dự đoán x gần 120. Detector tìm box tại x=122 nên ghép box đó vào ID 7 nếu các điều kiện phù hợp.

## 3. Kalman Filter

**Kalman Filter:** Phương pháp dự đoán và cập nhật trạng thái bằng cách kết hợp trạng thái dự đoán với quan sát và độ tin cậy.

Dạng đơn giản một chiều để hiểu:

$$
x_{\mathrm{new}}=x_{\mathrm{predict}}+K(x_{\mathrm{observe}}-x_{\mathrm{predict}}).
$$

**Ví dụ:** Dự đoán x=100, thấy x=110, K=0.6 → x mới=106.

**K là Kalman Gain**, không phải confidence của detector. Khi vật bị che và thiếu quan sát, phép dự đoán có thể tiếp tục một thời gian, không có nghĩa chắc chắn.

## 4. Data Association và Matching Cost [TRỌNG TÂM]

**Association:** Quyết định detection nào thuộc track nào. Một quy tắc chi phí đơn giản:

$$
\boxed{Cost=1-IoU}
$$

**Ví dụ IoU:**

| | Box A | Box B |
|---|---:|---:|
| Track 1 | 0.90 | 0.80 |
| Track 2 | 0.85 | 0.10 |

Ghép Track1→A và Track2→B: tổng 1.00. Ghép Track1→B và Track2→A: tổng 1.65. Nếu tối đa tổng IoU, **chọn cách hai**.

**Hungarian Algorithm:** Một thuật toán giải bài toán ghép cặp một-một tối ưu theo ma trận chi phí.

## 5. Các thuật toán trong bài

**SORT:** Ghép box bằng mô hình chuyển động, IoU và matching.

**DeepSORT:** Bổ sung đặc trưng ngoại hình.

**ByteTrack:** Giữ cả một số detection confidence thấp để ghép với các track hiện hữu trong lượt ghép sau.

**BoT-SORT:** Hỗ trợ bù chuyển động camera và thông tin ReID trong cấu hình phù hợp.

**OC-SORT:** Cải thiện liên kết khi che khuất và chuyển động khó.

**Không đúng:** “ByteTrack chắc chắn giữ ID dù xe biến mất mãi mãi”.

## 6. Occluded, Outside, Keyframe, Interpolation

**Occluded:** Xe vẫn trong cảnh nhưng bị xe tải che; xử lý theo guideline, không xóa ID tùy tiện.

**Outside:** Xe không còn xuất hiện trong đoạn track, ví dụ đã ra khỏi FOV; đánh dấu kết thúc đúng để tránh box treo.

**Keyframe:** Frame người đặt box hoặc trạng thái làm mốc.

**Interpolation:** Công cụ ước lượng box ở các frame giữa keyframe. Xe đi đường cong nhưng chỉ có hai mốc đầu-cuối có thể khiến box nội suy đi thẳng sai; cần thêm mốc.

## 7. FPS và Track Buffer [TRỌNG TÂM]

**FPS – Frames per Second:** Số frame trong 1 giây.

$$
\boxed{\text{Frame}=FPS\times\text{Giây}},\qquad
\boxed{\text{Giây}=\text{Frame}/FPS}
$$

**Track Buffer:** Giới hạn frame track bị mất được giữ theo cơ chế tracker.

**Ví dụ:** 25 FPS, che 3 giây =75 frame; track_buffer=50 nghĩa khoảng 2 giây → có nguy cơ mất ID cũ.

## 8. ID Switch, Fragmentation và box treo [TRỌNG TÂM]

**ID Switch (IDSW):** Hai xe chạy cắt nhau, xe đỏ từ ID 5 thành ID 8 và xe xanh từ ID 8 thành ID 5. Hai vật đổi ID cho nhau có thể được đếm **2 lần IDSW** trong ví dụ trên slide, mỗi vật một lần; phần mềm benchmark có quy tắc matching cụ thể.

**Track Fragmentation:** Cùng xe ID 5 bị che, hiện lại thành ID 9 → một vật bị chia track.

**Box treo:** Xe đã rời khung hình nhưng box vẫn xuất hiện.

## 9. MOTA [TRỌNG TÂM]

**MOTA – Multiple Object Tracking Accuracy:** Tổng hợp lỗi FN, FP, đổi ID.

$$
\boxed{MOTA=1-\frac{FN+FP+IDSW}{GT}}
$$

**GT:** Tổng **lần xuất hiện có nhãn của các vật** qua frame được chấm, không phải số xe duy nhất.

**Ví dụ:** 4 frame, mỗi frame 2 xe → GT=8. Nếu **đã đếm từ bảng so sánh** được FN=1, FP=1, IDSW=2:

$$
MOTA=1-(1+1+2)/8=50\%.
$$

Không thể tự suy ra FN=1, FP=1, IDSW=2 chỉ từ “4 frame × 2 xe”. Những lỗi đó phải **cho sẵn hoặc đếm từ kết quả tracker so với Ground Truth**.

**MOTA có thể âm** khi tổng lỗi vượt GT.

## 10. IDF1 [TRỌNG TÂM]

**IDF1 – Identification F1:** Đánh giá độ đúng khi liên kết danh tính qua chuỗi.

$$
\boxed{IDF1=\frac{2IDTP}{2IDTP+IDFP+IDFN}}
$$

- **IDTP:** Lần xuất hiện liên kết đúng identity.
- **IDFP:** Lần xuất hiện thừa/sai identity theo phép đánh giá.
- **IDFN:** Lần xuất hiện thật không được nhận đúng identity.

**Ví dụ:** IDTP=30, IDFP=10, IDFN=10 → IDF1=60/80=75%.

**MOTA cao, IDF1 thấp?** Có. Detector có thể tìm đủ box nhưng duy trì sai ID trong nhiều frame.

## 11. HOTA và ReID

**HOTA:** Đánh giá kết hợp chất lượng detection (**DetA**) và association (**AssA**). Tại một ngưỡng \(\alpha\):

$$
HOTA_\alpha=\sqrt{DetA_\alpha\,AssA_\alpha}.
$$

**ReID – Re-Identification:** Dùng **vector đặc trưng ngoại hình** để giúp nối lại cùng đối tượng. Với vector đã chuẩn hóa, độ tương đồng cosine tính bằng tích vô hướng:

$$
Similarity(A,B)=\sum_i A_iB_i.
$$

**Ví dụ:** A=(1,0), B=(0.8,0.6), C=(0.6,0.8) → B giống A hơn theo cosine. ReID vẫn nhầm hai xe cùng màu/mẫu và không tự bảo đảm khôi phục một track đã bị xóa khỏi bộ nhớ.

## 12. Ground Truth tracking và file MOT

Một nhãn tracking cần frame, class, box, Track ID và những thuộc tính do format quy định. Một dạng MOT trong slide:

~~~text
frame,id,x,y,w,h,flag,class,visibility
~~~

**Frame** để biết thời điểm, **id** để biết danh tính. Xuất ra format detection chỉ có box/class có thể không giữ được ID.

### Tự kiểm tra Day 03

1. 10 frame × 2 xe → GT bao nhiêu?
2. GT=20, FN=2, FP=1, IDSW=1 → MOTA?
3. Xe bị che sau xe buýt rồi hiện ra là Occluded hay Outside?
4. ByteTrack có giữ ID mãi mãi không?

---

# DAY 04 – KEYPOINT VÀ POSE

## 1. Keypoint, Skeleton, Topology, Pose

**Keypoint:** Một điểm mốc có tên và ý nghĩa cố định trên người hoặc vật: mũi, vai, khuỷu, gối.

**Skeleton:** Tập keypoint và các đường nối theo cấu trúc cơ thể.

**Topology:** Quy tắc xác định thứ tự keypoint và các điểm nối. Nếu 5 là vai trái thì không thể hôm sau tự coi 5 là khuỷu tay phải.

**Pose Estimation:** Ước lượng vị trí/tư thế cơ thể trong ảnh/video.

**Ví dụ:** Box cho biết một người ngồi trong xe; pose còn giúp biết vai, khuỷu và cổ tay ở đâu.

## 2. COCO-17 có những điểm nào? [TRỌNG TÂM]

| Index | Keypoint | Tiếng Việt |
|---:|---|---|
| 0 | nose | Mũi |
| 1 | left_eye | Mắt trái |
| 2 | right_eye | Mắt phải |
| 3 | left_ear | Tai trái |
| 4 | right_ear | Tai phải |
| 5 | left_shoulder | Vai trái |
| 6 | right_shoulder | Vai phải |
| 7 | left_elbow | Khuỷu tay trái |
| 8 | right_elbow | Khuỷu tay phải |
| 9 | left_wrist | Cổ tay trái |
| 10 | right_wrist | Cổ tay phải |
| 11 | left_hip | Hông trái |
| 12 | right_hip | Hông phải |
| 13 | left_knee | Đầu gối trái |
| 14 | right_knee | Đầu gối phải |
| 15 | left_ankle | Mắt cá chân trái |
| 16 | right_ankle | Mắt cá chân phải |

**Trái/phải** tính theo **cơ thể người được annotate**, không theo hướng người đứng nhìn màn hình.

## 3. Visibility v=0, 1, 2 [TRỌNG TÂM]

Trong quy ước COCO của bài:

- **v=2:** Nhìn thấy điểm rõ.
- **v=1:** Bị che nhưng vẫn gán điểm.
- **v=0:** Không gán nhãn điểm (trong slide thường minh họa ở ngoài khung).

**Ví dụ:** Cổ tay sau vô-lăng nhưng vẫn nằm trong cảnh theo rule → đặt điểm và đánh dấu bị che, không dùng v=0 chỉ vì khó nhìn.

**Dễ nhầm:** Occluded (bị che) **khác** Outside (ngoài khung/không gán theo rule).

## 4. Top-down vs Bottom-up

**Top-down:** Tìm người trước → cắt vùng → tìm 17 keypoint cho từng người. Nếu detector không tìm người, không có pose của người đó.

**Bottom-up:** Tìm các keypoint trên toàn ảnh → nhóm các điểm thành từng người. Khó khăn là ghép đúng điểm khi nhiều người đứng sát nhau.

## 5. Heatmap và OKS

**Heatmap:** Bản đồ cho thấy vùng nào có khả năng chứa một keypoint.

**OKS – Object Keypoint Similarity:** Metric đo mức khớp của keypoint dự đoán với GT theo khoảng cách, kích thước object và dung sai từng keypoint. Không phải IoU của person box.

**Ví dụ:** Sai 4 pixel ở một khớp dễ định vị có thể khác ý nghĩa với sai 4 pixel ở một khớp khó định vị. OKS có cơ chế xét dung sai.

## 6. Số giá trị một bộ pose

Mỗi điểm COCO biểu diễn bằng \((x,y,v)\) → 3 giá trị.

$$
17\times3=51\text{ giá trị keypoint}.
$$

Ví dụ nhãn YOLO Pose gồm 1 class + 4 giá trị box + 51 giá trị keypoint:

$$
1+4+51=56.
$$

Con số thực tế phụ thuộc định dạng export của model/task.

## 7. QC Pose trên CVAT

Dựng đúng Skeleton/Topology từ đầu; slide dùng mẫu SVG để tái sử dụng label. Khi label, kiểm tra **đảo trái/phải, sai index, điểm nằm ngoài hình, visibility, thiếu keypoint, đường nối vô lý**. Với video phải giữ track đúng. Thống kê visibility theo khớp để thấy người nào dùng v=0/v=1 sai quy tắc.

### Tự kiểm tra Day 04

1. COCO-17 có mấy điểm, mỗi điểm có mấy giá trị?
2. Cổ tay bị che bởi vô-lăng cần dùng v=1 hay v=0 theo bài?
3. Top-down bắt đầu bằng detection người hay tìm mọi khớp?

---

# DAY 05 – SEGMENTATION

## 1. Segmentation là gì và tại sao khác Bounding Box? [TRỌNG TÂM]

**Segmentation:** Bài toán quyết định **pixel nào thuộc đối tượng hoặc vùng nào**. Nếu bounding box chỉ vẽ một hình chữ nhật bao xe, segmentation tô đúng phần thân xe theo mức chi tiết của dữ liệu.

**Ví dụ:** Một cột điện rất mảnh. Hình chữ nhật bao cột chứa nhiều pixel trời và cây; mask giúp mô tả vùng cột chính xác hơn.

## 2. Semantic, Instance, Panoptic [TRỌNG TÂM]

**Semantic Segmentation:** Gán class cho từng pixel. Hai xe cùng class car có thể đều mang nhãn “car” mà không phân biệt cá thể.

**Instance Segmentation:** Vừa xác định vùng vừa phân biệt **từng cá thể**. Hai xe cùng class vẫn có instance ID khác nhau.

**Panoptic Segmentation:** Kết hợp thông tin lớp cho toàn cảnh và thông tin cá thể phù hợp. Vùng road có thể được biểu diễn như vùng stuff, các xe như instance.

**Ví dụ:** Ảnh có hai xe đỏ và một con đường. Semantic: pixel xe = car, pixel đường = road. Instance: xe thứ nhất và xe thứ hai có hai mask tách riêng.

**Lỗi:** Chọn nhầm yêu cầu semantic/instance ngay từ đầu có thể khiến cả dataset không đáp ứng mục tiêu.

## 3. Mask, Polygon, Brush và RLE

**Mask:** Ma trận mô tả vùng pixel; trong trường hợp nhị phân, 1 nghĩa là thuộc vùng, 0 nghĩa là không thuộc vùng.

**Polygon:** Đa giác nối nhiều đỉnh để bao vùng. Phù hợp vật có biên khá rõ.

**Brush:** Tô/xóa trực tiếp trên ảnh, tiện với đường biên phức tạp.

**RLE – Run-Length Encoding:** Cách mã hóa mask bằng các đoạn pixel chạy liên tiếp để giảm dung lượng.

**CVAT:** Có thể vẽ mask bằng polygon/brush hoặc dùng công cụ AI hỗ trợ; chỉ chọn shape và format mà guideline/export cho phép.

## 4. U-Net, Mask R-CNN và SAM khác nhau thế nào?

**U-Net:** Mạng encoder–decoder: encoder thu nhỏ biểu diễn ảnh để học đặc trưng; decoder mở rộng về kích thước để tạo mask. **Skip connections** nối các mức đặc trưng để hỗ trợ khôi phục chi tiết.

**Mask R-CNN:** Mô hình phát hiện từng đối tượng và dự đoán mask của từng instance.

**SAM – Segment Anything Model:** Hỗ trợ đề xuất mask theo prompt như điểm hoặc box theo khả năng của phiên bản công cụ. SAM giỏi đề xuất **vùng hình học**, nhưng **không tự đảm bảo đúng class và guideline**.

**Ví dụ:** Annotator bấm vào xe, SAM tô thân xe nhưng ăn cả bóng xe → reviewer phải sửa. Đây là lỗi vùng, dù mask trông rất đẹp.

## 5. Tensor, Logit, Sigmoid, Threshold, Loss

**Tensor:** Mảng số nhiều chiều. Ảnh màu có thể biểu diễn bằng tensor chiều cao × chiều rộng × ba kênh; mask nhị phân có thể chỉ gồm H×W.

**Logit:** Giá trị đầu ra thô trước khi biến thành điểm xác suất trong bài toán nhị phân.

**Sigmoid:** Hàm biến một số bất kỳ thành giá trị từ 0 đến 1. Đây là công thức nền tảng bổ sung để giải thích nội dung slide:

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Ví dụ z=0 → 0.5; z dương lớn → gần 1; z âm lớn → gần 0.

**Threshold:** Ngưỡng biến xác suất dự đoán thành mask. Ví dụ pixel score=0.8, ngưỡng=0.5 → pixel được tính vào foreground.

**Loss:** Giá trị được dùng trong quá trình model học. **Metric** là phép đo kết quả để báo cáo/so sánh. Loss thấp không tự chứng minh mask phù hợp yêu cầu vận hành.

Slide đề cập **BCEWithLogitsLoss**: hàm này kết hợp phần sigmoid và binary cross-entropy theo cách ổn định; không áp sigmoid trước loss đó như thể đầu vào đã là xác suất.

## 6. TP, FP, FN, TN ở cấp độ pixel

Chọn class “road” là dương tính:

- **TP:** Pixel thật là đường, dự đoán là đường.
- **FP:** Pixel thật không phải đường nhưng bị tô thành đường.
- **FN:** Pixel thật là đường nhưng bị bỏ sót.
- **TN:** Pixel không phải đường và được đoán đúng là không phải đường.

**Pixel Accuracy – Tỷ lệ pixel được phân loại đúng:**

$$
\boxed{Accuracy=\frac{TP+TN}{TP+TN+FP+FN}}.
$$

**Ví dụ:** 10.000 pixel, 9.900 pixel nền và 100 pixel người đi bộ. Model dự đoán toàn bộ là nền thì Accuracy=99% nhưng **Recall của lớp người bằng 0%**.

## 7. IoU, Dice và mIoU [TRỌNG TÂM]

**IoU mask:** Độ chồng lấp giữa vùng dự đoán và vùng GT.

$$
\boxed{IoU=\frac{TP}{TP+FP+FN}}.
$$

**Dice:** Một chỉ số khác về độ trùng khớp mask.

$$
\boxed{Dice=\frac{2TP}{2TP+FP+FN}}.
$$

**mIoU:** IoU trung bình các class được đánh giá.

$$
\boxed{mIoU=\frac{IoU_1+\cdots+IoU_C}{C}}.
$$

**Ví dụ:** TP=30, FP=10, FN=20:

$$
IoU=30/(30+10+20)=50\%.
$$

$$
Dice=60/(60+10+20)\approx66.7\%.
$$

**Quan hệ Dice–IoU (để hiểu thêm):**

$$
Dice=\frac{2IoU}{1+IoU}.
$$

Không được so Dice của mô hình A với IoU của B rồi kết luận một bên thắng nếu chưa thống nhất thước đo.

## 8. Sáu lỗi mask dễ gặp trong bài học

**Tràn biên:** Mask ăn ra vùng nền.

**Lỗ thủng:** Mask rỗng sai bên trong vùng cần tô.

**Mất chi tiết mảnh:** Bỏ sót phần mỏng như anten, gương theo guideline.

**Ăn bóng đổ:** Bóng trên đường bị tô nhầm thành thân xe.

**Merge:** Hai chiếc xe bị gộp vào cùng một instance.

**Split/Thiếu vùng:** Một object bị tách sai hoặc bỏ sót phần cần gán.

**Bài học:** Một mask có IoU khá cao vẫn có thể lỗi nghiêm trọng ở biên đường, vỉa hè hoặc vật nhỏ quan trọng.

## 9. Quy trình gán và QC mask

Xác định semantic/instance trong mục tiêu → chọn class/mask tool → vẽ thủ công hoặc SAM đề xuất → kiểm chỗ biên, lỗ, vật chạm nhau, bóng đổ → xuất nhãn → kiểm file đầu ra → đo IoU/Dice theo class và điều kiện khó.

**Tự kiểm tra Day 05:** (1) Hai xe cần hai mask độc lập thuộc semantic hay instance? (2) TP=20, FP=10, FN=10 thì Dice? (3) Vì sao accuracy 99% vẫn có thể bỏ sót hoàn toàn vật thể nhỏ?

---

# DAY 06 – DATA ANNOTATION LIFECYCLE VÀ QC

## 1. Chín bước vòng đời annotation chính thức trong bài [TRỌNG TÂM]

**Lifecycle** là một **vòng lặp**, không phải gán nhãn xong là kết thúc. Mỗi bước phải có **artifact** (đầu ra cụ thể) và **gate** (điều kiện chuyển tiếp).

| Bước | Tên trong slide | Giải thích bằng ví dụ |
|---:|---|---|
| 1 | Objective & Scope | Mục tiêu “phát hiện xe/người”; loại trừ biển báo nếu ngoài phạm vi |
| 2 | Collection & Selection | Lựa chọn clip đêm/ngày, nguồn camera và quyền sử dụng |
| 3 | Preparation & Preprocess | Kiểm file hỏng, tên file, tách dữ liệu, chuẩn metadata |
| 4 | Annotation Guideline | Quy tắc class, độ chặt box, occluded, ignore |
| 5 | Pilot Annotation | Cho người gán thử ca khó để sửa quy tắc |
| 6 | Annotation | Gán nhãn chính thức cho batch |
| 7 | QA/QC | Soát quy trình, nhãn, lỗi class/box/track và rework |
| 8 | Release | Xuất dataset version đã đạt ngưỡng, có bằng chứng |
| 9 | Monitoring & Feedback | Đọc lỗi sau bàn giao, cập nhật vòng sau |

**Ví dụ:** Hai annotator tranh cãi xe máy chở người có nên vẽ một hay hai box. Nếu guideline thiếu rule, **sửa guideline và pilot**, không chỉ ép họ sửa thủ công từng frame.

## 2. QA và QC khác nhau thế nào? [TRỌNG TÂM]

**QA – Quality Assurance:** Tạo và kiểm soát **quy trình** để phòng ngừa lỗi. Ví dụ xây guideline, đào tạo, calibration round, lựa chọn gold set.

**QC – Quality Control:** Kiểm tra **kết quả nhãn** để phát hiện và sửa lỗi. Ví dụ reviewer kiểm 100 frame và thấy 3 xe bị bỏ sót.

QA không loại trừ QC; cả hai cùng cần thiết.

## 3. Guideline, Taxonomy, Include/Exclude, Ambiguity

**Annotation Guideline:** Tài liệu quy định mọi người phải gán thế nào.

**Taxonomy:** Danh sách/lớp phân cấp class, ví dụ Vehicle > Car/Truck nếu dự án cho phép.

**Include:** Những object phải gán.

**Exclude/Ignore:** Những object hoặc vùng không gán/chấm theo quy tắc.

**Ambiguous Case:** Tình huống mơ hồ; ghi frame, ảnh và lý do, rồi **escalate** cho người chốt rule.

**Decision Log:** Nhật ký trường hợp đã được quyết định để người khác áp dụng nhất quán.

## 4. Pilot Annotation và Calibration Round

**Pilot:** Làm thử một batch nhỏ **trước khi triển khai diện rộng**, kiểm xem rule rõ chưa.

**Calibration Round:** Nhiều người gán cùng ví dụ rồi cùng thảo luận những điểm bất đồng.

**Ví dụ:** 100 ảnh pilot có 20 tình huống rider/pedestrian bị gán khác nhau. Kết quả pilot phải đưa ra rule nhất quán, không chỉ chấm “người này đúng, người kia sai” nếu guideline chưa rõ.

## 5. IAA – Inter-Annotator Agreement

**IAA:** Mức độ hai hay nhiều annotator **đồng ý** về annotation.

Một cách tính tỷ lệ thống nhất:

$$
\boxed{Agreement=\frac{\text{Ca hai người đồng ý}}{\text{Tổng ca đã so}}}.
$$

**Ví dụ:** 90 ca đồng ý/100 ca → 90%.

**Nhưng:** Hai người có thể cùng sai. **Agreement không phải Ground Truth, không phải Accuracy tuyệt đối**.

Với nhãn hình học, cần **match đúng object trước** rồi mới đo IoU giữa box của hai annotator. Đối tượng thiếu/thừa phải báo riêng, không trộn bừa vào IoU.

## 6. Cohen's Kappa và các thước đo liên quan

**Cohen's Kappa:** Đo đồng thuận giữa hai người sau khi điều chỉnh phần đồng thuận kỳ vọng do ngẫu nhiên:

$$
\boxed{\kappa=\frac{p_o-p_e}{1-p_e}}.
$$

- \(p_o\): Đồng thuận quan sát.
- \(p_e\): Đồng thuận kỳ vọng từ phân bố nhãn.

**Ví dụ:** \(p_o=0.9,\ p_e=0.5\):

$$
\kappa=(0.9-0.5)/(1-0.5)=0.8.
$$

**Fleiss' Kappa/Krippendorff's Alpha:** Các chỉ số IAA khác cho tình huống nhiều người và kiểu nhãn phù hợp; bài học nhắc tên để phân biệt mục đích, không phải lúc nào cũng yêu cầu tự tính.

## 7. Micro, Macro và Worst-Class Recall

**Recall mỗi class:**

$$
R_i=\frac{TP_i}{TP_i+FN_i}.
$$

**Micro Recall:** Gộp TP, FN trên tất cả class rồi tính:

$$
\boxed{MicroR=\frac{\sum TP_i}{\sum TP_i+\sum FN_i}}.
$$

**Macro Recall:** Trung bình Recall các class:

$$
\boxed{MacroR=\frac{R_1+R_2+\cdots+R_C}{C}}.
$$

**Worst-Class Recall:** Giá trị Recall thấp nhất trong những class được đánh giá.

**Ví dụ:** Car=95%, Person=80%, Motorcycle=50% → Macro=(95+80+50)/3=75%; Worst=50%.

**Bẫy:** Class car rất phổ biến có thể làm Micro đẹp dù motorcycle bị bỏ sót nhiều.

## 8. Gold Set, Review, Rework, Reviewer Recall

**Gold Set:** Một số ca có nhãn tham chiếu được rà soát tốt để đánh giá người gán/reviewer.

**Self-QC:** Người gán tự kiểm.

**Peer Review:** Người khác kiểm.

**Adjudication:** Người chịu trách nhiệm quyết định ca bất đồng dựa trên evidence và rule.

**Rework:** Gửi batch về sửa, sau đó kiểm lại.

**Reviewer Recall:** Phần lỗi tham chiếu mà reviewer tìm thấy:

$$
ReviewerRecall=\frac{\text{Lỗi phát hiện}}{\text{Tổng lỗi tham chiếu}}.
$$

Ví dụ 3/4 lỗi được phát hiện → 75%.

## 9. FPR và FDR – Hai tỷ lệ dễ nhầm

**FPR – False Positive Rate:** Trong những ca **sạch**, bao nhiêu bị báo nhầm?

$$
FPR=\frac{\text{Ca sạch bị flag sai}}{\text{Tổng ca sạch}}.
$$

**FDR – False Discovery Rate:** Trong **mọi cảnh báo**, bao nhiêu cảnh báo sai?

$$
FDR=\frac{\text{Cảnh báo sai}}{\text{Tổng cảnh báo}}.
$$

**Ví dụ:** 20 ca sạch có 2 ca bị báo nhầm → FPR=10%. Toàn bộ 5 cảnh báo có 2 báo nhầm → FDR=40%.

## 10. Release Gate, PASS/HOLD, Provenance, Audit

**Quality Gate:** Điều kiện định lượng/định tính để duyệt batch.

**PASS:** Đạt điều kiện; **HOLD:** Chưa phát hành vì lỗi hoặc thiếu bằng chứng.

**Manifest:** Bản kê file/nhãn/metadata.

**Dataset Version:** Phiên bản dữ liệu được quản lý.

**Provenance:** Dữ liệu đến từ đâu.

**Data Lineage:** Các bước xử lý dữ liệu đã trải qua.

**Audit Evidence:** Bằng chứng đo và quyết định QC, để người khác truy vết.

**Ví dụ:** Box IoU đẹp nhưng còn nhiều biển số chưa xử lý privacy theo chính sách → có thể phải HOLD.

**Tự kiểm tra Day 06:** (1) Pilot trước hay sau Annotation hàng loạt? (2) QA khác QC thế nào? (3) Agreement 95% có chứng minh nhãn đúng 95% không? (4) PASS/HOLD dùng để làm gì?

---

# DAY 07 – DATA PIPELINE VÀ GOVERNANCE

## 1. Data Pipeline là gì?

**Data Pipeline:** Chuỗi bước dữ liệu được thu thập, kiểm tra, chuyển đổi, lưu, chia tập và bàn giao để train/evaluate.

**Ví dụ:** Từ video 30 FPS → trích ảnh theo clip → giữ liên kết nhãn theo frame → chia dataset → xuất format YOLO → train.

**Failure mode:** File vẫn mở, model vẫn train nhưng metadata và class mapping đã sai.

## 2. Train, Validation, Test [TRỌNG TÂM]

**Train Set:** Dữ liệu model nhìn và dùng để học tham số.

**Validation Set:** Dữ liệu hỗ trợ lựa chọn mô hình/threshold, đánh giá trong phát triển.

**Test Set:** Dữ liệu tách riêng để đo chất lượng cuối cùng; không nên dùng đi dùng lại để chọn model.

**Data Split:** Cách chia tập, phải phù hợp tính độc lập thực tế của dữ liệu.

## 3. Data Leakage [TRỌNG TÂM]

**Data Leakage:** Khi thông tin thuộc tập phải giữ riêng hoặc thuộc tương lai/tập chấm bị lọt vào training/chọn cấu hình, khiến metric lạc quan giả.

**Ví dụ:** Frame 100 của một video trong train, frame 101 gần y hệt trong test. Mô hình được chấm trên thông tin quá giống dữ liệu đã học.

**Cách tránh theo bài:** Chia theo **clip/scene/group key**, kiểm ảnh trùng/gần trùng, không chia video liên tiếp thành từng frame random mà không xét quan hệ.

**Dễ nhầm:** Train 70%, Val 20%, Test 10% **không bảo đảm không leakage**; cách chọn nhóm quan trọng hơn tỷ lệ đẹp.

## 4. Các format và nguy cơ khi convert

**COCO:** Một hệ biểu diễn dataset thường dùng JSON, có thể chứa box, category, mask và keypoint.

**YOLO Detection:** Thường lưu class và box chuẩn hóa; khác YOLO Pose/Segmentation.

**CVAT Export:** Có thể giữ attribute, track, shape và metadata tùy format.

**Datumaro:** Công cụ hỗ trợ chuyển định dạng và quản lý dữ liệu.

**Ví dụ:** Chuyển CVAT Track sang format chỉ có box/class → **Track ID có thể mất**, dù số box giữ nguyên.

## 5. Round-trip Test và Class Mapping

**Round-trip Test:** Chuyển từ A sang B rồi đối chiếu hoặc chuyển ngược, kiểm thông tin có được bảo toàn không.

**Class Mapping:** Bảng ánh xạ class giữa các hệ. Nếu motorcycle vô tình map thành bicycle, model học sai nhưng file vẫn hợp lệ về cú pháp.

Cần so: số file, số object, phân phối class, geometry, Track ID, attributes, ignore zones và metadata quan trọng.

**Bẫy:** “Trước convert 100 box, sau 100 box” không chứng minh class hay tọa độ đúng.

## 6. Manifest, Checksum, Version, Provenance

**Manifest:** Danh sách file, split, class và metadata; giúp đối chiếu batch.

**Checksum/Hash:** Giá trị hỗ trợ phát hiện nội dung file đã đổi. Không chứng minh class annotation đúng.

**Version:** Nhãn/manifest/guideline có mốc phát hành rõ, để biết model train trên bản nào.

**Provenance:** Nguồn gốc, quyền sử dụng, thời gian thu thập.

**Lineage:** Dữ liệu đã được xử lý qua những bước nào.

## 7. Privacy, PII và Audit

**PII – Personally Identifiable Information:** Thông tin có thể dùng nhận diện người, ví dụ khuôn mặt/biển số trong phạm vi chính sách.

**Anonymization/Blurring:** Che/biến đổi để giảm khả năng nhận diện.

**Privacy Miss Rate:** Phần vùng PII thật bị bỏ sót:

$$
\boxed{MissRate=\frac{\text{Vùng PII bị sót}}{\text{Tổng vùng PII thật}}}.
$$

Ví dụ bỏ sót 2 mặt trên 200 mặt cần che → 1%.

**Governance:** Ai được xem file gốc, sử dụng trong mục đích gì, bao lâu, có log truy cập/version không.

## 8. Release Gate

Kiểm **Integrity → Format/Schema → Split/Leakage → Class Mapping → Privacy → Version/Evidence**. Một pipeline tốt phải phát hiện lỗi âm thầm chứ không chỉ chạy không crash.

**Tự kiểm tra Day 07:** (1) Vì sao chia frame liền nhau vào train/test dễ sai? (2) 100 box trước/sau có đủ chứng minh convert tốt không? (3) Hash giải quyết vấn đề gì?

---

# DAY 08 – AI ASSISTANCE VÀ ACTIVE LEARNING

## 1. Model-Assisted Annotation/Pre-label là gì? [TRỌNG TÂM]

**Pre-label:** AI tạo annotation đề xuất; con người kiểm tra, sửa và bổ sung thay vì vẽ từ đầu.

**Ví dụ:** Ảnh có 10 xe, AI tạo 8 box; reviewer cần chủ động tìm 2 xe bị sót, kiểm các box hiện có có sai class/biên không.

**Automation Bias:** Con người vì thấy AI làm sẵn nên dễ chấp nhận quá nhanh, đặc biệt bỏ sót vật nhỏ.

## 2. Các dạng AI hỗ trợ

**Detection Pre-label:** Model sinh box.

**Interactive Segmentation:** Annotator click hoặc đưa box, công cụ như SAM sinh mask đề xuất.

**Promptable Pre-label:** Dùng mô tả văn bản hoặc ảnh mẫu theo khả năng của model/phần mềm để tìm đối tượng.

**VLM – Vision-Language Model:** Mô hình xử lý ảnh và ngôn ngữ, có thể hỗ trợ diễn giải hoặc đề xuất thông tin nhưng vẫn cần người kiểm.

Không mặc định mọi phiên bản SAM/AI Tool có cùng tính năng.

## 3. Bốn trạng thái review pre-label [TRỌNG TÂM]

| Trạng thái | Giải thích | Ví dụ |
|---|---|---|
| **Accepted** | Giữ nguyên nhãn AI đã đúng | Box car chính xác |
| **Edited** | Có vật thật nhưng box/class/attribute AI sai | Sửa box lệch hoặc class sai |
| **Deleted** | AI tạo box trên thứ không thuộc nhãn thật | Xóa box ảo trên bóng đổ |
| **Added** | AI bỏ sót object cần nhãn | Vẽ thêm người đi bộ |

Đây là bốn hành động **trong quy trình review pre-label**. Không có nghĩa được xóa annotation thật hoặc sửa hàng loạt dữ liệu ngoài quyền được giao.

**AI Error Profile:** Thống kê số Accepted/Edited/Deleted/Added theo từng class, camera, ngày/đêm và điều kiện khó.

## 4. Residual Error và New Error

**Residual Error:** Lỗi AI còn sót sau khi người review vì không nhìn thấy hoặc chấp nhận sai.

**New Error:** Người reviewer làm nhãn vốn đúng thành sai.

**Ví dụ:** Box car AI đúng nhưng reviewer kéo lệch vào nền → New Error.

Vì vậy “tỷ lệ Accepted cao” không tự chứng minh reviewer làm tốt.

## 5. Synthetic Data và Sim-to-Real

**Synthetic Data:** Dữ liệu được tạo từ mô phỏng hoặc hệ thống sinh thay vì camera đường phố thực.

**CARLA:** Một môi trường mô phỏng giao thông được nhắc trong bài.

**Domain Randomization:** Thay đổi thời tiết, ánh sáng, vị trí hoặc kiểu cảnh để tăng đa dạng mô phỏng.

**Domain Shift:** Phân phối dữ liệu triển khai khác dữ liệu học.

**Sim-to-Real Gap:** Model chạy tốt trong mô phỏng nhưng kém ngoài đời vì có khác biệt về ánh sáng, thời tiết, noise hoặc vật thể.

**Ví dụ:** Người đi bộ mô phỏng rõ nét, nhưng người thật dưới mưa và đèn pha lóa khó hơn nhiều.

## 6. Active Learning là gì? [TRỌNG TÂM]

Có 100.000 ảnh chưa gán nhưng chỉ đủ kinh phí gán 5.000 ảnh. **Active Learning (AL)** tìm những ảnh có khả năng đem lại giá trị học lớn, thay vì chỉ chọn ngẫu nhiên.

Chu trình: train model ban đầu → chấm tập ảnh chưa label → chọn batch → con người gán → huấn luyện lại → so với random baseline.

**Uncertainty Sampling:** Chọn những ảnh model thiếu tự tin.

**Diversity Sampling:** Chọn nhiều kiểu cảnh để tránh batch gồm hàng trăm frame gần trùng nhau.

## 7. Least Confidence, Margin, Entropy

Các biểu thức dưới đây là **cách viết chuẩn để giải thích phương pháp được slide mô tả**, không có nghĩa tất cả đều là công thức bắt buộc thuộc.

**Least Confidence:**

$$
\boxed{U(x)=1-\max_i p_i}
$$

Nếu class cao nhất 0.55 → U=0.45; nếu 0.95 → U=0.05. Mẫu đầu bất định hơn theo thước đo này.

**Margin Sampling:**

$$
\boxed{M=p_{\mathrm{top1}}-p_{\mathrm{top2}}}
$$

Hai class 0.51/0.49 → margin 0.02, model chưa phân biệt rõ; 0.90/0.05 → margin 0.85.

**Entropy:**

$$
\boxed{H=-\sum_i p_i\log p_i}
$$

Xác suất càng rải đều trên nhiều class thì entropy thường càng cao. Không cần tự tính log bằng tay nếu chỉ học nhận diện khái niệm.

## 8. Ba bẫy của Active Learning

**Cold Start:** Model ban đầu còn quá yếu nên uncertainty không đáng tin.

**Collective Outlier:** Chọn cả đống ảnh lạ/noise nhưng không hữu ích cho mục tiêu.

**Model Coupling:** Chọn mẫu quá bám điểm yếu của một model khiến không tổng quát khi thay model.

**Small Objects:** Biển báo nhỏ, người đi bộ xa, xe máy cuối hàng thường dễ bị bỏ sót; cần đánh giá theo lát cắt thay vì chỉ score tổng.

**Tự kiểm tra Day 08:** (1) AI thiếu người đi bộ → trạng thái nào? (2) AI vẽ box trên nền không có vật → trạng thái nào? (3) Vì sao không lấy toàn bộ các frame có uncertainty cao từ cùng một đoạn video?

---

# DAY 09 – ROAD ELEMENTS

## 1. Road Elements là gì?

Đây là những yếu tố giao thông cần nhận dạng theo **hình học lẫn ý nghĩa vận hành**, gồm làn đường, vùng có thể lái, đèn và biển báo.

**Ego Vehicle:** Chiếc xe đang mang sensor và được dùng làm điểm tham chiếu của hệ thống.

**Ví dụ:** Box đèn xanh có thể rất chính xác nhưng gán sai “đèn điều khiển ego lane”, khiến hệ thống hiểu sai quy tắc giao thông.

## 2. Lane Marking và Polyline [TRỌNG TÂM]

**Lane Marking:** Vạch hoặc ranh giới liên quan đến làn đường.

**Polyline:** Chuỗi đỉnh nối thành đường, phù hợp vạch dài/cong/đứt.

**Solid Line:** Vạch liền. **Dashed Line:** Vạch đứt. **Single/Double:** Một/hai đường vạch.

**Lane Boundary:** Ranh giới làn. **Lane Centerline:** Đường giữa làn theo schema. **Curb/Road Edge:** Mép đường/bó vỉa; không tự động là lane marking.

**Tình huống:** Vạch bị ô tô che. Có nối xuyên qua hay tách polyline? Không có đáp án phổ quát: cần theo guideline về phần không nhìn thấy.

## 3. Drivable Area/Free Space [TRỌNG TÂM]

**Drivable Area:** Vùng mà ego được xem là có thể di chuyển theo điều kiện/schema của bài.

**Polygon:** Hình đa giác biểu diễn một vùng kín.

**Direct Drivable:** Vùng đường đi chính cho ego.

**Alternative Drivable:** Vùng thay thế có thể sử dụng theo định nghĩa dataset.

**Non-drivable/Ignore:** Vùng không thuộc vùng chạy hợp lệ hoặc nằm ngoài phạm vi đánh giá.

**Ví dụ:** Vỉa hè là một bề mặt có pixel nhìn thấy được nhưng không được tự tô thành vùng chạy trực tiếp của ô tô.

## 4. Traffic Light – State và Relevance [TRỌNG TÂM]

**State:** Red/Yellow/Green/Off/Unknown tùy schema.

**Relevance:** Đèn này có áp dụng cho làn/hướng đi của ego không?

**Temporal Consistency:** Trạng thái qua các frame phải hợp lý, không nhảy tùy ý.

**Ví dụ:** Đèn xanh của làn rẽ trái không có nghĩa xe ego đi thẳng được đi. Box đúng mà relevance sai vẫn là lỗi nghiêm trọng.

## 5. Traffic Sign – Phân cấp class

**Sign Family:** Regulatory, Warning, Guide, Temporary/Construction...

**Sign Type:** Loại cụ thể trong family.

**Exact Code/Text:** Mã hoặc chữ số khi đủ rõ theo quy tắc.

**Readable:** Có đọc được nội dung không?

**Relevant to Ego:** Biển có áp dụng cho đường/làn ego không?

**Occluded/Temporary:** Bị che/biển tạm thời.

**Ví dụ:** Biển tốc độ mờ chỉ đọc được “?0” không nên tự đoán chắc là 50 theo trí nhớ của người gán.

## 6. Road Elements QC

**Geometry QC:** Polyline bám vạch không, polygon tràn sidewalk không, box đèn/sign đúng phần cần vẽ không.

**Attribute QC:** Đúng state, class, direct/alternative, relevance không.

**Temporal QC:** Cùng đèn/biển/vạch qua frame có thay đổi thuộc tính vô lý không.

**Sampling:** Cần lấy mẫu khó như giao lộ, rẽ, vạch mờ, biển xa, ban đêm, mưa, glare. Một điểm accuracy tổng không cho thấy lỗi nào ảnh hưởng ego lane.

**Tự kiểm tra Day 09:** (1) Lane marking dùng polyline hay polygon? (2) Đèn xanh làn trái có tự liên quan ego đi thẳng không? (3) Vì sao asphalt không phải luôn là direct drivable?

---

# DAY 10 – DMS VÀ OMS

## 1. DMS khác OMS như thế nào? [TRỌNG TÂM]

**DMS – Driver Monitoring System (Giám sát tài xế):** Hệ thống dùng camera hướng vào cabin để quan sát trạng thái và hành vi người cầm lái: mắt nhắm kéo dài, buồn ngủ, ngáp, quay đầu, phân tâm.

**OMS – Occupant Monitoring System (Giám sát người trong xe):** Quan sát người ngồi trong cabin, vị trí, tư thế và các tình huống liên quan đến an toàn. Không chỉ nhìn tài xế.

**Ví dụ:** DMS hỏi “tài xế có đang nhìn đường và tỉnh táo không?”. OMS hỏi “những người trong cabin đang ở vị trí/tư thế nào, có tình huống an toàn cần lưu ý không?”.

**ADAS:** Nhóm hệ thống hỗ trợ lái, thường sử dụng cảm nhận môi trường bên ngoài. Camera DMS/OMS hướng vào trong, vì vậy còn có yêu cầu bảo vệ dữ liệu nhạy cảm.

## 2. Landmark khuôn mặt là gì?

**Facial Landmark:** Điểm mốc có vị trí và ý nghĩa cố định trên mặt, ví dụ góc mắt, mí mắt, sống mũi, viền môi. Khác box khuôn mặt vì nó mô tả cấu trúc chi tiết.

Bài DMS dùng **50 điểm khuôn mặt** theo cấu trúc:

| Bộ phận | Chỉ số | Số điểm |
|---|---|---:|
| Lông mày trái | 0–4 | 5 |
| Lông mày phải | 5–9 | 5 |
| Sống mũi | 10–13 | 4 |
| Mắt trái | 14–21 | 8 |
| Mắt phải | 22–29 | 8 |
| Môi ngoài | 30–41 | 12 |
| Môi trong | 42–49 | 8 |
| **Tổng** | **0–49** | **50** |

**Lưu ý:** Đây là chỉ số trong bộ thực hành DMS đã học, **không thay thế bằng các bộ 68 hoặc 98 điểm từ Internet**. Vị trí chính xác từng chỉ số phải theo topology của task.

## 3. EAR – Eye Aspect Ratio [TRỌNG TÂM]

**EAR là gì?** Tỷ lệ hình học cho biết mắt mở hay nhắm ở một frame, được tính bằng **hai khoảng cách dọc giữa mí trên/dưới** so với **khoảng cách ngang giữa hai góc mắt**.

Với sáu điểm quanh mắt đặt theo quy ước của công thức:

$$
\boxed{EAR=\frac{\|p_2-p_6\|+\|p_3-p_5\|}{2\|p_1-p_4\|}}.
$$

Trong đó:
- \(\|p_2-p_6\|\): Khoảng cách dọc cặp mí thứ nhất.
- \(\|p_3-p_5\|\): Khoảng cách dọc cặp mí thứ hai.
- \(\|p_1-p_4\|\): Khoảng cách ngang hai góc mắt.
- Ký hiệu \(\|a-b\|\) nghĩa là **khoảng cách** giữa hai điểm, không phải số âm.

**Ví dụ:** Hai khoảng cách dọc 4 px và 6 px; ngang 20 px:

$$
EAR=(4+6)/(2\times20)=10/40=0.25.
$$

**Ý nghĩa:** Mắt nhắm → khoảng cách mí dọc thường giảm → EAR giảm. Nhưng **một EAR thấp trong một frame không đủ chẩn đoán buồn ngủ**. Phải xét chuỗi thời gian, ngưỡng phù hợp, chất lượng landmark và tình trạng che khuất.

## 4. MAR – Mouth Aspect Ratio

**MAR:** Tỷ lệ hình học mô tả mức mở miệng, hỗ trợ nhận diện ngáp. Theo slide, nguyên lý tương tự EAR nhưng dùng điểm quanh môi.

Một cách viết **minh họa nguyên lý sáu điểm**:

$$
MAR=\frac{\|q_2-q_6\|+\|q_3-q_5\|}{2\|q_1-q_4\|}.
$$

**Quan trọng:** Phải xác định đúng các \(q_i\) trong bộ landmark cụ thể; **không lấy trực tiếp chỉ số điểm mắt gán sang điểm môi**. Có nhiều biến thể công thức MAR giữa những nghiên cứu/model.

Nếu MAR tăng mạnh và kéo dài khoảng thời gian phù hợp, đó là **tín hiệu hỗ trợ** nghi ngờ ngáp, không phải kết luận chắc chắn.

## 5. PERCLOS – Tỷ lệ thời gian nhắm mắt

**PERCLOS:** Chỉ số diễn tả tỷ lệ thời gian mắt được xem là nhắm trong một khoảng quan sát theo ngưỡng và cách xử lý chớp mắt của hệ thống.

Công thức **diễn giải đơn giản**:

$$
PERCLOS=\frac{\text{Thời gian mắt được xác định là nhắm}}{\text{Tổng thời gian quan sát hợp lệ}}\times100\%.
$$

**Ví dụ:** Trong 60 giây hợp lệ, mắt được tính là nhắm 6 giây → PERCLOS=10%.

**Dễ nhầm:** Chớp mắt bình thường ngắn không đồng nghĩa với “buồn ngủ”. Phải dùng temporal window và quy tắc của DMS.

## 6. Head Pose: Yaw, Pitch, Roll [TRỌNG TÂM]

**Yaw:** Quay đầu trái/phải (ví dụ nhìn về cửa sổ trái).

**Pitch:** Cúi đầu hoặc ngẩng đầu.

**Roll:** Nghiêng đầu về vai trái/phải.

**Gaze:** Hướng nhìn của mắt. Gaze **khác** hướng đầu: mắt có thể liếc ngang khi đầu vẫn thẳng.

**Từ keypoint đến behavior:** Landmark chỉ là tọa độ; cần biến thành **đặc trưng theo hình học và thời gian** như EAR, MAR, PERCLOS, head pose, sau đó mới áp quy tắc hoặc model nhận diện hành vi. Bài học nhắc ST-GCN như một cách học đặc trưng skeleton theo không gian và thời gian.

## 7. OMS, COCO-17, Seat Belt, Track và Group

**COCO-17 trong OMS:** Mỗi người có skeleton 17 điểm như Day 04.

**Seat-belt annotation:** Gắn nhãn dây nhìn thấy theo schema, có thể là các đoạn chéo/ngang hoặc dạng shape phù hợp. Không tự suy luận một người có đeo dây chỉ vì thấy một đoạn đen gần ngực.

**Track ID:** Duy trì danh tính cùng người qua nhiều frame.

**Group:** Liên kết những annotation cùng thuộc một người ở thời điểm/task được quy định, ví dụ bbox và skeleton. **Group không tự thay thế Track ID.**

**Occluded/Outside:** Điểm bị che vẫn thuộc vùng theo guideline; điểm ra khỏi khung/không có nhãn là tình huống khác.

**Privacy:** Dữ liệu mặt người trong cabin cần được quản lý quyền truy cập và mục đích sử dụng.

### Tự kiểm tra Day 10

1. DMS và OMS khác nhau về đối tượng quan sát thế nào?
2. 50 điểm khuôn mặt có bao nhiêu điểm môi trong và bao nhiêu điểm mắt?
3. EAR nhỏ một frame đã đủ khẳng định người lái buồn ngủ chưa?
4. Yaw và Gaze có đồng nghĩa không?

---

# DAY 11 – SURROUND VIEW VÀ FISHEYE

## 1. SVM, Fisheye và Distortion [TRỌNG TÂM]

**SVM – Surround View Monitoring:** Hệ thống quan sát xung quanh xe nhờ nhiều camera, thường hỗ trợ đỗ xe, lùi, lái tốc độ thấp.

**Fisheye Camera:** Camera góc rất rộng; vật ở gần rìa ảnh có thể cong, kéo giãn hoặc khác mạnh so với ảnh camera thường.

**Distortion:** Biến dạng hình học do ống kính và phép chiếu.

**Ví dụ:** Cùng một chiếc xe máy thẳng ngoài đời, hình của nó ở sát rìa ảnh fisheye có thể cong. **Một guideline dành cho ảnh camera thường không được áp máy móc lên ảnh fisheye**.

## 2. Raw, Rectified, Stitched, BEV

**Raw Fisheye:** Pixel gốc từ sensor, còn distortion.

**Rectified:** Ảnh được biến đổi để giảm/điều chỉnh distortion theo mô hình camera.

**Stitched/Composite Image:** Ghép ảnh của nhiều camera để tạo góc nhìn rộng; có thể có lỗi ở đường nối.

**BEV – Bird's-Eye View:** Biểu diễn từ trên nhìn xuống, cần phép chiếu và calibration phù hợp.

**Ví dụ:** Một ô tô có box trên ảnh fisheye raw khác box trên ảnh rectified. Nếu dùng tọa độ của không gian này trong không gian kia mà không transform → box lệch.

**Quan trọng:** Khi annotate, phải biết mình đang vẽ trong **không gian ảnh nào**.

## 3. Edge Zone, Lens Border, Seam và Overlap

**Edge Zone:** Vùng rìa ảnh, thường bị distortion mạnh.

**Lens Border/Vignetting:** Viền đen/vùng tối do hệ quang học.

**Ego Body:** Một phần thân xe tự che ảnh camera.

**Stitching Seam:** Ranh giới nơi ảnh nhiều camera được ghép, có thể xuất hiện vật bị cắt đôi, lặp hình hoặc thiếu vùng.

**Overlap Zone:** Vùng hai camera cùng quan sát. **Một xe xuất hiện trong hai camera khác nhau không tự động là duplicate annotation**.

**Ignore Region:** Vùng được quy định bỏ qua vì không hợp lệ/không thuộc phạm vi đánh giá.

## 4. Occluded khác Truncated

**Occluded:** Bị che bởi một vật khác (người đi sau cột điện).

**Truncated:** Bị cắt bởi **mép khung hình** hoặc giới hạn FOV.

**Ví dụ:** Đuôi xe nằm ngoài mép ảnh → truncated. Xe bị xe tải che một nửa giữa ảnh → occluded.

**Visible Extent:** Phần vật thể thật sự nhìn thấy trên pixel. Trong bài SVM, nguyên tắc box thường theo visible extent theo guideline, không vẽ theo “hình xe mình tưởng tượng” khi task không yêu cầu.

## 5. Annotation Road/Lane trong ảnh fisheye

**Free-space Polygon:** Phải dừng tại ranh giới quan sát hợp lệ, curb, parked vehicle hoặc seam theo rule; không tô xuyên qua vùng không thấy.

**Lane Polyline:** Không tự động nối mọi đoạn vạch bị che thành đường liên tục; cần theo quy tắc visibility.

**Bounding Box:** Theo vùng nhìn thấy và cờ edge_zone/truncated/occluded nếu schema yêu cầu.

## 6. Camera-specific QC [TRỌNG TÂM]

**Camera-specific QC:** Đánh giá chất lượng riêng theo camera trước/sau/trái/phải, không chỉ một con số chung.

**Gold Set theo camera:** Bộ mẫu tham chiếu phải có ảnh giữa/rìa, ngày/đêm, mưa, glare, vùng seam, vật nhỏ, lùi/đỗ.

**Metric phù hợp:**
- Box: IoU, Precision, Recall, lỗi class.
- Polygon free-space: IoU và boundary error gần ego.
- Lane polyline: sai lệch đường, thiếu đoạn, đầu mút.
- Attribute: tỷ lệ đúng của truncated, visibility, state, edge_zone.

**Metadata hữu ích:** camera_id, time/frame/sequence, calibration version, ignore mask, site, điều kiện sáng.

### Tự kiểm tra Day 11

1. Tại sao box ở ảnh fisheye raw và rectified có thể khác nhau?
2. Xe bị cắt bởi mép ảnh nên xem trạng thái truncated hay occluded?
3. Vì sao QC tổng hợp có thể che lỗi nặng ở camera sau?

---

# DAY 12 – LiDAR, POINT CLOUD VÀ CUBOID 3D

## 1. LiDAR là gì? [TRỌNG TÂM]

**LiDAR – Light Detection and Ranging:** Hệ đo khoảng cách bằng ánh sáng để lấy thông tin hình học 3D.

**Time-of-Flight:** Ý tưởng đo thời gian tín hiệu đi từ sensor đến vật rồi quay lại. Công thức vật lý **bổ sung để hiểu nguyên lý**:

$$
d=\frac{c\Delta t}{2},
$$

trong đó \(c\) là vận tốc ánh sáng, \(\Delta t\) thời gian khứ hồi. Chia 2 vì tín hiệu đi **hai chiều**.

**Camera:** Nhìn đặc trưng thị giác/màu/chữ qua pixel.

**LiDAR:** Đo khoảng cách và hình học của các điểm trong không gian.

**Radar:** Có khả năng cung cấp thông tin vận tốc xuyên tâm nhờ hiệu ứng Doppler trong các hệ thống thích hợp.

Không có cảm biến nào giải quyết mọi điều kiện một mình.

## 2. Point Cloud, XYZ, Ego Frame, Density

**Point Cloud:** Tập điểm rời trong không gian:

$$
P=(x,y,z).
$$

**X/Y/Z:** Tọa độ theo một hệ quy chiếu **phải được xác định trước**.

**Ego Frame:** Hệ tọa độ gắn với xe hoặc gốc chuẩn của hệ thống.

**Point Density:** Mức nhiều/ít điểm của một vùng. Vật càng xa hoặc bị che thường có ít điểm.

**Ground Plane:** Mặt đất làm tham chiếu cho vị trí đáy cuboid.

**Ví dụ:** Xe ở 100 m có thể chỉ có vài điểm, xe ở 10 m có cụm rất dày. Không có nghĩa xe khi ở xa nhỏ đi về kích thước thật.

**Dễ nhầm:** Point cloud **không phải ảnh** (lưới pixel đều) và **không phải mesh** (các điểm được nối thành mặt kín).

## 3. Ba góc nhìn [TRỌNG TÂM]

**Top/BEV:** Từ trên xuống → kiểm x, y, Length, Width, Yaw.

**Front:** Phía trước → kiểm bề ngang, Height, đáy.

**Side:** Phía bên → kiểm z, Height, vị trí đáy.

**Ví dụ:** Top view cho thấy xe dài 4 m, rộng 1.8 m nhưng không thể xác nhận chiều cao 1.6 m. Muốn kiểm H phải nhìn Front/Side.

## 4. Năm class Robotaxi 3D trong slide [TRỌNG TÂM]

| Class | Nghĩa theo bài học | Ví dụ |
|---|---|---|
| **Vehicle** | Phương tiện cơ giới nhiều bánh trên đường | Car, Truck, Bus |
| **Two-wheels** | Phương tiện hai bánh | Motorcycle, Bicycle |
| **Pedestrian** | Người đi/đứng | Người đi bộ |
| **Animal** | Động vật | Chó, mèo, gia súc |
| **Obstacle** | Vật cản ngoài bốn nhóm trên | Rào, cọc, vật rơi |

**Không copy taxonomy ngoài** như KITTI/nuScenes vào dự án khi chưa map class theo guideline chính thức.

## 5. 3D Cuboid có bảy tham số [TRỌNG TÂM]

**3D Cuboid:** Hình hộp mô tả tâm, kích thước và hướng của vật thể 3D:

$$
\boxed{(x,y,z,L,W,H,yaw)}
$$

- \(x,y,z\): Tâm cuboid theo hệ quy chiếu.
- \(L\): Length – chiều dài dọc vật.
- \(W\): Width – chiều rộng ngang.
- \(H\): Height – chiều cao.
- \(yaw\): Góc hướng của đầu vật thể quanh trục đứng.

**Ground Contact:** Ràng buộc đáy hộp với mặt đất, **không phải tham số thứ tám**.

**Ví dụ:** Một chiếc ô tô có center (10,2,1), L=4.2m, W=1.8m, H=1.6m, yaw=0° theo trục/quy ước của dự án.

## 6. Yaw / Heading: hướng đầu xe khác hướng chuyển động

**Yaw:** Hướng của thân/đầu vật thể, đọc trên Top View.

**Heading:** Hướng đầu xe theo quy ước annotation, không bắt buộc là hướng vật đang chạy.

**Ví dụ:** Xe đang lùi; đầu xe vẫn hướng về phía trước thân xe nhưng vận tốc lại theo hướng ngược. Xe đỗ cũng có heading dù vận tốc bằng 0.

**Lỗi lật 180°:** Hộp cùng vị trí/kích thước nhưng mũi tên ngược đầu xe. Với point cloud rất thưa có thể khó phân biệt đầu/đuôi; cần evidence từ camera/sequence hoặc escalate.

## 7. Ground Contact, Oversize, Undersize

**Ground Contact:** Đáy hộp phải phù hợp mặt đất/bề mặt đỡ vật thể theo guideline.

**Box nổi:** Đáy cao hơn mặt đất một cách không hợp lý.

**Box xuyên đất:** Đáy thấp hơn mặt đất.

**Oversize/Undersize:** Dài/rộng/cao quá nhiều hoặc quá ít so với đối tượng.

Nếu **z được định nghĩa là tâm hình học**, quan hệ hình học minh họa:

$$
z_{\mathrm{bottom}}=z-\frac{H}{2}.
$$

Không dùng công thức này nếu hệ thống định nghĩa z là tọa độ đáy.

## 8. Thứ tự vẽ cuboid

1. Nhìn cả scene, xác định Ego Frame, ground, các cụm điểm.
2. Chọn class theo năm nhóm/guideline.
3. Top View: đặt center x-y, L/W, yaw.
4. Side/Front: kiểm z/H và đáy.
5. Kiểm kích thước và orientation ở nhiều góc.
6. Save, ghi chú case khó, escalate nếu không đủ evidence.

### Tự kiểm tra Day 12

1. Bảy tham số cuboid là gì?
2. Top View có kiểm được Height chính xác không?
3. Xe chạy lùi có nhất thiết yaw ngược lại không?
4. Ít điểm LiDAR ở xa có chứng minh xe thật nhỏ hơn không?

---

# DAY 13 – 3D OBJECT DETECTION VÀ PRE-LABEL

## 1. Sáu bài toán thường gặp trên Point Cloud

**Classification:** Phân loại object/tập điểm.

**Detection:** Tìm đối tượng, class và cuboid 3D trong scene.

**Segmentation:** Gán class cho điểm hoặc tách instance điểm.

**Tracking:** Giữ identity theo thời gian.

**Registration:** Căn chỉnh các lần quét/tập point cloud trong một hệ tọa độ.

**SLAM – Simultaneous Localization and Mapping:** Vừa ước lượng vị trí, vừa xây dựng bản đồ.

**Ví dụ:** Registration giúp ghép hai lần quét cho khớp, không tự bảo đảm nhãn Vehicle/Pedestrian đúng.

## 2. Point-based, Voxel-based, Pillar-based [TRỌNG TÂM]

**Point-based:** Xử lý trực tiếp các điểm hoặc nhóm điểm. Ví dụ PointNet/PointNet++; giữ nhiều chi tiết vị trí điểm.

**Voxel-based:** Chia không gian thành các ô 3D gọi là voxel. Ví dụ VoxelNet; phù hợp xử lý khối không gian có cấu trúc.

**Pillar-based:** Gom điểm thành các cột theo mặt phẳng ngang, tạo biểu diễn BEV rồi xử lý hiệu quả bằng mạng 2D. Ví dụ PointPillars.

**Center-based:** Tìm tâm object trên không gian biểu diễn rồi hồi quy chiều cao, kích thước, heading. Ví dụ CenterPoint.

**Đánh đổi:** Giữ nhiều chi tiết 3D có thể nặng tính toán; gom thành pillar nhanh hơn ở một số pipeline nhưng có thể làm mất bớt chi tiết chiều cao.

**Transformer/Fusion:** Các hướng cải thiện xử lý quan hệ rộng hoặc kết hợp dữ liệu cảm biến; không phải “class nhãn mới”.

## 3. Model 3D Detection dự đoán gì?

Thông thường có:

**Class + Score + (x,y,z,L,W,H,yaw)**.

Ví dụ score=0.95 nghĩa model tự tin về dự đoán theo cách tính score; **không đảm bảo** đáy, yaw, class đều đúng.

## 4. LiDAR-based, Camera-based và Sensor Fusion

**LiDAR-based 3D Detection:** Đọc thông tin khoảng cách/hình học 3D từ point cloud.

**Camera-based 3D Detection:** Ước lượng cấu trúc 3D từ ảnh, có thể khó xác định khoảng cách với một camera đơn.

**Sensor Fusion:** Kết hợp nhiều cảm biến/đặc trưng để cải thiện khả năng quan sát.

Trong thực hành Day 13, **đặt và QC cuboid 3D theo point cloud và guideline**, còn ảnh camera hỗ trợ đối chiếu class/đầu–đuôi; không kéo L/W/H cho “vừa pixel” mà bỏ qua hình học.

## 5. Pre-label bằng pretrained model [TRỌNG TÂM]

**Pretrained Detector:** Model đã được học trước, sinh nhãn đề xuất.

**Pre-label Workflow:** Point Cloud → Model → Dự đoán Cuboid → Human Review → Sửa nhãn → QC.

Reviewer kiểm:
- **Missing Object:** Bỏ sót vật thể.
- **False Positive:** Hộp trên cụm không phải vật cần label.
- **Wrong Class:** Chọn sai class.
- **Wrong Center:** Tâm sai.
- **Wrong Dimensions:** L/W/H không đúng.
- **Wrong Yaw:** Đầu xe sai.
- **Wrong Ground Contact:** Box nổi/xuyên đất.

**Ví dụ:** PointPillars đề xuất cuboid đúng xe nhưng yaw quay 180° → vẫn phải sửa, dù score rất cao.

## 6. Chuyển hệ tọa độ Z trong pipeline minh họa [TRỌNG TÂM]

Slide sử dụng phép biến đổi Z trước model:

$$
\boxed{z_{\mathrm{model}}=z_{\mathrm{source}}-z_{\mathrm{ground}}-h_{\mathrm{sensor}}}
$$

Và chiều ngược sau dự đoán:

$$
\boxed{z_{\mathrm{source}}=z_{\mathrm{model}}+h_{\mathrm{sensor}}+z_{\mathrm{ground}}}
$$

- \(z_{\mathrm{source}}\): Z gốc của dữ liệu/box.
- \(z_{\mathrm{ground}}\): Mốc ground ước lượng.
- \(h_{\mathrm{sensor}}\): Độ cao sensor theo convention pipeline.
- \(z_{\mathrm{model}}\): Z trong hệ đưa cho model.

**Ví dụ:** z_source=3; z_ground=0.5; h_sensor=1.5 → z_model=1.0. Đổi ngược: 1+0.5+1.5=3.

**Cực quan trọng:** Đây là **quy tắc của pipeline minh họa**, không phải phép chuyển đúng cho mọi sensor/dataset. Phải kiểm file calibration/convention thật.

## 7. Lỗi lệch Z hàng loạt

**Một box lệch** có thể do người gán hoặc điểm quá thưa. **Nhiều box cùng lệch một khoảng Z** qua nhiều frame/class là dấu hiệu đáng nghi lỗi chuyển đổi hệ tọa độ.

**Cách xử lý:** Dừng nghi ngờ chất lượng cả batch, kiểm bước chuyển tọa độ, metadata, phép đổi ngược; không sửa thủ công hàng loạt từng box để giấu lỗi pipeline.

### Tự kiểm tra Day 13

1. PointPillars thuộc nhóm point, voxel hay pillar?
2. Score 0.95 có bảo đảm yaw đúng không?
3. Tất cả cuboid nổi khỏi mặt đất cùng 1.5m thì kiểm cái gì trước?

---

# DAY 14 – 3D TRACKING VÀ CAMERA–LiDAR CALIBRATION

## 1. 3D Tracking là gì? [TRỌNG TÂM]

**3D Tracking:** Theo dõi cùng vật thể qua nhiều timestamp bằng **3D cuboid + Track ID nhất quán**.

**Một object – một identity:** Một chiếc xe không được tạo ID mới tùy tiện chỉ vì đi sau vật che hoặc LiDAR ít điểm trong một frame.

**Ví dụ theo slide:** Track 94 giữ kích thước vật cứng **L=4.19m, W=2.04m, H=1.82m** từ frame ít điểm đến frame dày điểm. Độ thưa điểm thay đổi, kích thước thật không thay đổi.

## 2. Các trạng thái một track

**Enter:** Object xuất hiện/đủ evidence để đưa vào track theo guideline.

**Visible:** Đủ điểm và không bị che đáng kể.

**Occluded:** Bị che; có thể thiếu một phần/nhiều điểm, nhưng không tự biến thành object mới.

**Reappear:** Vật xuất hiện lại sau đoạn che, giữ cùng ID nếu có bằng chứng đúng danh tính.

**Exit:** Object ra khỏi vùng theo dõi, kết thúc track phù hợp.

**Lưu ý:** Không được đoán vị trí 3D vô căn cứ ở frame không đủ evidence; làm theo guideline và ghi nhận ca khó.

## 3. Bảy lỗi Temporal [TRỌNG TÂM]

| Tên lỗi | Khái niệm | Ví dụ |
|---|---|---|
| **ID Switch** | Hai object đổi nhầm tên/ID | Hai xe cắt nhau, ID 3 và ID 7 đổi cho nhau |
| **Merge** | Hai vật bị gộp một cuboid/track | Hai xe máy sát nhau thành một box |
| **Fragmentation** | Một vật bị tách nhiều track | Xe ID 5 bị che, hiện lại ID 9 |
| **Dimension Drift** | Kích thước L/W/H trôi bất hợp lý | Xe dài 4.0m rồi 4.7m dù là cùng vật |
| **Orientation Drift** | Heading trôi/lật bất hợp lý | Yaw nhảy 180° khi chỉ còn vài điểm |
| **Sparsity Error** | Box co/nhảy/mất vì điểm thưa | Xe xa bị vẽ rất nhỏ theo cụm ít điểm |
| **Velocity Jump** | Vị trí nhảy làm vận tốc ước lượng phi lý | Box nhảy nhiều mét chỉ sau một frame |

**Phân biệt nhanh:** Merge = **hai vật thành một**; Fragmentation = **một vật bị tách nhiều đoạn**; IDSW = **đổi nhầm danh tính**.

## 4. Keyframe và Interpolation

**Keyframe:** Frame người gán trực tiếp quyết định box/mốc vị trí/hướng.

**Interpolation:** Công cụ nội suy trạng thái ở frame giữa.

**Ví dụ:** Xe rẽ cua, hai keyframe đầu/cuối nằm trên hai nhánh đường; nội suy thẳng có thể khiến hộp chạy xuyên giữa góc cua. Cần thêm keyframe ở đoạn rẽ và kiểm **cả frame nội suy**.

## 5. Calibration là gì? Intrinsic vs Extrinsic [TRỌNG TÂM]

**Calibration:** Xác định các tham số hình học để đưa dữ liệu các sensor sang hệ tọa độ phù hợp, giúp chiếu LiDAR lên ảnh.

**Camera Intrinsic:** Các tham số bên trong mô hình camera như tiêu cự \(f_x,f_y\), tâm ảnh \(c_x,c_y\), distortion.

**Sensor Extrinsic:** Quan hệ **xoay và tịnh tiến** giữa các hệ tọa độ (LiDAR, ego, camera).

**Ví dụ:** Camera nghiêng 4° so với thông tin calibration cũ. Khi overlay điểm LiDAR lên ảnh, **nhiều object và cả cảnh cùng lệch một hướng**. Đó là dấu hiệu hệ thống chứ không phải hàng trăm box riêng lẻ cùng sai.

## 6. LiDAR → Ego → Camera → Pixel

**LiDAR Frame:** Hệ tọa độ từ sensor LiDAR.

**Ego Frame:** Hệ tọa độ chuẩn của xe.

**Camera Frame:** Hệ tọa độ của camera.

**Pixel Frame:** Tọa độ ảnh (u,v).

Chuỗi trong bài:

~~~text
Điểm LiDAR (mét)
    ↓ T_LiDAR→Ego
Điểm Ego (mét)
    ↓ T_Ego→Camera
Điểm Camera (mét)
    ↓ chia độ sâu, nhân tiêu cự, cộng tâm ảnh, xử lý distortion
Pixel trên ảnh (u,v)
~~~

Cách viết gọn bằng ma trận thuần nhất:

$$
P_{\mathrm{camera}}=
T_{\mathrm{ego}\to\mathrm{camera}}
T_{\mathrm{LiDAR}\to\mathrm{ego}}
P_{\mathrm{LiDAR}}.
$$

**Bẫy:** Không tự lấy nghịch đảo ma trận khi file đã cung cấp đúng chiều transform; thứ tự nhân phải theo convention.

## 7. Chiếu điểm 3D lên ảnh 2D [TRỌNG TÂM]

Với tọa độ **trong hệ camera** \((X,Y,Z)\), mô hình camera lỗ kim (chưa tính méo ống kính):

$$
\boxed{u=f_x\frac{X}{Z}+c_x},\qquad
\boxed{v=f_y\frac{Y}{Z}+c_y}.
$$

Trong đó:
- \(f_x,f_y\): Tiêu cự trên hai trục ảnh, tính bằng pixel.
- \(c_x,c_y\): Tâm ảnh.
- \(X,Y,Z\): Tọa độ trong hệ camera; \(Z\) là độ sâu theo convention.
- \(u,v\): Tọa độ pixel.

**Ví dụ:** X=2, Y=0, Z=10; fx=fy=1000; cx=500, cy=300:

$$
u=1000\times(2/10)+500=700
$$

$$
v=1000\times(0/10)+300=300.
$$

Kết quả pixel **(700,300)**.

**Vì sao chia Z?** Cùng kích thước thật nhưng vật càng xa → trên ảnh càng nhỏ. Độ sâu rất quan trọng.

**Không thể suy ngược duy nhất:** Một pixel (u,v) không cho biết chính xác Z vì nhiều điểm 3D nằm trên cùng tia chiếu có thể rơi một pixel.

## 8. Khi Camera và LiDAR không khớp [TRỌNG TÂM]

**FOV – Field of View:** Object có điểm LiDAR nhưng nằm ngoài khung camera đang xem. Chọn camera khác nếu cần; không tự xóa vật thật khỏi point cloud.

**Occlusion:** Sensor này có thể bị chắn nhưng sensor khác vẫn nhìn được vì vị trí và hướng khác nhau.

**Sparse Points:** Camera nhìn rõ xe, LiDAR chỉ có ít điểm. Không kéo L/W/H theo kích thước pixel rồi làm sai cuboid.

**Calibration Error:** Mọi object bị lệch overlay có hệ thống cùng hướng qua nhiều frame. Kiểm calibration, timestamp và transform trước, không kéo từng box che lỗi.

## 9. Radar và tốc độ

**Radial Velocity:** Thành phần vận tốc dọc theo hướng nhìn của radar. Không nhất thiết bằng tốc độ toàn phần nếu xe đi chéo.

**Velocity Jump QC:** Nếu cùng một Track ID đột ngột nhảy vị trí thì vận tốc giữa hai frame có thể phi thực tế.

Công thức khoảng cách chia thời gian là **cách diễn giải vật lý bổ sung**:

$$
v=\frac{\sqrt{(\Delta x)^2+(\Delta y)^2+(\Delta z)^2}}{\Delta t}.
$$

Ví dụ object đổi vị trí 6m sau 0.1s → vận tốc suy ra 60m/s; nếu cảnh thật không thể như vậy, cần kiểm lại cuboid hoặc track.

## 10. Quy trình QC một 3D sequence

Xem cả sequence từ Enter đến Exit → kiểm Track ID → chốt kích thước ở frame điểm dày → kiểm Yaw ở Top → kiểm H và ground contact ở Side/Front → rà Interpolation → đối chiếu camera đúng FOV và calibration → phân loại lỗi → ghi **frame, track ID, sensor/camera, dấu hiệu, evidence** và escalate.

**Không làm:** Co hộp vì frame ít điểm; xoay mũi tên 180° tùy tiện; xóa object LiDAR vì không thấy trên một camera; kéo mọi cuboid theo ảnh để giấu lỗi calibration.

### Tự kiểm tra Day 14

1. Xe xa ít điểm có được tự co L/W/H?
2. Hai object bị một cuboid/ID bao chung là lỗi gì?
3. Intrinsic khác Extrinsic ở đâu?
4. Trong công thức chiếu 3D→2D, tại sao phải chia cho Z?
5. Cả cảnh overlay lệch cùng một hướng thì nghi lỗi gì?

---

# SỔ TAY CÔNG THỨC VÀ BÀI TẬP

## A. Công thức cần luyện trước

**1. Bounding Box – Day 02**

$$
w=x_2-x_1,\qquad h=y_2-y_1.
$$

$$
x_c=(x_1+x_2)/2,\qquad y_c=(y_1+y_2)/2.
$$

$$
(x_n,y_n,w_n,h_n)=
(x_c/W,\ y_c/H,\ w/W,\ h/H).
$$

**2. IoU – Day 02 và 05**

$$
IoU=\frac{Giao}{Hợp},\qquad Hợp=A+B-Giao.
$$

**3. Precision / Recall – Day 02**

$$
Precision=\frac{TP}{TP+FP},\qquad
Recall=\frac{TP}{TP+FN}.
$$

**4. mAP – Day 02**

$$
mAP=\frac{AP_1+\cdots+AP_C}{C}.
$$

**5. FPS – Day 03**

$$
Frames=FPS\times Seconds.
$$

**6. MOTA – Day 03**

$$
MOTA=1-\frac{FN+FP+IDSW}{GT}.
$$

**7. IDF1 – Day 03**

$$
IDF1=\frac{2IDTP}{2IDTP+IDFP+IDFN}.
$$

**8. Dice / mIoU – Day 05**

$$
Dice=\frac{2TP}{2TP+FP+FN},\qquad
mIoU=\frac{\sum IoU_i}{C}.
$$

**9. Cohen's Kappa – Day 06**

$$
\kappa=\frac{p_o-p_e}{1-p_e}.
$$

**10. EAR – Day 10**

$$
EAR=\frac{\|p_2-p_6\|+\|p_3-p_5\|}{2\|p_1-p_4\|}.
$$

**11. Hệ tọa độ Z – Day 13**

$$
z_m=z_s-z_g-h_s,\qquad z_s=z_m+h_s+z_g.
$$

**12. Chiếu 3D→2D – Day 14**

$$
u=f_xX/Z+c_x,\qquad v=f_yY/Z+c_y.
$$

**Một số công thức khác chỉ cần hiểu khái niệm trước:** HOTA, cosine similarity/ReID, Pixel Accuracy, Sigmoid, Micro/Macro Recall, Least Confidence, Margin, Entropy, MAR/PERCLOS theo định nghĩa dataset, vận tốc.

## B. 28 câu tự kiểm tra: hai câu mỗi ngày

Hãy tự làm trước khi mở đáp án. Đề có cả khái niệm, tình huống và tính toán.

1. **Day 01:** ML và DL có quan hệ như thế nào?
2. **Day 01:** Ground Truth khác Prediction, và GT có thể sai không?

3. **Day 02:** XYXY=(20,30,120,90). Đổi sang COCO XYWH và CENTER XYWH.
4. **Day 02:** TP=40, FP=10, FN=20. Tính Precision và Recall.

5. **Day 03:** Video có 5 frame, mỗi frame 2 xe; FN=1, FP=1, IDSW=2. MOTA?
6. **Day 03:** Xe bị che rồi hiện lại thành ID mới dù là cùng xe. Lỗi gì?

7. **Day 04:** COCO visibility v=1 nghĩa là gì?
8. **Day 04:** COCO-17 có bao nhiêu giá trị nếu mỗi điểm chứa x,y,v?

9. **Day 05:** Hai ô tô cùng class nhưng cần mask độc lập: Semantic hay Instance?
10. **Day 05:** TP=20, FP=10, FN=10. Tính IoU và Dice.

11. **Day 06:** Pilot Annotation làm trước hay sau annotation hàng loạt?
12. **Day 06:** IAA 95% đã đủ chứng minh Ground Truth đúng 95% chưa?

13. **Day 07:** Vì sao frame liên tiếp cùng video có thể gây leakage nếu chia random?
14. **Day 07:** Convert CVAT Track sang format chỉ box/class dễ mất thông tin nào?

15. **Day 08:** AI vẽ box trên bóng xe không có object hợp lệ. Reviewer chọn trạng thái gì?
16. **Day 08:** AI bỏ sót người đi bộ, trạng thái review nào?

17. **Day 09:** Lane marking thường biểu diễn bằng polyline hay polygon?
18. **Day 09:** Đèn xanh của làn rẽ trái có tự relevant cho ego đang đi thẳng không?

19. **Day 10:** Bộ DMS của bài có bao nhiêu facial landmark, bao nhiêu điểm môi trong?
20. **Day 10:** Một EAR thấp duy nhất đã chứng minh tài xế buồn ngủ chưa?

21. **Day 11:** Xe bị cắt bởi mép khung hình: Occluded hay Truncated?
22. **Day 11:** Vì sao cần gold set và QC theo từng camera?

23. **Day 12:** Liệt kê bảy tham số cuboid.
24. **Day 12:** Xe chạy lùi: yaw có nhất thiết cùng hướng vận tốc không?

25. **Day 13:** PointPillars là họ point-based, voxel-based hay pillar-based?
26. **Day 13:** Mọi box đều cao hơn mặt đất 1.5m: sửa từng box hay kiểm pipeline trước?

27. **Day 14:** fx=1000; X=2; Z=10; cx=500 → tọa độ u bằng bao nhiêu?
28. **Day 14:** Cả overlay Camera–LiDAR lệch đồng đều sang một bên: nghi điều gì?

<details>
<summary><strong>Bấm để xem đáp án + giải thích ngắn</strong></summary>

1. DL là nhánh ML; ML là nhánh AI.
2. GT là nhãn tham chiếu; Prediction do model sinh. GT vẫn có thể sai, cần QC.
3. COCO=(20,30,100,60); CENTER=(70,60,100,60).
4. Precision=40/(40+10)=80%; Recall=40/(40+20)=66.7%.
5. GT=5×2=10; MOTA=1−(1+1+2)/10=60%.
6. Track Fragmentation.
7. Điểm được gán nhãn nhưng bị che.
8. 17×3=51.
9. Instance Segmentation.
10. IoU=20/(20+10+10)=50%; Dice=40/(40+10+10)=66.7%.
11. Trước; để phát hiện và sửa guideline trước khi triển khai hàng loạt.
12. Không; hai annotator có thể cùng sai.
13. Vì frame quá giống nhau, làm test không đủ độc lập.
14. Track ID và có thể attribute/metadata không có trong format đích.
15. Deleted (trong phạm vi pre-label được phép review).
16. Added.
17. Polyline.
18. Không; phải xét relevance theo ego lane/hướng đi.
19. 50 điểm; môi trong 8.
20. Không; cần xét chuỗi thời gian/ngưỡng và các yếu tố khác.
21. Truncated nếu do biên ảnh; Occluded là bị vật khác che.
22. Vì mỗi camera có góc nhìn, distortion, edge-zone, failure mode riêng.
23. x,y,z,L,W,H,yaw.
24. Không. Yaw/heading mô tả hướng thân/đầu vật, không bắt buộc cùng hướng vận tốc.
25. Pillar-based.
26. Kiểm hệ tọa độ, transform Z, sensor/pipeline trước; không kéo cả batch để giấu lỗi.
27. u=1000×(2/10)+500=700.
28. Lỗi Calibration/Transform hoặc đồng bộ sensor, cần kiểm có hệ thống.

</details>

## C. 12 cặp khái niệm tuyệt đối không nhầm

| Khái niệm 1 | Khái niệm 2 | Khác biệt |
|---|---|---|
| AI | ML | AI là lĩnh vực rộng; ML học quy luật từ dữ liệu |
| Ground Truth | Prediction | Nhãn tham chiếu vs dự đoán của model |
| Classification | Detection | Class vs Class + vị trí |
| COCO XYWH | CENTER XYWH | Tọa độ góc vs tọa độ tâm |
| Precision | Recall | Trong dự đoán có bao nhiêu đúng vs trong vật thật tìm được bao nhiêu |
| NMS Threshold | Evaluation IoU Threshold | Loại box trùng vs chấm khớp GT |
| Detection | Tracking | Từng frame vs identity xuyên thời gian |
| Occluded | Outside | Bị che vs không còn hiện diện trong đoạn track |
| ID Switch | Fragmentation | Đổi nhầm ID vs một vật bị tách nhiều đoạn |
| Semantic | Instance | Lớp pixel vs cá thể riêng |
| QA | QC | Thiết kế đảm bảo quy trình vs kiểm kết quả |
| Intrinsic | Extrinsic | Tham số camera vs vị trí/hướng giữa các sensor |

## D. Thứ tự ôn tập gợi ý

**Lượt 1 – Xây nền:** Day 01–05: đọc tất cả khái niệm và tự giải thích bằng lời.  
**Lượt 2 – Tính toán:** Bounding Box, IoU, Precision, Recall, mAP, FPS, MOTA, IDF1, Dice, EAR.  
**Lượt 3 – Quy trình:** Day 06–09: guideline, QA/QC, pipeline, active learning, ý nghĩa road element.  
**Lượt 4 – Sensor/3D:** Day 10–14: DMS/OMS, fisheye, point cloud, cuboid, temporal errors, projection.  
**Lượt 5 – Tự kiểm tra:** Làm 28 câu mà không mở đáp án, quay lại đúng đoạn sai.

---

## Nguồn và giới hạn

Biên soạn theo các tài liệu Level 2 đã có trong buổi học:

- **Day 01:** *Hiểu về dữ liệu*.
- **Day 02:** *Detector hoạt động thế nào, xây từ con số 0, đo ra sao*.
- **Day 03:** *Tracker hoạt động thế nào, đo ra sao, xây từ con số 0*.
- **Day 04:** *Keypoint & Pose*.
- **Day 05:** *Segmentation — Từ Box Sang Mask*; *Từ pixel đến quyết định, từ U-Net số 0 đến AI hỗ trợ*.
- **Day 06:** *Quy Trình Dữ Liệu và Kiểm Soát Chất Lượng*; *Chín bước của Data Annotation Lifecycle*; *Từ mục tiêu đến release có thể audit*.
- **Day 07:** *Data Pipeline & Governance*.
- **Day 08:** *AI Data Assistance and Active Learning*.
- **Day 09:** *Road Elements: Lane, Drivable Area, Traffic Light, Traffic Sign*.
- **Day 10:** *DMS & OMS — Từ Keypoint Đến Hành Vi*.
- **Day 11:** *SVM / 360° External Cameras*.
- **Day 12:** *Data Annotation — Point Cloud & Cuboid*.
- **Day 13:** *3D Object Detection: từ point cloud đến cuboid*.
- **Day 14:** *Robotaxi B — Across Time & Sensors*.

Tài liệu diễn giải lại để dễ học, **không chép nguyên văn slide và không bao gồm Level 3**. Các ví dụ số và công thức vật lý/toán học bổ sung phục vụ giải thích. **Guideline của dự án là nguồn chuẩn** cho class, yaw, occlusion, hệ tọa độ, format, thang đo và quyền thao tác trên annotation thật.

**HẾT – VINUNI AI20K LEVEL 2 DAY 01–14.**
