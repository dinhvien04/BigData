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
