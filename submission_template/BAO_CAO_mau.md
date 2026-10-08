# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** làm cá nhân **Thành viên:** Đặng Đỉnh Đoàn (2A202602927)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

Chạy trên Colab (T4), Python 3.10. Cả năm file `video_1.txt` … `video_5.txt` chạy đủ frame (600 / 1050 / 837 / 900 / 750), frame cuối có hộp khớp số ảnh.

## 1. Cấu hình đã chọn

Mỗi video: tracker nộp, `conf`, `iou`, quan sát, và cấu hình đã thử rồi loại.

**Cách đọc cột "Quan sát".** Với `video_1` là số do TrackEval tính. Với `video_2`–`video_5` là số đo từ chính file nộp (số hộp mỗi frame, số ID, độ dài track). Đây là chỉ báo gián tiếp về độ ổn định ID. Bài này **chưa xem lại video bản nộp để đối chiếu bằng mắt**, nên các nhận xét về ID đổi màu hay hộp nhảy sang người khác chưa được kiểm tra.

| Video | Tracker | conf | iou | Quan sát | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.3 | 0.7 | HOTA 29.97, MOTA 19.03, IDF1 29.70, 33 lần đổi ID. Bỏ sót nhiều: chỉ khớp 22.5% người trong nhãn (14402 người bị bỏ sót, chỉ 611 hộp giả). | bytetrack 0.3/0.5: HOTA 26.91, thấp nhất. botsort conf 0.5: HOTA 27.17, bỏ sót thêm (khớp 16.5%). |
| video_2 (phố đêm, tĩnh, rất đông) | botsort | 0.3 | 0.5 | 11.7 hộp/frame dù cảnh rất đông. 67 ID cho 1050 frame, track trung vị 102 frame, 24% track dưới 15 frame. 44 trong 67 ID xuất hiện ở nửa đầu video. | Đã chạy thử 5 tracker và botsort ở conf 0.15 / 0.5, iou 0.4 / 0.7 (150 frame đầu). Chưa có số để so, chưa loại cấu hình nào bằng số. |
| video_3 (camera di động, ảnh nhỏ) | botsort | 0.3 | 0.5 | 5.7 hộp/frame, **167 ID** cho 837 frame, track trung vị chỉ 13 frame, 56% track dưới 15 frame: nhiều track ngắn, ID bị tách. | Như video_2. |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.3 | 0.5 | 6.9 hộp/frame, 73 ID, track trung vị 45 frame, 32% track ngắn. Người ở gần, hộp cao trung vị 458 px. | Như video_2. |
| video_5 (trên xe bus, giao lộ đông) | botsort | 0.3 | 0.5 | Chỉ 4.1 hộp/frame, người nhỏ (hộp cao trung vị 136 px), confidence trung bình thấp nhất (0.56). 77 ID, 40% track ngắn. | Như video_2. |

**Lưu ý về độ tin cậy.** Cấu hình `botsort / 0.3 / 0.5` của bốn video không nhãn là điểm xuất phát, không phải kết quả so sánh có số. Bảng `video_1` cho thấy botsort ở cấu hình này đứng đầu hoặc sát đầu trên một cảnh có nhãn, nhưng các cảnh còn lại khác `video_1` nhiều (đêm, camera di chuyển, trong nhà, rung lắc). Số liệu so sánh 5 tracker trên bốn video này chưa được dùng để chọn.

## 2. Số liệu video_1

Bản nộp (`botsort`, `conf` 0.3, `iou` 0.7), do `scripts/evaluate_practice.py` in:

```
HOTA: nop_bai_video1-pedestrian    HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA
COMBINED                           29.969    18.408    49.061    19.176    74.385    52.38     80.959    83.019    30.624

CLEAR: nop_bai_video1-pedestrian   MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag
COMBINED                           19.025    80.817    19.202    22.491    87.244    14.516    17.742    67.742    4179      14402     611       33        9         11        42        105

Identity: nop_bai_video1-pedestrian IDF1      IDR       IDP       IDTP      IDFN      IDFP
COMBINED                           29.703    18.68     72.463    3471      15110     1319
```

Cách chọn cấu hình nộp (quy tắc đặt trước: lấy HOTA cao nhất; mỗi lần chỉ đổi một số):

| Tracker | conf | iou | HOTA | DetA | AssA | MOTA | IDF1 | Đổi ID |
|---|---|---|---|---|---|---|---|---|
| bytetrack | 0.3 | 0.5 | 26.91 | 15.07 | 48.13 | 17.29 | 25.71 | 12 |
| ocsort | 0.3 | 0.5 | 27.45 | 17.88 | 42.35 | 19.81 | 28.73 | 42 |
| botsort | 0.3 | 0.5 | 29.46 | 18.10 | 48.22 | 19.81 | 29.35 | 25 |
| strongsort | 0.3 | 0.5 | 28.66 | 17.71 | 46.60 | 19.70 | 29.85 | 41 |
| deepocsort | 0.3 | 0.5 | 27.38 | 17.84 | 42.21 | 19.76 | 27.80 | 51 |
| botsort | 0.15 | 0.5 | 29.34 | 19.24 | 45.11 | 20.73 | 29.56 | 27 |
| botsort | 0.5 | 0.5 | 27.17 | 14.30 | 51.65 | 15.25 | 24.56 | 10 |
| botsort | 0.3 | 0.4 | 29.32 | 17.36 | 49.63 | 19.47 | 29.82 | 19 |
| **botsort** | **0.3** | **0.7** | **29.97** | 18.41 | 49.06 | 19.03 | 29.70 | 33 |

Các cấu hình botsort ở `conf` 0.15 và 0.3 chỉ chênh nhau dưới 1 điểm HOTA, và mỗi cấu hình chạy một lần. Không nên xem `iou` 0.7 hơn `iou` 0.5 là chắc chắn. Nếu chọn theo MOTA thì `conf` 0.15 (20.73) thắng.

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

**video_1 (tĩnh, ban ngày, vừa đông).** Điểm bị chặn chủ yếu bởi detector chứ không phải tracker: `DetA` chỉ khoảng 18 và hơn 14000 trong 18581 hộp nhãn bị bỏ sót, nên MOTA chỉ khoảng 19 dù `AssA` tới 49. Khung hình 1920×1080 bị thu về 640 px nên người nhỏ khó phát hiện (giải thích này là suy luận từ kích thước ảnh, chưa kiểm chứng bằng thí nghiệm). Trong đó tracker vẫn tạo khác biệt thấy được. Trong năm tracker ở cấu hình gốc, ByteTrack có `DetA` thấp nhất (15.07) và nhiều người bị bỏ sót nhất (15249), nên HOTA thấp nhất dù chỉ đổi ID 12 lần, ít nhất cả nhóm. ByteTrack chỉ dùng chuyển động nhưng vẫn giữ `AssA` 48.1, ngang botsort, nên ở cảnh tĩnh chuyển động đủ để nối track; còn OCSort cũng chỉ dùng chuyển động mà `AssA` chỉ 42.4, nên Re-ID không phải yếu tố duy nhất. OCSort và DeepOCSort có `AssA` thấp nhất (khoảng 42), 42 và 51 lần đổi ID, tức hay gán nhầm khi người cắt nhau. Botsort và StrongSORT giữ `AssA` cao hơn (48 và 47) nhờ Re-ID, và botsort ít đổi ID hơn StrongSORT (25 so với 41). Ba kết luận này là số chạy một lần trên một video 600 frame, chưa có lặp lại.

**video_3 (camera di chuyển, ảnh nhỏ).** Số liệu chỉ báo cho thấy dấu hiệu ID bị tách vụn: 167 ID cho trung bình 5.7 hộp mỗi frame, track trung vị chỉ 13 frame, 56% track dưới 15 frame (so với 24% ở video_2 cảnh tĩnh). Khi camera di chuyển, vị trí hộp thay đổi theo camera nên tracker chỉ dùng chuyển động dễ mất liên kết; Re-ID có lợi vì không phụ thuộc vị trí. Đây là giả thuyết từ số liệu và từ nguyên lý, chưa so với ByteTrack hay OCSort trên chính video này.

**video_2 (đêm, tĩnh, rất đông).** Chỉ 11.7 hộp mỗi frame trong cảnh rất đông cho thấy detector bỏ sót nhiều (đêm, người nhỏ, che khuất). Tracker có Re-ID có thể bị ngoại hình đêm tối làm nhiễu; camera tĩnh thì chuyển động vẫn dùng được. Không có số nào ở đây cho thấy Re-ID hơn hay kém chuyển động, nên chọn botsort là chưa có căn cứ riêng cho cảnh này.

## 4. Nếu có thêm thời gian

Xem lại cả năm video ở các đoạn người cắt nhau và đoạn camera quay nhanh để kiểm tra các nhận xét dựa trên số đo ở trên, rồi chạy bản nộp full-frame của bốn video không nhãn với tracker mỗi video chọn theo mắt. Lặp mỗi cấu hình `video_1` vài lần để biết chênh 0.5 điểm HOTA có thật không. Phần mở rộng (không thuộc bài nộp chính vì đổi luật chơi): cắt ảnh thành từng ô trước khi detect hoặc tăng `imgsz`, vì `video_1` cho thấy bỏ sót mới là nút thắt.
