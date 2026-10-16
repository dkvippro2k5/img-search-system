# Kịch bản sử dụng và pain list của hệ thống tìm ảnh

Tài liệu tổng hợp các tình huống có thể xảy ra với hệ thống tìm ảnh người nổi tiếng Việt Nam của capstone, xuất phát từ pain point của người dùng. Nguồn gồm nghiên cứu về hành vi tìm ảnh, các hệ thống lớn đang vận hành và giới hạn đã biết của mô hình. Mỗi kịch bản có hành vi mong đợi và cách kiểm thử, để chuyển thẳng thành test case (xem `PLAN.md` mục 5).

Ví dụ dùng tên có thật trong dữ liệu. Quan hệ giữa người trong ví dụ multi-hop chỉ là dạng truy vấn; đáp án phải lấy từ đồ thị tri thức có bằng chứng, không lấy từ tài liệu này.

---

## 1. Bằng chứng từ nghiên cứu và hệ thống thực tế

**Người dùng tìm ảnh với nhiều ý định khác nhau, và mức hài lòng phụ thuộc ý định.**
Xie và cộng sự (WSDM 2018) chia ý định tìm ảnh thành ba nhóm: khám phá/tìm hiểu, giải trí, và định vị/lấy ảnh cụ thể. Hành vi khác nhau rõ giữa các nhóm (thời gian click đầu, sửa truy vấn, thời gian xem); nhóm "tìm ảnh cụ thể" sửa truy vấn nhiều hơn. Phân loại ý định của tìm kiếm web không áp dụng nguyên cho tìm ảnh. [1]
→ Hệ thống cần phục vụ cả "xem nhiều ảnh về một người" lẫn "tìm đúng một tấm ảnh".

**Truy vấn thất bại thường là truy vấn cụ thể.**
Pu (2008) phân tích truy vấn tìm ảnh thất bại: chúng cụ thể hơn truy vấn thành công và dựa nhiều vào khái niệm (ý nghĩa) hơn đặc điểm thị giác; tìm theo nội dung ảnh chỉ giải quyết được một phần nhỏ. Một phân tích log Excite (2001) cho thấy 26% truy vấn tìm ảnh ban đầu bị sửa lại. [2]
→ Truy vấn dài, nhiều điều kiện là chỗ dễ trượt; caption và metadata quan trọng hơn chỉ embedding ảnh.

**Kết quả ảnh hiển thị dạng lưới, người dùng xem khác danh sách.**
Xie và cộng sự (WWW 2019): người dùng chú ý vùng giữa, xem chậm dần theo hàng và có thể bỏ qua cả hàng; chỉ số dạng danh sách (DCG) tương quan kém với mức hài lòng ở tìm ảnh. Tsukuda và Goto (SIGIR 2020) thấy mức hài lòng và chỉ số dạng lưới bị ảnh hưởng bởi ý định tìm kiếm. [3]
→ Ảnh đúng nhất phải nằm ở hàng đầu; khi báo cáo nên cân nhắc chỉ số theo lưới bên cạnh nDCG.

**Truy vấn mơ hồ hoặc nhiều khía cạnh cần đa dạng hóa kết quả.**
Các phương pháp như xQuAD và chỉ số α-nDCG, intent-aware precision đánh giá việc phủ nhiều ý định. Máy tìm kiếm đa dạng hóa tốt cho truy vấn có nhiều nghĩa, nhưng kém cho truy vấn một nghĩa nhiều khía cạnh. [4]
→ Áp dụng trực tiếp cho multi-intent: đo độ phủ ý định, không chỉ độ chính xác.

**Tìm theo tên người luôn gặp trùng tên.**
Bài toán Web People Search (WePS) tồn tại vì tên người rất mơ hồ; kết quả tìm một tên thường trộn nhiều người khác nhau. [5]
→ Dữ liệu có ví dụ thật: "Sơn Tùng" (nhà văn) và "Sơn Tùng M-TP" (ca sĩ).

**Mô hình ảnh–văn bản kiểu CLIP có lỗi đã biết.**
CLIP hoạt động gần như "túi từ": khó gắn đúng thuộc tính với đối tượng khi có nhiều đối tượng. Truy vấn phủ định ("không có", "không đeo kính") thường trả về chính đối tượng bị loại trừ, một phần vì từ phủ định chỉ chiếm khoảng 0,7% caption huấn luyện. Đếm số lượng cũng là điểm yếu được nêu. [6]
→ Không dựa vào dense retrieval cho phủ định và số lượng; xử lý bằng phân tích truy vấn hoặc báo không hỗ trợ.

**Tìm ảnh bằng ngôn ngữ tự nhiên ở quy mô lớn vẫn khó.**
Google tạm dừng mở rộng "Ask Photos" năm 2025 vì chưa đạt về độ trễ, chất lượng và trải nghiệm; sau đó kết hợp tìm kiếm cổ điển (trả kết quả ngay) với Gemini xử lý nền. [7]
→ Bài học trực tiếp: đường nhanh bằng chỉ mục cổ điển trước, mô hình lớn chỉ bổ sung; không để LLM chặn đường trả kết quả.

**Pinterest: chi phí và độ trễ là ràng buộc gốc.**
Pinterest nhấn mạnh hệ thống phải rẻ khi kho ảnh lớn và dùng xếp hạng nhiều tầng (model nhẹ lọc từ hàng triệu xuống hàng nghìn, model nặng xếp lại) để cân bằng độ trễ và chất lượng. [8]

**Tiếng Việt không dấu: tăng recall nhưng dễ nhầm nghĩa.**
Bỏ dấu giúp khớp truy vấn gõ không dấu, nhưng các từ khác nghĩa trùng nhau khi bỏ dấu; cách làm phổ biến là khớp cả hai dạng, ưu tiên khớp có dấu và hiển thị văn bản gốc. [9]

**Ảnh thật nhưng sai ngữ cảnh.**
NewsCLIPpings (EMNLP 2021) xây dữ liệu ảnh và chú thích đều thật nhưng ghép sai ngữ cảnh; đây là dạng thông tin sai khó phát hiện. [10]
→ Ảnh lấy theo tiêu đề bài báo có thể không phải người trong tiêu đề; caption VLM cũng có thể sai.

**Kết quả ảnh có thể khuếch đại định kiến.**
Kay và cộng sự (CHI 2015) thấy kết quả ảnh cho nghề nghiệp phóng đại định kiến giới và thiếu đại diện nữ; Pew thấy 57% nghề trông "nam hơn" thực tế trong kết quả ảnh. [11]
→ Truy vấn theo nghề ("doanh nhân", "nhà khoa học") cần kiểm tra phân bố kết quả.

**Tìm theo khuôn mặt gây lo ngại quyền riêng tư.**
PimEyes bị khiếu nại vì xử lý dữ liệu sinh trắc học và có thể bị lạm dụng để theo dõi người khác. [12]
→ Không cung cấp chức năng "tải ảnh mặt lên để tìm người"; giới hạn trong người nổi tiếng của dữ liệu.

**Ý định theo thời gian.**
Kim và Xing (WSDM 2013) cho thấy yếu tố thời gian giúp suy ra ý định ẩn trong tìm ảnh; nghiên cứu truy vấn web ước tính khoảng 7% truy vấn có ý định thời gian ngầm. [13]
→ "Mỹ Tâm hồi nhỏ" và "Mỹ Tâm hiện tại" là hai nhu cầu khác nhau.

**Không có kết quả là điểm rời bỏ.**
Nguồn ngành thương mại điện tử (chủ yếu là blog của nhà cung cấp, độ tin cậy vừa phải) khuyên theo dõi tỷ lệ truy vấn rỗng và luôn đưa gợi ý thay vì trang trống. [14]

---

## 2. Pain list của người dùng

| Mã | Pain point | Biểu hiện | Nguồn gốc trong hệ thống |
|---|---|---|---|
| P1 | Không tìm thấy | Kết quả rỗng hoặc thiếu ảnh có thật | Không dấu, sai chính tả, biệt danh, entity phân mảnh, caption thiếu |
| P2 | Sai người | Ảnh người trùng tên, người giống, người đứng cạnh | Liên kết entity sai, nhãn gốc sai, ảnh nhóm |
| P3 | Hiểu sai ý định | Bỏ mất một vế, trộn hai vế, hiểu "và" thành "cùng nhau" | Tách ý định, CLIP kiểu túi từ |
| P4 | Kết quả lặp lại | Cùng một ảnh (resize/crop) chiếm nhiều ô | Dedupe chưa đủ, ảnh trùng giữa nhiều entity |
| P5 | Chất lượng ảnh kém | Mờ, nhỏ, watermark, mặt quá nhỏ | Tầng kiểm tra chất lượng của Phần A |
| P6 | Nội dung không phù hợp | Ảnh nhạy cảm xuất hiện cho truy vấn bình thường | Kiểm duyệt bỏ sót |
| P7 | Chậm | Chờ lâu, nhất là truy vấn phức tạp | Gọi LLM đồng bộ, không cache, tài nguyên yếu |
| P8 | Không tin được | Multi-hop trả ảnh nhưng không biết vì sao; quan hệ bịa | Thiếu bằng chứng, LLM tự trả lời sự kiện |
| P9 | Sai thời điểm | Muốn ảnh hiện tại nhưng ra ảnh cũ hoặc ngược lại | Không có thông tin thời gian |
| P10 | Sai ngữ cảnh | Ảnh đúng người nhưng sai sự kiện được nêu | Caption/tiêu đề sai, ảnh minh họa trong bài |
| P11 | Lo ngại riêng tư / đạo đức | Hệ thống dùng để tìm người thường, ảnh nhạy cảm của người thật | Phạm vi chức năng, chính sách hiển thị |
| P12 | Thiên lệch | Truy vấn nghề nghiệp chỉ ra một nhóm người | Phân bố dữ liệu, xếp hạng theo độ phổ biến |

---

## 3. Kịch bản chi tiết

Cột "Ưu tiên": **B** = bắt buộc có test case trong capstone, **N** = nên có, **T** = tùy chọn / ghi vào hạn chế.

### 3.1 Nhập liệu và ngôn ngữ
| ID | Kịch bản | Ví dụ | Hành vi mong đợi | Kiểm thử | Pain | Ưu tiên |
|---|---|---|---|---|---|---|
| L1 | Gõ không dấu | "den vau", "ho ngoc ha" | Ra đúng person như khi có dấu | Cặp có/không dấu cho cùng ground truth; so Recall@10 | P1 | B |
| L2 | Sai chính tả nhẹ | "Hồ Ngọc Hạ", "Sơn Tùng MTP" | Sửa về person gần nhất, hiện "Bạn có muốn tìm…" | Biến thể sai 1–2 ký tự | P1 | B |
| L3 | Biệt danh / tên khác | "Hà Hồ", "Đen", "Sếp Tùng" | Liên kết qua bảng alias | Alias có nguồn (Wikidata/tiêu đề) | P1 | N |
| L4 | Truy vấn rất ngắn, mơ hồ | "ca sĩ" | Trả ảnh đa dạng nhiều người, không lặp một người | Đo số person khác nhau trong top-20 | P3, P12 | N |
| L5 | Truy vấn dài, nhiều điều kiện | "Hồ Ngọc Hà mặc váy đỏ trên sân khấu năm 2019" | Ưu tiên ràng buộc người; điều kiện không thỏa được thì nới lỏng và nói rõ | Kiểm tra giải thích "đã bỏ điều kiện…" | P1, P3 | N |
| L6 | Trộn tiếng Anh | "Sơn Tùng concert", "Hari Won red carpet" | Hiểu cả hai ngôn ngữ | Cặp truy vấn Việt/Anh cùng ground truth | P1 | N |
| L7 | Phủ định | "Trấn Thành không đeo kính" | Không hứa hẹn; nếu không hỗ trợ thì báo rõ thay vì trả ảnh có kính | Kiểm tra thông báo | P3 | T |
| L8 | Số lượng | "ảnh ba người", "hai cầu thủ" | Dùng `face_count` nếu có; không thì báo hạn chế | Truy vấn có số | P3 | T |
| L9 | Ký tự lạ, rỗng, quá dài, chèn mã | "", "<script>", 2.000 ký tự | Trả lỗi hợp lệ, không crash, không thực thi | Kiểm thử đơn vị API | P7 | B |

### 3.2 Danh tính và thực thể
| ID | Kịch bản | Ví dụ | Hành vi mong đợi | Kiểm thử | Pain | Ưu tiên |
|---|---|---|---|---|---|---|
| E1 | Tên đầy đủ, người nổi tiếng | "Đen Vâu" | Ảnh của person, không bỏ ảnh nằm ở entity mô tả khác | Ground truth = mọi ảnh approved của person sau gộp | P1 | B |
| E2 | Trùng tên khác người | "Sơn Tùng" | Hỏi lại hoặc chia nhóm: ca sĩ / nhà văn | Kiểm tra cả hai person xuất hiện, có nhãn phân biệt | P2 | B |
| E3 | Tên + nghề để phân biệt | "nhà văn Sơn Tùng" | Chỉ ra người đúng nghề | Precision@10 | P2 | B |
| E4 | Entity phân mảnh | "Phạm Nhật Vượng" (có nhiều entity mô tả) | Gộp thành một person | So kết quả trước/sau gộp entity | P1 | B |
| E5 | Ảnh nhóm có nhiều người | "Hari Won" ra ảnh gia đình Trấn Thành – Hari Won | Cho phép, nhưng xếp sau ảnh chân dung | Vị trí ảnh nhóm trong top-k | P2 | N |
| E6 | Người không có trong dữ liệu | Tên một người bình thường bất kỳ | Báo không có, không trả người gần giống | Truy vấn ngoài phạm vi; tỷ lệ trả rỗng đúng | P1, P11 | B |
| E7 | Ảnh trùng giữa nhiều entity | Cùng ảnh trong hai entity cùng người | Hiện một lần | Đếm trùng sha256 trong top-k | P4 | B |
| E8 | Nhãn gốc sai | Entity có nhiều ảnh bị duyệt nhầm | Ghi nhận vào hạn chế; dùng điểm đúng người để hạ hạng | Phân tích lỗi | P2 | T |

### 3.3 Single-intent theo thuộc tính, bối cảnh
| ID | Kịch bản | Ví dụ | Hành vi mong đợi | Kiểm thử | Pain | Ưu tiên |
|---|---|---|---|---|---|---|
| S1 | Người + hoạt động | "Mỹ Tâm biểu diễn trên sân khấu" | Ảnh người đó, ưu tiên ảnh sân khấu | Pooling + chấm tay 0/1/2 | P3 | B |
| S2 | Người + trang phục | "Hồ Ngọc Hà mặc áo dài" | Như trên | Pooling | P3 | N |
| S3 | Chỉ chủ đề, không tên | "vận động viên cầu lông", "doanh nhân phát biểu" | Ảnh đúng nghề, đa dạng người | Precision@10 + số person khác nhau | P3, P12 | B |
| S4 | Kiểu ảnh | "ảnh chân dung studio", "ảnh tư liệu đen trắng" | Dựa caption/thuộc tính ảnh | Pooling | P3 | N |
| S5 | Thời điểm | "Mỹ Tâm hồi nhỏ", "Đen Vâu gần đây" | Dùng caption/tiêu đề/`crawled_at`; nếu không đủ thông tin thì nói rõ | Ghi vào hạn chế nếu không làm | P9 | T |
| S6 | Sự kiện cụ thể | "Phạm Nhật Vượng tại sự kiện VinFast" | Ưu tiên ảnh có tiêu đề/caption nhắc sự kiện | Pooling | P10 | N |

### 3.4 Multi-intent
| ID | Kịch bản | Ví dụ | Hành vi mong đợi | Kiểm thử | Pain | Ưu tiên |
|---|---|---|---|---|---|---|
| M1 | Hai người, hiểu là "mỗi người" | "Trấn Thành và Hari Won" | Hai nhóm kết quả cân bằng; nếu có ảnh chung thì có thêm nhóm "cùng xuất hiện" | Độ phủ ý định@10, tỷ lệ nhỏ nhất của một nhóm | P3 | B |
| M2 | Hai người, yêu cầu cùng khung hình | "ảnh Trấn Thành chụp cùng Hari Won" | Chỉ ảnh có cả hai; rỗng thì nói rõ và gợi ý xem từng người | Giao ground truth | P3 | B |
| M3 | Một người, hai bối cảnh | "Mỹ Tâm trên sân khấu và Mỹ Tâm thời thơ ấu" | Hai nhóm theo bối cảnh | Độ phủ ý định | P3 | N |
| M4 | Hai chủ đề không tên | "ca sĩ nhạc rap và cầu thủ bóng đá" | Hai nhóm theo nghề | Độ phủ ý định | P3 | N |
| M5 | Một vế không có kết quả | "Đen Vâu và <người không có>" | Hiện vế có kết quả, báo vế còn lại rỗng | Kiểm tra thông báo | P1, P3 | B |
| M6 | Mất cân bằng | Một người có 100 ảnh, người kia 5 | Vế ít ảnh vẫn có mặt ở top-10 | Tỷ lệ nhỏ nhất của một nhóm | P3 | B |
| M7 | "và" nằm trong tên | "Gia đình Trấn Thành – Hari Won" | Không tách sai | Kiểm thử bộ phân tích | P3 | N |

### 3.5 Multi-hop
| ID | Kịch bản | Ví dụ | Hành vi mong đợi | Kiểm thử | Pain | Ưu tiên |
|---|---|---|---|---|---|---|
| H1 | Quan hệ gia đình, 1 bước | "ảnh con trai chủ tịch Phạm Nhật Vượng" | Liên kết người gốc → cạnh "con" có bằng chứng → ảnh người đích; hiện chuỗi và link bằng chứng | Độ đúng người gốc, độ đúng đường, P@10 | P8 | B |
| H2 | Có nhiều đáp án | "ảnh con của …" khi có nhiều con | Nhóm theo từng người đích | Độ phủ đáp án | P3, P8 | B |
| H3 | Quan hệ không có bằng chứng | Quan hệ không tồn tại trong KG | Trả "không tìm thấy bằng chứng", không trả ảnh | Truy vấn phủ định; tỷ lệ bịa | P8 | B |
| H4 | Có quan hệ nhưng người đích không có ảnh | Cạnh có bằng chứng, người đích không có entity | Hiện bằng chứng quan hệ, báo không có ảnh | Kiểm tra thông báo | P1, P8 | B |
| H5 | Người gốc qua chức danh | "ảnh vợ của chủ tịch Vingroup" | Chức danh → người (cần cạnh chức vụ có bằng chứng) → quan hệ | Đường 2 bước | P8 | N |
| H6 | Quan hệ nghề nghiệp/nhóm | "thành viên cùng nhóm với …", "đồng đội của …" | Cạnh thành viên/đội có nguồn | Như H1 | P8 | T |
| H7 | Người gốc mơ hồ | "vợ của Sơn Tùng" | Hỏi lại người gốc, không chọn bừa | Kiểm tra thông báo | P2, P8 | N |
| H8 | Truy vấn trông như multi-hop nhưng là tên chương trình | "anh trai say hi" | Không coi "anh trai" là quan hệ | Kiểm thử bộ phân tích | P3 | N |

### 3.6 Kết quả, chất lượng và an toàn
| ID | Kịch bản | Hành vi mong đợi | Kiểm thử | Pain | Ưu tiên |
|---|---|---|---|---|---|
| Q1 | Ảnh mờ, nhỏ, mặt quá nhỏ trong top-k | Hạ hạng theo điểm chất lượng | Tỷ lệ ảnh dưới ngưỡng trong top-10 | P5 | N |
| Q2 | Ảnh watermark lớn | Hạ hạng hoặc ẩn theo chính sách | Tỷ lệ ảnh có điểm watermark cao trong top-10 | P5 | N |
| Q3 | Ảnh gần trùng chiếm nhiều ô | Gộp, hiện một ảnh đại diện | Số cặp gần trùng trong top-20 | P4 | B |
| Q4 | Ảnh nhạy cảm lọt vào truy vấn bình thường | Không hiện; chế độ lọc an toàn bật mặc định | Bộ truy vấn "bẫy" + kiểm tra tay | P6 | B |
| Q5 | Ảnh đúng người sai sự kiện | Hiện tiêu đề nguồn để người dùng tự đánh giá | Phân tích lỗi | P10 | T |
| Q6 | Truy vấn nghề nghiệp lệch giới / độ tuổi | Đa dạng hóa, ghi nhận phân bố trong báo cáo | Thống kê top-20 | P12 | T |

### 3.7 Hiệu năng và vận hành
| ID | Kịch bản | Hành vi mong đợi | Kiểm thử | Pain | Ưu tiên |
|---|---|---|---|---|---|
| R1 | Truy vấn phức tạp gọi LLM | Trả kết quả nhanh bằng đường quy tắc, LLM có timeout | p95 theo loại truy vấn | P7 | B |
| R2 | Nhiều người dùng cùng lúc | p95 < 3 s ở mức đồng thời đã công bố | k6 1/4/8/16 VU | P7 | B |
| R3 | Khởi động lạnh | Health chỉ OK khi index đã nạp; đo thời gian khởi động | Đo từ `docker compose up` đến healthy | P7 | N |
| R4 | Ảnh nguồn/CDN lỗi | Ảnh hỏng hiện ô thay thế, không vỡ trang | Giả lập URL lỗi | P1 | N |
| R5 | Mất kết nối API LLM / S3 | Rơi về đường quy tắc; dùng index đã cache | Tắt mạng khi chạy | P7 | N |
| R6 | Truy vấn lặp lại nhiều | Cache kết quả; báo cáo số có/không cache riêng | Đo hai chế độ | P7 | N |

### 3.8 Quyền riêng tư và lạm dụng
| ID | Kịch bản | Hành vi mong đợi | Pain | Ưu tiên |
|---|---|---|---|---|
| X1 | Tìm người thường (không phải người nổi tiếng) | Chỉ tìm trong danh sách person của dữ liệu; không có chức năng tìm bằng ảnh khuôn mặt | P11 | B |
| X2 | Truy vấn tìm ảnh nhạy cảm của người thật | Từ chối hoặc không trả kết quả; ghi log | P6, P11 | B |
| X3 | Ảnh của trẻ em (ví dụ ảnh gia đình người nổi tiếng) | Theo chính sách mentor; mặc định hạn chế hiển thị nổi bật | P11 | T |
| X4 | Demo công khai | Bucket private, link tạm có hạn, cân nhắc đăng nhập | P11 | N |

---

## 4. Gợi ý dùng cho capstone

- **Tối thiểu cho test set:** tất cả kịch bản mức B, mỗi kịch bản 2–5 truy vấn, chia dev/test.
- **Chỉ số đi kèm:** P@10, Recall@50, nDCG@10; với multi-intent thêm độ phủ ý định và α-nDCG; với multi-hop thêm độ đúng đường quan hệ, tỷ lệ từ chối đúng, tỷ lệ bịa; với an toàn thêm số ảnh vi phạm trong top-k của bộ truy vấn bẫy; với hiệu năng là p50/p95/p99 theo loại truy vấn.
- **Trình bày:** dùng bảng pain list ở mục 2 làm slide "vấn đề", rồi ánh xạ mỗi pain sang thành phần giải quyết và số đo tương ứng.
- **Hạn chế nên ghi thẳng:** phủ định, đếm số lượng, thời gian, ảnh sai ngữ cảnh; đây đều là điểm yếu đã biết của các hệ thống lớn và của mô hình kiểu CLIP.

---

## 5. Nguồn

1. Xie et al., "Why People Search for Images using Web Search Engines", WSDM 2018. https://arxiv.org/abs/1711.09559
2. Pu, "An analysis of failed queries for web image retrieval", Journal of Information Science, 2008. https://scholar.lib.ntnu.edu.tw/en/publications/an-analysis-of-failed-queries-for-web-image-retrieval-2/ ; Yoon (ICDM 2009 poster) qua tổng hợp tìm kiếm, chưa đọc bản gốc.
3. Xie et al., "Grid-based Evaluation Metrics for Web Image Search", WWW 2019. https://irlab.science.uva.nl/wp-content/papercite-data/pdf/xie-2019-grid-based.pdf ; Tsukuda & Goto, SIGIR 2020. https://ir.webis.de/anthology/2020.sigirconf_conference-2020.250/
4. Santos et al. (xQuAD) và các chỉ số đa dạng hóa. https://theses.gla.ac.uk/id/eprint/4106 ; https://arxiv.org/pdf/1105.4255
5. Web People Search (WePS). https://github.com/jotaefea/weps ; https://www.ics.uci.edu/~dvk/pub/TKDE08_dvk_WePS.pdf
6. Giới hạn của CLIP: túi từ và gắn thuộc tính https://www.alphaxiv.org/abs/2502.03566.md ; phủ định https://openaccess.thecvf.com/content/ICCV2025/papers/Park_Know_No_Better_A_Data-Driven_Approach_for_Enhancing_Negation_Awareness_ICCV_2025_paper.pdf ; https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Pokhrel_Negation_Matters_Training-Free_Negation-Aware_Image_Retrieval_CVPRW_2026_paper.pdf ; đếm số lượng https://openaccess.thecvf.com/content/ICCV2025/supplemental/Kang_Is_CLIP_ideal_ICCV_2025_supplemental.pdf
7. Google Photos "Ask Photos": https://heise.de/-10424624 ; https://techcrunch.com/2025/06/26/google-photos-merges-classic-search-with-ai-to-speed-up-results
8. Pinterest: https://arxiv.org/pdf/1505.07647 ; https://arxiv.org/pdf/1803.09799
9. Tiếng Việt không dấu: https://manticoresearch.com/blog/vietnamese/ ; https://meta.discourse.org/t/do-not-strip-diacritics-for-search-in-vietnamese/96050
10. Ảnh sai ngữ cảnh (NewsCLIPpings và các nghiên cứu sau): https://arxiv.org/pdf/2403.08783 ; https://arxiv.org/pdf/2311.01766
11. Định kiến trong kết quả ảnh: Kay et al., CHI 2015 https://mdsoar.org/handle/11603/11254 ; tổng hợp Pew qua https://hackernoon.com/are-women-visible-enough-online-an-analysis-of-gender-representation-in-google-image-search-results
12. Tìm theo khuôn mặt và quyền riêng tư (PimEyes): https://netzpolitik.org/2020/pimeyes-face-search-company-is-abolishing-our-anonymity/ ; https://freevacy.com/news/bbc-news/big-brother-watch-files-ico-complaint-against-online-face-search-engine/3026
13. Ý định thời gian: Kim & Xing, WSDM 2013 https://vision.snu.ac.kr/gunhee/r_timesensitive.html ; https://archives.iw3c2.org/www2014/proceedings/companion/p845.pdf
14. Truy vấn không có kết quả (nguồn ngành, độ tin cậy vừa phải): https://lucidworks.com/blog/ecommerce-search-metrics ; https://ai.webscale.com/blog/no-results-found-ecommerce-search
15. Benchmark multi-hop đa phương thức (tham khảo cách dựng test case): M3-VQA https://arxiv.org/pdf/2604.25122 ; CrossModalQA https://arxiv.org/pdf/2609.05518
