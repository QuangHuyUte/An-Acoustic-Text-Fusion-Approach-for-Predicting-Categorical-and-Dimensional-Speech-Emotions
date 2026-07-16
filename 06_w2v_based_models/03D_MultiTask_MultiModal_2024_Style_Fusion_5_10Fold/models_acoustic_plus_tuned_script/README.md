# 03D Models - Acoustic + Tuned Script

Folder này dành cho checkpoint của hướng:

```text
03B acoustic expert + 03C tuned script/text expert -> 03D expert fusion head
```

Output hiện tại trong máy chỉ có report/prediction/fusion features, chưa có checkpoint `.pt` của 03D fusion head. Vì vậy không thể dựng lại model đúng từ CSV/report đã tải về.

Sau khi chạy lại notebook:

```text
03D_03B_Acoustic_Expert_03C_Text_Expert_Fusion_5_10Fold.ipynb
```

checkpoint sẽ được lưu tự động vào:

```text
output_03d_03b_03c_expert_fusion/models_acoustic_plus_tuned_script/<protocol>/
```

và được đóng gói thêm thành:

```text
output_03d_03b_03c_expert_fusion/03D_models_acoustic_plus_tuned_script.zip
```

Các file `.pt` trong folder output không nên push lên GitHub nếu dung lượng lớn; chỉ nên giữ local hoặc upload Kaggle Dataset/Drive để tái sử dụng.
