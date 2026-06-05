import os

label_dir = r"datasets/merged_yolo/labels/train"
img_dir = r"datasets/merged_yolo/images/train"

bad_files = [
"04_3420_jpg.rf.3681a6910d567601904531386a4e5a71.txt",
"04_3420_jpg.rf.ee261ecb8ef973560aeef51b7bc2c0ad.txt",
"07_1035_jpg.rf.21ea9aefd077993db7033f24acebbc61.txt",
"07_1035_jpg.rf.47d0758994a67561b00473c3e62885e9.txt",
"108_1830_jpg.rf.1a68cb3381f6396bfa8a2b0d5e38cf1c.txt",
"108_1830_jpg.rf.92b9afbfac03b7bb19ea3c8895a47bce.txt",
"18_00215.rf.0d6de1e2ea977c119a701ed7211860c1.txt",
"18_00215.rf.25c418f3f320e0b917b3cd8bcbf2e82a.txt",
"19_00656.rf.63df33bc1d0e0da0b268cb6ea64a1a70.txt",
"19_00656.rf.83b477a90163690dbef242862343c27d.txt",
"4_00425.rf.21eed91afd79bbfb4d8186b5ec038ba8.txt",
"4_00425.rf.d6db41a146c7dfaaffa509fd923b6c89.txt",
"63_01836_jpg.rf.58fc9e82269833c2c76edb1b37d40b33.txt",
"63_01836_jpg.rf.81a69343ee5f6e4623af0bf296eda925.txt",
"63_01871_jpg.rf.cab4ca0cc205e40889f2d5d442e2d6d4.txt",
"63_01871_jpg.rf.ea53552e66213710d0c0fbe6329438bc.txt",
"63_01946_jpg.rf.1b8d4cd6c1169aeb117ab996e9d86054.txt",
"63_01946_jpg.rf.b7a4e00ef8a17d4b0b770d67f97dc874.txt",
"64_00036.rf.1a0efbe8b130cfee69eac2dd4b379cff.txt",
"64_00036.rf.c0dbb80db7862208e34495643e4f7e24.txt"
]

for label_file in bad_files:
    label_path = os.path.join(label_dir, label_file)
    img_file = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(img_dir, img_file)

    if os.path.exists(label_path):
        os.remove(label_path)
        print("✅ Deleted label:", label_file)

    if os.path.exists(img_path):
        os.remove(img_path)
        print("🖼️ Deleted image:", img_file)
