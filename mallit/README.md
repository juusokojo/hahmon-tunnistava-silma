# Mallit

`face_detection_yunet_2023mar.onnx` – YuNet-kasvontunnistin (~230 kt).

- Lähde: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Lisenssi: MIT (ks. lähderepositorio)
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- Vaatii OpenCV ≥ 4.8 (`cv2.FaceDetectorYN`).

Lataus uudelleen:

```sh
curl -L -o mallit/face_detection_yunet_2023mar.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

---

`object_detection_nanodet_2022nov.onnx` – NanoDet-Plus-objektintunnistin
(COCO 80 luokkaa, käytössä vain "henkilö"; ~3,8 Mt).

- Lähde: https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet
- Lisenssi: Apache 2.0
- SHA-256: `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186`

```sh
curl -L -o mallit/object_detection_nanodet_2022nov.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx
```
