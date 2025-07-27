import cv2
import datetime
import os

name = input("Nhập tên: ").strip().replace(" ", "_")
mssv = input("Nhập MSSV: ").strip() # Đã đổi từ 'phone' sang 'mssv' và thay đổi lời nhắc

if not mssv or not name: # Điều kiện kiểm tra đã được cập nhật
    print("⚠️ Tên và MSSV không được để trống!")
    exit()


save_dir = "customer"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{mssv}-{name}.jpg") # Tên file ảnh sử dụng MSSV

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

def detect_and_annotate(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            face = frame[max(0, y1-20):min(y2+20, h-1),
                         max(0, x1-20):min(x2+20, w-1)]

            blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                              MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob_face)
            gender = genderList[genderNet.forward()[0].argmax()]

            ageNet.setInput(blob_face)
            age = ageList[ageNet.forward()[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            return frame, True

    return frame, False


print("🚀 Mở webcam... (Nhấn Q để thoát thủ công)")
cap = cv2.VideoCapture(0)

image_saved = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể mở webcam.")
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

    result_frame, detected = detect_and_annotate(frame)

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(result_frame, f'Time: {current_time}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if detected and not image_saved:
        cv2.imwrite(save_path, result_frame)
        print(f"✅ Đã lưu ảnh tại: {save_path}")
        image_saved = True

    cv2.imshow("Detecting age and gender", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Thoát bằng tay.")
        break

cap.release()
cv2.destroyAllWindows()