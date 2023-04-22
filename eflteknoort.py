import cv2
import time

cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2()

# İlk 30 saniye için ortalama değişim değerlerini hesaplamak için zamanı ölçmek için başlangıç zamanı alınır.
start_time = time.time()
prev_time = start_time
prev_average = 0
count = 0
while True:
    ret, frame = cap.read()

    # Maske görüntüsünün formatını belirle
    mask = object_detector.apply(frame)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Maske görüntüsünü 3 kanallı BGR formatına dönüştür

    contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    change_sum = 0
    change_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue

        change_sum += w*h
        change_count += 1

    if change_count > 0:
        average_change = change_sum/change_count
    else:
        average_change = 0

    # İlk 30 saniye boyunca biriken değerlerin ortalamasını hesapla
    if time.time() - start_time < 30:
        prev_average += average_change
        count += 1
    # 30 saniye sonra her 2 dakikada bir ortalama değişim değerini hesapla ve ilk ortalama ile karşılaştır
    else:
        current_time = time.time()
        if current_time - prev_time > 120:
            current_average = prev_average/count
            change_rate = (current_average-prev_average)/prev_average
            print("Degisim oranı:", change_rate)
            prev_average = current_average
            prev_time = current_time
            count = 0

    # Görüntüyü çiz
    cv2.putText(frame, "Ortalama degisim: {:.2f}".format(average_change), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    # Birleştirilmiş görüntüyü oluştur
    combined_frame = cv2.hconcat([frame, mask])

    cv2.imshow("Frame", combined_frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()