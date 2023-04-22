import cv2

cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()

    # Maske görüntüsünün formatını belirle
    mask = object_detector.apply(frame)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Maske görüntüsünü 3 kanallı BGR formatına dönüştür

    contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue

        cv2.putText(frame, "Durum: {}".format('Hareketli'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    else:
        cv2.putText(frame, "Durum: {}".format(''), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # Birleştirilmiş görüntüyü oluştur
    combined_frame = cv2.hconcat([frame, mask])

    cv2.imshow("Frame", combined_frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()