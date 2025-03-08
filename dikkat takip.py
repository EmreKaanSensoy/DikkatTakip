import cv2
import mediapipe as mp
import time
import math
import pygame
import threading

# Mediapipe modülleri
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Sesli uyarı için pygame
pygame.mixer.init()
ALARM_SOUND_PATH = "C:\\Users\\Emre Kaan\\Downloads\\honk-alarm-repeat-loop-101015.mp3"

# Eşik değerler
WARNING_THRESHOLD_SECONDS = 4
HEAD_TILT_THRESHOLD = 15  # Derece
EYE_CLOSED_THRESHOLD = 0.02  # Göz kapanma oranı için eşik

# Zamanlayıcılar
head_tilted_start_time = None
head_warning_active = False
eye_closed_start_time = None
eye_warning_active = False
alarm_playing = False
head_initial_angle = None  # Başlangıçtaki baş açısı

# Sesli uyarı fonksiyonu
def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        pygame.mixer.music.load(ALARM_SOUND_PATH)
        pygame.mixer.music.play(-1)  # Sürekli çalma

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        alarm_playing = False
        pygame.mixer.music.stop()

# Açı hesaplama fonksiyonu
def calculate_angle(p1, p2):
    """
    İki nokta arasındaki yatay düzlemle yaptığı açıyı hesaplar (derece cinsinden).
    """
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return abs(angle)

# Göz durumu kontrol fonksiyonu
def detect_eye_status(landmarks):
    left_eye_top = landmarks[159]  # Sol gözün üst kısmı
    left_eye_bottom = landmarks[145]  # Sol gözün alt kısmı
    right_eye_top = landmarks[386]  # Sağ gözün üst kısmı
    right_eye_bottom = landmarks[374]  # Sağ gözün alt kısmı
    
    # Gözlerin kapanıp açıldığını kontrol et
    left_eye_open = left_eye_bottom.y - left_eye_top.y
    right_eye_open = right_eye_bottom.y - right_eye_top.y
    
    if left_eye_open < EYE_CLOSED_THRESHOLD and right_eye_open < EYE_CLOSED_THRESHOLD:  # Gözlerin kapalı olduğunu kontrol et
        return True  # Gözler kapalı
    return False  # Gözler açık

# Kamera başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(frame_rgb)

    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            # Baş eğimi kontrolü
            nose_bridge = landmarks.landmark[168]  # Burun kökü
            nose_tip = landmarks.landmark[1]  # Burun ucu
            chin = landmarks.landmark[152]  # Çene noktası (başın eğimini kontrol etmek için)

            nose_bridge_coords = (int(nose_bridge.x * frame.shape[1]), int(nose_bridge.y * frame.shape[0]))
            nose_tip_coords = (int(nose_tip.x * frame.shape[1]), int(nose_tip.y * frame.shape[0]))
            chin_coords = (int(chin.x * frame.shape[1]), int(chin.y * frame.shape[0]))

            # Burun kökü ve çene arasındaki açıyı hesapla
            head_angle = calculate_angle(nose_bridge_coords, chin_coords)

            # Baş eğimi kontrolü
            if head_initial_angle is None:
                head_initial_angle = head_angle  # Başlangıç açısını kaydet

            if abs(head_angle - head_initial_angle) > HEAD_TILT_THRESHOLD:
                if head_tilted_start_time is None:
                    head_tilted_start_time = time.time()
                elapsed_time = time.time() - head_tilted_start_time
                if elapsed_time >= WARNING_THRESHOLD_SECONDS:
                    head_warning_active = True
                    if not alarm_playing:
                        threading.Thread(target=play_alarm).start()
            else:
                # Baş normal pozisyona dönerse
                if head_warning_active:  # Eğer alarm aktifse, durdurulmalı
                    stop_alarm()
                head_tilted_start_time = None
                head_warning_active = False

            # Göz kapanma durumu kontrolü
            eye_status = detect_eye_status(landmarks.landmark)
            if eye_status:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                eye_elapsed_time = time.time() - eye_closed_start_time
                if eye_elapsed_time >= WARNING_THRESHOLD_SECONDS:
                    eye_warning_active = True
                    if not alarm_playing:
                        threading.Thread(target=play_alarm).start()
            else:
                # Gözler açıldığında alarm durdurulmalı
                if eye_warning_active:  # Eğer alarm aktifse, durdurulmalı
                    stop_alarm()
                eye_closed_start_time = None
                eye_warning_active = False

            # Görselleştirme
            cv2.putText(frame, f"Head Angle: {int(head_angle)} deg", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if head_warning_active:
                cv2.putText(frame, "WARNING: Head Tilted!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if eye_warning_active:
                cv2.putText(frame, "WARNING: Eyes Closed!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Burun ve çene noktalarını çizin
            cv2.circle(frame, nose_bridge_coords, 5, (0, 255, 0), -1)
            cv2.circle(frame, nose_tip_coords, 5, (0, 0, 255), -1)
            cv2.circle(frame, chin_coords, 5, (255, 0, 0), -1)

            # Göz durumu yazdır
            cv2.putText(frame, f"Eye Status: {'Closed' if eye_status else 'Open'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Çerçeveyi göster
    cv2.imshow("Dikkat Kontrolu", frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
