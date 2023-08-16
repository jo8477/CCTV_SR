import cv2
import timeit
import subprocess

# 영상 검출기
def videoDetector(cam, cascade):
    
    while True:
        
        start_t = timeit.default_timer()
        # 알고리즘 시작 시점
        """ 알고리즘 연산 """
        
        # 캡처 이미지 불러오기
        ret, img = cam.read()
        # 영상 압축
        img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)
        # 그레이 스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # cascade 얼굴 탐지 알고리즘 
        results = cascade.detectMultiScale(gray,            # 입력 이미지
                                           scaleFactor=1.1, # 이미지 피라미드 스케일 factor
                                           minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                           minSize=(20, 20) # 탐지 객체 최소 크기
                                          )
        
        # 이미지 복사
        img_with_boxes = img.copy()
        
        for box in results:
            x, y, w, h = box
            x, y = x-10, y-10
            w, h = w+20, h+20
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
     
        """ 알고리즘 연산 """ 
        # 알고리즘 종료 시점
        terminate_t = timeit.default_timer()
        FPS = 'fps' + str(int(1. / (terminate_t - start_t)))
        cv2.putText(img_with_boxes, FPS, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
         # 영상 출력        
        cv2.imshow('facenet', img_with_boxes)
        
        key = cv2.waitKey(1)
        if key > 0:
            if key == ord('d'):
                for idx, box in enumerate(results):
                    x, y, w, h = box
                    face_image = img[y:y+h, x:x+w]
                    cv2.imwrite(f'./data/test/SRF_4/data/face_{start_t}_{idx}.jpg', face_image)
        
            elif key == 27:  # ESC 키를 누르면 종료
                #subprocess.run(["python", "img_gen.py"])   #SR을 위한 파이썬 파일 자동 실행
                break

# 가중치 파일 경로
cascade_filename = 'haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

# 영상 파일 
cam = cv2.VideoCapture('./data/video/face.mp4')

# 영상 탐지기
videoDetector(cam, cascade)
