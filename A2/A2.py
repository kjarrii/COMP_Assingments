import cv2, numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

cap = cv2.VideoCapture(0)
low_threshold = 50
high_threshold = 150
ransac_residual_threshold = 5.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 2:
        cv2.imshow("Line Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    ransac = RANSACRegressor(residual_threshold=ransac_residual_threshold)
    poly = make_pipeline(PolynomialFeatures(1), ransac)
    try:
        poly.fit(coords[:, 1].reshape(-1, 1), coords[:, 0])  
        x_min, x_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        y_min, y_max = poly.predict([[x_min], [x_max]])
        cv2.line(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    except ValueError:
        pass

    cv2.imshow("Line Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
