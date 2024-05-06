from scipy.stats import wilcoxon

YOLOV8_scores = []
RetinaNet_scores = []

statistic, p_value = wilcoxon(YOLOV8_scores, RetinaNet_scores)

print("Test statistic:", statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("There is a significant difference between the models.")
else:
    print("There is no significant difference between the models.")
