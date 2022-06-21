# Arrhythmia detection based on ECG Signal by Image using 2D Deep CNN
- 참고 논문 : [A Hybrid Deep CNN Model for Abnormal Arrhythmia Detection Based on Cardiac ECG Signal](https://www.mdpi.com/1424-8220/21/3/951)

ECG 데이터를 그래프로 변환하여 2D CNN에 심전도를 구별하는 모델이다. 해당 모델은 위 논문에서 2D CNN 의 모델 구조를 참고하였다. 논문에서 입력되는 데이터와 실제 데이터가 차이가 존재하여 이를 명확히 따라하지는 않는다. 어디까지나 __참고__ 일 뿐. 

## Model 구성
![2D_CNN_Model_Structure](https://www.mdpi.com/sensors/sensors-21-00951/article_deploy/html/images/sensors-21-00951-g005.png)
차이가 있다면 끝단에서 출력하는 나오는 Class의 개수가 다르다. 우리는 5개 (N, S, V, F, Q) 를 갖고 했으나 해당 논문에서는 8개로 분류를 진행하였다. 또한 GrayScale이 아닌 Color 그대로 들어간다는 점도 있다.

# Data
그래프로 변환한 이미지 데이터가 2GB가 넘어 LFS을 통해서도 업로드가 어렵다. 코드와 원본 데이터를 올려두었으니 *cutting_graph.py* 파일을 사용하면 파일을 구할 수 있다.