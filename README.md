# Arrhythmia detection based on ECG Signal by Image using 2D Deep CNN
- 참고 논문 : [A Hybrid Deep CNN Model for Abnormal Arrhythmia Detection Based on Cardiac ECG Signal](https://www.mdpi.com/1424-8220/21/3/951)

ECG 데이터를 그래프로 변환하여 2D CNN에 심전도를 구별하는 모델이다. 해당 모델은 위 논문에서 2D CNN 의 모델 구조를 참고하였다. 논문에서 입력되는 데이터와 실제 데이터가 차이가 존재하여 이를 명확히 따라하지는 않는다. 어디까지나 __참고__ 일 뿐. 

## Model 구성
![2D_CNN_Model_Structure](https://www.mdpi.com/sensors/sensors-21-00951/article_deploy/html/images/sensors-21-00951-g005.png)
차이가 있다면 끝단에서 출력하는 나오는 Class의 개수가 다르다. 우리는 5개 (N, S, V, F, Q) 를 갖고 했으나 해당 논문에서는 8개로 분류를 진행하였다. 또한 GrayScale이 아닌 Color 그대로 들어간다는 점도 있다.

# Data
그래프로 변환한 이미지 데이터가 2GB가 넘어 LFS을 통해서도 업로드가 어렵다. 코드와 원본 데이터를 올려두었으니 *cutting_graph.py* 파일을 사용하면 파일을 구할 수 있다. Input data의 예시를 하나 보여주면 아래와 같은 이미지가 들어가게 된다.
![Normal_beat_sliced_image](./docs/fig1.png)
이미지 크기가 좀 크긴하다만 어차피 모델에 들어가기 이전에 조정을 하고 들어간다. 512 * 512 사이즈로 들어가서 하나의 데이터셋을 구성하게 되고 데이터셋은 Tensorflow 모듈을 쓰게 된다. 해당 비트들은 이전에 미리 잘라둔 비트를 그대로 활용하기 때문에 비트를 따로 구하는 작업을 진행하진 않았다.