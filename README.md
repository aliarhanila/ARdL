# 🧠 ARdL — Minimal Deep Learning Library (NumPy Only)

**ARdL**, saf **NumPy** kullanılarak sıfırdan yazılmış, küçük veri setleri üzerinde derin öğrenme (Deep Learning) algoritmalarını anlamak ve denemek için oluşturulmuş özel bir projedir.  
Bu kütüphane, hem **Yapay Sinir Ağlarını (NN / MLP)** hem de **Konvolüsyonel Sinir Ağlarını (CNN)** destekler.

---

## 🚀 Özellikler | Features

- 💡 Pure NumPy implementation — no TensorFlow / PyTorch
- 🧩 Two models included:
  - Feedforward Neural Network (MLP)
  - Convolutional Neural Network (CNN)
- 🔁 Custom forward & backward propagation
- ⚙️ Adjustable hyperparameters (learning rate, epochs)
- 🧠 Educational — designed for learning, debugging and visualization

---
## 🗂️ Proje Yapısı | Project Structure

ARdL/
├── ARdL_lib/
│   ├── ardl/
│   │   ├── __init__.py
│   │   ├── train_cnn.py      # Convolutional Neural Network implementation
│   │   └── train_nn.py       # Fully Connected Neural Network (MLP)
├── main.py                   # Example training script (CNN by default)
├── xyz.npz                    # Dataset file (NumPy format)
└── README.md
## MLP İris Dataset Sonuçları












## MLP Nasıl Çalışır? | How the NN Works
### Forward Propagation:

**Tr:**
Girdi verisini alır ve rastgele ağırlıklara sahip bir yapay sinir ağından geçirir, aktivasyon fonksiyonları uygulanır.
Bu sırada loss (kayıp) fonksiyonları ile hatalar hesaplanır.
Bu projede kayıp fonksiyonu sadece CategoricalCrossEntropy’dir.

**En:**
The input data is passed once through a neural network with randomly initialized weights.
Activation functions are applied at each layer.
Meanwhile, a loss function calculates the model's error.
In this project, the only loss function used is Categorical Cross Entropy.

### Back Propagation:

**Tr:**
Hesaplanan kayıplar (hatalar) ile learning rate çarpılır, bunların türevleri (gradyanları) alınır.
Yapay sinir ağındaki her bir nöronun ağırlık ve bias sayıları, bu türevler üzerinden yeniden güncellenir.

**En:**
The calculated losses are multiplied by the learning rate, and the gradients (derivatives) are computed.
Based on these gradients, the neurons’ weights and biases are updated.

### Epochs:

**Tr:**
Forward Propagation ve Back Propagation, her döngüde veriye göre tekrar hesaplanır.
Hesaplanan değerler ile nöronların ağırlıkları ve bias sayıları, her adımda biraz daha istenen seviyeye gelir.

**En:**
During each epoch, the Forward and Back Propagation steps are repeated.
With each iteration, the weights and biases are adjusted closer to the desired values.

### Optimization

**Tr:**
Modelin öğrenmesini hızlandırmak ve daha doğru hale getirmek için optimizasyon algoritmaları kullanılır. 
En yaygın kullanılan algoritmalar arasında SGD (Stochastic Gradient Descent), Adam, RMSProp gibi yöntemler vardır.
Bu algoritmalar, ağırlık güncellemelerini daha verimli ve istikrarlı bir şekilde yapmamızı sağlar.

Şu anda bu projede kullanılan tek optimizasyon algoritması SGD (Stochastic Gradient Descent) dir. 

**En:**
Optimization algorithms are used to accelerate learning and improve model accuracy.
Common algorithms include SGD (Stochastic Gradient Descent), Adam, RMSProp, among others.
These methods make the weight updates more efficient and stable, helping the network converge faster and achieve 
better performance.

Currently the only optimization algorithm used in this project is SGD (Stochastic Gradient Descent).

## CNN Nasıl Çalışır | How the CNN works?

### Convolution 
**Tr:**
Girdi verisini, örnek olarak bir resmi alır. Ardından, önceden tanımlanmış Kernel (filtre) adı verilen küçük matrislerle
görseldeki pikseller çarpılır. Bu işlem, görselden özellik çıkarmak için yapılır. Kerneller piksellerle çarpıldığında,
görseldeki kenarlar, şekiller, renk geçişleri gibi resme özel olan özellikler, resimde daha baskın olarak gösterilir.

Kernel, bütün resmin pikselleri ile teker teker çarpıldığında, ortaya yeni bir resim çıkar. Buna feature map denir.
Feature map (özellik haritası), ana resimdeki kenar, şekil gibi özniteliklerin daha belirgin olarak gözüktüğü
yeni bir resimdir.

**En:**
The input data, for example an image, is taken and multiplied with small matrices called kernels (filters),
which are predefined. This operation is performed to extract features from the image.
When the kernels are multiplied with the pixels, the unique characteristics of the image — such as edges, shapes,
and color transitions — become more prominent in the resulting representation.

When the kernel is applied across all the pixels of the image, a new image is produced, called a feature map.
The feature map is a new version of the image where attributes like edges and shapes appear more clearly.


### ReLu (Rectified Linear Unit)
**Tr:**
Feature map üzerinde sıfırın altında değerler gösteren yerler de çıkacaktır ancak bu değerler genelde modeller için 
önemsizdir bu sebeple feature map a relu aktivasyonunu uygularız bu sıfırın altındaki değerleri sıfıra sabitler
ve sıfırın üzerindeki değerleri de aynen kendisine geri getirir elimizde artık yeni bir feature map var 
sıfırın altında değeri olmayan bu feature map artık poolinge girmee hazırdır 

**En:**
Some values in the feature map may be below zero, but these negative values are usually not important for the model.
Therefore, we apply the ReLU activation to the feature map. ReLU sets all negative values to zero while keeping positive values unchanged. 
The resulting feature map, now without negative values, is ready to be passed to the pooling layer.


### Pooling
**Tr:**
Elde ettiğimiz feature map bize özellikleri detaylıca gösterir ancak hem bu kadar detaylı özelliklere ihtiyaç olmaz 
hemde görselde gereksiz özellikler de barınır
Yüksek detaylı görüntüleri işlemek hem işlem gücünü boşa harcamak hemde eğitim süresini aşırı uzatmaya sebep olur 
bu yüzden yeni bir görsele ihtiyacımız olur daha küçük ama aradığımız özellikleri de içinde barındıran bir görsel
bu görseli elde edebilmek için pooling kullanırız feature map e GlobalAvgPool veya MaxPooling gibi pooling algoritmalarına
ihtiyaç duyarız Pooling algoritmaları ile görüntüdeki gereksiz özellikleri ve çok detaylı olan bu özellikleri filtrelemiş oluruz
bu yöntem ile hem yapay sinir ağımız öğrenmesi gereken veriye daha çok odaklanabilir hemde verinin boyutu azaldığı için 
iş yükümüz azalır 
Bu Projede kullanılan tek pooling algoritması MaxPooling dir 

**En:**
The obtained feature map shows the image features in detail; however, such a high level of detail is often unnecessary,
and redundant information may also exist in the image.
Processing high-resolution, detailed data wastes computational power and significantly increases training time.
Therefore, we need a new image representation that is smaller but still contains the essential features we want to preserve.

To achieve this, we use pooling. Pooling algorithms such as Global Average Pooling or MaxPooling
are applied to the feature map. These algorithms filter out redundant or overly detailed features in the image.

With this method, the neural network can focus more on the important data it needs to learn,
and since the data size is reduced, the computational load decreases as well.

The only pooling algorithm used in this Project is MaxPooling.


### Classification
**Tr:**
Pooling işleminden sonra verimiz bir MLP ağına girer ve sınıflandırılır. İkili sınıflandırmalarda genellikle Sigmoid,
çoklu sınıflandırmalarda ise Softmax fonksiyonu kullanılır.

**En:**
After pooling, the data is passed to an MLP network for classification. In binary classification, 
Sigmoid is generally used, while in multi-class classification, Softmax is applied.


### 🖊️ Author | Yazar

**Ali Arhan İla**  
[GitHub Profile](https://github.com/aliarhanila)  
[LinkedIn Profile](https://www.linkedin.com/in/ali-arhan-ila-693a2830b/)
