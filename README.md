# Naive-Bayes-Classifier

Naive Bayes Modeli Kullanılarak Yolcu Memnuniyeti Tahmini
1. Giriş
Bu çalışma, Naive Bayes sınıflandırıcısı kullanılarak havayolu yolcu memnuniyetini tahmin etmeyi amaçlamaktadır. NaiveBayes.py dosyasında scikit-learn kütüphanesindeki GaussianNB modeli kullanılırken, NaiveBayes2.py dosyasında bu modelin manuel bir implementasyonu gerçekleştirilmiştir.
2. Veri Seti
Çalışmada kullanılan veri seti airline_passenger_satisfaction.csv dosyasından alınmıştır. Veri setindeki temel sütunlar:
Kişisel Bilgiler (Cinsiyet, Müşteri Türü, Seyahat Türü vb.)
Hizmet Kalitesi (Check-in servisi, Yemek Kalitesi vb.)
Uçuş Bilgileri (Varış Gecikmesi, Uçuş Süresi vb.)
Bağımlı Değişken (Satisfaction - Memnuniyet): Yolcuların memnuniyet seviyesini gösterir.
3. Veri Ön İşleme
Kimlik sütunu (ID) çıkarıldı.
Eksik veriler (Arrival Delay - Varış Gecikmesi) ortalama değer ile dolduruldu.
Kategorik veriler, Label Encoding yöntemi ile sayısal hale getirildi.
Veri seti eğitim ve test olarak %80-%20 oranında bölündü.
4. Kullanılan Yöntemler
4.1 GaussianNB (Scikit-learn Kütüphanesi)
NaiveBayes.py dosyasında scikit-learn kütüphanesinin GaussianNB modeli kullanılmıştır.
Model, eğitim verileri ile eğitilmiş ve test verileri üzerinde doğruluğu hesaplanmıştır.
4.2 Özel Gaussian Naive Bayes Modeli (Manuel Uygulama)
NaiveBayes2.py dosyasında, MyGaussianNB adlı özel bir sınıf yazılmıştır.
Aşamalar:
fit() fonksiyonu ile her sınıf için ortalama ve varyans hesaplandı.
predict() fonksiyonu ile olasılık hesaplanarak en yüksek olasılıklı sınıf tahmin edildi.
score() fonksiyonu ile modelin doğruluk oranı ölçüldü.

5. Model Performansı
Model
Eğitim Süresi (sn)
Test Doğruluğu
Scikit-learn GaussianNB
Hızlı
Yüksek
Manuel GaussianNB
Daha Yavaş
Benzer


Scikit-learn modeli daha hızlı eğitim süresi sunarken, doğruluk oranı benzerlik göstermektedir.
6. Sonuç ve Değerlendirme
GaussianNB kullanılarak yolcu memnuniyeti başarılı bir şekilde tahmin edilmiştir.
Scikit-learn modeli daha verimli çalışırken, manuel implementasyon makine öğrenmesi algoritmalarının anlaşılmasını kolaylaştırmaktadır.
Manuel model geliştirilerek hız ve doğruluk artırılabilir.
Bu çalışmada, Naive Bayes yönteminin müşteri memnuniyeti tahmininde etkili olduğu gösterilmiştir.
7. Kaynaklar
https://www.geeksforgeeks.org/naive-bayes-classifiers/
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
https://youtu.be/QiLsuwtbw28?si=l2RUnHmVOElfKLqu


