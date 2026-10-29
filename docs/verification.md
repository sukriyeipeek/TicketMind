# İlk sürüm doğrulaması

Tarih: 2026-10-29. Ortam: Windows x64, CPython 3.12.14, görev için ayrılmış sanal ortam; sürümler `requirements-lock.txt` dosyasında. Python yürütücüsü masaüstü uygulamasının paketlenmiş runtime'ından geldi. Stok Windows Python kurulumu için ayrı yerel kurulum testi yapılmadı; GitHub Actions iş akışı normal bir Python 3.12 ortamında aynı kontrolleri çalıştırır.

## Çalıştırılan kontroller

- `python -m pytest -q`: **19 passed**. Veri grupları ve sözlük sızıntısı, model kaydetme/yükleme, kaynak metindeki Unicode alan konumları, istek doğrulama ve model bulunamadığında 503 davranışı kontrol edildi.
- `python -m pip check`: **No broken requirements found**.
- `python -m ticketmind.train`: 96 eğitim, 24 doğrulama ve 24 test mesajıyla gerçek TF-IDF/LR eğitimi tamamlandı. Test macro F1: **0.793706**, most-frequent dummy: **0.100000**. Tam sınıf ölçümleri, hata matrisi ve başarısız test mesajları `reports/baseline.md` içinde.
- Gerçek Uvicorn API'sine HTTP isteği: `/health` 200; geçerli Türkçe mesajda `/analyze` 200; metin yerine sayı gönderilince 422.
- Streamlit AppTest, gerçek yerel API'ye bağlı olarak formu çalıştırdı: kategori, model skoru ve iki çıkarılmış alan ekranda üretildi; uygulama exception oluşturmadı.

Uçtan uca örnek: “TM-1042 numaralı siparişimdeki kulaklığı iade etmek istiyorum.” → `iade`, `TM-1042`, `kulaklığı`, `needs_review=true`. Model skoru yaklaşık 0.554'tür ve doğruluk olasılığı olarak yorumlanmaz.

## Kontrol sınırları

- TestClient, mevcut Starlette sürümünde `httpx` yerine `httpx2` kullanılması yönünde bir deprecation uyarısı verdi; testler geçti. Bu uyarı bastırılmadı.
- Masaüstü kısıtlı yürütme ortamının geçici dizin/IPC erişimi testleri engelledi; kontroller normal yerel yürütmede, geçici dosyalar görev klasöründe tutularak tamamlandı. Bu bir uygulama doğruluk metriği değildir.
- AppTest ekranın işlevini doğruladı; farklı tarayıcı, ekran boyutu veya erişilebilirlik denetimi tamamlandığı iddia edilmez.
- Sentetik veri skoru gerçek müşteri başarısı değildir. Bağımsız insan yazımı veri, kalibrasyon, öğrenilmiş OOD/NER ve LLM/RAG henüz değerlendirilmedi.

