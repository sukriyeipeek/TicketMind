# Başlangıç modeli: sentetik demo değerlendirmesi

Bu sonuçlar AI ile yazılmış küçük demo verisine aittir; gerçek müşteri mesajlarında başarı iddiası taşımaz.

Veri SHA-256: `345158bf882131cbe3654de507aac6ed9b848cb2d4639ac7c1f6e20bf01a5217`

| Bölme | Mesaj | Grup |
|---|---:|---:|
| train | 96 | 32 |
| validation | 24 | 8 |
| test | 24 | 8 |

| Model | Test macro F1 | Test accuracy |
|---|---:|---:|
| Most-frequent dummy | 0.100 | 0.250 |
| TF-IDF + LR | 0.794 | 0.792 |

Doğrulama macro F1: 0.916. Ayarlar sabittir; testten eşik veya hiperparametre seçilmedi.

| Etiket | Precision | Recall | F1 | Test örneği |
|---|---:|---:|---:|---:|
| kargo | 0.800 | 0.667 | 0.727 | 6 |
| odeme | 0.714 | 0.833 | 0.769 | 6 |
| iade | 0.714 | 0.833 | 0.769 | 6 |
| teknik | 1.000 | 0.833 | 0.909 | 6 |

## Hata matrisi

Satır gerçek, sütun tahmin edilen etikettir.

| Gerçek / Tahmin | kargo | odeme | iade | teknik |
|---|---:|---:|---:|---:|
| kargo | 4 | 0 | 2 | 0 |
| odeme | 1 | 5 | 0 | 0 |
| iade | 0 | 1 | 5 | 0 |
| teknik | 0 | 1 | 0 | 5 |

## Yanlış sınıflandırılan test örnekleri

- `kargo-035`: kargo → iade: Ben ürünü geri göndermek istemedim; kargo kendiliğinden depoya dönüşe geçmiş. Teslimatı tamamlatabilir miyiz?
- `kargo-036`: kargo → iade: Paketin alıcıya ulaşmadan geri yollandığını görüyorum, teslimat sürecini incelemenizi rica ederim.
- `odeme-033`: odeme → kargo: Tahsilatta kullanılan para birimi ve kur farkı hakkında açıklama alabilir miyim?
- `iade-035`: iade → odeme: Kutusu yırtıldığı için farklı koli kullanacağım. Ürün iadesinde paketleme şartı var mı?
- `teknik-032`: teknik → odeme: Yalnız klavyeyle gezinirken ödeme formuna geçiş yapamıyorum. Odak sırası bozuk görünüyor.

## Sınırlar

- AI-authored synthetic data; scores do not estimate real customer performance.
- Scenario groups are split-disjoint; this does not guarantee semantic independence.
- All predictions require human review; model scores are uncalibrated.
- Entity rules are not trained NER and have no held-out extraction score yet.
