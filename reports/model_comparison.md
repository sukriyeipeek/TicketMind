# Sabit model karşılaştırması

AI ile yazılmış sentetik demo verisi. Ölçümler gerçek müşteri başarısını göstermez.

Seçim yalnız validation macro F1 ile yapılır. Eski test daha önce incelendiği için legacy regression olarak gösterilir; yeni holdout değildir.

Seçilen model: **tinybert**. Eşitlik sırası: word, char, tinybert.

| Model | Validation macro F1 | Legacy regression macro F1 | Eğitim (ms) | Validation batch (ms) |
|---|---:|---:|---:|---:|
| word | 0.916 | 0.794 | 138.6 | 2.8 |
| char | 0.917 | 1.000 | 79.7 | 6.7 |
| tinybert | 0.958 | 0.958 | 3905.1 | 19.4 |

Süreler tek yerel koşunun duvar saati ölçümüdür; indirme/yükleme eğitim süresine dahil değildir. Üretim gecikmesi garantisi değildir.

## word

Artifact: `models/word.joblib`

### validation

| Etiket | Precision | Recall | F1 | Örnek |
|---|---:|---:|---:|---:|
| kargo | 1.000 | 0.833 | 0.909 | 6 |
| odeme | 0.833 | 0.833 | 0.833 | 6 |
| iade | 0.857 | 1.000 | 0.923 | 6 |
| teknik | 1.000 | 1.000 | 1.000 | 6 |

Hata matrisi: satır gerçek, sütun tahmin.

| Gerçek / Tahmin | kargo | odeme | iade | teknik |
|---|---:|---:|---:|---:|
| kargo | 5 | 1 | 0 | 0 |
| odeme | 0 | 5 | 1 | 0 |
| iade | 0 | 0 | 6 | 0 |
| teknik | 0 | 0 | 0 | 6 |

Yanlış sınıflandırmalar:

- `kargo-025`: kargo → odeme: Sipariş hazır deniyor ama henüz kargoya verilmemiş. Depodan ne zaman çıkar?
- `odeme-029`: odeme → iade: Mağaza cüzdanıyla ödediğim tutarın bakiye hareketini görüntülemek istiyorum.
### legacy_regression

| Etiket | Precision | Recall | F1 | Örnek |
|---|---:|---:|---:|---:|
| kargo | 0.800 | 0.667 | 0.727 | 6 |
| odeme | 0.714 | 0.833 | 0.769 | 6 |
| iade | 0.714 | 0.833 | 0.769 | 6 |
| teknik | 1.000 | 0.833 | 0.909 | 6 |

Hata matrisi: satır gerçek, sütun tahmin.

| Gerçek / Tahmin | kargo | odeme | iade | teknik |
|---|---:|---:|---:|---:|
| kargo | 4 | 0 | 2 | 0 |
| odeme | 1 | 5 | 0 | 0 |
| iade | 0 | 1 | 5 | 0 |
| teknik | 0 | 1 | 0 | 5 |

Yanlış sınıflandırmalar:

- `kargo-035`: kargo → iade: Ben ürünü geri göndermek istemedim; kargo kendiliğinden depoya dönüşe geçmiş. Teslimatı tamamlatabilir miyiz?
- `kargo-036`: kargo → iade: Paketin alıcıya ulaşmadan geri yollandığını görüyorum, teslimat sürecini incelemenizi rica ederim.
- `odeme-033`: odeme → kargo: Tahsilatta kullanılan para birimi ve kur farkı hakkında açıklama alabilir miyim?
- `iade-035`: iade → odeme: Kutusu yırtıldığı için farklı koli kullanacağım. Ürün iadesinde paketleme şartı var mı?
- `teknik-032`: teknik → odeme: Yalnız klavyeyle gezinirken ödeme formuna geçiş yapamıyorum. Odak sırası bozuk görünüyor.

## char

Artifact: `models/char.joblib`

### validation

| Etiket | Precision | Recall | F1 | Örnek |
|---|---:|---:|---:|---:|
| kargo | 0.833 | 0.833 | 0.833 | 6 |
| odeme | 1.000 | 1.000 | 1.000 | 6 |
| iade | 0.833 | 0.833 | 0.833 | 6 |
| teknik | 1.000 | 1.000 | 1.000 | 6 |

Hata matrisi: satır gerçek, sütun tahmin.

| Gerçek / Tahmin | kargo | odeme | iade | teknik |
|---|---:|---:|---:|---:|
| kargo | 5 | 0 | 1 | 0 |
| odeme | 0 | 6 | 0 | 0 |
| iade | 1 | 0 | 5 | 0 |
| teknik | 0 | 0 | 0 | 6 |

Yanlış sınıflandırmalar:

- `kargo-025`: kargo → iade: Sipariş hazır deniyor ama henüz kargoya verilmemiş. Depodan ne zaman çıkar?
- `iade-030`: iade → kargo: Aldığım dijital üyelik paketinden vazgeçtim. İptal talebinin yolunu soruyorum.
### legacy_regression

| Etiket | Precision | Recall | F1 | Örnek |
|---|---:|---:|---:|---:|
| kargo | 1.000 | 1.000 | 1.000 | 6 |
| odeme | 1.000 | 1.000 | 1.000 | 6 |
| iade | 1.000 | 1.000 | 1.000 | 6 |
| teknik | 1.000 | 1.000 | 1.000 | 6 |

Hata matrisi: satır gerçek, sütun tahmin.

| Gerçek / Tahmin | kargo | odeme | iade | teknik |
|---|---:|---:|---:|---:|
| kargo | 6 | 0 | 0 | 0 |
| odeme | 0 | 6 | 0 | 0 |
| iade | 0 | 0 | 6 | 0 |
| teknik | 0 | 0 | 0 | 6 |

Yanlış sınıflandırmalar:

Bu küçük bölmede yanlış sınıflandırma bulunmadı.

## tinybert

Artifact: `models/tinybert`

### validation

| Etiket | Precision | Recall | F1 | Örnek |
|---|---:|---:|---:|---:|
| kargo | 1.000 | 1.000 | 1.000 | 6 |
| odeme | 1.000 | 1.000 | 1.000 | 6 |
| iade | 1.000 | 0.833 | 0.909 | 6 |
| teknik | 0.857 | 1.000 | 0.923 | 6 |

Hata matrisi: satır gerçek, sütun tahmin.

| Gerçek / Tahmin | kargo | odeme | iade | teknik |
|---|---:|---:|---:|---:|
| kargo | 6 | 0 | 0 | 0 |
| odeme | 0 | 6 | 0 | 0 |
| iade | 0 | 0 | 5 | 1 |
| teknik | 0 | 0 | 0 | 6 |

Yanlış sınıflandırmalar:

- `iade-029`: iade → teknik: İndirme bağlantısını kullanmadığım yazılım lisansını geri vermek için başvuru açacağım.
### legacy_regression

| Etiket | Precision | Recall | F1 | Örnek |
|---|---:|---:|---:|---:|
| kargo | 1.000 | 0.833 | 0.909 | 6 |
| odeme | 1.000 | 1.000 | 1.000 | 6 |
| iade | 0.857 | 1.000 | 0.923 | 6 |
| teknik | 1.000 | 1.000 | 1.000 | 6 |

Hata matrisi: satır gerçek, sütun tahmin.

| Gerçek / Tahmin | kargo | odeme | iade | teknik |
|---|---:|---:|---:|---:|
| kargo | 5 | 0 | 1 | 0 |
| odeme | 0 | 6 | 0 | 0 |
| iade | 0 | 0 | 6 | 0 |
| teknik | 0 | 0 | 0 | 6 |

Yanlış sınıflandırmalar:

- `kargo-031`: kargo → iade: Paket bir teslimat dolabına bırakılmış. Dolaptan almak için gereken kod gelmedi.

## Sınırlar

- Artificial scenario data; real-world accuracy and robustness are unknown.
- Original validation is small and reused; selection can overfit it.
- Legacy test was inspected before these experiments; it is not independent holdout evidence.
- All outputs need human review; probabilities are not calibrated confidence.
- Transformer tokenization truncates at 96 tokens by default; this is recorded in metadata.
