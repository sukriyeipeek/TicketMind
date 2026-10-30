# Alan çıkarımı: sentetik silver değerlendirmesi

Bu etiketler AI tarafından elle seçilmiş aralıklardır; bağımsız insan doğrulaması yoktur.

Örnek sayısı: 24. Veri SHA-256: `4a8cd463e4252a3301105bd79934d228bf8655e3b617c70cdb897c3141e64f7f`.

Eşleşme: alan türü + başlangıç + bitiş konumunun tamamı aynı olmalıdır. Kısmi puan verilmez.

| Alan | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| error_code | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| order_id | 10 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| product | 11 | 3 | 5 | 0.786 | 0.688 | 0.733 |
| micro | 26 | 3 | 5 | 0.897 | 0.839 | 0.867 |

## Hata bulunan örnekler

- `entity-silver-03`: TM-9003 içindeki kahve makinesini değiştirmek istiyorum. Yanlış eklenen: `[]`; kaçırılan: `[['product', 17, 33]]`.
- `entity-silver-04`: Masa lambasını geri gönderme koşullarını öğrenebilir miyim? Yanlış eklenen: `[]`; kaçırılan: `[['product', 0, 14]]`.
- `entity-silver-10`: Oyuncu klavyesi yerine başka model geldi, değişim istiyorum. Yanlış eklenen: `[['product', 7, 15]]`; kaçırılan: `[['product', 0, 15]]`.
- `entity-silver-12`: TM-9012 paketindeki sırt çantası yanlış renkte. Yanlış eklenen: `[]`; kaçırılan: `[['product', 20, 32]]`.
- `entity-silver-13`: Kutuyu depoda bir fare kemirmiş, ürün zarar görmüş. Yanlış eklenen: `[['product', 18, 22]]`; kaçırılan: `[]`.
- `entity-silver-23`: Telefonla görüştüğüm görevli paketimin yolda olduğunu söyledi. Yanlış eklenen: `[['product', 0, 9]]`; kaçırılan: `[]`.
- `entity-silver-24`: Elektrikli süpürgemi geri vermek için TM-9024 talebini açtım. Yanlış eklenen: `[]`; kaçırılan: `[['product', 0, 20]]`.

## Sınırlar

- Rules and a small product dictionary, not a learned NER model.
- Silver labels were written for this project; results do not estimate customer performance.
- Unknown products and ambiguous words deliberately remain in this set; failures are retained.
