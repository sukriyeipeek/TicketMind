# İlk ders: mesajdan tahmine

Bu dersin amacı, çalışan bir modele kendi cümlenle ne yaptığını açıklayabilmek.

## 1. Önce etiketin anlamını belirle

> Kargom gecikti, artık ürünü iade etmek istiyorum.

Bu cümlede iki konu geçiyor. Etiketleme kuralımız kullanıcının açık eylem talebini temel alır: **iade**. Modelin bir kelimeyi görünce otomatik etiket seçmesi yerine, bu kurala uygun çeşitli örneklerden öğrenmesini istiyoruz.

`data/tickets.jsonl` dosyasındaki her satır bağımsız bir JSON kaydıdır. `text` modelin okuyacağı mesaj; `label` eğitim hedefi; `group_id` aynı senaryonun varyantlarını bir arada tutan kimliktir. `split`, kaydın eğitim, doğrulama veya test grubunda olduğunu belirtir. Bu yardımcı alanlar model girdisine eklenmez.

## 2. Metni sayıya çevir

Model doğrudan metinle hesap yapamaz. TF-IDF, eğitim verisindeki kelime ve ikili kelime grupları için sayısal özellikler üretir. Eğitimde sık görülen, pek çok mesajda geçen kelimelerin etkisi ile daha ayırt edici ifadelerin etkisi farklılaşır. Bu yöntem cümlenin bütün anlamını anladığımızı göstermez.

`text.py` Türkçe I/İ dönüşümünü ve boşlukları düzenler. Orijinal mesaj değişmez; alan çıkarımı orijinal metni kullanır. Olumsuzluklar veya Türkçe karakterler silinmez.

## 3. Sınıflandırıcı bağlantıları öğrenir

`train.py` içindeki lojistik regresyon, özelliklerle etiketler arasındaki ağırlıkları öğrenir. Eğitim tamamlanınca TF-IDF ve sınıflandırıcı aynı `Pipeline` içinde kaydedilir. Yeni mesajda aynı dönüşümü yeniden kullanırız; sözlüğü yeniden öğrenmeyiz.

Kontrol sorusu: Test mesajlarını da TF-IDF'nin `fit` adımına verirsek, model değerlendirmesinde hangi bilgi sızmış olur?

## 4. Çıkarılan bilgiyle tahmin edilen niyeti ayır

`TM-1042` biçimindeki bir sipariş numarası kuralla bulunur. `iade` kategorisi model tahminidir. Ürün sözlüğü de bir kuraldır; eğitilmiş NER modeli değildir. Bu ayrım, sistemin ne öğrendiğini doğru anlatmamızı sağlar.

Alanların `start` ve `end` değerleri Python karakter indeksleridir. `text[start:end]`, çıkarılan `value` ile aynı olmalıdır. Bunlar UTF-8 bayt konumları değildir.

## 5. Sonuçlara bak

Önce `reports/baseline.md` dosyasını aç. Dummy model, bütün mesajlarda eğitimde en sık görülen kategoriyi seçen basit bir karşılaştırmadır. Modelimizin bunun üzerine ne koyduğuna bakarız.

Macro F1, dört kategorinin F1 değerlerini eşit ağırlıkla özetler. Hata matrisinde satır gerçek etiket, sütun tahmindir. Yüksek sentetik test skoru gerçek müşterilerde aynı sonucu alacağımızı göstermez.

## İlk küçük alıştırma

Modeli çalıştırmadan şu mesajları etiketle ve gerekçeni yaz:

1. Paketim gelmedi, nerede olduğunu öğrenebilir miyim?
2. Paketim gelmediği için siparişi iade etmek istiyorum.
3. Uygulamada ödeme ekranı açılmıyor ve E42 hatası görünüyor.
4. Merhaba, bugün hava nasıl?

Beklenen yaklaşım: 1 kargo, 2 iade, 3 teknik, 4 kapsam dışı/insan incelemesi. Dördüncü örneğin modelden her zaman `null` dönmesini bekleyemeyiz: bu sürüm öğrenilmiş bir kapsam dışı dedektör içermez. Bütün sonuçlar inceleme gerektirir.

Bu örnekleri mevcut test kümesine ekleyip başarıyı yükseltmeye çalışma. İyileştirmeyi eğitim/doğrulama verisinde araştır; son karşılaştırma için bağımsız test verisi tut.

## Referanslar

- [scikit-learn: metin özellikleri ve TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [scikit-learn: veri sızıntısı ve Pipeline](https://scikit-learn.org/stable/common_pitfalls.html)
- [Hugging Face: metin sınıflandırma](https://huggingface.co/docs/transformers/tasks/sequence_classification)

