# TicketMind-TR başlangıç veri kartı

## Köken ve amaç

- Sürüm: `synthetic-starter-v1`.
- Oluşturulma tarihi: 2026-10-29.
- Dil: Türkçe; bazı uygulama terimleri ve biçimsel hata kodları içerir.
- Kaynak: AI asistanı tarafından bu proje için yazılmış örnekler (`source: ai_authored_synthetic`).
- Amaç: niyet sınıflandırma, bölme kontrolü, eğitim/değerlendirme ve API akışını öğrenmek.
- Gerçek müşteri kaydı, web kazıma verisi, şirket içi yazışma veya gerçek kişi iletişim bilgisi kullanılmadı. Sipariş ve hata kodları kurgusaldır.
- Etiketler, örnekleri üreten aynı AI asistanı tarafından kılavuza göre atanmıştır. Bağımsız insan doğrulaması yoktur.

Bu veriyle elde edilen puanlar gerçek müşteri trafiğine genellenemez. Korpus, halka açık bir Türkçe NLP benchmark'ı veya bağımsız insan üretimi bir test seti olarak sunulmamalıdır.

## `tickets.jsonl`

Dosya UTF-8 JSON Lines biçimindedir. Her satır ayrı bir JSON nesnesidir.

| Alan | Açıklama |
| --- | --- |
| `id` | Veri sürümü içinde benzersiz örnek kimliği |
| `text` | Orijinal sentetik mesaj |
| `label` | `kargo`, `odeme`, `iade` veya `teknik` |
| `group_id` | Aynı senaryonun yeniden yazımlarını bağlayan kimlik |
| `split` | `train`, `validation` veya `test` |
| `source` | `ai_authored_synthetic` |

Her etiket için 12 senaryo grubu ve her grup için üç mesaj vardır: toplam **48 grup / 144 mesaj**. Gruplar bölmeler arasında paylaşılmaz. Etiketler kod içinde ASCII adlarıyla tutulur; ekranda Türkçe açıklamalar gösterilebilir.

| Bölme | Etiket başına grup | Etiket başına mesaj | Toplam grup | Toplam mesaj |
| --- | ---: | ---: | ---: | ---: |
| `train` | 8 | 24 | 32 | 96 |
| `validation` | 2 | 6 | 8 | 24 |
| `test` | 2 | 6 | 8 | 24 |
| Toplam | 12 | 36 | 48 | 144 |

Bölme önceden kayıtlıdır ve eğitim sırasında rastgele yeniden yapılmaz. Denge, gerçek müşteri sıklıklarını temsil etme amacıyla değil küçük eğitim deneyini okunabilir tutmak için seçildi. Her senaryo grubunun üç örneği birbiriyle ilişkili olduğundan 144 bağımsız gözlem varmış gibi yorumlanmamalıdır.

Gruplar, teslimat dolabı veya hediye değişimi gibi belirli senaryo ailelerine göre asistan tarafından elle atanmıştır. Bu işlem, bütün anlamsal yakınlıkları yakalayan bağımsız bir sızıntı denetimi değildir. Bölmeler arasında aynı sınıfa ait ortak kelimeler doğal olarak bulunur. Doğrudan normalleştirilmiş kopyalar ve grup çakışmaları veri doğrulamasında ayrıca denetlenmelidir.

## `review_examples.jsonl`

Bu dosyada **12 etiketsiz inceleme örneği** vardır: altısı `validation`, altısı `test`. Her satır `id`, `text`, `reason`, `split` ve `source` alanlarını taşır. Eğitimde kullanılmaz ve dört sınıflı sınıflandırma metriğinin paydasına eklenmez.

Örnek türleri: selamlama/teşekkür, alan dışı istek, eşit ağırlıklı çoklu niyet, olumsuzluk, yetersiz bağlam ve belirsiz ifade. `reason` eğitilmiş bir hedef etiket veya otomatik karar kuralı değildir. Örneğin olumsuzluk içeren bir mesajın ana niyeti insan için açık olabilir; burada inceleme akışının ve hata analizinin görünür olması için tutulur.

Bu küçük liste, alan dışı istekleri güvenilir biçimde saptama başarısını kanıtlamaz. Demo sisteminin bütün sonuçları insan incelemesine açık tutulmalıdır. İleride bağımsız değerlendirme yapılacaksa inceleme kararları için de açık bir etiketleme rubriği gerekir.

## Bilinen sınırlamalar

- Mesajlar çoğunlukla düzgün yazılmış, kısa ve tek niyetlidir. Gerçek konuşmalardaki yazım hatası, ağız, emoji, kod değiştirme, öfke, uzun bağlam ve eksik bilgi dağılımları yeterince temsil edilmez.
- Üretici ile etiketleyici aynıdır. Kelime seçimi ve etiketler arasında yapay ipuçları bulunabilir; sentetik doğrulama ve testte yüksek başarı gerçek dünyada yüksek başarı anlamına gelmez.
- Gruplu bölme kopya/yeniden yazım sızıntısını azaltmayı amaçlar; tek başına bağımsız dağılım kanıtı sağlamaz.
- Dört etiket, örnek bir destek süreç tasarımıdır. Örneğin `iade`, değişim ve sipariş iptalini birlikte içerir; `odeme`, onaylı geri ödemenin banka durumunu kapsar.
- Her test sınıfında yalnız altı mesaj bulunur. Birkaç hata toplam puanı belirgin değiştirir; model sıralaması kararsız olabilir.
- Alan çıkarımını karşılaştırmak için ürün/sipariş/hata kodu konumlarıyla bağımsız olarak etiketlenmiş bir değerlendirme kümesi henüz yoktur.
- Mesajların içerdiği iade, ödeme ve teslimat soruları herhangi bir şirketin gerçek politikası veya hukuki süre bilgisi değildir.

## Sonraki veri adımı

Kullanım izni açık, bağımsız insanlar tarafından yazılmış Türkçe destek örnekleri toplanmalı; kişisel veriler korunmalı; aynı konuşma ve benzer şablonlar birlikte gruplanmalıdır. İnsan etiketleyiciler kılavuzla çalışmalı, anlaşmazlıklar kaydedilmeli ve model/ayar seçimine hiç açılmayan yeni bir test kümesi ayrılmalıdır. Sentetik korpus bu yeni kümeden ayrı raporlanmalıdır.
