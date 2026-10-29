# TicketMind-TR etiketleme kılavuzu

Bu kılavuz ilk eğitim deneyi için dört sınıfın sınırlarını tanımlar. Veri, 29 Ekim 2026 tarihinde AI asistanı tarafından eğitim amacıyla yazılmış **sentetik Türkçe destek mesajlarından** oluşur. Gerçek bir destek sisteminden toplanmamıştır; insan etiketleyiciler arası uzlaşma ölçülmemiştir.

## Etiket birimi ve temel kural

Bir kayıt, kendi başına okunabilen tek bir müşteri mesajıdır. İlk sürümde bir mesaja **tek ana niyet** atanır. Mesajdaki kelimeleri saymak yerine kullanıcının açıkça istediği işlemi esas alın. Gerçek kullanımda belirsiz mesajı dört etiketten birine zorlamak yerine insan incelemesine yönlendirin.

| Kod | Kapsam | Örnek |
| --- | --- | --- |
| `kargo` | Gönderim, taşıma, takip, dağıtım, teslim ve taşıma hasarı kaydı | “Paketim nerede, teslimat ne zaman?” |
| `odeme` | Tahsilat, kart/banka işlemi, fazla veya eksik çekim, fatura, taksit, daha önce onaylanmış para iadesinin banka durumu | “İade onaylandı ama ücret hesabıma yatmadı.” |
| `iade` | Ürünü geri verme, değişim, sipariş iptali, geri gönderim başvurusu ve ürün kabul koşulları | “Ürünü geri göndermek istiyorum.” |
| `teknik` | Site veya uygulamanın çalışması, hesap erişimi, form/düğme/ekran hatası ve erişilebilirlik | “Giriş düğmesi çalışmıyor, E42 hatası çıkıyor.” |

Bu tanımlar örnek mağaza senaryosunun ürün kararlarıdır. Her destek şirketinin kullanacağı evrensel etiketler oldukları iddia edilmez.

## Kesişen niyetlerde karar örnekleri

- **“Kargom gecikti, iade istiyorum.” → `iade`**: gecikme gerekçe; açık istek ürün veya sipariş iadesidir.
- **“İade istemiyorum, paket nerede?” → `kargo`**: olumsuzlanan işlem ana niyet olamaz.
- **“Ürün dönüşü onaylandı, param yatmadı.” → `odeme`**: mevcut soru banka/tahsilat durumudur.
- **“İade için kargo kodu almak istiyorum.” → `iade`**: kodun amacı geri gönderim başvurusudur.
- **“Taşıyıcı ürünü kendiliğinden geri götürüyor, teslim almak istiyorum.” → `kargo`**: müşterinin geri verme talebi yoktur.
- **“Ödeme sayfası açılmıyor, henüz para çekilmedi.” → `teknik`**: gözlenen sorun arayüzün çalışmamasıdır.
- **“Banka kartı reddetti, başka ödeme yöntemi var mı?” → `odeme`**: istek ödeme yöntemi ve banka işlemiyle ilgilidir.
- **“Kutu kurye elindeyken ezik, hasar kaydı nasıl tutulur?” → `kargo`**; **“Kırık ürünü geri vereceğim.” → `iade`**.
- **“Kargoyu bulamıyorum ve bağımsız çift çekim sorununu da çözün.” → inceleme**: iki eşit talebi tek etiketle temsil etmek bilgi kaybıdır.

Önceki konuşmaya bağımlı mesajlara bağlam uydurmayın. Selamlama, teşekkür, alan dışı soru, yetersiz bağlam veya eşit ağırlıklı birden fazla niyet için inceleme durumu kullanın. Bunlar bu veri sürümünde ayrı bir eğitilmiş beşinci sınıf değildir.

## Metin ve alan çıkarımı

Orijinal metni saklayın. Türkçe karakterleri veya olumsuzlukları silmeyin; “geldi” ile “gelmedi” farklı bilgi taşır. Sayı ve hata kodlarını eğitim metninden körlemesine çıkarmayın.

Sipariş numarası ve hata kodu, varsa metinden birebir çıkarılabilir; örneğin `TM-1042` ve `E42`. Ürün adı için yalnız mesajın içerdiği ifadeyi kullanın. Bilinmeyen alanı uydurmayın. Kaydedilecek başlangıç/bitiş konumları orijinal metne bağlanmalıdır. Mevcut veri dosyası yalnız niyet etiketleri taşır; alan çıkarımı için tamamlanmış bir altın standart NER veri seti değildir.

## Bölme ve kalite kontrolü

`group_id`, aynı senaryonun üç yeniden yazımını bir arada tutar. Bir grubun bütün kayıtları aynı `split` içinde kalmalıdır. Aynı konuşmanın devam mesajları, kopyalar ve aynı üretim şablonunun yeniden yazımları da gerçek veri eklendiğinde birlikte gruplanmalıdır.

Öğrenilen kelime sözlüğü dahil bütün `fit` işlemleri yalnız `train` üzerinde yapılır. Model veya eşik seçimi `validation` üzerinde yapılır. `test` sonuçlarına bakarak seçim yapılırsa test artık bağımsız nihai ölçüm sayılmaz; bu olay deney raporuna yazılmalıdır.

Yeni veri eklerken sıra:

1. Kaynak ve kullanım iznini kaydet; kişisel bilgileri ayıkla veya uygun biçimde anonimleştir.
2. Mesajı bağımsız olarak oku ve açık ana niyeti bul.
3. Kararsızlık varsa açıklama notuyla incelemeye ayır.
4. Bağlantılı mesajların grup kimliğini koru ve grubu tek bölmeye ata.
5. En az iki bağımsız insanın etiketlediği bir alt kümede uyuşmazlıkları tartış.

Bu sentetik başlangıç korpusu geliştirme ve öğrenme içindir. Üretim kullanımı veya gerçek müşteri başarısı için kanıt sağlamaz.
