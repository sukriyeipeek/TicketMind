# Veri çalışma alanı ve alan çıkarımı deneyi

## Yerel kullanıcı örnekleri

`WorkbenchStore`, kullanıcı tarafından girilen mesaj, niyet etiketi, senaryo grubu ve açıklama notunu yerel SQLite veritabanında saklar. Varsayılan konum `data/local/reviews.sqlite3`; `TICKETMIND_WORKBENCH_DB` ile farklı bir yerel dosya seçilebilir. Bu veriler Git'e eklenmez. Mesajlar uygulama günlüğüne yazılmaz; veritabanı ve dışa aktarılan dosya mesajı düz metin olarak içerir.

Kabul edilen niyetler `kargo`, `odeme`, `iade`, `teknik` ve `review` değerleridir. `review`, belirsiz veya kapsam dışı örneği saklamaya yarar; otomatik olarak eğitilmiş beşinci sınıf oluşturmaz. Bir geliştirme grubundaki kayıtlar aynı etiketi taşımalıdır. Aynı destek olayının yeniden yazımlarını aynı grup altında tutun.

Kaynak `human_authored`, bölme `development` ve zaman damgası sunucu tarafından atanır. Bu kaynak alanı, kullanıcının giriş yaptığını kaydeder; metnin insanlarca yazıldığını bağımsız olarak doğrulamaz. Başka yerden kopyalanan veya AI ile üretilen mesajları insan yazımı bağımsız bir test kümesi gibi sunmayın.

Metin en fazla 4000 karakter, grup kimliği en fazla 100 karakter, not en fazla 2000 karakter olabilir. Türkçe normalleştirmeden sonra birebir aynı olan mesajın ikinci kaydı reddedilir. Bu, anlamsal benzerlik denetimi değildir.

Varsayılan olarak `data/` altındaki JSONL dosyalarında bulunan mesajlar ve grup kimlikleri korunur. Böylece başlangıç korpusunu veya mevcut bir holdout mesajını kopyalayıp geliştirme verisine katmak engellenir. Başka konumdaki korpusları kullanan uygulama, `protected_paths` ile bu dosyaları açıkça sağlamalıdır. Yerel çalışma alanında önceden kaydedilmiş bir mesajın daha sonra yeni bir korpusa eklenmesi halinde eğitim öncesi sızıntı denetimi yine gerekir.

`list_records` sayfalı kayıt okur; `stats` sınıf ve grup sayıları ile alan etiketlemesi yapılmış kayıt sayısını verir. `export_jsonl` bütün kayıtları UTF-8 JSONL biçiminde döndürür. Dışa aktarma eğitim veya test dosyasını değiştirmez; yeniden eğitim otomatik tetiklenmez. Özel müşteri verisi içeriyorsa dışa aktarılan dosyayı da özel tutun.

## Kullanıcının seçtiği alan aralıkları

Bir alan etiketi `field`, `value`, `start`, `end` içerir. Alanlar `order_id`, `error_code`, `product` olabilir. `start` ve `end`, orijinal metnin Python karakter konumlarıdır; bitiş hariçtir. `text[start:end] == value` koşulu zorunludur. Kodlama bayt konumu veya ekranda görünen karakter kümelerinin sayısı kullanılmaz.

`entities=null` henüz alan etiketlenmediğini, `entities=[]` mesajın incelendiğini ve alan bulunmadığını ifade eder. Etiketlenen aralıklar model tahmininden kopyalanmak zorunda değildir; kullanıcı orijinal metinde doğru aralığı seçmelidir.

## Geliştirme mesajları

`data/development.jsonl`, başlangıçtaki 144 mesaja eklenmeyen **32 yeni AI yazımı sentetik örnek** içerir. Her niyet için sekiz örnek vardır; yazım hatası, olumsuzluk ve birden fazla konunun geçtiği açık ana niyet örnekleri bulunur. `split=development` bu mesajların model geliştirme ve hata analizi için kullanılacağını belirtir. Bağımsız nihai test değildir ve ana test puanıyla karıştırılmamalıdır.

Örneğin “Kargom gelmedi ama iade istemiyorum” kargo; “Ödeme sayfası donuyor, henüz para çekilmedi” teknik; onaylı geri ödemenin banka durumu ise ödeme niyetidir. Etiket kılavuzu geçerlidir.

## Alan çıkarımı için silver değerlendirme

`data/entities_gold.jsonl` adı değerlendirme arayüzündeki referans rolünü ifade eder. İçindeki **24 kaydın etiketleri AI tarafından bu proje için seçilmiştir; bağımsız insan doğrulaması yapılmış gold etiketler değildir**. Her kayıt bunu `source=ai_authored_synthetic`, `annotation_status=ai_authored_silver`, `split=evaluation` ile belirtir.

Aralıklar mevcut çıkarıcı çağrılmadan, beklenen ürün/sipariş/hata ifadeleri elle seçilerek yazılmıştır. Türkçe ekleri bulunan ürün ifadeleri metindeki biçimiyle seçilir. “Oyuncu klavyesi” gibi bir ürün türü tam isim öbeği olarak etiketlenir. Gerçek hayvandaki “fare” veya “Telefonla görüştüm” cümlesindeki iletişim aracı, bu örneklerde satın alınan ürün alanı değildir. Referanslar bu proje etiketleme politikasını yansıtır; ürün varlığı tanımı evrensel değildir.

Korpus, küçük sözlükte bulunmayan kahve makinesi, sırt çantası ve elektrikli süpürge gibi ürünleri; çoklu kodları; olumsuz örnekleri ve Türkçe ekleri içerir. Kurala uymayan sonuçlar raporda tutulur. Bu küme üzerinde kural değiştirildikten sonra aynı kümenin sonucu bir regresyon ölçümü olarak yorumlanmalıdır; yeni ve bağımsız performans kanıtı sayılmaz.

Çalıştırma:

```powershell
python -m ticketmind.entity_eval
```

Komut `reports/entities.json` ve `reports/entities.md` üretir. Tam eşleşme için alan türü, başlangıç ve bitiş konumlarının hepsi aynı olmalıdır. Yanlış sınır hem false positive hem false negative sayılır; alanın bir bölümünü yakalamaya kısmi puan verilmez. Yöntem adı gibi tahmin metadata alanları metriği etkilemez. Her alan için precision, recall ve F1 ile bütün alanlar için micro F1 raporlanır. Pozitif referans/tahmin bulunmayan durumda F1 sıfır olarak tanımlıdır.

Bu deney öğrenilmiş bir NER modelini veya gerçek müşteri başarısını ölçmez. Sonraki adım, açık bir aralık etiketleme kılavuzuyla bağımsız insanlar tarafından doğrulanmış, geliştirme sırasında görülmemiş bir değerlendirme korpusu hazırlamaktır.
