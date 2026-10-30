# Model deneyleri: word, character ve TinyBERT

Bu çalışma aynı destek talebini üç farklı sayısal gösterimle sınıflandırmayı öğretir. Sonuçlar AI ile yazılmış küçük sentetik veriye aittir; gerçek müşteri başarısı veya üretime hazır model iddiası değildir.

## Sabit deney düzeni

| Model kimliği | Gösterim | Sınıflandırıcı / eğitim |
|---|---|---|
| `word` | Kelime TF-IDF, 1–2 gram | Lojistik regresyon; C=2, balanced sınıf ağırlığı, seed=42 |
| `char` | Kelime sınırlarına göre karakter TF-IDF, 3–5 gram | Aynı lojistik regresyon ayarları |
| `tinybert` | Türkçe küçük BERT'nin bağlamsal gösterimi | Gerçek fine-tuning; CPU, 3 epoch, batch=8, AdamW lr=0.0005, seed=42 |

Karakter modeli yazım biçimleri ve eklerden farklı örüntüler öğrenebilir. Bu deneyde gerçekten yardımcı olup olmadığını ölçeriz; iyileşme varsayılmaz. TinyBERT daha karmaşık bir gösterim kullanır; küçük veri üzerinde daha iyi sonuç vereceği garanti edilmez.

Sadece `data/tickets.jsonl` içindeki **orijinal train** satırları eğitilir. TF-IDF sözlüğü ve IDF değerleri de yalnız bu satırlardan öğrenilir. `data/development.jsonl` oluşturulmuş olsa bile karşılaştırma tarafından okunmaz. Aynı senaryo grubunun birden fazla bölmede bulunması veri yükleme sırasında reddedilir.

Model seçimi yalnız **validation macro F1** ile yapılır. Eşitlikte önceden belirlenen sıra `word → char → tinybert` uygulanır. Eski test daha önce incelendiği için raporda `legacy_regression` adını alır; model seçimi için kullanılmaz ve yeni, bağımsız holdout gibi sunulmaz. Küçük validation kümesinin tekrar kullanılması da seçim yanlılığı oluşturabilir. Yeni bir bağımsız değerlendirme için ayrı, tercihen insan yazımı ve önceden bakılmamış veri gerekir.

## Çalıştırma

Proje kökünde, etkin sanal ortamda:

```powershell
python -m ticketmind.experiments
```

Bu komut gerçek word ve char modellerini eğitir, kaydeder ve ölçer. Transformer paketi yüklemez, model indirmez. Çıktılar:

```text
models/word.joblib
models/char.joblib
reports/model_comparison.json
reports/model_comparison.md
```

README'deki isteğe bağlı transformer bağımlılıklarını kurduktan sonra:

```powershell
python -m ticketmind.experiments --include-transformer --epochs 3
```

Bu komut ayrıca aşağıdaki sabit kaynaktan ağırlık ve tokenizer indirir:

- Model: [`ytu-ce-cosmos/turkish-tiny-bert-uncased`](https://huggingface.co/ytu-ce-cosmos/turkish-tiny-bert-uncased)
- Revision: `208794046047fc7445f7a4179636423802691268`
- Ağırlık biçimi: `model.safetensors`
- `trust_remote_code=False`

Tokenizer'a `do_lower_case=False` verilir; metin önce `ticketmind.text.normalize_text` ile Türkçe `I/İ` dönüşümü ve NFC normalleştirmesinden geçer. Train ve kullanımda aynı dönüşüm uygulanır. Varsayılan token bütçesi özel tokenlar dahil 96'dır; uzun metinler bu bütçeye kırpılır. Eğitimde kırpılan örnek sayısı manifestte saklanır. Karakter sınırı ile token bütçesi aynı şey değildir.

Transformer'ın bütün parametreleri destek talebi etiketleriyle eğitilir; bu çalışma LoRA deneyi değildir. Eğitim boyunca yalnız train batch'leri gradyan üretir; validation ve legacy regression yalnız eğitim bittikten sonra ileri geçişle ölçülür. İndirme/yükleme, ön işleme ve eğitim süreleri ayrı saklanır.

Eğitilmiş transformer şu klasöre yazılır:

```text
models/tinybert/
  config.json
  model.safetensors
  tokenizer dosyaları
  metadata.json
```

`metadata.json`; kaynak revision, kütüphane sürümleri, train kimlik/grupları ve hash'i, sabit hiperparametreler, parametre sayısı, epoch kayıpları ve gerçek ölçülen süreleri içerir. Artifact dosyaları yalnız yerel eğitim çıktısıdır; her API isteğinde yeniden eğitim yapılmaz. Yerel `TransformerClassifier` yüklemesi ağ kullanmaz.

İlk indirme tamamlandıysa cache'den çalışma açıkça seçilebilir:

```powershell
python -m ticketmind.experiments --include-transformer --local-files-only
```

İndirme cache'i proje içindeki `models/hf-cache` klasörüdür ve Git'e eklenmez. İsteğe bağlı ortam ayarı PowerShell'de `$env:HF_HOME = Join-Path (Get-Location) 'models/hf-cache'` biçiminde yapılabilir; eğitim fonksiyonu da indirirken bu yerel cache yolunu açıkça kullanır.

Word/char artifact'ları ve karşılaştırma raporları kardeş geçici dosyadan `os.replace` ile yayımlanır; orijinal `models/baseline.joblib` değiştirilmez. Transformer önce ayrı bir geçici klasöre kaydedilir. Yerel model dosyaları yayımlanırken eski metadata kaldırılır ve yeni metadata en son görünür olur. Bu kısa aralıkta yeni yükleme hazır değildir; çalışan servisin yeniden eğitim sırasında model değiştirmesi ayrıca engellenir. Bu mekanizma klasörün bütün dosyaları için tek bir filesystem transaction veya elektrik kesintisi dayanıklılığı iddiası taşımaz.

Epoch sayısını değiştirerek yapılan sonraki koşu yeni bir geliştirme deneyidir. Böyle koşular arasından eski teste bakarak seçim yapmak, eski test sonuçlarını bağımsız başarı kanıtına dönüştürmez.

## Raporu okuma

JSON'da her model için `fit_ms`, `validation` ve `legacy_regression` alanları vardır. Her değerlendirme; accuracy, macro F1, etiket bazında precision/recall/F1, hata matrisi, hatalı örnekler ve batch tahmin süresini taşır. Model seçimi `selection` alanında kaydedilir. Süreler bu makinedeki tek yerel koşunun duvar saati ölçümüdür; yük testi veya p95 üretim gecikmesi değildir. Transformer indirme ve yükleme süresi `fit_ms` içine katılmaz.

Testler veri bölmelerinin modele sızmamasını, kararın yalnız validation metriğini okumasını ve artifact'ların yerel servisle açılmasını denetler. İsteğe bağlı offline transformer testi, ağdan indirme yapmadan rastgele başlatılmış çok küçük bir BERT ile safetensors roundtrip yapar; bu test model kalitesi ölçümü değildir. Gerçek model kalitesi yalnız gerçek `--include-transformer` koşusunun raporunda gösterilir.

Model skorları kalibre edilmiş kesinlik değildir. Bu eğitim demosunun bütün sonuçları insan incelemesi gerektirir.

Teknik kaynaklar: [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction), [Hugging Face Auto sınıfları](https://huggingface.co/docs/transformers/model_doc/auto), [PyTorch optimizer API](https://docs.pytorch.org/docs/stable/optim.html).
