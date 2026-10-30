# TicketMind RAG: kaynaklı destek taslağı

Bu modül sınıflandırıcıdan bağımsızdır. `TicketService` etiket/alan analizi yaparken `RagService` kurgusal yardım belgelerini arar ve destek çalışanına incelenecek bir taslak verir. Yanlış kategori tahmini ilgili belgeyi gizlemesin diye kategoriyle kesin filtreleme yapılmaz.

## Veri ve akış

`data/knowledge_base.json` içinde dört kategoride 16 AI yazımı kurgusal FAQ vardır. Bunlar gerçek bir mağazanın politikası veya yasal hak açıklaması değildir. Sipariş, ödeme veya iade sistemlerine bağlanan araç yoktur; uygulama işlem yapmaz.

```text
Mesaj
  -> boşluk/uzunluk doğrulaması
  -> Türkçe normalleştirme + içerik sözcüğü örtüşmesi
  -> TF-IDF veya yerel Ollama embedding ile sıralama
  -> en fazla 3 kaynak
  -> extractive alıntı VEYA Ollama ile JSON yanıtı
  -> kaynak ID + birebir kanıt alıntısı kontrolü
  -> çalışan incelemesine tabi taslak
```

Ortak sözcük kapısı, stopword'ler çıkarıldıktan sonra en az bir içerik sözcüğünün belgeyle örtüşmesini ister. Domain sözlüğü `kargom -> kargo` gibi dar dönüşümler içerir; genel Türkçe kök bulucu değildir. Kapı başarısızsa embedding ve generator çağrılmaz. Bu kontrol semantik alaka veya prompt injection dayanıklılığı garantisi değildir; tek ortak sözcük içeren alakasız bir sorgu yine geçebilir. Bu sınırlama bağımsız bir değerlendirme kümesinde araştırılmalıdır.

TF-IDF, öğrenme ve offline geliştirme için deterministiktir. Ollama embedding'leri L2 normalize edilir, sıralama cosine similarity ile yapılır. Sonlu olmayan, sıfır veya boyutu tutmayan vektörler reddedilir. Benzerlik skorları güven/başarı yüzdesi değildir. Embedding çağrısında `keep_alive=0` kullanılarak üreticiyle aynı anda iki modelin GPU'da tutulması azaltılır; sonraki sorguda embedding modelini yeniden yüklemek gecikme ekleyebilir. Üreticide `think=false`, 4096 context ve en fazla 384 yeni token istenir.

## Kullanım

```python
from ticketmind.rag import RagService

offline = RagService()  # ağ/model çağrısı yok
offline.draft("İade paketleme sırasında nelere dikkat etmeliyim?")

local_llm = RagService(provider="ollama", generator="ollama", timeout=90)
local_llm.draft("Kargo takip bağlantısını nerede bulurum?")
```

Varsayılan yerel adres `http://127.0.0.1:11434` olur. `TICKETMIND_OLLAMA_BASE_URL` yalnız port içeren loopback HTTP adresi kabul eder; bu değer kullanıcı API girdisi değildir. `TICKETMIND_EMBED_MODEL=embeddinggemma` ve `TICKETMIND_LLM_MODEL=qwen2.5:7b` sunucu yapılandırmasıdır. Uygulama model indirmez. Modellerin yerel Ollama'ya ayrıca kurulması gerekir.

Import ve constructor yalnız yerel FAQ dosyasını okur. İlk ilgili sorguda belge embedding'i hazırlanır. Servis nesnesi model adı + corpus SHA-256 anahtarıyla cache tutar; bir kilit aynı nesnede eşzamanlı ilk yüklemeyi tekilleştirir. Cache disk kalıcılığı veya farklı prosesler arasında paylaşım sağlamaz. Mutable model etiketi güncellenirse servis yeniden başlatılmalıdır; etiket immutable revision iddiası değildir.

HTTP çağrılarının her biri en fazla yapılandırılmış timeout ile sınırlandırılır; 120 saniyeden büyük değer kabul edilmez. İlk corpus embedding'i, query embedding'i ve generation ayrı çağrılar olduğundan toplam süre tek timeout'u aşabilir. HTTP timeout, Ollama'nın sunucuda başlamış native işini kesin sonlandırma garantisi değildir.

## Yanıt sözleşmesi

`draft(text, category=None)` şu alanları döndürür:

| Alan | Anlamı |
| --- | --- |
| `status` | `draft`, `no_evidence`, `model_unavailable`, `invalid_generation` |
| `draft` | İncelenecek metin veya `null` |
| `sources` | `id`, `title`, `text`, `score` içeren belge listesi |
| `mode` | `extractive` veya `ollama` |
| `model` | Ollama üretici adı; extractive için `null` |
| `needs_review` | Her zaman `true` |

`extractive`, birinci belgenin metnini kaynak ID'siyle alıntılar; LLM üretimi gibi sunulmaz. `ollama` seçildiyse model hatasında sessizce extractive çıktıya dönülmez. Eksik model `model_unavailable` döndürür; kanıt yoksa `no_evidence` döner ve üretici hiç çağrılmaz.

Ollama yanıtında `answer`, `citations`, `evidence_quotes` alanları beklenir. Bütün ID'ler getirilen kaynaklardan olmalı; her atıf ilgili belgeden birebir bir alıntıyla desteklenmelidir. Geçersiz JSON, bilinmeyen ID ve uydurma alıntı `invalid_generation` üretir. Bu yapısal doğrulama, cevaptaki her iddianın alıntıdan mantıksal olarak çıktığını kanıtlamaz. Bir model geçerli alıntının yanına desteklenmeyen iddia ekleyebilir; çalışan incelemesi sürer.

Mesaj ve belgeler JSON verisi olarak system talimatından ayrı serileştirilir. Bu ayrım tek başına prompt injection çözümü değildir. API/UI otomatik mesaj göndermez, ödeme veya iade işlemi yapmaz.

## Değerlendirme

```bash
python -m ticketmind.rag_eval --provider tfidf --with-drafts --output reports/rag-offline.json
python -m ticketmind.rag_eval --provider ollama --generator ollama --with-drafts --output reports/rag-ollama.json
```

`data/rag_eval.jsonl` 20 cevaplanabilir, 4 konu dışı/belirsiz geliştirme sorgusu içerir. FAQ ve sorular birlikte yazılmıştır; bağımsız holdout veya gerçek kullanıcı başarısı değildir. Recall@k ve MRR yalnız etiketlenen belge kimliklerini bulmayı ölçer. Cevaplanamayan sorularda `no_evidence_accuracy` ayrı verilir. `draft_status_counts` üretimin teknik sözleşmeye uyumunu gösterir; cevapların semantik doğruluğunu ölçmez.

Offline fixture sonuçları ile gerçek Ollama çalıştırması ayrı raporlanmalıdır. Gerçek model sonucu yoksa varsayım veya fake HTTP testi model kalite sonucu diye yazılmaz. `tests/test_rag.py` fake HTTP ile kontrol akışı ve reddetme sınırlarını doğrular; model kalitesini değerlendirmez.

Kullanılan yerel uçlar: [Ollama embedding API](https://docs.ollama.com/api/embed), [Ollama generation API](https://docs.ollama.com/api/generate).
