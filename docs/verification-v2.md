# v0.2 doğrulama · 30 Ekim 2026

- Yerel ortamda 71 test geçti; pip check başarılı.
- Gerçek CPU TinyBERT fine-tuning: 3 epoch; validation macro F1 0.958042. Kelime 0.916375, karakter 0.916667. Küçük sentetik doğrulama ölçümüdür.
- Dört model /analyze HTTP200; beş Streamlit sekmesi AppTest ile açıldı ve gerçek analiz gönderildi.
- Silver alan çıkarımı micro F1 0.866667.
- Yerel embeddinggemma + qwen3.5:4b ile gerçek RAG yanıtı üretildi; FAQ-IADE-01 kaynak kimliği ve alıntı sözleşmesi doğrulandı. reports/rag-real-smoke.json tek örnek smoke çıktısıdır, genel doğruluk ölçümü değildir.
- Kendi mesajlarını henüz kullanıcı yazmadı; 32 geliştirme örneği sentetiktir. Yeni bağımsız insan yazımı holdout henüz yoktur.
