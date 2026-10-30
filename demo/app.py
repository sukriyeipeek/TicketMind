"""Five learning stages, communicating only with the local HTTP API."""
import os
import httpx
import streamlit as st

BASE = os.getenv("TICKETMIND_API_URL", "http://127.0.0.1:8000/analyze").removesuffix("/analyze")
CATEGORIES = {"kargo": "Kargo", "odeme": "Ödeme", "iade": "İade", "teknik": "Teknik sorun"}

def call(method, path, **kwargs):
    try:
        response = httpx.request(method, BASE + path, timeout=150, trust_env=False, **kwargs)
        if response.is_error:
            st.info(response.json().get("detail", "İstek tamamlanamadı."))
            return None
        return response
    except (httpx.HTTPError, ValueError):
        st.error("Yerel API'ye erişilemiyor. README'deki sunucu komutunu kontrol edin.")
        return None

st.set_page_config(page_title="TicketMind-TR", page_icon="🎫", layout="wide")
st.title("TicketMind-TR · NLP atölyesi")
st.caption("Sentetik eğitim verisi ve kurgusal mağaza belgeleri. Bütün analizler ve taslaklar insan incelemesi gerektirir.")
analysis, data, experiments, entities, rag = st.tabs(["1 · Analiz", "2 · Veri atölyesi", "3 · Model deneyleri", "4 · Alan değerlendirmesi", "5 · Kaynaklı yanıt"])

with analysis:
    catalog = call("GET", "/models")
    available = [row for row in catalog.json()["models"] if row["available"]] if catalog else []
    choices = {row["key"]: row["label"] for row in available} or {"baseline": "Başlangıç modeli"}
    selected = st.selectbox("Model", list(choices), format_func=choices.get)
    with st.form("analysis"):
        text = st.text_area("Destek mesajı", "TM-1042 numaralı siparişimdeki kulaklığı iade etmek istiyorum.", max_chars=4000)
        submit = st.form_submit_button("Mesajı analiz et", type="primary")
    if submit:
        response = call("POST", "/analyze", json={"text": text, "model": selected})
        if response:
            result = response.json()
            st.metric("Önerilen kategori", CATEGORIES.get(result["category"], "Belirlenemedi"))
            st.bar_chart({"Kategori": [CATEGORIES[key] for key in result["category_scores"]], "Model skoru": list(result["category_scores"].values())}, x="Kategori", y="Model skoru")
            st.caption("Skorlar kalibre edilmemiştir; doğruluk yüzdesi değildir.")
            st.warning("Destek çalışanının incelemesi gerekli.")
            if result.get("input_truncated"):
                st.info("Mesaj transformer token sınırında kısaltıldı.")
            if result["entities"]:
                st.table(result["entities"])
            st.caption(result["model_version"])

with data:
    st.write("Kendi yazdığın mesajları etiketle. Kayıtlar yalnız yerel geliştirme verisidir; eğitim veya test dosyalarını değiştirmez.")
    with st.form("annotation"):
        message = st.text_area("Yeni örnek", max_chars=4000)
        label = st.selectbox("Ana niyet", [*CATEGORIES, "review"], format_func=lambda key: CATEGORIES.get(key, "İnceleme / belirsiz"))
        group = st.text_input("Senaryo grubu", placeholder="benim-kargo-senaryom-01")
        note = st.text_input("Etiketleme gerekçesi", max_chars=1000)
        st.caption("İsteğe bağlı alan etiketleri: metindeki ifadeyi aynen yaz. Birden fazla geçen ifadelerde bu form kayıt yapmaz.")
        values = {field: st.text_input(title) for field, title in [("order_id", "Sipariş numarası ifadesi"), ("product", "Ürün ifadesi"), ("error_code", "Hata kodu ifadesi")]}
        confirmed = st.checkbox("Bu örneği kendim yazdım ve etiketledim.")
        save = st.form_submit_button("Geliştirme örneğini kaydet")
    if save:
        spans, valid = [], True
        for field, value in values.items():
            if value:
                if message.count(value) != 1:
                    st.error("Her alan ifadesi metinde tam bir kez bulunmalı.")
                    valid = False
                else:
                    start = message.index(value)
                    spans.append({"field": field, "value": value, "start": start, "end": start + len(value)})
        if not confirmed:
            st.info("Kaynak beyanını onaylayın.")
        elif valid:
            result = call("POST", "/workbench", json={"text": message, "label": label, "group_id": group, "note": note, "author_confirmed": True, "entities": spans or None})
            if result:
                st.success("Yerel geliştirme kaydı eklendi.")
    records = call("GET", "/workbench")
    if records:
        payload = records.json()
        st.caption(f"{payload['stats']['total']} kayıt · {payload['stats']['groups']} senaryo grubu")
        if payload["records"]:
            st.dataframe(payload["records"], hide_index=True)
        exported = call("GET", "/workbench/export")
        if exported:
            st.download_button("Kayıtları JSONL indir", exported.content, "ticketmind-development.jsonl", "application/x-ndjson")
    with st.expander("32 sentetik geliştirme örneği"):
        examples = call("GET", "/data/development")
        if examples:
            st.dataframe(examples.json()["records"], hide_index=True)

with experiments:
    st.write("Kelime TF-IDF, karakter TF-IDF ve Türkçe TinyBERT karşılaştırması. Seçim doğrulama macro F1 ile yapılır; eski test sonuçları regresyon kontrolüdür.")
    transformer = st.checkbox("Türkçe TinyBERT fine-tuning deneyini de çalıştır (CPU, 3 epoch)")
    if st.button("Karşılaştırmayı çalıştır"):
        response = call("POST", "/experiments/run", json={"include_transformer": transformer, "epochs": 3})
        if response:
            st.session_state["job_id"] = response.json()["id"]
    if st.session_state.get("job_id"):
        st.button("Deney durumunu yenile")
        response = call("GET", "/experiments/jobs/" + st.session_state["job_id"])
        if response:
            st.json(response.json())
    report_response = call("GET", "/experiments")
    if report_response:
        report = report_response.json()
        st.dataframe([{"Model": name, "Doğrulama macro F1": row["validation"]["macro_f1"], "Eski test macro F1": row["legacy_regression"]["macro_f1"], "Eğitim ms": row["fit_ms"]} for name, row in report["models"].items()], hide_index=True)
        st.info("Doğrulamada seçilen model: " + report["selection"]["selected_model"])
        for name, row in report["models"].items():
            with st.expander(name + " · hatalar ve eğitim bilgisi"):
                st.write(row["validation"]["errors"])
                st.json(row["metadata"])

with entities:
    st.write("Tam alan türü ve karakter konumu eşleşmesi. Bu ölçüm AI tarafından etiketlenmiş 24 sentetik silver örneğe aittir.")
    response = call("GET", "/evaluation/entities")
    if response:
        report = response.json()
        st.metric("Micro F1", f"{report['micro']['f1']:.3f}")
        st.dataframe([{"Alan": field, **scores} for field, scores in report["per_field"].items()], hide_index=True)
        with st.expander("Yanlış çıkarılan veya kaçırılan alanlar"):
            st.write([row for row in report["examples"] if not row["exact_match"]])

with rag:
    st.write("Kurgusal mağazanın yardım belgelerinden kaynak bul ve destek çalışanı için taslak hazırla.")
    provider = st.selectbox("Kaynak arama", ["tfidf", "ollama"], format_func=lambda value: "Kelime arama · çevrimdışı" if value == "tfidf" else "Anlamsal arama · yerel embedding")
    generator = st.selectbox("Yanıt yöntemi", ["extractive", "ollama"], format_func=lambda value: "Kaynak pasajı · LLM kullanmaz" if value == "extractive" else "Yerel LLM · kaynaklı taslak")
    question = st.text_area("Destek sorusu", "Ürün iade başvurusunu nereden açabilirim?", max_chars=4000)
    if st.button("Kaynaklı taslak hazırla", type="primary"):
        with st.spinner("Kaynaklar aranıyor ve taslak hazırlanıyor…"):
            response = call("POST", "/rag/draft", json={"text": question, "provider": provider, "generator": generator})
        if response:
            result = response.json()
            if result["status"] == "draft":
                st.write(result["draft"])
                st.warning("Taslağı bir destek çalışanı incelemeli.")
            else:
                st.info({"no_evidence": "Yeterli kaynak bulunamadı.", "model_unavailable": "Yerel model hazır değil; Ollama kurulumunu kontrol edin.", "invalid_generation": "Model çıktısı kaynak doğrulamasını geçemedi."}.get(result["status"], result["status"]))
            st.caption(f"Yöntem: {result['mode']} · Model: {result['model'] or 'kullanılmadı'}")
            for source in result["sources"]:
                with st.expander(source["id"] + " · " + source["title"]):
                    st.write(source["text"])
                    st.caption(f"Arama skoru: {source['score']:.3f}")
