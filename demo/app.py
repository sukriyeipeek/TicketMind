"""A small Turkish screen that calls the API; it never loads/trains a model."""

import os

import httpx
import streamlit as st


API_URL = os.environ.get("TICKETMIND_API_URL", "http://127.0.0.1:8000/analyze")
CATEGORIES = {
    "kargo": "Kargo",
    "odeme": "Ödeme",
    "iade": "İade",
    "teknik": "Teknik sorun",
}
FIELDS = {
    "order_id": "Sipariş numarası",
    "error_code": "Hata kodu",
    "product": "Ürün",
}
REASONS = {
    "synthetic_demo_model": "Model yalnızca sentetik örneklerden öğrendi.",
    "no_known_features": "Mesajda modelin tanıdığı yeterli özellik bulunamadı.",
}


def error_detail(response: httpx.Response) -> str:
    """Turn API errors into readable text without displaying a traceback."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return "API geçerli bir yanıt döndürmedi. Sunucuyu kontrol edin."
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        return "Mesaj geçersiz. 1–4000 karakter arasında, boş olmayan bir metin girin."
    return "İstek işlenemedi. API sunucusunu kontrol edin."


st.set_page_config(page_title="TicketMind-TR", page_icon="🎫", layout="centered")
st.title("TicketMind-TR")
st.write("Türkçe destek mesajının konusunu ve içindeki bilgileri incele.")
st.info(
    "Öğrenme demosu · Model sentetik mesajlarla eğitilir. Sonuçlar insan "
    "incelemesi gerektirir. Gerçek müşteri bilgileri girmeyin."
)

with st.form("analyze_ticket"):
    message = st.text_area(
        "Destek mesajı",
        value=(
            "TM-1042 numaralı siparişimdeki kulaklığı iade etmek istiyorum. "
            "Nasıl başvurabilirim?"
        ),
        max_chars=4000,
        height=130,
    )
    submitted = st.form_submit_button("Mesajı analiz et", type="primary")

if submitted:
    if not message.strip():
        st.error("Analiz etmek için bir mesaj yazın.")
    else:
        try:
            with st.spinner("Mesaj analiz ediliyor…"):
                response = httpx.post(API_URL, json={"text": message}, timeout=10.0)
            if response.is_error:
                st.error(error_detail(response))
            else:
                result = response.json()
                category = CATEGORIES.get(result.get("category"), "Belirlenemedi")
                st.subheader("Analiz sonucu")
                left, right = st.columns(2)
                left.metric("Önerilen kategori", category)
                score = result["model_score"]
                right.metric("Model skoru", f"{score:.3f}")
                st.caption(
                    "Model skoru kalibre edilmemiştir; doğruluk olasılığı olarak "
                    "yorumlanmamalıdır."
                )
                if result.get("needs_review"):
                    st.warning("Destek çalışanının incelemesi gerekli.")
                    for reason in result.get("review_reasons", []):
                        if reason in REASONS:
                            st.write(f"• {REASONS[reason]}")
                entities = result.get("entities", [])
                if entities:
                    st.table(
                        [
                            {
                                "Alan": FIELDS.get(entity["field"], entity["field"]),
                                "Bulunan bilgi": entity["value"],
                            }
                            for entity in entities
                        ]
                    )
                else:
                    st.write("Tanımlı alanlardan bir bilgi bulunamadı.")
                st.caption(f"Model sürümü: {result['model_version']}")
        except httpx.TimeoutException:
            st.error("API zamanında yanıt vermedi. Biraz sonra yeniden deneyin.")
        except httpx.RequestError:
            st.error(
                "Analiz servisine bağlanılamadı. README'deki FastAPI başlatma "
                "komutunu ayrı bir terminalde çalıştırın."
            )
        except (ValueError, KeyError, TypeError):
            st.error("API yanıtı beklenen biçimde değil. API sürümünü kontrol edin.")
