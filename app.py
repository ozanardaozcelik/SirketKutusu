import os
import re
import pandas as pd
import gspread
import gradio as gr
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from transformers import pipeline
from deep_translator import GoogleTranslator

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SHEET_ID = '1RANFj3ANp7jBquIU_516o8TJW0gVOPjsyTPzZkneigo'

def write_secrets_to_files():
    client_secret_json = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
    token_json = os.getenv("GOOGLE_TOKEN_JSON")
    if client_secret_json is None:
        raise ValueError("GOOGLE_CLIENT_SECRET_JSON environment variable not set.")
    with open("client_secret.json", "w", encoding="utf-8") as f:
        f.write(client_secret_json)
    if token_json:
        with open("token.json", "w", encoding="utf-8") as f:
            f.write(token_json)

def get_gspread_client():
    creds = None
    if not os.path.exists("client_secret.json"):
        write_secrets_to_files()
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w', encoding='utf-8') as token_file:
            token_file.write(creds.to_json())
    return gspread.authorize(creds)

def load_data():
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID).sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df['Zaman damgası'] = pd.to_datetime(df['Zaman damgası'], errors='coerce')
    df = df.dropna(subset=['Zaman damgası'])
    df['Görüş'] = df['Görüş'].fillna("").astype(str)
    return df

def translate_tr_en(text):
    if not text.strip():
        return ""
    try:
        return GoogleTranslator(source='tr', target='en').translate(text)
    except Exception:
        return text

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
LABELS = ["complaint", "suggestion", "comment"]

def classify_text(text):
    if not text.strip():
        return "yorum"
    result = classifier(text, LABELS)
    label_en = result['labels'][0]
    label_tr_map = {"complaint": "şikayet", "suggestion": "öneri", "comment": "yorum"}
    return label_tr_map.get(label_en, "yorum")

def preprocess_dataframe():
    global df
    df = load_data()
    df['Seçim'] = df['Görüş'].apply(lambda x: classify_text(translate_tr_en(x)))

preprocess_dataframe()

chat_history = []

def render_chat_html(chat_history):
    html = "<div id='chat-scroll' style='overflow-y:auto; max-height: 60vh; padding-right:10px;'>"
    for sender, message in chat_history:
        if sender == "user":
            html += f"<div class='user-message'>{message}</div>"
        else:
            html += f"<div class='bot-message'>{message}</div>"
    html += "</div>"
    # JS ile container yüksekliği güncelleniyor
    html += """
    <script>
    function updateContainerHeight() {
        const chatScroll = document.getElementById('chat-scroll');
        const chatContainer = document.getElementById('chat-container');
        if(!chatScroll || !chatContainer) return;
        const contentHeight = chatScroll.scrollHeight;
        const maxHeight = window.innerHeight * 0.8; // maksimum yükseklik (80vh)
        if(contentHeight < maxHeight){
            chatContainer.style.height = contentHeight + 40 + "px";
            chatContainer.style.overflowY = "visible";
            chatScroll.style.overflowY = "visible";
        } else {
            chatContainer.style.height = maxHeight + "px";
            chatContainer.style.overflowY = "auto";
            chatScroll.style.overflowY = "auto";
        }
    }
    setTimeout(updateContainerHeight, 50);
    </script>
    """
    return html

def chatbot_answer(user_input):
    global chat_history, df
    if 'Seçim' not in df.columns:
        preprocess_dataframe()

    user_input = user_input.strip()
    chat_history = []

    if not user_input:
        cevaplar = []
        for kategori_tr in ["şikayet", "öneri", "yorum"]:
            filt = df[df['Seçim'] == kategori_tr].sort_values(by='Zaman damgası', ascending=False)
            metinler = filt['Görüş'].tolist()
            if metinler:
                cevaplar.append(f"Tüm {kategori_tr}ler:\n" + "\n".join(f"{i + 1}. {m}" for i, m in enumerate(metinler)))
        cevap = "\n\n".join(cevaplar) if cevaplar else "Veri bulunamadı."
        chat_history = [("user", user_input), ("bot", cevap)]
    elif re.fullmatch(r"\d+", user_input):
        cevap = "Sadece sayı girdiniz. Lütfen ne istediğinizi cümle ile yazın. Örneğin, 'son 3 öneri' gibi."
        chat_history = [("user", user_input), ("bot", cevap)]
    else:
        def extract_number(text):
            match = re.search(r"son\s+(\d+)", text)
            return int(match.group(1)) if match else None
        adet = extract_number(user_input)
        kategoriler = {
            "öneri": "öneri",
            "şikayet": "şikayet",
            "yorum": "yorum",
            "görüş": "yorum",
        }
        found = False
        for anahtar_kelime, kategori_adı in kategoriler.items():
            if anahtar_kelime in user_input:
                filt = df[df['Seçim'] == kategori_adı].sort_values(by='Zaman damgası', ascending=False)
                metinler = filt['Görüş'].tolist()
                if not metinler:
                    cevap = f"{kategori_adı.capitalize()} bulunamadı."
                    chat_history = [("user", user_input), ("bot", cevap)]
                    found = True
                    break
                sayi = adet if adet is not None else len(metinler)
                metinler = metinler[:sayi]
                cevap = f"Son {sayi} {kategori_adı}:\n" + "\n".join(f"{i + 1}. {m}" for i, m in enumerate(metinler))
                chat_history = [("user", user_input), ("bot", cevap)]
                found = True
                break
        if not found:
            if "son" in user_input and adet is not None:
                filt = df.sort_values(by='Zaman damgası', ascending=False)
                metinler = filt['Görüş'].head(adet).tolist()
                cevap = f"Son {adet} görüş:\n" + "\n".join(f"{i + 1}. {m}" for i, m in enumerate(metinler))
                chat_history = [("user", user_input), ("bot", cevap)]
            else:
                cevap = "Anlayamadım, lütfen daha açık bir şekilde sorunuzu yazınız."
                chat_history = [("user", user_input), ("bot", cevap)]
    return "", render_chat_html(chat_history)

def refresh_data():
    global chat_history
    chat_history.append(("bot", "Veriler güncelleniyor..."))
    preprocess_dataframe()
    chat_history.append(("bot", "Veriler güncellendi."))
    return render_chat_html(chat_history)

css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
body {
    background: linear-gradient(135deg, #ffc1b3 0%, #ffb3a7 100%);
    color: #eee;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
#chat-container {
    width: 80vw;
    /* height dinamik JS ile ayarlanıyor */
    margin: auto;
    margin-top: 2vh;
    background-color: #0e0e0e;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    padding: 20px;
    box-shadow:
        0 0 15px #ffc1b3,
        0 0 40px #ffb3a7,
        0 0 70px #ff9e80,
        0 0 120px #ff8c72;
    overflow-y: hidden; /* scroll dış container için JS kontrolünde */
}
#header {
    font-size: 48px;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    color: #d2691e;
    text-align: center;
    margin-bottom: 20px;
    text-shadow: 1px 1px 2px #000000;
}
#usage-guide {
    color: #ffb74d;
    font-weight: 600;
    margin-bottom: 18px;
    font-size: 15.5px;
    line-height: 1.45;
    user-select: none;
    overflow-x: visible;
    white-space: normal;
    word-wrap: break-word;
    max-width: 100%;
}
.input-area {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-top: 10px;
}
.input-area textarea {
    flex: 1;
    padding: 12px;
    font-size: 15px;
    border-radius: 10px;
    border: none;
    resize: none;
    max-height: 70px;
    background-color: #3a2800;
    color: #fff3e0;
    box-shadow: 0 0 8px #ffb3a7cc;
}
.send-btn, .refresh-btn {
    background-color: #ffb3a7;
    color: black;
    border: none;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 15px;
    cursor: pointer;
    height: 42px;
    min-width: 90px;
    font-weight: 700;
    box-shadow: 0 0 8px #ffb3a7cc;
    transition: background-color 0.3s ease;
}
.send-btn:hover, .refresh-btn:hover {
    background-color: #e89a85;
    box-shadow: 0 0 15px #e89a85cc;
}
.user-message, .bot-message {
    padding: 12px;
    border-radius: 15px;
    max-width: 60%;
    display: inline-block;
    width: fit-content;
    word-wrap: break-word;
    margin-bottom: 8px;
}
.user-message {
    background: #e78f7ccc;
    color: black;
    border-radius: 15px 15px 0 15px;
    margin-left: auto;
}
.bot-message {
    background: #e78f7ccc;
    color: #fff3e0;
    border-radius: 15px 15px 15px 0;
    margin-right: auto;
}
.chat-display {
    overflow-y: visible !important;
    max-height: none !important;
    flex: none !important;
    padding: 12px;
    font-size: 17px;
    color: #3a1f00;
    background-color: #000000;
    border-radius: 10px;
    margin-bottom: 10px;
    box-shadow: inset 0 0 15px #ffaa77aa;
    white-space: pre-wrap;
    word-wrap: break-word;
}
"""

with gr.Blocks(css=css) as iface:
    with gr.Column(elem_id="chat-container"):
        gr.Markdown("SirketKutusu ChatBOT", elem_id="header")
        gr.Markdown("""\
**📘 Kullanım Kılavuzu:**

- ✅ Boş mesaj gönderirseniz, tüm **şikayet**, **öneri** ve **yorumlar** listelenir.
- ✍️ 'Son 3 öneri' ya da 'son 2 yorum' gibi yazarak istediğiniz sayıda sonuç alabilirsiniz.
- 🔄 "Verileri Güncelle" butonuyla Google Sheets bağlantısını yenileyebilirsiniz.
""", elem_id="usage-guide")
        chat_display = gr.HTML(elem_classes="chat-display")
        with gr.Row(elem_classes="input-area"):
            user_input = gr.Textbox(placeholder="Mesajınızı yazın...", lines=2)
            with gr.Column(scale=0, min_width=90):
                send_button = gr.Button(value="Gönder", elem_classes=["send-btn"])
                refresh_button = gr.Button(value="Verileri Güncelle", elem_classes=["refresh-btn"])

    send_button.click(fn=chatbot_answer, inputs=user_input, outputs=[user_input, chat_display])
    refresh_button.click(fn=refresh_data, inputs=None, outputs=chat_display)

if __name__ == "__main__":
    iface.launch(share=True)
