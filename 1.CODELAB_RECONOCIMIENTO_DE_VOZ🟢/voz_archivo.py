import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import tempfile, os
import requests
import re
import webbrowser

SRATE = 16000     # tasa de muestreo
DUR = 5           # segundos

print("Grabando...!")
audio = sd.rec(int(DUR*SRATE), samplerate=SRATE, channels=1, dtype='int16')
sd.wait()
print("Listo, procesando...")

# guarda a WAV temporal
tmp_wav = tempfile.mktemp(suffix=".wav")
write(tmp_wav, SRATE, audio)

# reconoce con SpeechRecognition
r = sr.Recognizer()
with sr.AudioFile(tmp_wav) as source:
    data = r.record(source)

try:
    texto = r.recognize_google(data, language="es-ES")
    print("Dijiste:", texto)
    cmd = texto.lower()
    if "hola" in cmd:
        print("¡Hola, bienvenido al curso!")
    elif "abrir google" in cmd:
        import webbrowser
        webbrowser.open("https://www.google.com")
    elif "abrir youtube" in cmd:
        webbrowser.open("https://www.youtube.com")
    elif "datos de" in cmd:
        consulta = re.search(r"datos de ([\w\s]+)", cmd)
        if consulta:
            tema = consulta.group(1).strip()
            print(f"Buscando datos acerca de: {tema}")
            # Usar DuckDuckGo Instant Answer API para obtener datos generales
            url = f"https://api.duckduckgo.com/?q={tema}&format=json&no_redirect=1&no_html=1"
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    info = resp.json()
                    extracto = info.get("Abstract")
                    if extracto:
                        print(extracto)
                    else:
                        print("No se encontraron datos relevantes.")
                else:
                    print("No se pudo obtener información.")
            except Exception as e:
                print("Error al consultar datos:", e)
        else:
            print("No se reconoció el tema para buscar datos.")
    elif "noticias de" in cmd:
        consulta = re.search(r"noticia de ([\w\s]+)", cmd)
        if consulta:
            tema = consulta.group(2).strip()
            print(f"Buscando noticias acerca de: {tema}")
            url = f"https://newsapi.org/v2/everything?q={tema}&language=es&apiKey=8b5140f86d494fb58129155b4682ce67"
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    noticias = resp.json().get("articles", [])
                    if noticias:
                        print("Titulares de noticias:")
                    for noticia in noticias[:3]:
                        print("-", noticia.get("title"))
                    else:
                        print("No se encontraron noticias relevantes.")
                else:
                    print("No se pudo obtener información de noticias.")
            except Exception as e:
                print("Error al consultar noticias:", e)
        else:
            print("No se reconoció el tema para buscar noticias.")
    elif "buscar en wikipedia" in cmd:
        termino = re.search(r"buscar en wikipedia ([\w\s]+)", cmd)
        if termino:
            consulta = termino.group(1).strip()
            url = f"https://es.wikipedia.org/wiki/{consulta.replace(' ', '_')}"
            print(f"Abriendo Wikipedia para: {consulta}")
            webbrowser.open(url)
        else:
            print("No se reconoció el término para buscar en Wikipedia.")
    elif "hora" in cmd:
        from datetime import datetime

        print("Hora actual:", datetime.now().strftime("%H:%M"))
    elif "capital de" in cmd:
        pais = re.search(r"capital de ([\w\s]+)", cmd)
        if pais:
            nombre_pais = pais.group(1).strip()
            url = f"https://restcountries.com/v3.1/name/{nombre_pais}"
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if data and "capital" in data[0]:
                        print(f"La capital de {nombre_pais.title()} es {data[0]['capital'][0]}")
                    else:
                        print(f"No se encontró la capital de {nombre_pais.title()}.")
                else:
                    print("No se pudo obtener información del país.")
            except Exception as e:
                print("Error al consultar la capital:", e)
        else:
            print("No se reconoció el país.")
    else:
        print("Comando no reconocido.")
    
except sr.UnknownValueError:
    print("No se entendió el audio.")
except sr.RequestError as e:
    print("Error:", e)
finally:
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)