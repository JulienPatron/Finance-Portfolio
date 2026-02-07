import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# --- CONFIGURATION ---
# L'URL de ta page d'accueil (pour vérifier le bouton de réveil)
HOME_URL = "https://julien-patron-projects.streamlit.app"

# L'URL d'une sous-page (pour simuler de l'activité si l'app est déjà réveillée)
# Assure-toi que cette URL existe bien (ex: l'onglet F1 ou Cinema)
PROJECT_URL = "https://julien-patron-projects.streamlit.app/F1_Elo_System"

def wake_up_app():
    # 1. JITTER : On attend un temps aléatoire entre 1 minute (60s) et 10 minutes (600s)
    # Cela rend l'intervalle d'exécution irrégulier aux yeux de Streamlit.
    delay = random.randint(60, 600)
    print(f"🕒 Pause aléatoire de {delay} secondes avant démarrage...")
    time.sleep(delay)

    print("🚀 Démarrage du robot...")
    
    # Configuration du navigateur "Headless" (Invisible, sans interface graphique)
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # On se fait passer pour un vrai navigateur PC pour ne pas être bloqué
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # --- PHASE A : Vérification du sommeil ---
        print(f"🌍 Connexion à l'accueil : {HOME_URL}")
        driver.get(HOME_URL)
        time.sleep(10) # On laisse le temps à la page de charger
        
        # On cherche le bouton spécifique "Yes, get this app back up!"
        # Le XPath cherche un bouton contenant ce texte exact
        try:
            buttons = driver.find_elements(By.XPATH, "//button[div[contains(text(), 'Yes, get this app back up')]]")
            
            if buttons:
                print("💤 DÉTECTION : L'application est en veille.")
                print("👆 ACTION : Clic sur le bouton de réveil !")
                buttons[0].click()
                time.sleep(20) # On attend que le serveur redémarre
                print("✅ RÉUSSITE : Le bouton a été cliqué.")
                return # Mission accomplie, on arrête le script ici.
            else:
                print("⚡ ANALYSE : Pas de bouton de veille trouvé. L'app est déjà éveillée.")
                
        except Exception as e:
            print(f"Info : Pas de bouton détecté ou erreur de lecture ({e})")

        # --- PHASE B : Simulation d'activité (Si pas de bouton trouvé) ---
        print("🔄 ACTION : Navigation vers un projet pour maintenir l'activité...")
        
        driver.get(PROJECT_URL)
        print(f"👉 Visite de la page : {PROJECT_URL}")
        
        # On reste sur la page 15 secondes
        time.sleep(15) 
        
        # On scroll un peu vers le bas (action humaine)
        driver.execute_script("window.scrollTo(0, 300);")
        time.sleep(2)
        
        print("✅ ACTIVITÉ SIMULÉE : Visite terminée.")

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE : {e}")
    finally:
        driver.quit()
        print("🏁 Fin du script.")

if __name__ == "__main__":
    wake_up_app()